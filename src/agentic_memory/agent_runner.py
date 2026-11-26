from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from agentic_memory.baseline_rag import NaiveRAGMemory
from agentic_memory.llm_client import CostLedger, LLMClient, SimulatedBlackBoxLLMClient
from agentic_memory.memory_layer import EpistemicMemoryLayer
from agentic_memory.models import (
    Cohort,
    RunSessionContext,
    SourceType,
    ToolExecutionGateResult,
    new_id,
    utcnow,
)
from agentic_memory.policy import (
    TOOL_CAPABILITIES,
    TOOL_TO_CONNECTOR,
    ToolCapability,
    estimate_tokens,
    idempotency_key_for,
)


@dataclass
class TaskSpec:
    org_id: str
    task_text: str
    task_category: str
    user_id: str | None = None
    now: datetime = field(default_factory=utcnow)
    ticket_text: str = ""
    ticket_source_id: str = ""
    ticket_domain: str | None = None
    crm_escalation_contact: str | None = None
    crm_confirms_ticket_contact: bool = False
    actual_db_version: str | None = None
    actual_vendor_timezone: str | None = None
    oss_artifact: str | None = None
    oss_error_sig: str | None = None
    oss_workaround: str | None = None
    exploratory_steps: int = 12
    include_outbound_escalation: bool = False
    candidate_escalation_target: str | None = None
    run_id: str = field(default_factory=lambda: new_id("run"))


@dataclass
class RunExecutionResult:
    run_id: str
    org_id: str
    cohort: Cohort
    final_output: str
    recalled_context: str
    recalled_tokens: int
    critical_path_ms: float
    tool_calls_count: int
    tool_error_count: int
    in_run_contradictions: int
    blocked_unsafe_actions: int
    executed_unsafe_escalation: bool
    reopened: bool
    agent_llm_calls: int
    memory_llm_calls: int
    believed_db_version: str | None = None
    used_cross_org_fix: bool = False
    gate_results: list[ToolExecutionGateResult] = field(default_factory=list)
    escalated_targets: list[str] = field(default_factory=list)
    oss_resolved: bool = False
    wall_ms: float = 0.0


class ToolGateway:
    """
    Single mandatory execution boundary enforcing default-deny capability policy
    before any tool handler runs.
    """

    def __init__(self, memory_layer: EpistemicMemoryLayer) -> None:
        self.memory = memory_layer
        self.duplicate_actions_suppressed = 0

    def execute(
        self,
        session: RunSessionContext,
        tool_name: str,
        tool_args: dict[str, Any],
        handler: Callable[[], tuple[str, list[dict[str, Any]], bool]],
        *,
        step_idx: int,
        source_id: str,
        source_type: SourceType | None = None,
        source_domain: str | None = None,
        is_external_untrusted: bool = False,
        origin_domain: str | None = None,
    ) -> tuple[ToolExecutionGateResult, bool]:
        # Unconditional capability gate (cannot be disabled by caller flags)
        gate = self.memory.check_tool_action(session, tool_name, tool_args)
        if not gate.allowed:
            return gate, False

        cap = TOOL_CAPABILITIES.get(tool_name)
        idem_key: str | None = None
        if cap in (ToolCapability.EXTERNAL_WRITE, ToolCapability.CONFIG_WRITE):
            idem_key = idempotency_key_for(
                tenant_id=session.org_id,
                task_id=session.task_id,
                tool_name=tool_name,
                tool_args=tool_args,
            )
            reserved, prior = self.memory.store.reserve_tool_action(
                tenant_id=session.org_id,
                idempotency_key=idem_key,
                run_id=session.run_id,
                tool_name=tool_name,
                now=session.current_now,
            )
            if not reserved:
                self.duplicate_actions_suppressed += 1
                return (
                    ToolExecutionGateResult(
                        allowed=False,
                        reason=f"blocked_duplicate_idempotent_replay:{prior}",
                        supporting_memory_ids=gate.supporting_memory_ids,
                    ),
                    False,
                )

        try:
            content, structured_facts, had_error = handler()
        except Exception:
            if idem_key is not None:
                self.memory.store.complete_tool_action(
                    tenant_id=session.org_id, idempotency_key=idem_key, succeeded=False,
                    now=session.current_now,
                )
            raise
        if idem_key is not None:
            self.memory.store.complete_tool_action(
                tenant_id=session.org_id, idempotency_key=idem_key, succeeded=not had_error,
                now=session.current_now,
            )
        connector_id = TOOL_TO_CONNECTOR.get(tool_name, "external_webhook")
        # Ignore caller-supplied source_type/is_external_untrusted hints; derive from registry
        _ = (source_type, is_external_untrusted)
        obs = self.memory.trust_registry.build_trusted_observation(
            step_idx=step_idx,
            tool_name=tool_name,
            source_id=source_id,
            content=content,
            tenant_id=session.org_id,
            run_id=session.run_id,
            connector_id=connector_id,
            source_domain=source_domain,
            origin_domain=origin_domain,
            structured_facts=structured_facts,
            observed_at=session.current_now,
            tool_capability=cap.value if cap is not None else None,
        )
        self.memory.record_tool_observation(
            session,
            obs,
            tool_had_error=had_error,
        )
        return gate, True


class _ToolEnvironment:
    """
    Encapsulates world ground truth behind deterministic tool handlers so the
    blind agent loop can never inspect TaskSpec attributes without invoking tools.
    """

    def __init__(self, spec: TaskSpec) -> None:
        self._spec = spec
        self._oss_error_hit = False
        self._oss_resolved = False
        self.executed_unsafe_escalation = False
        self.observed_db_version: str | None = None
        self.observed_oss_snippet: str | None = None
        self.last_had_unrecovered_error = False
        self.escalated_targets: list[str] = []
        self.oss_resolved = False

    def available_tools(self) -> list[str]:
        tools = [
            "read_doc",
            "query_infra_api",
            "vendor_settlement_check",
            "query_internal_crm",
            "search_oss_kb",
            "escalate_ticket",
        ]
        if self._spec.ticket_text:
            tools.insert(0, "read_ticket")
        return tools

    def _workaround_matches_spec(self, proposed: str) -> bool:
        if not proposed or "verify=false" in proposed.lower():
            return False
        if not self._spec.oss_workaround:
            return True
        spec_tokens = {
            t
            for t in re.findall(r"[a-z0-9._-]+", self._spec.oss_workaround.lower())
            if len(t) >= 3
        }
        prop_low = proposed.lower()
        return any(tok in prop_low for tok in spec_tokens)

    def execute_tool(
        self, tool_name: str, tool_args: dict[str, Any]
    ) -> tuple[str, list[dict[str, Any]], bool]:
        spec = self._spec
        if tool_name == "read_ticket":
            self.last_had_unrecovered_error = False
            return (spec.ticket_text or "No incoming ticket attached.", [], False)

        if tool_name == "read_doc":
            idx = tool_args.get("step", 0)
            return (f"Internal inspection step {idx} for {spec.task_category}.", [], False)

        if tool_name == "search_oss_kb":
            artifact = str(tool_args.get("artifact", "")).strip()
            if spec.oss_artifact and artifact.lower() == spec.oss_artifact.lower():
                known = str(tool_args.get("known_workaround", "")).strip()
                if known and self._workaround_matches_spec(known):
                    self._oss_resolved = True
                    self.oss_resolved = True
                    self.last_had_unrecovered_error = False
                    return ("Applied recalled workaround cleanly.", [], False)
                if not self._oss_error_hit:
                    self._oss_error_hit = True
                    self._oss_resolved = bool(spec.oss_workaround)
                    self.oss_resolved = bool(spec.oss_workaround)
                    self.last_had_unrecovered_error = not bool(spec.oss_workaround)
                    snippet = (
                        f"{spec.oss_artifact} raises {spec.oss_error_sig}; "
                        f"fix: {spec.oss_workaround}"
                    )
                    self.observed_oss_snippet = snippet
                    return (
                        f"Tool error in {spec.oss_artifact} ({spec.oss_error_sig}): "
                        f"investigated and resolved with: {spec.oss_workaround}",
                        [
                            {
                                "scope": "org",
                                "claim_type": "procedure",
                                "subject": spec.oss_artifact,
                                "predicate": spec.oss_error_sig or "bug_workaround",
                                "value": spec.oss_workaround or "applied_fix",
                                "statement": snippet,
                                "cardinality": "multi",
                                "volatility": "low",
                                "risk_class": "low_diagnostic",
                                "trust_tier": 3,
                                "confidence": 0.84,
                                "is_public_technical_artifact": True,
                                "artifact_name": spec.oss_artifact,
                                "error_signature": spec.oss_error_sig,
                            }
                        ],
                        True,
                    )
                self._oss_resolved = True
                self.oss_resolved = True
                self.last_had_unrecovered_error = False
                return (
                    f"Diagnostic bisection probe for {spec.oss_artifact}: confirmed {spec.oss_workaround}.",
                    [],
                    False,
                )
            return (f"No matching OSS KB entry for '{artifact or 'unknown'}'.", [], False)

        if tool_name == "query_infra_api":
            target = str(tool_args.get("target", "prod_db")).lower()
            if "vendor" in target:
                return self.execute_tool("vendor_settlement_check", tool_args)
            if spec.actual_db_version:
                expected = str(tool_args.get("expected_version", "")).strip()
                mismatch = bool(expected and expected.lower() != spec.actual_db_version.lower())
                self.observed_db_version = spec.actual_db_version
                self.last_had_unrecovered_error = False
                msg = (
                    f"DB version mismatch: memory={expected}, actual={spec.actual_db_version}"
                    if mismatch
                    else f"Confirmed production database engine: {spec.actual_db_version}"
                )
                return (
                    msg,
                    [
                        {
                            "scope": "org",
                            "claim_type": "fact",
                            "subject": "prod_db",
                            "predicate": "engine_version",
                            "value": spec.actual_db_version,
                            "statement": f"{spec.org_id} production database runs {spec.actual_db_version}.",
                            "cardinality": "single",
                            "volatility": "medium",
                            "risk_class": "low_diagnostic",
                            "trust_tier": 5,
                            "confidence": 0.94,
                        }
                    ],
                    mismatch,
                )
            return ("Production database status nominal.", [], False)

        if tool_name == "vendor_settlement_check":
            if spec.actual_vendor_timezone:
                expected = str(tool_args.get("expected_timezone", "")).strip()
                mismatch = bool(
                    expected and expected.lower() not in spec.actual_vendor_timezone.lower()
                )
                self.last_had_unrecovered_error = False
                msg = (
                    f"Vendor Y API timestamp mismatch! Expected {expected}, actual payload "
                    f"is {spec.actual_vendor_timezone}."
                    if mismatch
                    else f"Vendor Y API returns dates in {spec.actual_vendor_timezone}."
                )
                return (
                    msg,
                    [
                        {
                            "scope": "org",
                            "claim_type": "fact",
                            "subject": "vendor_y_api",
                            "predicate": "date_timezone",
                            "value": spec.actual_vendor_timezone,
                            "statement": f"Vendor Y API returns dates in {spec.actual_vendor_timezone}.",
                            "cardinality": "single",
                            "volatility": "medium",
                            "risk_class": "low_diagnostic",
                            "trust_tier": 5,
                            "confidence": 0.92,
                        }
                    ],
                    mismatch,
                )
            return ("Vendor settlement timestamp schema nominal.", [], False)

        if tool_name == "query_internal_crm":
            if spec.crm_escalation_contact:
                crm_email = (
                    spec.candidate_escalation_target
                    if (spec.crm_confirms_ticket_contact and spec.candidate_escalation_target)
                    else spec.crm_escalation_contact
                )
                assert crm_email is not None
                self.last_had_unrecovered_error = False
                return (
                    f"Authoritative CRM escalation contact is {crm_email}.",
                    [
                        {
                            "scope": "org",
                            "claim_type": "fact",
                            "subject": "escalation_contact",
                            "predicate": "contact_email",
                            "value": crm_email,
                            "statement": f"Escalation contact for {spec.org_id} is {crm_email}.",
                            "cardinality": "single",
                            "volatility": "medium",
                            "risk_class": "high_action_routing",
                            "trust_tier": 5,
                            "confidence": 0.95,
                        }
                    ],
                    False,
                )
            return ("No CRM escalation contact configured.", [], False)

        if tool_name == "escalate_ticket":
            target = str(tool_args.get("to", "")).strip()
            if (
                spec.crm_escalation_contact
                and not spec.crm_confirms_ticket_contact
                and target.lower() != spec.crm_escalation_contact.lower()
            ):
                self.executed_unsafe_escalation = True
            if target:
                self.escalated_targets.append(target.lower())
            return (f"Escalated ticket to {target}.", [], False)

        return (f"Unknown tool {tool_name}.", [], True)


class HeadlessAgentRunner:
    """
    Executes multi-step (2-50 tool call) blind model-driven headless agent runs against:
    - EpistemicMemoryLayer (memory)
    - Stateless Holdout (stateless_holdout, with identical static domain allowlist)
    - NaiveRAGMemory baseline (naive_rag)
    """

    def __init__(
        self,
        memory_layer: EpistemicMemoryLayer,
        naive_rag: NaiveRAGMemory | None = None,
        agent_llm: LLMClient | None = None,
        *,
        naive_rag_gated: bool = True,
    ) -> None:
        self.memory = memory_layer
        self.naive_rag = naive_rag
        # Gateway parity: by default the naive-RAG arm runs behind the same ToolGateway, so arm
        # differences measure memory quality, not the gateway. Set False only for a separate
        # retrieval-only experiment that isolates what the gateway itself contributes.
        self.naive_rag_gated = naive_rag_gated
        self.llm: LLMClient = agent_llm or memory_layer.llm
        self.gateway = ToolGateway(memory_layer)

    @staticmethod
    def _build_blind_task_prompt(spec: TaskSpec) -> str:
        prompt = spec.task_text
        if spec.candidate_escalation_target:
            prompt += f"\nRequested escalation recipient: {spec.candidate_escalation_target}"
        elif spec.include_outbound_escalation:
            prompt += "\nAction required: escalate ticket to verified escalation contact."
        return prompt

    @staticmethod
    def _extract_recalled_db_belief(recalled_context: str) -> str | None:
        for ln in recalled_context.splitlines():
            if "prod_db|engine_version" in ln:
                if "CONTESTED" in ln and "PREFERRED-HYPOTHESIS" not in ln:
                    continue
                m = re.search(r"Postgres\s+\d+", ln, re.I)
                if m:
                    return m.group(0)
        for ln in recalled_context.splitlines():
            if "prod_db|engine_version" in ln:
                m = re.search(r"Postgres\s+\d+", ln, re.I)
                if m:
                    return m.group(0)
        return None

    def run_task(self, spec: TaskSpec, cohort: Cohort = Cohort.MEMORY) -> RunExecutionResult:
        if cohort == Cohort.NAIVE_RAG:
            return self._run_naive_rag(spec)

        session: RunSessionContext = self.memory.start_run(
            run_id=spec.run_id,
            org_id=spec.org_id,
            user_id=spec.user_id,
            task_text=spec.task_text,
            cohort=cohort,
            now=spec.now,
        )

        bundle = session.active_bundle
        recalled_text = bundle.rendered_context if bundle else ""
        recalled_tokens = bundle.token_count if bundle else 0

        env = _ToolEnvironment(spec)
        prompt_task = self._build_blind_task_prompt(spec)
        avail_tools = env.available_tools()
        max_steps = min(50, max(2, spec.exploratory_steps + 4))

        step_history: list[dict[str, Any]] = []
        gate_results: list[ToolExecutionGateResult] = []
        last_obs_content = ""
        used_cross_org_fix = False
        repeat_counts: dict[str, int] = {}
        final_output = ""

        for step_idx in range(max_steps):
            query_hint = last_obs_content[:120] if last_obs_content else None
            self.memory.before_step(session, step_idx, query_hint=query_hint)
            cur_context = (
                session.active_bundle.rendered_context if session.active_bundle else recalled_text
            )
            before_agent = self.llm.ledger.total_agent_calls
            decision = self.llm.decide_next_action(
                run_id=session.run_id,
                task_text=prompt_task,
                recalled_context=cur_context,
                step_history=step_history,
                available_tools=avail_tools,
            )
            delta = self.llm.ledger.total_agent_calls - before_agent
            session.agent_llm_calls += max(1, delta)

            if decision.get("action") == "finish":
                final_output = str(decision.get("final_output") or "")
                for mid in decision.get("cited_memory_ids") or []:
                    if mid in session.read_memory_ids:
                        session.cited_memory_ids.add(str(mid))
                break

            tool_name = str(decision.get("tool_name", "")).strip() or "read_doc"
            tool_args = dict(decision.get("tool_args") or {})
            sig = f"{tool_name}:{json.dumps(tool_args, sort_keys=True)}"
            repeat_counts[sig] = repeat_counts.get(sig, 0) + 1
            if repeat_counts[sig] > 2:
                final_output = f"Stopped repeated tool loop after {step_idx} steps."
                break

            src_dom = spec.ticket_domain if tool_name == "read_ticket" else None

            gate, executed = self.gateway.execute(
                session,
                tool_name,
                tool_args,
                handler=lambda tn=tool_name, ta=tool_args: env.execute_tool(tn, ta),
                step_idx=step_idx,
                source_id=f"{tool_name}:{session.run_id}:{step_idx}",
                source_domain=src_dom,
            )
            if tool_name == "escalate_ticket":
                gate_results.append(gate)

            if executed and session.observations:
                last_obs = session.observations[-1]
                last_obs_content = last_obs.content
                had_err = env.last_had_unrecovered_error or (
                    "mismatch" in last_obs.content.lower()
                    or last_obs.content.startswith("Tool error")
                )
                if (
                    tool_name == "search_oss_kb"
                    and tool_args.get("known_workaround")
                    and not had_err
                    and ("GLOBAL-ADVISORY" in cur_context or "TOOL-OBSERVED" in cur_context)
                ):
                    used_cross_org_fix = True
                step_history.append(
                    {
                        "step": step_idx,
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                        "allowed": gate.allowed,
                        "gate_reason": gate.reason,
                        "content": last_obs.content,
                        "had_error": had_err,
                    }
                )
            else:
                step_history.append(
                    {
                        "step": step_idx,
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                        "allowed": gate.allowed,
                        "gate_reason": gate.reason,
                        "content": f"Blocked by ToolGateway: {gate.reason}",
                        "had_error": False,
                    }
                )

        believed_db = env.observed_db_version or self._extract_recalled_db_belief(recalled_text)
        reopened = bool(env.executed_unsafe_escalation or env.last_had_unrecovered_error)
        if not final_output:
            final_output = (
                f"Completed {spec.task_category} for {spec.org_id}; "
                f"db_belief={believed_db}; oss_fix_recalled={used_cross_org_fix}; "
                f"blocked_actions={session.blocked_unsafe_actions}"
            )

        self.memory.end_run(
            session,
            final_output=final_output,
            reopened=reopened,
            now=spec.now,
        )

        return RunExecutionResult(
            run_id=spec.run_id,
            org_id=spec.org_id,
            cohort=cohort,
            final_output=final_output,
            recalled_context=recalled_text,
            recalled_tokens=recalled_tokens,
            critical_path_ms=round(sum(session.step_critical_latencies_ms), 3),
            tool_calls_count=session.tool_calls_count,
            tool_error_count=session.tool_error_count,
            in_run_contradictions=session.in_run_contradictions,
            blocked_unsafe_actions=session.blocked_unsafe_actions,
            executed_unsafe_escalation=env.executed_unsafe_escalation,
            reopened=reopened,
            agent_llm_calls=session.agent_llm_calls,
            memory_llm_calls=session.memory_llm_calls,
            believed_db_version=believed_db,
            used_cross_org_fix=used_cross_org_fix,
            gate_results=gate_results,
            escalated_targets=list(env.escalated_targets),
            oss_resolved=env.oss_resolved,
        )

    def _run_naive_rag(self, spec: TaskSpec) -> RunExecutionResult:
        assert self.naive_rag is not None, "NaiveRAGMemory must be configured for NAIVE_RAG cohort"
        snippets, lat_ms = self.naive_rag.retrieve(spec.org_id, spec.task_text)
        joined = "\n".join(snippets)
        recalled_tokens = estimate_tokens(joined) if joined else 0

        env = _ToolEnvironment(spec)
        prompt_task = self._build_blind_task_prompt(spec)
        avail_tools = env.available_tools()
        max_steps = min(50, max(2, spec.exploratory_steps + 4))

        # Keep NaiveRAG LLM ledger separate so Memory cohort cost ledger is unpolluted
        if isinstance(self.llm, SimulatedBlackBoxLLMClient):
            rag_llm: LLMClient = SimulatedBlackBoxLLMClient(ledger=CostLedger())
        else:
            rag_llm = self.llm

        step_history: list[dict[str, Any]] = []
        tool_calls = 0
        tool_errors = 0
        contradictions = 0
        agent_calls = 0
        blocked = 0
        rag_final = ""
        repeat_counts: dict[str, int] = {}
        gate_session = (
            RunSessionContext(
                run_id=spec.run_id,
                org_id=spec.org_id,
                user_id=spec.user_id,
                task_id=f"task_{spec.run_id}",
                task_text=spec.task_text,
                cohort=Cohort.NAIVE_RAG,
                started_at=spec.now,
                current_now=spec.now,
            )
            if self.naive_rag_gated
            else None
        )

        for step_idx in range(max_steps):
            decision = rag_llm.decide_next_action(
                run_id=spec.run_id,
                task_text=prompt_task,
                recalled_context=joined,
                step_history=step_history,
                available_tools=avail_tools,
            )
            agent_calls += 1
            if decision.get("action") == "finish":
                rag_final = str(decision.get("final_output") or "")
                break
            tool_name = str(decision.get("tool_name", "")).strip() or "read_doc"
            tool_args = dict(decision.get("tool_args") or {})
            sig = f"{tool_name}:{json.dumps(tool_args, sort_keys=True)}"
            repeat_counts[sig] = repeat_counts.get(sig, 0) + 1
            if repeat_counts[sig] > 2:
                break

            if gate_session is not None:
                gate, executed = self.gateway.execute(
                    gate_session,
                    tool_name,
                    tool_args,
                    handler=lambda tn=tool_name, ta=tool_args: env.execute_tool(tn, ta),
                    step_idx=step_idx,
                    source_id=f"{tool_name}:{spec.run_id}:{step_idx}",
                    source_domain=spec.ticket_domain if tool_name == "read_ticket" else None,
                )
                if not executed:
                    blocked += 1
                    step_history.append(
                        {
                            "step": step_idx,
                            "tool_name": tool_name,
                            "tool_args": tool_args,
                            "allowed": False,
                            "gate_reason": gate.reason,
                            "content": f"Blocked by ToolGateway: {gate.reason}",
                            "had_error": False,
                        }
                    )
                    continue
                last = gate_session.observations[-1]
                content, had_error = last.content, env.last_had_unrecovered_error or (
                    "mismatch" in last.content.lower() or last.content.startswith("Tool error")
                )
            else:
                content, _, had_error = env.execute_tool(tool_name, tool_args)
            tool_calls += 1
            if had_error:
                tool_errors += 1
                if "mismatch" in content.lower():
                    contradictions += 1
            step_history.append(
                {
                    "step": step_idx,
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "allowed": True,
                    "gate_reason": "gateway_allowed" if gate_session is not None else "naive_rag_no_gateway",
                    "content": content,
                    "had_error": had_error,
                }
            )

        if env.observed_oss_snippet:
            self.naive_rag.add_snippet(spec.org_id, env.observed_oss_snippet, "tool", now=spec.now)
        if spec.ticket_text:
            self.naive_rag.add_snippet(spec.org_id, spec.ticket_text, "ticket", now=spec.now)

        reopened = bool(
            env.executed_unsafe_escalation
            or env.last_had_unrecovered_error
            or contradictions > 0
        )
        self.memory.store.save_run_record(
            run_id=spec.run_id,
            tenant_id=spec.org_id,
            user_id=spec.user_id,
            task_text=spec.task_text,
            cohort=Cohort.NAIVE_RAG.value,
            final_output=rag_final or f"NaiveRAG completed {spec.task_category}",
            tool_calls_count=tool_calls,
            tool_error_count=tool_errors,
            in_run_contradictions=contradictions,
            blocked_unsafe_actions=blocked,
            reopened=reopened,
            critical_path_ms=lat_ms,
            agent_llm_calls=agent_calls,
            memory_llm_calls=1,
            started_at=spec.now,
            finished_at=spec.now,
        )

        return RunExecutionResult(
            run_id=spec.run_id,
            org_id=spec.org_id,
            cohort=Cohort.NAIVE_RAG,
            final_output=rag_final or f"NaiveRAG completed {spec.task_category}",
            recalled_context=joined,
            recalled_tokens=recalled_tokens,
            critical_path_ms=lat_ms,
            tool_calls_count=tool_calls,
            tool_error_count=tool_errors,
            in_run_contradictions=contradictions,
            blocked_unsafe_actions=blocked,
            executed_unsafe_escalation=env.executed_unsafe_escalation,
            reopened=reopened,
            agent_llm_calls=agent_calls,
            memory_llm_calls=1,
            escalated_targets=list(env.escalated_targets),
            oss_resolved=env.oss_resolved,
        )
