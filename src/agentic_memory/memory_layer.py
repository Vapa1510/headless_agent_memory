from __future__ import annotations

import json
import math
import re
import time
from datetime import datetime
from typing import Any

from rapidfuzz import fuzz

from agentic_memory.llm_client import CostLedger, LLMClient, SimulatedBlackBoxLLMClient
from agentic_memory.models import (
    GLOBAL_TENANT_ID,
    MAX_AVG_LLM_CALLS_PER_RUN,
    MAX_CONTEXT_TOKENS,
    PACKING_SOFT_CAP_TOKENS,
    CandidateClaim,
    Cardinality,
    ClaimType,
    Cohort,
    ContainmentAuditEvent,
    FeedbackVerdict,
    HumanFeedback,
    MemoryClaim,
    MemoryStatus,
    RecallBundle,
    RecallItem,
    RiskClass,
    RunSessionContext,
    Scope,
    SourceType,
    TenantPolicy,
    ToolExecutionGateResult,
    ToolObservation,
    TrustTier,
    Volatility,
    WriteOutcome,
    new_id,
    utcnow,
)
from agentic_memory.policy import (
    DEFAULT_AUTHORITY_REGISTRY,
    EXTERNAL_UNTRUSTED_SOURCE_TYPES,
    HALF_LIFE_DAYS,
    TOOL_CAPABILITIES,
    SourceAuthorityRegistry,
    TokenCounter,
    ToolCapability,
    _collect_arg_strings,
    authorize_tool_action,
    canonicalize_predicate,
    canonicalize_slot,
    canonicalize_subject,
    claim_freshness,
    detect_prompt_injection,
    evidence_confidence,
    evidence_fingerprint,
    global_value_supported,
    hash_source_id,
    infer_minimum_risk_class,
    normalize_candidate_authority,
    sanitize_for_global,
    validate_global_metadata_field,
    validate_global_safety,
)
from agentic_memory.store_protocol import MemoryStoreProtocol

TOKEN_RE = re.compile(r"[a-z0-9_]{2,}")
_HALF_LIFE_STR = {k.value: v for k, v in HALF_LIFE_DAYS.items()}


class EpistemicMemoryLayer:
    """
    Epistemic belief-maintenance and error-containment memory layer
    sitting between the headless agent loop and persistent storage.
    """

    def __init__(
        self,
        store: MemoryStoreProtocol,
        llm_client: LLMClient | None = None,
        *,
        k_anonymity: int = 5,
        max_active_per_org: int = 500,
        max_negative_per_org: int = 100,
        max_global_advisories: int = 2000,
        trust_registry: SourceAuthorityRegistry | None = None,
        token_counter: TokenCounter | None = None,
        stale_revalidation: bool = True,
    ) -> None:
        self.store: MemoryStoreProtocol = store
        self.llm = llm_client or SimulatedBlackBoxLLMClient()
        self.ledger: CostLedger = self.llm.ledger
        self.ledger.attach_store(self.store)
        self.k_anonymity = k_anonymity
        self.max_active_per_org = max_active_per_org
        self.max_negative_per_org = max_negative_per_org
        self.max_global_advisories = max_global_advisories
        self.trust_registry = trust_registry or DEFAULT_AUTHORITY_REGISTRY
        if hasattr(self.llm, "_authority_registry"):
            self.llm._authority_registry = self.trust_registry
        self.token_counter = token_counter or TokenCounter()
        self.last_gap_reextraction: dict[str, Any] | None = None
        # Ablation switch for experiments only: False hides staleness labels/revalidation requests
        # in recall. The gateway's freshness requirement for actions stays on regardless.
        self.stale_revalidation = stale_revalidation

    def register_tenant(self, policy: TenantPolicy) -> None:
        self.store.upsert_tenant(policy)

    def _tokenize(self, text: str) -> set[str]:
        return {
            t
            for t in TOKEN_RE.findall(text.lower())
            if t not in {"the", "and", "for", "with", "this", "that", "from", "task", "check"}
        }

    def _format_claim_line(
        self, claim: MemoryClaim, eff_conf: float, now: datetime | None = None
    ) -> str:
        if claim.claim_type == ClaimType.NEGATIVE_CONSTRAINT:
            tag = "ANTI-MEMORY/REFUTED"
        elif claim.status == MemoryStatus.CANDIDATE_UNVERIFIED:
            tag = "UNVERIFIED-DO-NOT-ACT"
        elif claim.status == MemoryStatus.CONTESTED:
            hyp = " | PREFERRED-HYPOTHESIS" if claim.metadata.get("working_hypothesis") else ""
            tag = f"CONTESTED{hyp}"
        elif claim.scope == Scope.GLOBAL:
            org_cnt = claim.metadata.get("source_org_count", "5+")
            tag = f"GLOBAL-ADVISORY({org_cnt}orgs)"
        elif claim.trust_tier >= TrustTier.INTERNAL_API_VERIFIED:
            tag = f"VERIFIED-{claim.trust_tier.name}"
        elif claim.trust_tier == TrustTier.TOOL_OBSERVED:
            tag = "TOOL-OBSERVED"
        else:
            tag = "PROVISIONAL"

        hint = ""
        if claim.metadata.get("requires_verification"):
            hint = f" [ACTION: {claim.metadata.get('verification_hint', 'verify before acting')}]"
        # Confidence (evidence quality) and freshness (age of last confirmation) are shown
        # separately; a stale fact is labelled and must be re-checked before it is stated or used.
        conf = evidence_confidence(claim) if claim.scope != Scope.GLOBAL else eff_conf
        fresh = claim_freshness(claim, now) if now is not None else None
        age_part = f" | confirmed={int(fresh['age_days'])}d_ago" if fresh is not None else ""
        if (
            self.stale_revalidation
            and fresh is not None
            and fresh["stale"]
            and claim.status == MemoryStatus.ACTIVE
        ):
            tag = f"{tag} | STALE-REVALIDATE"
            hint += (
                f" [ACTION: last confirmed {int(fresh['age_days'])}d ago (re-check window "
                f"{int(fresh['window_days'])}d) - re-check with an authoritative tool before stating or acting]"
            )
        return (
            f"- [{tag} | id={claim.memory_id} | slot={claim.slot_key} | "
            f"conf={conf:.2f}{age_part}] "
            f"{claim.statement}{hint}"
        )

    def _score_and_pack_rows(
        self,
        *,
        org_id: str,
        run_id: str,
        query_text: str,
        org_rows: list[tuple[Any, ...]],
        global_rows: list[tuple[Any, ...]],
        now: datetime,
    ) -> RecallBundle:
        t0 = time.perf_counter()
        q_tokens = self._tokenize(query_text)
        q_lower = query_text.lower()
        q_token_count = max(1, len(q_tokens))
        q_exact_tokens = set(TOKEN_RE.findall(q_lower))
        q_words_str = f" {' '.join(re.sub(r'[^a-z0-9]+', ' ', q_lower).split())} "
        small_pool = len(org_rows) <= 45
        now_ts = now.timestamp()

        org_hits_list: list[set[str]] = []
        df: dict[str, int] = dict.fromkeys(q_tokens, 0)
        if q_tokens and org_rows:
            for r in org_rows:
                blob_tokens = set(TOKEN_RE.findall(f"{r[7]} {r[8]} {r[9]} {r[11]}".lower()))
                hits = q_tokens & blob_tokens
                org_hits_list.append(hits)
                for t in hits:
                    df[t] += 1
            n_docs = len(org_rows)
            idf_map = {t: (math.log((n_docs + 1.0) / (df[t] + 1.0)) + 0.5) for t in q_tokens}
            matched_idf_sum = sum(idf_map[t] for t in q_tokens if df[t] > 0)
        else:
            idf_map = {}
            matched_idf_sum = 0.0

        def _row_eff_conf(r: tuple[Any, ...], meta: dict[str, Any]) -> float:
            status_str = r[16]
            if status_str in ("REFUTED", "ARCHIVED", "QUARANTINED"):
                return 0.0
            trust_int = int(r[15])
            risk_str = r[14]
            if status_str == "CANDIDATE_UNVERIFIED":
                ceiling = 0.25
            elif trust_int == 1:
                if risk_str == "high_action_routing":
                    ceiling = 0.25
                elif risk_str == "low_diagnostic" and meta.get("error_signature"):
                    ceiling = 0.65
                else:
                    ceiling = 0.40
            elif trust_int == 2:
                ceiling = 0.50 if int(r[18]) <= 1 else 0.72
            elif trust_int == 3:
                ceiling = 0.90
            elif trust_int == 4:
                ceiling = 0.88
            elif trust_int == 5:
                ceiling = 0.96
            else:
                ceiling = 1.0

            half_life = _HALF_LIFE_STR[r[13]]
            if half_life is None:
                decay = 1.0
            else:
                updated_ts = datetime.fromisoformat(r[25]).timestamp()
                age_days = max(0.0, (now_ts - updated_ts) / 86400.0)
                decay = 0.5 ** (age_days / half_life)

            if trust_int == 1 and risk_str == "high_action_routing":
                corrob = 1.0
            else:
                ev = int(r[18])
                corrob = 1.0 if ev <= 1 else (1.0 + 0.08 * math.log2(ev))
            chits = int(r[29])
            penalty = 1.0 if chits == 0 else (0.65**chits)
            raw = float(r[17]) * decay * corrob * penalty
            return round(min(ceiling, max(0.0, raw)), 4)

        def _score_row_base(
            r: tuple[Any, ...], pre_hits: set[str] | None = None
        ) -> tuple[float, float, tuple[Any, ...]]:
            meta_raw = r[32]
            meta = {} if meta_raw == "{}" else json.loads(meta_raw)
            eff_conf = _row_eff_conf(r, meta)
            subj = r[7]
            stmt = r[11]
            if q_tokens:
                if pre_hits is not None:
                    overlap = (
                        sum(idf_map[t] for t in pre_hits) / matched_idf_sum
                        if matched_idf_sum > 0.0
                        else 0.0
                    )
                else:
                    blob = f"{subj} {r[8]} {r[9]} {stmt}".lower()
                    blob_tokens = set(TOKEN_RE.findall(blob))
                    overlap = len(q_tokens & blob_tokens) / q_token_count
            else:
                overlap = 0.2
            subj_low = subj.lower()
            if subj_low in q_exact_tokens:
                subj_match = True
            elif "_" in subj_low or "-" in subj_low:
                subj_phrase = subj_low.replace("_", " ").replace("-", " ")
                subj_match = f" {subj_phrase} " in q_words_str
            else:
                subj_match = False
            rel = 0.55 * min(1.0, overlap) + (0.25 if subj_match else 0.0) + 0.25 * eff_conf
            if r[5] == "negative_constraint":
                rel += 0.35
            if meta.get("working_hypothesis"):
                rel += 0.20
            helps = int(r[28])
            if helps > 0:
                rel += min(0.15, 0.03 * helps)
            return round(rel, 4), eff_conf, r

        def _rerank_fuzzy(pool: list[tuple[float, float, tuple[Any, ...]]]) -> list[tuple[float, float, tuple[Any, ...]]]:
            if not pool:
                return pool
            pool.sort(key=lambda x: x[0], reverse=True)
            limit = len(pool) if small_pool else min(len(pool), 60)
            reranked: list[tuple[float, float, tuple[Any, ...]]] = []
            for idx, (rel, eff_conf, r) in enumerate(pool):
                if idx < limit:
                    fuzzy_boost = (fuzz.token_set_ratio(q_lower, r[11].lower()) / 100.0) * 0.25
                    reranked.append((round(rel + fuzzy_boost, 4), eff_conf, r))
                else:
                    reranked.append((rel, eff_conf, r))
            return reranked

        neg_pool: list[tuple[float, float, tuple[Any, ...]]] = []
        active_pool: list[tuple[float, float, tuple[Any, ...]]] = []
        contested_pool: list[tuple[float, float, tuple[Any, ...]]] = []
        global_pool: list[tuple[float, float, tuple[Any, ...]]] = []

        for idx, r in enumerate(org_rows):
            pre_hits = org_hits_list[idx] if org_hits_list else None
            scored = _score_row_base(r, pre_hits=pre_hits)
            if r[5] == "negative_constraint":
                neg_pool.append(scored)
            elif r[16] == "ACTIVE":
                active_pool.append(scored)
            elif r[16] in ("CONTESTED", "CANDIDATE_UNVERIFIED"):
                contested_pool.append(scored)

        for gr in global_rows:
            global_pool.append(_score_row_base(gr))

        neg_pool = _rerank_fuzzy(neg_pool)
        active_pool = _rerank_fuzzy(active_pool)
        contested_pool = _rerank_fuzzy(contested_pool)
        global_pool = _rerank_fuzzy(global_pool)

        neg_pool.sort(key=lambda x: x[0], reverse=True)
        active_pool.sort(key=lambda x: x[0], reverse=True)
        contested_pool.sort(key=lambda x: (x[0], x[1]), reverse=True)
        global_pool.sort(key=lambda x: x[0], reverse=True)

        header_overhead = 80
        budgets = {
            "negative_constraints": 550,
            "active_org": 1850,
            "contested_unverified": 570,
            "global_advisories": 550,
        }
        selected: dict[str, list[RecallItem]] = {k: [] for k in budgets}
        used_tokens = header_overhead
        selected_ids: set[str] = set()
        verification_reqs: list[dict[str, str]] = []

        def _pack_bucket(
            bucket_name: str, pool: list[tuple[float, float, tuple[Any, ...]]], cap: int
        ) -> int:
            nonlocal used_tokens
            bucket_tokens = 0
            misses = 0
            for rel, eff_conf, r in pool:
                if cap - bucket_tokens < 24 or PACKING_SOFT_CAP_TOKENS - used_tokens < 24:
                    break
                mid = r[0]
                if mid in selected_ids:
                    continue
                claim = self.store.hydrate_claim_tuple(r)
                line = self._format_claim_line(claim, eff_conf, now)
                tok = self.token_counter.count(line)
                if bucket_tokens + tok > cap or used_tokens + tok > PACKING_SOFT_CAP_TOKENS:
                    misses += 1
                    if misses >= 5:
                        break
                    continue
                misses = 0
                selected[bucket_name].append(
                    RecallItem.model_construct(
                        claim=claim,
                        effective_confidence=eff_conf,
                        relevance_score=rel,
                        bucket=bucket_name,
                        rendered_line=line,
                        token_estimate=tok,
                    )
                )
                selected_ids.add(mid)
                bucket_tokens += tok
                used_tokens += tok
                stale = (
                    self.stale_revalidation
                    and claim.status == MemoryStatus.ACTIVE
                    and claim.scope != Scope.GLOBAL
                    and claim_freshness(claim, now)["stale"]
                )
                if claim.metadata.get("requires_verification") or stale:
                    verification_reqs.append(
                        {
                            "memory_id": claim.memory_id,
                            "slot_key": claim.slot_key,
                            "value": claim.value,
                            "hint": (
                                "stale: re-check with an authoritative tool before stating or acting"
                                if stale
                                else str(claim.metadata.get("verification_hint", ""))
                            ),
                        }
                    )
            return bucket_tokens

        _pack_bucket("negative_constraints", neg_pool, budgets["negative_constraints"])
        _pack_bucket("active_org", active_pool, budgets["active_org"])
        _pack_bucket("contested_unverified", contested_pool, budgets["contested_unverified"])
        _pack_bucket("global_advisories", global_pool, budgets["global_advisories"])

        if used_tokens < PACKING_SOFT_CAP_TOKENS - 60:
            _pack_bucket("active_org", active_pool, PACKING_SOFT_CAP_TOKENS - used_tokens)

        sections: list[str] = []
        section_titles = [
            ("negative_constraints", "## Critical Warnings & Refuted Beliefs"),
            ("active_org", "## Verified & Active Org/User Memory"),
            ("contested_unverified", "## Contested & Unverified Candidates (Verify Before Acting)"),
            ("global_advisories", "## Cross-Org Technical Advisories (Anonymized)"),
        ]
        all_items: list[RecallItem] = []
        for key, title in section_titles:
            items = selected[key]
            if items:
                sections.append(title + "\n" + "\n".join(it.rendered_line for it in items))
                all_items.extend(items)

        rendered = "\n\n".join(sections)
        total_tokens = self.token_counter.count(rendered) if rendered else 0

        while (total_tokens > PACKING_SOFT_CAP_TOKENS or total_tokens > MAX_CONTEXT_TOKENS) and all_items:
            droppable = [i for i in all_items if i.bucket != "negative_constraints"] or all_items
            victim = min(droppable, key=lambda it: it.relevance_score)
            all_items.remove(victim)
            selected[victim.bucket].remove(victim)
            sections = []
            for key, title in section_titles:
                if selected[key]:
                    sections.append(
                        title + "\n" + "\n".join(it.rendered_line for it in selected[key])
                    )
            rendered = "\n\n".join(sections)
            total_tokens = self.token_counter.count(rendered) if rendered else 0

        surviving_ids = {it.claim.memory_id for it in all_items}
        verification_reqs = [vr for vr in verification_reqs if vr["memory_id"] in surviving_ids]

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        return RecallBundle.model_construct(
            org_id=org_id,
            run_id=run_id,
            items=all_items,
            rendered_context=rendered,
            token_count=total_tokens,
            latency_ms=elapsed_ms,
            read_memory_ids=[it.claim.memory_id for it in all_items],
            verification_requests=verification_reqs,
        )

    def start_run(
        self,
        *,
        run_id: str,
        org_id: str,
        task_text: str,
        user_id: str | None = None,
        task_id: str | None = None,
        cohort: Cohort = Cohort.MEMORY,
        now: datetime | None = None,
    ) -> RunSessionContext:
        """
        Critical-path run entry (0 LLM calls, 0 SQLite write locks).
        """
        ts = now or utcnow()
        if cohort == Cohort.MEMORY:
            self.ledger.register_run(run_id)
        t0 = time.perf_counter()

        session = RunSessionContext(
            run_id=run_id,
            org_id=org_id,
            user_id=user_id,
            task_id=task_id or new_id("task"),
            task_text=task_text,
            cohort=cohort,
            started_at=ts,
            current_now=ts,
            seen_query_tokens=self._tokenize(task_text),
        )

        if cohort == Cohort.STATELESS_HOLDOUT:
            empty = RecallBundle(org_id=org_id, run_id=run_id, latency_ms=0.05)
            session.initial_bundle = empty
            session.active_bundle = empty
            session.step_critical_latencies_ms.append(round((time.perf_counter() - t0) * 1000.0, 3))
            return session

        org_rows, global_rows = self.store.query_recall_rows(
            org_id=org_id, user_id=user_id, query_text=task_text, now=ts
        )
        bundle = self._score_and_pack_rows(
            org_id=org_id,
            run_id=run_id,
            query_text=task_text,
            org_rows=org_rows,
            global_rows=global_rows,
            now=ts,
        )
        total_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        bundle.latency_ms = total_ms
        session.initial_bundle = bundle
        session.active_bundle = bundle
        session.read_memory_ids.update(bundle.read_memory_ids)
        session.step_critical_latencies_ms.append(total_ms)
        return session

    def before_step(
        self,
        session: RunSessionContext,
        step_idx: int,
        query_hint: str | None = None,
    ) -> RecallBundle:
        """
        Critical-path inter-step recall (0 LLM calls).
        Uses cached working set unless novel tool tokens warrant an incremental refresh.
        """
        t0 = time.perf_counter()
        if session.cohort == Cohort.STATELESS_HOLDOUT:
            ms = round((time.perf_counter() - t0) * 1000.0, 3)
            session.step_critical_latencies_ms.append(ms)
            return session.active_bundle or RecallBundle(org_id=session.org_id, run_id=session.run_id)

        if query_hint and session.refresh_count < 2:
            new_tokens = self._tokenize(query_hint) - session.seen_query_tokens
            if len(new_tokens) >= 2:
                session.seen_query_tokens.update(new_tokens)
                session.refresh_count += 1
                merged_query = f"{session.task_text} {query_hint}"
                org_rows, global_rows = self.store.query_recall_rows(
                    org_id=session.org_id,
                    user_id=session.user_id,
                    query_text=merged_query,
                    now=session.current_now,
                )
                bundle = self._score_and_pack_rows(
                    org_id=session.org_id,
                    run_id=session.run_id,
                    query_text=merged_query,
                    org_rows=org_rows,
                    global_rows=global_rows,
                    now=session.current_now,
                )
                ms = round((time.perf_counter() - t0) * 1000.0, 3)
                bundle.latency_ms = ms
                session.active_bundle = bundle
                session.read_memory_ids.update(bundle.read_memory_ids)
                session.step_critical_latencies_ms.append(ms)
                return bundle

        ms = round((time.perf_counter() - t0) * 1000.0, 3)
        session.step_critical_latencies_ms.append(ms)
        return session.active_bundle or RecallBundle(org_id=session.org_id, run_id=session.run_id)

    def check_tool_action(
        self,
        session: RunSessionContext,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> ToolExecutionGateResult:
        """Critical-path capability gate preventing unverified external memory execution."""
        t0 = time.perf_counter()
        tenant_policy = self.store.get_tenant(session.org_id)
        items = list(session.active_bundle.items) if session.active_bundle else []
        cap = TOOL_CAPABILITIES.get(tool_name)
        if cap is not None and cap != ToolCapability.READ:
            # Durable refutations are memory, so the stateless holdout does not get them.
            # In-run verification from attested internal tools is NOT memory: every arm gets it,
            # so the gateway is identical across arms and only memory differs.
            durable_refs = []
            if session.cohort != Cohort.STATELESS_HOLDOUT:
                serialized = " ".join(_collect_arg_strings(tool_args))
                durable_refs = self.store.find_matching_refutations(
                    session.org_id,
                    serialized,
                    exclude_ids={it.claim.memory_id for it in items},
                )
            for ref_claim in durable_refs:
                items.append(
                    RecallItem.model_construct(
                        claim=ref_claim,
                        effective_confidence=0.98,
                        relevance_score=1.0,
                        bucket="negative_constraints",
                        rendered_line="",
                        token_estimate=0,
                    )
                )
            for obs in session.observations:
                obs_trust = self.trust_registry.resolve_observation(
                    obs, expected_tenant_id=session.org_id, expected_run_id=session.run_id
                )
                if obs_trust < TrustTier.INTERNAL_API_VERIFIED:
                    continue
                for sf in obs.structured_facts:
                    if not self.trust_registry.is_authoritative_for(
                        obs.connector_id,
                        str(sf.get("subject", "")),
                        str(sf.get("predicate", "")),
                        str(sf.get("value", "")),
                    ):
                        continue  # authentic source, but not an authority for this predicate
                    in_run_claim = MemoryClaim(
                        memory_id=f"inrun:{obs.source_id}",
                        tenant_id=session.org_id,
                        scope=Scope.ORG,
                        claim_type=ClaimType(sf.get("claim_type", "fact")),
                        slot_key=canonicalize_slot(
                            Scope.ORG,
                            session.org_id,
                            str(sf.get("subject", "")),
                            str(sf.get("predicate", "")),
                        ),
                        subject=str(sf.get("subject", "")),
                        predicate=str(sf.get("predicate", "")),
                        value=str(sf.get("value", "")),
                        value_hash="inrun",
                        statement=str(sf.get("statement", "")),
                        cardinality=Cardinality.SINGLE,
                        volatility=Volatility.MEDIUM,
                        risk_class=RiskClass.HIGH_ACTION_ROUTING,
                        trust_tier=obs_trust,
                        status=MemoryStatus.ACTIVE,
                        base_confidence=0.95,
                        created_in_run_id=session.run_id,
                        created_at=obs.observed_at,
                        updated_at=obs.observed_at,
                        metadata={"last_confirmed_at": obs.observed_at.isoformat()},
                        source_type=obs.source_type,
                        source_id=obs.source_id,
                        source_domain=obs.source_domain,
                    )
                    items.append(
                        RecallItem.model_construct(
                            claim=in_run_claim,
                            effective_confidence=0.95,
                            relevance_score=1.0,
                            bucket="active_org",
                            rendered_line="",
                            token_estimate=0,
                        )
                    )
        res = authorize_tool_action(
            tool_name, tool_args, items, tenant_policy, now=session.current_now
        )
        if not res.allowed:
            session.blocked_unsafe_actions += 1
        session.step_critical_latencies_ms.append(round((time.perf_counter() - t0) * 1000.0, 3))
        return res

    def record_tool_observation(
        self,
        session: RunSessionContext,
        observation: ToolObservation,
        *,
        tool_had_error: bool = False,
    ) -> None:
        """Critical-path in-memory observation staging + live contradiction detection."""
        t0 = time.perf_counter()
        session.tool_calls_count += 1
        if tool_had_error:
            session.tool_error_count += 1
        # Ingestion boundary: a seal minted for another tenant or run, a seal with no context,
        # or content changed after sealing all demote the observation to untrusted input.
        if observation.authority_seal and not self.trust_registry.verify_observation(
            observation, expected_tenant_id=session.org_id, expected_run_id=session.run_id
        ):
            observation = observation.model_copy(
                update={"is_external_untrusted": True, "authority_seal": None}
            )
            session.attestation_rejections += 1
        session.observations.append(observation)

        # Check live authoritative tool contradictions against recalled active beliefs
        if (
            observation.source_type in (SourceType.INTERNAL_API, SourceType.TOOL_OUTPUT)
            and session.active_bundle
        ):
            for sf in observation.structured_facts:
                subj = canonicalize_subject(str(sf.get("subject", "")))
                pred = canonicalize_predicate(str(sf.get("predicate", "")))
                val = str(sf.get("value", "")).strip().lower()
                for item in session.active_bundle.items:
                    c = item.claim
                    if (
                        canonicalize_subject(c.subject) == subj
                        and canonicalize_predicate(c.predicate) == pred
                        and c.value.strip().lower() != val
                        and c.status == MemoryStatus.ACTIVE
                    ):
                        session.in_run_contradictions += 1
                        session.cited_memory_ids.add(f"contradict:{c.memory_id}")
        session.step_critical_latencies_ms.append(round((time.perf_counter() - t0) * 1000.0, 3))

    _BOILERPLATE_RE = re.compile(
        r"(?:^internal inspection step\b|routine queue check|^status\s*[:=]?\s*ok\b|"
        r"^heartbeat\b|\bno\s+(?:changes|incidents|errors)\b)",
        re.I,
    )
    _ACK_CHITCHAT_RE = re.compile(
        r"\b(?:thank\s+you|thanks|got\s+it|looks\s+good|closing\s+(?:this\s+)?ticket|acknowledged)\b",
        re.I,
    )
    _STRUCTURAL_SIGNAL_RE = re.compile(
        r"(?:@|://|\b\d{1,3}(?:\.\d{1,3}){3}\b|\b\d+\.\d+\b|\b\d+(?:ms|s|m|h|%|gb|mb|kb|rps)\b"
        r"|`[^`]+`|\b[a-z]+_[a-z0-9_]+\b|\b[a-z]+-[a-z0-9]+-[a-z0-9]+\b"
        r"|\b(?:sk|ghp|xox[baprs])-[A-Za-z0-9_-]{8,}\b"
        r"|\b[A-Z]{2,}[_-][A-Z0-9_]+\b|\b[A-Za-z0-9]+(?:Error|Exception|Failure)\b"
        r"|\b(?:ERROR|WARN|WARNING|FATAL|CRITICAL)\s*:"
        r"|\b(?:HTTP|TLS|SSL|UTC|EST|EDT|PST|CET|CEST|GMT|BST|IST|SEPA|SLA|SQL|DNS|API|DB|S3|SFTP)\b"
        r"|\b\d{1,2}:\d{2}\b|\bv\d+\b|\bport\s+\d{2,5}\b"
        r"|\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:com|net|org|io|co|dev|internal|cloud)\b)",
    )
    _STATE_PREDICATE_RE = re.compile(
        r"\b(?:switched|migrated|moved|updated|changed|deprecated|pinned|blocked|threw|"
        r"upgraded|downgraded|enabled|disabled|configured|installed|deployed|rotated|"
        r"arrives?|produces?|supports?|defaults?|failed|fixed|resolved|"
        r"returns|requires|uses|lands|routes|escalat\w*|always|never|only)\b",
        re.I,
    )
    _SALIENCE_STOPWORDS = frozenset(
        {
            "the", "and", "for", "with", "this", "that", "from", "into", "about",
            "our", "your", "their", "have", "has", "been", "were", "will", "just",
            "very", "much", "here", "there", "please", "help", "today", "now",
        }
    )

    def _is_salient_text(self, text: str, seen_tokens: set[str] | None = None) -> bool:
        stripped = text.strip()
        if len(stripped) < 28 or self._BOILERPLATE_RE.search(stripped):
            return False
        if self._STRUCTURAL_SIGNAL_RE.search(stripped):
            return True
        has_predicate = bool(self._STATE_PREDICATE_RE.search(stripped))
        if self._ACK_CHITCHAT_RE.search(stripped) and not has_predicate:
            return False
        tokens = {t for t in TOKEN_RE.findall(stripped.lower()) if t not in self._SALIENCE_STOPWORDS}
        novel = (tokens - seen_tokens) if seen_tokens else tokens
        return len(tokens) >= 6 and len(novel) >= 2 and has_predicate

    def _should_extract(self, session: RunSessionContext) -> bool:
        if session.cohort == Cohort.STATELESS_HOLDOUT:
            return False
        if not session.observations:
            return False
        if any(o.structured_facts for o in session.observations):
            return True
        has_salient = any(
            self._is_salient_text(o.content, session.seen_query_tokens)
            for o in session.observations
        )
        if not has_salient:
            return False
        if self.ledger.avg_memory_calls_per_run >= (MAX_AVG_LLM_CALLS_PER_RUN - 0.25):
            return any(o.is_external_untrusted for o in session.observations)
        return True

    def _extract_structured_observations_zero_llm(
        self,
        *,
        session: RunSessionContext,
        recalled_claims: list[MemoryClaim],
    ) -> list[CandidateClaim]:
        """Deterministic zero-LLM extraction when observations already supply structured tool payloads."""
        out: list[CandidateClaim] = []
        for obs in session.observations:
            raw_ext = obs.is_external_untrusted or (
                obs.source_type in EXTERNAL_UNTRUSTED_SOURCE_TYPES
            )
            resolved_trust = self.trust_registry.resolve_observation(
                obs, expected_tenant_id=session.org_id, expected_run_id=session.run_id
            )
            is_ext = raw_ext or (resolved_trust == TrustTier.UNTRUSTED_EXTERNAL)
            for sf in obs.structured_facts:
                derived_ids = list(sf.get("derived_from_ids", []))
                # derived_from is only what the source DECLARES. Being recalled into the same run is
                # exposure (run_reads), not dependence: inferring links from a shared subject would make
                # a fresh API result that CORRECTS a stale memory look like that memory's descendant.

                if is_ext:
                    trust = TrustTier.UNTRUSTED_EXTERNAL
                elif "trust_tier" in sf:
                    try:
                        req_t = max(1, min(int(resolved_trust), int(sf["trust_tier"])))
                    except (TypeError, ValueError):
                        req_t = int(resolved_trust)
                    trust = TrustTier(req_t)
                else:
                    trust = resolved_trust
                req_scope = Scope(sf.get("scope", "org"))
                eff_scope = Scope.ORG if (is_ext and req_scope == Scope.GLOBAL) else req_scope
                req_rc = (
                    RiskClass(sf["risk_class"])
                    if sf.get("risk_class") in {r.value for r in RiskClass}
                    else None
                )
                subj = str(sf["subject"])
                pred = str(sf["predicate"])
                val = str(sf["value"])
                stmt = str(sf["statement"])
                eff_rc = infer_minimum_risk_class(subj, pred, val, stmt, requested=req_rc)
                cand = CandidateClaim(
                    scope=eff_scope,
                    user_id=session.user_id if eff_scope == Scope.USER else sf.get("user_id"),
                    task_id=sf.get("task_id"),
                    claim_type=ClaimType(sf.get("claim_type", "fact")),
                    subject=subj,
                    predicate=pred,
                    value=val,
                    statement=stmt,
                    cardinality=Cardinality(sf.get("cardinality", "single")),
                    volatility=Volatility(sf.get("volatility", "medium")),
                    risk_class=eff_rc,
                    trust_tier=trust,
                    base_confidence=float(sf.get("confidence", 0.85)),
                    derived_from_ids=derived_ids,
                    provenance=self.trust_registry.provenance_from_observation(
                        obs, run_id=session.run_id
                    ).model_copy(update={"is_external_untrusted": is_ext}),
                    is_public_technical_artifact=(
                        False if is_ext else bool(sf.get("is_public_technical_artifact", False))
                    ),
                    artifact_name=None if is_ext else sf.get("artifact_name"),
                    error_signature=sf.get("error_signature"),
                    metadata=dict(sf.get("metadata", {})),
                )
                out.append(
                    normalize_candidate_authority(
                        cand, registry=self.trust_registry, expected_tenant_id=session.org_id
                    )
                )
        return out

    @staticmethod
    def _flag_observation_injection(
        cand: CandidateClaim, obs_by_src: dict[str, ToolObservation]
    ) -> CandidateClaim:
        """Carry an observation-level injection verdict onto every claim drawn from it (0 LLM calls)."""
        obs = obs_by_src.get(cand.provenance.source_id)
        if obs is None or not (obs.is_external_untrusted or cand.provenance.is_external_untrusted):
            return cand
        injected, reason = detect_prompt_injection(obs.content)
        if not injected:
            return cand
        return cand.model_copy(
            update={"metadata": {**cand.metadata, "source_observation_injection": reason or "injection"}}
        )

    # Identifier kinds whose omission from a partially extracted observation signals a lost
    # compound fact (e.g. a retirement date extracted but the replacement route token dropped).
    _GAP_REEXTRACT_KINDS = frozenset({"route_token", "email", "url", "iso_date", "version"})

    def _focused_gap_reextraction(
        self,
        *,
        session: RunSessionContext,
        unstructured_obs: list[ToolObservation],
        recalled_claims: list[MemoryClaim],
        final_output: str,
        first_pass: list[CandidateClaim],
    ) -> list[CandidateClaim]:
        """
        At most ONE extra extraction call per run, only when an observation already yielded
        >= 1 accepted claim but left salient identifiers uncovered (compound fact partially
        extracted). Observations the model abstained on entirely (e.g. injections) never
        trigger it. Skipped when the fleet budget is within 0.5 calls of the 2.0 ceiling.
        """
        stats = getattr(self.llm, "last_extraction_stats", None) or {}
        gaps = stats.get("coverage_gaps") or []
        if not gaps or not first_pass:
            return []
        extracted_src = {c.provenance.source_id for c in first_pass}
        focus: dict[str, list[str]] = {}
        for g in gaps:
            if g.get("source_id") in extracted_src and g.get("kind") in self._GAP_REEXTRACT_KINDS:
                focus.setdefault(str(g["source_id"]), []).append(str(g["span"]))
        if not focus:
            return []
        if self.ledger.avg_memory_calls_per_run >= (MAX_AVG_LLM_CALLS_PER_RUN - 0.5):
            return []
        target_obs = [o for o in unstructured_obs if o.source_id in focus]
        spans = [sp for o in target_obs for sp in focus[o.source_id]]
        first_stats = dict(stats)
        before = self.ledger.total_memory_calls
        try:
            second = self.llm.extract_claims(
                run_id=session.run_id,
                org_id=session.org_id,
                user_id=session.user_id,
                task_text=session.task_text,
                observations=target_obs,
                recalled_claims=recalled_claims,
                final_output=final_output,
                focus_spans=spans,
            )
        finally:
            session.memory_llm_calls += self.ledger.total_memory_calls - before
            second_stats = getattr(self.llm, "last_extraction_stats", None) or {}
            merged: dict[str, Any] = {}
            for k in set(first_stats) | set(second_stats):
                a, b = first_stats.get(k), second_stats.get(k)
                if isinstance(a, int) and isinstance(b, int):
                    merged[k] = a + b
                elif k == "coverage_gaps":
                    merged[k] = list(b or [])  # gaps remaining after the focused pass
                elif isinstance(a, list) or isinstance(b, list):
                    merged[k] = [*(a or []), *(b or [])]
                else:
                    merged[k] = a if a is not None else b
            merged["gap_reextraction_calls"] = 1
            if hasattr(self.llm, "last_extraction_stats"):
                self.llm.last_extraction_stats = merged
        seen = {
            (canonicalize_subject(c.subject), canonicalize_predicate(c.predicate), c.value.strip().lower())
            for c in first_pass
        }
        out: list[CandidateClaim] = []
        for c in second:
            key = (canonicalize_subject(c.subject), canonicalize_predicate(c.predicate), c.value.strip().lower())
            if key in seen:
                continue
            seen.add(key)
            out.append(normalize_candidate_authority(c, registry=self.trust_registry))
        self.last_gap_reextraction = {"focus_spans": spans, "new_claims": len(out)}
        return out

    def end_run(
        self,
        session: RunSessionContext,
        *,
        final_output: str,
        reopened: bool = False,
        explicit_candidates: list[CandidateClaim] | None = None,
        now: datetime | None = None,
    ) -> list[tuple[MemoryClaim | None, WriteOutcome]]:
        """
        Off-critical-path post-run write, revision, derivation linking, and global staging.
        """
        ts = now or session.current_now or utcnow()
        tenant_policy = self.store.get_tenant(session.org_id)

        # 1) Flush run_reads off critical path
        if session.read_memory_ids:
            cited_only = {
                mid for mid in session.cited_memory_ids if not mid.startswith("contradict:")
            }
            self.store.record_run_reads(
                run_id=session.run_id,
                tenant_id=session.org_id,
                memory_ids=sorted(session.read_memory_ids),
                step_idx=max(0, session.tool_calls_count),
                read_at=ts,
                cited_memory_ids=cited_only,
            )

        recalled_claims = [it.claim for it in session.active_bundle.items] if session.active_bundle else []
        candidates: list[CandidateClaim] = list(explicit_candidates or [])

        # 2) Post-run batched extraction: deterministic structured facts cost 0 LLM calls;
        # unstructured salient text invokes 1 consolidated extraction LLM call.
        if not candidates and self._should_extract(session):
            structured_cands = self._extract_structured_observations_zero_llm(
                session=session, recalled_claims=recalled_claims
            )
            unstructured_obs = [o for o in session.observations if not o.structured_facts]
            has_unstructured_salient = any(
                self._is_salient_text(o.content, session.seen_query_tokens)
                for o in unstructured_obs
            )
            unstructured_cands: list[CandidateClaim] = []
            if has_unstructured_salient:
                within_budget = (
                    self.ledger.avg_memory_calls_per_run < (MAX_AVG_LLM_CALLS_PER_RUN - 0.25)
                    or any(o.is_external_untrusted for o in unstructured_obs)
                )
                if within_budget:
                    before_calls = self.ledger.total_memory_calls
                    unstructured_cands = [
                        normalize_candidate_authority(c, registry=self.trust_registry)
                        for c in self.llm.extract_claims(
                            run_id=session.run_id,
                            org_id=session.org_id,
                            user_id=session.user_id,
                            task_text=session.task_text,
                            observations=unstructured_obs,
                            recalled_claims=recalled_claims,
                            final_output=final_output,
                        )
                    ]
                    session.memory_llm_calls += self.ledger.total_memory_calls - before_calls
                    unstructured_cands.extend(
                        self._focused_gap_reextraction(
                            session=session,
                            unstructured_obs=unstructured_obs,
                            recalled_claims=recalled_claims,
                            final_output=final_output,
                            first_pass=unstructured_cands,
                        )
                    )
            candidates = [*structured_cands, *unstructured_cands]

        def _adjudicate(existing: MemoryClaim, incoming: CandidateClaim) -> str:
            # Hard cost circuit-breaker: fall back to deterministic CONTESTED if approaching 2.0 avg LLM budget
            if self.ledger.avg_memory_calls_per_run >= (MAX_AVG_LLM_CALLS_PER_RUN - 0.15):
                return "contest_both"
            before = self.ledger.total_memory_calls
            res = self.llm.adjudicate_conflict(
                run_id=session.run_id, existing=existing, incoming=incoming
            )
            session.memory_llm_calls += self.ledger.total_memory_calls - before
            return res.get("resolution", "contest_both")

        results: list[tuple[MemoryClaim | None, WriteOutcome]] = []
        allowed_parents = set(session.read_memory_ids)
        obs_by_src = {o.source_id: o for o in session.observations}
        candidates = [self._flag_observation_injection(c, obs_by_src) for c in candidates]
        for cand in candidates:
            claim, outcome = self.store.upsert_claim(
                tenant_id=session.org_id,
                candidate=cand,
                tenant_policy=tenant_policy,
                now=ts,
                conflict_adjudicator=_adjudicate,
                allowed_parent_ids=allowed_parents,
                authority_registry=self.trust_registry,
            )
            results.append((claim, outcome))

            # 3) Cross-tenant sanitization staging (zero PII/org leakage)
            if claim is not None and outcome in (
                WriteOutcome.INSERTED_ACTIVE,
                WriteOutcome.CORROBORATED,
                WriteOutcome.SUPERSEDED,
            ):
                ok, payload, _ = sanitize_for_global(claim, tenant_policy)
                if ok and payload is not None:
                    origin_fp = (
                        hash_source_id(f"dom:{claim.source_domain.lower()}")
                        if claim.source_domain
                        else (
                            hash_source_id(f"tdom:{tenant_policy.authorized_domains[0].lower()}")
                            if tenant_policy.authorized_domains
                            else None
                        )
                    )
                    src_obs = obs_by_src.get(claim.source_id)
                    ev_text = " ".join(
                        [
                            src_obs.content if src_obs is not None else "",
                            *(
                                json.dumps(sf, sort_keys=True)
                                for sf in (src_obs.structured_facts if src_obs is not None else [])
                            ),
                        ]
                    )
                    self.store.stage_global_candidate(
                        org_id=session.org_id,
                        cluster_key=payload["cluster_key"],
                        artifact=payload["artifact"],
                        error_sig=payload["error_sig"],
                        sanitized_summary=payload["sanitized_summary"],
                        now=ts,
                        origin_fingerprint=origin_fp,
                        source_hash=hash_source_id(claim.source_id),
                        trust_tier=claim.trust_tier,
                        sanitized_value=payload.get("sanitized_value"),
                        evidence_fp=evidence_fingerprint(
                            observation_text=ev_text,
                            tenant_salted_id=hash_source_id(f"tenant:{session.org_id}"),
                            source_id=claim.source_id,
                            upstream_ref=(claim.metadata or {}).get("upstream_ref"),
                        ),
                    )

        contradicted_ids = [
            tag.split(":", 1)[1]
            for tag in session.cited_memory_ids
            if tag.startswith("contradict:")
        ]
        helped_ids = (
            [
                mid
                for mid in session.read_memory_ids
                if mid not in contradicted_ids
            ]
            if (session.tool_error_count == 0 and not reopened and session.in_run_contradictions == 0)
            else []
        )

        cumulative_cp_ms = round(sum(session.step_critical_latencies_ms), 3)
        self.store.save_run_record(
            run_id=session.run_id,
            tenant_id=session.org_id,
            user_id=session.user_id,
            task_text=session.task_text,
            cohort=session.cohort.value,
            final_output=final_output,
            tool_calls_count=session.tool_calls_count,
            tool_error_count=session.tool_error_count,
            in_run_contradictions=session.in_run_contradictions,
            blocked_unsafe_actions=session.blocked_unsafe_actions,
            reopened=reopened,
            critical_path_ms=cumulative_cp_ms,
            agent_llm_calls=session.agent_llm_calls,
            memory_llm_calls=session.memory_llm_calls,
            started_at=session.started_at,
            finished_at=ts,
            recalled_ids_helped=helped_ids,
            contradicted_ids=contradicted_ids,
        )
        return results

    @staticmethod
    def _slot_belongs_to_tenant(slot_key: str, org_id: str) -> bool:
        from agentic_memory.policy import normalize_token

        return slot_key.startswith(f"org:{normalize_token(org_id)}|")

    @staticmethod
    def _feedback_supports_root(feedback: HumanFeedback, claim: MemoryClaim) -> bool:
        """Deterministic check that the human's own feedback points at this claim."""
        if feedback.corrected_slot_key and claim.slot_key == feedback.corrected_slot_key:
            return True
        text = f"{feedback.faulty_claim_hint or ''} {feedback.correction_text or ''}".lower()
        if not text.strip():
            return False
        value = claim.value.strip().lower()
        if len(value) >= 3 and value in text:
            return True
        subject = claim.subject.strip().lower()
        variants = {subject, subject.replace("_", " "), subject.replace("_", "")}
        return any(len(v) >= 3 and v in text for v in variants)

    def ingest_delayed_feedback(
        self,
        feedback: HumanFeedback,
        *,
        now: datetime | None = None,
    ) -> list[ContainmentAuditEvent]:
        """
        Processes sparse (~2%) delayed human feedback days after runs finish.
        APPROVE strengthens trust; REJECT/CORRECTION attributes root memory (1 LLM call)
        and cascades quarantine + anti-memory containment (0 LLM calls).
        """
        ts = now or feedback.submitted_at or utcnow()
        # Tenant isolation: feedback may only act on a run that belongs to the submitting org.
        if self.store.run_exists_for_other_tenant(feedback.run_id, feedback.org_id):
            raise PermissionError(
                f"feedback {feedback.feedback_id}: run {feedback.run_id} does not belong to {feedback.org_id}"
            )
        if feedback.corrected_slot_key and not self._slot_belongs_to_tenant(
            feedback.corrected_slot_key, feedback.org_id
        ):
            raise PermissionError(
                f"feedback {feedback.feedback_id}: corrected_slot_key is outside tenant {feedback.org_id}"
            )
        if feedback.verdict == FeedbackVerdict.APPROVE:
            self.store.approve_run_memories(feedback.run_id, feedback.org_id, ts)
            return []

        recalled_claims = self.store.get_run_recalled_claims(feedback.run_id, feedback.org_id)
        created_claims = self.store.get_run_created_claims(feedback.run_id, feedback.org_id)
        seen_fb_ids: set[str] = set()
        candidate_claims: list[MemoryClaim] = []
        for rc in [*recalled_claims, *created_claims]:
            if rc.claim_type == ClaimType.NEGATIVE_CONSTRAINT:
                continue
            if rc.memory_id not in seen_fb_ids:
                candidate_claims.append(rc)
                seen_fb_ids.add(rc.memory_id)

        run_rec = self.store.get_run_record(feedback.run_id, feedback.org_id)
        run_output = str(run_rec["final_output"]) if run_rec else ""

        allowed_root_ids = {c.memory_id for c in candidate_claims}
        if not allowed_root_ids:
            return []

        before = self.ledger.total_memory_calls
        raw_root_ids = self.llm.attribute_feedback_root(
            feedback=feedback, recalled_claims=candidate_claims, run_output=run_output
        )
        proposed = [rid for rid in dict.fromkeys(raw_root_ids) if rid in allowed_root_ids]
        llm_used = self.ledger.total_memory_calls - before

        audits: list[ContainmentAuditEvent] = []
        reason = feedback.correction_text or feedback.faulty_claim_hint or "human_rejected_report"
        tenant_policy = self.store.get_tenant(feedback.org_id)

        # The attribution call only PROPOSES roots. A proposed root is refuted only when the human
        # feedback itself supports it (names its slot, subject or value). Proposed but unsupported
        # roots are contested and made non-actionable; nothing is refuted on a guess.
        by_id = {c.memory_id: c for c in candidate_claims}
        root_ids = [rid for rid in proposed if self._feedback_supports_root(feedback, by_id[rid])]
        uncertain = [rid for rid in proposed if rid not in root_ids and by_id[rid].tenant_id == feedback.org_id]
        if uncertain or not proposed:
            changed = self.store.contest_uncertain_roots(
                tenant_id=feedback.org_id,
                memory_ids=uncertain,
                run_id=feedback.run_id,
                reason=("ambiguous_attribution:" if uncertain else "unattributed_rejection:") + reason,
                now=ts,
            )
            audits.append(
                ContainmentAuditEvent(
                    tenant_id=feedback.org_id,
                    root_memory_id="",
                    anti_memory_id="",
                    contested_memory_ids=changed,
                    affected_run_ids=[feedback.run_id],
                    llm_calls_used=llm_used,
                    created_at=ts,
                    notes="ambiguous_attribution_contested_not_refuted" if uncertain else "no_root_attributed_run_flagged",
                )
            )
        for root_id in root_ids:
            claim = self.store.get_claim(root_id, feedback.org_id)
            if claim is None:
                continue

            # If a tenant rejects a GLOBAL advisory, create a tenant-scoped anti-memory override
            # and increment global contradiction count; only refute globally once >=3 orgs reject.
            if claim.tenant_id == GLOBAL_TENANT_ID:
                updated_g = self.store.increment_contradiction_hit(
                    claim.memory_id, ts, tenant_id=feedback.org_id
                )
                override_cand = CandidateClaim(
                    scope=Scope.ORG,
                    claim_type=ClaimType.NEGATIVE_CONSTRAINT,
                    slot_key=f"{canonicalize_slot(Scope.ORG, feedback.org_id, claim.subject, claim.predicate)}|anti",
                    subject=claim.subject,
                    predicate=f"not_{claim.predicate}",
                    value=f"NOT:{claim.value}",
                    statement=f"TENANT OVERRIDE: Do NOT apply global advisory '{claim.statement}' ({reason}).",
                    cardinality=Cardinality.MULTI,
                    volatility=Volatility.PERMANENT,
                    risk_class=claim.risk_class,
                    trust_tier=TrustTier.HUMAN_VERIFIED,
                    base_confidence=0.96,
                    provenance=self.trust_registry.build_trusted_provenance(
                        connector_id="human_review",
                        source_id=f"global_override:{feedback.org_id}:{claim.memory_id}",
                        source_type=SourceType.HUMAN_FEEDBACK,
                        run_id=feedback.run_id,
                        tenant_id=feedback.org_id,
                        content=f"NOT:{claim.value}",
                        observed_at=ts,
                    ),
                )
                anti_c, _ = self.store.upsert_claim(
                    tenant_id=feedback.org_id,
                    candidate=override_cand,
                    tenant_policy=tenant_policy,
                    now=ts,
                    authority_registry=self.trust_registry,
                )
                if updated_g is not None and updated_g.contradiction_hits >= 3:
                    audits.append(
                        self.store.refute_and_contain(
                            tenant_id=GLOBAL_TENANT_ID,
                            root_memory_id=claim.memory_id,
                            reason=f"multi_tenant_rejection:{reason}",
                            now=ts,
                            llm_calls_used=llm_used,
                        )
                    )
                elif anti_c is not None:
                    audits.append(
                        ContainmentAuditEvent(
                            tenant_id=feedback.org_id,
                            root_memory_id=claim.memory_id,
                            anti_memory_id=anti_c.memory_id,
                            affected_run_ids=[feedback.run_id],
                            llm_calls_used=llm_used,
                            created_at=ts,
                            notes=f"tenant_override_for_global_advisory:{reason}",
                        )
                    )
                continue

            if claim.tenant_id != feedback.org_id:
                continue

            # Walk upstream through unverified derivation ancestors to catch compounding root cause
            target_roots: list[str] = []
            visited_up: set[str] = {root_id}
            up_queue: list[str] = list(claim.derived_from_ids)
            earliest_unverified: list[str] = []
            while up_queue:
                pid = up_queue.pop(0)
                if pid in visited_up:
                    continue
                visited_up.add(pid)
                p = self.store.get_claim(pid, feedback.org_id, include_global=False)
                if (
                    p is not None
                    and p.tenant_id == feedback.org_id
                    and p.status not in (MemoryStatus.REFUTED, MemoryStatus.ARCHIVED)
                    and p.trust_tier <= TrustTier.AGENT_INFERRED
                ):
                    unverified_parents = [
                        gp
                        for gp in p.derived_from_ids
                        if (gp_c := self.store.get_claim(gp, feedback.org_id, include_global=False)) is not None
                        and gp_c.tenant_id == feedback.org_id
                        and gp_c.status not in (MemoryStatus.REFUTED, MemoryStatus.ARCHIVED)
                        and gp_c.trust_tier <= TrustTier.AGENT_INFERRED
                    ]
                    if not unverified_parents:
                        earliest_unverified.append(p.memory_id)
                    else:
                        up_queue.extend(unverified_parents)

            target_roots = earliest_unverified or [root_id]
            for tr in dict.fromkeys(target_roots):
                audit = self.store.refute_and_contain(
                    tenant_id=feedback.org_id,
                    root_memory_id=tr,
                    reason=reason,
                    now=ts,
                    llm_calls_used=llm_used,
                )
                audits.append(audit)

        # If human supplied an explicit correction, store it with HUMAN_VERIFIED trust
        if feedback.corrected_slot_key and feedback.corrected_value:
            parts = feedback.corrected_slot_key.split("|")
            subj = parts[-2] if len(parts) >= 3 else "corrected_entity"
            pred = parts[-1] if len(parts) >= 2 else "corrected_attribute"
            tenant_policy = self.store.get_tenant(feedback.org_id)
            corr = CandidateClaim(
                scope=Scope.ORG,
                claim_type=ClaimType.FACT,
                slot_key=feedback.corrected_slot_key,
                subject=subj,
                predicate=pred,
                value=feedback.corrected_value,
                statement=feedback.correction_text or f"{subj} {pred}: {feedback.corrected_value}",
                cardinality=Cardinality.SINGLE,
                volatility=Volatility.LOW,
                risk_class=RiskClass.MEDIUM_OPERATIONAL,
                trust_tier=TrustTier.HUMAN_VERIFIED,
                base_confidence=0.98,
                provenance=self.trust_registry.build_trusted_provenance(
                    connector_id="human_review",
                    source_id=feedback.feedback_id,
                    source_type=SourceType.HUMAN_FEEDBACK,
                    run_id=feedback.run_id,
                    tenant_id=feedback.org_id,
                    content=feedback.corrected_value,
                    observed_at=ts,
                ),
            )
            self.store.upsert_claim(
                tenant_id=feedback.org_id,
                candidate=corr,
                tenant_policy=tenant_policy,
                now=ts,
                authority_registry=self.trust_registry,
            )

        return audits

    def run_background_maintenance(
        self,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """
        Scheduled off-critical-path maintenance:
        1. Promote cross-org clusters reaching >= k_anonymity distinct orgs (1 LLM synth/cluster)
        2. Enforce 2+ year bounded retention and expiry (0 LLM calls)
        """
        ts = now or utcnow()
        promoted_ids: list[str] = []
        clusters = self.store.get_promotable_global_clusters(k_anonymity=self.k_anonymity)

        for cl in clusters:
            if self.ledger.avg_memory_calls_per_run >= (MAX_AVG_LLM_CALLS_PER_RUN - 0.05):
                break
            synth = self.llm.synthesize_global_advisory(
                cluster_key=cl["cluster_key"],
                artifact=cl["artifact"],
                error_sig=cl["error_sig"],
                org_summaries=cl["summaries"],
            )
            clean_stmt = str(synth.get("statement", "")).strip()
            ok_synth, _ = validate_global_safety(clean_stmt)
            if not ok_synth:
                continue
            # The published value comes from staged tenant evidence, never from the synthesis
            # call; the store re-verifies this inside the promotion transaction.
            raw_val = str(cl.get("consensus_value") or "").strip()
            if not raw_val:
                synth_val = str(synth.get("value", "")).strip()
                raw_val = (
                    synth_val
                    if synth_val
                    and global_value_supported(synth_val, [], cl["summaries"], self.k_anonymity)
                    else f"{cl['artifact']}:{cl['error_sig']}"
                )
            ok_val, _ = validate_global_metadata_field(raw_val, max_len=200)
            if not ok_val:
                continue
            claim, outcome = self.store.promote_global_cluster(
                cluster_key=cl["cluster_key"],
                artifact=cl["artifact"],
                error_sig=cl["error_sig"],
                statement=clean_stmt,
                value=raw_val,
                k_anonymity=self.k_anonymity,
                now=ts,
            )
            if claim is not None and outcome == WriteOutcome.INSERTED_ACTIVE:
                promoted_ids.append(claim.memory_id)

        self.store.sync_promoted_global_counts(ts)
        compaction = self.store.compact_and_forget(
            now=ts,
            max_active_per_org=self.max_active_per_org,
            max_negative_per_org=self.max_negative_per_org,
            max_global_advisories=self.max_global_advisories,
        )
        return {
            "promoted_global_ids": promoted_ids,
            "compaction": compaction,
        }
