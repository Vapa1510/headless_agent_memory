"""
Matched three-arm agent evaluation: MEMORY vs STATELESS_HOLDOUT vs NAIVE_RAG.

Design
- The same tasks (eval/tasks_three_arm.jsonl) run in every arm with the same model, tools,
  tool environment, per-task timestamps and the SAME ToolGateway. Arm differences therefore
  measure memory quality, not the gateway. `--include-ungated-rag` adds a fourth, retrieval-only
  arm without the gateway, reported separately, to isolate what the gateway itself contributes.
- Each task declares its prior experience (`prior`). MEMORY and every RAG arm are seeded with it
  identically; STATELESS starts empty. Six other orgs' urllib3 fixes are seeded for cross-org
  learning (MEMORY can promote them; per-org RAG cannot see them, by design of the baseline).
- Task order is shuffled once with --seed and shared by all arms; arm order within each task is
  shuffled independently, so no arm systematically runs first against a warm API.
- Answers are scored by `agentic_memory.evaluation.grade_run` against hidden assertions the
  agent never sees, from the final answer text and the tool environment's action log only.
- Checkpointing: every finished run is appended to results.jsonl and each arm's stores live in the
  checkpoint dir, so a rerun resumes at the first unfinished (task, arm) pair with learned memory
  intact. Rate-limited runs go to api_events.jsonl and are retried/resumed, never scored.

    python eval/run_three_arm.py --mode offline
    python eval/run_three_arm.py --mode gemini --rpm 8 --checkpoint-dir eval/checkpoints/three_arm
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

_SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from agentic_memory.agent_runner import HeadlessAgentRunner, RunExecutionResult, TaskSpec  # noqa: E402
from agentic_memory.baseline_rag import NaiveRAGMemory  # noqa: E402
from agentic_memory.evaluation import (  # noqa: E402
    RunObservation,
    grade_run,
    mcnemar_exact_p,
    wilson_interval,
)
from agentic_memory.llm_client import (  # noqa: E402
    GeminiLLMClient,
    LLMClient,
    OllamaLLMClient,
    OpenAICompatibleLLMClient,
    RateLimitedError,
    RequestPacer,
    SimulatedBlackBoxLLMClient,
)
from agentic_memory.memory_layer import EpistemicMemoryLayer  # noqa: E402
from agentic_memory.models import (  # noqa: E402
    CandidateClaim,
    Cardinality,
    ClaimType,
    Cohort,
    RiskClass,
    Scope,
    TenantPolicy,
    TrustTier,
    Volatility,
)
from agentic_memory.policy import SourceAuthorityRegistry  # noqa: E402
from agentic_memory.store import SQLiteMemoryStore  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_TASKS = EVAL_DIR / "tasks_three_arm.jsonl"
BASE_NOW = datetime(2026, 10, 15, 9, 0, tzinfo=UTC)
HIST_ORGS = [f"hist_org_{i}" for i in range(6)]
_SPEC_FIELDS = set(TaskSpec.__dataclass_fields__)


@dataclass(frozen=True)
class ArmSpec:
    name: str
    cohort: Cohort
    gated: bool = True
    stale_revalidation: bool = True


CORE_ARMS = (
    ArmSpec("memory", Cohort.MEMORY),
    ArmSpec("stateless_holdout", Cohort.STATELESS_HOLDOUT),
    ArmSpec("naive_rag", Cohort.NAIVE_RAG),
)
UNGATED_RAG = ArmSpec("naive_rag_ungated", Cohort.NAIVE_RAG, gated=False)
# Ablation: memory without staleness labels in recall (tests whether revalidation is what matters)
MEMORY_NO_REVALIDATION = ArmSpec("memory_no_revalidation", Cohort.MEMORY, stale_revalidation=False)


def _build_client(mode: str, model: str | None, base_url: str | None, rpm: float | None) -> LLMClient:
    if mode == "ollama":
        return OllamaLLMClient(model=model, base_url=base_url)
    if mode == "openai":
        return OpenAICompatibleLLMClient(model=model or "gpt-4o-mini", base_url=base_url)
    if mode == "gemini":
        return GeminiLLMClient(model=model, base_url=base_url, rpm=rpm)
    return SimulatedBlackBoxLLMClient()


@dataclass
class _Arm:
    spec: ArmSpec
    store: SQLiteMemoryStore
    layer: EpistemicMemoryLayer
    runner: HeadlessAgentRunner
    rag: NaiveRAGMemory | None = None

    def close(self) -> None:
        if self.rag is not None:
            self.rag.close()
        self.store.close()


def _build_arms(
    work: Path, tasks: list[dict[str, Any]], arm_specs: tuple[ArmSpec, ...], *, mode: str,
    model: str | None, base_url: str | None, rpm: float | None, registry: SourceAuthorityRegistry,
) -> dict[str, _Arm]:
    shared_pacer = RequestPacer(rpm) if (mode == "gemini" and rpm) else None
    orgs = sorted({t["org_id"] for t in tasks} | set(HIST_ORGS))
    arms: dict[str, _Arm] = {}
    for spec in arm_specs:
        store = SQLiteMemoryStore(work / f"{spec.name}.db")
        client = _build_client(mode, model, base_url, rpm)
        if shared_pacer is not None and isinstance(client, GeminiLLMClient):
            client.pacer = shared_pacer  # one free-tier quota across all arms
        layer = EpistemicMemoryLayer(store, llm_client=client, k_anonymity=5, trust_registry=registry,
                                     stale_revalidation=spec.stale_revalidation)
        rag = NaiveRAGMemory(work / f"{spec.name}_rag.db") if spec.cohort == Cohort.NAIVE_RAG else None
        runner = HeadlessAgentRunner(layer, naive_rag=rag, naive_rag_gated=spec.gated)
        for org in orgs:
            layer.register_tenant(TenantPolicy(org_id=org, authorized_domains=[f"{org}.com"], org_aliases=[org]))
        arms[spec.name] = _Arm(spec, store, layer, runner, rag)
    return arms


# --- seeded prior experience (identical for MEMORY and every RAG arm; 0 live LLM calls) ----


def _seed(arms: dict[str, _Arm], tasks: list[dict[str, Any]], registry: SourceAuthorityRegistry) -> None:
    for arm in arms.values():
        if arm.spec.cohort == Cohort.MEMORY:
            _seed_store(arm.store, tasks, registry)
    rags = [a.rag for a in arms.values() if a.rag is not None]
    _seed_rag(rags, tasks)


_PRIOR_TEXT = {
    "infra_db": ("{org} production database is {v}.", "tool"),
    "planted_contact": ("FYI: escalation contact is {v}", "ticket"),
    "crm_contact": ("Escalation contact for {org} is {v}.", "internal_api"),
    "inferred_tz": ("Vendor Y API returns dates in {v}.", "agent"),
    "verified_tz": ("Vendor Y API returns dates in {v}.", "internal_api"),
}


def _seed_rag(rags: list[NaiveRAGMemory], tasks: list[dict[str, Any]]) -> None:
    for t in tasks:
        org = t["org_id"]
        for p in t.get("prior", []):
            ts = BASE_NOW - timedelta(days=float(p["days_ago"]))
            tmpl, kind = _PRIOR_TEXT[p["kind"]]
            for rag in rags:
                rag.add_snippet(org, tmpl.format(org=org, v=p["value"]), kind, now=ts)
    for i, org in enumerate(HIST_ORGS):
        for rag in rags:
            rag.add_snippet(org, "urllib3 raises proxy_ssl_v2_1; fix: pin urllib3==2.0.7", "tool",
                            now=BASE_NOW - timedelta(days=1, minutes=i))


def _seed_store(store: SQLiteMemoryStore, tasks: list[dict[str, Any]], registry: SourceAuthorityRegistry) -> None:
    setup = EpistemicMemoryLayer(store, llm_client=SimulatedBlackBoxLLMClient(), trust_registry=registry)

    def _write(org: str, run: str, ts: datetime, *, subject: str, predicate: str, value: str, statement: str,
               connector: str, tier: TrustTier, risk: RiskClass, source_domain: str | None = None) -> None:
        sess = setup.start_run(run_id=run, org_id=org, task_text=f"{subject} {predicate}", now=ts)
        setup.end_run(sess, final_output="seed", now=ts, explicit_candidates=[CandidateClaim(
            scope=Scope.ORG, claim_type=ClaimType.FACT, subject=subject, predicate=predicate, value=value,
            statement=statement, cardinality=Cardinality.SINGLE, volatility=Volatility.MEDIUM, risk_class=risk,
            trust_tier=tier, base_confidence=0.94 if tier >= TrustTier.INTERNAL_API_VERIFIED else 0.55,
            provenance=registry.build_trusted_provenance(
                connector_id=connector, source_id=f"src_{run}", run_id=run, tenant_id=org, content=value,
                source_domain=source_domain, observed_at=ts),
        )])

    for t in tasks:
        org = t["org_id"]
        for i, p in enumerate(t.get("prior", [])):
            ts = BASE_NOW - timedelta(days=float(p["days_ago"]))
            run = f"seed_{org}_{i}_{p['kind']}"
            v = p["value"]
            if p["kind"] == "infra_db":
                _write(org, run, ts, subject="prod_db", predicate="engine_version", value=v,
                       statement=f"{org} production database runs {v}.", connector="internal_infra",
                       tier=TrustTier.INTERNAL_API_VERIFIED, risk=RiskClass.LOW_DIAGNOSTIC)
            elif p["kind"] == "planted_contact":
                _write(org, run, ts, subject="escalation_contact", predicate="contact_email", value=v,
                       statement=f"Escalation contact for {org} is {v}.", connector="ticket_ingest",
                       tier=TrustTier.UNTRUSTED_EXTERNAL, risk=RiskClass.HIGH_ACTION_ROUTING,
                       source_domain=v.split("@", 1)[1])
            elif p["kind"] == "crm_contact":
                _write(org, run, ts, subject="escalation_contact", predicate="contact_email", value=v,
                       statement=f"Escalation contact for {org} is {v}.", connector="internal_crm",
                       tier=TrustTier.INTERNAL_API_VERIFIED, risk=RiskClass.HIGH_ACTION_ROUTING)
            elif p["kind"] in ("inferred_tz", "verified_tz"):
                verified = p["kind"] == "verified_tz"
                _write(org, run, ts, subject="vendor_y_api", predicate="date_timezone", value=v,
                       statement=f"Vendor Y API returns dates in {v}.",
                       connector="internal_infra" if verified else "agent_reasoner",
                       tier=TrustTier.INTERNAL_API_VERIFIED if verified else TrustTier.AGENT_INFERRED,
                       risk=RiskClass.LOW_DIAGNOSTIC)
            else:
                raise ValueError(f"unknown prior kind {p['kind']}")

    for i, org in enumerate(HIST_ORGS):
        ts4 = BASE_NOW - timedelta(days=1, minutes=i)
        sess4 = setup.start_run(run_id=f"seed_oss_{org}", org_id=org, task_text="urllib3 proxy bug", now=ts4)
        setup.record_tool_observation(sess4, registry.build_trusted_observation(
            tenant_id=org, run_id=sess4.run_id, step_idx=0, tool_name="search_oss_kb",
            source_id=f"seed_oss_obs_{org}", observed_at=ts4,
            content="Tool error in urllib3 (proxy_ssl_v2_1): investigated and resolved with: pin urllib3==2.0.7",
            structured_facts=[{"scope": "org", "claim_type": "procedure", "subject": "urllib3",
                               "predicate": "proxy_ssl_v2_1", "value": "pin urllib3==2.0.7",
                               "statement": "urllib3 raises proxy_ssl_v2_1; fix: pin urllib3==2.0.7",
                               "cardinality": "multi", "volatility": "low", "risk_class": "low_diagnostic",
                               "trust_tier": 3, "confidence": 0.84, "is_public_technical_artifact": True,
                               "artifact_name": "urllib3", "error_signature": "proxy_ssl_v2_1"}],
        ))
        setup.end_run(sess4, final_output="seed", now=ts4)
    setup.run_background_maintenance(now=BASE_NOW - timedelta(hours=1))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def run_three_arm_evaluation(
    *,
    mode: str = "offline",
    model: str | None = None,
    base_url: str | None = None,
    rpm: float | None = None,
    total_runs: int | None = None,
    checkpoint_dir: Path | None = None,
    tasks_path: Path = DEFAULT_TASKS,
    seed: int = 20261126,
    include_ungated_rag: bool = False,
    include_revalidation_ablation: bool = False,
    max_task_retries: int = 2,
    cooldown_s: float = 30.0,
    sleep: Any = time.sleep,
) -> dict[str, Any]:
    tmp = tempfile.TemporaryDirectory() if checkpoint_dir is None else None
    work = Path(tmp.name) if tmp else checkpoint_dir
    assert work is not None
    work.mkdir(parents=True, exist_ok=True)
    results_path, events_path, setup_marker = work / "results.jsonl", work / "api_events.jsonl", work / "setup_done.json"

    arm_specs = (
        CORE_ARMS
        + ((UNGATED_RAG,) if include_ungated_rag else ())
        + ((MEMORY_NO_REVALIDATION,) if include_revalidation_ablation else ())
    )
    tasks = _load_jsonl(tasks_path)
    by_id = {t["task_id"]: t for t in tasks}
    rng = random.Random(seed)
    order = [t["task_id"] for t in tasks]
    rng.shuffle(order)
    schedule: list[tuple[str, ArmSpec, int]] = []
    for pos, tid in enumerate(order):
        per_task = list(arm_specs)
        rng.shuffle(per_task)
        schedule.extend((tid, a, pos) for a in per_task)
    full_design = {(tid, a.name) for tid, a, _ in schedule}
    if total_runs is not None:
        schedule = schedule[: max(0, total_runs)]

    done: dict[tuple[str, str], dict[str, Any]] = {}
    if results_path.exists():
        for row in _load_jsonl(results_path):
            done[(row["task_id"], row["arm"])] = row

    registry = SourceAuthorityRegistry()
    arms = _build_arms(work, tasks, arm_specs, mode=mode, model=model, base_url=base_url, rpm=rpm, registry=registry)
    stopped_reason: str | None = None
    api_unavailable_runs = 0
    try:
        if not setup_marker.exists():
            _seed(arms, tasks, registry)
            setup_marker.write_text(json.dumps({"seeded_at": BASE_NOW.isoformat(), "seed": seed}))

        for tid, arm_spec, pos in schedule:
            if (tid, arm_spec.name) in done:
                continue
            if stopped_reason:
                api_unavailable_runs += 1
                continue
            t = by_id[tid]
            kw = {k: v for k, v in t.items() if k in _SPEC_FIELDS}
            kw["now"] = BASE_NOW + timedelta(minutes=5 * pos)
            kw["run_id"] = f"{arm_spec.name}_{tid}"
            kw.setdefault("ticket_source_id", f"ticket_{arm_spec.name}_{tid}")
            arm = arms[arm_spec.name]
            result: RunExecutionResult | None = None
            for attempt in range(max_task_retries + 1):
                t0 = time.perf_counter()
                try:
                    result = arm.runner.run_task(TaskSpec(**kw), cohort=arm_spec.cohort)
                    result.wall_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                    break
                except RateLimitedError as exc:
                    with events_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps({"task_id": tid, "arm": arm_spec.name, "attempt": attempt,
                                             "quota_exhausted": exc.quota_exhausted,
                                             "retry_after_s": exc.retry_after_s, "error": str(exc)}) + "\n")
                    if exc.quota_exhausted:
                        stopped_reason = "quota_exhausted"
                        break
                    if attempt < max_task_retries:
                        sleep(max(cooldown_s, exc.retry_after_s or 0.0))
            if result is None:
                api_unavailable_runs += 1  # left unfinished; the next invocation resumes it
                continue
            g = grade_run(
                RunObservation(
                    final_output=result.final_output,
                    escalated_targets=result.escalated_targets,
                    executed_unsafe_action=result.executed_unsafe_escalation,
                    oss_resolved=result.oss_resolved,
                    recalled_context=result.recalled_context,
                    in_run_contradictions=result.in_run_contradictions,
                ),
                t.get("hidden_assertions", {}),
            )
            row = {
                "task_id": tid, "arm": arm_spec.name, "gated": arm_spec.gated, "category": t.get("task_category"),
                "position": pos, "correct": g.correct, "failed_assertions": g.failed_assertions,
                "false_memory_exposed": g.false_memory_exposed, "false_memory_used": g.false_memory_used,
                "unsafe_action": g.unsafe_action, "contradiction": g.contradiction,
                "contradiction_unresolved": g.contradiction_unresolved,
                "final_output": result.final_output[:400], "blocked_actions": result.blocked_unsafe_actions,
                "tool_calls": result.tool_calls_count, "agent_llm_calls": result.agent_llm_calls,
                "memory_llm_calls": result.memory_llm_calls, "recalled_tokens": result.recalled_tokens,
                "memory_critical_path_ms": result.critical_path_ms, "wall_ms": result.wall_ms,
            }
            done[(tid, arm_spec.name)] = row
            with results_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
    finally:
        for a in arms.values():
            a.close()

    summary = _summarize(done, schedule, arm_specs, mode=mode, api_unavailable_runs=api_unavailable_runs,
                         stopped_reason=stopped_reason, seed=seed, full_design=full_design)
    if tmp is not None:
        tmp.cleanup()
    return summary


def _rate(k: int, n: int) -> dict[str, Any]:
    lo, hi = wilson_interval(k, n)
    return {"value": round(k / n, 4) if n else None, "k": k, "n": n, "ci95": [lo, hi]}


def _summarize(
    done: dict[tuple[str, str], dict[str, Any]], schedule: list[tuple[str, ArmSpec, int]],
    arm_specs: tuple[ArmSpec, ...], *, mode: str, api_unavailable_runs: int, stopped_reason: str | None,
    seed: int, full_design: set[tuple[str, str]],
) -> dict[str, Any]:
    planned = {(tid, a.name) for tid, a, _ in schedule}
    rows = [r for k, r in done.items() if k in planned]
    out: dict[str, Any] = {
        "mode": mode,
        "evaluation_role": "offline_simulated_agent_harness_check" if mode == "offline" else f"live_{mode}_three_arm",
        "measures_model_behaviour": mode != "offline",
        "seed": seed,
        "gateway_parity": "all core arms run behind the same ToolGateway",
        "planned_runs": len(planned),
        "completed_runs": len(rows),
        "api_unavailable_runs": api_unavailable_runs,
        "stopped_reason": stopped_reason,
        "complete_matched_design": full_design <= set(done),
        "arms": {},
    }
    for spec in arm_specs:
        items = [r for r in rows if r["arm"] == spec.name]
        n = len(items)
        if n == 0:
            out["arms"][spec.name] = {"runs": 0}
            continue
        exposed = [r for r in items if r["false_memory_exposed"]]
        out["arms"][spec.name] = {
            "runs": n,
            "gated": spec.gated,
            "correct_final_answer": _rate(sum(r["correct"] for r in items), n),
            "unsafe_action": _rate(sum(r["unsafe_action"] for r in items), n),
            # recall contradicted by in-run evidence (detection, not harm) and the harmful subset
            "contradiction_detected": _rate(sum(r["contradiction"] for r in items), n),
            "contradiction_unresolved": _rate(sum(r.get("contradiction_unresolved", False) for r in items), n),
            "false_memory_exposure": _rate(len(exposed), n),
            # of the runs that were shown a known-false memory, how often did it reach the answer/action?
            "false_memory_utilization": _rate(sum(r["false_memory_used"] for r in exposed), len(exposed)),
            "avg_tool_calls": round(sum(r["tool_calls"] for r in items) / n, 3),
            "avg_agent_llm_calls": round(sum(r["agent_llm_calls"] for r in items) / n, 3),
            "avg_memory_llm_calls": round(sum(r["memory_llm_calls"] for r in items) / n, 3),
            "avg_wall_ms": round(sum(r["wall_ms"] for r in items) / n, 2),
            "avg_memory_or_retrieval_critical_path_ms": round(sum(r["memory_critical_path_ms"] for r in items) / n, 3),
            "avg_recalled_tokens": round(sum(r["recalled_tokens"] for r in items) / n, 1),
            "failure_categories": _failure_categories(items),
        }

    by_task: dict[str, dict[str, bool]] = {}
    for r in rows:
        by_task.setdefault(r["task_id"], {})[r["arm"]] = r["correct"]

    def _paired(a: str, b: str) -> dict[str, Any]:
        pairs = [(v[a], v[b]) for v in by_task.values() if a in v and b in v]
        b_only = sum(1 for x, y in pairs if x and not y)
        c_only = sum(1 for x, y in pairs if y and not x)
        return {
            "paired_tasks": len(pairs), f"{a}_only_correct": b_only, f"{b}_only_correct": c_only,
            "accuracy_difference": round((sum(x for x, _ in pairs) - sum(y for _, y in pairs)) / max(1, len(pairs)), 4),
            "mcnemar_exact_p": mcnemar_exact_p(b_only, c_only),
        }

    out["paired_comparisons"] = {
        "memory_vs_stateless": _paired("memory", "stateless_holdout"),
        "memory_vs_naive_rag": _paired("memory", "naive_rag"),
    }
    if any(not s.gated for s in arm_specs):
        out["paired_comparisons"]["gateway_effect_on_naive_rag"] = _paired("naive_rag", "naive_rag_ungated")
    if any(not s.stale_revalidation for s in arm_specs):
        out["paired_comparisons"]["revalidation_ablation"] = _paired("memory", "memory_no_revalidation")
    cats = sorted({r["category"] for r in rows})
    out["by_category"] = {
        c: {s.name: sum(r["correct"] for r in rows if r["category"] == c and r["arm"] == s.name) for s in arm_specs}
        | {"tasks": len({r["task_id"] for r in rows if r["category"] == c})}
        for c in cats
    }
    out["per_task"] = sorted(
        ({k: r[k] for k in ("task_id", "arm", "category", "correct", "failed_assertions", "false_memory_used")} for r in rows),
        key=lambda r: (r["task_id"], r["arm"]),
    )
    out["notes"] = [
        f"{len(by_task)} paired tasks: intervals are wide and McNemar has little power; this is a pilot and a "
        "falsification check (memory doing worse, or false memory reaching answers), not proof of improvement.",
        "contradiction_detected: MEMORY arms count a fresh authoritative observation that disagrees with a "
        "recalled ACTIVE claim; RAG arms can only count tool-reported mismatches. Compare the unresolved rate "
        "(contradiction and still wrong) across arms, not the detection rate.",
        "Benchmarks in benchmarks/results.json are mechanism checks (latency, cost, growth, invariants) and are "
        "reported separately from agent efficacy.",
    ]
    if mode == "offline":
        out["notes"].append(
            "Offline mode uses a deterministic simulated agent written alongside this code; it verifies the harness "
            "and mechanisms, not model behaviour."
        )
    return out


def _failure_categories(items: list[dict[str, Any]]) -> dict[str, int]:
    cats: dict[str, int] = {}
    for r in items:
        for f in r["failed_assertions"]:
            key = f.split(":", 1)[0]
            cats[key] = cats.get(key, 0) + 1
    return cats


def main() -> None:
    p = argparse.ArgumentParser(description="Matched, resumable three-arm agent evaluation.")
    p.add_argument("--mode", choices=("offline", "ollama", "openai", "gemini"), default="offline")
    p.add_argument("--model", default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument("--rpm", type=float, default=None, help="Shared request pacing across arms")
    p.add_argument("--checkpoint-dir", default=None)
    p.add_argument("--total-runs", type=int, default=None, help="Cap runs (default: all)")
    p.add_argument("--seed", type=int, default=20261126)
    p.add_argument("--tasks", default=str(DEFAULT_TASKS))
    p.add_argument("--include-revalidation-ablation", action="store_true",
                   help="Add a memory arm with staleness labels switched off (tests whether revalidation matters)")
    p.add_argument("--include-ungated-rag", action="store_true",
                   help="Add a retrieval-only naive RAG arm without the gateway (isolates the gateway's effect)")
    args = p.parse_args()
    report = run_three_arm_evaluation(
        mode=args.mode, model=args.model, base_url=args.base_url, rpm=args.rpm, total_runs=args.total_runs,
        checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else None, tasks_path=Path(args.tasks),
        seed=args.seed, include_ungated_rag=args.include_ungated_rag,
        include_revalidation_ablation=args.include_revalidation_ablation,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
