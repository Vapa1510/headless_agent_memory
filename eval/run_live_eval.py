"""
Extraction / admission / safety evaluation over hand-labelled tickets.

Reports an explicit outcome taxonomy (correct admission, correct rejection, safe abstention,
false rejection, under-admission, unsafe admission, over-retention) instead of one accuracy
number, separately for the development set (tickets_30.jsonl) and the held-out set
(tickets_heldout.jsonl). Live runs checkpoint after every ticket and resume from the first
unfinished one; rate-limited tickets are reported as API-unavailable, never as failures.

    python eval/run_live_eval.py --mode offline
    python eval/run_live_eval.py --mode offline --fixture-variant mislabelled_scope
    python eval/run_live_eval.py --mode gemini --rpm 10 --checkpoint-dir eval/checkpoints/g1
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch

_SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from agentic_memory.agent_runner import ToolGateway  # noqa: E402
from agentic_memory.evaluation import (  # noqa: E402
    ClaimExpectation,
    EvaluationOutcome,
    classify_ticket_outcome,
    score_claim_expectations,
    summarize_extraction_outcomes,
)
from agentic_memory.llm_client import (  # noqa: E402
    GeminiLLMClient,
    LLMClient,
    OllamaLLMClient,
    OpenAICompatibleLLMClient,
    RateLimitedError,
    SimulatedBlackBoxLLMClient,
)
from agentic_memory.memory_layer import EpistemicMemoryLayer  # noqa: E402
from agentic_memory.models import (  # noqa: E402
    RiskClass,
    SourceType,
    TenantPolicy,
    TrustTier,
)
from agentic_memory.policy import sanitize_for_global  # noqa: E402
from agentic_memory.store import SQLiteMemoryStore  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_DATASETS = {
    "development": EVAL_DIR / "tickets_30.jsonl",
    "heldout": EVAL_DIR / "tickets_heldout.jsonl",
}
FIXTURE_VARIANTS = ("as_labelled", "mislabelled_scope")


def load_tickets(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


class _FixtureHttpResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._raw = json.dumps({"choices": [{"message": {"content": json.dumps(payload)}}]}).encode(
            "utf-8"
        )

    def __enter__(self) -> _FixtureHttpResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._raw


def apply_fixture_variant(payload: dict[str, Any], variant: str) -> dict[str, Any]:
    """
    `mislabelled_scope` reproduces the failure mode seen in the live Gemini run: the model tags
    every claim as task-scoped. The deterministic scope classifier must recover durable facts
    while still keeping genuinely ephemeral ones out. (Reconstructed from the reported behaviour;
    the raw Gemini responses were not retained.)
    """
    if variant == "as_labelled":
        return payload
    if variant == "mislabelled_scope":
        claims = []
        for c in payload.get("claims", []):
            c2 = dict(c)
            c2["scope"] = "task"
            c2["suggested_durability"] = "ephemeral"
            claims.append(c2)
        return {**payload, "claims": claims}
    raise ValueError(f"Unknown fixture variant: {variant}")


def build_llm_client(
    *,
    mode: str,
    model: str | None = None,
    base_url: str | None = None,
    rpm: float | None = None,
    for_extraction_offline: bool = False,
) -> LLMClient:
    if mode == "ollama":
        return OllamaLLMClient(model=model, base_url=base_url)
    if mode == "openai":
        return OpenAICompatibleLLMClient(model=model or "gpt-4o-mini", base_url=base_url)
    if mode == "gemini":
        return GeminiLLMClient(model=model, base_url=base_url, rpm=rpm)
    if for_extraction_offline:
        return OpenAICompatibleLLMClient(api_key="offline-eval-key", model="gpt-4o-mini")
    return SimulatedBlackBoxLLMClient()


def _gate_args(tool: str, target: str) -> dict[str, Any]:
    if tool == "update_webhook":
        return {"url": target}
    if tool == "change_billing_route":
        return {"route_id": target}
    return {"to": target}


def _read_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    done: dict[str, dict[str, Any]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[str(row["ticket_id"])] = row
    return done


def evaluate_dataset(
    tickets: list[dict[str, Any]],
    *,
    mode: str,
    dataset: str,
    variant: str = "as_labelled",
    model: str | None = None,
    base_url: str | None = None,
    rpm: float | None = None,
    checkpoint_dir: Path | None = None,
    max_tickets: int | None = None,
) -> dict[str, Any]:
    work_dir_ctx = tempfile.TemporaryDirectory() if checkpoint_dir is None else None
    work_dir = Path(work_dir_ctx.name) if work_dir_ctx else checkpoint_dir
    assert work_dir is not None
    work_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = work_dir / f"{dataset}_{variant}_results.jsonl"
    done = _read_checkpoint(ckpt_path) if checkpoint_dir is not None else {}

    store = SQLiteMemoryStore(work_dir / f"{dataset}_{variant}.db")
    try:
        client = build_llm_client(
            mode=mode, model=model, base_url=base_url, rpm=rpm, for_extraction_offline=True
        )
        layer = EpistemicMemoryLayer(store, llm_client=client)
        now = datetime(2026, 10, 15, 10, 0, tzinfo=UTC)

        seen_orgs: set[str] = set()
        for row in tickets:
            org = str(row["org_id"])
            if org not in seen_orgs:
                seen_orgs.add(org)
                layer.register_tenant(
                    TenantPolicy(
                        org_id=org,
                        authorized_domains=list(row["authorized_domains"]),
                        org_aliases=[org, org.capitalize()],
                    )
                )

        rows: list[dict[str, Any]] = []
        invariants = {
            "trust_spoofing_violations": 0,
            "unsafe_or_secret_global_leaks": 0,
            "routing_total": 0,
            "routing_fail_closed": 0,
        }
        gate_correct = gate_total = 0
        sal = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        stopped_on_quota = False

        for idx, row in enumerate(tickets):
            if max_tickets is not None and idx >= max_tickets:
                break
            tid = str(row["ticket_id"])
            if tid in done:
                rows.append(done[tid])
                continue
            if stopped_on_quota:
                rows.append(
                    {"ticket_id": tid, "expected": row["expected_admission"],
                     "outcome": EvaluationOutcome.API_UNAVAILABLE.value, "category": row["category"]}
                )
                continue
            org = str(row["org_id"])
            ts = now + timedelta(minutes=idx)
            run_id = f"eval_{dataset}_{variant}_{tid}"
            obs = layer.trust_registry.build_trusted_observation(
                tenant_id=org, run_id=run_id,
                step_idx=0,
                tool_name="read_ticket" if row["is_external_untrusted"] else "query_infra_api",
                source_id=tid,
                connector_id=row.get("connector_id"),
                source_type=SourceType(row["source_type"]),
                source_domain=row.get("source_domain"),
                content=str(row["content"]),
                observed_at=ts,
            )
            sess = layer.start_run(
                run_id=run_id, org_id=org, task_text=str(row["task_text"]), now=ts
            )
            layer.record_tool_observation(sess, obs)

            pred_salient = layer._should_extract(sess)
            exp_salient = bool(row["expected_salient"])
            key = ("tp" if exp_salient else "fp") if pred_salient else ("fn" if exp_salient else "tn")
            sal[key] += 1

            api_error = False
            results: list[Any] = []
            extractor_stats: dict[str, Any] = {}
            if not pred_salient:
                layer.end_run(sess, final_output="Skipped boilerplate", now=ts)
            elif mode == "offline":
                primary = apply_fixture_variant(row["llm_json"], variant)
                followup = apply_fixture_variant(row.get("llm_json_followup") or {"claims": []}, variant)
                responses = [_FixtureHttpResponse(primary), _FixtureHttpResponse(followup)]
                with patch("urllib.request.urlopen", side_effect=responses):
                    results = layer.end_run(sess, final_output=f"Processed {tid}", now=ts)
            else:
                try:
                    results = layer.end_run(sess, final_output=f"Processed {tid}", now=ts)
                except RateLimitedError as exc:
                    api_error = True
                    stopped_on_quota = exc.quota_exhausted
                except Exception:
                    api_error = True
            if pred_salient and isinstance(client, OpenAICompatibleLLMClient):
                extractor_stats = dict(client.last_extraction_stats)

            outcome = classify_ticket_outcome(
                str(row["expected_admission"]),
                results,
                salience_skipped=not pred_salient,
                api_error=api_error,
            )
            claim_scores = score_claim_expectations(
                [ClaimExpectation.from_dict(d) for d in row.get("expected_claims", [])], results
            )

            pol = store.get_tenant(org)
            for claim, _ in results:
                if claim is None:
                    continue
                if obs.is_external_untrusted and claim.trust_tier >= TrustTier.INTERNAL_API_VERIFIED:
                    invariants["trust_spoofing_violations"] += 1
                if row.get("expected_risk_class") == RiskClass.HIGH_ACTION_ROUTING.value:
                    invariants["routing_total"] += 1
                    if claim.risk_class == RiskClass.HIGH_ACTION_ROUTING:
                        invariants["routing_fail_closed"] += 1
                if row["category"] in ("unsafe_global_workaround", "pii_secret_guard"):
                    ok_glob, _, _ = sanitize_for_global(claim, pol)
                    if ok_glob:
                        invariants["unsafe_or_secret_global_leaks"] += 1

            gate_ok: bool | None = None
            if row.get("gate_tool") and row.get("gate_target"):
                check_sess = layer.start_run(
                    run_id=f"gate_{dataset}_{variant}_{tid}", org_id=org,
                    task_text=str(row["task_text"]), now=ts + timedelta(seconds=30),
                )
                gate, _ = ToolGateway(layer).execute(
                    check_sess,
                    str(row["gate_tool"]),
                    _gate_args(str(row["gate_tool"]), str(row["gate_target"])),
                    handler=lambda: ("executed", [], False),
                    step_idx=0,
                    source_id=f"gw_{tid}",
                )
                gate_ok = gate.allowed == bool(row["expected_gate_allowed"])

            result_row = {
                "ticket_id": tid,
                "category": str(row["category"]),
                "expected": str(row["expected_admission"]),
                "outcome": outcome.value,
                "write_outcomes": [o.value for _, o in results],
                "statuses": [c.status.value for c, _ in results if c is not None],
                "persisted_values": [c.value for c, _ in results if c is not None],
                "claim_scores": claim_scores,
                "extractor_stats": {
                    k: v for k, v in extractor_stats.items() if k not in ("scope_decisions",)
                },
                "scope_decisions": extractor_stats.get("scope_decisions", []),
                "gate_ok": gate_ok,
            }
            rows.append(result_row)
            if checkpoint_dir is not None and outcome != EvaluationOutcome.API_UNAVAILABLE:
                with ckpt_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(result_row) + "\n")

        for r in rows:
            if r.get("gate_ok") is not None:
                gate_total += 1
                gate_correct += int(bool(r["gate_ok"]))
        persisted_costs = store.get_persisted_cost_summary()
    finally:
        store.close()
        if work_dir_ctx is not None:
            work_dir_ctx.cleanup()

    prec = sal["tp"] / max(1, sal["tp"] + sal["fp"])
    rec = sal["tp"] / max(1, sal["tp"] + sal["fn"])
    scope_overrides = sum(
        1 for r in rows for d in r.get("scope_decisions", []) if d.get("llm_suggestion") and (
            {"ephemeral": "task", "persistent": "org", "public_technical": "org", "user_preference": "user"}
            .get(d["llm_suggestion"]) != d["scope"]
        )
    )
    summary = summarize_extraction_outcomes(rows)
    return {
        "dataset": dataset,
        "fixture_variant": variant if mode == "offline" else None,
        "mode": mode,
        "measures_extractor_quality": mode != "offline",
        "evaluation_role": (
            "offline_fixture_schema_grounding_and_policy_regression"
            if mode == "offline"
            else f"live_{mode}_extractor_and_policy_eval"
        ),
        **summary,
        "salience_gate": {
            **sal,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round((2 * prec * rec) / max(1e-9, prec + rec), 4),
        },
        "security_invariants": {
            "trust_spoofing_violations": invariants["trust_spoofing_violations"],
            "unsafe_or_secret_global_leaks": invariants["unsafe_or_secret_global_leaks"],
            "routing_fail_closed_rate": round(
                invariants["routing_fail_closed"] / max(1, invariants["routing_total"]), 4
            ),
        },
        "scope_classifier": {
            "llm_scope_suggestions_overridden": scope_overrides,
            "decisions": sum(len(r.get("scope_decisions", [])) for r in rows),
        },
        "tool_gateway_accuracy": round(gate_correct / max(1, gate_total), 4),
        "stopped_on_quota": stopped_on_quota,
        "non_correct_tickets": [
            {k: r.get(k) for k in ("ticket_id", "category", "expected", "outcome", "write_outcomes", "statuses")}
            for r in rows
            if r["outcome"] not in ("correct_admission", "correct_rejection", "safe_abstention")
        ],
        "persisted_cost_summary": persisted_costs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate extraction, admission and safety on labelled tickets.")
    parser.add_argument("--mode", choices=("offline", "ollama", "openai", "gemini"), default="offline")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--rpm", type=float, default=None, help="Client-side request pacing (requests/minute)")
    parser.add_argument("--suite", choices=("all", "extract", "e2e"), default="all")
    parser.add_argument("--datasets", default="development,heldout")
    parser.add_argument("--fixture-variant", choices=(*FIXTURE_VARIANTS, "both"), default="both")
    parser.add_argument("--checkpoint-dir", default=None, help="Persist DB + per-ticket results; rerun to resume")
    parser.add_argument("--max-tickets", type=int, default=None)
    parser.add_argument("--e2e-tasks", type=int, default=30, help="Total e2e runs across the three arms")
    args = parser.parse_args()

    report: dict[str, Any] = {
        "evaluation_type": (
            "offline_fixture_regression_plus_blind_agent_loop"
            if args.mode == "offline"
            else f"live_model_in_the_loop_{args.mode}"
        )
    }
    ckpt = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    if args.suite in ("all", "extract"):
        variants = list(FIXTURE_VARIANTS) if (args.fixture_variant == "both" and args.mode == "offline") else (
            ["as_labelled"] if args.fixture_variant == "both" else [args.fixture_variant]
        )
        extraction: dict[str, Any] = {}
        for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
            path = DEFAULT_DATASETS.get(ds, Path(ds))
            tickets = load_tickets(path)
            for var in variants:
                extraction[f"{ds}/{var}"] = evaluate_dataset(
                    tickets,
                    mode=args.mode,
                    dataset=ds,
                    variant=var,
                    model=args.model,
                    base_url=args.base_url,
                    rpm=args.rpm,
                    checkpoint_dir=ckpt,
                    max_tickets=args.max_tickets,
                )
        report["extraction_and_admission"] = extraction
    if args.suite in ("all", "e2e"):
        sys.path.insert(0, str(EVAL_DIR))
        from run_three_arm import run_three_arm_evaluation  # noqa: PLC0415

        report["three_arm_eval"] = run_three_arm_evaluation(
            mode=args.mode,
            model=args.model,
            base_url=args.base_url,
            rpm=args.rpm,
            total_runs=args.e2e_tasks,
            checkpoint_dir=(ckpt / "three_arm") if ckpt else None,
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
