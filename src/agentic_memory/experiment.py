from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentic_memory.models import (
    MAX_AVG_LLM_CALLS_PER_RUN,
    MAX_CRITICAL_PATH_P95_MS,
    Cohort,
)


@dataclass
class FalsificationReport:
    passed: bool
    efficiency_gain_pct: float
    error_reduction_pct: float
    contradiction_rate_memory: float
    contradiction_rate_rag: float | None
    reopen_delta_pp: float
    p95_critical_path_ms: float
    avg_memory_llm_calls: float
    failures: list[str]
    summary: dict[str, Any]


def evaluate_falsification(
    telemetry_summary: dict[str, dict[str, Any]],
    ledger_snapshot: dict[str, Any],
    *,
    min_tool_call_reduction_pct: float = 10.0,
    max_reopen_regression_pp: float = 0.5,
) -> FalsificationReport:
    """
    Applies pre-registered 7-day falsification criteria against the 5% stateless
    holdout (and optional naive RAG arm) using label-free telemetry.
    """
    mem = telemetry_summary.get(Cohort.MEMORY.value)
    stateless = telemetry_summary.get(Cohort.STATELESS_HOLDOUT.value)
    rag = telemetry_summary.get(Cohort.NAIVE_RAG.value)

    failures: list[str] = []
    if not mem or not stateless:
        return FalsificationReport(
            passed=False,
            efficiency_gain_pct=0.0,
            error_reduction_pct=0.0,
            contradiction_rate_memory=0.0,
            contradiction_rate_rag=None,
            reopen_delta_pp=0.0,
            p95_critical_path_ms=0.0,
            avg_memory_llm_calls=0.0,
            failures=["missing_required_memory_or_stateless_cohort"],
            summary=telemetry_summary,
        )

    s_calls = max(1e-6, float(stateless["avg_tool_calls"]))
    m_calls = float(mem["avg_tool_calls"])
    efficiency_gain_pct = round(((s_calls - m_calls) / s_calls) * 100.0, 2)

    s_err = float(stateless["avg_tool_errors"])
    m_err = float(mem["avg_tool_errors"])
    error_reduction_pct = (
        round(((s_err - m_err) / s_err) * 100.0, 2) if s_err > 0 else (0.0 if m_err == 0 else -100.0)
    )

    m_contra = float(mem["in_run_contradiction_rate"])
    rag_contra = float(rag["in_run_contradiction_rate"]) if rag else None

    m_reopen = float(mem["reopen_rate"])
    s_reopen = float(stateless["reopen_rate"])
    reopen_delta_pp = round((m_reopen - s_reopen) * 100.0, 3)

    p95_ms = float(mem["p95_critical_path_ms"])
    # Use ledger rolling memory LLM calls per run across memory runs (or mem cohort average)
    avg_mem_llm = float(mem["avg_memory_llm_calls"])
    if ledger_snapshot.get("runs_tracked", 0) > 0:
        avg_mem_llm = max(avg_mem_llm, float(ledger_snapshot["avg_memory_calls_per_run"]))

    if efficiency_gain_pct < min_tool_call_reduction_pct:
        failures.append(
            f"FALSIFIED_EFFICIENCY: tool-call reduction {efficiency_gain_pct:.2f}% < {min_tool_call_reduction_pct:.1f}%"
        )
    if m_err > s_err + 1e-6:
        failures.append(
            f"FALSIFIED_TOOL_ERRORS: memory tool errors ({m_err:.4f}) exceed stateless ({s_err:.4f})"
        )
    if reopen_delta_pp > max_reopen_regression_pp:
        failures.append(
            f"FALSIFIED_REOPEN_RATE: reopen delta +{reopen_delta_pp:.3f}pp > +{max_reopen_regression_pp}pp"
        )
    if rag_contra is not None and rag_contra > 0.0 and m_contra >= rag_contra:
        failures.append(
            f"FALSIFIED_BELIEF_CLEANLINESS: memory contradiction rate ({m_contra:.4f}) >= naive RAG ({rag_contra:.4f})"
        )
    if p95_ms > MAX_CRITICAL_PATH_P95_MS:
        failures.append(
            f"FALSIFIED_LATENCY_BUDGET: p95 critical-path latency {p95_ms:.2f}ms > {MAX_CRITICAL_PATH_P95_MS}ms"
        )
    if avg_mem_llm > MAX_AVG_LLM_CALLS_PER_RUN:
        failures.append(
            f"FALSIFIED_COST_BUDGET: avg memory LLM calls/run {avg_mem_llm:.3f} > {MAX_AVG_LLM_CALLS_PER_RUN}"
        )

    return FalsificationReport(
        passed=(len(failures) == 0),
        efficiency_gain_pct=efficiency_gain_pct,
        error_reduction_pct=error_reduction_pct,
        contradiction_rate_memory=m_contra,
        contradiction_rate_rag=rag_contra,
        reopen_delta_pp=reopen_delta_pp,
        p95_critical_path_ms=p95_ms,
        avg_memory_llm_calls=round(avg_mem_llm, 4),
        failures=failures,
        summary=telemetry_summary,
    )
