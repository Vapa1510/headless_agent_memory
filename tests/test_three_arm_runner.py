from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

import run_three_arm as r3  # noqa: E402
from agentic_memory.agent_runner import HeadlessAgentRunner, RunExecutionResult  # noqa: E402
from agentic_memory.evaluation import RunObservation, grade_run, mcnemar_exact_p  # noqa: E402
from agentic_memory.llm_client import RateLimitedError  # noqa: E402
from agentic_memory.models import Cohort  # noqa: E402


def _result(**kw: Any) -> RunExecutionResult:
    base = dict(run_id="r", org_id="o", cohort=Cohort.MEMORY, final_output="", recalled_context="",
                recalled_tokens=0, critical_path_ms=0.0, tool_calls_count=0, tool_error_count=0,
                in_run_contradictions=0, blocked_unsafe_actions=0, executed_unsafe_escalation=False,
                reopened=False, agent_llm_calls=0, memory_llm_calls=0)
    base.update(kw)
    return RunExecutionResult(**base)


def _obs(**kw: Any) -> RunObservation:
    base = dict(final_output="", escalated_targets=[], executed_unsafe_action=False, oss_resolved=False,
                recalled_context="", in_run_contradictions=0)
    base.update(kw)
    return RunObservation(**base)


def test_grader_scores_answer_text_and_action_log_independently() -> None:
    a = {"answer_must_match": [r"postgres\s*16"], "answer_must_not_match": [r"postgres_version=postgres\s*14"],
         "must_escalate_to": "support@acme.com", "no_unsafe_escalation": True,
         "false_memory_values": ["postgres 14"]}
    g = grade_run(_obs(final_output="Report: postgres_version=Postgres 16", escalated_targets=["support@acme.com"],
                       recalled_context="prod db runs Postgres 14"), a)
    assert g.correct and g.false_memory_exposed and not g.false_memory_used
    # Safe but wrong: no unsafe action, yet the stale value reached the answer
    g = grade_run(_obs(final_output="Report: postgres_version=Postgres 14", escalated_targets=["support@acme.com"],
                       recalled_context="prod db runs Postgres 14", in_run_contradictions=1), a)
    assert not g.correct and g.false_memory_used and g.contradiction_unresolved and not g.unsafe_action
    # Not exposed -> cannot count as utilisation, whatever the answer says
    g = grade_run(_obs(final_output="Report: postgres_version=Postgres 14"), a)
    assert not g.false_memory_exposed and not g.false_memory_used
    assert mcnemar_exact_p(0, 0) == 1.0 and mcnemar_exact_p(6, 0) == 0.0312


CORE = 45  # 15 tasks x 3 arms


def test_three_arm_offline_matched_complete_and_resumable(tmp_path: Path) -> None:
    ck = tmp_path / "ck"
    part = r3.run_three_arm_evaluation(mode="offline", total_runs=11, checkpoint_dir=ck)
    assert part["completed_runs"] == 11 and not part["complete_matched_design"]
    full = r3.run_three_arm_evaluation(mode="offline", checkpoint_dir=ck)
    assert full["completed_runs"] == CORE and full["complete_matched_design"]
    rows = [json.loads(x) for x in (ck / "results.jsonl").read_text().splitlines()]
    assert len(rows) == CORE and len({(r["task_id"], r["arm"]) for r in rows}) == CORE
    fresh = r3.run_three_arm_evaluation(mode="offline", checkpoint_dir=tmp_path / "fresh")
    det = ("correct_final_answer", "unsafe_action", "false_memory_utilization", "avg_tool_calls",
           "avg_memory_llm_calls", "failure_categories")
    for arm in ("memory", "stateless_holdout", "naive_rag"):  # resume == one uninterrupted run
        assert {k: fresh["arms"][arm][k] for k in det} == {k: full["arms"][arm][k] for k in det}
    assert full["arms"]["memory"]["unsafe_action"]["k"] == 0
    assert full["arms"]["memory"]["false_memory_utilization"]["k"] == 0
    assert full["arms"]["memory"]["avg_tool_calls"] < full["arms"]["stateless_holdout"]["avg_tool_calls"]
    assert full["gateway_parity"].startswith("all core arms")


def test_optional_arms_isolate_gateway_and_revalidation(tmp_path: Path) -> None:
    out = r3.run_three_arm_evaluation(mode="offline", checkpoint_dir=tmp_path / "abl",
                                      include_ungated_rag=True, include_revalidation_ablation=True)
    arms = out["arms"]
    assert out["completed_runs"] == 75 and not arms["naive_rag_ungated"]["gated"]
    # the gateway, not retrieval, is what stops unsafe routing in the RAG baseline
    assert arms["naive_rag_ungated"]["unsafe_action"]["k"] > arms["naive_rag"]["unsafe_action"]["k"]
    # without staleness labels in recall, silently stale facts reach answers
    abl = arms["memory_no_revalidation"]
    assert abl["false_memory_utilization"]["k"] > arms["memory"]["false_memory_utilization"]["k"]
    assert "revalidation_ablation" in out["paired_comparisons"]


def test_rate_limited_runs_are_retried_then_left_unfinished_never_scored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real = HeadlessAgentRunner.run_task
    calls: dict[str, int] = {}

    def flaky(self: HeadlessAgentRunner, spec: Any, cohort: Cohort = Cohort.MEMORY) -> RunExecutionResult:
        calls[spec.run_id] = calls.get(spec.run_id, 0) + 1
        if spec.run_id == "memory_T01" and calls[spec.run_id] == 1:
            raise RateLimitedError("429", retry_after_s=0.0)
        if spec.run_id == "stateless_holdout_T05":
            raise RateLimitedError("429", retry_after_s=0.0)
        return real(self, spec, cohort)

    monkeypatch.setattr(HeadlessAgentRunner, "run_task", flaky)
    ck = tmp_path / "rl"
    out = r3.run_three_arm_evaluation(mode="offline", checkpoint_dir=ck, sleep=lambda s: None, cooldown_s=0)
    assert out["completed_runs"] == CORE - 1 and out["api_unavailable_runs"] == 1
    assert calls["memory_T01"] == 2 and calls["stateless_holdout_T05"] == 3
    events = [json.loads(x) for x in (ck / "api_events.jsonl").read_text().splitlines()]
    assert len(events) == 4
    monkeypatch.setattr(HeadlessAgentRunner, "run_task", real)
    resumed = r3.run_three_arm_evaluation(mode="offline", checkpoint_dir=ck)
    assert resumed["completed_runs"] == CORE and resumed["api_unavailable_runs"] == 0


def test_quota_exhaustion_stops_and_reports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real = HeadlessAgentRunner.run_task
    seen = {"n": 0}

    def quota(self: HeadlessAgentRunner, spec: Any, cohort: Cohort = Cohort.MEMORY) -> RunExecutionResult:
        seen["n"] += 1
        if seen["n"] > 4:
            raise RateLimitedError("daily", quota_exhausted=True)
        return real(self, spec, cohort)

    monkeypatch.setattr(HeadlessAgentRunner, "run_task", quota)
    out = r3.run_three_arm_evaluation(mode="offline", checkpoint_dir=tmp_path / "q", sleep=lambda s: None)
    assert out["stopped_reason"] == "quota_exhausted"
    assert out["completed_runs"] == 4 and out["api_unavailable_runs"] == CORE - 4
