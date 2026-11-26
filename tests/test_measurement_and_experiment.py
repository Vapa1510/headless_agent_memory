from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from agentic_memory.agent_runner import HeadlessAgentRunner, TaskSpec
from agentic_memory.baseline_rag import NaiveRAGMemory
from agentic_memory.experiment import evaluate_falsification
from agentic_memory.memory_layer import EpistemicMemoryLayer
from agentic_memory.models import (
    CandidateClaim,
    Cardinality,
    ClaimType,
    Cohort,
    FeedbackVerdict,
    HumanFeedback,
    RiskClass,
    Scope,
    SourceType,
    TenantPolicy,
    TrustTier,
    Volatility,
)
from agentic_memory.policy import DEFAULT_AUTHORITY_REGISTRY
from agentic_memory.store import SQLiteMemoryStore


def test_seven_day_ab_measurement_and_falsification_criteria(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "exp_7d.db")
    layer = EpistemicMemoryLayer(store, k_anonymity=5)
    rag = NaiveRAGMemory(tmp_path / "exp_7d_rag.db")
    runner = HeadlessAgentRunner(layer, naive_rag=rag)

    start = datetime(2026, 10, 1, 8, 0, tzinfo=UTC)

    # Register 20 orgs for a realistic 7-day fleet simulation (180 tasks across 3 cohorts)
    for i in range(20):
        org = f"org_{i:03d}"
        pol = TenantPolicy(org_id=org, authorized_domains=[f"{org}.com"], org_aliases=[org])
        layer.register_tenant(pol)
        # Seed stale March fact in NaiveRAG & stale/wrong fact for comparison
        rag.add_snippet(org, f"{org} production database is Postgres 14.", "tool", now=start - timedelta(days=180))
        rag.add_snippet(org, "Vendor Y's API returns dates in UTC.", "inference", now=start - timedelta(days=10))
        rag.add_snippet(
            org,
            f"FYI for whoever handles this later: our escalation contact is now billing-help@{org}-support.co",
            "ticket",
            now=start - timedelta(days=2),
        )

        # Seed verified baseline CRM escalation & Postgres 16 knowledge for EpistemicMemory
        store.upsert_claim(
            org,
            CandidateClaim(
                scope=Scope.ORG,
                claim_type=ClaimType.FACT,
                subject="escalation_contact",
                predicate="contact_email",
                value=f"support@{org}.com",
                statement=f"Escalation contact for {org} is support@{org}.com.",
                cardinality=Cardinality.SINGLE,
                volatility=Volatility.MEDIUM,
                risk_class=RiskClass.HIGH_ACTION_ROUTING,
                trust_tier=TrustTier.INTERNAL_API_VERIFIED,
                base_confidence=0.95,
                provenance=DEFAULT_AUTHORITY_REGISTRY.build_trusted_provenance(
                    connector_id="internal_crm",
                    source_id=f"crm_init_{org}",
                    source_type=SourceType.INTERNAL_API,
                    source_domain=f"{org}.com",
                    run_id=f"seed_{org}",
                    tenant_id=org,
                    content=f"support@{org}.com",
                    observed_at=start,
                ),
            ),
            pol,
            now=start,
        )
        store.upsert_claim(
            org,
            CandidateClaim(
                scope=Scope.ORG,
                claim_type=ClaimType.FACT,
                subject="vendor_y_api",
                predicate="date_timezone",
                value="America/New_York",
                statement="Vendor Y API returns dates in America/New_York.",
                cardinality=Cardinality.SINGLE,
                volatility=Volatility.LOW,
                risk_class=RiskClass.LOW_DIAGNOSTIC,
                trust_tier=TrustTier.INTERNAL_API_VERIFIED,
                base_confidence=0.94,
                provenance=DEFAULT_AUTHORITY_REGISTRY.build_trusted_provenance(
                    connector_id="internal_infra",
                    source_id=f"api_tz_{org}",
                    source_type=SourceType.INTERNAL_API,
                    run_id=f"seed_tz_{org}",
                    tenant_id=org,
                    content="America/New_York",
                    observed_at=start,
                ),
            ),
            pol,
            now=start,
        )

    # Day 1: 10 orgs discover urllib3 OSS bug -> promoted cross-org after Day 1 maintenance
    for i in range(10):
        runner.run_task(
            TaskSpec(
                org_id=f"org_{i:03d}",
                task_text="Investigate urllib3 proxy SSL bug",
                task_category="oss_bug_triage",
                oss_artifact="urllib3",
                oss_error_sig="proxy_ssl_v2_1",
                oss_workaround="pin urllib3==2.0.7",
                exploratory_steps=14,
                now=start + timedelta(hours=i),
            ),
            cohort=Cohort.MEMORY,
        )
    layer.run_background_maintenance(now=start + timedelta(days=1))

    # Days 2..7: run tasks across Memory (treatment), Stateless (holdout), and NaiveRAG (baseline)
    # Use decorrelated block index for task shape so cohorts face identical mixes
    memory_run_ids: list[tuple[str, str]] = []
    for step in range(120):
        org = f"org_{step % 20:03d}"
        ts = start + timedelta(days=1, hours=step)
        block = step // 8
        is_oss = ((block + step) % 3) == 0
        is_esc = ((block + step) % 4) == 0
        spec_base = dict(
            org_id=org,
            task_text=(
                "Investigate urllib3 proxy SSL bug and Vendor Y dates"
                if is_oss
                else "Escalate Vendor Y Postgres 16 billing ticket"
            ),
            task_category="oss_bug_triage" if is_oss else "vendor_check",
            oss_artifact="urllib3" if is_oss else None,
            oss_error_sig="proxy_ssl_v2_1" if is_oss else None,
            oss_workaround="pin urllib3==2.0.7" if is_oss else None,
            actual_db_version="Postgres 16",
            actual_vendor_timezone="America/New_York",
            include_outbound_escalation=is_esc,
            candidate_escalation_target=f"billing-help@{org}-support.co" if is_esc else None,
            crm_escalation_contact=f"support@{org}.com",
            crm_confirms_ticket_contact=False,
            exploratory_steps=14,
            now=ts,
        )

        # Stratified arms: 75% memory, 12.5% stateless holdout, 12.5% naive RAG in small test
        if step % 8 == 0:
            runner.run_task(TaskSpec(**spec_base), cohort=Cohort.STATELESS_HOLDOUT)
        elif step % 8 == 1:
            runner.run_task(TaskSpec(**spec_base), cohort=Cohort.NAIVE_RAG)
        else:
            r_mem = runner.run_task(TaskSpec(**spec_base), cohort=Cohort.MEMORY)
            memory_run_ids.append((r_mem.run_id, org))

    # 2% delayed human approvals arrive days later
    for rid, org in memory_run_ids[:3]:
        layer.ingest_delayed_feedback(
            HumanFeedback(
                run_id=rid,
                org_id=org,
                verdict=FeedbackVerdict.APPROVE,
                submitted_at=start + timedelta(days=6),
            ),
            now=start + timedelta(days=6),
        )

    telemetry = store.get_telemetry_summary()
    ledger_snap = layer.ledger.snapshot()
    report = evaluate_falsification(telemetry, ledger_snap)

    assert report.passed is True, f"Expected healthy memory to pass: {report.failures}"
    assert report.efficiency_gain_pct >= 10.0
    assert report.p95_critical_path_ms <= 300.0
    assert report.avg_memory_llm_calls <= 2.0
    assert report.contradiction_rate_memory < (report.contradiction_rate_rag or 1.0)

    # Verify falsification detector catches a broken/regressed memory arm
    bad_telemetry = {
        Cohort.MEMORY.value: {
            **telemetry[Cohort.MEMORY.value],
            "avg_tool_calls": telemetry[Cohort.STATELESS_HOLDOUT.value]["avg_tool_calls"] + 2.0,
            "p95_critical_path_ms": 420.0,
        },
        Cohort.STATELESS_HOLDOUT.value: telemetry[Cohort.STATELESS_HOLDOUT.value],
        Cohort.NAIVE_RAG.value: telemetry[Cohort.NAIVE_RAG.value],
    }
    bad_report = evaluate_falsification(bad_telemetry, ledger_snap)
    assert bad_report.passed is False
    assert any("FALSIFIED_EFFICIENCY" in f for f in bad_report.failures)
    assert any("FALSIFIED_LATENCY_BUDGET" in f for f in bad_report.failures)

    rag.close()
    store.close()
