from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from agentic_memory.memory_layer import EpistemicMemoryLayer
from agentic_memory.models import (
    MAX_CONTEXT_TOKENS,
    CandidateClaim,
    Cardinality,
    ClaimType,
    RiskClass,
    Scope,
    SourceProvenance,
    SourceType,
    TenantPolicy,
    TrustTier,
    Volatility,
)
from agentic_memory.policy import DEFAULT_AUTHORITY_REGISTRY, estimate_tokens
from agentic_memory.store import SQLiteMemoryStore


def test_context_budget_strictly_capped_at_4000_tokens_and_preserves_anti_memories(
    tmp_path: Path,
) -> None:
    store = SQLiteMemoryStore(tmp_path / "tokens.db")
    layer = EpistemicMemoryLayer(store)
    policy = TenantPolicy(org_id="acme", authorized_domains=["acme.com"], org_aliases=["Acme"])
    layer.register_tenant(policy)
    now = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

    # Insert 140 verbose claims (~10,000+ total tokens)
    for i in range(140):
        cand = CandidateClaim(
            scope=Scope.ORG,
            claim_type=ClaimType.FACT,
            subject=f"service_{i}",
            predicate="runtime_config",
            value=f"cfg_{i}",
            statement=(
                f"Acme service_{i} billing API pipeline handles batch settlement jobs using "
                f"dedicated connection pool profile cfg_{i} with retry backoff policy and SLA alerts."
            ),
            cardinality=Cardinality.SINGLE,
            volatility=Volatility.LOW,
            risk_class=RiskClass.MEDIUM_OPERATIONAL,
            trust_tier=TrustTier.INTERNAL_API_VERIFIED,
            base_confidence=0.9,
            provenance=DEFAULT_AUTHORITY_REGISTRY.build_trusted_provenance(
                connector_id="internal_crm",
                source_id=f"tok_src_{i}",
                source_type=SourceType.INTERNAL_API,
                run_id=f"seed_tok_{i}",
                tenant_id="acme",
                content=f"cfg_{i}",
                observed_at=now,
            ),
        )
        store.upsert_claim("acme", cand, policy, now=now)

    # Also insert 2 critical anti-memories
    bad_claim, _ = store.upsert_claim(
        "acme",
        CandidateClaim(
            scope=Scope.ORG,
            claim_type=ClaimType.FACT,
            subject="billing_api",
            predicate="timezone",
            value="UTC",
            statement="Acme billing API uses UTC.",
            cardinality=Cardinality.SINGLE,
            volatility=Volatility.MEDIUM,
            risk_class=RiskClass.LOW_DIAGNOSTIC,
            trust_tier=TrustTier.AGENT_INFERRED,
            base_confidence=0.45,
            provenance=DEFAULT_AUTHORITY_REGISTRY.build_trusted_provenance(
                connector_id="agent_reasoner",
                source_id="bad_tz",
                source_type=SourceType.AGENT_INFERENCE,
                run_id="r_bad",
                tenant_id="acme",
                content="UTC",
                observed_at=now,
            ),
        ),
        policy,
        now=now,
    )
    assert bad_claim is not None
    store.refute_and_contain(
        tenant_id="acme",
        root_memory_id=bad_claim.memory_id,
        reason="Human proved billing API returns EST",
        now=now + timedelta(hours=1),
    )

    sess = layer.start_run(
        run_id="r_big_recall",
        org_id="acme",
        task_text="Check Acme billing API pipeline batch settlement timezone",
        now=now + timedelta(hours=2),
    )
    assert sess.active_bundle is not None
    assert sess.active_bundle.token_count <= MAX_CONTEXT_TOKENS
    assert estimate_tokens(sess.active_bundle.rendered_context) <= MAX_CONTEXT_TOKENS
    assert "ANTI-MEMORY/REFUTED" in sess.active_bundle.rendered_context
    assert sess.memory_llm_calls == 0
    store.close()


def test_two_year_bounded_forgetting_with_zero_llm_calls(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "forgetting_2yr.db")
    layer = EpistemicMemoryLayer(
        store,
        max_active_per_org=40,
        max_negative_per_org=15,
        max_global_advisories=50,
    )
    policy = TenantPolicy(org_id="acme", authorized_domains=["acme.com"], org_aliases=["Acme"])
    layer.register_tenant(policy)

    start = datetime(2025, 1, 1, 9, 0, tzinfo=UTC)
    pinned_id = ""

    # Simulate 2+ years (760 days) of writes, decayed trivia, candidates, and reads
    for day in range(0, 760, 4):
        ts = start + timedelta(days=day)
        is_pinned = day == 20
        cand = CandidateClaim(
            scope=Scope.ORG,
            claim_type=ClaimType.PROCEDURE if is_pinned else ClaimType.FACT,
            subject="critical_runbook" if is_pinned else f"ephemeral_fact_{day}",
            predicate="rule",
            value=f"val_{day}",
            statement=(
                "Critical human-verified failover runbook for Acme primary cluster."
                if is_pinned
                else f"Transient ticket note #{day} for Acme environment."
            ),
            cardinality=Cardinality.SINGLE,
            volatility=Volatility.LOW if is_pinned else Volatility.HIGH,
            risk_class=RiskClass.HIGH_ACTION_ROUTING if not is_pinned and day % 8 == 0 else RiskClass.MEDIUM_OPERATIONAL,
            trust_tier=(
                TrustTier.HUMAN_VERIFIED
                if is_pinned
                else TrustTier.UNTRUSTED_EXTERNAL
                if day % 8 == 0
                else TrustTier.TOOL_OBSERVED
            ),
            base_confidence=0.96 if is_pinned else 0.70,
            provenance=(
                DEFAULT_AUTHORITY_REGISTRY.build_trusted_provenance(
                    connector_id="human_review",
                    source_id=f"src_2yr_{day}",
                    source_type=SourceType.HUMAN_FEEDBACK,
                    run_id=f"r_2yr_{day}",
                    tenant_id="acme",
                    content=f"val_{day}",
                    observed_at=ts,
                )
                if is_pinned
                else SourceProvenance(
                    source_id=f"src_2yr_{day}",
                    source_type=SourceType.SUPPORT_TICKET,
                    is_external_untrusted=True,
                    connector_id="ticket_ingest",
                    run_id=f"r_2yr_{day}",
                    observed_at=ts,
                )
                if day % 8 == 0
                else DEFAULT_AUTHORITY_REGISTRY.build_trusted_provenance(
                    connector_id="tool_runner",
                    source_id=f"src_2yr_{day}",
                    source_type=SourceType.TOOL_OUTPUT,
                    run_id=f"r_2yr_{day}",
                    tenant_id="acme",
                    content=f"val_{day}",
                    observed_at=ts,
                )
            ),
        )
        c, _ = store.upsert_claim("acme", cand, policy, now=ts)
        if is_pinned and c is not None:
            pinned_id = c.memory_id
        # Simulate frequent retrieval utility on pinned runbook
        if pinned_id and day % 20 == 0:
            store.record_run_reads(
                run_id=f"read_{day}",
                tenant_id="acme",
                memory_ids=[pinned_id],
                step_idx=0,
                read_at=ts,
            )

    end_ts = start + timedelta(days=760)
    calls_before = layer.ledger.total_memory_calls
    maint = layer.run_background_maintenance(now=end_ts)
    assert layer.ledger.total_memory_calls == calls_before  # 0 LLM calls for compaction/forgetting!

    stats = store.count_stats()
    assert stats["active"] + stats["contested"] + stats["candidates"] <= 40
    assert maint["compaction"]["expired_candidates"] > 0
    assert maint["compaction"]["purged_run_reads"] > 0

    # High-utility human-verified runbook survives 2+ years of forgetting!
    assert store.get_claim(pinned_id) is not None
    store.close()
