from __future__ import annotations

import json
import threading
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from agentic_memory.agent_runner import HeadlessAgentRunner, TaskSpec
from agentic_memory.llm_client import LLMClient, OpenAICompatibleLLMClient
from agentic_memory.memory_layer import EpistemicMemoryLayer
from agentic_memory.models import (
    MAX_CONTEXT_TOKENS,
    MAX_CRITICAL_PATH_P95_MS,
    CandidateClaim,
    Cardinality,
    ClaimType,
    CostOperation,
    FeedbackVerdict,
    HumanFeedback,
    MemoryStatus,
    RiskClass,
    Scope,
    SourceProvenance,
    SourceType,
    TenantPolicy,
    ToolObservation,
    TrustTier,
    Volatility,
    WriteOutcome,
)
from agentic_memory.policy import DEFAULT_AUTHORITY_REGISTRY
from agentic_memory.store import SQLiteMemoryStore


def test_independent_worker_stores_racing_on_same_disk_db(tmp_path: Path) -> None:
    db_path = tmp_path / "multi_worker_stores.db"
    workers = 8
    barrier = threading.Barrier(workers)
    errors: list[Exception] = []
    lock = threading.Lock()

    def _worker(idx: int) -> None:
        store: SQLiteMemoryStore | None = None
        try:
            barrier.wait(timeout=5.0)
            store = SQLiteMemoryStore(db_path)
            pol = TenantPolicy(org_id="acme", authorized_domains=["acme.com"], org_aliases=["Acme"])
            store.upsert_tenant(pol)
            cand = CandidateClaim(
                scope=Scope.ORG,
                claim_type=ClaimType.FACT,
                subject="api_gateway",
                predicate="rate_limit",
                value=f"{100 + (idx % 3)}rps",
                statement=f"Acme API gateway rate limit is {100 + (idx % 3)}rps.",
                cardinality=Cardinality.SINGLE,
                volatility=Volatility.MEDIUM,
                risk_class=RiskClass.LOW_DIAGNOSTIC,
                trust_tier=TrustTier.TOOL_OBSERVED,
                base_confidence=0.82,
                provenance=DEFAULT_AUTHORITY_REGISTRY.build_trusted_provenance(
                    connector_id="tool_runner",
                    source_id=f"worker_src_{idx}",
                    source_type=SourceType.TOOL_OUTPUT,
                    run_id=f"wr_{idx}",
                    tenant_id="acme",
                    content=f"{100 + (idx % 3)}rps",
                ),
            )
            store.upsert_claim("acme", cand, pol)
        except Exception as exc:  # pragma: no cover
            with lock:
                errors.append(exc)
        finally:
            if store is not None:
                store.close()

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15.0)

    assert not errors, f"Separate worker stores hit errors: {errors}"
    check_store = SQLiteMemoryStore(db_path)
    claims = check_store.get_claims_by_slot("acme", "org:acme|api_gateway|rate_limit")
    active = [c for c in claims if c.status == MemoryStatus.ACTIVE]
    assert len(active) <= 1
    assert len(claims) >= 1
    check_store.close()


def test_multiparent_partial_containment_and_boundary_writes(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "edge_multiparent.db")
    layer = EpistemicMemoryLayer(store)
    pol = TenantPolicy(org_id="acme", authorized_domains=["acme.com"], org_aliases=["Acme"])
    layer.register_tenant(pol)
    now = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

    # Boundary write tests: empty & oversized raw transcript dump rejected
    for label, txt in (("empty", ""), ("too_short", "ok"), ("transcript_dump", "x" * 520)):
        _, out = store.upsert_claim(
            "acme",
            CandidateClaim(
                scope=Scope.ORG,
                subject="log",
                predicate="dump",
                value=label,
                statement=txt,
                provenance=DEFAULT_AUTHORITY_REGISTRY.build_trusted_provenance(
                    connector_id="tool_runner",
                    source_id=f"bnd_{label}",
                    source_type=SourceType.TOOL_OUTPUT,
                    run_id="r_bnd",
                    tenant_id="acme",
                    content=label,
                    observed_at=now,
                ),
            ),
            pol,
            now=now,
        )
        assert out == WriteOutcome.REJECTED_LOW_VALUE

    # Multi-parent containment: M_bad (unverified) + M_clean (INTERNAL_API_VERIFIED)
    m_bad, _ = store.upsert_claim(
        "acme",
        CandidateClaim(
            scope=Scope.ORG,
            subject="vendor_y_api",
            predicate="date_timezone",
            value="UTC",
            statement="Vendor Y API returns dates in UTC.",
            trust_tier=TrustTier.AGENT_INFERRED,
            base_confidence=0.45,
            provenance=DEFAULT_AUTHORITY_REGISTRY.build_trusted_provenance(
                connector_id="agent_reasoner",
                source_id="src_bad",
                source_type=SourceType.AGENT_INFERENCE,
                run_id="r_bad",
                tenant_id="acme",
                content="UTC",
                observed_at=now,
            ),
        ),
        pol,
        now=now,
    )
    m_clean, _ = store.upsert_claim(
        "acme",
        CandidateClaim(
            scope=Scope.ORG,
            subject="billing_schedule",
            predicate="cutoff_hour",
            value="17:00",
            statement="Acme internal billing settlement closes at 17:00 local.",
            trust_tier=TrustTier.INTERNAL_API_VERIFIED,
            base_confidence=0.95,
            provenance=DEFAULT_AUTHORITY_REGISTRY.build_trusted_provenance(
                connector_id="internal_infra",
                source_id="src_clean",
                source_type=SourceType.INTERNAL_API,
                run_id="r_clean",
                tenant_id="acme",
                content="17:00",
                observed_at=now,
            ),
        ),
        pol,
        now=now,
    )
    assert m_bad is not None and m_clean is not None

    # Child 1 depends ONLY on M_bad -> should be QUARANTINED
    d_pure, _ = store.upsert_claim(
        "acme",
        CandidateClaim(
            scope=Scope.ORG,
            subject="vendor_y_api",
            predicate="utc_rule",
            value="shift_0",
            statement="Apply 0h UTC offset to Vendor Y timestamps.",
            derived_from_ids=[m_bad.memory_id],
            trust_tier=TrustTier.AGENT_INFERRED,
            base_confidence=0.48,
            provenance=DEFAULT_AUTHORITY_REGISTRY.build_trusted_provenance(
                connector_id="agent_reasoner",
                source_id="src_pure",
                source_type=SourceType.AGENT_INFERENCE,
                run_id="r_der",
                tenant_id="acme",
                content="shift_0",
                observed_at=now,
            ),
        ),
        pol,
        now=now,
    )
    # Child 2 depends on BOTH M_bad and M_clean -> should be CONTESTED (re-verify), not deleted
    d_mixed, _ = store.upsert_claim(
        "acme",
        CandidateClaim(
            scope=Scope.ORG,
            subject="billing_reconcile",
            predicate="window",
            value="aligned",
            statement="Reconcile billing settlement at 17:00 against Vendor Y UTC batch.",
            derived_from_ids=[m_bad.memory_id, m_clean.memory_id],
            trust_tier=TrustTier.AGENT_INFERRED,
            base_confidence=0.48,
            provenance=DEFAULT_AUTHORITY_REGISTRY.build_trusted_provenance(
                connector_id="agent_reasoner",
                source_id="src_mixed",
                source_type=SourceType.AGENT_INFERENCE,
                run_id="r_der",
                tenant_id="acme",
                content="aligned",
                observed_at=now,
            ),
        ),
        pol,
        now=now,
    )
    assert d_pure is not None and d_mixed is not None

    audit = store.refute_and_contain(
        tenant_id="acme",
        root_memory_id=m_bad.memory_id,
        reason="Vendor Y is EST not UTC",
        now=now + timedelta(hours=1),
    )
    assert d_pure.memory_id in audit.quarantined_memory_ids
    assert d_mixed.memory_id in audit.contested_memory_ids
    assert store.get_claim(d_pure.memory_id).status == MemoryStatus.QUARANTINED  # type: ignore[union-attr]
    assert store.get_claim(d_mixed.memory_id).status == MemoryStatus.CONTESTED  # type: ignore[union-attr]
    assert store.get_claim(m_clean.memory_id).status == MemoryStatus.ACTIVE  # type: ignore[union-attr]
    store.close()


def test_50_step_incremental_refresh_and_punctuated_fts_query(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "edge_50step.db")
    layer = EpistemicMemoryLayer(store)
    pol = TenantPolicy(org_id="acme", authorized_domains=["acme.com"], org_aliases=["Acme"])
    layer.register_tenant(pol)
    now = datetime(2026, 9, 5, 10, 0, tzinfo=UTC)

    # Seed a global advisory via staging + promote_global_cluster
    for gi in range(5):
        store.stage_global_candidate(
            org_id=f"g_edge_{gi}",
            cluster_key="global|urllib3|sslerror",
            artifact="urllib3",
            error_sig="sslerror",
            sanitized_summary="urllib3 2.1.0 SSLError behind HTTP proxy: pin urllib3==2.0.7.",
            now=now,
        )
    store.promote_global_cluster(
        cluster_key="global|urllib3|sslerror",
        artifact="urllib3",
        error_sig="sslerror",
        statement="urllib3 2.1.0 SSLError behind HTTP proxy: pin urllib3==2.0.7.",
        value="pin_2_0_7",
        k_anonymity=5,
        now=now,
    )

    sess = layer.start_run(
        run_id="run_50_steps",
        org_id="acme",
        task_text="Summarize weekly customer incidents",
        now=now,
    )
    for step in range(1, 50):
        hint = (
            "Discovered error in vendor-y/API: (urllib3) v2.1.0 SSLError!"
            if step == 12
            else None
        )
        bundle = layer.before_step(sess, step, query_hint=hint)
        assert bundle.token_count <= MAX_CONTEXT_TOKENS

    assert sess.refresh_count == 1
    assert sess.active_bundle is not None
    assert "urllib3" in sess.active_bundle.rendered_context
    cumulative_ms = sum(sess.step_critical_latencies_ms)
    assert cumulative_ms < MAX_CRITICAL_PATH_P95_MS
    store.close()


def test_transitive_unlinked_containment_and_post_purge_anti_memory_guard(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "transitive.db")
    layer = EpistemicMemoryLayer(store)
    pol = TenantPolicy(org_id="acme", authorized_domains=["acme.com"], org_aliases=["Acme"])
    layer.register_tenant(pol)
    day0 = datetime(2026, 6, 1, 9, 0, tzinfo=UTC)

    m0, _ = store.upsert_claim(
        "acme",
        CandidateClaim(
            scope=Scope.ORG,
            subject="vendor_y_api",
            predicate="date_timezone",
            value="UTC",
            statement="Vendor Y API returns dates in UTC.",
            trust_tier=TrustTier.AGENT_INFERRED,
            base_confidence=0.45,
            provenance=DEFAULT_AUTHORITY_REGISTRY.build_trusted_provenance(
                connector_id="agent_reasoner",
                source_id="s0",
                source_type=SourceType.AGENT_INFERENCE,
                run_id="r0",
                tenant_id="acme",
                content="UTC",
                observed_at=day0,
            ),
        ),
        pol,
        now=day0,
    )
    assert m0 is not None

    # Run 1 reads M0 and writes unlinked semantic derivation D1
    r1 = layer.start_run(run_id="r1", org_id="acme", task_text="Vendor Y API dates", now=day0)
    res1 = layer.end_run(
        r1,
        final_output="ok",
        explicit_candidates=[
            CandidateClaim(
                scope=Scope.ORG,
                subject="vendor_y_api",
                predicate="offset",
                value="0h",
                statement="Vendor Y settlement offset is 0h from UTC.",
                trust_tier=TrustTier.AGENT_INFERRED,
                base_confidence=0.45,
                provenance=DEFAULT_AUTHORITY_REGISTRY.build_trusted_provenance(
                    connector_id="agent_reasoner",
                    source_id="s1",
                    source_type=SourceType.AGENT_INFERENCE,
                    run_id="r1",
                    tenant_id="acme",
                    content="0h",
                    observed_at=day0,
                ),
            )
        ],
        now=day0,
    )
    d1 = res1[0][0]
    assert d1 is not None

    # Run 2 reads D1 and writes explicit child D2 on a different subject
    r2 = layer.start_run(run_id="r2", org_id="acme", task_text="settlement offset billing", now=day0 + timedelta(hours=1))
    res2 = layer.end_run(
        r2,
        final_output="ok",
        explicit_candidates=[
            CandidateClaim(
                scope=Scope.ORG,
                subject="billing_sla",
                predicate="cutoff",
                value="midnight",
                statement="Run billing batch at midnight.",
                derived_from_ids=[d1.memory_id],
                trust_tier=TrustTier.AGENT_INFERRED,
                base_confidence=0.45,
                provenance=DEFAULT_AUTHORITY_REGISTRY.build_trusted_provenance(
                    connector_id="agent_reasoner",
                    source_id="s2",
                    source_type=SourceType.AGENT_INFERENCE,
                    run_id="r2",
                    tenant_id="acme",
                    content="midnight",
                    observed_at=day0 + timedelta(hours=1),
                ),
            )
        ],
        now=day0 + timedelta(hours=1),
    )
    d2 = res2[0][0]
    assert d2 is not None

    audit = store.refute_and_contain(
        tenant_id="acme", root_memory_id=m0.memory_id, reason="Wrong UTC", now=day0 + timedelta(days=1)
    )
    # d1 has no declared link to m0 (exposure only) -> suspect, flagged for re-verification;
    # d2 declares derived_from d1 -> also suspect. Neither is quarantined on exposure alone.
    assert store.get_claim(d1.memory_id).status == MemoryStatus.CONTESTED  # type: ignore[union-attr]
    assert store.get_claim(d2.memory_id).status == MemoryStatus.CONTESTED  # type: ignore[union-attr]
    assert set(audit.suspect_memory_ids) == {d1.memory_id, d2.memory_id}
    assert audit.quarantined_memory_ids == []
    assert "r1" in audit.affected_run_ids

    # After 95-day tombstone purge, ANTI-MEMORY still blocks re-learning UTC!
    store.compact_and_forget(now=day0 + timedelta(days=95))
    relearned, out = store.upsert_claim(
        "acme",
        CandidateClaim(
            scope=Scope.ORG,
            subject="vendor_y_api",
            predicate="date_timezone",
            value="UTC",
            statement="Vendor Y API returns dates in UTC.",
            trust_tier=TrustTier.AGENT_INFERRED,
            base_confidence=0.45,
            provenance=DEFAULT_AUTHORITY_REGISTRY.build_trusted_provenance(
                connector_id="agent_reasoner",
                source_id="s_relearn",
                source_type=SourceType.AGENT_INFERENCE,
                run_id="r_95",
                tenant_id="acme",
                content="UTC",
                observed_at=day0 + timedelta(days=95),
            ),
        ),
        pol,
        now=day0 + timedelta(days=95),
    )
    assert out == WriteOutcome.DEDUPLICATED_SOURCE
    assert relearned is not None and relearned.claim_type == ClaimType.NEGATIVE_CONSTRAINT
    store.close()


def test_adjudication_keep_existing_verified_protection_and_global_feedback(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "edge_adj_glob.db")
    layer = EpistemicMemoryLayer(store, k_anonymity=5)
    pol = TenantPolicy(org_id="acme", authorized_domains=["acme.com"], org_aliases=["Acme"])
    layer.register_tenant(pol)
    now = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

    # 1) Unverified AGENT_INFERRED guess must NOT demote fresh HUMAN_VERIFIED belief
    verified, _ = store.upsert_claim(
        "acme",
        CandidateClaim(
            scope=Scope.ORG,
            subject="vendor_y_api",
            predicate="date_timezone",
            value="America/New_York",
            statement="Vendor Y API uses America/New_York.",
            trust_tier=TrustTier.HUMAN_VERIFIED,
            base_confidence=0.98,
            provenance=DEFAULT_AUTHORITY_REGISTRY.build_trusted_provenance(
                connector_id="human_review",
                source_id="h1",
                source_type=SourceType.HUMAN_FEEDBACK,
                run_id="r0",
                tenant_id="acme",
                content="America/New_York",
                observed_at=now,
            ),
        ),
        pol,
        now=now,
    )
    guess, out_guess = store.upsert_claim(
        "acme",
        CandidateClaim(
            scope=Scope.ORG,
            subject="vendor_y_api",
            predicate="date_timezone",
            value="UTC",
            statement="Vendor Y API uses UTC.",
            trust_tier=TrustTier.AGENT_INFERRED,
            base_confidence=0.45,
            provenance=DEFAULT_AUTHORITY_REGISTRY.build_trusted_provenance(
                connector_id="agent_reasoner",
                source_id="a1",
                source_type=SourceType.AGENT_INFERENCE,
                run_id="r1",
                tenant_id="acme",
                content="UTC",
                observed_at=now,
            ),
        ),
        pol,
        now=now,
    )
    assert verified is not None and guess is not None
    assert store.get_claim(verified.memory_id).status == MemoryStatus.ACTIVE  # type: ignore[union-attr]
    assert out_guess == WriteOutcome.INSERTED_CANDIDATE

    # Repeating the same guess while verified is ACTIVE must NOT promote candidate or supersede verified
    guess2, out_guess2 = store.upsert_claim(
        "acme",
        CandidateClaim(
            scope=Scope.ORG,
            subject="vendor_y_api",
            predicate="date_timezone",
            value="UTC",
            statement="Vendor Y API uses UTC.",
            trust_tier=TrustTier.AGENT_INFERRED,
            base_confidence=0.48,
            provenance=DEFAULT_AUTHORITY_REGISTRY.build_trusted_provenance(
                connector_id="agent_reasoner",
                source_id="a2",
                source_type=SourceType.AGENT_INFERENCE,
                run_id="r1b",
                tenant_id="acme",
                content="UTC",
                observed_at=now,
            ),
        ),
        pol,
        now=now,
    )
    assert guess2 is not None and out_guess2 == WriteOutcome.CORROBORATED
    assert guess2.status == MemoryStatus.CANDIDATE_UNVERIFIED
    assert store.get_claim(verified.memory_id).status == MemoryStatus.ACTIVE  # type: ignore[union-attr]

    # 2) Explicit conflict_adjudicator returning "keep_existing" preserves existing ACTIVE
    t1, _ = store.upsert_claim(
        "acme",
        CandidateClaim(
            scope=Scope.ORG,
            subject="gateway",
            predicate="timeout",
            value="30s",
            statement="Gateway timeout is 30s.",
            trust_tier=TrustTier.TOOL_OBSERVED,
            base_confidence=0.80,
            provenance=DEFAULT_AUTHORITY_REGISTRY.build_trusted_provenance(
                connector_id="tool_runner",
                source_id="g1",
                source_type=SourceType.TOOL_OUTPUT,
                run_id="rg1",
                tenant_id="acme",
                content="30s",
                observed_at=now,
            ),
        ),
        pol,
        now=now,
    )
    t2, out2 = store.upsert_claim(
        "acme",
        CandidateClaim(
            scope=Scope.ORG,
            subject="gateway",
            predicate="timeout",
            value="60s",
            statement="Gateway timeout is 60s.",
            trust_tier=TrustTier.TOOL_OBSERVED,
            base_confidence=0.82,
            provenance=DEFAULT_AUTHORITY_REGISTRY.build_trusted_provenance(
                connector_id="tool_runner",
                source_id="g2",
                source_type=SourceType.TOOL_OUTPUT,
                run_id="rg2",
                tenant_id="acme",
                content="60s",
                observed_at=now,
            ),
        ),
        pol,
        now=now,
        conflict_adjudicator=lambda _e, _i: "keep_existing",
    )
    assert t1 is not None and t2 is not None
    assert out2 == WriteOutcome.SUPERSEDED
    assert store.get_claim(t1.memory_id).status == MemoryStatus.ACTIVE  # type: ignore[union-attr]
    assert store.get_claim(t2.memory_id).status == MemoryStatus.SUPERSEDED  # type: ignore[union-attr]

    # 3) FTS5 global recall finds urllib3 despite 40 noisy advisories & generic query prefix,
    #    and 6th org joining cluster does NOT trigger an extra LLM synthesis call
    for i in range(6):
        store.stage_global_candidate(
            org_id=f"org_{i}",
            cluster_key="global|urllib3|proxy_ssl",
            artifact="urllib3",
            error_sig="proxy_ssl",
            sanitized_summary="Pin urllib3==2.0.7 for proxy SSL error.",
            now=now,
        )
    for i in range(40):
        ckey = f"global|noisy_pkg_{i}|worker_error"
        for oi in range(5):
            store.stage_global_candidate(
                org_id=f"noisy_org_{i}_{oi}",
                cluster_key=ckey,
                artifact=f"noisy_pkg_{i}",
                error_sig="worker_error",
                sanitized_summary=f"Worker failure workaround for noisy_pkg_{i}.",
                now=now,
            )
        store.promote_global_cluster(
            cluster_key=ckey,
            artifact=f"noisy_pkg_{i}",
            error_sig="worker_error",
            statement=f"Worker failure workaround for noisy_pkg_{i}.",
            # Promoted values must be supported by staged evidence (substitution guard)
            value=f"workaround noisy_pkg_{i}",
            k_anonymity=5,
            now=now,
        )
    layer.run_background_maintenance(now=now)
    synth_calls_1 = layer.ledger.snapshot()["memory_global_synth_calls"]
    store.stage_global_candidate(
        org_id="org_6",
        cluster_key="global|urllib3|proxy_ssl",
        artifact="urllib3",
        error_sig="proxy_ssl",
        sanitized_summary="Pin urllib3==2.0.7 for proxy SSL error.",
        now=now + timedelta(days=1),
    )
    layer.run_background_maintenance(now=now + timedelta(days=1))
    assert layer.ledger.snapshot()["memory_global_synth_calls"] == synth_calls_1

    s_fts = layer.start_run(
        run_id="r_fts",
        org_id="acme",
        task_text="Investigate customer ticket worker failure today please help debug urllib3",
        now=now + timedelta(days=1),
    )
    assert s_fts.active_bundle is not None
    assert "urllib3" in s_fts.active_bundle.rendered_context

    # 4) Rejecting a run attributing a GLOBAL advisory must create a tenant override, not crash!
    layer.end_run(s_fts, final_output="Applied urllib3 fix.", now=now + timedelta(days=1))
    audits = layer.ingest_delayed_feedback(
        HumanFeedback(
            run_id="r_fts",
            org_id="acme",
            verdict=FeedbackVerdict.REJECT,
            faulty_claim_hint="urllib3 proxy_ssl",
            correction_text="urllib3 pin incompatible with Acme FIPS build",
            submitted_at=now + timedelta(days=2),
        ),
        now=now + timedelta(days=2),
    )
    assert len(audits) == 1
    assert audits[0].tenant_id == "acme"

    # Subsequent task for acme respects the tenant override and does NOT use the overridden global fix
    runner = HeadlessAgentRunner(layer)
    after_override = runner.run_task(
        TaskSpec(
            org_id="acme",
            task_text="Debug urllib3 proxy_ssl in FIPS build",
            task_category="oss_bug_triage",
            oss_artifact="urllib3",
            oss_error_sig="proxy_ssl",
            oss_workaround="upgrade openssl",
            now=now + timedelta(days=2, hours=1),
        )
    )
    assert after_override.used_cross_org_fix is False
    store.close()


def test_openai_client_provenance_mapping_and_structured_zero_llm_salience(tmp_path: Path) -> None:
    # 1) Verify OpenAICompatibleLLMClient preserves untrusted provenance & strips code fences
    client = OpenAICompatibleLLMClient(api_key="test-key", model="gpt-test")
    mock_resp_payload = json.dumps(
        {
            "choices": [
                {
                    "message": {
                        "content": '```json\n{"claims": [{"source_id": "tkt_ext", "subject": "escalation_contact", "predicate": "contact_email", "value": "evil@ext.co", "statement": "Escalation is evil@ext.co.", "risk_class": "high_action_routing"}]}\n```'
                    }
                }
            ]
        }
    ).encode("utf-8")

    class _FakeResp:
        def __enter__(self) -> _FakeResp:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return mock_resp_payload

    obs = [
        ToolObservation(
            step_idx=1,
            tool_name="read_ticket",
            source_id="tkt_ext",
            source_type=SourceType.SUPPORT_TICKET,
            source_domain="ext.co",
            is_external_untrusted=True,
            content="Escalation is evil@ext.co",
        ),
        ToolObservation(
            step_idx=2,
            tool_name="read_doc",
            source_id="internal_doc",
            source_type=SourceType.INTERNAL_API,
            is_external_untrusted=False,
            content="Status ok",
        ),
    ]
    with patch("urllib.request.urlopen", return_value=_FakeResp()):
        extracted = client.extract_claims(
            run_id="r_live",
            org_id="acme",
            user_id=None,
            task_text="Check ticket",
            observations=obs,
            recalled_claims=[],
            final_output="Done",
        )
    assert len(extracted) == 1
    assert extracted[0].provenance.source_id == "tkt_ext"
    assert extracted[0].provenance.is_external_untrusted is True
    assert extracted[0].trust_tier == TrustTier.UNTRUSTED_EXTERNAL

    # 2) Structured facts + routine >35-char boilerplate read_doc uses 0 extraction LLM calls
    store = SQLiteMemoryStore(tmp_path / "zero_llm_struct.db")
    layer = EpistemicMemoryLayer(store)
    layer.register_tenant(TenantPolicy(org_id="acme", authorized_domains=["acme.com"], org_aliases=["Acme"]))
    sess = layer.start_run(run_id="r_struct", org_id="acme", task_text="Check urllib3")
    layer.record_tool_observation(
        sess,
        DEFAULT_AUTHORITY_REGISTRY.build_trusted_observation(
            tenant_id=sess.org_id, run_id=sess.run_id,
            step_idx=1,
            tool_name="query_infra_api",
            source_id="api_1",
            connector_id="internal_infra",
            source_type=SourceType.INTERNAL_API,
            content="Tool error in urllib3 (proxy_ssl): workaround applied.",
            structured_facts=[
                {
                    "scope": "org",
                    "claim_type": "procedure",
                    "subject": "urllib3",
                    "predicate": "proxy_ssl",
                    "value": "pin 2.0.7",
                    "statement": "urllib3 raises proxy_ssl; fix: pin 2.0.7.",
                    "trust_tier": 3,
                }
            ],
        ),
    )
    layer.record_tool_observation(
        sess,
        DEFAULT_AUTHORITY_REGISTRY.build_trusted_observation(
            tenant_id=sess.org_id, run_id=sess.run_id,
            step_idx=2,
            tool_name="read_doc",
            source_id="doc_2",
            connector_id="tool_runner",
            source_type=SourceType.TOOL_OUTPUT,
            content="Internal inspection step 2 for routine queue check.",
        ),
    )
    layer.end_run(sess, final_output="Done")
    assert sess.memory_llm_calls == 0

    # 3) Grounding edge cases: trailing period on version/date & diagnostic error_signature pass;
    #    ungrounded numeric fabrication without diagnostic signature still fails closed.
    ground_payload = json.dumps(
        {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "claims": [
                                    {
                                        "source_id": "obs_pg",
                                        "source_quote": "permission denied for view pg_stat_io",
                                        "subject": "prod_db",
                                        "predicate": "engine_version",
                                        "value": "Postgres 16",
                                        "statement": "DB exhibits Postgres 16 pg_stat_io error.",
                                        "risk_class": "low_diagnostic",
                                        "error_signature": "pg16_pg_stat_io",
                                    },
                                    {
                                        "source_id": "obs_ver",
                                        "source_quote": "pinning urllib3==2.0.7",
                                        "subject": "urllib3",
                                        "predicate": "fix",
                                        "value": "pin urllib3==2.0.7",
                                        "statement": "Pin urllib3==2.0.7 for proxy.",
                                        "risk_class": "low_diagnostic",
                                    },
                                    {
                                        "source_id": "obs_ver",
                                        "source_quote": "pinning urllib3==2.0.7",
                                        "subject": "urllib3",
                                        "predicate": "port",
                                        "value": "port 9999",
                                        "statement": "Use port 9999.",
                                        "risk_class": "medium_operational",
                                    },
                                    {
                                        "source_id": "obs_ver",
                                        "source_quote": "pinning urllib3==2.0.7",
                                        "subject": "prod_db",
                                        "predicate": "engine_version",
                                        "value": "Postgres 99",
                                        "statement": "Fabricated Postgres 99 via bogus signature.",
                                        "risk_class": "low_diagnostic",
                                        "error_signature": "bogus_fabricated_sig",
                                    },
                                    {
                                        "source_id": "obs_ext_sla",
                                        "source_quote": "standard queue mode",
                                        "subject": "sla_policy",
                                        "predicate": "queue_mode",
                                        "value": "standard queue mode",
                                        "statement": "Acme uses standard queue mode.",
                                        "risk_class": "low_diagnostic",
                                        "error_signature": "spoofed_external_sig",
                                    },
                                ]
                            }
                        )
                    }
                }
            ]
        }
    ).encode("utf-8")

    class _GroundResp:
        def __enter__(self) -> _GroundResp:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return ground_payload

    ground_obs = [
        ToolObservation(
            step_idx=1,
            tool_name="read_ticket",
            source_id="obs_pg",
            source_type=SourceType.SUPPORT_TICKET,
            is_external_untrusted=True,
            connector_id="ticket_ingest",
            content="ERROR: permission denied for view pg_stat_io.",
        ),
        DEFAULT_AUTHORITY_REGISTRY.build_trusted_observation(
            step_idx=2,
            tool_name="read_doc",
            tenant_id="acme",
            run_id="r_ground_edge",
            source_id="obs_ver",
            connector_id="tool_runner",
            source_type=SourceType.TOOL_OUTPUT,
            content="Fixed by pinning urllib3==2.0.7.",
        ),
        ToolObservation(
            step_idx=3,
            tool_name="read_ticket",
            source_id="obs_ext_sla",
            source_type=SourceType.SUPPORT_TICKET,
            is_external_untrusted=True,
            connector_id="ticket_ingest",
            content="Acme SLA uses standard queue mode.",
        ),
    ]
    with patch("urllib.request.urlopen", return_value=_GroundResp()):
        g_claims = client.extract_claims(
            run_id="r_ground_edge",
            org_id="acme",
            user_id=None,
            task_text="Check grounding",
            observations=ground_obs,
            recalled_claims=[],
            final_output="Done",
        )
    assert len(g_claims) == 3
    assert {c.value for c in g_claims} == {"Postgres 16", "pin urllib3==2.0.7", "standard queue mode"}
    sla_claim = next(c for c in g_claims if c.value == "standard queue mode")
    assert sla_claim.error_signature is None  # ungrounded error_signature stripped on external ticket
    assert client.last_extraction_stats["dropped_ungrounded_span"] == 2
    store.close()


def test_recall_idf_subject_boundary_anti_memory_and_mixed_extraction_fixes(tmp_path: Path) -> None:
    # 1) Boundary-safe subj_match + IDF prevents svc_6 from beating svc_61 and finds aged shard_115
    store = SQLiteMemoryStore(tmp_path / "recall_idf.db")
    layer = EpistemicMemoryLayer(store)
    pol = TenantPolicy(org_id="acme", authorized_domains=["acme.com"], org_aliases=["Acme"])
    layer.register_tenant(pol)
    now = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)

    for idx in (6, 61, 115, 124, 129):
        age_days = 115 if idx == 115 else (80 if idx == 61 else 2)
        store.upsert_claim(
            "acme",
            CandidateClaim(
                scope=Scope.ORG,
                subject=f"svc_{idx}",
                predicate="config_attr",
                value=f"val_{idx}",
                statement=(
                    f"acme service svc_{idx} (shard_{idx:03d}) uses billing queue profile p_1 "
                    f"with timeout 32s."
                ),
                trust_tier=TrustTier.INTERNAL_API_VERIFIED,
                base_confidence=0.90,
                provenance=DEFAULT_AUTHORITY_REGISTRY.build_trusted_provenance(
                    connector_id="internal_infra",
                    source_id=f"s_idf_{idx}",
                    source_type=SourceType.INTERNAL_API,
                    run_id=f"r_idf_{idx}",
                    tenant_id="acme",
                    content=f"val_{idx}",
                    observed_at=now - timedelta(days=age_days),
                ),
            ),
            pol,
            now=now - timedelta(days=age_days),
        )

    s_subj = layer.start_run(
        run_id="q_subj_boundary",
        org_id="acme",
        task_text="Investigate acme service svc_61 billing queue profile",
        now=now,
    )
    assert s_subj.active_bundle is not None
    assert s_subj.active_bundle.items[0].claim.subject == "svc_61"

    s_shard = layer.start_run(
        run_id="q_shard_idf",
        org_id="acme",
        task_text="Investigate acme billing queue shard_115 profile p_1 timeout 32s",
        now=now,
    )
    assert s_shard.active_bundle is not None
    assert s_shard.active_bundle.items[0].claim.subject == "svc_115"

    # 2) Refutation reason mentioning valid support@acme.com must NOT block support@acme.com,
    #    bare domain refutation blocks via ledger, and HUMAN_VERIFIED reinstatement clears block.
    bad_dom, _ = store.upsert_claim(
        "acme",
        CandidateClaim(
            scope=Scope.ORG,
            subject="webhook_partner",
            predicate="domain",
            value="evil-partner.co",
            statement="Partner webhook domain is evil-partner.co.",
            risk_class=RiskClass.HIGH_ACTION_ROUTING,
            trust_tier=TrustTier.AGENT_INFERRED,
            provenance=DEFAULT_AUTHORITY_REGISTRY.build_trusted_provenance(
                connector_id="agent_reasoner",
                source_id="bad_dom",
                source_type=SourceType.AGENT_INFERENCE,
                run_id="r_bdom",
                tenant_id="acme",
                content="evil-partner.co",
                observed_at=now,
            ),
        ),
        pol,
        now=now,
    )
    assert bad_dom is not None
    store.refute_and_contain(
        tenant_id="acme",
        root_memory_id=bad_dom.memory_id,
        reason="Use support@acme.com instead of evil-partner.co",
        now=now,
    )
    sess_gate = layer.start_run(run_id="r_gate_fix", org_id="acme", task_text="Escalate", now=now)
    g_good = layer.check_tool_action(sess_gate, "send_email", {"to": "support@acme.com"})
    assert g_good.allowed is True
    # Bare domain refutation in ledger blocks email & URL to that domain even with empty active_bundle
    sess_gate.active_bundle.items.clear()  # type: ignore[union-attr]
    g_bare_dom = layer.check_tool_action(sess_gate, "send_email", {"to": "ops@evil-partner.co"})
    assert g_bare_dom.allowed is False
    assert g_bare_dom.reason == "blocked_by_active_negative_constraint"

    # Non-UTC offset expires_at (+05:30 already expired in UTC) must be excluded at recall time
    ist = timezone(timedelta(hours=5, minutes=30))
    cand_tz, _ = store.upsert_claim(
        "acme",
        CandidateClaim(
            scope=Scope.ORG,
            subject="escalation_contact",
            predicate="contact_email",
            value="tz-exp@external.co",
            statement="Temp contact tz-exp@external.co.",
            risk_class=RiskClass.HIGH_ACTION_ROUTING,
            trust_tier=TrustTier.UNTRUSTED_EXTERNAL,
            provenance=SourceProvenance(
                source_id="tz_exp_1",
                source_type=SourceType.SUPPORT_TICKET,
                source_domain="external.co",
                is_external_untrusted=True,
                run_id="r_tz_exp",
                observed_at=datetime(2026, 9, 17, 14, 0, tzinfo=ist),  # 08:30 UTC -> expires Oct 1 08:30 UTC
            ),
        ),
        pol,
        now=datetime(2026, 9, 17, 14, 0, tzinfo=ist),
    )
    assert cand_tz is not None
    s_tz = layer.start_run(run_id="r_tz_chk", org_id="acme", task_text="tz-exp external", now=now)
    assert s_tz.active_bundle is not None
    assert "tz-exp@external.co" not in s_tz.active_bundle.rendered_context

    # HUMAN_VERIFIED reinstatement archives matching NEGATIVE_CONSTRAINT
    store.upsert_claim(
        "acme",
        CandidateClaim(
            scope=Scope.ORG,
            subject="webhook_partner",
            predicate="domain",
            value="evil-partner.co",
            statement="Human reinstated evil-partner.co as verified partner.",
            risk_class=RiskClass.HIGH_ACTION_ROUTING,
            trust_tier=TrustTier.HUMAN_VERIFIED,
            base_confidence=0.98,
            provenance=DEFAULT_AUTHORITY_REGISTRY.build_trusted_provenance(
                connector_id="human_review",
                source_id="hum_reinstate",
                source_type=SourceType.HUMAN_FEEDBACK,
                run_id="r_reinstate",
                tenant_id="acme",
                content="evil-partner.co",
                observed_at=now + timedelta(hours=1),
            ),
        ),
        pol,
        now=now + timedelta(hours=1),
    )
    s_rein = layer.start_run(
        run_id="r_rein_chk",
        org_id="acme",
        task_text="webhook_partner domain",
        now=now + timedelta(hours=2),
    )
    assert layer.check_tool_action(s_rein, "update_webhook", {"url": "https://evil-partner.co/v1"}).allowed is True

    # 3) Mixed structured + unstructured salient observations preserve structured facts without dropping them
    #    AND unsafe synthesize_global_advisory value is blocked.
    class _MixedLLM(LLMClient):
        def extract_claims(self, **kwargs: object) -> list[CandidateClaim]:
            self.ledger.record(CostOperation.MEMORY_EXTRACT, run_id=str(kwargs.get("run_id")))
            return []

        def adjudicate_conflict(self, **kwargs: object) -> dict[str, str]:
            return {"resolution": "contest_both", "rationale": "stub"}

        def attribute_feedback_root(self, **kwargs: object) -> list[str]:
            return []

        def synthesize_global_advisory(self, **kwargs: object) -> dict[str, str]:
            self.ledger.record(CostOperation.MEMORY_GLOBAL_SYNTH)
            return {"statement": "Pin urllib3 cleanly.", "value": "verify=False"}

        def decide_next_action(self, **kwargs: object) -> dict[str, object]:
            return {"action": "finish", "final_output": "stub"}

    layer_mix = EpistemicMemoryLayer(store, llm_client=_MixedLLM(), k_anonymity=5)
    s_mix = layer_mix.start_run(run_id="r_mix_obs", org_id="acme", task_text="Check infra", now=now)
    layer_mix.record_tool_observation(
        s_mix,
        ToolObservation(
            step_idx=0,
            tool_name="read_ticket",
            source_id="tkt_mix",
            source_type=SourceType.SUPPORT_TICKET,
            source_domain="acme.com",
            is_external_untrusted=True,
            content="Acme switched settlement payouts to Frankfurt SEPA rail with 400ms batching.",
        ),
    )
    layer_mix.record_tool_observation(
        s_mix,
        DEFAULT_AUTHORITY_REGISTRY.build_trusted_observation(
            tenant_id=s_mix.org_id, run_id=s_mix.run_id,
            step_idx=1,
            tool_name="query_infra_api",
            source_id="api_mix",
            connector_id="internal_infra",
            source_type=SourceType.INTERNAL_API,
            content="Infra API returned cache mode.",
            structured_facts=[
                {
                    "scope": "org",
                    "subject": "redis_cluster",
                    "predicate": "eviction_policy",
                    "value": "allkeys-lru",
                    "statement": "Redis cluster uses allkeys-lru.",
                    "trust_tier": 5,
                }
            ],
        ),
    )
    mix_res = layer_mix.end_run(s_mix, final_output="Done", now=now)
    assert any(c is not None and c.subject == "redis_cluster" for c, _ in mix_res)

    for oi in range(5):
        store.stage_global_candidate(
            org_id=f"g_test_{oi}",
            cluster_key="global|urllib3|tls_test",
            artifact="urllib3",
            error_sig="tls_test",
            sanitized_summary="Pin urllib3>=2.0.7 for proxy TLS.",
            now=now,
        )
    maint = layer_mix.run_background_maintenance(now=now + timedelta(hours=1))
    assert maint["promoted_global_ids"] == []
    store.close()
