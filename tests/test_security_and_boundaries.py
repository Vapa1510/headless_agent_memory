from __future__ import annotations

import inspect
import io
import json
import os
import sqlite3
import subprocess
import sys
import urllib.error
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from agentic_memory import MemoryStoreProtocol
from agentic_memory.agent_runner import HeadlessAgentRunner, TaskSpec, ToolGateway
from agentic_memory.llm_client import (
    GeminiLLMClient,
    OpenAICompatibleLLMClient,
    RateLimitedError,
    SimulatedBlackBoxLLMClient,
)
from agentic_memory.memory_layer import EpistemicMemoryLayer
from agentic_memory.models import (
    GLOBAL_TENANT_ID,
    CandidateClaim,
    ClaimType,
    Cohort,
    CostOperation,
    MemoryStatus,
    RiskClass,
    Scope,
    SourceProvenance,
    SourceType,
    TenantPolicy,
    TrustTier,
    WriteOutcome,
)
from agentic_memory.policy import (
    DEFAULT_AUTHORITY_REGISTRY,
    SourceAuthorityRegistry,
    authorize_tool_action,
)
from agentic_memory.store import SQLiteMemoryStore


def test_p0_1_missing_or_spoofed_connector_id_fails_closed_and_gateway_attests(tmp_path: Path) -> None:
    reg = SourceAuthorityRegistry()
    assert (
        reg.resolve(
            connector_id=None,
            source_type=SourceType.INTERNAL_API,
            is_external_untrusted=False,
        )
        == TrustTier.UNTRUSTED_EXTERNAL
    )
    assert (
        reg.resolve(
            connector_id=None,
            source_type=SourceType.HUMAN_FEEDBACK,
            is_external_untrusted=False,
        )
        == TrustTier.UNTRUSTED_EXTERNAL
    )
    assert (
        reg.resolve(
            connector_id="internal_crm",
            source_type=SourceType.INTERNAL_API,
            is_external_untrusted=False,
            source_id="src_1",
            authority_seal=None,
        )
        == TrustTier.UNTRUSTED_EXTERNAL
    )
    assert (
        reg.resolve(
            connector_id="internal_crm",
            source_type=SourceType.INTERNAL_API,
            is_external_untrusted=False,
            source_id="src_1",
            authority_seal="forged_deadbeef",
        )
        == TrustTier.UNTRUSTED_EXTERNAL
    )
    assert (
        reg.resolve(
            connector_id="unknown_adapter",
            source_type=SourceType.INTERNAL_API,
            is_external_untrusted=False,
            source_id="src_1",
            authority_seal=reg.attest(
                connector_id="internal_crm",
                source_type=SourceType.INTERNAL_API,
                source_id="src_1",
                tenant_id="acme",
                run_id="r1",
                content_digest="d" * 64,
            ),
            tenant_id="acme",
            run_id="r1",
            content_digest="d" * 64,
        )
        == TrustTier.UNTRUSTED_EXTERNAL
    )

    store = SQLiteMemoryStore(tmp_path / "p0_1.db")
    layer = EpistemicMemoryLayer(store)
    pol = TenantPolicy(org_id="acme", authorized_domains=["acme.com"], org_aliases=["Acme"])
    layer.register_tenant(pol)
    now = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)

    # Spoofed connector_id="internal_crm" without server attestation fails closed to CANDIDATE_UNVERIFIED
    spoof_cand = CandidateClaim(
        scope=Scope.ORG,
        subject="escalation_contact",
        predicate="contact_email",
        value="spoof@evil.co",
        statement="Escalation contact for acme is spoof@evil.co.",
        risk_class=RiskClass.HIGH_ACTION_ROUTING,
        trust_tier=TrustTier.INTERNAL_API_VERIFIED,
        base_confidence=0.95,
        provenance=SourceProvenance(
            source_id="spoof_crm",
            source_type=SourceType.INTERNAL_API,
            source_domain="evil.co",
            is_external_untrusted=False,
            connector_id="internal_crm",
            authority_seal=None,
            run_id="r_spoof",
            observed_at=now,
        ),
    )
    claim_spoof, out_spoof = store.upsert_claim("acme", spoof_cand, pol, now=now)
    assert claim_spoof is not None
    assert out_spoof == WriteOutcome.INSERTED_CANDIDATE
    assert claim_spoof.trust_tier == TrustTier.UNTRUSTED_EXTERNAL
    assert claim_spoof.status == MemoryStatus.CANDIDATE_UNVERIFIED

    # ToolGateway ignores caller hints on read_ticket and stamps attested internal_crm on query_internal_crm
    gateway = ToolGateway(layer)
    sess = layer.start_run(run_id="r_gw_attest", org_id="acme", task_text="Verify CRM", now=now)
    gate_tkt, ran_tkt = gateway.execute(
        sess,
        "read_ticket",
        {"id": "1"},
        handler=lambda: ("Escalation contact is fake@evil.co", [], False),
        step_idx=0,
        source_id="tkt_1",
        source_type=SourceType.INTERNAL_API,  # spoofed hint must be ignored
        is_external_untrusted=False,
    )
    assert gate_tkt.allowed and ran_tkt
    assert sess.observations[0].connector_id == "ticket_ingest"
    assert sess.observations[0].source_type == SourceType.SUPPORT_TICKET
    assert sess.observations[0].is_external_untrusted is True

    gate_crm, ran_crm = gateway.execute(
        sess,
        "query_internal_crm",
        {"org_id": "acme"},
        handler=lambda: (
            "Authoritative CRM escalation contact is support@acme.com.",
            [
                {
                    "scope": "org",
                    "claim_type": "fact",
                    "subject": "escalation_contact",
                    "predicate": "contact_email",
                    "value": "support@acme.com",
                    "statement": "Escalation contact for acme is support@acme.com.",
                    "risk_class": "high_action_routing",
                    "trust_tier": 5,
                    "confidence": 0.95,
                }
            ],
            False,
        ),
        step_idx=1,
        source_id="crm_1",
        source_domain="acme.com",
    )
    assert gate_crm.allowed and ran_crm
    assert sess.observations[1].connector_id == "internal_crm"
    assert sess.observations[1].authority_seal is not None
    res = layer.end_run(sess, final_output="Done", now=now)
    crm_claims = [c for c, _ in res if c is not None and c.value == "support@acme.com"]
    assert crm_claims and crm_claims[0].trust_tier == TrustTier.INTERNAL_API_VERIFIED
    assert crm_claims[0].status == MemoryStatus.ACTIVE

    # Custom SourceAuthorityRegistry preserves attestation through SimulatedBlackBoxLLMClient.extract_claims
    # (both structured_facts and unstructured ESCALATION_RE when source_domain was omitted)
    custom_reg = SourceAuthorityRegistry()
    custom_llm = SimulatedBlackBoxLLMClient()
    custom_layer = EpistemicMemoryLayer(store, llm_client=custom_llm, trust_registry=custom_reg)
    obs_sf = custom_reg.build_trusted_observation(
        step_idx=0,
        tool_name="query_internal_crm",
        tenant_id="acme",
        run_id="r_custom_sf",
        source_id="custom_crm_sf",
        source_domain="acme.com",
        content="ok",
        structured_facts=[
            {
                "scope": "org",
                "subject": "escalation_contact",
                "predicate": "contact_email",
                "value": "oncall@acme.com",
                "statement": "Escalation contact is oncall@acme.com.",
                "risk_class": "high_action_routing",
                "trust_tier": int(TrustTier.INTERNAL_API_VERIFIED),
            }
        ],
    )
    sf_cands = custom_llm.extract_claims(
        run_id="r_custom_sf",
        org_id="acme",
        user_id=None,
        task_text="crm",
        observations=[obs_sf],
        recalled_claims=[],
        final_output="done",
    )
    assert len(sf_cands) == 1
    assert sf_cands[0].trust_tier == TrustTier.INTERNAL_API_VERIFIED
    assert sf_cands[0].provenance.is_external_untrusted is False

    obs_unstruct = custom_reg.build_trusted_observation(
        step_idx=0,
        tool_name="query_internal_crm",
        tenant_id="acme",
        run_id="r_custom_unstruct",
        source_id="custom_crm_unstruct",
        content="Authoritative CRM escalation contact is escalations-v2@acme.com.",
    )
    unstruct_cands = custom_llm.extract_claims(
        run_id="r_custom_unstruct",
        org_id="acme",
        user_id=None,
        task_text="crm",
        observations=[obs_unstruct],
        recalled_claims=[],
        final_output="done",
    )
    assert len(unstruct_cands) == 1
    c_unstruct, out_unstruct = store.upsert_claim(
        "acme", unstruct_cands[0], pol, now=now, authority_registry=custom_layer.trust_registry
    )
    assert c_unstruct is not None
    assert out_unstruct == WriteOutcome.SUPERSEDED
    assert c_unstruct.trust_tier == TrustTier.INTERNAL_API_VERIFIED
    assert c_unstruct.status == MemoryStatus.ACTIVE
    store.close()


def test_p0_2_scope_global_bypass_rejected_and_sybil_origin_diversity(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "p0_2.db")
    pol = TenantPolicy(org_id="acme", authorized_domains=["acme.com"], org_aliases=["Acme"])
    store.upsert_tenant(pol)
    now = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)

    # 1) Direct Scope.GLOBAL upsert is rejected
    bypass_cand = CandidateClaim(
        scope=Scope.GLOBAL,
        claim_type=ClaimType.PROCEDURE,
        subject="urllib3",
        predicate="ssl_fix",
        value="pin urllib3==2.0.7",
        statement="Pin urllib3==2.0.7 to avoid SSL proxy errors.",
        trust_tier=TrustTier.CROSS_ORG_CORROBORATED,
        provenance=DEFAULT_AUTHORITY_REGISTRY.build_trusted_provenance(
            connector_id="cross_org_synthesizer",
            source_id="bypass_1",
            source_type=SourceType.CROSS_ORG_CLUSTER,
            run_id="r_bypass",
            tenant_id="acme",
            content="pin urllib3==2.0.7",
            observed_at=now,
        ),
    )
    c_bypass, out_bypass = store.upsert_claim("acme", bypass_cand, pol, now=now)
    assert c_bypass is None
    assert out_bypass == WriteOutcome.REJECTED_GLOBAL_BYPASS
    assert store.count_stats()["global_active"] == 0

    # 2) Untrusted trust tier rejected in stage_global_candidate
    cnt_untrusted = store.stage_global_candidate(
        org_id="org_u",
        cluster_key="global|urllib3|ssl_fix",
        artifact="urllib3",
        error_sig="ssl_fix",
        sanitized_summary="Pin urllib3==2.0.7.",
        now=now,
        origin_fingerprint="orig_u",
        source_hash="src_u",
        trust_tier=TrustTier.UNTRUSTED_EXTERNAL,
    )
    assert cnt_untrusted == 0

    # 3) 5 distinct orgs sharing 1 origin_fingerprint fail promote_global_cluster
    for i in range(5):
        store.stage_global_candidate(
            org_id=f"syb_org_{i}",
            cluster_key="global|httpx|shared_origin_err",
            artifact="httpx",
            error_sig="shared_origin_err",
            sanitized_summary="Close AsyncClient in finally block.",
            now=now,
            origin_fingerprint="single_shared_webhook_origin",
            source_hash=f"src_shared_{i}",
            trust_tier=TrustTier.TOOL_OBSERVED,
        )
    c_syb, out_syb = store.promote_global_cluster(
        cluster_key="global|httpx|shared_origin_err",
        artifact="httpx",
        error_sig="shared_origin_err",
        statement="Close AsyncClient in finally block.",
        value="close AsyncClient",
        k_anonymity=5,
        now=now,
    )
    assert c_syb is None
    assert out_syb == WriteOutcome.REJECTED_LOW_VALUE

    # 4) 5 distinct orgs replaying 1 identical source_hash fail promote_global_cluster
    for i in range(5):
        store.stage_global_candidate(
            org_id=f"rep_org_{i}",
            cluster_key="global|httpx|replay_src_err",
            artifact="httpx",
            error_sig="replay_src_err",
            sanitized_summary="Close AsyncClient in finally block.",
            now=now,
            origin_fingerprint=f"orig_rep_{i}",
            source_hash="identical_replayed_hash",
            trust_tier=TrustTier.TOOL_OBSERVED,
        )
    c_rep, out_rep = store.promote_global_cluster(
        cluster_key="global|httpx|replay_src_err",
        artifact="httpx",
        error_sig="replay_src_err",
        statement="Close AsyncClient in finally block.",
        value="close AsyncClient",
        k_anonymity=5,
        now=now,
    )
    assert c_rep is None
    assert out_rep == WriteOutcome.REJECTED_LOW_VALUE

    # 5) 5 distinct orgs with diverse origins + source hashes promote atomically
    for i in range(5):
        store.stage_global_candidate(
            org_id=f"legit_org_{i}",
            cluster_key="global|httpx|pool_timeout",
            artifact="httpx",
            error_sig="pool_timeout",
            sanitized_summary="Close AsyncClient in finally block.",
            now=now,
            origin_fingerprint=f"legit_origin_{i}",
            source_hash=f"legit_src_{i}",
            trust_tier=TrustTier.TOOL_OBSERVED,
        )
    c_ok, out_ok = store.promote_global_cluster(
        cluster_key="global|httpx|pool_timeout",
        artifact="httpx",
        error_sig="pool_timeout",
        statement="Close AsyncClient in finally block.",
        value="close AsyncClient",
        k_anonymity=5,
        now=now,
    )
    assert c_ok is not None
    assert out_ok == WriteOutcome.INSERTED_ACTIVE
    assert c_ok.tenant_id == GLOBAL_TENANT_ID
    assert c_ok.scope == Scope.GLOBAL
    assert store.count_stats()["global_active"] == 1
    store.close()


def test_p0_3_mandatory_gateway_untargeted_sensitive_writes_and_hmac_strictness(tmp_path: Path) -> None:
    # 1) enforce_gate parameter is completely removed
    assert "enforce_gate" not in inspect.signature(ToolGateway.execute).parameters
    assert not hasattr(HeadlessAgentRunner, "_run_tool_step")

    pol = TenantPolicy(org_id="acme", authorized_domains=["acme.com"], org_aliases=["Acme"])
    # Untargeted EXTERNAL_WRITE and CONFIG_WRITE fail closed without verified backing memory
    for tool_name, args, expected_reason in (
        ("send_email", {"body": "hi"}, "blocked_missing_verifiable_external_target"),
        ("send_email", {"recipient_id": "ext_1"}, "blocked_missing_verifiable_external_target"),
        ("escalate_ticket", {"note": "urgent"}, "blocked_missing_verifiable_external_target"),
        ("update_webhook", {"mode": "silent"}, "blocked_missing_verifiable_external_target"),
        ("change_billing_route", {"route_id": "attacker"}, "blocked_unverified_config_write_fail_closed"),
    ):
        gate = authorize_tool_action(tool_name, args, [], pol)
        assert gate.allowed is False
        assert gate.reason == expected_reason
        assert gate.requires_verification is True

    # INTERNAL_WRITE without external targets succeeds
    g_int = authorize_tool_action("add_internal_note", {"note": "checked logs"}, [], pol)
    assert g_int.allowed is True

    # Verified CONFIG_WRITE route backed by INTERNAL_API_VERIFIED ACTIVE memory succeeds
    store = SQLiteMemoryStore(tmp_path / "p0_3_gate.db")
    layer = EpistemicMemoryLayer(store)
    layer.register_tenant(pol)
    now = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)
    store.upsert_claim(
        "acme",
        CandidateClaim(
            scope=Scope.ORG,
            subject="billing_route",
            predicate="active_route",
            value="billing_v2",
            statement="Acme active billing route is billing_v2.",
            risk_class=RiskClass.HIGH_ACTION_ROUTING,
            trust_tier=TrustTier.INTERNAL_API_VERIFIED,
            base_confidence=0.95,
            provenance=DEFAULT_AUTHORITY_REGISTRY.build_trusted_provenance(
                connector_id="internal_infra",
                source_id="cfg_v2",
                source_type=SourceType.INTERNAL_API,
                run_id="r_cfg",
                tenant_id="acme",
                content="billing_v2",
                observed_at=now,
            ),
        ),
        pol,
        now=now,
    )
    sess = layer.start_run(run_id="r_cfg_chk", org_id="acme", task_text="Update billing_route billing_v2", now=now)
    assert layer.check_tool_action(sess, "change_billing_route", {"route_id": "billing_v2"}).allowed is True
    assert layer.check_tool_action(sess, "change_billing_route", {"route_id": "attacker", "cc": "billing_v2"}).allowed is False
    # Body/note leakage, substring suffixing, and in-domain email piggyback must ALL fail closed
    assert layer.check_tool_action(sess, "send_email", {"body": "billing_v2"}).allowed is False
    assert (
        layer.check_tool_action(
            sess, "change_billing_route", {"new_route": "attacker", "note": "replacing billing_v2"}
        ).allowed
        is False
    )
    assert layer.check_tool_action(sess, "change_billing_route", {"route_id": "billing_v2:attacker"}).allowed is False
    assert (
        layer.check_tool_action(
            sess, "change_billing_route", {"route_id": "attacker", "cc": "support@acme.com"}
        ).allowed
        is False
    )
    assert (
        layer.check_tool_action(
            sess, "send_email", {"to": "support@acme.com", "recipient_id": "ext_attacker"}
        ).allowed
        is False
    )

    # 2) Strict HMAC enforcement & random non-hardcoded dev secrets
    old_env = os.environ.pop("MEMORY_ORG_HMAC_SECRET", None)
    try:
        with pytest.raises(ValueError, match="MEMORY_ORG_HMAC_SECRET"):
            SQLiteMemoryStore(tmp_path / "strict_fail.db", require_explicit_hmac_secret=True)
        with pytest.raises(ValueError, match="MEMORY_ORG_HMAC_SECRET"):
            SQLiteMemoryStore(
                tmp_path / "strict_short.db",
                hmac_secret=b"too-short",
                require_explicit_hmac_secret=True,
            )
        os.environ["MEMORY_ORG_HMAC_SECRET"] = "                "
        with pytest.raises(ValueError, match="MEMORY_ORG_HMAC_SECRET"):
            SQLiteMemoryStore(tmp_path / "strict_ws.db", require_explicit_hmac_secret=True)
        os.environ.pop("MEMORY_ORG_HMAC_SECRET", None)
        strict_ok = SQLiteMemoryStore(
            tmp_path / "strict_ok.db",
            hmac_secret=b"a-very-strong-32-byte-secret-key!",
            require_explicit_hmac_secret=True,
        )
        strict_ok.close()

        dev_a = SQLiteMemoryStore(tmp_path / "dev_a.db")
        dev_b = SQLiteMemoryStore(tmp_path / "dev_b.db")
        dev_a_reopen = SQLiteMemoryStore(tmp_path / "dev_a.db")
        assert len(dev_a._hmac_secret) == 32
        assert dev_a._hmac_secret != dev_b._hmac_secret
        assert dev_a._hmac_secret == dev_a_reopen._hmac_secret
        assert b"intelligen-epistemic-memory-salt-v1" not in dev_a._hmac_secret
        dev_a.close()
        dev_b.close()
        dev_a_reopen.close()
    finally:
        if old_env is not None:
            os.environ["MEMORY_ORG_HMAC_SECRET"] = old_env
    store.close()


def test_p1_1_and_p1_2_retention_rollups_and_window_locked_cost_math(tmp_path: Path) -> None:
    db_path = tmp_path / "rollups.db"
    store = SQLiteMemoryStore(db_path)
    pol = TenantPolicy(org_id="acme", authorized_domains=["acme.com"], org_aliases=["Acme"])
    store.upsert_tenant(pol)
    day0 = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)

    # Emit runs & cost events across Day 0 (aged > 90d), Day 150 (30d..90d redact window), Day 205 (recent < 30d)
    for d_off, tag in ((0, "old"), (150, "mid"), (205, "recent")):
        ts = day0 + timedelta(days=d_off)
        for j in range(3):
            rid = f"run_{tag}_{j}"
            store.save_run_record(
                run_id=rid,
                tenant_id="acme",
                user_id=None,
                task_text=f"Sensitive task payload {rid}",
                cohort=Cohort.MEMORY.value,
                final_output=f"Sensitive output {rid}",
                tool_calls_count=4,
                tool_error_count=0,
                in_run_contradictions=0,
                blocked_unsafe_actions=0,
                reopened=False,
                critical_path_ms=10.0,
                agent_llm_calls=4,
                memory_llm_calls=1,
                started_at=ts,
                finished_at=ts,
            )
            store.record_cost_event(
                operation=CostOperation.MEMORY_EXTRACT,
                run_id=rid,
                attempt_num=1,
                tokens_in=100,
                tokens_out=30,
                succeeded=True,
                created_at=ts,
            )

    # Add containment audit at Day 0 (> 180d) and promoted global_staging cluster at Day 0 (> 180d)
    root, _ = store.upsert_claim(
        "acme",
        CandidateClaim(
            scope=Scope.ORG,
            subject="old_err",
            predicate="flag",
            value="bad",
            statement="Old erroneous flag bad.",
            trust_tier=TrustTier.AGENT_INFERRED,
            provenance=DEFAULT_AUTHORITY_REGISTRY.build_trusted_provenance(
                connector_id="agent_reasoner",
                source_id="old_flag",
                source_type=SourceType.AGENT_INFERENCE,
                run_id="run_old_0",
                tenant_id="acme",
                content="bad",
                observed_at=day0,
            ),
        ),
        pol,
        now=day0,
    )
    assert root is not None
    store.refute_and_contain(
        tenant_id="acme", root_memory_id=root.memory_id, reason="old audit", now=day0
    )
    for oi in range(5):
        store.stage_global_candidate(
            org_id=f"st_org_{oi}",
            cluster_key="global|urllib3|old_sig",
            artifact="urllib3",
            error_sig="old_sig",
            sanitized_summary="Pin urllib3==2.0.7.",
            now=day0,
            origin_fingerprint=f"orig_{oi}",
            source_hash=f"src_{oi}",
            trust_tier=TrustTier.TOOL_OBSERVED,
        )
    store.promote_global_cluster(
        cluster_key="global|urllib3|old_sig",
        artifact="urllib3",
        error_sig="old_sig",
        statement="Pin urllib3==2.0.7.",
        value="pin 2.0.7",
        k_anonymity=5,
        now=day0,
    )

    before_lifetime = store.get_persisted_cost_summary()
    assert before_lifetime["runs_tracked"] == 9
    assert before_lifetime["memory_calls_total"] == 9
    assert before_lifetime["avg_memory_calls_per_run"] == 1.0

    day_210 = day0 + timedelta(days=210)
    comp = store.compact_and_forget(now=day_210)
    assert comp["pruned_runs"] == 3
    assert comp["redacted_runs"] == 3
    assert comp["pruned_cost_events"] == 3
    assert comp["pruned_audits"] == 1
    assert comp["pruned_global_staging"] == 5

    # Verify mid run was redacted, recent run intact, old run deleted
    mid_rec = store.get_run_record("run_mid_0")
    assert mid_rec is not None
    assert mid_rec["task_text"] == "[REDACTED]"
    assert mid_rec["final_output"] == "[REDACTED]"
    rec_recent = store.get_run_record("run_recent_0")
    assert rec_recent is not None and rec_recent["task_text"] != "[REDACTED]"
    assert store.get_run_record("run_old_0") is None

    # Lifetime cost math remains exact across rollups + uncompacted rows (no double counting!)
    after_lifetime = store.get_persisted_cost_summary()
    assert after_lifetime["runs_tracked"] == 9
    assert after_lifetime["memory_calls_total"] == 9
    assert after_lifetime["avg_memory_calls_per_run"] == 1.0

    # Windowed cost math over last 70 days aligns denominator (6 runs) & numerator (6 calls)
    windowed = store.get_persisted_cost_summary(window_days=70, now=day_210)
    assert windowed["runs_tracked"] == 6
    assert windowed["memory_calls_total"] == 6
    assert windowed["avg_memory_calls_per_run"] == 1.0

    # New org joining old already-promoted cluster does not create unpromoted duplicate
    store.stage_global_candidate(
        org_id="st_org_5",
        cluster_key="global|urllib3|old_sig",
        artifact="urllib3",
        error_sig="old_sig",
        sanitized_summary="Pin urllib3==2.0.7.",
        now=day_210,
        origin_fingerprint="orig_5",
        source_hash="src_5",
        trust_tier=TrustTier.TOOL_OBSERVED,
    )
    assert store.get_promotable_global_clusters(k_anonymity=5) == []
    store.close()

    # Verify legacy global_staging migration adds origin_fingerprint, source_hash, trust_tier cleanly
    legacy_db = tmp_path / "legacy.db"
    conn = sqlite3.connect(legacy_db)
    conn.execute(
        """
        CREATE TABLE global_staging (
            cluster_key TEXT NOT NULL,
            artifact TEXT NOT NULL,
            error_sig TEXT NOT NULL,
            org_hmac TEXT NOT NULL,
            sanitized_summary TEXT NOT NULL,
            created_at TEXT NOT NULL,
            promoted INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (cluster_key, org_hmac)
        )
        """
    )
    conn.commit()
    conn.close()
    migrated = SQLiteMemoryStore(legacy_db)
    cols = {r["name"] for r in migrated._get_conn().execute("PRAGMA table_info(global_staging)").fetchall()}
    assert {"origin_fingerprint", "source_hash", "trust_tier"}.issubset(cols)
    migrated.close()


def test_p1_3_blind_agent_loop_never_peeks_ground_truth(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "blind.db")
    seen_inputs: list[tuple[str, str, list[str]]] = []

    class _SpyBlindLLM(SimulatedBlackBoxLLMClient):
        def decide_next_action(
            self,
            *,
            run_id: str,
            task_text: str,
            recalled_context: str,
            step_history: list[dict[str, Any]],
            available_tools: list[str],
        ) -> dict[str, Any]:
            if not step_history:
                seen_inputs.append((task_text, recalled_context, list(available_tools)))
            return super().decide_next_action(
                run_id=run_id,
                task_text=task_text,
                recalled_context=recalled_context,
                step_history=step_history,
                available_tools=available_tools,
            )

    llm = _SpyBlindLLM()
    layer = EpistemicMemoryLayer(store, llm_client=llm)
    layer.register_tenant(TenantPolicy(org_id="acme", authorized_domains=["acme.com"], org_aliases=["Acme"]))
    runner = HeadlessAgentRunner(layer)
    now = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)

    res = runner.run_task(
        TaskSpec(
            org_id="acme",
            task_text="Check Acme production database, Vendor Y timezone, urllib3 bug, and escalate billing dispute",
            task_category="full_triage",
            actual_db_version="Postgres 99_SECRET_DB",
            actual_vendor_timezone="Secret/Zone_42",
            crm_escalation_contact="secret_crm@acme.com",
            oss_artifact="urllib3",
            oss_error_sig="proxy_ssl",
            oss_workaround="secret_pin==9.9.9",
            include_outbound_escalation=True,
            exploratory_steps=10,
            now=now,
        ),
        cohort=Cohort.MEMORY,
    )
    assert len(seen_inputs) == 1
    first_task_text, first_recalled, first_tools = seen_inputs[0]
    joined_initial = f"{first_task_text}\n{first_recalled}\n{' '.join(first_tools)}"
    for secret_truth in ("Postgres 99_SECRET_DB", "Secret/Zone_42", "secret_crm@acme.com", "secret_pin==9.9.9"):
        assert secret_truth not in joined_initial
    # Yet via ToolEnvironment handlers, the blind loop discovered the DB version and completed safely
    assert res.believed_db_version == "Postgres 99_SECRET_DB"
    assert res.executed_unsafe_escalation is False

    # Also verify OpenAICompatibleLLMClient.decide_next_action JSON parsing & normalization
    oa_client = OpenAICompatibleLLMClient(api_key="test-key", model="gpt-test")

    class _MockResp:
        def __init__(self, payload: dict[str, Any]) -> None:
            self._raw = json.dumps({"choices": [{"message": {"content": json.dumps(payload)}}]}).encode("utf-8")

        def __enter__(self) -> _MockResp:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return self._raw

    with patch(
        "urllib.request.urlopen",
        side_effect=[
            _MockResp({"action": "tool_call", "tool_name": "query_infra_api", "tool_args": None}),
            _MockResp({"action": "finish", "final_output": "DB ok", "cited_memory_ids": "mem_1"}),
        ],
    ):
        d1 = oa_client.decide_next_action(
            run_id="r_oa",
            task_text="Check DB",
            recalled_context="",
            step_history=[],
            available_tools=["query_infra_api"],
        )
        d2 = oa_client.decide_next_action(
            run_id="r_oa",
            task_text="Check DB",
            recalled_context="",
            step_history=[{"step": 0, "tool_name": "query_infra_api", "content": "Postgres 16"}],
            available_tools=["query_infra_api"],
        )
    assert d1["action"] == "call_tool" and d1["tool_name"] == "query_infra_api" and d1["tool_args"] == {}
    assert d2["action"] == "finish" and d2["cited_memory_ids"] == []
    assert oa_client.ledger.total_agent_calls == 2
    store.close()


def test_p2_1_and_p2_2_separated_pg16_diagnostics_and_protocol_compliance(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "p2.db")
    assert isinstance(store, MemoryStoreProtocol)
    layer = EpistemicMemoryLayer(store)
    layer.register_tenant(TenantPolicy(org_id="acme", authorized_domains=["acme.com"], org_aliases=["Acme"]))
    now = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)

    # Separated standalone PG16 error strings
    for idx, content in enumerate(
        (
            "ERROR: permission denied for view pg_stat_io.",
            'ERROR: option "process_main" requires a Boolean value.',
            "VACUUM (PROCESS_MAIN false) public.events;",
        )
    ):
        sess = layer.start_run(run_id=f"pg16_sep_{idx}", org_id="acme", task_text="Triage DB error", now=now)
        layer.record_tool_observation(
            sess,
            layer.trust_registry.build_trusted_observation(
                tenant_id=sess.org_id, run_id=sess.run_id,
                step_idx=0,
                tool_name="read_ticket",
                source_id=f"tkt_pg_{idx}",
                connector_id="ticket_ingest",
                source_domain="acme.com",
                content=content,
                observed_at=now,
            ),
        )
        res = layer.end_run(sess, final_output="Done", now=now)
        assert any(c is not None and c.value == "Postgres 16" for c, _ in res)
    store.close()


def test_p1_4_contested_and_candidate_recall_surfaces_above_unrelated_contested(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "p1_4_recall.db")
    layer = EpistemicMemoryLayer(store)
    pol = TenantPolicy(org_id="acme", authorized_domains=["acme.com"], org_aliases=["Acme"])
    layer.register_tenant(pol)
    now = datetime(2026, 10, 10, 9, 0, tzinfo=UTC)

    # 1) Seed 25 unrelated higher-confidence CONTESTED slots (~0.90 eff_conf)
    for i in range(25):
        for ver, day_off in ((f"legacy_mode_{i}_a", 2), (f"legacy_mode_{i}_b", 1)):
            sess_u = layer.start_run(run_id=f"u_{i}_{ver}", org_id="acme", task_text="unrelated", now=now - timedelta(days=day_off))
            layer.record_tool_observation(
                sess_u,
                layer.trust_registry.build_trusted_observation(
                    tenant_id=sess_u.org_id, run_id=sess_u.run_id,
                    step_idx=0,
                    tool_name="check_vendor_api",
                    source_id=f"src_u_{i}_{ver}",
                    connector_id="tool_runner",
                    source_domain="acme.com",
                    content=f"Unrelated module_{i} config is {ver}.",
                    structured_facts=[
                        {
                            "scope": "org",
                            "claim_type": "fact",
                            "subject": f"unrelated_mod_{i}",
                            "predicate": "flag_mode",
                            "value": ver,
                            "statement": f"Unrelated module_{i} flag_mode is {ver}.",
                            "risk_class": "low_diagnostic",
                            "trust_tier": int(TrustTier.TOOL_OBSERVED),
                            "confidence": 0.90,
                        }
                    ],
                    observed_at=now - timedelta(days=day_off),
                ),
            )
            layer.end_run(sess_u, final_output="Done", now=now - timedelta(days=day_off))

    # 2) Seed matched CONTESTED pair (stale Postgres 14 vs fresher PG16 working_hypothesis=True)
    sess_old = layer.start_run(run_id="c_old", org_id="acme", task_text="svc_contested engine", now=now - timedelta(days=45))
    layer.record_tool_observation(
        sess_old,
        layer.trust_registry.build_trusted_observation(
            tenant_id=sess_old.org_id, run_id=sess_old.run_id,
            step_idx=0,
            tool_name="check_db",
            source_id="src_c_old",
            connector_id="tool_runner",
            source_domain="acme.com",
            content="svc_contested uses Postgres 14.",
            structured_facts=[
                {
                    "scope": "org",
                    "claim_type": "fact",
                    "subject": "svc_contested",
                    "predicate": "engine_version",
                    "value": "Postgres 14",
                    "statement": "Acme svc_contested database engine_version was Postgres 14.",
                    "risk_class": "low_diagnostic",
                    "trust_tier": int(TrustTier.TOOL_OBSERVED),
                    "confidence": 0.85,
                }
            ],
            observed_at=now - timedelta(days=45),
        ),
    )
    layer.end_run(sess_old, final_output="Done", now=now - timedelta(days=45))

    sess_new = layer.start_run(run_id="c_new", org_id="acme", task_text="svc_contested engine", now=now - timedelta(days=1))
    layer.record_tool_observation(
        sess_new,
        layer.trust_registry.build_trusted_observation(
            tenant_id=sess_new.org_id, run_id=sess_new.run_id,
            step_idx=0,
            tool_name="check_db",
            source_id="src_c_new",
            connector_id="tool_runner",
            source_domain="acme.com",
            content="svc_contested PG16 check -> Postgres 16.",
            structured_facts=[
                {
                    "scope": "org",
                    "claim_type": "fact",
                    "subject": "svc_contested",
                    "predicate": "engine_version",
                    "value": "Postgres 16",
                    "statement": "Acme svc_contested shows pg_stat_io -> Postgres 16.",
                    "risk_class": "low_diagnostic",
                    "trust_tier": int(TrustTier.TOOL_OBSERVED),
                    "confidence": 0.56,
                }
            ],
            observed_at=now - timedelta(days=1),
        ),
    )
    layer.end_run(sess_new, final_output="Done", now=now - timedelta(days=1))

    # 3) Seed off-domain external CANDIDATE_UNVERIFIED claim (<=0.25 ceiling)
    store.upsert_claim(
        "acme",
        CandidateClaim(
            scope=Scope.ORG,
            claim_type=ClaimType.FACT,
            subject="escalation_contact_cand",
            predicate="contact_email",
            value="cand@external.co",
            statement="Candidate escalation contact is cand@external.co.",
            risk_class=RiskClass.HIGH_ACTION_ROUTING,
            trust_tier=TrustTier.UNTRUSTED_EXTERNAL,
            base_confidence=0.22,
            provenance=SourceProvenance(
                source_id="ext_cand_1",
                source_type=SourceType.SUPPORT_TICKET,
                source_domain="external.co",
                is_external_untrusted=True,
                run_id="r_cand_seed",
                observed_at=now - timedelta(days=1),
            ),
        ),
        pol,
        now=now - timedelta(days=1),
    )

    # Verify CONTESTED pair surfaces with PREFERRED-HYPOTHESIS first above unrelated 0.90 CONTESTED rows
    s_cont = layer.start_run(run_id="q_cont", org_id="acme", task_text="Check svc_contested engine_version Postgres", now=now)
    assert s_cont.active_bundle is not None
    cont_items = [
        it
        for it in s_cont.active_bundle.items
        if it.bucket == "contested_unverified"
        and it.claim.subject == "svc_contested"
        and it.claim.status == MemoryStatus.CONTESTED
    ]
    assert len(cont_items) == 2
    assert cont_items[0].claim.metadata.get("working_hypothesis") is True
    assert "PREFERRED-HYPOTHESIS" in cont_items[0].rendered_line
    assert s_cont.active_bundle.items.index(cont_items[0]) < max(
        idx for idx, it in enumerate(s_cont.active_bundle.items) if it.bucket == "contested_unverified"
    )

    # Verify low-conf CANDIDATE_UNVERIFIED quarantine claim surfaces under UNVERIFIED-DO-NOT-ACT
    s_cand = layer.start_run(run_id="q_cand", org_id="acme", task_text="Check escalation_contact_cand contact_email", now=now)
    assert s_cand.active_bundle is not None
    assert not any(
        it.bucket == "active_org" and it.claim.status == MemoryStatus.CANDIDATE_UNVERIFIED
        for it in s_cand.active_bundle.items
    )
    assert any(
        it.bucket == "contested_unverified"
        and it.claim.subject == "escalation_contact_cand"
        and "[UNVERIFIED-DO-NOT-ACT" in it.rendered_line
        for it in s_cand.active_bundle.items
    )
    store.close()


def test_p0_4_arbitrary_target_keys_blocked_and_metadata_exclusion_preserved(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "p0_4_keys.db")
    layer = EpistemicMemoryLayer(store)
    pol = TenantPolicy(org_id="acme", authorized_domains=["acme.com"], org_aliases=["Acme"])
    layer.register_tenant(pol)
    now = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)

    # Seed verified routing target (`billing_v2`) and refuted in-domain address (`deprecated@acme.com`)
    store.upsert_claim(
        "acme",
        CandidateClaim(
            scope=Scope.ORG,
            subject="billing_route",
            predicate="active_route",
            value="billing_v2",
            statement="Acme active billing route is billing_v2.",
            risk_class=RiskClass.HIGH_ACTION_ROUTING,
            trust_tier=TrustTier.INTERNAL_API_VERIFIED,
            base_confidence=0.95,
            provenance=DEFAULT_AUTHORITY_REGISTRY.build_trusted_provenance(
                connector_id="internal_infra",
                source_id="cfg_v2_keys",
                source_type=SourceType.INTERNAL_API,
                run_id="r_cfg_keys",
                tenant_id="acme",
                content="billing_v2",
                observed_at=now,
            ),
        ),
        pol,
        now=now,
    )
    bad_in_dom, _ = store.upsert_claim(
        "acme",
        CandidateClaim(
            scope=Scope.ORG,
            subject="old_mailbox",
            predicate="contact_email",
            value="deprecated@acme.com",
            statement="Legacy mailbox deprecated@acme.com.",
            risk_class=RiskClass.HIGH_ACTION_ROUTING,
            trust_tier=TrustTier.AGENT_INFERRED,
            provenance=DEFAULT_AUTHORITY_REGISTRY.build_trusted_provenance(
                connector_id="agent_reasoner",
                source_id="old_mbx",
                source_type=SourceType.AGENT_INFERENCE,
                run_id="r_old_mbx",
                tenant_id="acme",
                content="deprecated@acme.com",
                observed_at=now,
            ),
        ),
        pol,
        now=now,
    )
    assert bad_in_dom is not None
    store.refute_and_contain(
        tenant_id="acme",
        root_memory_id=bad_in_dom.memory_id,
        reason="Deprecated unmonitored mailbox",
        now=now,
    )

    sess = layer.start_run(run_id="r_keys_chk", org_id="acme", task_text="Check billing_route billing_v2", now=now)

    # 1) Novel/unlisted sink/destination keys, composite strings, URL authority confusion,
    #    nested metadata-named target dicts, and bytes/opaque values all fail closed
    #    even alongside valid allowlisted cc or verified route_id.
    for tool_name, args in (
        ("change_billing_route", {"custom_sink_x": "bad", "cc": "support@acme.com"}),
        ("change_billing_route", {"sink": "bad"}),
        ("change_billing_route", {"route_id": "billing_v2", "custom_sink_x": "bad"}),
        ("change_billing_route", {"sink": 999, "cc": "support@acme.com"}),
        ("change_billing_route", {"sink": True, "cc": "support@acme.com"}),
        ("change_billing_route", {"note": {"sink": "bad"}, "cc": "support@acme.com"}),
        ("change_billing_route", {"notes": [{"destination": "bad"}], "cc": "support@acme.com"}),
        ("change_billing_route", {"custom_sink_x": {"mode": "evil_sink"}, "cc": "support@acme.com"}),
        ("change_billing_route", {"custom_sink_x": {"note": "bad"}, "cc": "support@acme.com"}),
        ("change_billing_route", {"custom_sink_x": "evil.co", "cc": "support@acme.com"}),
        ("change_billing_route", {"custom_sink_x": "support@acme.com; attacker_sink"}),
        ("change_billing_route", {"sink": "support@acme.com, extra_unverified_token"}),
        ("change_billing_route", {"custom_sink_x": "support@acme.com; billing_v2"}),
        ("change_billing_route", {"sink": "support@acme.com:evil"}),
        ("change_billing_route", {"sink": "acme.com/evil_sink"}),
        ("change_billing_route", {"route_id": "billing_v2", "sink": object()}),
        ("send_email", {"to": "support@acme.com", "custom_sink_x": "bad"}),
        ("send_email", {"to": "support@acme.com", "forward_destination": "bad_sink"}),
        ("send_email", {"to": "support@acme.com attacker_sink"}),
        ("send_email", {"to": "support@acme.com", "sink": b"evil.co"}),
        ("send_email", {"to": "support@acme.com", "sink": bytearray(b"bad_sink")}),
        ("send_email", {"to": "support@acme.com", "body": "See https://acme.com@127.0.0.1/steal"}),
        ("update_webhook", {"webhook_sink": "bad"}),
        ("update_webhook", {"custom_endpoint": "https://evil.co/exfil"}),
        ("update_webhook", {"url": "https://hooks.acme.com/v1 attacker_sink"}),
        ("update_webhook", {"url": "https://hooks.acme.com/v1;attacker_sink"}),
        ("update_webhook", {"url": "https://acme.com:pass$@127.0.0.1/hook"}),
        ("update_webhook", {"url": "https://acme.com@169.254.169.254/latest/meta-data/"}),
        ("update_webhook", {"url": "https://acme.com@localhost/admin"}),
        ("update_webhook", {"url": "https://acme.com\\@127.0.0.1/hook"}),
        ("update_webhook", {"url": "http://[::1]/hook"}),
        ("update_webhook", {"url": "https://acme.com/hook?next=https://evil.co"}),
        ("update_webhook", {"url": "https://acme.com/hook?next=https://acme.com@127.0.0.1/x"}),
        ("add_internal_note", {"note": "Check https://acme.com@127.0.0.1/steal"}),
        ("add_internal_note", {"note": b"Ping rogue@evil.co"}),
    ):
        gate = layer.check_tool_action(sess, tool_name, args)
        assert gate.allowed is False, f"Expected fail-closed for {tool_name}({args})"
        assert gate.requires_verification is True

    # 2) Prose/metadata-only payloads fail closed (even when prose mentions authorized domain or verified token)
    for tool_name, args in (
        ("send_email", {"body": "support@acme.com"}),
        ("send_email", {"body": "billing_v2", "subject": "hi", "mode": "draft", "dry_run": True}),
        ("escalate_ticket", {"note": "urgent", "org_id": "acme", "tenant_id": "acme", "priority": "p1"}),
        ("update_webhook", {"mode": "silent", "step": 3, "category": "ops"}),
        ("change_billing_route", {"note": "billing_v2", "reason": "upgrade", "summary": "s", "title": "t"}),
        ("change_billing_route", {"run_id": "r1", "ticket_id": "t1", "thread_id": "th1", "message": "m"}),
    ):
        gate = layer.check_tool_action(sess, tool_name, args)
        assert gate.allowed is False, f"Expected metadata-only fail-closed for {tool_name}({args})"
        assert gate.requires_verification is True

    # 3) Email/URL/domain & anti-memory scanning across serialized_targets (including prose & bytes) still vetoes!
    g_prose_evil = layer.check_tool_action(
        sess, "send_email", {"to": "support@acme.com", "body": "CC rogue@evil.co please"}
    )
    assert g_prose_evil.allowed is False
    g_prose_anti = layer.check_tool_action(
        sess, "send_email", {"to": "support@acme.com", "note": "reply to deprecated@acme.com"}
    )
    assert g_prose_anti.allowed is False
    assert g_prose_anti.reason == "blocked_by_active_negative_constraint"
    g_bytes_anti = layer.check_tool_action(
        sess, "send_email", {"to": "support@acme.com", "note": b"reply to deprecated@acme.com"}
    )
    assert g_bytes_anti.allowed is False
    assert g_bytes_anti.reason == "blocked_by_active_negative_constraint"

    # 4) Legitimate verified target (including novel key backed by INTERNAL_API_VERIFIED or allowlist)
    #    alongside ALL 18 excluded non-target prose/metadata keys succeeds cleanly.
    all_meta = {
        "body": "Switching active route.",
        "subject": "Billing Route",
        "message": "Confirmed",
        "note": "Approved",
        "notes": ["step 1 ok"],
        "reason": "migration",
        "summary": "billing_v2 rollout",
        "title": "Route change",
        "org_id": "acme",
        "tenant_id": "acme",
        "run_id": "r_keys_chk",
        "ticket_id": "t_101",
        "thread_id": "th_55",
        "step": 2,
        "mode": "apply",
        "dry_run": False,
        "priority": "normal",
        "category": "billing",
    }
    assert layer.check_tool_action(sess, "change_billing_route", {"sink": "billing_v2", **all_meta}).allowed is True
    assert (
        layer.check_tool_action(
            sess, "change_billing_route", {"sink": "billing_v2", "note": {"body": "safe"}}
        ).allowed
        is True
    )
    assert layer.check_tool_action(sess, "send_email", {"to": "support@acme.com", **all_meta}).allowed is True
    assert (
        layer.check_tool_action(
            sess, "send_email", {"custom_recipients": "support@acme.com, ops@acme.com", **all_meta}
        ).allowed
        is True
    )
    assert (
        layer.check_tool_action(
            sess,
            "send_email",
            {"to": "Support@Acme.com", "body": "Check main.py and (https://acme.com/docs).", "step": 2, "dry_run": False},
        ).allowed
        is True
    )
    assert (
        layer.check_tool_action(
            sess, "update_webhook", {"custom_endpoint": "https://hooks.acme.com/v1", **all_meta}
        ).allowed
        is True
    )
    assert (
        layer.check_tool_action(
            sess, "update_webhook", {"custom_endpoint": "https://hooks.acme.com:8443/v1?token=abc", **all_meta}
        ).allowed
        is True
    )
    store.close()


def test_standalone_entrypoints_import_src_without_pythonpath(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    clean_env = {k: v for k, v in os.environ.items() if k.upper() != "PYTHONPATH"}

    eval_script = repo_root / "eval" / "run_live_eval.py"
    eval_proc = subprocess.run(
        [sys.executable, str(eval_script), "--mode", "offline", "--suite", "extract"],
        cwd=str(tmp_path),
        env=clean_env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    assert eval_proc.returncode == 0, f"eval/run_live_eval.py failed:\n{eval_proc.stdout}\n{eval_proc.stderr}"
    parsed = json.loads(eval_proc.stdout)
    dev = parsed["extraction_and_admission"]["development/as_labelled"]
    assert dev["outcome_accuracy"]["value"] == 1.0
    assert dev["unsafe_admissions_total"] == 0
    assert dev["tool_gateway_accuracy"] == 1.0
    # Reconstructed Gemini failure mode (every claim tagged task-scoped) must not cause under-learning
    mislabelled = parsed["extraction_and_admission"]["development/mislabelled_scope"]
    assert mislabelled["false_rejection_rate"]["value"] == 0.0
    assert mislabelled["scope_classifier"]["llm_scope_suggestions_overridden"] >= 20
    for key, ds in parsed["extraction_and_admission"].items():
        assert ds["unsafe_admissions_total"] == 0, key
        assert ds["security_invariants"]["trust_spoofing_violations"] == 0, key

    bench_script = repo_root / "benchmarks" / "run_benchmarks.py"
    bench_proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib.util, sys; "
                f"spec = importlib.util.spec_from_file_location('bench_mod', {str(bench_script)!r}); "
                "mod = importlib.util.module_from_spec(spec); "
                "spec.loader.exec_module(mod); "
                "assert hasattr(mod, 'run_scale_and_latency_benchmark')"
            ),
        ],
        cwd=str(tmp_path),
        env=clean_env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    assert bench_proc.returncode == 0, f"benchmarks/run_benchmarks.py failed:\n{bench_proc.stdout}\n{bench_proc.stderr}"


def test_experimental_gemini_client_auth_thought_filtering_retries_and_policy_invariants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    unconfigured = GeminiLLMClient(model="gemini-3-flash-preview")
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        unconfigured.decide_next_action(
            run_id="r_no_key",
            task_text="Check DB",
            recalled_context="",
            step_history=[],
            available_tools=["query_infra_api"],
        )

    store = SQLiteMemoryStore(tmp_path / "gemini_unit.db")
    gemini = GeminiLLMClient(api_key="unit-test-gemini-secret", model="gemini-3-flash-preview")
    layer = EpistemicMemoryLayer(store, llm_client=gemini)
    pol = TenantPolicy(org_id="acme", authorized_domains=["acme.com"], org_aliases=["Acme"])
    layer.register_tenant(pol)
    now = datetime(2026, 10, 15, 10, 0, tzinfo=UTC)

    seen_urls: list[str] = []
    seen_headers: list[dict[str, str]] = []
    attempt_counter = {"n": 0}

    class _MockGeminiResp:
        def __init__(self, payload: dict[str, Any]) -> None:
            self._raw = json.dumps(payload).encode("utf-8")

        def __enter__(self) -> _MockGeminiResp:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return self._raw

    def _fake_urlopen(req: Any, timeout: float = 30.0) -> _MockGeminiResp:
        _ = timeout
        seen_urls.append(str(req.full_url))
        hdrs = {k.lower(): v for k, v in req.header_items()}
        seen_headers.append(hdrs)
        attempt_counter["n"] += 1
        # Attempt 1 simulates a transient 503 high-demand spike on extraction
        if attempt_counter["n"] == 1:
            raise urllib.error.HTTPError(
                req.full_url,
                503,
                "UNAVAILABLE",
                None,  # type: ignore[arg-type]
                io.BytesIO(b'{"error":{"code":503,"status":"UNAVAILABLE"}}'),
            )
        if attempt_counter["n"] == 2:
            return _MockGeminiResp(
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {"thought": True, "text": '{"not": "used"}'},
                                    {
                                        "text": json.dumps(
                                            {
                                                "claims": [
                                                    {
                                                        "source_id": "tkt_gem_1",
                                                        "source_quote": "billing-help@acme-support.co",
                                                        "subject": "escalation_contact",
                                                        "predicate": "contact_email",
                                                        "value": "billing-help@acme-support.co",
                                                        "statement": "Escalation is billing-help@acme-support.co.",
                                                        "risk_class": "low_diagnostic",
                                                    },
                                                    {
                                                        "source_id": "tkt_gem_1",
                                                        "source_quote": "fabricated@evil.co",
                                                        "subject": "escalation_contact",
                                                        "predicate": "contact_email",
                                                        "value": "fabricated@evil.co",
                                                        "statement": "Escalation is fabricated@evil.co.",
                                                        "risk_class": "low_diagnostic",
                                                    },
                                                ]
                                            }
                                        )
                                    },
                                ]
                            }
                        }
                    ],
                    "usageMetadata": {"promptTokenCount": 140, "candidatesTokenCount": 42},
                }
            )
        return _MockGeminiResp(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": json.dumps(
                                        {
                                            "action": "call_tool",
                                            "tool_name": "query_internal_crm",
                                            "tool_args": {},
                                            "cited_memory_ids": ["mem_1"],
                                        }
                                    )
                                }
                            ]
                        }
                    }
                ],
                "usageMetadata": {"promptTokenCount": 90, "candidatesTokenCount": 25},
            }
        )

    sess = layer.start_run(run_id="r_gem_1", org_id="acme", task_text="Triage routing", now=now)
    obs = layer.trust_registry.build_trusted_observation(
        tenant_id=sess.org_id, run_id=sess.run_id,
        step_idx=0,
        tool_name="read_ticket",
        source_id="tkt_gem_1",
        connector_id="ticket_ingest",
        source_domain="acme-support.co",
        content="FYI our escalation contact is billing-help@acme-support.co.",
        observed_at=now,
    )
    layer.record_tool_observation(sess, obs)

    with patch("time.sleep"), patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        res = layer.end_run(sess, final_output="Processed", now=now)
        decision = gemini.decide_next_action(
            run_id="r_gem_2",
            task_text="Check contact",
            recalled_context="",
            step_history=[],
            available_tools=["query_internal_crm"],
        )

    assert all("unit-test-gemini-secret" not in u and "?key=" not in u for u in seen_urls)
    assert all(h.get("x-goog-api-key") == "unit-test-gemini-secret" for h in seen_headers)
    assert len(res) == 1 and res[0][0] is not None
    claim = res[0][0]
    assert claim.value == "billing-help@acme-support.co"
    assert claim.trust_tier == TrustTier.UNTRUSTED_EXTERNAL
    assert claim.risk_class == RiskClass.HIGH_ACTION_ROUTING
    assert claim.status == MemoryStatus.CANDIDATE_UNVERIFIED
    assert gemini.last_extraction_stats["dropped_ungrounded_span"] == 1
    assert decision["action"] == "call_tool" and decision["tool_name"] == "query_internal_crm"
    cost = store.get_persisted_cost_summary()
    assert cost["memory_extract_calls"] == 2
    assert cost["failed_attempts"] == 1
    assert cost["total_tokens_in"] > 140  # failed attempt est_tokens_in + succeeded usageMetadata
    assert cost["total_tokens_out"] == 42

    # Non-retryable quota exhaustion fails fast in 1 attempt (typed RateLimitedError so eval
    # harnesses checkpoint/resume instead of scoring the run as an agent failure)
    quota_calls = {"n": 0}

    def _quota_urlopen(req: Any, timeout: float = 30.0) -> Any:
        _ = timeout
        quota_calls["n"] += 1
        raise urllib.error.HTTPError(
            req.full_url,
            429,
            "Too Many Requests",
            None,  # type: ignore[arg-type]
            io.BytesIO(b'{"error":{"code":429,"message":"You exceeded your current quota"}}'),
        )

    with (
        patch("time.sleep"),
        patch("urllib.request.urlopen", side_effect=_quota_urlopen),
        pytest.raises(RateLimitedError) as quota_exc,
    ):
        gemini.decide_next_action(
            run_id="r_quota",
            task_text="Check DB",
            recalled_context="",
            step_history=[],
            available_tools=["query_infra_api"],
        )
    assert quota_calls["n"] == 1
    assert quota_exc.value.quota_exhausted is True
    store.close()


