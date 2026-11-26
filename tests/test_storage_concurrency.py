from __future__ import annotations

import threading
from datetime import UTC, datetime
from pathlib import Path

from agentic_memory.models import (
    CandidateClaim,
    Cardinality,
    ClaimType,
    MemoryStatus,
    RiskClass,
    Scope,
    SourceType,
    TenantPolicy,
    TrustTier,
    Volatility,
    WriteOutcome,
)
from agentic_memory.policy import DEFAULT_AUTHORITY_REGISTRY
from agentic_memory.store import SQLiteMemoryStore


def _make_candidate(
    *,
    run_id: str,
    source_id: str,
    value: str,
    trust_tier: TrustTier = TrustTier.TOOL_OBSERVED,
    confidence: float = 0.85,
) -> CandidateClaim:
    return CandidateClaim(
        scope=Scope.ORG,
        claim_type=ClaimType.FACT,
        subject="prod_db",
        predicate="engine_version",
        value=value,
        statement=f"Acme production database runs {value}.",
        cardinality=Cardinality.SINGLE,
        volatility=Volatility.MEDIUM,
        risk_class=RiskClass.LOW_DIAGNOSTIC,
        trust_tier=trust_tier,
        base_confidence=confidence,
        provenance=DEFAULT_AUTHORITY_REGISTRY.build_trusted_provenance(
            connector_id=(
                "internal_infra" if trust_tier >= TrustTier.INTERNAL_API_VERIFIED else "tool_runner"
            ),
            source_id=source_id,
            source_type=(
                SourceType.INTERNAL_API
                if trust_tier >= TrustTier.INTERNAL_API_VERIFIED
                else SourceType.TOOL_OUTPUT
            ),
            run_id=run_id,
            tenant_id="acme",
            content=value,
            observed_at=datetime(2026, 9, 1, 12, 0, tzinfo=UTC),
        ),
    )


def test_parallel_conflicting_inserts_maintain_single_active_invariant(tmp_path: Path) -> None:
    db_path = tmp_path / "concurrent_inserts.db"
    store = SQLiteMemoryStore(db_path)
    policy = TenantPolicy(org_id="acme", authorized_domains=["acme.com"], org_aliases=["Acme"])
    store.upsert_tenant(policy)

    workers = 16
    barrier = threading.Barrier(workers)
    errors: list[Exception] = []
    results: list[WriteOutcome] = []
    lock = threading.Lock()

    def _worker(idx: int) -> None:
        try:
            cand = _make_candidate(
                run_id=f"run_{idx}",
                source_id=f"src_{idx}",
                value=f"Postgres {14 + (idx % 4)}",
            )
            barrier.wait(timeout=5.0)
            _, outcome = store.upsert_claim("acme", cand, policy)
            with lock:
                results.append(outcome)
        except Exception as exc:  # pragma: no cover - test fails if populated
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15.0)

    assert not errors, f"Unexpected concurrency exceptions: {errors}"
    assert len(results) == workers

    claims = store.get_claims_by_slot("acme", "org:acme|prod_db|engine_version")
    active_claims = [c for c in claims if c.status == MemoryStatus.ACTIVE]
    contested_claims = [c for c in claims if c.status == MemoryStatus.CONTESTED]

    # Hard invariant: never more than 1 ACTIVE claim on a single-cardinality slot
    assert len(active_claims) <= 1
    assert len(active_claims) + len(contested_claims) >= 1
    store.close()


def test_parallel_updates_and_idempotent_source_dedup(tmp_path: Path) -> None:
    db_path = tmp_path / "concurrent_updates.db"
    store = SQLiteMemoryStore(db_path)
    policy = TenantPolicy(org_id="acme", authorized_domains=["acme.com"], org_aliases=["Acme"])
    store.upsert_tenant(policy)

    # Seed initial claim
    seed = _make_candidate(run_id="run_0", source_id="src_0", value="Postgres 15")
    initial_claim, outcome = store.upsert_claim("acme", seed, policy)
    assert outcome == WriteOutcome.INSERTED_ACTIVE
    assert initial_claim is not None

    # 1) Same source_id replayed concurrently by 10 workers must deduplicate cleanly (evidence_count stays 1)
    barrier = threading.Barrier(10)
    errors: list[Exception] = []

    def _replay_same_source(idx: int) -> None:
        try:
            barrier.wait(timeout=5.0)
            cand = _make_candidate(run_id=f"dup_{idx}", source_id="src_0", value="Postgres 15")
            store.upsert_claim("acme", cand, policy)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=_replay_same_source, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15.0)

    assert not errors
    after_dup = store.get_claim(initial_claim.memory_id)
    assert after_dup is not None
    assert after_dup.evidence_count == 1

    # 2) Distinct sources concurrently corroborating must accumulate all 8 distinct observations
    barrier2 = threading.Barrier(8)
    outcomes2: list[WriteOutcome] = []
    olock = threading.Lock()

    def _corroborate(idx: int) -> None:
        try:
            barrier2.wait(timeout=5.0)
            cand = _make_candidate(run_id=f"corr_{idx}", source_id=f"distinct_src_{idx}", value="Postgres 15")
            _, out = store.upsert_claim("acme", cand, policy)
            with olock:
                outcomes2.append(out)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads2 = [threading.Thread(target=_corroborate, args=(i,)) for i in range(8)]
    for t in threads2:
        t.start()
    for t in threads2:
        t.join(timeout=15.0)

    assert not errors
    assert all(o == WriteOutcome.CORROBORATED for o in outcomes2)
    after_corr = store.get_claim(initial_claim.memory_id)
    assert after_corr is not None
    assert after_corr.evidence_count == 9
    store.close()
