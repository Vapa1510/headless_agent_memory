from __future__ import annotations

import contextlib
import hashlib
import json
import os
import random
import re
import secrets
import sqlite3
import threading
import time
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, TypeVar

from agentic_memory.global_semantics import check_against_evidence, contributor_conflict
from agentic_memory.models import (
    GLOBAL_TENANT_ID,
    CandidateClaim,
    Cardinality,
    ClaimType,
    ContainmentAuditEvent,
    CostOperation,
    MemoryClaim,
    MemoryStatus,
    RiskClass,
    Scope,
    SourceType,
    TenantPolicy,
    TrustTier,
    Volatility,
    WriteOutcome,
    new_id,
    utcnow,
)
from agentic_memory.policy import (
    SourceAuthorityRegistry,
    assert_valid_status_transition,
    canonicalize_slot,
    claims_semantically_related,
    compute_retention_utility,
    effective_confidence,
    evaluate_candidate_write,
    global_value_supported,
    hash_source_id,
    normalize_candidate_authority,
    org_hmac,
    statement_grounded_in_evidence,
    validate_global_metadata_field,
    validate_global_safety,
    value_digest,
)

T = TypeVar("T")

_SCOPE_MAP = {e.value: e for e in Scope}
_CLAIM_TYPE_MAP = {e.value: e for e in ClaimType}
_CARDINALITY_MAP = {e.value: e for e in Cardinality}
_VOLATILITY_MAP = {e.value: e for e in Volatility}
_RISK_MAP = {e.value: e for e in RiskClass}
_STATUS_MAP = {e.value: e for e in MemoryStatus}
_SOURCE_TYPE_MAP = {e.value: e for e in SourceType}
_TRUST_MAP = {int(e): e for e in TrustTier}

_CLAIM_COLS = (
    "memory_id",
    "tenant_id",
    "scope",
    "user_id",
    "task_id",
    "claim_type",
    "slot_key",
    "subject",
    "predicate",
    "value",
    "value_hash",
    "statement",
    "cardinality",
    "volatility",
    "risk_class",
    "trust_tier",
    "status",
    "base_confidence",
    "evidence_count",
    "derived_from_json",
    "created_in_run_id",
    "source_type",
    "source_id",
    "source_domain",
    "created_at",
    "updated_at",
    "last_accessed_at",
    "access_count",
    "verified_helps",
    "contradiction_hits",
    "version",
    "expires_at",
    "metadata_json",
)
_CLAIM_SELECT = ", ".join(_CLAIM_COLS)
_M_CLAIM_SELECT = ", ".join(f"m.{c}" for c in _CLAIM_COLS)


_GLOBAL_FTS_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "for",
        "with",
        "this",
        "that",
        "from",
        "into",
        "about",
        "check",
        "investigate",
        "triage",
        "debug",
        "summarize",
        "customer",
        "ticket",
        "task",
        "issue",
        "error",
        "failure",
        "worker",
        "run",
        "please",
        "help",
        "today",
    }
)


def _chunked(items: list[str], size: int = 400) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


class SQLiteMemoryStore:
    """
    Disk-backed SQLite WAL persistence store supporting concurrent workers,
    strict tenant isolation, partial unique single-cardinality invariants,
    provenance DAG containment, k-anonymous global staging, and bounded retention.
    """

    def __init__(
        self,
        db_path: str | Path,
        hmac_secret: bytes | None = None,
        *,
        require_explicit_hmac_secret: bool = False,
        hmac_key_version: int = 1,
    ) -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._hmac_key_version = hmac_key_version
        env_secret = os.environ.get("MEMORY_ORG_HMAC_SECRET", "").strip()
        raw_secret: bytes | None = (
            hmac_secret
            if hmac_secret is not None
            else (env_secret.encode("utf-8") if env_secret else None)
        )
        if require_explicit_hmac_secret and (raw_secret is None or len(raw_secret) < 16):
            raise ValueError(
                "MEMORY_ORG_HMAC_SECRET (>= 16 bytes) must be set in strict/production mode"
            )
        self._local = threading.local()
        self._conn_lock = threading.Lock()
        self._all_conns: list[sqlite3.Connection] = []
        self._init_schema()
        if raw_secret:
            self._hmac_secret = hashlib.sha256(
                f"key_v{hmac_key_version}:".encode() + raw_secret
            ).digest()
        else:
            dev_raw = self._load_or_create_dev_hmac_secret()
            self._hmac_secret = hashlib.sha256(
                f"key_v{hmac_key_version}:".encode() + dev_raw
            ).digest()

    def _load_or_create_dev_hmac_secret(self) -> bytes:
        if self.db_path == ":memory:":
            return secrets.token_bytes(32)

        def _tx(conn: sqlite3.Connection) -> bytes:
            row = conn.execute(
                "SELECT value FROM store_meta WHERE key = 'dev_hmac_secret'"
            ).fetchone()
            if row is not None and row["value"]:
                return bytes.fromhex(str(row["value"]))
            generated = secrets.token_bytes(32)
            conn.execute(
                "INSERT OR IGNORE INTO store_meta(key, value) VALUES ('dev_hmac_secret', ?)",
                (generated.hex(),),
            )
            row2 = conn.execute(
                "SELECT value FROM store_meta WHERE key = 'dev_hmac_secret'"
            ).fetchone()
            return bytes.fromhex(str(row2["value"])) if row2 is not None else generated

        return self._run_write_tx(_tx)

    def _get_conn(self) -> sqlite3.Connection:
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is not None:
            return conn
        conn = sqlite3.connect(
            self.db_path,
            timeout=15.0,
            check_same_thread=False,
            isolation_level=None,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 15000;")
        for attempt in range(8):
            try:
                conn.execute("PRAGMA journal_mode = WAL;")
                break
            except sqlite3.OperationalError as exc:
                if ("locked" in str(exc).lower() or "busy" in str(exc).lower()) and attempt < 7:
                    time.sleep(0.01 * (attempt + 1))
                    continue
                raise
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA temp_store = MEMORY;")
        conn.execute("PRAGMA cache_size = -16384;")
        self._local.conn = conn
        with self._conn_lock:
            self._all_conns.append(conn)
        return conn

    def close(self) -> None:
        with self._conn_lock:
            for conn in self._all_conns:
                with contextlib.suppress(sqlite3.Error):
                    conn.close()
            self._all_conns.clear()
        if hasattr(self._local, "conn"):
            delattr(self._local, "conn")

    def _run_write_tx(self, fn: Callable[[sqlite3.Connection], T], max_retries: int = 12) -> T:
        conn = self._get_conn()
        if conn.in_transaction:
            sp_name = new_id("sp")
            conn.execute(f"SAVEPOINT {sp_name};")
            try:
                result = fn(conn)
                conn.execute(f"RELEASE SAVEPOINT {sp_name};")
                return result
            except Exception:
                with contextlib.suppress(sqlite3.Error):
                    conn.execute(f"ROLLBACK TO SAVEPOINT {sp_name};")
                    conn.execute(f"RELEASE SAVEPOINT {sp_name};")
                raise
        for attempt in range(max_retries):
            try:
                conn.execute("BEGIN IMMEDIATE;")
                try:
                    result = fn(conn)
                    conn.execute("COMMIT;")
                    return result
                except Exception:
                    with contextlib.suppress(sqlite3.Error):
                        conn.execute("ROLLBACK;")
                    raise
            except sqlite3.OperationalError as exc:
                msg = str(exc).lower()
                if ("locked" in msg or "busy" in msg) and attempt + 1 < max_retries:
                    time.sleep(min(0.05, 0.002 * (2**attempt)) + random.uniform(0.0, 0.003))
                    continue
                raise
        raise RuntimeError("Exceeded max SQLite write retries")

    def _init_schema(self) -> None:
        def _apply(conn: sqlite3.Connection) -> None:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS tenants (
                    org_id TEXT PRIMARY KEY,
                    authorized_domains_json TEXT NOT NULL,
                    org_aliases_json TEXT NOT NULL,
                    hmac_id TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    user_id TEXT,
                    task_id TEXT,
                    claim_type TEXT NOT NULL,
                    slot_key TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    value TEXT NOT NULL,
                    value_hash TEXT NOT NULL,
                    statement TEXT NOT NULL,
                    cardinality TEXT NOT NULL,
                    volatility TEXT NOT NULL,
                    risk_class TEXT NOT NULL,
                    trust_tier INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    base_confidence REAL NOT NULL,
                    evidence_count INTEGER NOT NULL,
                    derived_from_json TEXT NOT NULL,
                    created_in_run_id TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    source_domain TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_accessed_at TEXT NOT NULL,
                    access_count INTEGER NOT NULL,
                    verified_helps INTEGER NOT NULL,
                    contradiction_hits INTEGER NOT NULL,
                    version INTEGER NOT NULL,
                    expires_at TEXT,
                    metadata_json TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_memories_recall
                    ON memories(tenant_id, status, scope);
                CREATE INDEX IF NOT EXISTS idx_memories_slot
                    ON memories(tenant_id, slot_key, status);
                CREATE INDEX IF NOT EXISTS idx_memories_run
                    ON memories(tenant_id, created_in_run_id);
                CREATE INDEX IF NOT EXISTS idx_memories_status_updated
                    ON memories(status, updated_at);
                CREATE INDEX IF NOT EXISTS idx_memories_org_priority
                    ON memories(tenant_id, status, verified_helps DESC, updated_at DESC);

                CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_single_active
                    ON memories(tenant_id, slot_key)
                    WHERE status = 'ACTIVE' AND cardinality = 'single';

                CREATE TABLE IF NOT EXISTS evidence_sources (
                    tenant_id TEXT NOT NULL,
                    slot_key TEXT NOT NULL,
                    value_hash TEXT NOT NULL,
                    source_hash TEXT NOT NULL,
                    memory_id TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, slot_key, value_hash, source_hash)
                );

                CREATE TABLE IF NOT EXISTS memory_derivations (
                    parent_memory_id TEXT NOT NULL,
                    child_memory_id TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    PRIMARY KEY (parent_memory_id, child_memory_id)
                );
                CREATE INDEX IF NOT EXISTS idx_derivations_child
                    ON memory_derivations(child_memory_id);

                CREATE TABLE IF NOT EXISTS run_reads (
                    run_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    memory_id TEXT NOT NULL,
                    step_idx INTEGER NOT NULL,
                    was_cited INTEGER NOT NULL DEFAULT 0,
                    read_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, memory_id, step_idx)
                );
                CREATE INDEX IF NOT EXISTS idx_run_reads_mem
                    ON run_reads(tenant_id, memory_id);
                CREATE INDEX IF NOT EXISTS idx_run_reads_time
                    ON run_reads(read_at);

                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    user_id TEXT,
                    task_text TEXT NOT NULL,
                    cohort TEXT NOT NULL,
                    final_output TEXT NOT NULL,
                    tool_calls_count INTEGER NOT NULL,
                    tool_error_count INTEGER NOT NULL,
                    in_run_contradictions INTEGER NOT NULL,
                    blocked_unsafe_actions INTEGER NOT NULL,
                    reopened INTEGER NOT NULL DEFAULT 0,
                    critical_path_ms REAL NOT NULL,
                    agent_llm_calls INTEGER NOT NULL,
                    memory_llm_calls INTEGER NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_runs_cohort_time
                    ON runs(cohort, finished_at);

                CREATE TABLE IF NOT EXISTS global_staging (
                    cluster_key TEXT NOT NULL,
                    artifact TEXT NOT NULL,
                    error_sig TEXT NOT NULL,
                    org_hmac TEXT NOT NULL,
                    sanitized_summary TEXT NOT NULL,
                    origin_fingerprint TEXT NOT NULL,
                    source_hash TEXT NOT NULL,
                    trust_tier INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    promoted INTEGER NOT NULL DEFAULT 0,
                    sanitized_value TEXT NOT NULL DEFAULT '',
                    evidence_fingerprint TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY (cluster_key, org_hmac)
                );
                CREATE INDEX IF NOT EXISTS idx_global_staging_cluster_promoted
                    ON global_staging(cluster_key, promoted, created_at);

                CREATE TABLE IF NOT EXISTS tool_action_ledger (
                    tenant_id TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, idempotency_key)
                );
                CREATE INDEX IF NOT EXISTS idx_tool_action_ledger_time
                    ON tool_action_ledger(updated_at);

                CREATE TABLE IF NOT EXISTS report_review_flags (
                    tenant_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    root_memory_id TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    reviewed INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (tenant_id, run_id, root_memory_id)
                );

                CREATE TABLE IF NOT EXISTS containment_audits (
                    audit_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    root_memory_id TEXT NOT NULL,
                    anti_memory_id TEXT NOT NULL,
                    quarantined_ids_json TEXT NOT NULL,
                    contested_ids_json TEXT NOT NULL,
                    affected_run_ids_json TEXT NOT NULL,
                    llm_calls_used INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    notes TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_containment_audits_time
                    ON containment_audits(created_at);

                CREATE TABLE IF NOT EXISTS refutation_ledger (
                    tenant_id TEXT NOT NULL,
                    slot_key TEXT NOT NULL,
                    refuted_value_hash TEXT NOT NULL,
                    refuted_value TEXT NOT NULL,
                    target_norm TEXT NOT NULL,
                    root_memory_id TEXT NOT NULL,
                    anti_memory_id TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, slot_key, refuted_value_hash)
                );
                CREATE INDEX IF NOT EXISTS idx_refutation_tenant_target
                    ON refutation_ledger(tenant_id, target_norm);

                CREATE TABLE IF NOT EXISTS llm_cost_events (
                    event_id TEXT PRIMARY KEY,
                    run_id TEXT,
                    operation TEXT NOT NULL,
                    worker_pid INTEGER NOT NULL,
                    attempt_num INTEGER NOT NULL,
                    tokens_in INTEGER NOT NULL,
                    tokens_out INTEGER NOT NULL,
                    succeeded INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_cost_events_run
                    ON llm_cost_events(run_id, operation);
                CREATE INDEX IF NOT EXISTS idx_cost_events_time
                    ON llm_cost_events(created_at);

                CREATE TABLE IF NOT EXISTS daily_cohort_rollups (
                    day_utc TEXT NOT NULL,
                    cohort TEXT NOT NULL,
                    runs_count INTEGER NOT NULL DEFAULT 0,
                    tool_calls_sum INTEGER NOT NULL DEFAULT 0,
                    tool_errors_sum INTEGER NOT NULL DEFAULT 0,
                    contradiction_runs INTEGER NOT NULL DEFAULT 0,
                    blocked_unsafe_sum INTEGER NOT NULL DEFAULT 0,
                    reopened_sum INTEGER NOT NULL DEFAULT 0,
                    agent_llm_sum INTEGER NOT NULL DEFAULT 0,
                    memory_llm_sum INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (day_utc, cohort)
                );

                CREATE TABLE IF NOT EXISTS daily_cost_rollups (
                    day_utc TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    failed_attempts INTEGER NOT NULL DEFAULT 0,
                    tokens_in INTEGER NOT NULL DEFAULT 0,
                    tokens_out INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (day_utc, operation)
                );

                CREATE TABLE IF NOT EXISTS promoted_cluster_counts (
                    cluster_key TEXT PRIMARY KEY,
                    org_count INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS store_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    memory_id UNINDEXED,
                    tenant_id UNINDEXED,
                    statement,
                    subject,
                    predicate
                );
                """
            )
            existing_cols = {
                str(r["name"])
                for r in conn.execute("PRAGMA table_info(global_staging)").fetchall()
            }
            if "origin_fingerprint" not in existing_cols:
                conn.execute(
                    "ALTER TABLE global_staging ADD COLUMN origin_fingerprint TEXT NOT NULL DEFAULT ''"
                )
            if "source_hash" not in existing_cols:
                conn.execute(
                    "ALTER TABLE global_staging ADD COLUMN source_hash TEXT NOT NULL DEFAULT ''"
                )
            if "trust_tier" not in existing_cols:
                conn.execute(
                    "ALTER TABLE global_staging ADD COLUMN trust_tier INTEGER NOT NULL DEFAULT 0"
                )
            for col in ("sanitized_value", "evidence_fingerprint"):
                if col not in existing_cols:
                    # Pre-existing DBs only; tolerate a concurrent worker winning the ALTER race
                    try:
                        conn.execute(
                            f"ALTER TABLE global_staging ADD COLUMN {col} TEXT NOT NULL DEFAULT ''"
                        )
                    except sqlite3.OperationalError as exc:
                        if "duplicate column name" not in str(exc).lower():
                            raise

        conn = self._get_conn()
        for attempt in range(8):
            try:
                _apply(conn)
                return
            except sqlite3.OperationalError as exc:
                if ("locked" in str(exc).lower() or "busy" in str(exc).lower()) and attempt < 7:
                    time.sleep(0.015 * (attempt + 1))
                    continue
                raise

    def upsert_tenant(self, policy: TenantPolicy) -> None:
        h = org_hmac(self._hmac_secret, policy.org_id)

        def _tx(conn: sqlite3.Connection) -> None:
            conn.execute(
                """
                INSERT INTO tenants(org_id, authorized_domains_json, org_aliases_json, hmac_id)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(org_id) DO UPDATE SET
                    authorized_domains_json = excluded.authorized_domains_json,
                    org_aliases_json = excluded.org_aliases_json,
                    hmac_id = excluded.hmac_id
                """,
                (
                    policy.org_id,
                    json.dumps(policy.authorized_domains),
                    json.dumps(policy.org_aliases),
                    h,
                ),
            )

        self._run_write_tx(_tx)

    def get_tenant(self, org_id: str) -> TenantPolicy:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT org_id, authorized_domains_json, org_aliases_json FROM tenants WHERE org_id = ?",
            (org_id,),
        ).fetchone()
        if row is None:
            return TenantPolicy(org_id=org_id, authorized_domains=[], org_aliases=[org_id])
        return TenantPolicy(
            org_id=row["org_id"],
            authorized_domains=json.loads(row["authorized_domains_json"]),
            org_aliases=json.loads(row["org_aliases_json"]),
        )

    def hydrate_claim_tuple(self, r: tuple[Any, ...]) -> MemoryClaim:
        der_raw = r[19]
        meta_raw = r[32]
        exp_raw = r[31]
        return MemoryClaim.model_construct(
            memory_id=r[0],
            tenant_id=r[1],
            scope=_SCOPE_MAP[r[2]],
            user_id=r[3],
            task_id=r[4],
            claim_type=_CLAIM_TYPE_MAP[r[5]],
            slot_key=r[6],
            subject=r[7],
            predicate=r[8],
            value=r[9],
            value_hash=r[10],
            statement=r[11],
            cardinality=_CARDINALITY_MAP[r[12]],
            volatility=_VOLATILITY_MAP[r[13]],
            risk_class=_RISK_MAP[r[14]],
            trust_tier=_TRUST_MAP[int(r[15])],
            status=_STATUS_MAP[r[16]],
            base_confidence=float(r[17]),
            evidence_count=int(r[18]),
            derived_from_ids=[] if der_raw == "[]" else json.loads(der_raw),
            created_in_run_id=r[20],
            source_type=_SOURCE_TYPE_MAP[r[21]],
            source_id=r[22],
            source_domain=r[23],
            created_at=datetime.fromisoformat(r[24]),
            updated_at=datetime.fromisoformat(r[25]),
            last_accessed_at=datetime.fromisoformat(r[26]),
            access_count=int(r[27]),
            verified_helps=int(r[28]),
            contradiction_hits=int(r[29]),
            version=int(r[30]),
            expires_at=datetime.fromisoformat(exp_raw) if exp_raw else None,
            metadata={} if meta_raw == "{}" else json.loads(meta_raw),
        )

    def _row_to_claim(self, row: sqlite3.Row) -> MemoryClaim:
        der_raw = row["derived_from_json"]
        meta_raw = row["metadata_json"]
        exp_raw = row["expires_at"]
        return MemoryClaim.model_construct(
            memory_id=row["memory_id"],
            tenant_id=row["tenant_id"],
            scope=_SCOPE_MAP[row["scope"]],
            user_id=row["user_id"],
            task_id=row["task_id"],
            claim_type=_CLAIM_TYPE_MAP[row["claim_type"]],
            slot_key=row["slot_key"],
            subject=row["subject"],
            predicate=row["predicate"],
            value=row["value"],
            value_hash=row["value_hash"],
            statement=row["statement"],
            cardinality=_CARDINALITY_MAP[row["cardinality"]],
            volatility=_VOLATILITY_MAP[row["volatility"]],
            risk_class=_RISK_MAP[row["risk_class"]],
            trust_tier=_TRUST_MAP[int(row["trust_tier"])],
            status=_STATUS_MAP[row["status"]],
            base_confidence=float(row["base_confidence"]),
            evidence_count=int(row["evidence_count"]),
            derived_from_ids=[] if der_raw == "[]" else json.loads(der_raw),
            created_in_run_id=row["created_in_run_id"],
            source_type=_SOURCE_TYPE_MAP[row["source_type"]],
            source_id=row["source_id"],
            source_domain=row["source_domain"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            last_accessed_at=datetime.fromisoformat(row["last_accessed_at"]),
            access_count=int(row["access_count"]),
            verified_helps=int(row["verified_helps"]),
            contradiction_hits=int(row["contradiction_hits"]),
            version=int(row["version"]),
            expires_at=datetime.fromisoformat(exp_raw) if exp_raw else None,
            metadata={} if meta_raw == "{}" else json.loads(meta_raw),
        )

    def _insert_claim_row(self, conn: sqlite3.Connection, claim: MemoryClaim) -> None:
        conn.execute(
            """
            INSERT INTO memories (
                memory_id, tenant_id, scope, user_id, task_id, claim_type, slot_key,
                subject, predicate, value, value_hash, statement, cardinality, volatility,
                risk_class, trust_tier, status, base_confidence, evidence_count,
                derived_from_json, created_in_run_id, source_type, source_id, source_domain,
                created_at, updated_at, last_accessed_at, access_count, verified_helps,
                contradiction_hits, version, expires_at, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                claim.memory_id,
                claim.tenant_id,
                claim.scope.value,
                claim.user_id,
                claim.task_id,
                claim.claim_type.value,
                claim.slot_key,
                claim.subject,
                claim.predicate,
                claim.value,
                claim.value_hash,
                claim.statement,
                claim.cardinality.value,
                claim.volatility.value,
                claim.risk_class.value,
                int(claim.trust_tier),
                claim.status.value,
                claim.base_confidence,
                claim.evidence_count,
                json.dumps(claim.derived_from_ids),
                claim.created_in_run_id,
                claim.source_type.value,
                claim.source_id,
                claim.source_domain,
                self._utc_iso(claim.created_at),
                self._utc_iso(claim.updated_at),
                self._utc_iso(claim.last_accessed_at),
                claim.access_count,
                claim.verified_helps,
                claim.contradiction_hits,
                claim.version,
                self._utc_iso(claim.expires_at) if claim.expires_at else None,
                json.dumps(claim.metadata),
            ),
        )
        if claim.scope == Scope.GLOBAL:
            conn.execute(
                "INSERT INTO memories_fts(memory_id, tenant_id, statement, subject, predicate) VALUES (?, ?, ?, ?, ?)",
                (claim.memory_id, claim.tenant_id, claim.statement, claim.subject, claim.predicate),
            )

    def _update_claim_row(self, conn: sqlite3.Connection, claim: MemoryClaim) -> None:
        prev = conn.execute(
            "SELECT status FROM memories WHERE memory_id = ? AND version = ?",
            (claim.memory_id, claim.version),
        ).fetchone()
        if prev is None:
            raise RuntimeError(f"CAS version mismatch updating {claim.memory_id}")
        assert_valid_status_transition(_STATUS_MAP[prev["status"]], claim.status)
        res = conn.execute(
            """
            UPDATE memories SET
                status = ?, trust_tier = ?, base_confidence = ?, evidence_count = ?,
                statement = ?, source_type = ?, source_id = ?, source_domain = ?,
                updated_at = ?, last_accessed_at = ?, access_count = ?, verified_helps = ?,
                contradiction_hits = ?, version = ?, expires_at = ?, metadata_json = ?
            WHERE memory_id = ? AND version = ?
            """,
            (
                claim.status.value,
                int(claim.trust_tier),
                claim.base_confidence,
                claim.evidence_count,
                claim.statement,
                claim.source_type.value,
                claim.source_id,
                claim.source_domain,
                self._utc_iso(claim.updated_at),
                self._utc_iso(claim.last_accessed_at),
                claim.access_count,
                claim.verified_helps,
                claim.contradiction_hits,
                claim.version + 1,
                self._utc_iso(claim.expires_at) if claim.expires_at else None,
                json.dumps(claim.metadata),
                claim.memory_id,
                claim.version,
            ),
        )
        if res.rowcount != 1:
            raise RuntimeError(f"CAS version mismatch updating {claim.memory_id}")
        claim.version += 1

    def get_claim(
        self,
        memory_id: str,
        tenant_id: str | None = None,
        *,
        include_global: bool = True,
    ) -> MemoryClaim | None:
        """
        Resource-level read. With tenant_id, returns the claim only if it belongs to that tenant
        (or is a global advisory when include_global). Every request-path caller passes tenant_id;
        the unscoped form is reserved for operator tooling and tests.
        """
        conn = self._get_conn()
        if tenant_id is None:
            row = conn.execute("SELECT * FROM memories WHERE memory_id = ?", (memory_id,)).fetchone()
        else:
            tenants = (tenant_id, GLOBAL_TENANT_ID) if include_global else (tenant_id, tenant_id)
            row = conn.execute(
                "SELECT * FROM memories WHERE memory_id = ? AND tenant_id IN (?, ?)",
                (memory_id, *tenants),
            ).fetchone()
        return self._row_to_claim(row) if row else None

    def get_claims_by_slot(self, tenant_id: str, slot_key: str) -> list[MemoryClaim]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM memories WHERE tenant_id = ? AND slot_key = ? ORDER BY updated_at DESC",
            (tenant_id, slot_key),
        ).fetchall()
        return [self._row_to_claim(r) for r in rows]

    def upsert_claim(
        self,
        tenant_id: str,
        candidate: CandidateClaim,
        tenant_policy: TenantPolicy,
        now: datetime | None = None,
        conflict_adjudicator: Callable[[MemoryClaim, CandidateClaim], str] | None = None,
        allowed_parent_ids: set[str] | None = None,
        authority_registry: SourceAuthorityRegistry | None = None,
    ) -> tuple[MemoryClaim | None, WriteOutcome]:
        """
        Concurrency-safe atomic write/corroboration/revision transaction.
        """
        candidate = normalize_candidate_authority(
            candidate, registry=authority_registry, expected_tenant_id=tenant_id
        )
        if candidate.scope == Scope.GLOBAL or tenant_id == GLOBAL_TENANT_ID:
            return None, WriteOutcome.REJECTED_GLOBAL_BYPASS
        ts = now or candidate.provenance.observed_at or utcnow()
        effective_tenant = tenant_id
        if allowed_parent_ids is not None and candidate.derived_from_ids:
            filtered_parents = [
                pid for pid in dict.fromkeys(candidate.derived_from_ids) if pid in allowed_parent_ids
            ]
            if filtered_parents != candidate.derived_from_ids:
                candidate = candidate.model_copy(update={"derived_from_ids": filtered_parents})
        decision, target_status, capped_conf, write_reason, expires_at = evaluate_candidate_write(
            candidate, tenant_policy, registry=authority_registry
        )
        if decision == "EPHEMERAL_TASK":
            return None, WriteOutcome.EPHEMERAL_TASK_ONLY
        if decision == "REJECT_INJECTION":
            return None, WriteOutcome.REJECTED_INJECTION
        if decision == "REJECT_LOW_VALUE":
            return None, WriteOutcome.REJECTED_LOW_VALUE

        slot_key = candidate.slot_key or canonicalize_slot(
            candidate.scope,
            effective_tenant,
            candidate.subject,
            candidate.predicate,
            user_id=candidate.user_id,
            task_id=candidate.task_id,
        )
        v_hash = value_digest(candidate.value)
        s_hash = hash_source_id(candidate.provenance.source_id)

        meta = dict(candidate.metadata)
        meta["write_reason"] = write_reason
        meta["last_confirmed_at"] = self._utc_iso(candidate.provenance.observed_at or ts)
        if candidate.is_public_technical_artifact:
            meta["is_public_technical_artifact"] = True
        if candidate.artifact_name:
            meta["artifact_name"] = candidate.artifact_name
        if candidate.error_signature:
            meta["error_signature"] = candidate.error_signature
        if target_status == MemoryStatus.CANDIDATE_UNVERIFIED:
            meta["requires_verification"] = True
            meta.setdefault(
                "verification_hint",
                "Verify via internal CRM/API or human review before executing outbound actions.",
            )

        def _tx(conn: sqlite3.Connection) -> tuple[MemoryClaim | None, WriteOutcome]:
            # 1) Idempotent evidence deduplication check
            dup = conn.execute(
                """
                SELECT memory_id FROM evidence_sources
                WHERE tenant_id = ? AND slot_key = ? AND value_hash = ? AND source_hash = ?
                """,
                (effective_tenant, slot_key, v_hash, s_hash),
            ).fetchone()
            if dup is not None:
                existing_row = conn.execute(
                    "SELECT * FROM memories WHERE memory_id = ?", (dup["memory_id"],)
                ).fetchone()
                if existing_row is not None:
                    return self._row_to_claim(existing_row), WriteOutcome.DEDUPLICATED_SOURCE

            # Respect durable refutation ledger (survives both 90d tombstone purge & >100 negative-constraint compaction)
            ref_row = conn.execute(
                """
                SELECT * FROM refutation_ledger
                WHERE tenant_id = ? AND slot_key = ? AND refuted_value_hash = ?
                LIMIT 1
                """,
                (effective_tenant, slot_key, v_hash),
            ).fetchone()
            if ref_row is not None:
                if candidate.trust_tier < TrustTier.HUMAN_VERIFIED:
                    anti_row = conn.execute(
                        "SELECT * FROM memories WHERE memory_id = ?",
                        (ref_row["anti_memory_id"],),
                    ).fetchone()
                    if anti_row is not None:
                        return self._row_to_claim(anti_row), WriteOutcome.DEDUPLICATED_SOURCE
                    return self._durable_refutation_claim(ref_row), WriteOutcome.DEDUPLICATED_SOURCE
                conn.execute(
                    "DELETE FROM refutation_ledger WHERE tenant_id = ? AND slot_key = ? AND refuted_value_hash = ?",
                    (effective_tenant, slot_key, v_hash),
                )

            existing_rows = conn.execute(
                """
                SELECT * FROM memories
                WHERE tenant_id = ? AND slot_key IN (?, ?)
                  AND status IN ('ACTIVE', 'CONTESTED', 'CANDIDATE_UNVERIFIED', 'REFUTED')
                ORDER BY updated_at DESC
                """,
                (effective_tenant, slot_key, f"{slot_key}|anti"),
            ).fetchall()
            existing_claims = [self._row_to_claim(r) for r in existing_rows]

            # Respect active negative constraints / human refutations on the same value
            # (survives even after REFUTED tombstones are purged at 90d!)
            not_v_hash = value_digest(f"NOT:{candidate.value}")
            for ex in existing_claims:
                is_refuted_match = ex.status == MemoryStatus.REFUTED and ex.value_hash == v_hash
                is_anti_match = (
                    ex.claim_type == ClaimType.NEGATIVE_CONSTRAINT
                    and ex.status == MemoryStatus.ACTIVE
                    and (
                        ex.value_hash == not_v_hash
                        or ex.metadata.get("refuted_value_hash") == v_hash
                    )
                )
                if is_refuted_match or is_anti_match:
                    if candidate.trust_tier < TrustTier.HUMAN_VERIFIED:
                        return ex, WriteOutcome.DEDUPLICATED_SOURCE
                    if is_anti_match:
                        ex.status = MemoryStatus.ARCHIVED
                        ex.metadata["reinstated_by_human_run"] = candidate.provenance.run_id
                        ex.updated_at = ts
                        self._update_claim_row(conn, ex)

            # Filter slot claims to non-anti claims on the exact slot_key
            slot_claims = [c for c in existing_claims if c.slot_key == slot_key]

            # Check same value match (corroboration or promotion of CANDIDATE_UNVERIFIED)
            same_value_claim = next(
                (
                    c
                    for c in slot_claims
                    if c.value_hash == v_hash
                    and c.status in (MemoryStatus.ACTIVE, MemoryStatus.CONTESTED, MemoryStatus.CANDIDATE_UNVERIFIED)
                ),
                None,
            )

            if same_value_claim is not None:
                # Anti-trust-laundering: untrusted external sources can NEVER promote CANDIDATE_UNVERIFIED,
                # and HIGH_ACTION_ROUTING candidates require >= INTERNAL_API_VERIFIED to promote.
                cannot_promote_candidate = (
                    target_status == MemoryStatus.CANDIDATE_UNVERIFIED
                    or candidate.trust_tier == TrustTier.UNTRUSTED_EXTERNAL
                    or (
                        (
                            same_value_claim.risk_class == RiskClass.HIGH_ACTION_ROUTING
                            or candidate.risk_class == RiskClass.HIGH_ACTION_ROUTING
                        )
                        and candidate.trust_tier < TrustTier.INTERNAL_API_VERIFIED
                    )
                )
                if (
                    same_value_claim.status == MemoryStatus.CANDIDATE_UNVERIFIED
                    and cannot_promote_candidate
                ):
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO evidence_sources
                        (tenant_id, slot_key, value_hash, source_hash, memory_id, observed_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (effective_tenant, slot_key, v_hash, s_hash, same_value_claim.memory_id, self._utc_iso(ts)),
                    )
                    same_value_claim.updated_at = ts
                    same_value_claim.metadata["last_confirmed_at"] = self._utc_iso(ts)
                    same_value_claim.evidence_count += 1
                    # Keep capped at unverified level
                    same_value_claim.base_confidence = min(same_value_claim.base_confidence, 0.22)
                    self._update_claim_row(conn, same_value_claim)
                    return same_value_claim, WriteOutcome.CORROBORATED

                # Check if same_value_claim is CONTESTED or CANDIDATE_UNVERIFIED against active/contested rivals
                rival_contested = [
                    other
                    for other in slot_claims
                    if other.memory_id != same_value_claim.memory_id
                    and other.status in (MemoryStatus.ACTIVE, MemoryStatus.CONTESTED)
                    and other.claim_type != ClaimType.NEGATIVE_CONSTRAINT
                ]
                can_resolve_contested = (
                    same_value_claim.status == MemoryStatus.CONTESTED
                    and target_status == MemoryStatus.ACTIVE
                    and candidate.trust_tier >= TrustTier.TOOL_OBSERVED
                    and all(
                        effective_confidence(other, ts) + 0.15 <= max(capped_conf, effective_confidence(same_value_claim, ts))
                        for other in rival_contested
                    )
                )

                # Guard against trust-hierarchy inversion when promoting CANDIDATE_UNVERIFIED over existing rivals
                cand_promote_allowed = False
                cand_contest_rivals = False
                if same_value_claim.status == MemoryStatus.CANDIDATE_UNVERIFIED and target_status == MemoryStatus.ACTIVE:
                    if (
                        candidate.cardinality != Cardinality.SINGLE
                        or not rival_contested
                        or candidate.trust_tier >= TrustTier.INTERNAL_API_VERIFIED
                    ):
                        cand_promote_allowed = True
                    else:
                        strongest_rival = max(
                            rival_contested,
                            key=lambda c: (int(c.trust_tier), effective_confidence(c, ts)),
                        )
                        eff_rival = effective_confidence(strongest_rival, ts)
                        if (
                            strongest_rival.status == MemoryStatus.ACTIVE
                            and (
                                strongest_rival.trust_tier >= TrustTier.INTERNAL_API_VERIFIED
                                or (
                                    strongest_rival.trust_tier >= TrustTier.TOOL_OBSERVED
                                    and not candidate.error_signature
                                )
                            )
                            and eff_rival >= 0.70
                            and candidate.trust_tier <= TrustTier.AGENT_INFERRED
                        ):
                            conn.execute(
                                """
                                INSERT OR IGNORE INTO evidence_sources
                                (tenant_id, slot_key, value_hash, source_hash, memory_id, observed_at)
                                VALUES (?, ?, ?, ?, ?, ?)
                                """,
                                (effective_tenant, slot_key, v_hash, s_hash, same_value_claim.memory_id, self._utc_iso(ts)),
                            )
                            same_value_claim.updated_at = ts
                            same_value_claim.metadata["last_confirmed_at"] = self._utc_iso(ts)
                            same_value_claim.evidence_count += 1
                            same_value_claim.base_confidence = min(same_value_claim.base_confidence, 0.25)
                            same_value_claim.metadata["conflicts_with_verified_id"] = strongest_rival.memory_id
                            self._update_claim_row(conn, same_value_claim)
                            return same_value_claim, WriteOutcome.CORROBORATED
                        if (
                            candidate.trust_tier > strongest_rival.trust_tier
                            and candidate.trust_tier >= TrustTier.TOOL_OBSERVED
                            and capped_conf >= eff_rival + 0.15
                        ):
                            cand_promote_allowed = True
                        else:
                            cand_contest_rivals = True

                # Verified promotion of a previously staged candidate (or corroboration/resolution)
                if target_status == MemoryStatus.ACTIVE and (
                    cand_promote_allowed
                    or (same_value_claim.status != MemoryStatus.CANDIDATE_UNVERIFIED and candidate.trust_tier >= TrustTier.INTERNAL_API_VERIFIED)
                    or can_resolve_contested
                ):
                    # Supersede any conflicting ACTIVE/CONTESTED rows on single-cardinality slot
                    if candidate.cardinality == Cardinality.SINGLE:
                        for other in rival_contested:
                            other.status = MemoryStatus.SUPERSEDED
                            other.metadata["superseded_by"] = same_value_claim.memory_id
                            other.updated_at = ts
                            self._update_claim_row(conn, other)

                    same_value_claim.status = MemoryStatus.ACTIVE
                    same_value_claim.metadata.pop("requires_verification", None)
                    same_value_claim.metadata.pop("working_hypothesis", None)
                    same_value_claim.metadata.pop("conflicts_with_verified_id", None)
                elif cand_contest_rivals:
                    strongest_rival = max(
                        rival_contested,
                        key=lambda c: (int(c.trust_tier), effective_confidence(c, ts)),
                    )
                    eff_rival = effective_confidence(strongest_rival, ts)
                    for other in rival_contested:
                        if other.status == MemoryStatus.ACTIVE:
                            other.status = MemoryStatus.CONTESTED
                            other.metadata.pop("working_hypothesis", None)
                            other.updated_at = ts
                            self._update_claim_row(conn, other)
                    same_value_claim.status = MemoryStatus.CONTESTED
                    same_value_claim.expires_at = None
                    if capped_conf > eff_rival:
                        same_value_claim.metadata["working_hypothesis"] = True
                        same_value_claim.metadata["requires_verification"] = True

                # Preserve higher-trust authoritative provenance when corroborated by lower-trust sources
                if int(candidate.trust_tier) >= int(same_value_claim.trust_tier):
                    same_value_claim.trust_tier = candidate.trust_tier
                    same_value_claim.source_type = candidate.provenance.source_type
                    same_value_claim.source_id = candidate.provenance.source_id
                    same_value_claim.source_domain = candidate.provenance.source_domain
                same_value_claim.base_confidence = max(same_value_claim.base_confidence, capped_conf)
                same_value_claim.evidence_count += 1
                same_value_claim.updated_at = ts
                same_value_claim.metadata["last_confirmed_at"] = self._utc_iso(ts)
                same_value_claim.expires_at = None if same_value_claim.status == MemoryStatus.ACTIVE else same_value_claim.expires_at
                self._update_claim_row(conn, same_value_claim)
                conn.execute(
                    """
                    INSERT OR IGNORE INTO evidence_sources
                    (tenant_id, slot_key, value_hash, source_hash, memory_id, observed_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (effective_tenant, slot_key, v_hash, s_hash, same_value_claim.memory_id, self._utc_iso(ts)),
                )
                return same_value_claim, (
                    WriteOutcome.CONTESTED if cand_contest_rivals else WriteOutcome.CORROBORATED
                )

            # Different value on SINGLE-cardinality slot: check active/contested conflicts
            active_conflicts = [
                c
                for c in slot_claims
                if c.status in (MemoryStatus.ACTIVE, MemoryStatus.CONTESTED)
                and c.value_hash != v_hash
                and c.claim_type != ClaimType.NEGATIVE_CONSTRAINT
            ]

            final_status = target_status
            outcome = (
                WriteOutcome.INSERTED_CANDIDATE
                if target_status == MemoryStatus.CANDIDATE_UNVERIFIED
                else WriteOutcome.INSERTED_ACTIVE
            )

            if (
                candidate.cardinality == Cardinality.SINGLE
                and candidate.claim_type != ClaimType.NEGATIVE_CONSTRAINT
                and active_conflicts
            ):
                if target_status == MemoryStatus.CANDIDATE_UNVERIFIED:
                    # Do NOT disturb existing ACTIVE verified belief; stage candidate separately
                    final_status = MemoryStatus.CANDIDATE_UNVERIFIED
                    outcome = WriteOutcome.INSERTED_CANDIDATE
                elif candidate.trust_tier >= TrustTier.INTERNAL_API_VERIFIED:
                    # Authoritative verification supersedes all conflicting active/contested beliefs
                    new_id_val = new_id("mem")
                    for conflict in active_conflicts:
                        conflict.status = MemoryStatus.SUPERSEDED
                        conflict.metadata["superseded_by"] = new_id_val
                        conflict.updated_at = ts
                        self._update_claim_row(conn, conflict)
                    final_status = MemoryStatus.ACTIVE
                    outcome = WriteOutcome.SUPERSEDED
                    claim = self._build_claim(
                        memory_id=new_id_val,
                        effective_tenant=effective_tenant,
                        slot_key=slot_key,
                        v_hash=v_hash,
                        candidate=candidate,
                        status=final_status,
                        base_conf=capped_conf,
                        ts=ts,
                        expires_at=expires_at,
                        meta=meta,
                    )
                    self._insert_claim_and_links(conn, claim, s_hash, ts)
                    return claim, outcome
                else:
                    # Compare staleness/confidence gap or use adjudicator
                    primary = max(
                        active_conflicts,
                        key=lambda c: (int(c.trust_tier), effective_confidence(c, ts)),
                    )
                    eff_existing = effective_confidence(primary, ts)
                    temp_incoming = self._build_claim(
                        memory_id="tmp",
                        effective_tenant=effective_tenant,
                        slot_key=slot_key,
                        v_hash=v_hash,
                        candidate=candidate,
                        status=MemoryStatus.ACTIVE,
                        base_conf=capped_conf,
                        ts=ts,
                        expires_at=expires_at,
                        meta=meta,
                    )
                    eff_incoming = effective_confidence(temp_incoming, ts)

                    # Do NOT let an unverified low-trust inference demote a fresh verified/observed truth
                    if (
                        primary.status == MemoryStatus.ACTIVE
                        and (
                            primary.trust_tier >= TrustTier.INTERNAL_API_VERIFIED
                            or (
                                primary.trust_tier >= TrustTier.TOOL_OBSERVED
                                and not candidate.error_signature
                            )
                        )
                        and eff_existing >= 0.70
                        and candidate.trust_tier <= TrustTier.AGENT_INFERRED
                    ):
                        meta["requires_verification"] = True
                        meta["conflicts_with_verified_id"] = primary.memory_id
                        claim = self._build_claim(
                            memory_id=new_id("mem"),
                            effective_tenant=effective_tenant,
                            slot_key=slot_key,
                            v_hash=v_hash,
                            candidate=candidate,
                            status=MemoryStatus.CANDIDATE_UNVERIFIED,
                            base_conf=min(capped_conf, 0.25),
                            ts=ts,
                            expires_at=ts + timedelta(days=14),
                            meta=meta,
                        )
                        self._insert_claim_and_links(conn, claim, s_hash, ts)
                        return claim, WriteOutcome.INSERTED_CANDIDATE

                    resolution = "contest_both"
                    if (
                        candidate.trust_tier > primary.trust_tier
                        and candidate.trust_tier >= TrustTier.TOOL_OBSERVED
                        and eff_incoming >= eff_existing + 0.15
                    ):
                        resolution = "supersede_incoming"
                    elif (
                        abs(eff_incoming - eff_existing) < 0.15
                        and conflict_adjudicator is not None
                    ):
                        resolution = conflict_adjudicator(primary, candidate)

                    if resolution == "supersede_incoming":
                        new_id_val = new_id("mem")
                        for conflict in active_conflicts:
                            conflict.status = MemoryStatus.SUPERSEDED
                            conflict.metadata["superseded_by"] = new_id_val
                            conflict.updated_at = ts
                            self._update_claim_row(conn, conflict)
                        final_status = MemoryStatus.ACTIVE
                        outcome = WriteOutcome.SUPERSEDED
                        claim = self._build_claim(
                            memory_id=new_id_val,
                            effective_tenant=effective_tenant,
                            slot_key=slot_key,
                            v_hash=v_hash,
                            candidate=candidate,
                            status=final_status,
                            base_conf=capped_conf,
                            ts=ts,
                            expires_at=expires_at,
                            meta=meta,
                        )
                        self._insert_claim_and_links(conn, claim, s_hash, ts)
                        return claim, outcome
                    elif resolution == "keep_existing":
                        meta["superseded_by"] = primary.memory_id
                        meta["adjudication"] = "keep_existing"
                        claim = self._build_claim(
                            memory_id=new_id("mem"),
                            effective_tenant=effective_tenant,
                            slot_key=slot_key,
                            v_hash=v_hash,
                            candidate=candidate,
                            status=MemoryStatus.SUPERSEDED,
                            base_conf=capped_conf,
                            ts=ts,
                            expires_at=expires_at,
                            meta=meta,
                        )
                        self._insert_claim_and_links(conn, claim, s_hash, ts)
                        return claim, WriteOutcome.SUPERSEDED
                    else:
                        # Transition conflicting ACTIVE rows to CONTESTED and mark incoming CONTESTED
                        for conflict in active_conflicts:
                            if conflict.status == MemoryStatus.ACTIVE:
                                conflict.status = MemoryStatus.CONTESTED
                                conflict.metadata.pop("working_hypothesis", None)
                                conflict.updated_at = ts
                                self._update_claim_row(conn, conflict)
                        final_status = MemoryStatus.CONTESTED
                        if eff_incoming > eff_existing:
                            meta["working_hypothesis"] = True
                            meta["requires_verification"] = True
                            meta["verification_hint"] = (
                                "Conflicting fresh observation outweighs decayed belief; verify with internal tool."
                            )
                        outcome = WriteOutcome.CONTESTED

            claim = self._build_claim(
                memory_id=new_id("mem"),
                effective_tenant=effective_tenant,
                slot_key=slot_key,
                v_hash=v_hash,
                candidate=candidate,
                status=final_status,
                base_conf=capped_conf,
                ts=ts,
                expires_at=expires_at,
                meta=meta,
            )
            self._insert_claim_and_links(conn, claim, s_hash, ts)
            return claim, outcome

        return self._run_write_tx(_tx)

    def _build_claim(
        self,
        *,
        memory_id: str,
        effective_tenant: str,
        slot_key: str,
        v_hash: str,
        candidate: CandidateClaim,
        status: MemoryStatus,
        base_conf: float,
        ts: datetime,
        expires_at: datetime | None,
        meta: dict[str, Any],
    ) -> MemoryClaim:
        return MemoryClaim(
            memory_id=memory_id,
            tenant_id=effective_tenant,
            scope=candidate.scope,
            user_id=candidate.user_id,
            task_id=candidate.task_id,
            claim_type=candidate.claim_type,
            slot_key=slot_key,
            subject=candidate.subject,
            predicate=candidate.predicate,
            value=candidate.value,
            value_hash=v_hash,
            statement=candidate.statement,
            cardinality=candidate.cardinality,
            volatility=candidate.volatility,
            risk_class=candidate.risk_class,
            trust_tier=candidate.trust_tier,
            status=status,
            base_confidence=base_conf,
            evidence_count=1,
            derived_from_ids=list(dict.fromkeys(candidate.derived_from_ids)),
            created_in_run_id=candidate.provenance.run_id,
            source_type=candidate.provenance.source_type,
            source_id=candidate.provenance.source_id,
            source_domain=candidate.provenance.source_domain,
            created_at=ts,
            updated_at=ts,
            last_accessed_at=ts,
            expires_at=expires_at,
            metadata=meta,
        )

    def _insert_claim_and_links(
        self,
        conn: sqlite3.Connection,
        claim: MemoryClaim,
        source_hash: str,
        ts: datetime,
    ) -> None:
        self._insert_claim_row(conn, claim)
        conn.execute(
            """
            INSERT OR IGNORE INTO evidence_sources
            (tenant_id, slot_key, value_hash, source_hash, memory_id, observed_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (claim.tenant_id, claim.slot_key, claim.value_hash, source_hash, claim.memory_id, self._utc_iso(ts)),
        )
        for parent_id in claim.derived_from_ids:
            if parent_id == claim.memory_id:
                continue
            parent_row = conn.execute(
                "SELECT tenant_id FROM memories WHERE memory_id = ?", (parent_id,)
            ).fetchone()
            # Never create cross-tenant derivation links into tenant memory
            if parent_row is not None and parent_row["tenant_id"] in (claim.tenant_id, GLOBAL_TENANT_ID):
                conn.execute(
                    """
                    INSERT OR IGNORE INTO memory_derivations(parent_memory_id, child_memory_id, relation_type)
                    VALUES (?, ?, 'derived_from')
                    """,
                    (parent_id, claim.memory_id),
                )

    @staticmethod
    def _utc_iso(dt: datetime | None = None) -> str:
        cur = dt or utcnow()
        cur = cur.replace(tzinfo=UTC) if cur.tzinfo is None else cur.astimezone(UTC)
        return cur.isoformat()

    def _durable_refutation_claim(self, ref_row: sqlite3.Row) -> MemoryClaim:
        parts = str(ref_row["slot_key"]).split("|")
        subj = parts[-2] if len(parts) >= 3 else "refuted_entity"
        pred = parts[-1] if len(parts) >= 2 else "refuted_attribute"
        created = datetime.fromisoformat(str(ref_row["created_at"]))
        return MemoryClaim(
            memory_id=str(ref_row["anti_memory_id"]),
            tenant_id=str(ref_row["tenant_id"]),
            scope=Scope.ORG,
            claim_type=ClaimType.NEGATIVE_CONSTRAINT,
            slot_key=f"{ref_row['slot_key']}|anti",
            subject=subj,
            predicate=f"not_{pred}",
            value=f"NOT:{ref_row['refuted_value']}",
            value_hash=value_digest(f"NOT:{ref_row['refuted_value']}"),
            statement=(
                f"REFUTED BELIEF (durable ledger): Do NOT use '{ref_row['refuted_value']}' "
                f"({ref_row['reason']})."
            ),
            cardinality=Cardinality.MULTI,
            volatility=Volatility.PERMANENT,
            risk_class=RiskClass.HIGH_ACTION_ROUTING,
            trust_tier=TrustTier.HUMAN_VERIFIED,
            status=MemoryStatus.ACTIVE,
            base_confidence=0.98,
            evidence_count=1,
            derived_from_ids=[str(ref_row["root_memory_id"])],
            created_in_run_id="durable_refutation_ledger",
            source_type=SourceType.HUMAN_FEEDBACK,
            source_id=f"refute:{ref_row['root_memory_id']}",
            created_at=created,
            updated_at=created,
            last_accessed_at=created,
            metadata={
                "refuted_memory_id": str(ref_row["root_memory_id"]),
                "refuted_value_hash": str(ref_row["refuted_value_hash"]),
                "durable_ledger": True,
            },
        )

    def find_matching_refutations(
        self,
        tenant_id: str,
        serialized_targets: str,
        exclude_ids: set[str] | None = None,
    ) -> list[MemoryClaim]:
        if not serialized_targets.strip():
            return []
        lower_targets = " ".join(serialized_targets.lower().split())
        raw_tokens = set(re.findall(r"[a-z0-9_.:/@-]+", lower_targets))
        target_tokens = raw_tokens | {t.rstrip(".") for t in raw_tokens}
        target_emails: set[str] = set()
        target_domains: set[str] = set()
        for tok in list(raw_tokens):
            if "@" in tok:
                clean_email = tok.rstrip(".")
                target_emails.add(clean_email)
                target_domains.add(clean_email.split("@", 1)[1])
            elif "://" in tok:
                host = tok.split("://", 1)[1].split("/", 1)[0].split(":", 1)[0].rstrip(".")
                if host:
                    target_domains.add(host)
            elif "." in tok:
                target_domains.add(tok.rstrip("."))

        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM refutation_ledger WHERE tenant_id = ?",
            (tenant_id,),
        ).fetchall()
        excluded = exclude_ids or set()
        matches: list[MemoryClaim] = []
        for r in rows:
            if str(r["anti_memory_id"]) in excluded:
                continue
            target_norm = " ".join(str(r["target_norm"]).strip().lower().split())
            if not target_norm:
                continue
            matched = target_norm in target_tokens
            if not matched and "@" in target_norm:
                matched = target_norm in target_emails
            elif not matched and "://" in target_norm:
                ref_host = target_norm.split("://", 1)[1].split("/", 1)[0].split(":", 1)[0].rstrip(".")
                matched = bool(ref_host and ref_host in target_domains)
            elif not matched and " " not in target_norm and "." in target_norm:
                matched = target_norm.rstrip(".") in target_domains
            elif not matched and " " in target_norm:
                matched = bool(
                    re.search(
                        rf"(?<![a-z0-9_./@-]){re.escape(target_norm)}(?![a-z0-9_./@-])",
                        lower_targets,
                    )
                )
            if matched:
                matches.append(self._durable_refutation_claim(r))
        return matches

    def query_recall_rows(
        self,
        *,
        org_id: str,
        user_id: str | None,
        query_text: str,
        global_limit: int = 20,
        now: datetime | None = None,
    ) -> tuple[list[tuple[Any, ...]], list[tuple[Any, ...]]]:
        """
        Strict tenant-partitioned raw tuple read for fast zero-hydration scoring.
        Enforces recall-time TTL expiration even prior to background compaction.
        Returns (org_and_user_rows, global_advisory_rows).
        """
        now_iso = self._utc_iso(now)
        conn = self._get_conn()
        cur = conn.cursor()
        cur.row_factory = None
        org_rows = cur.execute(
            f"""
            SELECT {_CLAIM_SELECT} FROM memories
            WHERE tenant_id = ?
              AND status IN ('ACTIVE', 'CONTESTED', 'CANDIDATE_UNVERIFIED')
              AND (scope != 'user' OR user_id = ?)
              AND (expires_at IS NULL OR expires_at > ?)
            """,
            (org_id, user_id, now_iso),
        ).fetchall()

        raw_tokens = list(dict.fromkeys(re.findall(r"[A-Za-z0-9]{2,}", query_text.lower())))
        informative = [t for t in raw_tokens if t not in _GLOBAL_FTS_STOPWORDS]
        tokens = (informative or raw_tokens)[:24]
        global_rows: list[tuple[Any, ...]] = []
        if tokens:
            fts_query = " OR ".join(f'"{t}"' for t in tokens)
            try:
                global_rows = cur.execute(
                    f"""
                    SELECT {_M_CLAIM_SELECT} FROM memories_fts f
                    JOIN memories m ON m.memory_id = f.memory_id
                    WHERE memories_fts MATCH ?
                      AND m.tenant_id = ?
                      AND m.scope = 'global'
                      AND m.status = 'ACTIVE'
                      AND (m.expires_at IS NULL OR m.expires_at > ?)
                    ORDER BY bm25(memories_fts), m.verified_helps DESC
                    LIMIT ?
                    """,
                    (fts_query, GLOBAL_TENANT_ID, now_iso, global_limit),
                ).fetchall()
            except sqlite3.OperationalError:
                global_rows = []

        if not global_rows:
            global_rows = cur.execute(
                f"""
                SELECT {_CLAIM_SELECT} FROM memories
                WHERE tenant_id = ? AND scope = 'global' AND status = 'ACTIVE'
                  AND (expires_at IS NULL OR expires_at > ?)
                ORDER BY verified_helps DESC, updated_at DESC
                LIMIT ?
                """,
                (GLOBAL_TENANT_ID, now_iso, global_limit),
            ).fetchall()

        return org_rows, global_rows

    def query_recall_pool(
        self,
        *,
        org_id: str,
        user_id: str | None,
        query_text: str,
        global_limit: int = 25,
        now: datetime | None = None,
    ) -> tuple[list[MemoryClaim], list[MemoryClaim]]:
        """
        Strict tenant-partitioned read:
        Returns (org_and_user_claims, global_advisories).
        """
        org_rows, global_rows = self.query_recall_rows(
            org_id=org_id,
            user_id=user_id,
            query_text=query_text,
            global_limit=global_limit,
            now=now,
        )
        return (
            [self.hydrate_claim_tuple(r) for r in org_rows],
            [self.hydrate_claim_tuple(r) for r in global_rows],
        )

    def record_run_reads(
        self,
        *,
        run_id: str,
        tenant_id: str,
        memory_ids: list[str],
        step_idx: int,
        read_at: datetime,
        cited_memory_ids: set[str] | None = None,
    ) -> None:
        if not memory_ids:
            return
        cited = cited_memory_ids or set()

        def _tx(conn: sqlite3.Connection) -> None:
            now_iso = self._utc_iso(read_at)
            allowed: list[str] = []
            for chunk in _chunked(memory_ids):
                ph = ",".join("?" for _ in chunk)
                allowed.extend(
                    str(r["memory_id"])
                    for r in conn.execute(
                        f"SELECT memory_id FROM memories WHERE memory_id IN ({ph}) AND tenant_id IN (?, ?)",
                        [*chunk, tenant_id, GLOBAL_TENANT_ID],
                    ).fetchall()
                )
            if not allowed:
                return
            conn.executemany(
                """
                INSERT OR IGNORE INTO run_reads(run_id, tenant_id, memory_id, step_idx, was_cited, read_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [(run_id, tenant_id, mid, step_idx, 1 if mid in cited else 0, now_iso) for mid in allowed],
            )
            for chunk in _chunked(allowed):
                placeholders = ",".join("?" for _ in chunk)
                conn.execute(
                    f"""
                    UPDATE memories
                    SET access_count = access_count + 1, last_accessed_at = ?
                    WHERE memory_id IN ({placeholders}) AND tenant_id IN (?, ?)
                    """,
                    [now_iso, *chunk, tenant_id, GLOBAL_TENANT_ID],
                )

        self._run_write_tx(_tx)

    def get_flagged_reports(self, tenant_id: str) -> list[dict[str, Any]]:
        rows = self._get_conn().execute(
            """
            SELECT run_id, root_memory_id, reason, created_at, reviewed FROM report_review_flags
            WHERE tenant_id = ? ORDER BY created_at, run_id
            """,
            (tenant_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_run_recalled_claims(self, run_id: str, tenant_id: str) -> list[MemoryClaim]:
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT DISTINCT m.* FROM run_reads r
            JOIN memories m ON m.memory_id = r.memory_id
            WHERE r.run_id = ? AND r.tenant_id = ?
            """,
            (run_id, tenant_id),
        ).fetchall()
        return [self._row_to_claim(r) for r in rows]

    def get_run_created_claims(self, run_id: str, tenant_id: str) -> list[MemoryClaim]:
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT * FROM memories
            WHERE tenant_id = ? AND created_in_run_id = ?
              AND claim_type != 'negative_constraint'
            """,
            (tenant_id, run_id),
        ).fetchall()
        return [self._row_to_claim(r) for r in rows]

    def increment_contradiction_hit(
        self,
        memory_id: str,
        now: datetime,
        *,
        tenant_id: str | None = None,
    ) -> MemoryClaim | None:
        """
        Records a rejection. For GLOBAL advisories the count is the number of DISTINCT rejecting
        tenants (keyed by org HMAC), so one tenant rejecting repeatedly cannot reach the
        multi-tenant refutation threshold and delete an advisory for everyone. Tenant-owned
        claims may only be hit by their own tenant.
        """

        def _tx(conn: sqlite3.Connection) -> MemoryClaim | None:
            row = conn.execute("SELECT * FROM memories WHERE memory_id = ?", (memory_id,)).fetchone()
            if row is None:
                return None
            c = self._row_to_claim(row)
            if tenant_id is not None and c.tenant_id not in (tenant_id, GLOBAL_TENANT_ID):
                return None
            if c.tenant_id == GLOBAL_TENANT_ID and tenant_id is not None:
                rejecting = set(c.metadata.get("rejecting_org_hmacs", []))
                rejecting.add(org_hmac(self._hmac_secret, tenant_id))
                c.metadata["rejecting_org_hmacs"] = sorted(rejecting)
                c.contradiction_hits = len(rejecting)
            else:
                c.contradiction_hits += 1
            c.updated_at = now
            self._update_claim_row(conn, c)
            return c

        return self._run_write_tx(_tx)

    def contest_uncertain_roots(
        self,
        *,
        tenant_id: str,
        memory_ids: list[str],
        run_id: str,
        reason: str,
        now: datetime,
    ) -> list[str]:
        """
        Ambiguous attribution: nothing is refuted. Plausible roots become CONTESTED and
        non-actionable (`requires_verification`), and the rejected run is flagged for review.
        Returns the ids that were changed.
        """

        def _tx(conn: sqlite3.Connection) -> list[str]:
            changed: list[str] = []
            for mid in dict.fromkeys(memory_ids):
                row = conn.execute(
                    "SELECT * FROM memories WHERE memory_id = ? AND tenant_id = ?", (mid, tenant_id)
                ).fetchone()
                if row is None:
                    continue
                c = self._row_to_claim(row)
                if c.status not in (MemoryStatus.ACTIVE, MemoryStatus.CANDIDATE_UNVERIFIED, MemoryStatus.CONTESTED):
                    continue
                if c.status == MemoryStatus.ACTIVE:
                    c.status = MemoryStatus.CONTESTED
                c.metadata["requires_verification"] = True
                c.metadata["reverify_reason"] = f"ambiguous_attribution:{run_id}"
                c.updated_at = now
                self._update_claim_row(conn, c)
                changed.append(mid)
            conn.execute(
                """
                INSERT OR IGNORE INTO report_review_flags (tenant_id, run_id, root_memory_id, reason, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (tenant_id, run_id, changed[0] if changed else "", reason, self._utc_iso(now)),
            )
            return changed

        return self._run_write_tx(_tx)

    def get_run_record(self, run_id: str, tenant_id: str | None = None) -> sqlite3.Row | None:
        conn = self._get_conn()
        if tenant_id is None:
            return conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return conn.execute(
            "SELECT * FROM runs WHERE run_id = ? AND tenant_id = ?", (run_id, tenant_id)
        ).fetchone()

    def run_exists_for_other_tenant(self, run_id: str, tenant_id: str) -> bool:
        row = self._get_conn().execute(
            "SELECT 1 FROM runs WHERE run_id = ? AND tenant_id != ? LIMIT 1", (run_id, tenant_id)
        ).fetchone()
        return row is not None

    def save_run_record(
        self,
        *,
        run_id: str,
        tenant_id: str,
        user_id: str | None,
        task_text: str,
        cohort: str,
        final_output: str,
        tool_calls_count: int,
        tool_error_count: int,
        in_run_contradictions: int,
        blocked_unsafe_actions: int,
        reopened: bool,
        critical_path_ms: float,
        agent_llm_calls: int,
        memory_llm_calls: int,
        started_at: datetime,
        finished_at: datetime,
        recalled_ids_helped: list[str] | None = None,
        contradicted_ids: list[str] | None = None,
    ) -> None:
        def _tx(conn: sqlite3.Connection) -> None:
            prior = conn.execute("SELECT tenant_id FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            if prior is not None and str(prior["tenant_id"]) != tenant_id:
                raise PermissionError(f"run_id {run_id} belongs to another tenant")
            conn.execute(
                """
                INSERT OR REPLACE INTO runs (
                    run_id, tenant_id, user_id, task_text, cohort, final_output,
                    tool_calls_count, tool_error_count, in_run_contradictions,
                    blocked_unsafe_actions, reopened, critical_path_ms,
                    agent_llm_calls, memory_llm_calls, started_at, finished_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    tenant_id,
                    user_id,
                    task_text,
                    cohort,
                    final_output,
                    tool_calls_count,
                    tool_error_count,
                    in_run_contradictions,
                    blocked_unsafe_actions,
                    1 if reopened else 0,
                    critical_path_ms,
                    agent_llm_calls,
                    memory_llm_calls,
                    self._utc_iso(started_at),
                    self._utc_iso(finished_at),
                ),
            )
            if recalled_ids_helped:
                for chunk in _chunked(recalled_ids_helped):
                    ph = ",".join("?" for _ in chunk)
                    conn.execute(
                        f"UPDATE memories SET verified_helps = verified_helps + 1 "
                        f"WHERE memory_id IN ({ph}) AND tenant_id = ?",
                        [*chunk, tenant_id],
                    )
            if contradicted_ids:
                for chunk in _chunked(contradicted_ids):
                    ph = ",".join("?" for _ in chunk)
                    conn.execute(
                        f"UPDATE memories SET contradiction_hits = contradiction_hits + 1 "
                        f"WHERE memory_id IN ({ph}) AND tenant_id = ?",
                        [*chunk, tenant_id],
                    )

        self._run_write_tx(_tx)

    def refute_and_contain(
        self,
        *,
        tenant_id: str,
        root_memory_id: str,
        reason: str,
        now: datetime,
        llm_calls_used: int = 0,
    ) -> ContainmentAuditEvent:
        """
        Refutes root_memory_id, creates a durable NEGATIVE_CONSTRAINT anti-memory,
        quarantines causal descendants in memory_derivations (preserving unrelated
        co-occurring memories), checks run-scoped semantic descendants, and emits
        a blast-radius audit event over all affected runs.
        """

        def _tx(conn: sqlite3.Connection) -> ContainmentAuditEvent:
            root_row = conn.execute(
                "SELECT * FROM memories WHERE memory_id = ? AND tenant_id = ?",
                (root_memory_id, tenant_id),
            ).fetchone()
            if root_row is None:
                raise KeyError(f"Root memory {root_memory_id} not found for tenant {tenant_id}")
            root_claim = self._row_to_claim(root_row)
            root_claim.status = MemoryStatus.REFUTED
            root_claim.base_confidence = 0.0
            root_claim.updated_at = now
            root_claim.metadata["refutation_reason"] = reason
            self._update_claim_row(conn, root_claim)

            # Idempotently reuse existing anti-memory if root was already refuted
            existing_anti = conn.execute(
                """
                SELECT memory_id FROM memories
                WHERE tenant_id = ? AND source_id = ? AND claim_type = 'negative_constraint'
                  AND status = 'ACTIVE'
                LIMIT 1
                """,
                (tenant_id, f"refute:{root_claim.memory_id}"),
            ).fetchone()
            if existing_anti is not None:
                anti_id = existing_anti["memory_id"]
            else:
                anti_id = new_id("anti")
                anti_slot = f"{root_claim.slot_key}|anti"
                anti_claim = MemoryClaim(
                    memory_id=anti_id,
                    tenant_id=tenant_id,
                    scope=root_claim.scope,
                    user_id=root_claim.user_id,
                    task_id=None,
                    claim_type=ClaimType.NEGATIVE_CONSTRAINT,
                    slot_key=anti_slot,
                    subject=root_claim.subject,
                    predicate=f"not_{root_claim.predicate}",
                    value=f"NOT:{root_claim.value}",
                    value_hash=value_digest(f"NOT:{root_claim.value}"),
                    statement=(
                        f"REFUTED BELIEF: Do NOT assume '{root_claim.statement}' "
                        f"({reason}). Verify directly via authoritative API."
                    ),
                    cardinality=Cardinality.MULTI,
                    volatility=Volatility.PERMANENT,
                    risk_class=root_claim.risk_class,
                    trust_tier=TrustTier.HUMAN_VERIFIED,
                    status=MemoryStatus.ACTIVE,
                    base_confidence=0.98,
                    evidence_count=1,
                    derived_from_ids=[root_claim.memory_id],
                    created_in_run_id=root_claim.created_in_run_id,
                    source_type=SourceType.HUMAN_FEEDBACK,
                    source_id=f"refute:{root_claim.memory_id}",
                    created_at=now,
                    updated_at=now,
                    last_accessed_at=now,
                    metadata={
                        "refuted_memory_id": root_claim.memory_id,
                        "refuted_value_hash": root_claim.value_hash,
                    },
                )
                self._insert_claim_row(conn, anti_claim)

            conn.execute(
                """
                INSERT OR REPLACE INTO refutation_ledger (
                    tenant_id, slot_key, refuted_value_hash, refuted_value, target_norm,
                    root_memory_id, anti_memory_id, reason, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tenant_id,
                    root_claim.slot_key,
                    root_claim.value_hash,
                    root_claim.value,
                    root_claim.value.strip().lower(),
                    root_claim.memory_id,
                    anti_id,
                    reason,
                    self._utc_iso(now),
                ),
            )

            # (1) derived_from closure: only DECLARED dependence propagates refutation.
            descendants: list[str] = []
            seen: set[str] = {root_memory_id}
            queue: deque[str] = deque([root_memory_id])
            while queue:
                curr = queue.popleft()
                for cr in conn.execute(
                    "SELECT child_memory_id FROM memory_derivations WHERE parent_memory_id = ?",
                    (curr,),
                ).fetchall():
                    cid = cr["child_memory_id"]
                    if cid not in seen:
                        seen.add(cid)
                        descendants.append(cid)
                        queue.append(cid)

            quarantined_ids: list[str] = []
            contested_ids: list[str] = []
            restored_ids: list[str] = []
            tainted: set[str] = {root_memory_id}
            # Process in BFS order so a child sees its parents' verdicts.
            for cid in descendants:
                c_row = conn.execute(
                    "SELECT * FROM memories WHERE memory_id = ? AND tenant_id = ?", (cid, tenant_id)
                ).fetchone()
                if c_row is None:
                    continue
                child = self._row_to_claim(c_row)
                if child.status in (MemoryStatus.REFUTED, MemoryStatus.ARCHIVED, MemoryStatus.SUPERSEDED):
                    continue
                parent_rows = conn.execute(
                    """
                    SELECT m.memory_id, m.status, m.trust_tier FROM memory_derivations d
                    JOIN memories m ON m.memory_id = d.parent_memory_id
                    WHERE d.child_memory_id = ?
                    """,
                    (cid,),
                ).fetchall()
                if not any(pr["memory_id"] in tainted for pr in parent_rows):
                    continue  # its tainted ancestor was itself spared, so this branch is clean
                if child.trust_tier >= TrustTier.INTERNAL_API_VERIFIED:
                    # Independent authoritative evidence for the child itself: keep it ACTIVE and
                    # record that its dependence on the refuted root no longer holds.
                    child.metadata["containment_spared_by"] = "independent_authoritative_evidence"
                    child.metadata["severed_parent"] = root_memory_id
                    child.updated_at = now
                    self._update_claim_row(conn, child)
                    restored_ids.append(cid)
                    continue
                has_clean_verified_parent = any(
                    pr["memory_id"] not in tainted
                    and pr["status"] == MemoryStatus.ACTIVE.value
                    and int(pr["trust_tier"]) >= int(TrustTier.INTERNAL_API_VERIFIED)
                    for pr in parent_rows
                )
                tainted.add(cid)
                if has_clean_verified_parent:
                    if child.status != MemoryStatus.CONTESTED:
                        child.status = MemoryStatus.CONTESTED
                        child.metadata["reverify_reason"] = f"partial_ancestor_refuted:{root_memory_id}"
                        child.updated_at = now
                        self._update_claim_row(conn, child)
                    contested_ids.append(cid)
                else:
                    if child.status != MemoryStatus.QUARANTINED:
                        child.status = MemoryStatus.QUARANTINED
                        child.metadata["quarantined_by_root"] = root_memory_id
                        child.updated_at = now
                        self._update_claim_row(conn, child)
                    quarantined_ids.append(cid)

            # (2) exposed_in: runs that were SHOWN the root or a quarantined/contested descendant.
            exposed_ids = [root_memory_id, *quarantined_ids, *contested_ids]
            affected_runs: list[str] = []
            seen_runs: set[str] = set()
            for chunk in _chunked(exposed_ids):
                ph = ",".join("?" for _ in chunk)
                for r in conn.execute(
                    f"""
                    SELECT run_id, MIN(read_at) AS first_read FROM run_reads
                    WHERE tenant_id = ? AND memory_id IN ({ph})
                    GROUP BY run_id ORDER BY first_read ASC
                    """,
                    [tenant_id, *chunk],
                ).fetchall():
                    if r["run_id"] not in seen_runs:
                        seen_runs.add(r["run_id"])
                        affected_runs.append(r["run_id"])

            # (3) Exposure-only suspects: claims written in an exposed run that touch the same
            # subject but declare no link. Exposure is not proof of dependence, so they are
            # flagged CONTESTED for re-verification rather than quarantined, and suspicion does
            # not propagate further.
            suspect_ids: list[str] = []
            tainted_claims = [root_claim] + [
                self._row_to_claim(r)
                for chunk in _chunked(quarantined_ids)
                for r in conn.execute(
                    f"SELECT * FROM memories WHERE memory_id IN ({','.join('?' for _ in chunk)})", chunk
                ).fetchall()
            ]
            for r_chunk in _chunked(affected_runs):
                r_ph = ",".join("?" for _ in r_chunk)
                for rrow in conn.execute(
                    f"""
                    SELECT * FROM memories
                    WHERE tenant_id = ? AND created_in_run_id IN ({r_ph})
                      AND status IN ('ACTIVE', 'CANDIDATE_UNVERIFIED')
                      AND claim_type != 'negative_constraint'
                    """,
                    [tenant_id, *r_chunk],
                ).fetchall():
                    rc = self._row_to_claim(rrow)
                    if rc.memory_id in seen or rc.memory_id == anti_id:
                        continue
                    if rc.trust_tier >= TrustTier.INTERNAL_API_VERIFIED:
                        continue
                    if any(
                        claims_semantically_related(p, rc.statement, rc.subject) for p in tainted_claims
                    ):
                        if rc.status == MemoryStatus.ACTIVE:
                            rc.status = MemoryStatus.CONTESTED
                        rc.metadata["suspect_exposed_to"] = root_memory_id
                        rc.metadata["reverify_reason"] = "written_in_run_exposed_to_refuted_memory"
                        rc.metadata["requires_verification"] = True
                        rc.updated_at = now
                        self._update_claim_row(conn, rc)
                        suspect_ids.append(rc.memory_id)
                        seen.add(rc.memory_id)

            # Claims that DECLARE a dependence on a suspect are suspects too (still not quarantined).
            s_queue: deque[str] = deque(suspect_ids)
            while s_queue:
                sid = s_queue.popleft()
                for cr in conn.execute(
                    "SELECT child_memory_id FROM memory_derivations WHERE parent_memory_id = ?", (sid,)
                ).fetchall():
                    cid = cr["child_memory_id"]
                    if cid in seen:
                        continue
                    seen.add(cid)
                    c_row = conn.execute(
                        "SELECT * FROM memories WHERE memory_id = ? AND tenant_id = ?", (cid, tenant_id)
                    ).fetchone()
                    if c_row is None:
                        continue
                    child = self._row_to_claim(c_row)
                    if child.status != MemoryStatus.ACTIVE or child.trust_tier >= TrustTier.INTERNAL_API_VERIFIED:
                        continue
                    child.status = MemoryStatus.CONTESTED
                    child.metadata["suspect_exposed_to"] = root_memory_id
                    child.metadata["reverify_reason"] = f"derived_from_suspect:{sid}"
                    child.metadata["requires_verification"] = True
                    child.updated_at = now
                    self._update_claim_row(conn, child)
                    suspect_ids.append(cid)
                    s_queue.append(cid)

            # (4) Flag every affected report for human review
            now_iso = self._utc_iso(now)
            conn.executemany(
                """
                INSERT OR IGNORE INTO report_review_flags (tenant_id, run_id, root_memory_id, reason, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                [(tenant_id, rid, root_memory_id, reason, now_iso) for rid in affected_runs],
            )

            audit = ContainmentAuditEvent(
                tenant_id=tenant_id,
                root_memory_id=root_memory_id,
                anti_memory_id=anti_id,
                quarantined_memory_ids=quarantined_ids,
                contested_memory_ids=contested_ids,
                restored_memory_ids=restored_ids,
                suspect_memory_ids=suspect_ids,
                affected_run_ids=affected_runs,
                llm_calls_used=llm_calls_used,
                created_at=now,
                notes=reason,
            )
            conn.execute(
                """
                INSERT INTO containment_audits (
                    audit_id, tenant_id, root_memory_id, anti_memory_id,
                    quarantined_ids_json, contested_ids_json, affected_run_ids_json,
                    llm_calls_used, created_at, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    audit.audit_id,
                    audit.tenant_id,
                    audit.root_memory_id,
                    audit.anti_memory_id,
                    json.dumps(audit.quarantined_memory_ids),
                    json.dumps(
                        {
                            "contested": audit.contested_memory_ids,
                            "suspect": audit.suspect_memory_ids,
                            "restored": audit.restored_memory_ids,
                        }
                    ),
                    json.dumps(audit.affected_run_ids),
                    audit.llm_calls_used,
                    self._utc_iso(audit.created_at),
                    audit.notes,
                ),
            )
            return audit

        return self._run_write_tx(_tx)

    def stage_global_candidate(
        self,
        *,
        org_id: str,
        cluster_key: str,
        artifact: str,
        error_sig: str,
        sanitized_summary: str,
        now: datetime,
        origin_fingerprint: str | None = None,
        source_hash: str | None = None,
        trust_tier: TrustTier | int = TrustTier.TOOL_OBSERVED,
        sanitized_value: str | None = None,
        evidence_fp: str | None = None,
    ) -> int:
        """
        Records a sanitized contribution keyed by org HMAC (never raw org_id).
        Enforces >= TOOL_OBSERVED trust tier and stores origin/source hashes for Sybil hardening.
        Returns current distinct eligible org count for this cluster.
        """
        tier_val = int(trust_tier)
        if tier_val < int(TrustTier.TOOL_OBSERVED):
            return 0
        h_org = org_hmac(self._hmac_secret, org_id)
        eff_origin = (
            origin_fingerprint.strip()
            if origin_fingerprint and origin_fingerprint.strip()
            else f"org:{h_org[:16]}"
        )
        eff_source = (
            source_hash.strip()
            if source_hash and source_hash.strip()
            else hash_source_id(f"{h_org}:{sanitized_summary}")
        )
        now_iso = self._utc_iso(now)
        # Without an explicit evidence fingerprint the contribution is treated as this tenant's
        # own execution (independent per tenant+source), matching pre-fingerprint behaviour.
        eff_evidence = (
            evidence_fp.strip()
            if evidence_fp and evidence_fp.strip()
            else f"exec:{hash_source_id(f'{h_org}|{eff_source}')}"
        )
        eff_value = (sanitized_value or "").strip()

        def _tx(conn: sqlite3.Connection) -> int:
            already_active = conn.execute(
                """
                SELECT 1 FROM memories
                WHERE tenant_id = ? AND slot_key = ? AND status = 'ACTIVE'
                LIMIT 1
                """,
                (GLOBAL_TENANT_ID, cluster_key),
            ).fetchone()
            promoted_flag = 1 if already_active is not None else 0
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO global_staging
                (cluster_key, artifact, error_sig, org_hmac, sanitized_summary,
                 origin_fingerprint, source_hash, trust_tier, created_at, promoted,
                 sanitized_value, evidence_fingerprint)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cluster_key,
                    artifact,
                    error_sig,
                    h_org,
                    sanitized_summary,
                    eff_origin,
                    eff_source,
                    tier_val,
                    now_iso,
                    promoted_flag,
                    eff_value,
                    eff_evidence,
                ),
            )
            if promoted_flag == 1 and cur.rowcount > 0:
                conn.execute(
                    """
                    INSERT INTO promoted_cluster_counts(cluster_key, org_count, updated_at)
                    VALUES (?, 1, ?)
                    ON CONFLICT(cluster_key) DO UPDATE SET
                        org_count = promoted_cluster_counts.org_count + 1,
                        updated_at = excluded.updated_at
                    """,
                    (cluster_key, now_iso),
                )
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM global_staging WHERE cluster_key = ? AND trust_tier >= ?",
                (cluster_key, int(TrustTier.TOOL_OBSERVED)),
            ).fetchone()
            return int(row["cnt"])

        return self._run_write_tx(_tx)

    def get_promotable_global_clusters(
        self,
        k_anonymity: int = 5,
        min_distinct_origins: int | None = None,
    ) -> list[dict[str, Any]]:
        req_origins = (
            min_distinct_origins
            if min_distinct_origins is not None
            else max(3, k_anonymity - 1)
        )
        conn = self._get_conn()
        clusters = conn.execute(
            """
            SELECT gs.cluster_key, gs.artifact, gs.error_sig,
                   COUNT(DISTINCT gs.org_hmac) AS org_count,
                   COUNT(DISTINCT gs.origin_fingerprint) AS origin_count,
                   COUNT(DISTINCT gs.source_hash) AS source_count,
                   COUNT(DISTINCT gs.evidence_fingerprint) AS evidence_count
            FROM global_staging gs
            WHERE gs.trust_tier >= ?
              AND NOT EXISTS (
                  SELECT 1 FROM memories m
                  WHERE m.tenant_id = ? AND m.slot_key = gs.cluster_key AND m.status = 'ACTIVE'
              )
            GROUP BY gs.cluster_key, gs.artifact, gs.error_sig
            HAVING org_count >= ?
               AND origin_count >= ?
               AND source_count >= ?
               AND evidence_count >= ?
               AND MAX(gs.promoted) = 0
            """,
            (
                int(TrustTier.TOOL_OBSERVED),
                GLOBAL_TENANT_ID,
                k_anonymity,
                req_origins,
                k_anonymity,
                req_origins,
            ),
        ).fetchall()
        out: list[dict[str, Any]] = []
        for c in clusters:
            rows = conn.execute(
                """
                SELECT sanitized_summary, sanitized_value FROM global_staging
                WHERE cluster_key = ? AND trust_tier >= ?
                """,
                (c["cluster_key"], int(TrustTier.TOOL_OBSERVED)),
            ).fetchall()
            per_contributor = [
                f"{r['sanitized_summary']} {r['sanitized_value'] or ''}".strip() for r in rows
            ]
            if contributor_conflict(per_contributor)[0]:
                continue  # contradictory workarounds: never spend a synthesis call on them
            out.append(
                {
                    "cluster_key": c["cluster_key"],
                    "artifact": c["artifact"],
                    "error_sig": c["error_sig"],
                    "org_count": int(c["org_count"]),
                    "origin_count": int(c["origin_count"]),
                    "evidence_count": int(c["evidence_count"]),
                    "summaries": [r["sanitized_summary"] for r in rows],
                    "consensus_value": self._consensus_value(
                        [str(r["sanitized_value"] or "") for r in rows]
                    ),
                }
            )
        return out

    @staticmethod
    def _consensus_value(values: list[str]) -> str | None:
        norm = [" ".join(v.strip().split()) for v in values if v and v.strip()]
        if not norm:
            return None
        counts: dict[str, int] = {}
        for v in norm:
            counts[v.lower()] = counts.get(v.lower(), 0) + 1
        best = max(counts.items(), key=lambda kv: (kv[1], -len(kv[0])))[0]
        return next(v for v in norm if v.lower() == best)

    def promote_global_cluster(
        self,
        *,
        cluster_key: str,
        artifact: str,
        error_sig: str,
        statement: str,
        value: str,
        k_anonymity: int = 5,
        min_distinct_origins: int | None = None,
        now: datetime | None = None,
    ) -> tuple[MemoryClaim | None, WriteOutcome]:
        """
        Sole server-side writer for Scope.GLOBAL advisories.
        Atomically checks K-anonymity org count, distinct origin diversity,
        distinct source hashes, trust >= TOOL_OBSERVED, and zero-leakage validation.
        """
        ok_stmt, _ = validate_global_safety(statement)
        ok_val, _ = validate_global_metadata_field(value, max_len=200)
        ok_art, _ = validate_global_metadata_field(artifact, max_len=120)
        ok_sig, _ = validate_global_metadata_field(error_sig, max_len=120)
        ok_ck, _ = validate_global_metadata_field(cluster_key, max_len=200)
        if not (ok_stmt and ok_val and ok_art and ok_sig and ok_ck):
            return None, WriteOutcome.REJECTED_LOW_VALUE

        ts = now or utcnow()
        ts_iso = self._utc_iso(ts)
        req_origins = (
            min_distinct_origins
            if min_distinct_origins is not None
            else max(3, k_anonymity - 1)
        )

        def _tx(conn: sqlite3.Connection) -> tuple[MemoryClaim | None, WriteOutcome]:
            existing_row = conn.execute(
                """
                SELECT * FROM memories
                WHERE tenant_id = ? AND slot_key = ? AND status = 'ACTIVE'
                LIMIT 1
                """,
                (GLOBAL_TENANT_ID, cluster_key),
            ).fetchone()
            if existing_row is not None:
                conn.execute(
                    "UPDATE global_staging SET promoted = 1 WHERE cluster_key = ?",
                    (cluster_key,),
                )
                return self._row_to_claim(existing_row), WriteOutcome.DEDUPLICATED_SOURCE

            agg = conn.execute(
                """
                SELECT COUNT(DISTINCT org_hmac) AS org_count,
                       COUNT(DISTINCT origin_fingerprint) AS origin_count,
                       COUNT(DISTINCT source_hash) AS source_count,
                       COUNT(DISTINCT evidence_fingerprint) AS evidence_count,
                       MAX(promoted) AS max_promoted,
                       MIN(trust_tier) AS min_trust_tier
                FROM global_staging
                WHERE cluster_key = ? AND trust_tier >= ?
                """,
                (cluster_key, int(TrustTier.TOOL_OBSERVED)),
            ).fetchone()
            if agg is None:
                return None, WriteOutcome.REJECTED_LOW_VALUE

            org_count = int(agg["org_count"] or 0)
            origin_count = int(agg["origin_count"] or 0)
            source_count = int(agg["source_count"] or 0)
            evidence_count = int(agg["evidence_count"] or 0)
            max_promoted = int(agg["max_promoted"] or 0)
            min_trust = int(agg["min_trust_tier"] or 0)
            if (
                org_count < k_anonymity
                or origin_count < req_origins
                or source_count < k_anonymity
                or evidence_count < req_origins
                or max_promoted != 0
                or min_trust < int(TrustTier.TOOL_OBSERVED)
            ):
                return None, WriteOutcome.REJECTED_LOW_VALUE

            # Bind the published advisory to what contributors actually staged: the value must
            # be the >= K consensus (or token-supported by >= K summaries), and any literal in
            # the synthesized statement must already exist in the staged evidence.
            ev_rows = conn.execute(
                """
                SELECT sanitized_summary, sanitized_value FROM global_staging
                WHERE cluster_key = ? AND trust_tier >= ?
                """,
                (cluster_key, int(TrustTier.TOOL_OBSERVED)),
            ).fetchall()
            staged_values = [str(r["sanitized_value"] or "") for r in ev_rows]
            staged_summaries = [str(r["sanitized_summary"] or "") for r in ev_rows]
            if not global_value_supported(value, staged_values, staged_summaries, k_anonymity):
                return None, WriteOutcome.REJECTED_UNSUPPORTED_GLOBAL
            per_contributor = [f"{sm} {sv}".strip() for sm, sv in zip(staged_summaries, staged_values, strict=True)]
            conflict, _conflict_reason = contributor_conflict(per_contributor)
            if conflict:
                return None, WriteOutcome.REJECTED_UNSUPPORTED_GLOBAL
            ok_val_sem, _ = check_against_evidence(value, per_contributor)
            if not ok_val_sem:
                return None, WriteOutcome.REJECTED_UNSUPPORTED_GLOBAL
            evidence_blob = " ".join([*staged_summaries, *staged_values, artifact, error_sig, cluster_key])
            ok_sem, sem_reason = check_against_evidence(statement, per_contributor)
            if statement_grounded_in_evidence(statement, evidence_blob) and ok_sem:
                final_statement = statement
                statement_source = "synthesis_grounded_in_staged_evidence"
            else:
                final_statement = (
                    f"[Cross-Org Advisory | {artifact} / {error_sig}] Observed independently in "
                    f"{org_count} organizations; staged workaround: {value}."
                )
                statement_source = (
                    "deterministic_template_synthesis_ungrounded"
                    if ok_sem
                    else f"deterministic_template_synthesis_{sem_reason.split(':', 1)[0]}"
                )
            ok_final, _ = validate_global_safety(final_statement)
            if not ok_final:
                return None, WriteOutcome.REJECTED_UNSUPPORTED_GLOBAL

            src_id = f"cluster:{cluster_key}"
            claim = MemoryClaim(
                memory_id=new_id("mem"),
                tenant_id=GLOBAL_TENANT_ID,
                scope=Scope.GLOBAL,
                claim_type=ClaimType.PROCEDURE,
                slot_key=cluster_key,
                subject=artifact,
                predicate=error_sig,
                value=value,
                value_hash=value_digest(value),
                statement=final_statement,
                cardinality=Cardinality.SINGLE,
                volatility=Volatility.LOW,
                risk_class=RiskClass.LOW_DIAGNOSTIC,
                trust_tier=TrustTier.CROSS_ORG_CORROBORATED,
                status=MemoryStatus.ACTIVE,
                base_confidence=0.86,
                evidence_count=org_count,
                derived_from_ids=[],
                created_in_run_id="background_maintenance",
                source_type=SourceType.CROSS_ORG_CLUSTER,
                source_id=src_id,
                created_at=ts,
                updated_at=ts,
                last_accessed_at=ts,
                metadata={
                    "source_org_count": org_count,
                    "distinct_origins": origin_count,
                    "distinct_evidence_sources": evidence_count,
                    "statement_source": statement_source,
                    "artifact_name": artifact,
                    "error_signature": error_sig,
                },
            )
            self._insert_claim_and_links(conn, claim, hash_source_id(src_id), ts)
            conn.execute(
                "UPDATE global_staging SET promoted = 1 WHERE cluster_key = ?",
                (cluster_key,),
            )
            conn.execute(
                """
                INSERT INTO promoted_cluster_counts(cluster_key, org_count, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(cluster_key) DO UPDATE SET
                    org_count = MAX(promoted_cluster_counts.org_count, excluded.org_count),
                    updated_at = excluded.updated_at
                """,
                (cluster_key, org_count, ts_iso),
            )
            return claim, WriteOutcome.INSERTED_ACTIVE

        return self._run_write_tx(_tx)

    def sync_promoted_global_counts(self, now: datetime) -> None:
        """Updates source_org_count on already-published global advisories with 0 LLM calls."""
        now_iso = self._utc_iso(now)

        def _tx(conn: sqlite3.Connection) -> None:
            conn.execute(
                """
                INSERT INTO promoted_cluster_counts(cluster_key, org_count, updated_at)
                SELECT cluster_key, COUNT(DISTINCT org_hmac), ?
                FROM global_staging
                WHERE promoted = 1 AND trust_tier >= ?
                GROUP BY cluster_key
                ON CONFLICT(cluster_key) DO UPDATE SET
                    org_count = MAX(promoted_cluster_counts.org_count, excluded.org_count),
                    updated_at = excluded.updated_at
                """,
                (now_iso, int(TrustTier.TOOL_OBSERVED)),
            )
            counts = conn.execute(
                "SELECT cluster_key, org_count FROM promoted_cluster_counts"
            ).fetchall()
            for row in counts:
                c_row = conn.execute(
                    """
                    SELECT * FROM memories
                    WHERE tenant_id = ? AND slot_key = ? AND status = 'ACTIVE'
                    LIMIT 1
                    """,
                    (GLOBAL_TENANT_ID, row["cluster_key"]),
                ).fetchone()
                if c_row is not None:
                    claim = self._row_to_claim(c_row)
                    new_cnt = int(row["org_count"])
                    if int(claim.metadata.get("source_org_count", 0)) < new_cnt:
                        claim.metadata["source_org_count"] = new_cnt
                        claim.evidence_count = max(claim.evidence_count, new_cnt)
                        claim.updated_at = now
                        self._update_claim_row(conn, claim)

        self._run_write_tx(_tx)

    def reserve_tool_action(
        self,
        *,
        tenant_id: str,
        idempotency_key: str,
        run_id: str,
        tool_name: str,
        now: datetime,
    ) -> tuple[bool, str | None]:
        """
        Atomically claims an external side effect before it executes. Returns (reserved,
        prior_status). A key already 'pending' or 'executed' is NOT re-reserved, so a retried
        or concurrently duplicated run cannot send the same email / change the same route twice.
        A 'failed' action may be retried.
        """
        now_iso = self._utc_iso(now)

        def _tx(conn: sqlite3.Connection) -> tuple[bool, str | None]:
            row = conn.execute(
                "SELECT status FROM tool_action_ledger WHERE tenant_id = ? AND idempotency_key = ?",
                (tenant_id, idempotency_key),
            ).fetchone()
            if row is None:
                conn.execute(
                    """
                    INSERT INTO tool_action_ledger
                    (tenant_id, idempotency_key, run_id, tool_name, status, attempts, created_at, updated_at)
                    VALUES (?, ?, ?, ?, 'pending', 1, ?, ?)
                    """,
                    (tenant_id, idempotency_key, run_id, tool_name, now_iso, now_iso),
                )
                return True, None
            prior = str(row["status"])
            if prior == "failed":
                conn.execute(
                    """
                    UPDATE tool_action_ledger
                    SET status = 'pending', attempts = attempts + 1, run_id = ?, updated_at = ?
                    WHERE tenant_id = ? AND idempotency_key = ?
                    """,
                    (run_id, now_iso, tenant_id, idempotency_key),
                )
                return True, prior
            return False, prior

        return self._run_write_tx(_tx)

    def complete_tool_action(
        self, *, tenant_id: str, idempotency_key: str, succeeded: bool, now: datetime
    ) -> None:
        now_iso = self._utc_iso(now)

        def _tx(conn: sqlite3.Connection) -> None:
            conn.execute(
                """
                UPDATE tool_action_ledger SET status = ?, updated_at = ?
                WHERE tenant_id = ? AND idempotency_key = ?
                """,
                ("executed" if succeeded else "failed", now_iso, tenant_id, idempotency_key),
            )

        self._run_write_tx(_tx)

    def mark_global_cluster_promoted(self, cluster_key: str) -> None:
        def _tx(conn: sqlite3.Connection) -> None:
            conn.execute(
                "UPDATE global_staging SET promoted = 1 WHERE cluster_key = ?",
                (cluster_key,),
            )

        self._run_write_tx(_tx)

    def _delete_memories_with_cleanup(self, conn: sqlite3.Connection, ids: list[str]) -> None:
        for chunk in _chunked(ids):
            ph = ",".join("?" for _ in chunk)
            conn.execute(
                f"DELETE FROM memory_derivations WHERE parent_memory_id IN ({ph}) OR child_memory_id IN ({ph})",
                chunk + chunk,
            )
            conn.execute(f"DELETE FROM memories_fts WHERE memory_id IN ({ph})", chunk)
            conn.execute(f"DELETE FROM memories WHERE memory_id IN ({ph})", chunk)

    def compact_and_forget(
        self,
        *,
        now: datetime,
        max_active_per_org: int = 500,
        max_negative_per_org: int = 100,
        max_global_advisories: int = 2000,
        tombstone_ttl_days: int = 90,
        run_reads_ttl_days: int = 60,
        redact_runs_after_days: int = 30,
        runs_ttl_days: int = 90,
        audits_ttl_days: int = 180,
        promoted_staging_ttl_days: int = 180,
    ) -> dict[str, int]:
        """
        Bounded 2+ year retention enforcement with zero LLM calls.
        Atomically rolls up aged telemetry/cost rows before deletion.
        """

        def _tx(conn: sqlite3.Connection) -> dict[str, int]:
            stats = {
                "expired_candidates": 0,
                "evicted_low_utility": 0,
                "purged_tombstones": 0,
                "purged_run_reads": 0,
                "evicted_global": 0,
                "redacted_runs": 0,
                "pruned_runs": 0,
                "pruned_cost_events": 0,
                "pruned_audits": 0,
                "pruned_global_staging": 0,
            }

            # 1) Expire stale unverified candidates past TTL
            now_iso = self._utc_iso(now)
            exp_rows = conn.execute(
                """
                SELECT memory_id FROM memories
                WHERE status = 'CANDIDATE_UNVERIFIED'
                  AND expires_at IS NOT NULL
                  AND expires_at <= ?
                """,
                (now_iso,),
            ).fetchall()
            if exp_rows:
                ids = [r["memory_id"] for r in exp_rows]
                self._delete_memories_with_cleanup(conn, ids)
                stats["expired_candidates"] = len(ids)

            # 2) Enforce per-org active/contested capacity via deterministic utility score
            org_rows = conn.execute(
                """
                SELECT DISTINCT tenant_id FROM memories
                WHERE tenant_id != ?
                  AND status IN ('ACTIVE', 'CONTESTED', 'CANDIDATE_UNVERIFIED')
                """,
                (GLOBAL_TENANT_ID,),
            ).fetchall()
            for orow in org_rows:
                tid = orow["tenant_id"]
                neg_rows = conn.execute(
                    """
                    SELECT * FROM memories
                    WHERE tenant_id = ? AND claim_type = 'negative_constraint'
                      AND status = 'ACTIVE'
                    """,
                    (tid,),
                ).fetchall()
                if len(neg_rows) > max_negative_per_org:
                    neg_claims = [self._row_to_claim(r) for r in neg_rows]
                    neg_claims.sort(key=lambda c: compute_retention_utility(c, now))
                    to_drop = [c.memory_id for c in neg_claims[: len(neg_claims) - max_negative_per_org]]
                    self._delete_memories_with_cleanup(conn, to_drop)
                    stats["evicted_low_utility"] += len(to_drop)

                act_rows = conn.execute(
                    """
                    SELECT * FROM memories
                    WHERE tenant_id = ? AND claim_type != 'negative_constraint'
                      AND status IN ('ACTIVE', 'CONTESTED', 'CANDIDATE_UNVERIFIED')
                    """,
                    (tid,),
                ).fetchall()
                if len(act_rows) > max_active_per_org:
                    claims = [self._row_to_claim(r) for r in act_rows]
                    claims.sort(key=lambda c: compute_retention_utility(c, now))
                    to_drop = [c.memory_id for c in claims[: len(claims) - max_active_per_org]]
                    self._delete_memories_with_cleanup(conn, to_drop)
                    stats["evicted_low_utility"] += len(to_drop)

            # 3) Enforce global advisory cap
            g_rows = conn.execute(
                "SELECT * FROM memories WHERE tenant_id = ? AND status = 'ACTIVE'",
                (GLOBAL_TENANT_ID,),
            ).fetchall()
            if len(g_rows) > max_global_advisories:
                g_claims = [self._row_to_claim(r) for r in g_rows]
                g_claims.sort(key=lambda c: compute_retention_utility(c, now))
                to_drop = [c.memory_id for c in g_claims[: len(g_claims) - max_global_advisories]]
                self._delete_memories_with_cleanup(conn, to_drop)
                stats["evicted_global"] += len(to_drop)

            # 4) Purge old tombstones (SUPERSEDED / REFUTED / QUARANTINED / ARCHIVED)
            tomb_cutoff = self._utc_iso(now - timedelta(days=tombstone_ttl_days))
            tomb_rows = conn.execute(
                """
                SELECT memory_id FROM memories
                WHERE status IN ('SUPERSEDED', 'REFUTED', 'QUARANTINED', 'ARCHIVED')
                  AND updated_at <= ?
                """,
                (tomb_cutoff,),
            ).fetchall()
            if tomb_rows:
                t_ids = [r["memory_id"] for r in tomb_rows]
                self._delete_memories_with_cleanup(conn, t_ids)
                stats["purged_tombstones"] = len(t_ids)

            # 5) Prune rolling run_reads audit window and aged evidence_sources
            reads_cutoff = self._utc_iso(now - timedelta(days=run_reads_ttl_days))
            res = conn.execute("DELETE FROM run_reads WHERE read_at <= ?", (reads_cutoff,))
            stats["purged_run_reads"] = res.rowcount
            # Idempotency keys only need to outlive any plausible retry of the same task.
            conn.execute(
                "DELETE FROM tool_action_ledger WHERE updated_at <= ? AND status != 'pending'",
                (self._utc_iso(now - timedelta(days=90)),),
            )
            conn.execute(
                """
                DELETE FROM evidence_sources
                WHERE observed_at <= ?
                  AND memory_id NOT IN (SELECT memory_id FROM memories)
                """,
                (tomb_cutoff,),
            )

            # 6) Redact surviving run payloads past 30d; roll up & prune runs past runs_ttl_days
            redact_cutoff = self._utc_iso(now - timedelta(days=redact_runs_after_days))
            runs_cutoff = self._utc_iso(now - timedelta(days=runs_ttl_days))
            res_red = conn.execute(
                """
                UPDATE runs
                SET task_text = '[REDACTED]', final_output = '[REDACTED]'
                WHERE finished_at <= ? AND finished_at > ?
                  AND (task_text != '[REDACTED]' OR final_output != '[REDACTED]')
                """,
                (redact_cutoff, runs_cutoff),
            )
            stats["redacted_runs"] = res_red.rowcount
            conn.execute(
                """
                INSERT INTO daily_cohort_rollups (
                    day_utc, cohort, runs_count, tool_calls_sum, tool_errors_sum,
                    contradiction_runs, blocked_unsafe_sum, reopened_sum, agent_llm_sum, memory_llm_sum
                )
                SELECT
                    substr(finished_at, 1, 10) AS day_utc,
                    cohort,
                    COUNT(*),
                    COALESCE(SUM(tool_calls_count), 0),
                    COALESCE(SUM(tool_error_count), 0),
                    COALESCE(SUM(CASE WHEN in_run_contradictions > 0 THEN 1 ELSE 0 END), 0),
                    COALESCE(SUM(blocked_unsafe_actions), 0),
                    COALESCE(SUM(reopened), 0),
                    COALESCE(SUM(agent_llm_calls), 0),
                    COALESCE(SUM(memory_llm_calls), 0)
                FROM runs
                WHERE finished_at <= ?
                GROUP BY substr(finished_at, 1, 10), cohort
                ON CONFLICT(day_utc, cohort) DO UPDATE SET
                    runs_count = daily_cohort_rollups.runs_count + excluded.runs_count,
                    tool_calls_sum = daily_cohort_rollups.tool_calls_sum + excluded.tool_calls_sum,
                    tool_errors_sum = daily_cohort_rollups.tool_errors_sum + excluded.tool_errors_sum,
                    contradiction_runs = daily_cohort_rollups.contradiction_runs + excluded.contradiction_runs,
                    blocked_unsafe_sum = daily_cohort_rollups.blocked_unsafe_sum + excluded.blocked_unsafe_sum,
                    reopened_sum = daily_cohort_rollups.reopened_sum + excluded.reopened_sum,
                    agent_llm_sum = daily_cohort_rollups.agent_llm_sum + excluded.agent_llm_sum,
                    memory_llm_sum = daily_cohort_rollups.memory_llm_sum + excluded.memory_llm_sum
                """,
                (runs_cutoff,),
            )
            res_runs = conn.execute("DELETE FROM runs WHERE finished_at <= ?", (runs_cutoff,))
            stats["pruned_runs"] = res_runs.rowcount

            # 7) Atomically roll up aged llm_cost_events before deleting
            conn.execute(
                """
                INSERT INTO daily_cost_rollups (
                    day_utc, operation, attempts, failed_attempts, tokens_in, tokens_out
                )
                SELECT
                    substr(created_at, 1, 10) AS day_utc,
                    operation,
                    COUNT(*),
                    COALESCE(SUM(CASE WHEN succeeded = 0 THEN 1 ELSE 0 END), 0),
                    COALESCE(SUM(tokens_in), 0),
                    COALESCE(SUM(tokens_out), 0)
                FROM llm_cost_events
                WHERE created_at <= ?
                GROUP BY substr(created_at, 1, 10), operation
                ON CONFLICT(day_utc, operation) DO UPDATE SET
                    attempts = daily_cost_rollups.attempts + excluded.attempts,
                    failed_attempts = daily_cost_rollups.failed_attempts + excluded.failed_attempts,
                    tokens_in = daily_cost_rollups.tokens_in + excluded.tokens_in,
                    tokens_out = daily_cost_rollups.tokens_out + excluded.tokens_out
                """,
                (tomb_cutoff,),
            )
            res_cost = conn.execute(
                "DELETE FROM llm_cost_events WHERE created_at <= ?",
                (tomb_cutoff,),
            )
            stats["pruned_cost_events"] = res_cost.rowcount

            # 8) Prune aged containment_audits & promoted global_staging rows (after syncing counts)
            audits_cutoff = self._utc_iso(now - timedelta(days=audits_ttl_days))
            res_aud = conn.execute(
                "DELETE FROM containment_audits WHERE created_at <= ?",
                (audits_cutoff,),
            )
            stats["pruned_audits"] = res_aud.rowcount
            conn.execute(
                "DELETE FROM report_review_flags WHERE reviewed = 1 AND created_at <= ?",
                (audits_cutoff,),
            )

            conn.execute(
                """
                INSERT INTO promoted_cluster_counts(cluster_key, org_count, updated_at)
                SELECT cluster_key, COUNT(DISTINCT org_hmac), ?
                FROM global_staging
                WHERE promoted = 1 AND trust_tier >= ?
                GROUP BY cluster_key
                ON CONFLICT(cluster_key) DO UPDATE SET
                    org_count = MAX(promoted_cluster_counts.org_count, excluded.org_count),
                    updated_at = excluded.updated_at
                """,
                (now_iso, int(TrustTier.TOOL_OBSERVED)),
            )
            staging_cutoff = self._utc_iso(now - timedelta(days=promoted_staging_ttl_days))
            res_stg = conn.execute(
                "DELETE FROM global_staging WHERE promoted = 1 AND created_at <= ?",
                (staging_cutoff,),
            )
            stats["pruned_global_staging"] = res_stg.rowcount

            return stats

        return self._run_write_tx(_tx)

    def approve_run_memories(self, run_id: str, tenant_id: str, now: datetime) -> list[str]:
        """
        Strengthens utility/confidence for ACTIVE claims consumed during an approved run.
        Never launders unverified external candidates or contested claims into HUMAN_VERIFIED,
        and never elevates a single org's approval onto global cross-org advisories.
        """

        def _tx(conn: sqlite3.Connection) -> list[str]:
            rows = conn.execute(
                """
                SELECT DISTINCT m.*, MAX(r.was_cited) AS max_cited FROM run_reads r
                JOIN memories m ON m.memory_id = r.memory_id
                WHERE r.run_id = ? AND r.tenant_id = ?
                  AND m.status = 'ACTIVE'
                  AND m.claim_type != 'negative_constraint'
                GROUP BY m.memory_id
                """,
                (run_id, tenant_id),
            ).fetchall()
            promoted: list[str] = []
            for r in rows:
                c = self._row_to_claim(r)
                c.verified_helps += 1
                c.updated_at = now
                # Global advisories only get utility help counts, not single-tenant trust escalation
                if c.tenant_id == GLOBAL_TENANT_ID:
                    self._update_claim_row(conn, c)
                    promoted.append(c.memory_id)
                    continue

                if c.trust_tier == TrustTier.AGENT_INFERRED and c.risk_class != RiskClass.HIGH_ACTION_ROUTING:
                    c.trust_tier = TrustTier.TOOL_OBSERVED
                elif (
                    c.trust_tier == TrustTier.TOOL_OBSERVED
                    and int(r["max_cited"] or 0) == 1
                    and c.risk_class != RiskClass.HIGH_ACTION_ROUTING
                ):
                    c.trust_tier = TrustTier.HUMAN_VERIFIED

                if c.risk_class != RiskClass.HIGH_ACTION_ROUTING:
                    c.base_confidence = min(0.98, max(c.base_confidence, 0.72) + 0.05)
                self._update_claim_row(conn, c)
                promoted.append(c.memory_id)
            return promoted

        return self._run_write_tx(_tx)

    def record_cost_event(
        self,
        *,
        operation: CostOperation,
        run_id: str | None = None,
        attempt_num: int = 1,
        tokens_in: int = 0,
        tokens_out: int = 0,
        succeeded: bool = True,
        count: int = 1,
        created_at: datetime | None = None,
    ) -> None:
        ts_iso = self._utc_iso(created_at)
        pid = os.getpid()
        rows = [
            (
                new_id("cost"),
                run_id,
                operation.value,
                pid,
                attempt_num,
                tokens_in,
                tokens_out,
                1 if succeeded else 0,
                ts_iso,
            )
            for _ in range(max(1, count))
        ]

        def _tx(conn: sqlite3.Connection) -> None:
            conn.executemany(
                """
                INSERT INTO llm_cost_events (
                    event_id, run_id, operation, worker_pid, attempt_num,
                    tokens_in, tokens_out, succeeded, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

        self._run_write_tx(_tx)

    def get_persisted_cost_summary(
        self,
        *,
        since: datetime | None = None,
        window_days: int | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        conn = self._get_conn()
        eff_since = since
        if eff_since is None and window_days is not None:
            ref_now = now or utcnow()
            eff_since = ref_now - timedelta(days=window_days)

        if eff_since is not None:
            since_iso = self._utc_iso(eff_since)
            since_day = since_iso[:10]
            raw_rows = conn.execute(
                """
                SELECT operation,
                       COUNT(*) AS attempts,
                       SUM(CASE WHEN succeeded = 0 THEN 1 ELSE 0 END) AS failed_attempts,
                       SUM(tokens_in) AS total_tokens_in,
                       SUM(tokens_out) AS total_tokens_out
                FROM llm_cost_events
                WHERE created_at >= ?
                GROUP BY operation
                """,
                (since_iso,),
            ).fetchall()
            rollup_rows = conn.execute(
                """
                SELECT operation,
                       COALESCE(SUM(attempts), 0) AS attempts,
                       COALESCE(SUM(failed_attempts), 0) AS failed_attempts,
                       COALESCE(SUM(tokens_in), 0) AS total_tokens_in,
                       COALESCE(SUM(tokens_out), 0) AS total_tokens_out
                FROM daily_cost_rollups
                WHERE day_utc >= ?
                GROUP BY operation
                """,
                (since_day,),
            ).fetchall()
            pid_count = int(
                conn.execute(
                    "SELECT COUNT(DISTINCT worker_pid) FROM llm_cost_events WHERE created_at >= ?",
                    (since_iso,),
                ).fetchone()[0]
            )
            mem_runs_raw = int(
                conn.execute(
                    "SELECT COUNT(*) FROM runs WHERE cohort = 'memory' AND finished_at >= ?",
                    (since_iso,),
                ).fetchone()[0]
            )
            mem_runs_rollup = int(
                conn.execute(
                    "SELECT COALESCE(SUM(runs_count), 0) FROM daily_cohort_rollups WHERE cohort = 'memory' AND day_utc >= ?",
                    (since_day,),
                ).fetchone()[0]
            )
            event_runs = int(
                conn.execute(
                    "SELECT COUNT(DISTINCT run_id) FROM llm_cost_events WHERE run_id IS NOT NULL AND created_at >= ?",
                    (since_iso,),
                ).fetchone()[0]
            )
        else:
            raw_rows = conn.execute(
                """
                SELECT operation,
                       COUNT(*) AS attempts,
                       SUM(CASE WHEN succeeded = 0 THEN 1 ELSE 0 END) AS failed_attempts,
                       SUM(tokens_in) AS total_tokens_in,
                       SUM(tokens_out) AS total_tokens_out
                FROM llm_cost_events
                GROUP BY operation
                """
            ).fetchall()
            rollup_rows = conn.execute(
                """
                SELECT operation,
                       COALESCE(SUM(attempts), 0) AS attempts,
                       COALESCE(SUM(failed_attempts), 0) AS failed_attempts,
                       COALESCE(SUM(tokens_in), 0) AS total_tokens_in,
                       COALESCE(SUM(tokens_out), 0) AS total_tokens_out
                FROM daily_cost_rollups
                GROUP BY operation
                """
            ).fetchall()
            pid_count = int(
                conn.execute("SELECT COUNT(DISTINCT worker_pid) FROM llm_cost_events").fetchone()[0]
            )
            mem_runs_raw = int(
                conn.execute("SELECT COUNT(*) FROM runs WHERE cohort = 'memory'").fetchone()[0]
            )
            mem_runs_rollup = int(
                conn.execute(
                    "SELECT COALESCE(SUM(runs_count), 0) FROM daily_cohort_rollups WHERE cohort = 'memory'"
                ).fetchone()[0]
            )
            event_runs = int(
                conn.execute(
                    "SELECT COUNT(DISTINCT run_id) FROM llm_cost_events WHERE run_id IS NOT NULL"
                ).fetchone()[0]
            )

        by_op: dict[str, int] = {}
        total_attempts = 0
        failed_attempts = 0
        tokens_in = 0
        tokens_out = 0
        for r in [*raw_rows, *rollup_rows]:
            op = str(r["operation"])
            att = int(r["attempts"] or 0)
            by_op[op] = by_op.get(op, 0) + att
            total_attempts += att
            failed_attempts += int(r["failed_attempts"] or 0)
            tokens_in += int(r["total_tokens_in"] or 0)
            tokens_out += int(r["total_tokens_out"] or 0)

        mem_runs_recorded = mem_runs_raw + mem_runs_rollup
        runs_tracked = max(mem_runs_recorded, event_runs)
        return {
            "runs_tracked": runs_tracked,
            "memory_calls_total": total_attempts,
            "failed_attempts": failed_attempts,
            "memory_extract_calls": by_op.get(CostOperation.MEMORY_EXTRACT.value, 0),
            "memory_adjudicate_calls": by_op.get(CostOperation.MEMORY_ADJUDICATE.value, 0),
            "memory_feedback_attrib_calls": by_op.get(CostOperation.MEMORY_FEEDBACK_ATTRIB.value, 0),
            "memory_global_synth_calls": by_op.get(CostOperation.MEMORY_GLOBAL_SYNTH.value, 0),
            "total_tokens_in": tokens_in,
            "total_tokens_out": tokens_out,
            "distinct_worker_pids": pid_count,
            "avg_memory_calls_per_run": round(total_attempts / runs_tracked, 4)
            if runs_tracked
            else 0.0,
        }

    def count_stats(self) -> dict[str, int]:
        conn = self._get_conn()
        total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        active = conn.execute("SELECT COUNT(*) FROM memories WHERE status = 'ACTIVE'").fetchone()[0]
        contested = conn.execute("SELECT COUNT(*) FROM memories WHERE status = 'CONTESTED'").fetchone()[0]
        candidates = conn.execute("SELECT COUNT(*) FROM memories WHERE status = 'CANDIDATE_UNVERIFIED'").fetchone()[0]
        quarantined = conn.execute("SELECT COUNT(*) FROM memories WHERE status = 'QUARANTINED'").fetchone()[0]
        refuted = conn.execute("SELECT COUNT(*) FROM memories WHERE status = 'REFUTED'").fetchone()[0]
        global_active = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE tenant_id = ? AND status = 'ACTIVE'",
            (GLOBAL_TENANT_ID,),
        ).fetchone()[0]
        reads = conn.execute("SELECT COUNT(*) FROM run_reads").fetchone()[0]
        durable_refs = conn.execute("SELECT COUNT(*) FROM refutation_ledger").fetchone()[0]
        runs_cnt = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        audits_cnt = conn.execute("SELECT COUNT(*) FROM containment_audits").fetchone()[0]
        staging_cnt = conn.execute("SELECT COUNT(*) FROM global_staging").fetchone()[0]
        cost_cnt = conn.execute("SELECT COUNT(*) FROM llm_cost_events").fetchone()[0]
        cohort_rollups_cnt = conn.execute("SELECT COUNT(*) FROM daily_cohort_rollups").fetchone()[0]
        cost_rollups_cnt = conn.execute("SELECT COUNT(*) FROM daily_cost_rollups").fetchone()[0]
        promoted_cnt = conn.execute("SELECT COUNT(*) FROM promoted_cluster_counts").fetchone()[0]
        ev_src_cnt = conn.execute("SELECT COUNT(*) FROM evidence_sources").fetchone()[0]
        deriv_cnt = conn.execute("SELECT COUNT(*) FROM memory_derivations").fetchone()[0]
        return {
            "total_memories": int(total),
            "active": int(active),
            "contested": int(contested),
            "candidates": int(candidates),
            "quarantined": int(quarantined),
            "refuted": int(refuted),
            "global_active": int(global_active),
            "run_reads": int(reads),
            "refutation_ledger": int(durable_refs),
            "runs": int(runs_cnt),
            "containment_audits": int(audits_cnt),
            "global_staging": int(staging_cnt),
            "llm_cost_events": int(cost_cnt),
            "daily_cohort_rollups": int(cohort_rollups_cnt),
            "daily_cost_rollups": int(cost_rollups_cnt),
            "promoted_cluster_counts": int(promoted_cnt),
            "evidence_sources": int(ev_src_cnt),
            "memory_derivations": int(deriv_cnt),
        }

    def get_telemetry_summary(self) -> dict[str, dict[str, Any]]:
        conn = self._get_conn()
        cohorts = conn.execute("SELECT DISTINCT cohort FROM runs").fetchall()
        summary: dict[str, dict[str, Any]] = {}
        for cr in cohorts:
            cohort = cr["cohort"]
            rows = conn.execute(
                """
                SELECT tool_calls_count, tool_error_count, in_run_contradictions,
                       blocked_unsafe_actions, reopened, critical_path_ms,
                       agent_llm_calls, memory_llm_calls
                FROM runs WHERE cohort = ?
                """,
                (cohort,),
            ).fetchall()
            n = len(rows)
            if n == 0:
                continue
            latencies = sorted(float(r["critical_path_ms"]) for r in rows)
            p95_idx = min(n - 1, int(0.95 * (n - 1)))
            summary[cohort] = {
                "runs": n,
                "avg_tool_calls": round(sum(r["tool_calls_count"] for r in rows) / n, 3),
                "avg_tool_errors": round(sum(r["tool_error_count"] for r in rows) / n, 4),
                "in_run_contradiction_rate": round(
                    sum(1 for r in rows if r["in_run_contradictions"] > 0) / n, 4
                ),
                "blocked_unsafe_actions": sum(r["blocked_unsafe_actions"] for r in rows),
                "reopen_rate": round(sum(r["reopened"] for r in rows) / n, 4),
                "p95_critical_path_ms": round(latencies[p95_idx], 3),
                "avg_agent_llm_calls": round(sum(r["agent_llm_calls"] for r in rows) / n, 3),
                "avg_memory_llm_calls": round(sum(r["memory_llm_calls"] for r in rows) / n, 3),
            }
        return summary
