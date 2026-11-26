from __future__ import annotations

import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path

from rapidfuzz import fuzz

from agentic_memory.models import utcnow
from agentic_memory.policy import estimate_tokens


class NaiveRAGMemory:
    """
    Comparative baseline: naive similarity-RAG memory over stored claim/snippet texts.
    - No trust tiers or capability gating
    - No temporal decay or slot-conflict detection
    - No provenance DAG containment or negative constraints
    - No cross-tenant advisory synthesis
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode = WAL;")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS rag_snippets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                org_id TEXT NOT NULL,
                text TEXT NOT NULL,
                source_type TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_rag_org ON rag_snippets(org_id)"
        )
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def add_snippet(
        self, org_id: str, text: str, source_type: str, now: datetime | None = None
    ) -> None:
        ts = (now or utcnow()).isoformat()
        with self._lock:
            self._conn.execute(
                "INSERT INTO rag_snippets(org_id, text, source_type, created_at) VALUES (?, ?, ?, ?)",
                (org_id, text, source_type, ts),
            )
            self._conn.commit()

    def retrieve(
        self, org_id: str, query: str, max_tokens: int = 4000, top_k: int = 20
    ) -> tuple[list[str], float]:
        t0 = time.perf_counter()
        with self._lock:
            rows = self._conn.execute(
                "SELECT text FROM rag_snippets WHERE org_id = ?", (org_id,)
            ).fetchall()
        q_low = query.lower()
        scored = sorted(
            (
                (fuzz.token_set_ratio(q_low, str(r["text"]).lower()), str(r["text"]))
                for r in rows
            ),
            key=lambda x: x[0],
            reverse=True,
        )
        out: list[str] = []
        used = 0
        for _, txt in scored[:top_k]:
            tok = estimate_tokens(txt)
            if used + tok > max_tokens:
                break
            out.append(txt)
            used += tok
        return out, round((time.perf_counter() - t0) * 1000.0, 3)
