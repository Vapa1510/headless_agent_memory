from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from agentic_memory.models import (
    CandidateClaim,
    ContainmentAuditEvent,
    CostOperation,
    MemoryClaim,
    TenantPolicy,
    TrustTier,
    WriteOutcome,
)

if TYPE_CHECKING:
    from agentic_memory.policy import SourceAuthorityRegistry


@runtime_checkable
class MemoryStoreProtocol(Protocol):
    """Persistence contract separating epistemic memory policy from storage engines."""

    def close(self) -> None:
        ...

    def upsert_tenant(self, policy: TenantPolicy) -> None:
        ...

    def get_tenant(self, org_id: str) -> TenantPolicy:
        ...

    def hydrate_claim_tuple(self, r: tuple[Any, ...]) -> MemoryClaim:
        ...

    def get_claim(
        self, memory_id: str, tenant_id: str | None = None, *, include_global: bool = True
    ) -> MemoryClaim | None:
        ...

    def get_claims_by_slot(self, tenant_id: str, slot_key: str) -> list[MemoryClaim]:
        ...

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
        ...

    def find_matching_refutations(
        self,
        tenant_id: str,
        serialized_targets: str,
        exclude_ids: set[str] | None = None,
    ) -> list[MemoryClaim]:
        ...

    def query_recall_rows(
        self,
        *,
        org_id: str,
        user_id: str | None,
        query_text: str,
        global_limit: int = 20,
        now: datetime | None = None,
    ) -> tuple[list[tuple[Any, ...]], list[tuple[Any, ...]]]:
        ...

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
        ...

    def get_run_recalled_claims(self, run_id: str, tenant_id: str) -> list[MemoryClaim]:
        ...

    def get_run_created_claims(self, run_id: str, tenant_id: str) -> list[MemoryClaim]:
        ...

    def increment_contradiction_hit(
        self, memory_id: str, now: datetime, *, tenant_id: str | None = None
    ) -> MemoryClaim | None:
        ...

    def get_run_record(self, run_id: str, tenant_id: str | None = None) -> Mapping[str, Any] | Any | None:
        ...

    def run_exists_for_other_tenant(self, run_id: str, tenant_id: str) -> bool:
        ...

    def contest_uncertain_roots(
        self, *, tenant_id: str, memory_ids: list[str], run_id: str, reason: str, now: datetime
    ) -> list[str]: ...

    def get_flagged_reports(self, tenant_id: str) -> list[dict[str, Any]]:
        ...

    def reserve_tool_action(
        self, *, tenant_id: str, idempotency_key: str, run_id: str, tool_name: str, now: datetime
    ) -> tuple[bool, str | None]:
        ...

    def complete_tool_action(
        self, *, tenant_id: str, idempotency_key: str, succeeded: bool, now: datetime
    ) -> None:
        ...

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
        ...

    def refute_and_contain(
        self,
        *,
        tenant_id: str,
        root_memory_id: str,
        reason: str,
        now: datetime,
        llm_calls_used: int = 0,
    ) -> ContainmentAuditEvent:
        ...

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
    ) -> int:
        ...

    def get_promotable_global_clusters(
        self,
        k_anonymity: int = 5,
        min_distinct_origins: int | None = None,
    ) -> list[dict[str, Any]]:
        ...

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
        ...

    def sync_promoted_global_counts(self, now: datetime) -> None:
        ...

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
        ...

    def approve_run_memories(self, run_id: str, tenant_id: str, now: datetime) -> list[str]:
        ...

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
        ...

    def get_persisted_cost_summary(
        self,
        *,
        since: datetime | None = None,
        window_days: int | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        ...
