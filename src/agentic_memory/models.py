from __future__ import annotations

from datetime import UTC, datetime
from enum import IntEnum, StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

GLOBAL_TENANT_ID = "__global__"
MAX_CONTEXT_TOKENS = 4000
PACKING_SOFT_CAP_TOKENS = 3600
MAX_AVG_LLM_CALLS_PER_RUN = 2.0
MAX_CRITICAL_PATH_P95_MS = 300.0


def utcnow() -> datetime:
    return datetime.now(UTC)


def content_digest(text: str) -> str:
    """SHA-256 of exact content; attestation seals bind to it."""
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def new_id(prefix: str = "mem") -> str:
    return f"{prefix}_{uuid4().hex[:16]}"


class Scope(StrEnum):
    TASK = "task"
    USER = "user"
    ORG = "org"
    GLOBAL = "global"


class ClaimType(StrEnum):
    FACT = "fact"
    PROCEDURE = "procedure"
    NEGATIVE_CONSTRAINT = "negative_constraint"


class Cardinality(StrEnum):
    SINGLE = "single"
    MULTI = "multi"


class Volatility(StrEnum):
    HIGH = "high"  # 14-day half-life (incident states, temporary outages)
    MEDIUM = "medium"  # 60-day half-life (infra versions, contacts, vendor schemas)
    LOW = "low"  # 365-day half-life (stable policies, library workarounds)
    PERMANENT = "permanent"  # No decay (human refutations / verified invariants)


class RiskClass(StrEnum):
    LOW_DIAGNOSTIC = "low_diagnostic"
    MEDIUM_OPERATIONAL = "medium_operational"
    HIGH_ACTION_ROUTING = "high_action_routing"


class TrustTier(IntEnum):
    UNTRUSTED_EXTERNAL = 1
    AGENT_INFERRED = 2
    TOOL_OBSERVED = 3
    CROSS_ORG_CORROBORATED = 4
    INTERNAL_API_VERIFIED = 5
    HUMAN_VERIFIED = 6


class SourceType(StrEnum):
    EMAIL = "email"
    SUPPORT_TICKET = "support_ticket"
    WEB_PAGE = "web_page"
    UPLOAD = "upload"
    AGENT_INFERENCE = "agent_inference"
    TOOL_OUTPUT = "tool_output"
    INTERNAL_API = "internal_api"
    HUMAN_FEEDBACK = "human_feedback"
    CROSS_ORG_CLUSTER = "cross_org_cluster"


class MemoryStatus(StrEnum):
    ACTIVE = "ACTIVE"
    CANDIDATE_UNVERIFIED = "CANDIDATE_UNVERIFIED"
    CONTESTED = "CONTESTED"
    SUPERSEDED = "SUPERSEDED"
    QUARANTINED = "QUARANTINED"
    REFUTED = "REFUTED"
    ARCHIVED = "ARCHIVED"


class Cohort(StrEnum):
    MEMORY = "memory"
    STATELESS_HOLDOUT = "stateless_holdout"
    NAIVE_RAG = "naive_rag"


class FeedbackVerdict(StrEnum):
    APPROVE = "approve"
    REJECT = "reject"
    CORRECTION = "correction"


class TenantPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    org_id: str
    authorized_domains: list[str] = Field(default_factory=list)
    org_aliases: list[str] = Field(default_factory=list)


class SourceProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    source_type: SourceType
    source_domain: str | None = None
    origin_domain: str | None = None
    is_external_untrusted: bool = True
    run_id: str
    observed_at: datetime = Field(default_factory=utcnow)
    connector_id: str | None = None
    authority_seal: str | None = None
    tool_capability: str | None = None
    # Attestation context: a seal is valid only for this tenant, this run and this exact content.
    tenant_id: str | None = None
    content_digest: str | None = None
    # "observation": digest of the tool output the claim was read from (checked at ingestion);
    # "claim_value": digest of the claim's own value (checked at write time).
    content_kind: str | None = None


class CandidateClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scope: Scope = Scope.ORG
    user_id: str | None = None
    task_id: str | None = None
    claim_type: ClaimType = ClaimType.FACT
    subject: str
    predicate: str
    value: str
    statement: str
    slot_key: str | None = None
    cardinality: Cardinality = Cardinality.SINGLE
    volatility: Volatility = Volatility.MEDIUM
    risk_class: RiskClass = RiskClass.LOW_DIAGNOSTIC
    trust_tier: TrustTier = TrustTier.AGENT_INFERRED
    base_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    derived_from_ids: list[str] = Field(default_factory=list)
    provenance: SourceProvenance
    is_public_technical_artifact: bool = False
    artifact_name: str | None = None
    error_signature: str | None = None
    source_quote: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_id: str = Field(default_factory=lambda: new_id("mem"))
    tenant_id: str
    scope: Scope
    user_id: str | None = None
    task_id: str | None = None
    claim_type: ClaimType
    slot_key: str
    subject: str
    predicate: str
    value: str
    value_hash: str
    statement: str
    cardinality: Cardinality
    volatility: Volatility
    risk_class: RiskClass
    trust_tier: TrustTier
    status: MemoryStatus
    base_confidence: float = Field(ge=0.0, le=1.0)
    evidence_count: int = Field(default=1, ge=1)
    derived_from_ids: list[str] = Field(default_factory=list)
    created_in_run_id: str
    source_type: SourceType
    source_id: str
    source_domain: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    last_accessed_at: datetime = Field(default_factory=utcnow)
    access_count: int = 0
    verified_helps: int = 0
    contradiction_hits: int = 0
    version: int = 1
    expires_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RecallItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: MemoryClaim
    effective_confidence: float
    relevance_score: float
    bucket: str
    rendered_line: str
    token_estimate: int


class RecallBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    org_id: str
    run_id: str
    items: list[RecallItem] = Field(default_factory=list)
    rendered_context: str = ""
    token_count: int = 0
    latency_ms: float = 0.0
    read_memory_ids: list[str] = Field(default_factory=list)
    verification_requests: list[dict[str, str]] = Field(default_factory=list)


class ToolExecutionGateResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allowed: bool
    reason: str
    blocking_memory_ids: list[str] = Field(default_factory=list)
    supporting_memory_ids: list[str] = Field(default_factory=list)
    requires_verification: bool = False


class ToolObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_idx: int
    tool_name: str
    source_id: str
    source_type: SourceType
    source_domain: str | None = None
    origin_domain: str | None = None
    is_external_untrusted: bool = True
    content: str
    structured_facts: list[dict[str, Any]] = Field(default_factory=list)
    observed_at: datetime = Field(default_factory=utcnow)
    connector_id: str | None = None
    authority_seal: str | None = None
    tool_capability: str | None = None
    tenant_id: str | None = None
    run_id: str | None = None
    content_digest: str | None = None


class CostOperation(StrEnum):
    AGENT_STEP = "agent_step"
    MEMORY_EXTRACT = "memory_extract"
    MEMORY_ADJUDICATE = "memory_adjudicate"
    MEMORY_FEEDBACK_ATTRIB = "memory_feedback_attrib"
    MEMORY_GLOBAL_SYNTH = "memory_global_synth"


class PersistedCostEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str | None = None
    operation: CostOperation
    worker_pid: int
    attempt_num: int = 1
    tokens_in: int = 0
    tokens_out: int = 0
    succeeded: bool = True
    created_at: datetime = Field(default_factory=utcnow)


class ClaimTransition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_id: str
    tenant_id: str
    from_status: MemoryStatus
    to_status: MemoryStatus
    reason: str = ""
    changed_at: datetime = Field(default_factory=utcnow)


class HumanFeedback(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feedback_id: str = Field(default_factory=lambda: new_id("fb"))
    run_id: str
    org_id: str
    verdict: FeedbackVerdict
    correction_text: str | None = None
    faulty_claim_hint: str | None = None
    corrected_slot_key: str | None = None
    corrected_value: str | None = None
    submitted_at: datetime = Field(default_factory=utcnow)


class ContainmentAuditEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    audit_id: str = Field(default_factory=lambda: new_id("audit"))
    tenant_id: str
    root_memory_id: str
    anti_memory_id: str
    # derived_from descendants with no independent authoritative support
    quarantined_memory_ids: list[str] = Field(default_factory=list)
    # derived_from descendants that also rest on a clean verified parent: kept, but re-verify
    contested_memory_ids: list[str] = Field(default_factory=list)
    # derived_from descendants whose own evidence is authoritative: left ACTIVE, edge severed
    restored_memory_ids: list[str] = Field(default_factory=list)
    # exposure-only suspects (written in an exposed run, same subject, no declared link):
    # flagged for re-verification, NOT quarantined — exposure is not proof of dependence
    suspect_memory_ids: list[str] = Field(default_factory=list)
    # runs that were shown the refuted root or a quarantined descendant; their reports are flagged
    affected_run_ids: list[str] = Field(default_factory=list)
    llm_calls_used: int = 0
    created_at: datetime = Field(default_factory=utcnow)
    notes: str = ""


class WriteOutcome(StrEnum):
    INSERTED_ACTIVE = "inserted_active"
    INSERTED_CANDIDATE = "inserted_candidate"
    CORROBORATED = "corroborated"
    DEDUPLICATED_SOURCE = "deduplicated_source"
    CONTESTED = "contested"
    SUPERSEDED = "superseded"
    REJECTED_INJECTION = "rejected_injection"
    REJECTED_LOW_VALUE = "rejected_low_value"
    REJECTED_GLOBAL_BYPASS = "rejected_global_bypass"
    REJECTED_UNSUPPORTED_GLOBAL = "rejected_unsupported_global"
    EPHEMERAL_TASK_ONLY = "ephemeral_task_only"


class RunSessionContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    org_id: str
    user_id: str | None = None
    task_id: str
    task_text: str
    cohort: Cohort = Cohort.MEMORY
    started_at: datetime = Field(default_factory=utcnow)
    current_now: datetime = Field(default_factory=utcnow)
    initial_bundle: RecallBundle | None = None
    active_bundle: RecallBundle | None = None
    seen_query_tokens: set[str] = Field(default_factory=set)
    task_scratchpad: list[str] = Field(default_factory=list)
    observations: list[ToolObservation] = Field(default_factory=list)
    read_memory_ids: set[str] = Field(default_factory=set)
    cited_memory_ids: set[str] = Field(default_factory=set)
    step_critical_latencies_ms: list[float] = Field(default_factory=list)
    in_run_contradictions: int = 0
    tool_calls_count: int = 0
    tool_error_count: int = 0
    blocked_unsafe_actions: int = 0
    agent_llm_calls: int = 0
    memory_llm_calls: int = 0
    refresh_count: int = 0
    attestation_rejections: int = 0
