from __future__ import annotations

import hashlib
import hmac
import math
import os
import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any

from rapidfuzz import fuzz

from agentic_memory.models import (
    CandidateClaim,
    ClaimType,
    MemoryClaim,
    MemoryStatus,
    RecallItem,
    RiskClass,
    Scope,
    SourceProvenance,
    SourceType,
    TenantPolicy,
    ToolExecutionGateResult,
    ToolObservation,
    TrustTier,
    Volatility,
    content_digest,
    utcnow,
)

HALF_LIFE_DAYS: dict[Volatility, float | None] = {
    Volatility.HIGH: 14.0,
    Volatility.MEDIUM: 60.0,
    Volatility.LOW: 365.0,
    Volatility.PERMANENT: None,
}


class ToolCapability(StrEnum):
    READ = "read"
    INTERNAL_WRITE = "internal_write"
    EXTERNAL_WRITE = "external_write"
    CONFIG_WRITE = "config_write"


TOOL_CAPABILITIES: dict[str, ToolCapability] = {
    "read_ticket": ToolCapability.READ,
    "read_doc": ToolCapability.READ,
    "query_infra_api": ToolCapability.READ,
    "vendor_settlement_check": ToolCapability.READ,
    "query_internal_crm": ToolCapability.READ,
    "search_oss_kb": ToolCapability.READ,
    "add_internal_note": ToolCapability.INTERNAL_WRITE,
    "send_email": ToolCapability.EXTERNAL_WRITE,
    "escalate_ticket": ToolCapability.EXTERNAL_WRITE,
    "update_webhook": ToolCapability.EXTERNAL_WRITE,
    "change_billing_route": ToolCapability.CONFIG_WRITE,
}

TOOL_TO_CONNECTOR: dict[str, str] = {
    "read_ticket": "ticket_ingest",
    "read_doc": "tool_runner",
    "query_infra_api": "internal_infra",
    "vendor_settlement_check": "internal_infra",
    "query_internal_crm": "internal_crm",
    "search_oss_kb": "tool_runner",
    "add_internal_note": "tool_runner",
    "send_email": "tool_runner",
    "escalate_ticket": "tool_runner",
    "update_webhook": "tool_runner",
    "change_billing_route": "tool_runner",
}

CONNECTOR_DEFAULT_SOURCE_TYPE: dict[str, SourceType] = {
    "internal_crm": SourceType.INTERNAL_API,
    "internal_infra": SourceType.INTERNAL_API,
    "human_review": SourceType.HUMAN_FEEDBACK,
    "tool_runner": SourceType.TOOL_OUTPUT,
    "cross_org_synthesizer": SourceType.CROSS_ORG_CLUSTER,
    "agent_reasoner": SourceType.AGENT_INFERENCE,
    "ticket_ingest": SourceType.SUPPORT_TICKET,
    "email_ingest": SourceType.EMAIL,
    "web_fetch": SourceType.WEB_PAGE,
    "file_upload": SourceType.UPLOAD,
    "external_webhook": SourceType.WEB_PAGE,
}

CAPABILITY_SENSITIVE_TOOLS = frozenset(
    name
    for name, cap in TOOL_CAPABILITIES.items()
    if cap in (ToolCapability.EXTERNAL_WRITE, ToolCapability.CONFIG_WRITE)
)

EXTERNAL_UNTRUSTED_SOURCE_TYPES = frozenset(
    {
        SourceType.EMAIL,
        SourceType.SUPPORT_TICKET,
        SourceType.WEB_PAGE,
        SourceType.UPLOAD,
    }
)

VALID_STATUS_TRANSITIONS: dict[MemoryStatus, frozenset[MemoryStatus]] = {
    MemoryStatus.CANDIDATE_UNVERIFIED: frozenset(
        {
            MemoryStatus.CANDIDATE_UNVERIFIED,
            MemoryStatus.ACTIVE,
            MemoryStatus.CONTESTED,
            MemoryStatus.SUPERSEDED,
            MemoryStatus.QUARANTINED,
            MemoryStatus.REFUTED,
            MemoryStatus.ARCHIVED,
        }
    ),
    MemoryStatus.ACTIVE: frozenset(
        {
            MemoryStatus.ACTIVE,
            MemoryStatus.CONTESTED,
            MemoryStatus.SUPERSEDED,
            MemoryStatus.QUARANTINED,
            MemoryStatus.REFUTED,
            MemoryStatus.ARCHIVED,
        }
    ),
    MemoryStatus.CONTESTED: frozenset(
        {
            MemoryStatus.CONTESTED,
            MemoryStatus.ACTIVE,
            MemoryStatus.SUPERSEDED,
            MemoryStatus.QUARANTINED,
            MemoryStatus.REFUTED,
            MemoryStatus.ARCHIVED,
        }
    ),
    MemoryStatus.SUPERSEDED: frozenset({MemoryStatus.SUPERSEDED, MemoryStatus.REFUTED, MemoryStatus.ARCHIVED}),
    MemoryStatus.QUARANTINED: frozenset(
        {MemoryStatus.QUARANTINED, MemoryStatus.REFUTED, MemoryStatus.ARCHIVED}
    ),
    MemoryStatus.REFUTED: frozenset({MemoryStatus.REFUTED, MemoryStatus.ARCHIVED}),
    MemoryStatus.ARCHIVED: frozenset({MemoryStatus.ARCHIVED}),
}


def is_valid_status_transition(old_status: MemoryStatus, new_status: MemoryStatus) -> bool:
    return new_status in VALID_STATUS_TRANSITIONS.get(old_status, frozenset())


def assert_valid_status_transition(old_status: MemoryStatus, new_status: MemoryStatus) -> None:
    if not is_valid_status_transition(old_status, new_status):
        raise ValueError(f"Invalid memory status transition: {old_status.value} -> {new_status.value}")


@dataclass(frozen=True)
class ConnectorAuthority:
    connector_id: str
    max_trust_tier: TrustTier
    is_external_untrusted: bool = False
    # Predicate domains this connector is authoritative for (None = every domain). A connector
    # outside its domains is still AUTHENTIC (its seal verifies) but cannot verify the fact:
    # its claim is capped at TOOL_OBSERVED. See `predicate_domain`.
    authoritative_domains: frozenset[str] | None = None


class SourceAuthorityRegistry:
    """
    Server-side immutable source authority resolver.
    Trust tier is determined strictly by authenticated connector/source provenance,
    never by caller-supplied or LLM-supplied trust_tier payloads.
    """

    def __init__(
        self,
        connectors: dict[str, ConnectorAuthority] | None = None,
        *,
        attestation_secret: bytes | None = None,
        require_attestation: bool = True,
        attestation_keys: dict[str, bytes] | None = None,
        active_key_id: str | None = None,
        require_shared_keys: bool = False,
    ) -> None:
        self._connectors: dict[str, ConnectorAuthority] = dict(
            connectors
            or {
                "internal_crm": ConnectorAuthority(
                    "internal_crm", TrustTier.INTERNAL_API_VERIFIED, False,
                    authoritative_domains=frozenset({"contact", "customer"}),
                ),
                "internal_infra": ConnectorAuthority(
                    "internal_infra", TrustTier.INTERNAL_API_VERIFIED, False,
                    authoritative_domains=frozenset({"infrastructure", "customer"}),
                ),
                "human_review": ConnectorAuthority("human_review", TrustTier.HUMAN_VERIFIED, False),
                "tool_runner": ConnectorAuthority("tool_runner", TrustTier.TOOL_OBSERVED, False),
                "cross_org_synthesizer": ConnectorAuthority(
                    "cross_org_synthesizer", TrustTier.CROSS_ORG_CORROBORATED, False
                ),
                "agent_reasoner": ConnectorAuthority("agent_reasoner", TrustTier.AGENT_INFERRED, False),
                "ticket_ingest": ConnectorAuthority("ticket_ingest", TrustTier.UNTRUSTED_EXTERNAL, True),
                "email_ingest": ConnectorAuthority("email_ingest", TrustTier.UNTRUSTED_EXTERNAL, True),
                "web_fetch": ConnectorAuthority("web_fetch", TrustTier.UNTRUSTED_EXTERNAL, True),
                "file_upload": ConnectorAuthority("file_upload", TrustTier.UNTRUSTED_EXTERNAL, True),
                "external_webhook": ConnectorAuthority("external_webhook", TrustTier.UNTRUSTED_EXTERNAL, True),
            }
        )
        keys, active_kid, source = self._resolve_attestation_keys(
            attestation_secret=attestation_secret,
            attestation_keys=attestation_keys,
            active_key_id=active_key_id,
        )
        if require_shared_keys and source == "ephemeral_dev":
            raise ValueError(
                "Shared connector attestation keys are required: set MEMORY_ATTESTATION_KEYS "
                "('kid:hex,...', first entry active) or MEMORY_ATTESTATION_SECRET, or pass "
                "attestation_keys explicitly. A per-process random key makes seals minted by one "
                "worker unverifiable by every other worker (all evidence fails closed to untrusted)."
            )
        self._attestation_keys: dict[str, bytes] = keys
        self._active_kid: str = active_kid
        self.key_source: str = source
        self._require_attestation = require_attestation

    @staticmethod
    def _resolve_attestation_keys(
        *,
        attestation_secret: bytes | None,
        attestation_keys: dict[str, bytes] | None,
        active_key_id: str | None,
    ) -> tuple[dict[str, bytes], str, str]:
        """
        Key precedence: explicit key ring > explicit single secret > MEMORY_ATTESTATION_KEYS >
        MEMORY_ATTESTATION_SECRET > per-process random (development only).
        A key ring keeps retired keys verifiable during rotation; only the active key signs.
        """
        if attestation_keys:
            ring = {str(k): bytes(v) for k, v in attestation_keys.items()}
            kid = active_key_id or next(iter(ring))
            if kid not in ring:
                raise ValueError(f"active_key_id {kid!r} not in attestation key ring")
            return ring, kid, "explicit_key_ring"
        if attestation_secret:
            return {"k1": attestation_secret}, "k1", "explicit_secret"
        env_ring = os.environ.get("MEMORY_ATTESTATION_KEYS", "").strip()
        if env_ring:
            ring: dict[str, bytes] = {}
            for part in env_ring.split(","):
                kid, _, hexkey = part.strip().partition(":")
                if not kid or not hexkey:
                    raise ValueError("MEMORY_ATTESTATION_KEYS must be 'kid:hex[,kid:hex...]'")
                raw = bytes.fromhex(hexkey.strip())
                if len(raw) < 16:
                    raise ValueError(f"attestation key {kid!r} shorter than 16 bytes")
                ring[kid.strip()] = raw
            kid = active_key_id or next(iter(ring))
            return ring, kid, "env_key_ring"
        env_secret = os.environ.get("MEMORY_ATTESTATION_SECRET", "").strip()
        if env_secret:
            if len(env_secret) < 16:
                raise ValueError("MEMORY_ATTESTATION_SECRET must be >= 16 characters")
            return {"k1": env_secret.encode("utf-8")}, "k1", "env_secret"
        return {"dev": secrets.token_bytes(32)}, "dev", "ephemeral_dev"

    @classmethod
    def from_environment(cls, *, strict: bool = True) -> SourceAuthorityRegistry:
        """Production constructor: every worker loads the same managed key ring."""
        return cls(require_shared_keys=strict)

    def register_connector(
        self,
        connector_id: str,
        max_trust_tier: TrustTier,
        *,
        is_external_untrusted: bool = False,
    ) -> None:
        self._connectors[connector_id] = ConnectorAuthority(
            connector_id=connector_id,
            max_trust_tier=(
                TrustTier.UNTRUSTED_EXTERNAL if is_external_untrusted else max_trust_tier
            ),
            is_external_untrusted=is_external_untrusted,
        )

    def get_connector(self, connector_id: str) -> ConnectorAuthority | None:
        return self._connectors.get(connector_id)

    @staticmethod
    def default_source_type_for_connector(connector_id: str) -> SourceType:
        return CONNECTOR_DEFAULT_SOURCE_TYPE.get(connector_id, SourceType.WEB_PAGE)

    # --- seals ---------------------------------------------------------------------------
    # A seal binds: key id, connector, source type, source id, external flag, origin domain,
    # TENANT, RUN and a SHA-256 CONTENT DIGEST. Verification recomputes the HMAC over exactly
    # those fields and fails closed if any context field is missing, so a seal minted for one
    # tenant, run or payload cannot be replayed on another, and a context-free seal is useless.

    def attest(
        self,
        *,
        connector_id: str,
        source_type: SourceType,
        source_id: str,
        tenant_id: str,
        run_id: str,
        content_digest: str,
        is_external_untrusted: bool = False,
        origin_domain: str | None = None,
    ) -> str:
        if not (tenant_id and run_id and content_digest):
            raise ValueError("attestation requires tenant_id, run_id and content_digest")
        return self._sign(
            self._active_kid,
            connector_id=connector_id,
            source_type=source_type,
            source_id=source_id,
            is_external_untrusted=is_external_untrusted,
            origin_domain=origin_domain,
            tenant_id=tenant_id,
            run_id=run_id,
            content_digest=content_digest,
        )

    def _sign(
        self,
        kid: str,
        *,
        connector_id: str,
        source_type: SourceType,
        source_id: str,
        is_external_untrusted: bool,
        origin_domain: str | None,
        tenant_id: str,
        run_id: str,
        content_digest: str,
    ) -> str:
        msg = "|".join(
            [
                kid,
                connector_id,
                source_type.value,
                source_id,
                str(int(bool(is_external_untrusted))),
                (origin_domain or "").strip().lower(),
                tenant_id,
                run_id,
                content_digest,
            ]
        ).encode()
        return f"{kid}:{hmac.new(self._attestation_keys[kid], msg, hashlib.sha256).hexdigest()}"

    def verify_attestation(
        self,
        *,
        connector_id: str | None,
        source_type: SourceType,
        source_id: str | None,
        is_external_untrusted: bool,
        origin_domain: str | None = None,
        authority_seal: str | None = None,
        tenant_id: str | None = None,
        run_id: str | None = None,
        content_digest: str | None = None,
        expected_tenant_id: str | None = None,
        expected_run_id: str | None = None,
    ) -> bool:
        # Missing context is never accepted
        if not (connector_id and authority_seal and source_id and tenant_id and run_id and content_digest):
            return False
        # Exact equality with the caller's context (when the caller knows it)
        if expected_tenant_id is not None and tenant_id != expected_tenant_id:
            return False
        if expected_run_id is not None and run_id != expected_run_id:
            return False
        kid, sep, _ = authority_seal.partition(":")
        if not sep or kid not in self._attestation_keys:
            return False  # unknown / retired-and-removed key id fails closed
        expected = self._sign(
            kid,
            connector_id=connector_id,
            source_type=source_type,
            source_id=source_id,
            is_external_untrusted=is_external_untrusted,
            origin_domain=origin_domain,
            tenant_id=tenant_id,
            run_id=run_id,
            content_digest=content_digest,
        )
        return hmac.compare_digest(expected, authority_seal)

    def build_trusted_provenance(
        self,
        *,
        connector_id: str,
        source_id: str,
        run_id: str,
        tenant_id: str,
        content: str,
        source_type: SourceType | None = None,
        source_domain: str | None = None,
        origin_domain: str | None = None,
        observed_at: datetime | None = None,
        tool_capability: str | None = None,
    ) -> SourceProvenance:
        """
        Provenance for a claim written directly by a connector (CRM sync, human review).
        `content` is the claim VALUE the connector vouches for; the store re-checks the digest
        against the candidate's value, so the seal cannot be reused for a substituted value.
        """
        auth = self._connectors.get(connector_id)
        if auth is None:
            raise ValueError(f"Unknown connector_id: {connector_id}")
        eff_source_type = source_type or self.default_source_type_for_connector(connector_id)
        is_ext = auth.is_external_untrusted or (eff_source_type in EXTERNAL_UNTRUSTED_SOURCE_TYPES)
        eff_origin = origin_domain if origin_domain is not None else (source_domain or "")
        digest = content_digest(content.strip())
        seal = self.attest(
            connector_id=connector_id,
            source_type=eff_source_type,
            source_id=source_id,
            is_external_untrusted=is_ext,
            origin_domain=eff_origin,
            tenant_id=tenant_id,
            run_id=run_id,
            content_digest=digest,
        )
        return SourceProvenance(
            source_id=source_id,
            source_type=eff_source_type,
            source_domain=source_domain,
            origin_domain=eff_origin,
            is_external_untrusted=is_ext,
            run_id=run_id,
            observed_at=observed_at or utcnow(),
            connector_id=connector_id,
            authority_seal=seal,
            tool_capability=tool_capability,
            tenant_id=tenant_id,
            content_digest=digest,
            content_kind="claim_value",
        )

    def build_trusted_observation(
        self,
        *,
        step_idx: int,
        tool_name: str,
        source_id: str,
        content: str,
        tenant_id: str,
        run_id: str,
        connector_id: str | None = None,
        source_type: SourceType | None = None,
        source_domain: str | None = None,
        origin_domain: str | None = None,
        structured_facts: list[dict[str, Any]] | None = None,
        observed_at: datetime | None = None,
        tool_capability: str | None = None,
    ) -> ToolObservation:
        """Seal a tool result for exactly one tenant, one run and its exact content (+ structured facts)."""
        eff_conn = connector_id or TOOL_TO_CONNECTOR.get(tool_name, "external_webhook")
        auth = self._connectors.get(eff_conn)
        if auth is None:
            eff_conn = "external_webhook"
            auth = self._connectors[eff_conn]
        eff_source_type = source_type or self.default_source_type_for_connector(eff_conn)
        is_ext = auth.is_external_untrusted or (eff_source_type in EXTERNAL_UNTRUSTED_SOURCE_TYPES)
        eff_origin = origin_domain if origin_domain is not None else (source_domain or "")
        facts = list(structured_facts or [])
        digest = observation_digest(content, facts)
        seal = self.attest(
            connector_id=eff_conn,
            source_type=eff_source_type,
            source_id=source_id,
            is_external_untrusted=is_ext,
            origin_domain=eff_origin,
            tenant_id=tenant_id,
            run_id=run_id,
            content_digest=digest,
        )
        return ToolObservation(
            step_idx=step_idx,
            tool_name=tool_name,
            source_id=source_id,
            source_type=eff_source_type,
            source_domain=source_domain,
            origin_domain=eff_origin,
            is_external_untrusted=is_ext,
            content=content,
            structured_facts=facts,
            observed_at=observed_at or utcnow(),
            connector_id=eff_conn,
            authority_seal=seal,
            tool_capability=tool_capability,
            tenant_id=tenant_id,
            run_id=run_id,
            content_digest=digest,
        )

    def verify_observation(
        self, obs: ToolObservation, *, expected_tenant_id: str, expected_run_id: str
    ) -> bool:
        """Ingestion check: seal valid, bound to THIS tenant and run, and content unmodified."""
        if obs.content_digest != observation_digest(obs.content, obs.structured_facts):
            return False
        return self.verify_attestation(
            connector_id=obs.connector_id,
            source_type=obs.source_type,
            source_id=obs.source_id,
            is_external_untrusted=obs.is_external_untrusted,
            origin_domain=obs.origin_domain if obs.origin_domain is not None else obs.source_domain,
            authority_seal=obs.authority_seal,
            tenant_id=obs.tenant_id,
            run_id=obs.run_id,
            content_digest=obs.content_digest,
            expected_tenant_id=expected_tenant_id,
            expected_run_id=expected_run_id,
        )

    @staticmethod
    def _default_for_source_type(source_type: SourceType) -> TrustTier:
        if source_type in EXTERNAL_UNTRUSTED_SOURCE_TYPES:
            return TrustTier.UNTRUSTED_EXTERNAL
        if source_type == SourceType.HUMAN_FEEDBACK:
            return TrustTier.HUMAN_VERIFIED
        if source_type == SourceType.INTERNAL_API:
            return TrustTier.INTERNAL_API_VERIFIED
        if source_type == SourceType.CROSS_ORG_CLUSTER:
            return TrustTier.CROSS_ORG_CORROBORATED
        if source_type == SourceType.TOOL_OUTPUT:
            return TrustTier.TOOL_OBSERVED
        return TrustTier.AGENT_INFERRED

    def is_authoritative_for(self, connector_id: str | None, subject: str, predicate: str, value: str = "") -> bool:
        """Source authenticity is not predicate authority: a CRM can verify contacts, not DB versions."""
        auth = self._connectors.get(connector_id or "")
        if auth is None or auth.authoritative_domains is None:
            return True
        return predicate_domain(subject, predicate, value) in auth.authoritative_domains

    def resolve(
        self,
        *,
        connector_id: str | None,
        source_type: SourceType,
        is_external_untrusted: bool,
        source_id: str | None = None,
        origin_domain: str | None = None,
        authority_seal: str | None = None,
        tenant_id: str | None = None,
        run_id: str | None = None,
        content_digest: str | None = None,
        expected_tenant_id: str | None = None,
        expected_run_id: str | None = None,
    ) -> TrustTier:
        if is_external_untrusted or source_type in EXTERNAL_UNTRUSTED_SOURCE_TYPES:
            return TrustTier.UNTRUSTED_EXTERNAL
        if not connector_id:
            return TrustTier.UNTRUSTED_EXTERNAL
        auth = self._connectors.get(connector_id)
        if auth is None or auth.is_external_untrusted:
            return TrustTier.UNTRUSTED_EXTERNAL
        if self._require_attestation and not self.verify_attestation(
            connector_id=connector_id,
            source_type=source_type,
            source_id=source_id,
            is_external_untrusted=is_external_untrusted,
            origin_domain=origin_domain,
            authority_seal=authority_seal,
            tenant_id=tenant_id,
            run_id=run_id,
            content_digest=content_digest,
            expected_tenant_id=expected_tenant_id,
            expected_run_id=expected_run_id,
        ):
            return TrustTier.UNTRUSTED_EXTERNAL
        base = self._default_for_source_type(source_type)
        return TrustTier(min(int(base), int(auth.max_trust_tier)))

    def resolve_observation(
        self, obs: ToolObservation, *, expected_tenant_id: str | None = None,
        expected_run_id: str | None = None,
    ) -> TrustTier:
        raw_ext = obs.is_external_untrusted or (obs.source_type in EXTERNAL_UNTRUSTED_SOURCE_TYPES)
        if obs.content_digest != observation_digest(obs.content, obs.structured_facts):
            return TrustTier.UNTRUSTED_EXTERNAL
        return self.resolve(
            connector_id=obs.connector_id,
            source_type=obs.source_type,
            is_external_untrusted=raw_ext,
            source_id=obs.source_id,
            origin_domain=obs.origin_domain if obs.origin_domain is not None else obs.source_domain,
            authority_seal=obs.authority_seal,
            tenant_id=obs.tenant_id,
            run_id=obs.run_id,
            content_digest=obs.content_digest,
            expected_tenant_id=expected_tenant_id,
            expected_run_id=expected_run_id,
        )

    def provenance_from_observation(self, obs: ToolObservation, *, run_id: str) -> SourceProvenance:
        """Carry the observation's sealed context onto a claim extracted from it."""
        return SourceProvenance(
            source_id=obs.source_id,
            source_type=obs.source_type,
            source_domain=obs.source_domain,
            origin_domain=obs.origin_domain if obs.origin_domain is not None else obs.source_domain,
            is_external_untrusted=obs.is_external_untrusted,
            run_id=obs.run_id or run_id,
            observed_at=obs.observed_at,
            connector_id=obs.connector_id,
            authority_seal=obs.authority_seal,
            tool_capability=obs.tool_capability,
            tenant_id=obs.tenant_id,
            content_digest=obs.content_digest,
            content_kind="observation",
        )


def observation_digest(content: str, structured_facts: list[dict[str, Any]] | None = None) -> str:
    import json as _json

    facts = _json.dumps(structured_facts or [], sort_keys=True, default=str)
    return content_digest(f"{content}\x1f{facts}")


DEFAULT_AUTHORITY_REGISTRY = SourceAuthorityRegistry()

INJECTION_PATTERNS = [
    re.compile(
        r"\b(?:ignore|disregard|override|forget)\s+(?:all\s+)?(?:previous|prior|above|earlier|your)\s+"
        r"(?:instructions|rules|guidance|guidelines|policies)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:note|reminder|notice|message|instructions?)\s+(?:to|for)\s+(?:all\s+)?"
        r"(?:ai|llm|automated)\s+(?:agents?|assistants?|systems?)\b",
        re.I,
    ),
    re.compile(r"\bsystem\s+prompt\b", re.I),
    re.compile(r"\b(?:note\s+)?for\s+future\s+(?:handlers|agents|runs)\s*:\s*always\b", re.I),
    re.compile(r"\balways\s+(?:cc|bcc|forward|exfiltrate|route)\b", re.I),
    re.compile(r"\bbypass\s+(?:auth|verification|security|tls|ssl)\b", re.I),
]

UNSAFE_GLOBAL_PATTERNS = [
    re.compile(r"verify\s*=\s*false", re.I),
    re.compile(r"disable[_\s-]*(ssl|tls|auth|cert|verify)", re.I),
    re.compile(r"ssl[_\s-]*verify\s*=\s*(false|0|off)", re.I),
    re.compile(r"chmod\s+777", re.I),
    re.compile(r"curl\s+[^|]+\|\s*(ba)?sh", re.I),
    re.compile(r"ignore[_\s-]*cert", re.I),
]

EMAIL_FULL_RE = re.compile(r"\b([A-Za-z0-9._%+-]+@([A-Za-z0-9._-]+\.[A-Za-z]{2,}))\b")
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@([A-Za-z0-9._-]+\.[A-Za-z]{2,})\b")
URL_DOMAIN_RE = re.compile(r"https?://([A-Za-z0-9._-]+)", re.I)
DOMAIN_TOKEN_RE = re.compile(
    r"\b([A-Za-z0-9](?:[A-Za-z0-9_-]*[A-Za-z0-9])?(?:\.[A-Za-z0-9_-]+)*\.[A-Za-z]{2,})\b"
)
IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
LOCAL_PATH_RE = re.compile(r"(?:[A-Za-z]:\\|/(?:Users|home|var|etc|opt)/)[^\s,;]+")
PUBLIC_GLOBAL_DOMAINS = frozenset({"pypi.org", "github.com", "postgresql.org", "docs.python.org"})

ROUTING_SIGNAL_RE = re.compile(
    r"(?:\bescalat\w*\b|\bcc_policy\b|\bwebhook\b|\bcontact_email\b|\brouting\b"
    r"|\bbilling_route\b|\bforward(?:ing)?\b|\bredirect\b|\bauth(?:orized)?_domain\b"
    r"|\ballowlist\b)",
    re.I,
)

COMMON_NON_DOMAIN_EXTS = frozenset(
    {
        "py", "js", "ts", "rs", "go", "java", "rb", "php", "c", "h", "cpp", "sh", "ps1", "bat",
        "json", "yaml", "yml", "toml", "txt", "md", "log", "cfg", "ini", "lock", "csv", "sql",
        "xml", "html", "css", "env", "pem", "key", "crt", "p12", "pdf", "png", "jpg", "jpeg",
        "gif", "svg", "webp", "zip", "tar", "gz", "tgz", "bz2", "xz", "exe", "dll", "so", "dylib",
        "whl", "doc", "docx", "xlsx", "parquet", "avro", "db", "sqlite", "out", "err",
    }
)


def _looks_like_domain_token(dom: str) -> bool:
    """Return True if a dotted token looks like a DNS hostname rather than a file or code symbol."""
    if "." not in dom:
        return False
    tld = dom.rsplit(".", 1)[-1].lower()
    if tld in COMMON_NON_DOMAIN_EXTS:
        return False
    if dom in PUBLIC_GLOBAL_DOMAINS:
        return True
    # DNS TLDs are lowercase 2-6 alpha chars (plus special local/test suffixes internal/example)
    return (2 <= len(tld) <= 6 and tld.isalpha()) or tld in {"internal", "example"}

SECRET_RE = re.compile(
    r"\b(?:sk-[A-Za-z0-9]{10,}|ghp_[A-Za-z0-9]{10,}|AKIA[0-9A-Z]{12,}|xox[baprs]-[A-Za-z0-9-]{10,}|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})\b"
)

SUBJECT_ALIASES = {
    "production_database": "prod_db",
    "production_db": "prod_db",
    "prod_database": "prod_db",
    "postgres_database": "prod_db",
    "primary_database": "prod_db",
    "escalation_email": "escalation_contact",
    "escalation_address": "escalation_contact",
    "billing_escalation": "escalation_contact",
    "vendor_y": "vendor_y_api",
    "vendor_y_service": "vendor_y_api",
}

PREDICATE_ALIASES = {
    "db_version": "engine_version",
    "postgres_version": "engine_version",
    "database_version": "engine_version",
    "email": "contact_email",
    "address": "contact_email",
    "timezone": "date_timezone",
    "returns_dates_in": "date_timezone",
    "timestamp_timezone": "date_timezone",
}


class TokenCounter:
    """
    Model-aware token budget helper.
    Uses a local HuggingFace `tokenizers` JSON file when configured (offline-safe),
    and otherwise falls back to a calibrated conservative BPE upper bound.
    """

    def __init__(self, tokenizer_json_path: str | None = None) -> None:
        self._tokenizer = None
        if tokenizer_json_path:
            try:
                from tokenizers import Tokenizer  # type: ignore[import-not-found]

                self._tokenizer = Tokenizer.from_file(tokenizer_json_path)
            except Exception:
                self._tokenizer = None

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._tokenizer is not None:
            return len(self._tokenizer.encode(text).ids)
        return estimate_tokens(text)


def estimate_tokens(text: str) -> int:
    """Conservative BPE-safe token estimate (>= real token count for English/code)."""
    if not text:
        return 0
    words = len(text.split())
    by_words = math.ceil(words / 0.72)
    by_chars = math.ceil(len(text) / 3.4)
    return max(by_words, by_chars, 1)


def normalize_token(raw: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", raw.strip().lower()).strip("_")
    return cleaned or "unknown"


def canonicalize_subject(subject: str) -> str:
    norm = normalize_token(subject)
    return SUBJECT_ALIASES.get(norm, norm)


def canonicalize_predicate(predicate: str) -> str:
    norm = normalize_token(predicate)
    return PREDICATE_ALIASES.get(norm, norm)


def canonicalize_slot(
    scope: Scope,
    tenant_id: str,
    subject: str,
    predicate: str,
    user_id: str | None = None,
    task_id: str | None = None,
) -> str:
    subj = canonicalize_subject(subject)
    pred = canonicalize_predicate(predicate)
    if scope == Scope.GLOBAL:
        return f"global|{subj}|{pred}"
    if scope == Scope.USER and user_id:
        return f"org:{normalize_token(tenant_id)}|user:{normalize_token(user_id)}|{subj}|{pred}"
    if scope == Scope.TASK and task_id:
        return f"org:{normalize_token(tenant_id)}|task:{normalize_token(task_id)}|{subj}|{pred}"
    return f"org:{normalize_token(tenant_id)}|{subj}|{pred}"


def value_digest(value: str) -> str:
    normalized = " ".join(value.strip().lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def hash_source_id(source_id: str) -> str:
    return hashlib.sha256(source_id.strip().encode("utf-8")).hexdigest()[:16]


def org_hmac(secret_salt: bytes, org_id: str) -> str:
    return hmac.new(secret_salt, org_id.encode("utf-8"), hashlib.sha256).hexdigest()


def extract_domains(text: str) -> list[str]:
    domains = [m.group(1).lower() for m in EMAIL_RE.finditer(text)]
    domains.extend(m.group(1).lower() for m in URL_DOMAIN_RE.finditer(text))
    # Preserve order while deduplicating
    seen: set[str] = set()
    ordered: list[str] = []
    for d in domains:
        if d not in seen:
            seen.add(d)
            ordered.append(d)
    return ordered


def trust_confidence_ceiling(claim: MemoryClaim) -> float:
    if claim.status in (MemoryStatus.REFUTED, MemoryStatus.ARCHIVED):
        return 0.0
    if claim.status == MemoryStatus.CANDIDATE_UNVERIFIED:
        return 0.25
    if claim.trust_tier == TrustTier.UNTRUSTED_EXTERNAL:
        if claim.risk_class == RiskClass.HIGH_ACTION_ROUTING:
            return 0.25
        if claim.risk_class == RiskClass.LOW_DIAGNOSTIC and claim.metadata.get("error_signature"):
            return 0.65
        return 0.40
    if claim.trust_tier == TrustTier.AGENT_INFERRED:
        # Single-run agent conclusions remain provisional until corroborated
        return 0.50 if claim.evidence_count <= 1 else 0.72
    if claim.trust_tier == TrustTier.TOOL_OBSERVED:
        return 0.90
    if claim.trust_tier == TrustTier.CROSS_ORG_CORROBORATED:
        return 0.88
    if claim.trust_tier == TrustTier.INTERNAL_API_VERIFIED:
        return 0.96
    return 1.0


def effective_confidence(claim: MemoryClaim, now: datetime) -> float:
    if claim.status in (MemoryStatus.REFUTED, MemoryStatus.ARCHIVED):
        return 0.0
    if claim.status == MemoryStatus.QUARANTINED:
        return 0.0

    age_days = max(0.0, (now - claim.updated_at).total_seconds() / 86400.0)
    half_life = HALF_LIFE_DAYS[claim.volatility]
    decay = 1.0 if half_life is None else 0.5 ** (age_days / half_life)

    # Anti-self-corroboration: untrusted routing claims gain zero boost from repetition
    if (
        claim.trust_tier == TrustTier.UNTRUSTED_EXTERNAL
        and claim.risk_class == RiskClass.HIGH_ACTION_ROUTING
    ):
        corrob_factor = 1.0
    else:
        corrob_factor = 1.0 + 0.08 * math.log2(max(1, claim.evidence_count))

    contradiction_penalty = 0.65**claim.contradiction_hits
    raw = claim.base_confidence * decay * corrob_factor * contradiction_penalty
    ceiling = trust_confidence_ceiling(claim)
    return round(min(ceiling, max(0.0, raw)), 4)


# --- Confidence, freshness and actionability are different questions ----------------------
#   confidence    how trustworthy is the evidence?            (trust tier, corroboration, contradictions)
#   freshness     how recently was the fact last confirmed?   (age vs a risk-dependent window)
#   actionability may it be used for THIS operation?          (status + trust + freshness + use)
# `effective_confidence` above blends confidence with age and is used only to RANK recall.
# A Postgres version confirmed by the infra API 185 days ago has high confidence and low freshness.

class MemoryUse(StrEnum):
    CONTEXT = "context"  # shown to the agent, labelled
    STATE_AS_FACT = "state_as_fact"  # written into an answer or report
    AUTHORIZE_ACTION = "authorize_action"  # backs an external write / config change


REVALIDATE_AFTER_DAYS_BY_RISK: dict[RiskClass, float] = {
    RiskClass.HIGH_ACTION_ROUTING: 30.0,
    RiskClass.MEDIUM_OPERATIONAL: 90.0,
    RiskClass.LOW_DIAGNOSTIC: 180.0,
}
REVALIDATE_AFTER_DAYS_BY_VOLATILITY: dict[Volatility, float | None] = {
    Volatility.HIGH: 14.0,
    Volatility.MEDIUM: 60.0,
    Volatility.LOW: 365.0,
    Volatility.PERMANENT: None,
}


def last_confirmed_at(claim: MemoryClaim) -> datetime:
    """When a source last re-observed this value (not merely when the row was touched)."""
    raw = (claim.metadata or {}).get("last_confirmed_at")
    if raw:
        try:
            return datetime.fromisoformat(str(raw))
        except ValueError:
            pass
    return claim.created_at


def revalidation_window_days(claim: MemoryClaim) -> float | None:
    """Risk-dependent: the more a wrong value could hurt, the sooner it must be re-confirmed."""
    if claim.claim_type == ClaimType.NEGATIVE_CONSTRAINT or claim.volatility == Volatility.PERMANENT:
        return None
    return min(
        REVALIDATE_AFTER_DAYS_BY_RISK[claim.risk_class],
        REVALIDATE_AFTER_DAYS_BY_VOLATILITY[claim.volatility] or float("inf"),
    )


def claim_freshness(claim: MemoryClaim, now: datetime) -> dict[str, Any]:
    age = max(0.0, (now - last_confirmed_at(claim)).total_seconds() / 86400.0)
    window = revalidation_window_days(claim)
    return {
        "age_days": round(age, 1),
        "window_days": window,
        "stale": window is not None and age > window,
    }


def evidence_confidence(claim: MemoryClaim) -> float:
    """Trustworthiness of the evidence alone — no time decay."""
    if claim.status in (MemoryStatus.REFUTED, MemoryStatus.ARCHIVED, MemoryStatus.QUARANTINED):
        return 0.0
    if claim.trust_tier == TrustTier.UNTRUSTED_EXTERNAL and claim.risk_class == RiskClass.HIGH_ACTION_ROUTING:
        corrob = 1.0
    else:
        corrob = 1.0 + 0.08 * math.log2(max(1, claim.evidence_count))
    raw = claim.base_confidence * corrob * (0.65**claim.contradiction_hits)
    return round(min(trust_confidence_ceiling(claim), max(0.0, raw)), 4)


def claim_actionability(claim: MemoryClaim, now: datetime, use: MemoryUse) -> tuple[bool, str]:
    if use == MemoryUse.CONTEXT:
        ok = claim.status in (MemoryStatus.ACTIVE, MemoryStatus.CONTESTED, MemoryStatus.CANDIDATE_UNVERIFIED)
        return ok, "recallable" if ok else f"status_{claim.status.value.lower()}"
    if claim.status != MemoryStatus.ACTIVE:
        return False, f"status_{claim.status.value.lower()}"
    if claim.claim_type == ClaimType.NEGATIVE_CONSTRAINT:
        return False, "anti_memory_is_a_veto_not_a_fact"
    if (claim.metadata or {}).get("requires_verification"):
        return False, "flagged_for_reverification"
    if claim_freshness(claim, now)["stale"]:
        return False, "stale_revalidate_first"
    if use == MemoryUse.AUTHORIZE_ACTION and claim.trust_tier < TrustTier.INTERNAL_API_VERIFIED:
        return False, "action_needs_internal_api_or_human_evidence"
    return True, "ok"


def detect_prompt_injection(text: str) -> tuple[bool, str | None]:
    for pat in INJECTION_PATTERNS:
        m = pat.search(text)
        if m:
            return True, f"explicit_directive_injection:{m.group(0)}"
    return False, None


def is_domain_authorized(domain: str | None, tenant_policy: TenantPolicy) -> bool:
    if not domain:
        return True
    d = domain.lower().strip(".")
    for allowed in tenant_policy.authorized_domains:
        a = allowed.lower().strip(".")
        if d == a or d.endswith("." + a):
            return True
    return False


def infer_minimum_risk_class(
    subject: str,
    predicate: str,
    value: str,
    statement: str,
    requested: RiskClass | None = None,
) -> RiskClass:
    """
    Fail-closed risk classifier:
    - Omitted/invalid risk_class defaults to HIGH_ACTION_ROUTING.
    - Any email, non-public URL, or routing/policy indicator upgrades to HIGH_ACTION_ROUTING.
    """
    blob = f"{subject} {predicate} {value} {statement}"
    if EMAIL_FULL_RE.search(blob) or ROUTING_SIGNAL_RE.search(f"{subject} {predicate}"):
        return RiskClass.HIGH_ACTION_ROUTING
    for m in URL_DOMAIN_RE.finditer(f"{value} {statement}"):
        if m.group(1).lower() not in PUBLIC_GLOBAL_DOMAINS:
            return RiskClass.HIGH_ACTION_ROUTING
    if requested is not None:
        return requested
    return RiskClass.HIGH_ACTION_ROUTING


_CONTACT_PREDICATE_RE = re.compile(r"(?:escalat|contact|owner|assignee|on_?call|recipient|forward|redirect)", re.I)
_INFRA_PREDICATE_RE = re.compile(
    r"(?:version|engine|timezone|tz\b|host|port|endpoint|region|cluster|database|\bdb\b|schema|"
    r"config|runtime|deploy|ssl|tls|cert|route|routing|webhook|allowlist|auth_?domain|cc_policy)",
    re.I,
)


def predicate_domain(subject: str, predicate: str, value: str = "") -> str:
    """
    Coarse fact domain used for connector authority: 'contact' (who to notify or escalate to),
    'infrastructure' (systems, versions, routes, configuration) or 'customer' (everything else).
    """
    sp = f"{subject} {predicate}"
    if _CONTACT_PREDICATE_RE.search(sp) or EMAIL_FULL_RE.search(value):
        return "contact"
    if _INFRA_PREDICATE_RE.search(sp):
        return "infrastructure"
    return "customer"


def normalize_candidate_authority(
    candidate: CandidateClaim,
    registry: SourceAuthorityRegistry | None = None,
    *,
    expected_tenant_id: str | None = None,
) -> CandidateClaim:
    """
    Enforce immutable source-derived trust and untrusted-input boundaries.
    External/untrusted inputs cannot spoof higher trust tiers, self-tag public artifacts,
    write Scope.GLOBAL directly, or downgrade routing risk classes.
    The provenance seal must verify for the writing tenant; a direct connector write must also
    carry the digest of exactly this value.
    """
    reg = registry or DEFAULT_AUTHORITY_REGISTRY
    prov = candidate.provenance
    raw_ext = prov.is_external_untrusted or (prov.source_type in EXTERNAL_UNTRUSTED_SOURCE_TYPES)
    value_bound = prov.content_kind != "claim_value" or (
        prov.content_digest == content_digest(candidate.value.strip())
    )
    resolved_trust = (
        reg.resolve(
            connector_id=prov.connector_id,
            source_type=prov.source_type,
            is_external_untrusted=raw_ext,
            source_id=prov.source_id,
            origin_domain=prov.origin_domain if prov.origin_domain is not None else prov.source_domain,
            authority_seal=prov.authority_seal,
            tenant_id=prov.tenant_id,
            run_id=prov.run_id,
            content_digest=prov.content_digest,
            expected_tenant_id=expected_tenant_id,
        )
        if value_bound and prov.content_kind in ("claim_value", "observation")
        else TrustTier.UNTRUSTED_EXTERNAL
    )
    is_ext = raw_ext or (resolved_trust == TrustTier.UNTRUSTED_EXTERNAL)
    eff_trust = (
        TrustTier.UNTRUSTED_EXTERNAL
        if is_ext
        else TrustTier(min(int(candidate.trust_tier), int(resolved_trust)))
    )
    authority_capped = False
    if (
        not is_ext
        and eff_trust >= TrustTier.INTERNAL_API_VERIFIED
        and not reg.is_authoritative_for(prov.connector_id, candidate.subject, candidate.predicate, candidate.value)
    ):
        eff_trust = TrustTier(min(int(eff_trust), int(TrustTier.TOOL_OBSERVED)))
        authority_capped = True
    eff_scope = Scope.ORG if (is_ext and candidate.scope == Scope.GLOBAL) else candidate.scope
    eff_public = False if is_ext else candidate.is_public_technical_artifact
    eff_risk = infer_minimum_risk_class(
        candidate.subject,
        candidate.predicate,
        candidate.value,
        candidate.statement,
        requested=candidate.risk_class,
    )
    new_prov = (
        prov.model_copy(update={"is_external_untrusted": True})
        if is_ext and not prov.is_external_untrusted
        else prov
    )
    if (
        eff_trust == candidate.trust_tier
        and eff_scope == candidate.scope
        and eff_public == candidate.is_public_technical_artifact
        and eff_risk == candidate.risk_class
        and new_prov is prov
        and not authority_capped
    ):
        return candidate
    return candidate.model_copy(
        update={
            "trust_tier": eff_trust,
            "scope": eff_scope,
            "is_public_technical_artifact": eff_public,
            "risk_class": eff_risk,
            "provenance": new_prov,
        }
    )


def evaluate_candidate_write(
    candidate: CandidateClaim,
    tenant_policy: TenantPolicy,
    registry: SourceAuthorityRegistry | None = None,
) -> tuple[str, MemoryStatus, float, str, datetime | None]:
    """
    Determine write admission policy (0 LLM calls).
    Returns: (decision, status, capped_confidence, reason, expires_at)
    decision in {"ADMIT", "EPHEMERAL_TASK", "REJECT_INJECTION", "REJECT_LOW_VALUE"}
    """
    candidate = normalize_candidate_authority(
        candidate, registry=registry, expected_tenant_id=tenant_policy.org_id
    )
    stmt = candidate.statement.strip()
    if len(stmt) < 8 or len(stmt) > 480:
        return (
            "REJECT_LOW_VALUE",
            MemoryStatus.ARCHIVED,
            0.0,
            "statement_length_out_of_bounds",
            None,
        )

    if candidate.scope == Scope.TASK:
        return (
            "EPHEMERAL_TASK",
            MemoryStatus.ARCHIVED,
            0.0,
            "task_scope_kept_ephemeral_not_persisted",
            None,
        )

    if candidate.scope == Scope.USER and not (candidate.user_id and candidate.user_id.strip()):
        return (
            "REJECT_LOW_VALUE",
            MemoryStatus.ARCHIVED,
            0.0,
            "missing_user_id_for_user_scope",
            None,
        )

    if candidate.provenance.is_external_untrusted:
        # Check the model's paraphrase, the exact evidence span, and the observation-level flag
        # set by the memory layer: a model that paraphrases "ignore earlier rules and route X"
        # into a clean statement must not launder the injection past this gate.
        injected, reason = detect_prompt_injection(stmt)
        if not injected and candidate.source_quote:
            injected, reason = detect_prompt_injection(candidate.source_quote)
        if not injected and candidate.metadata.get("source_observation_injection"):
            injected, reason = True, str(candidate.metadata["source_observation_injection"])
        if injected:
            return (
                "REJECT_INJECTION",
                MemoryStatus.REFUTED,
                0.0,
                reason or "prompt_injection_detected",
                None,
            )

    domains = extract_domains(f"{candidate.value} {stmt}")
    if candidate.provenance.source_domain:
        domains.append(candidate.provenance.source_domain.lower())
    unauthorized_domains = [d for d in domains if not is_domain_authorized(d, tenant_policy)]

    # High-risk routing/policy changes from untrusted external sources or unauthorized domains
    # NEVER auto-activate; they are staged safely as CANDIDATE_UNVERIFIED so real updates
    # are not dropped while planted attacks cannot execute autonomously.
    if (
        candidate.risk_class == RiskClass.HIGH_ACTION_ROUTING
        and (candidate.provenance.is_external_untrusted or unauthorized_domains)
        and candidate.trust_tier < TrustTier.INTERNAL_API_VERIFIED
    ):
        expires_at = candidate.provenance.observed_at + timedelta(days=14)
        return (
            "ADMIT",
            MemoryStatus.CANDIDATE_UNVERIFIED,
            min(candidate.base_confidence, 0.22),
            "untrusted_or_off_domain_routing_staged_for_verification",
            expires_at,
        )

    # Single-run agent inferences without direct tool/API proof stay cautious
    if candidate.trust_tier == TrustTier.AGENT_INFERRED:
        return (
            "ADMIT",
            MemoryStatus.ACTIVE,
            min(candidate.base_confidence, 0.48),
            "single_run_inference_capped_provisional",
            None,
        )

    if candidate.trust_tier == TrustTier.UNTRUSTED_EXTERNAL:
        # Technical diagnostics backed by an observable error signature may contest stale facts
        if candidate.risk_class == RiskClass.LOW_DIAGNOSTIC and candidate.error_signature:
            return (
                "ADMIT",
                MemoryStatus.ACTIVE,
                min(candidate.base_confidence, 0.62),
                "diagnostic_signature_from_external_ticket",
                None,
            )
        expires_at = candidate.provenance.observed_at + timedelta(days=30)
        return (
            "ADMIT",
            MemoryStatus.CANDIDATE_UNVERIFIED,
            min(candidate.base_confidence, 0.25),
            "untrusted_external_claim_staged",
            expires_at,
        )

    return (
        "ADMIT",
        MemoryStatus.ACTIVE,
        candidate.base_confidence,
        "verified_or_tool_observed_claim",
        None,
    )


def _collect_arg_strings(val: Any) -> list[str]:
    out: list[str] = []
    if isinstance(val, str):
        if val.strip():
            out.append(val)
    elif isinstance(val, (bytes, bytearray)):
        decoded = bytes(val).decode("utf-8", errors="replace").strip()
        if decoded:
            out.append(decoded)
    elif isinstance(val, dict):
        for v in val.values():
            out.extend(_collect_arg_strings(v))
    elif isinstance(val, (list, tuple, set)):
        for v in val:
            out.extend(_collect_arg_strings(v))
    return out


_URL_TARGET_STRICT_RE = re.compile(
    r"^https?://([A-Za-z0-9._-]+)(?::\d{1,5})?(?:[/?#][^\s,;]*)?$",
    re.I,
)


def _extract_claim_targets(claim: MemoryClaim) -> tuple[set[str], set[str], str]:
    c_val = claim.value.strip().lower()
    if claim.claim_type == ClaimType.NEGATIVE_CONSTRAINT:
        blob = c_val[4:].strip() if c_val.startswith("not:") else c_val
    else:
        blob = f"{claim.value} {claim.statement}".lower()
    emails = {m.group(1).lower() for m in EMAIL_FULL_RE.finditer(blob)}
    domains = {d for _, d in ((m.group(1).lower(), m.group(2).lower()) for m in EMAIL_FULL_RE.finditer(blob))}
    domains.update(m.group(1).lower().rstrip(".") for m in URL_DOMAIN_RE.finditer(blob))
    domains.update(m.group(1).lower().rstrip(".") for m in DOMAIN_TOKEN_RE.finditer(blob))
    return emails, domains, c_val


# Per-tool target contract: what kind of destination each side-effecting tool may address.
# Checked AFTER the generic allowlist/verified-memory gate, so a value that is "verified" in
# memory still cannot be pointed at the wrong kind of sink (e.g. an email in a billing route,
# a plaintext http:// webhook, or a URL as an escalation recipient).
TOOL_TARGET_KINDS: dict[str, str] = {
    "send_email": "email",
    "escalate_ticket": "email",
    "update_webhook": "https_url",
    "change_billing_route": "route_token",
}

NON_TARGET_METADATA_KEYS = frozenset(
    {
        "body", "subject", "message", "note", "notes", "reason", "summary", "title", "org_id",
        "tenant_id", "run_id", "ticket_id", "thread_id", "step", "mode", "dry_run", "priority",
        "category", "idempotency_key",
    }
)

_ROUTE_TOKEN_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{1,63}$")
_HTTPS_TARGET_RE = re.compile(r"^https://[a-z0-9._-]+(?::\d{1,5})?(?:[/?#][^\s,;]*)?$", re.I)


def _explicit_target_values(val: Any, under_metadata: bool = False) -> list[Any]:
    out: list[Any] = []
    if isinstance(val, dict):
        for k, v in val.items():
            is_meta = str(k).strip().lower() in NON_TARGET_METADATA_KEYS
            if isinstance(v, (dict, list, tuple, set)):
                out.extend(_explicit_target_values(v, under_metadata or is_meta))
            elif not is_meta and not under_metadata:
                out.append(v)
    elif isinstance(val, (list, tuple, set)):
        for item in val:
            out.extend(_explicit_target_values(item, under_metadata))
    elif not under_metadata:
        out.append(val)
    return out


def verify_tool_target_kinds(tool_name: str, tool_args: dict[str, Any]) -> tuple[bool, str]:
    kind = TOOL_TARGET_KINDS.get(tool_name)
    if kind is None:
        return True, "no_target_contract"
    values = _explicit_target_values(tool_args)
    if not values:
        return False, f"blocked_missing_target_for_contract:{tool_name}:{kind}"
    for raw in values:
        if not isinstance(raw, str):
            return False, f"blocked_target_kind_mismatch:{tool_name}:{kind}"
        for part in [p for p in re.split(r"[\s,;]+", raw.strip()) if p]:
            if kind == "email":
                ok = bool(EMAIL_FULL_RE.fullmatch(part))
            elif kind == "https_url":
                ok = bool(_HTTPS_TARGET_RE.fullmatch(part))
            else:
                ok = bool(_ROUTE_TOKEN_RE.fullmatch(part.lower())) and "@" not in part and "://" not in part
            if not ok:
                return False, f"blocked_target_kind_mismatch:{tool_name}:{kind}"
    return True, f"target_contract_ok:{kind}"


def authorize_tool_action(
    tool_name: str,
    tool_args: dict[str, Any],
    recalled_items: list[RecallItem],
    tenant_policy: TenantPolicy,
    tool_capabilities: dict[str, ToolCapability] | None = None,
    now: datetime | None = None,
) -> ToolExecutionGateResult:
    """Generic default-deny gate, then the tool-specific target contract."""
    gate = _authorize_tool_action_generic(
        tool_name, tool_args, recalled_items, tenant_policy, tool_capabilities, now=now
    )
    if not gate.allowed:
        return gate
    registry = tool_capabilities if tool_capabilities is not None else TOOL_CAPABILITIES
    if registry.get(tool_name) in (ToolCapability.EXTERNAL_WRITE, ToolCapability.CONFIG_WRITE):
        ok, reason = verify_tool_target_kinds(tool_name, tool_args)
        if not ok:
            return ToolExecutionGateResult(
                allowed=False,
                reason=reason,
                supporting_memory_ids=gate.supporting_memory_ids,
                requires_verification=True,
            )
    return gate


def idempotency_key_for(
    *, tenant_id: str, task_id: str, tool_name: str, tool_args: dict[str, Any]
) -> str:
    """Stable key: the same task asking for the same side effect twice is one action."""
    import json as _json

    explicit = tool_args.get("idempotency_key") if isinstance(tool_args, dict) else None
    if explicit:
        basis = f"{tenant_id}|explicit|{explicit}"
    else:
        canon = _json.dumps(
            {k: v for k, v in tool_args.items() if k not in ("dry_run",)},
            sort_keys=True,
            default=str,
        ).lower()
        basis = f"{tenant_id}|{task_id}|{tool_name}|{canon}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:32]


def _authorize_tool_action_generic(
    tool_name: str,
    tool_args: dict[str, Any],
    recalled_items: list[RecallItem],
    tenant_policy: TenantPolicy,
    tool_capabilities: dict[str, ToolCapability] | None = None,
    *,
    now: datetime | None = None,
) -> ToolExecutionGateResult:
    """
    Default-deny capability boundary for tool execution.
    - Unknown / unregistered tools fail closed.
    - Registered READ tools pass immediately.
    - INTERNAL_WRITE / EXTERNAL_WRITE / CONFIG_WRITE enforce active anti-memory veto,
      tenant domain allowlist, and INTERNAL_API_VERIFIED/HUMAN_VERIFIED backing.
    """
    registry = tool_capabilities if tool_capabilities is not None else TOOL_CAPABILITIES
    capability = registry.get(tool_name)
    if capability is None:
        return ToolExecutionGateResult(
            allowed=False,
            reason=f"blocked_unknown_tool_capability_default_deny:{tool_name}",
            requires_verification=True,
        )
    if capability == ToolCapability.READ:
        return ToolExecutionGateResult(allowed=True, reason="registered_read_tool")

    eval_now = now or utcnow()

    def _can_authorize(c: MemoryClaim) -> bool:
        # Evidence that may back an action: ACTIVE, internal-API/human trust, fresh, not flagged.
        return claim_actionability(c, eval_now, MemoryUse.AUTHORIZE_ACTION)[0]

    serialized_targets = " ".join(_collect_arg_strings(tool_args))
    lower_serialized = serialized_targets.lower()

    # Check active anti-memory constraints matching explicit targets or config route values
    email_targets = [(m.group(1).lower(), m.group(2).lower()) for m in EMAIL_FULL_RE.finditer(serialized_targets)]
    has_malformed_url = False
    url_domains: list[str] = []
    for chunk in re.split(r"https?://", serialized_targets, flags=re.I)[1:]:
        token = re.split(r"[\s,;'\"]", chunk, maxsplit=1)[0].rstrip(".,;:!?)]}'\"")
        authority = re.split(r"[/?#]", token, maxsplit=1)[0]
        if not authority or "@" in authority or "\\" in authority or "[" in authority or "]" in authority:
            has_malformed_url = True
            for seg in re.split(r"[@\\/:]+", authority):
                seg_clean = seg.lower().rstrip(".")
                if seg_clean and re.fullmatch(r"[a-z0-9._-]+", seg_clean):
                    url_domains.append(seg_clean)
            continue
        if ":" in authority:
            host_part, port_part = authority.split(":", 1)
            if not port_part.isdigit() or not (1 <= len(port_part) <= 5) or not host_part:
                has_malformed_url = True
                continue
        else:
            host_part = authority
        if not re.fullmatch(r"[A-Za-z0-9._-]+", host_part):
            has_malformed_url = True
            continue
        url_domains.append(host_part.lower().rstrip("."))
    target_entries: list[tuple[str | None, str]] = list(email_targets)
    seen_doms = {d for _, d in email_targets}
    for dom in url_domains:
        if dom not in seen_doms:
            seen_doms.add(dom)
            target_entries.append((None, dom))
    for m in DOMAIN_TOKEN_RE.finditer(serialized_targets):
        dom = m.group(1).lower().rstrip(".")
        if not _looks_like_domain_token(dom):
            continue
        if dom not in seen_doms:
            seen_doms.add(dom)
            target_entries.append((None, dom))

    # Anti-memory veto: any active NEGATIVE_CONSTRAINT matching a target blocks immediately
    anti_blocks: list[str] = []
    for item in recalled_items:
        c = item.claim
        if c.claim_type == ClaimType.NEGATIVE_CONSTRAINT and c.status == MemoryStatus.ACTIVE:
            c_emails, c_domains, c_val = _extract_claim_targets(c)
            raw_neg_val = c_val[4:] if c_val.startswith("not:") else c_val
            matched_anti = False
            if target_entries:
                for addr, dom in target_entries:
                    matched = (
                        (addr in c_emails or c_val == addr or raw_neg_val == addr or (not c_emails and dom in c_domains))
                        if addr is not None
                        else (dom in c_domains or c_val == dom or raw_neg_val == dom)
                    )
                    if matched:
                        matched_anti = True
                        break
            if (
                not matched_anti
                and capability in (ToolCapability.CONFIG_WRITE, ToolCapability.EXTERNAL_WRITE)
                and raw_neg_val
                and c.risk_class in (RiskClass.HIGH_ACTION_ROUTING, RiskClass.MEDIUM_OPERATIONAL)
                and re.search(
                    rf"(?<![a-z0-9_./@-]){re.escape(raw_neg_val)}(?![a-z0-9_./@-])",
                    lower_serialized,
                )
            ):
                matched_anti = True
            if matched_anti:
                anti_blocks.append(c.memory_id)

    if anti_blocks:
        return ToolExecutionGateResult(
            allowed=False,
            reason="blocked_by_active_negative_constraint",
            blocking_memory_ids=anti_blocks,
            requires_verification=True,
        )

    def _is_single_domain_target(tok: str) -> bool:
        if not tok:
            return False
        if EMAIL_FULL_RE.fullmatch(tok):
            return True
        if _URL_TARGET_STRICT_RE.fullmatch(tok):
            return True
        return bool(
            DOMAIN_TOKEN_RE.fullmatch(tok)
            and _looks_like_domain_token(tok.rstrip("."))
        )

    def _is_pure_domain_target(tv: str) -> bool:
        parts = [p for p in re.split(r"[\s,;]+", tv.strip()) if p]
        return bool(parts) and all(_is_single_domain_target(p) for p in parts)

    non_target_metadata_keys = frozenset(
        {
            "body",
            "subject",
            "message",
            "note",
            "notes",
            "reason",
            "summary",
            "title",
            "org_id",
            "tenant_id",
            "run_id",
            "ticket_id",
            "thread_id",
            "step",
            "mode",
            "dry_run",
            "priority",
            "category",
        }
    )

    def _collect_nested_dicts_only(val: Any) -> list[str]:
        out: list[str] = []
        if isinstance(val, dict):
            for nk, nv in val.items():
                if str(nk).strip().lower() in non_target_metadata_keys:
                    out.extend(_collect_nested_dicts_only(nv))
                else:
                    out.extend(_collect_target_values(nv))
        elif isinstance(val, (list, tuple, set)):
            for item in val:
                out.extend(_collect_nested_dicts_only(item))
        return out

    def _collect_target_values(val: Any) -> list[str]:
        out: list[str] = []
        if val is None:
            return out
        if isinstance(val, bool):
            out.append(f"__bool__:{str(val).lower()}")
        elif isinstance(val, (int, float)):
            out.append(str(val).strip().lower())
        elif isinstance(val, str):
            stripped = val.strip().lower()
            if stripped:
                out.append(stripped)
        elif isinstance(val, (bytes, bytearray)):
            decoded = bytes(val).decode("utf-8", errors="replace").strip().lower()
            if decoded:
                out.append(decoded)
        elif isinstance(val, dict):
            for nv in val.values():
                out.extend(_collect_target_values(nv))
        elif isinstance(val, (list, tuple, set)):
            for item in val:
                out.extend(_collect_target_values(item))
        else:
            out.append(f"__obj__:{val!r}".lower())
        return out

    explicit_target_vals = _collect_nested_dicts_only(tool_args)

    if capability == ToolCapability.INTERNAL_WRITE:
        if has_malformed_url or (not target_entries and ("@" in serialized_targets or "://" in serialized_targets)):
            return ToolExecutionGateResult(
                allowed=False,
                reason="blocked_unparseable_external_target_fail_closed",
                requires_verification=True,
            )
        if not target_entries:
            return ToolExecutionGateResult(allowed=True, reason="registered_internal_write_no_external_target")

    non_domain_support: list[str] = []
    if capability in (ToolCapability.EXTERNAL_WRITE, ToolCapability.CONFIG_WRITE):
        if has_malformed_url:
            return ToolExecutionGateResult(
                allowed=False,
                reason="blocked_unparseable_external_target_fail_closed",
                requires_verification=True,
            )
        for tv in explicit_target_vals:
            if not _is_pure_domain_target(tv) and ("@" in tv or "://" in tv):
                return ToolExecutionGateResult(
                    allowed=False,
                    reason="blocked_unparseable_external_target_fail_closed",
                    requires_verification=True,
                )
        if not target_entries and ("@" in serialized_targets or "://" in serialized_targets):
            return ToolExecutionGateResult(
                allowed=False,
                reason="blocked_unparseable_external_target_fail_closed",
                requires_verification=True,
            )
        if not explicit_target_vals:
            unauthorized_in_prose = [
                dom for _, dom in target_entries if not is_domain_authorized(dom, tenant_policy)
            ]
            if unauthorized_in_prose:
                return ToolExecutionGateResult(
                    allowed=False,
                    reason=(
                        f"blocked_unverified_external_target:{','.join(dict.fromkeys(unauthorized_in_prose))}; "
                        "requires internal CRM/API or human verification"
                    ),
                    requires_verification=True,
                )
            fail_reason = (
                "blocked_missing_verifiable_external_target"
                if capability == ToolCapability.EXTERNAL_WRITE
                else "blocked_unverified_config_write_fail_closed"
            )
            return ToolExecutionGateResult(
                allowed=False,
                reason=fail_reason,
                requires_verification=True,
            )

        non_domain_targets = [tv for tv in explicit_target_vals if not _is_pure_domain_target(tv)]
        if non_domain_targets:
            allowed_risks = (
                (RiskClass.HIGH_ACTION_ROUTING,)
                if capability == ToolCapability.EXTERNAL_WRITE
                else (RiskClass.HIGH_ACTION_ROUTING, RiskClass.MEDIUM_OPERATIONAL)
            )
            verified_val_ids: dict[str, list[str]] = {}
            for item in recalled_items:
                c = item.claim
                if c.risk_class not in allowed_risks or not _can_authorize(c):
                    continue
                raw_val = c.value.strip().lower()
                if len(raw_val) >= 3:
                    verified_val_ids.setdefault(raw_val, []).append(c.memory_id)

            if not all(
                not tv.startswith(("__bool__:", "__obj__:")) and tv in verified_val_ids
                for tv in non_domain_targets
            ):
                fail_reason = (
                    "blocked_missing_verifiable_external_target"
                    if capability == ToolCapability.EXTERNAL_WRITE
                    else "blocked_unverified_config_write_fail_closed"
                )
                return ToolExecutionGateResult(
                    allowed=False,
                    reason=fail_reason,
                    requires_verification=True,
                )
            for tv in non_domain_targets:
                non_domain_support.extend(verified_val_ids[tv])

        if not target_entries:
            ok_reason = (
                "external_write_backed_by_verified_route_token"
                if capability == ToolCapability.EXTERNAL_WRITE
                else "config_write_backed_by_verified_internal_or_human_memory"
            )
            return ToolExecutionGateResult(
                allowed=True,
                reason=ok_reason,
                supporting_memory_ids=list(dict.fromkeys(non_domain_support)),
            )
    elif not target_entries:
        return ToolExecutionGateResult(
            allowed=False,
            reason="blocked_unverified_write_fail_closed",
            requires_verification=True,
        )

    unauthorized = [(addr, dom) for addr, dom in target_entries if not is_domain_authorized(dom, tenant_policy)]
    if not unauthorized:
        # The allowlist says the domain is acceptable, not that this mailbox is still right. If the exact
        # target came from a recalled routing claim that is past its revalidation window, re-check first.
        stale_route_ids: list[str] = []
        for addr, _dom in target_entries:
            if addr is None:
                continue
            stale_here: list[str] = []
            fresh_confirmation = False
            for item in recalled_items:
                c = item.claim
                if c.claim_type == ClaimType.NEGATIVE_CONSTRAINT or c.status != MemoryStatus.ACTIVE:
                    continue
                if c.risk_class != RiskClass.HIGH_ACTION_ROUTING:
                    continue
                c_emails, _c_domains, c_val = _extract_claim_targets(c)
                if addr not in c_emails and c_val != addr:
                    continue
                if claim_freshness(c, eval_now)["stale"]:
                    stale_here.append(c.memory_id)
                elif _can_authorize(c):
                    fresh_confirmation = True  # e.g. the CRM re-confirmed it in this run
            if stale_here and not fresh_confirmation:
                stale_route_ids.extend(stale_here)
        if stale_route_ids:
            return ToolExecutionGateResult(
                allowed=False,
                reason="blocked_stale_routing_memory_revalidate_first",
                blocking_memory_ids=list(dict.fromkeys(stale_route_ids)),
                requires_verification=True,
            )
        return ToolExecutionGateResult(
            allowed=True,
            reason="all_target_domains_in_tenant_allowlist",
            supporting_memory_ids=list(dict.fromkeys(non_domain_support)),
        )

    # Every unauthorized off-domain target MUST be independently backed by an ACTIVE
    # non-negative-constraint memory with trust_tier >= INTERNAL_API_VERIFIED, with no blocking claims.
    verified_support: list[str] = list(non_domain_support)
    blocking_ids: list[str] = []
    unverified_domains: list[str] = []

    for addr, dom in unauthorized:
        entry_verified: list[str] = []
        entry_blocked: list[str] = []
        for item in recalled_items:
            c = item.claim
            if c.claim_type == ClaimType.NEGATIVE_CONSTRAINT:
                continue
            c_emails, c_domains, c_val = _extract_claim_targets(c)
            matched = (
                (addr in c_emails or c_val == addr)
                if addr is not None
                else (dom in c_domains or c_val == dom)
            )
            if matched:
                if _can_authorize(c):
                    entry_verified.append(c.memory_id)
                elif c.status == MemoryStatus.ACTIVE and c.trust_tier >= TrustTier.INTERNAL_API_VERIFIED:
                    # Verified once but now stale or flagged: no longer counts as support, and does
                    # not veto either; a fresh confirmation in this run can still authorize.
                    continue
                else:
                    entry_blocked.append(c.memory_id)
        if entry_blocked or not entry_verified:
            blocking_ids.extend(entry_blocked)
            unverified_domains.append(dom)
        else:
            verified_support.extend(entry_verified)

    if not unverified_domains and not blocking_ids and verified_support:
        return ToolExecutionGateResult(
            allowed=True,
            reason="off_domain_target_backed_by_verified_internal_or_human_memory",
            supporting_memory_ids=list(dict.fromkeys(verified_support)),
        )

    return ToolExecutionGateResult(
        allowed=False,
        reason=(
            f"blocked_unverified_external_target:{','.join(dict.fromkeys(unverified_domains))}; "
            "requires internal CRM/API or human verification"
        ),
        blocking_memory_ids=list(dict.fromkeys(blocking_ids)),
        requires_verification=True,
    )


_GLOBAL_DOMAIN_TLDS = frozenset(
    {
        "com", "co", "io", "net", "org", "ai", "dev", "app", "cloud", "xyz", "site", "tech",
        "online", "info", "biz", "edu", "gov", "me", "tv", "cc", "pro", "top", "club",
        "internal", "local", "example",
        "uk", "us", "ca", "de", "fr", "eu", "in", "jp", "au", "ru", "cn", "br", "nl", "se", "ch", "sg",
    }
)


def _check_global_common(
    stripped: str, tenant_policy: TenantPolicy | None = None
) -> tuple[bool, str]:
    for pat in UNSAFE_GLOBAL_PATTERNS + INJECTION_PATTERNS:
        if pat.search(stripped):
            return False, "unsafe_or_injection_workaround_blocked"

    if EMAIL_RE.search(stripped) or IPV4_RE.search(stripped) or SECRET_RE.search(stripped):
        return False, "contains_pii_email_ip_or_secret"

    if LOCAL_PATH_RE.search(stripped):
        return False, "contains_local_or_internal_path"

    for domain in extract_domains(stripped):
        if domain not in PUBLIC_GLOBAL_DOMAINS:
            return False, f"contains_non_public_domain:{domain}"

    for m in DOMAIN_TOKEN_RE.finditer(stripped):
        dom = m.group(1).lower().rstrip(".")
        tld = dom.rsplit(".", 1)[-1]
        if tld in COMMON_NON_DOMAIN_EXTS or dom in PUBLIC_GLOBAL_DOMAINS:
            continue
        # Reject bare non-public domain tokens (e.g. acme.com, evil.co, bad.xyz) in metadata/statements
        if tld in _GLOBAL_DOMAIN_TLDS:
            return False, f"contains_non_public_domain:{dom}"

    if tenant_policy is not None:
        lowered = stripped.lower()
        for alias in [tenant_policy.org_id, *tenant_policy.org_aliases]:
            a = alias.strip().lower()
            if a and re.search(rf"(?<![a-z0-9]){re.escape(a)}(?![a-z0-9])", lowered):
                return False, f"contains_tenant_identifier:{a}"

    return True, "ok"


def validate_global_metadata_field(
    field_value: str,
    tenant_policy: TenantPolicy | None = None,
    *,
    max_len: int = 120,
) -> tuple[bool, str]:
    """Validate short global metadata fields (artifact_name, error_signature, cluster_key, value)."""
    stripped = field_value.strip()
    if not stripped or len(stripped) > max_len:
        return False, "global_metadata_length_out_of_bounds"
    return _check_global_common(stripped, tenant_policy)


def validate_global_safety(
    text: str, tenant_policy: TenantPolicy | None = None
) -> tuple[bool, str]:
    """Validate text against PII, org identifiers, non-public domains, and unsafe commands."""
    stripped = text.strip()
    if len(stripped) < 8 or len(stripped) > 480:
        return False, "global_statement_length_out_of_bounds"
    return _check_global_common(stripped, tenant_policy)


def sanitize_global_statement(
    text: str, tenant_policy: TenantPolicy | None = None
) -> tuple[bool, str, str]:
    """Validate and clean candidate text for zero-leakage global sharing."""
    ok, reason = validate_global_safety(text, tenant_policy)
    if not ok:
        return False, "", reason
    sanitized = re.sub(r"\[[^\]]*\]", "", text.strip()).strip()
    return True, sanitized, "ok"


def sanitize_for_global(
    claim: MemoryClaim, tenant_policy: TenantPolicy
) -> tuple[bool, dict[str, str] | None, str]:
    """
    Two-stage zero-leakage gate for cross-tenant generalization:
    - Stage 1 requires verified tool/API trust (>= TOOL_OBSERVED, never UNTRUSTED_EXTERNAL
      or single-run AGENT_INFERRED) and LOW_DIAGNOSTIC risk.
    - Stage 2 validates every exported field (statement, value, artifact_name,
      error_signature, cluster_key) for PII, tenant identifiers, secrets, and unsafe workarounds.
    """
    if claim.status in (
        MemoryStatus.REFUTED,
        MemoryStatus.QUARANTINED,
        MemoryStatus.CANDIDATE_UNVERIFIED,
        MemoryStatus.ARCHIVED,
    ):
        return False, None, "ineligible_status"

    if claim.trust_tier < TrustTier.TOOL_OBSERVED or claim.source_type in EXTERNAL_UNTRUSTED_SOURCE_TYPES:
        return False, None, "insufficient_evidence_trust_for_global_sharing"

    if claim.risk_class != RiskClass.LOW_DIAGNOSTIC:
        return False, None, "only_low_diagnostic_technical_claims_eligible"

    if not claim.metadata.get("is_public_technical_artifact"):
        return False, None, "not_marked_as_public_technical_artifact"

    raw_artifact = str(claim.metadata.get("artifact_name") or claim.subject).strip()
    raw_err_sig = str(claim.metadata.get("error_signature") or claim.predicate).strip()
    if not raw_artifact or not raw_err_sig:
        return False, None, "missing_artifact_or_error_signature"

    for field_name, field_val in (
        ("artifact_name", raw_artifact),
        ("error_signature", raw_err_sig),
        ("value", claim.value),
    ):
        ok_f, reason_f = validate_global_metadata_field(field_val, tenant_policy, max_len=240)
        if not ok_f:
            return False, None, f"{field_name}:{reason_f}"

    ok, sanitized, reason = sanitize_global_statement(claim.statement, tenant_policy)
    if not ok:
        return False, None, reason

    norm_artifact = normalize_token(raw_artifact)
    norm_err = normalize_token(raw_err_sig)
    cluster_key = f"global|{norm_artifact}|{norm_err}"

    ok_ck, reason_ck = validate_global_metadata_field(cluster_key, tenant_policy, max_len=200)
    if not ok_ck:
        return False, None, f"cluster_key:{reason_ck}"

    return (
        True,
        {
            "cluster_key": cluster_key,
            "artifact": norm_artifact,
            "error_sig": norm_err,
            "sanitized_summary": sanitized,
            "sanitized_value": claim.value.strip(),
        },
        "sanitized_for_k_anonymity_staging",
    )


def claims_semantically_related(parent: MemoryClaim, candidate_text: str, candidate_subject: str) -> bool:
    """
    Deterministic causal relevance check used to bound error containment
    so unrelated memories that co-occurred in a run are not quarantined.
    """
    if canonicalize_subject(parent.subject) == canonicalize_subject(candidate_subject):
        return True
    p_tokens = {
        t
        for t in re.findall(r"[a-z0-9_]{3,}", f"{parent.subject} {parent.statement}".lower())
        if t not in {"the", "and", "for", "with", "that", "this", "from", "uses", "api"}
    }
    c_tokens = {
        t
        for t in re.findall(r"[a-z0-9_]{3,}", f"{candidate_subject} {candidate_text}".lower())
        if t not in {"the", "and", "for", "with", "that", "this", "from", "uses", "api"}
    }
    if len(p_tokens & c_tokens) >= 2:
        return True
    return fuzz.token_set_ratio(parent.statement.lower(), candidate_text.lower()) >= 65


def compute_retention_utility(claim: MemoryClaim, now: datetime) -> float:
    """
    Zero-LLM deterministic retention score used for 2+ year bounded forgetting.
    Higher score = retained longer.
    """
    conf = effective_confidence(claim, now)
    age_since_access_days = max(0.0, (now - claim.last_accessed_at).total_seconds() / 86400.0)
    recency_factor = math.exp(-age_since_access_days / 90.0)
    help_bonus = 0.25 * math.log1p(claim.verified_helps)
    access_bonus = 0.08 * math.log1p(claim.access_count)
    trust_bonus = 0.12 * (int(claim.trust_tier) / 6.0)
    type_bonus = 0.35 if claim.claim_type == ClaimType.NEGATIVE_CONSTRAINT else 0.0
    if claim.trust_tier == TrustTier.HUMAN_VERIFIED:
        type_bonus += 0.25
    return round(conf * 0.6 + recency_factor * 0.25 + help_bonus + access_bonus + trust_bonus + type_bonus, 6)


# --- Cross-tenant evidence binding (global promotion) ------------------------------------

_UPSTREAM_REF_RE = re.compile(
    r"(?:https?://[^\s,;)\]]+|\bGHSA(?:-[a-z0-9]{4}){3}\b|\bCVE-\d{4}-\d{4,}\b"
    r"|\b(?:issue|pr|pull)\s*#\d+\b|\b[a-z0-9_.-]+/[a-z0-9_.-]+#\d+\b)",
    re.I,
)


def evidence_fingerprint(
    *,
    observation_text: str,
    tenant_salted_id: str,
    source_id: str,
    upstream_ref: str | None = None,
) -> str:
    """
    Identity of the *underlying* evidence behind a staged cross-org contribution.

    - If the evidence cites an upstream document (URL, GHSA/CVE id, issue #), the fingerprint
      is the cited reference set: five tenants that all read the same GitHub issue or blog
      post count as ONE source, however many distinct tenants/source_ids they have.
    - Otherwise the evidence is the tenant's own tool execution, which is independent per
      (tenant, execution) by construction.
    """
    refs = sorted(
        {m.group(0).lower().rstrip(".,;:)") for m in _UPSTREAM_REF_RE.finditer(observation_text)}
        | ({upstream_ref.strip().lower()} if upstream_ref and upstream_ref.strip() else set())
    )
    if refs:
        return "ref:" + hashlib.sha256("|".join(refs).encode("utf-8")).hexdigest()[:16]
    return "exec:" + hashlib.sha256(f"{tenant_salted_id}|{source_id}".encode()).hexdigest()[:16]


def _norm_ws(s: str) -> str:
    return " ".join(s.strip().lower().split())


def global_value_supported(value: str, staged_values: list[str], staged_summaries: list[str], k: int) -> bool:
    """
    A promoted advisory value must be the one >= k contributing tenants actually staged.
    When contributors staged no explicit value (legacy callers), every alphanumeric token of
    the value must occur in >= k staged summaries. Content the tenants never wrote
    (e.g. a swapped package pin) is therefore unpromotable even if K tenants staged *something*.
    """
    norm_v = _norm_ws(value)
    if not norm_v:
        return False
    nonempty = [_norm_ws(v) for v in staged_values if v and v.strip()]
    if nonempty:
        return sum(1 for v in nonempty if v == norm_v) >= k
    toks = [t for t in re.split(r"[^a-z0-9]+", norm_v) if t]
    if not toks:
        return False
    support = 0
    for s in staged_summaries:
        s_toks = set(re.split(r"[^a-z0-9]+", _norm_ws(s)))
        if all(t in s_toks for t in toks):
            support += 1
    return support >= k


_STATEMENT_LITERAL_RE = re.compile(
    r"[a-z0-9_.:/@=<>+~^-]*\d[a-z0-9_.:/@=<>+~^-]*|\b[a-z0-9-]+\.[a-z]{2,}\b|`[^`]+`"
)


def statement_grounded_in_evidence(statement: str, evidence_blob: str) -> bool:
    """Every literal (version, pin, flag, host, code span) in a synthesized statement must
    already appear in the staged evidence; an LLM synthesis cannot introduce new specifics."""
    hay = _norm_ws(evidence_blob)
    for lit in _STATEMENT_LITERAL_RE.findall(_norm_ws(statement)):
        lit = lit.strip("`").rstrip(".,;:!?)]}").lstrip("([{")
        if len(lit) >= 2 and lit not in hay:
            return False
    return True
