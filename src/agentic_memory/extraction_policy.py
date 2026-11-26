"""
Deterministic post-extraction policy (0 LLM calls).

The LLM proposes atomic claims with evidence spans and a *suggested* durability.
Everything below decides, without the model, whether each proposal is:

1. grounded in its own evidence span (not merely somewhere in the observation),
2. semantically coherent (predicate type matches value type),
3. ephemeral task state, durable org knowledge, or a user preference (system-controlled scope),
4. leaving salient identifiers in the observation uncovered (claim-level recall signal).

The model never grants itself scope, trust, or global-sharing permission.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from agentic_memory.models import ClaimType, RiskClass, Scope
from agentic_memory.policy import (
    EMAIL_FULL_RE,
    ROUTING_SIGNAL_RE,
    URL_DOMAIN_RE,
    canonicalize_predicate,
    canonicalize_subject,
)


class SuggestedDurability(StrEnum):
    EPHEMERAL = "ephemeral"
    PERSISTENT = "persistent"
    USER_PREFERENCE = "user_preference"
    PUBLIC_TECHNICAL = "public_technical"


_LEGACY_SCOPE_TO_DURABILITY = {
    "task": SuggestedDurability.EPHEMERAL,
    "org": SuggestedDurability.PERSISTENT,
    "user": SuggestedDurability.USER_PREFERENCE,
    "global": SuggestedDurability.PUBLIC_TECHNICAL,
}


def parse_suggested_durability(
    raw_durability: object, raw_legacy_scope: object = None
) -> SuggestedDurability | None:
    """Accept the new `suggested_durability` field; map a legacy `scope` field as a hint only."""
    if raw_durability is not None:
        key = str(raw_durability).strip().lower()
        for d in SuggestedDurability:
            if key == d.value:
                return d
        if key in ("durable", "permanent", "org", "organization"):
            return SuggestedDurability.PERSISTENT
        if key in ("temporary", "transient", "task"):
            return SuggestedDurability.EPHEMERAL
    if raw_legacy_scope is not None:
        return _LEGACY_SCOPE_TO_DURABILITY.get(str(raw_legacy_scope).strip().lower())
    return None


# --- Scope classification -------------------------------------------------------------

_EPHEMERAL_PHRASE_RE = re.compile(
    r"\b(?:for\s+(?:this|the\s+current)\s+(?:ticket|task|run|replay|session|request)"
    r"|current\s+(?:ticket|task|run|replay|session)|replay\s+only|only\s+for\s+now"
    r"|temporar(?:y|ily)|one[-\s]off|scratch(?:pad)?|for\s+today|right\s+now"
    r"|until\s+(?:this|the)\s+(?:ticket|task)\s+(?:is\s+)?(?:closed|resolved))\b",
    re.I,
)
_REQUEST_ONLY_RE = re.compile(
    r"^\s*(?:please|kindly|can\s+you|could\s+you|check\s+whether|verify\s+whether|"
    r"confirm\s+whether|look\s+into|investigate)\b",
    re.I,
)
_EPHEMERAL_PREDICATE_TOKENS = frozenset(
    {
        "trace_id", "request_id", "session_id", "ticket_id", "run_id", "task_status",
        "next_step", "todo", "requested_action", "request", "ask", "current_status",
        "is_responding", "replay",
    }
)
_EPHEMERAL_SUBJECT_TOKENS = frozenset({"debug_flag", "current_ticket", "this_ticket", "task", "request"})

_DURABLE_PREDICATE_PARTS = (
    "version", "engine", "cluster", "region", "endpoint", "url", "route", "routing", "contact",
    "email", "mailbox", "schedule", "cutoff", "timezone", "format", "schema", "bucket",
    "queue", "owner", "policy", "config", "rail", "window", "destination", "provider",
    "replica", "tier", "sla", "limit", "retention", "workaround", "fix", "error", "bug",
    "timeout", "lock", "leak", "bypass", "tls", "ssl", "reload", "forward", "cc_",
    "approval", "retire", "deprecat", "migrat", "host", "database", "warehouse", "storage",
    "dependency", "library", "proxy", "certificate", "auth", "token", "port", "language",
)
_STATE_CHANGE_RE = re.compile(
    r"\b(?:switched|migrated|moved|now|starting|upgraded|downgraded|retires?|retiring|"
    r"deprecated|rotated|pinned|runs?|uses?|land(?:s)?|requires?|routes?|arrive[sd]?|"
    r"resolved\s+by|fixes|fixed)\b",
    re.I,
)
_IDENTIFIER_RE = re.compile(
    r"(?:\b[a-z][a-z0-9]*_[a-z0-9_]+\b|\b[a-z0-9]+(?:-[a-z0-9]+){2,}\b|\b\d+\.\d+(?:\.\d+)*\b"
    r"|\bv\d+\b|\b[A-Za-z]+\s?\d{1,3}\b|\b\d{1,2}:\d{2}\b|\b\d+\s?(?:ms|s|gb|mb|%)\b"
    r"|\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:com|net|org|io|co|dev|internal|cloud)\b|^\s*\d+\s*$)",
    re.I,
)
_USER_PREF_PARTS = ("prefer", "preference", "notify", "language", "signature", "locale")


@dataclass(frozen=True)
class ScopeDecision:
    scope: Scope
    reason: str
    llm_suggestion: SuggestedDurability | None
    overrode_llm: bool


def _tokens(s: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9_]+", s.lower()) if t}


def classify_scope(
    *,
    subject: str,
    predicate: str,
    value: str,
    statement: str,
    source_quote: str | None,
    claim_type: ClaimType,
    risk_class: RiskClass,
    suggested_durability: SuggestedDurability | None,
    error_signature: str | None = None,
    user_id: str | None = None,
) -> ScopeDecision:
    """
    System-controlled scope decision. The LLM's suggestion is one input; deterministic
    evidence about ephemerality and durable org relevance takes precedence over it.

    Order: explicit ephemerality marker > durable-org evidence > LLM suggestion > TASK default.
    Scope never grants trust: an ORG decision still passes through `evaluate_candidate_write`,
    which caps untrusted claims as CANDIDATE_UNVERIFIED.
    """
    subj = canonicalize_subject(subject)
    pred = canonicalize_predicate(predicate)
    text = f"{statement} {source_quote or ''}"
    sugg = suggested_durability

    def _decide(scope: Scope, reason: str) -> ScopeDecision:
        llm_scope = {
            SuggestedDurability.EPHEMERAL: Scope.TASK,
            SuggestedDurability.PERSISTENT: Scope.ORG,
            SuggestedDurability.PUBLIC_TECHNICAL: Scope.ORG,
            SuggestedDurability.USER_PREFERENCE: Scope.USER,
        }.get(sugg) if sugg is not None else None
        return ScopeDecision(
            scope=scope,
            reason=reason,
            llm_suggestion=sugg,
            overrode_llm=llm_scope is not None and llm_scope != scope,
        )

    # 1) Explicit ephemerality markers always keep a claim out of durable storage.
    pred_tokens = _tokens(pred) | {pred}
    if _EPHEMERAL_PHRASE_RE.search(text):
        return _decide(Scope.TASK, "explicit_ephemerality_marker")
    if pred in _EPHEMERAL_PREDICATE_TOKENS or pred_tokens & _EPHEMERAL_PREDICATE_TOKENS:
        return _decide(Scope.TASK, "ephemeral_predicate")
    if subj in _EPHEMERAL_SUBJECT_TOKENS:
        return _decide(Scope.TASK, "ephemeral_subject")

    # 2) User preference (requires a concrete user_id; otherwise falls through to org rules).
    if user_id and (
        sugg == SuggestedDurability.USER_PREFERENCE or any(p in pred for p in _USER_PREF_PARTS)
    ):
        return _decide(Scope.USER, "user_preference_with_user_id")

    # 3) Durable organization-specific evidence.
    if claim_type in (ClaimType.PROCEDURE, ClaimType.NEGATIVE_CONSTRAINT):
        return _decide(Scope.ORG, "durable_procedure_or_constraint")
    if risk_class == RiskClass.HIGH_ACTION_ROUTING or ROUTING_SIGNAL_RE.search(f"{subj} {pred}"):
        return _decide(Scope.ORG, "routing_policy_is_org_durable")
    if error_signature:
        return _decide(Scope.ORG, "diagnostic_signature_about_org_infra")
    durable_pred = any(part in pred for part in _DURABLE_PREDICATE_PARTS)
    has_identifier = bool(_IDENTIFIER_RE.search(value)) or bool(
        EMAIL_FULL_RE.search(value) or URL_DOMAIN_RE.search(value)
    )
    state_change = bool(_STATE_CHANGE_RE.search(text))
    if durable_pred and (has_identifier or state_change):
        return _decide(Scope.ORG, "durable_predicate_with_identifier_or_state_change")
    if durable_pred and sugg != SuggestedDurability.EPHEMERAL:
        return _decide(Scope.ORG, "durable_predicate")
    if has_identifier and state_change and not _REQUEST_ONLY_RE.search(statement):
        return _decide(Scope.ORG, "identifier_with_state_change")

    # 4) Weak or no deterministic evidence: follow the model's suggestion.
    if sugg in (SuggestedDurability.PERSISTENT, SuggestedDurability.PUBLIC_TECHNICAL):
        return _decide(Scope.ORG, "llm_suggested_persistent_no_contrary_evidence")
    if sugg == SuggestedDurability.EPHEMERAL:
        return _decide(Scope.TASK, "llm_suggested_ephemeral")
    if _REQUEST_ONLY_RE.search(statement):
        return _decide(Scope.TASK, "request_without_durable_fact")
    if durable_pred:
        return _decide(Scope.ORG, "durable_predicate")
    return _decide(Scope.TASK, "no_durability_evidence")


# --- Semantic coherence -----------------------------------------------------------------

_DATE_LIKE_RE = re.compile(
    r"(?:\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b|\bq[1-4]\b|"
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b|"
    r"\b(?:mon|tues|wednes|thurs|fri|satur|sun)day\b|\b\d{1,2}:\d{2}\b)",
    re.I,
)


def validate_claim_semantics(predicate: str, value: str) -> tuple[bool, str]:
    """Cheap type check that the value is the kind of thing the predicate names."""
    pred = canonicalize_predicate(predicate)
    val = value.strip()
    if any(p in pred for p in ("email", "mailbox", "cc_policy", "forward_to")) and not EMAIL_FULL_RE.search(val):
        return False, "email_predicate_without_email_value"
    if any(p in pred for p in ("url", "endpoint", "webhook")) and not (
        URL_DOMAIN_RE.search(val) or re.search(r"\b[a-z0-9-]+\.[a-z]{2,}\b", val, re.I)
    ):
        return False, "url_predicate_without_url_value"
    is_tz_or_format = any(p in pred for p in ("timezone", "tz", "format", "schema", "window"))
    if (
        not is_tz_or_format
        and any(p in pred for p in ("date", "retire", "deadline", "sunset", "expires", "expiry"))
        and not _DATE_LIKE_RE.search(val)
    ):
        return False, "date_predicate_without_date_value"
    if "version" in pred and not re.search(r"\d", val):
        return False, "version_predicate_without_version_number"
    if pred in ("active_route", "route_id", "billing_route", "route") and (
        len(val.split()) > 3 or _DATE_LIKE_RE.fullmatch(val)
    ):
        return False, "route_predicate_value_is_not_a_route_token"
    return True, "ok"


# --- Per-claim span binding -------------------------------------------------------------

_NUM_RE = re.compile(r"\b\d+(?:\.\d+)*\b")


def value_literals_bound_to_quote(value: str, source_quote: str | None) -> bool:
    """
    High-signal literals in the value (emails, URL hosts, numbers/versions, snake_case or
    kebab identifiers) must appear in the claim's *own* evidence span, not merely somewhere
    in the observation. This is what lets one observation yield several independently
    grounded claims without one span silently vouching for another claim's value.
    """
    if source_quote is None or not source_quote.strip():
        return True  # observation-level grounding is applied separately
    q = " ".join(source_quote.lower().split())
    v = " ".join(value.lower().split())
    for m in EMAIL_FULL_RE.finditer(v):
        if m.group(1) not in q:
            return False
    for m in URL_DOMAIN_RE.finditer(v):
        if m.group(1) not in q:
            return False
    for tok in _NUM_RE.findall(v):
        if not re.search(rf"(?<![a-z0-9]){re.escape(tok)}(?![0-9])", q):
            return False
    return all(tok in q for tok in re.findall(r"\b[a-z][a-z0-9]*_[a-z0-9_]+\b", v))


# --- Coverage gaps (claim-level recall signal) -----------------------------------------

_SALIENT_SPAN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("url", re.compile(r"https?://[^\s,;]+", re.I)),
    ("route_token", re.compile(r"\b[a-z][a-z0-9]*_v\d+\b|\bv\d+_[a-z0-9_]+\b", re.I)),
    ("iso_date", re.compile(r"\b\d{4}-\d{2}-\d{2}\b")),
    ("version", re.compile(r"\b[A-Za-z][A-Za-z0-9]*\s?(?:==)?\d+\.\d+(?:\.\d+)?\b")),
)


def find_coverage_gaps(observation_text: str, covered_texts: list[str]) -> list[dict[str, str]]:
    """
    Salient identifiers present in an observation that no accepted claim mentions.
    Used to (a) report claim-level recall gaps and (b) optionally trigger one bounded,
    focused re-extraction. Never used to synthesize claims deterministically.
    """
    hay = " ".join(" ".join(covered_texts).lower().split())
    gaps: list[dict[str, str]] = []
    seen: set[str] = set()
    for kind, pat in _SALIENT_SPAN_PATTERNS:
        for m in pat.finditer(observation_text):
            span = m.group(0).rstrip(".,;:)")
            key = span.lower()
            if key in seen:
                continue
            seen.add(key)
            if key not in hay:
                gaps.append({"kind": kind, "span": span})
    return gaps
