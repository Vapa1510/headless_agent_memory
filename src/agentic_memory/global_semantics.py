"""
Deterministic semantic checks for cross-tenant advisories (0 LLM calls).

Literal grounding (every version/host/flag in a synthesized statement must exist in the staged
evidence) does not catch meaning changes built from the same words: "leave AsyncClient open"
uses only words that appear in "close AsyncClient in finally". This module extracts two kinds of
structure and compares them:

- directives: (action, object, polarity), with negation ("do not close", "never pin") and
  antonyms ("open" vs "close", "upgrade" vs "downgrade") folded onto one polarity axis;
- version constraints: (package, operator, version), so "pin urllib3==2.0.7" cannot become
  "upgrade urllib3 to >=2.0.7".

A synthesized statement must be supported by the evidence on both; contributors whose staged
evidence contradicts each other block promotion entirely.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

# antonym pairs collapse onto the first element as the polarity axis
_ANTONYMS: dict[str, tuple[str, bool]] = {
    "close": ("close", True), "open": ("close", False),
    "pin": ("pin", True), "unpin": ("pin", False),
    "upgrade": ("upgrade", True), "downgrade": ("upgrade", False),
    "enable": ("enable", True), "disable": ("enable", False),
    "increase": ("increase", True), "decrease": ("increase", False),
    "reduce": ("increase", False),  # "raise"/"lower" omitted: ambiguous ("raises SSLError")
    "add": ("add", True), "remove": ("add", False), "delete": ("add", False),
    "use": ("use", True), "avoid": ("use", False),
    "reuse": ("reuse", True), "recreate": ("reuse", False),
    "verify": ("verify", True), "skip": ("verify", False), "bypass": ("verify", False),
    "set": ("set", True), "restart": ("restart", True), "retry": ("retry", True),
    "reload": ("reload", True), "rotate": ("rotate", True), "shut": ("close", True),
}
_STOP = frozenset(
    {"the", "a", "an", "in", "on", "to", "of", "for", "with", "and", "or", "after", "before", "it",
     "its", "each", "every", "all", "any", "block", "blocks", "your", "their", "this", "that", "is", "be"}
)
_NEG = frozenset({"not", "never", "don't", "dont", "without", "no", "stop", "cannot", "can't", "shouldn't"})


def _inflections(verb: str) -> set[str]:
    forms = {verb, verb + "s", verb + "d", verb + "ed", verb + "ing"}
    if verb.endswith("e"):
        forms |= {verb[:-1] + "ing", verb + "d"}
    if re.fullmatch(r"[a-z]*[^aeiou][aeiou][^aeiouwxy]", verb):
        forms |= {verb + verb[-1] + "ing", verb + verb[-1] + "ed"}
    return forms


_FORM_TO_VERB: dict[str, str] = {f: v for v in _ANTONYMS for f in _inflections(v)}
_TOKEN_RE = re.compile(r"[a-z0-9_.+-]+(?:[=<>~!]=?[a-z0-9_.+-]+)?|[^\sa-z0-9]")


@dataclass(frozen=True)
class Directive:
    axis: str
    polarity: bool
    objects: frozenset[str]


def extract_directives(text: str) -> list[Directive]:
    # Keep clause punctuation as separators so objects and negations never cross ";" or ","
    raw = [t.rstrip(".") for t in _TOKEN_RE.findall(text.lower())]
    toks = [t if re.match(r"[a-z0-9]", t) else "|" for t in raw if t]
    out: list[Directive] = []
    for i, t in enumerate(toks):
        verb = _FORM_TO_VERB.get(t)
        if verb is None:
            continue
        axis, pol = _ANTONYMS[verb]
        window: list[str] = []
        for w in reversed(toks[max(0, i - 3) : i]):
            if w == "|":
                break
            window.insert(0, w)
        if any(w in _NEG for w in window) or (len(window) >= 2 and window[-2:] == ["do", "not"]):
            pol = not pol
        objs: list[str] = []
        for o in toks[i + 1 : i + 6]:
            if o == "|":
                break
            if o not in _STOP and o not in _FORM_TO_VERB:
                objs.append(o)
        obj_set: set[str] = set()
        for o in objs[:3]:
            obj_set.add(o)
            obj_set.update(p for p in re.split(r"[=<>~!]+", o) if p)  # "urllib3==2.0.7" -> both halves
        out.append(Directive(axis, pol, frozenset(obj_set)))
    # "leave/keep X open" = open (close axis, negative)
    for m in re.finditer(r"\b(?:leave|keep)\s+((?:[a-z0-9_.-]+\s+){0,3})open\b", text.lower()):
        objs = [o for o in m.group(1).split() if o not in _STOP]
        out.append(Directive("close", False, frozenset(objs[:3])))
    return out


_VERSION_RE = re.compile(
    r"\b([a-z][a-z0-9_.-]*)\s*(==|>=|<=|~=|!=|>|<)\s*(\d+(?:\.\d+)*)", re.I
)
_VERB_VERSION_RE = re.compile(
    r"\b(pin|upgrade|downgrade)\w*\s+(?:to\s+)?([a-z][a-z0-9_.-]*)\s*(?:to\s+|@\s*|==\s*)?v?(\d+(?:\.\d+)+)",
    re.I,
)
_OP_CLASS = {"==": "eq", "~=": "compat", ">=": "min", ">": "min", "<=": "max", "<": "max", "!=": "ne"}


def extract_version_constraints(text: str) -> set[tuple[str, str, str]]:
    out: set[tuple[str, str, str]] = set()
    for pkg, op, ver in _VERSION_RE.findall(text):
        out.add((pkg.lower(), _OP_CLASS[op], ver))
    for verb, pkg, ver in _VERB_VERSION_RE.findall(text):
        v = verb.lower()
        op = "eq" if v.startswith("pin") else ("min" if v.startswith("upgrade") else "max")
        out.add((pkg.lower(), op, ver))
    return out


def _supported(d: Directive, evidence: list[Directive]) -> tuple[bool, bool]:
    """(supported_same_polarity, contradicted_by_opposite_polarity) for one directive."""
    same = opposite = False
    for e in evidence:
        if e.axis != d.axis:
            continue
        if d.objects and e.objects and not (d.objects & e.objects):
            continue
        if e.polarity == d.polarity:
            same = True
        else:
            opposite = True
    return same, opposite


def check_against_evidence(text: str, evidence_texts: Iterable[str]) -> tuple[bool, str]:
    ev_list = list(evidence_texts)
    ev_dirs = [d for t in ev_list for d in extract_directives(t)]
    unsupported: str | None = None
    for d in extract_directives(text):
        same, opposite = _supported(d, ev_dirs)
        if opposite and not same:
            return False, f"polarity_conflict:{'' if d.polarity else 'not_'}{d.axis}"
        if (
            unsupported is None
            and not same
            and not opposite
            and d.axis in {"close", "pin", "upgrade", "enable", "use", "add", "verify", "increase"}
        ):
            unsupported = f"unsupported_directive:{d.axis}"
    if unsupported:
        return False, unsupported
    ev_versions = set().union(*(extract_version_constraints(t) for t in ev_list)) if ev_list else set()
    for vc in extract_version_constraints(text):
        if vc not in ev_versions:
            return False, f"unsupported_version_constraint:{vc[0]}:{vc[1]}:{vc[2]}"
    return True, "ok"


def contributor_conflict(per_contributor_texts: list[str], *, max_dissent_share: float = 0.2) -> tuple[bool, str]:
    """
    True if contributors disagree: some say "close X", others "leave X open", or they pin
    different versions of the same package. Dissent above `max_dissent_share` (or >= 2
    contributors) blocks promotion.
    """
    n = len(per_contributor_texts)
    if n == 0:
        return False, "no_contributors"
    per: list[list[Directive]] = [extract_directives(t) for t in per_contributor_texts]
    pins: dict[str, set[str]] = {}
    pin_votes: dict[tuple[str, str], int] = {}
    for t in per_contributor_texts:
        for pkg, op, ver in extract_version_constraints(t):
            if op == "eq":
                pins.setdefault(pkg, set()).add(ver)
                pin_votes[(pkg, ver)] = pin_votes.get((pkg, ver), 0) + 1
    axes = {d.axis for ds in per for d in ds}
    for axis in sorted(axes):
        pos = [i for i, ds in enumerate(per) if any(d.axis == axis and d.polarity for d in ds)]
        neg = [i for i, ds in enumerate(per) if any(d.axis == axis and not d.polarity for d in ds)]
        if not pos or not neg:
            continue
        pos_objs = frozenset().union(*(d.objects for i in pos for d in per[i] if d.axis == axis and d.polarity))
        neg_objs = frozenset().union(*(d.objects for i in neg for d in per[i] if d.axis == axis and not d.polarity))
        if pos_objs and neg_objs and not (pos_objs & neg_objs):
            continue  # same action on different things
        minority = min(len(set(pos) - set(neg)), len(set(neg) - set(pos))) or min(len(pos), len(neg))
        if minority >= 2 or minority / n > max_dissent_share:
            return True, f"contributors_disagree:{axis}"
    for pkg, vers in pins.items():
        if len(vers) > 1:
            votes = sorted((pin_votes[(pkg, v)] for v in vers), reverse=True)
            minority = sum(votes[1:])
            if minority >= 2 or minority / n > max_dissent_share:
                return True, f"contributors_pin_different_versions:{pkg}"
    return False, "consistent"
