"""Conflict detection between retrieved sources (ARCH §7.6; PRD-014).

Flags material disagreement: (a) the same section of the same document in two
`active` versions with different recommendation text (DEVIATIONS.md #261), or (b) a lightweight
NLI/contradiction pass between top recommendation chunks. Any flag =>
escalation (conflicting_sources); both sides are surfaced with citations, never
auto-resolved. Starts conservative (over-flag), tuned on the eval set
(self-critique §21a).

(b) is a lexical heuristic, not a real NLI model call (DEVIATIONS.md #53): two
`recommendation` chunks that share enough vocabulary to be "about the same
topic" but disagree on negation ("recommended" vs "not recommended") are
flagged. This intentionally over-flags rather than under-flags — a missed
contradiction is a safety defect, a false positive just costs a review.
"""

from __future__ import annotations

import difflib
import re
from itertools import combinations

from app.agents.state import RetrievalItem
from app.retrieval.sparse import analyze

_NEGATION_RE = re.compile(
    r"\b(not|no longer|avoid|contraindicated|should not|must not|is not|are not|discontinued)\b",
    re.IGNORECASE,
)
_MATERIAL_DIFFERENCE_MAX_RATIO = 0.85  # below this similarity ratio => "different text"
_MIN_SHARED_TOKENS_FOR_SAME_TOPIC = 3


def _materially_different(a: str, b: str) -> bool:
    ratio = difflib.SequenceMatcher(None, a.strip().lower(), b.strip().lower()).ratio()
    return ratio < _MATERIAL_DIFFERENCE_MAX_RATIO


def _lexically_contradicts(a: str, b: str) -> bool:
    shared = set(analyze(a)) & set(analyze(b))
    if len(shared) < _MIN_SHARED_TOKENS_FOR_SAME_TOPIC:
        return False
    return bool(_NEGATION_RE.search(a)) != bool(_NEGATION_RE.search(b))


def detect_conflicts(items: list[RetrievalItem]) -> list[dict]:
    flags: list[dict] = []

    # (a) same section of the SAME document in two different active versions,
    # with different text. Keyed on (document_id, section_number): before
    # DEVIATIONS.md #261 it was keyed on section_number alone, so two
    # different guidelines that merely number a section alike (Kenya "1.10"
    # vs NICE "1.10", MoH "5" vs WHO "5") were flagged as conflicting
    # versions -- every conflict flag in the four-area trial was this.
    by_section: dict[tuple[str, str], list[RetrievalItem]] = {}
    for item in items:
        section_number = item.get("section_number")
        if item.get("version_status") == "active" and section_number:
            key = (str(item.get("document_id")), section_number)
            by_section.setdefault(key, []).append(item)
    for (_doc, section_number), group in by_section.items():
        for a, b in combinations(group, 2):
            if a["document_version_id"] == b["document_version_id"]:
                continue
            if _materially_different(a["text"], b["text"]):
                flags.append(
                    {
                        "chunk_id_a": a["chunk_id"],
                        "chunk_id_b": b["chunk_id"],
                        "reason": "same_section_diff_versions",
                        "section_number": section_number,
                    }
                )

    # (b) lexical contradiction pass over recommendation chunks.
    recs = [it for it in items if it.get("chunk_type") == "recommendation"]
    for a, b in combinations(recs, 2):
        if a["chunk_id"] == b["chunk_id"]:
            continue
        if _lexically_contradicts(a["text"], b["text"]):
            flags.append(
                {
                    "chunk_id_a": a["chunk_id"],
                    "chunk_id_b": b["chunk_id"],
                    "reason": "lexical_contradiction",
                }
            )

    return flags
