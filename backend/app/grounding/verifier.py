"""Grounding gate (ARCH §8.3, ARCH-015).

Per claim segment: (1) citation resolves to a chunk in THIS turn's snapshot,
(2) quote is a verbatim substring of that chunk + offsets check, (3) entailment
(lexical overlap + one constrained model call), (4) scope/wording
(app.grounding.wording).

Citation resolution (DEVIATIONS.md #69): the synthesis agent labels sources
`c1..cN` in the same order as `retrieval_snapshot` when it builds the SOURCES
block (see `app.agents.guideline_synthesis_agent`), so a claim's
`citation_ids` resolve positionally against `retrieval_snapshot` without the
model needing to reproduce a chunk UUID verbatim — a real, observed
reliability failure mode for long opaque ids. The verbatim `chunk_id` is
still accepted as a fallback for a model that echoes it anyway.

Verdict policy (ARCH §8.3):
  all supported            -> release (+ disclaimer)
  only weak                -> release marked + queue for review
  unsupported, strip ok    -> partial-strip, re-check, release reduced + log stripped
  unsupported, strip bad / directive scope_violation -> ESCALATE, hold
  low-confidence/empty      -> never reaches here with claims
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from app.citations.model import build_citation, find_verbatim_quote
from app.config import get_settings
from app.grounding.wording import scan_segment
from app.ingestion.review import NOT_RETRIEVABLE
from app.schemas.citation import Citation
from app.schemas.enums import GroundingVerdict, SegmentType

EntailmentFn = Callable[[str, str], str]  # (claim_text, quote) -> "yes" | "partly" | "no"

# Lexical-overlap thresholds (ARCH §8.3 step 3; DEVIATIONS.md #69).
_LEXICAL_YES_THRESHOLD = 0.6
_LEXICAL_PARTLY_THRESHOLD = 0.3
_HYBRID_CONFIDENT_YES_THRESHOLD = 0.75
_HYBRID_CONFIDENT_NO_THRESHOLD = 0.05

_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "of",
        "to",
        "for",
        "in",
        "on",
        "and",
        "or",
        "is",
        "are",
        "with",
        "this",
        "that",
        "at",
        "by",
        "as",
        "be",
        "it",
        "was",
        "were",
    }
)


@dataclass
class SegmentVerdict:
    segment_index: int
    verdict: GroundingVerdict
    # reason: citation_not_retrieved | quote_mismatch | chunk_under_review |
    # not_entailed | scope_violation
    reason: str | None = None
    # For a SUPPORTED/WEAK claim segment (DEVIATIONS.md #107): the subset of
    # the segment's own citation_ids whose chunk actually contains the quote
    # — a segment is kept if ANY cited id verifies, but that does not mean
    # every cited id does. `None` for framing segments / unsupported verdicts,
    # where it's unused.
    verified_citation_ids: list[str] | None = None


@dataclass
class GroundingReport:
    per_segment: list[SegmentVerdict]
    action: str  # release | release_marked | partial_strip | escalate
    stripped_segment_indexes: list[int] = field(default_factory=list)


def _label_maps(retrieval_snapshot: list[dict]) -> tuple[dict[str, dict], dict[str, dict]]:
    by_label = {f"c{i + 1}": item for i, item in enumerate(retrieval_snapshot)}
    by_id = {item["chunk_id"]: item for item in retrieval_snapshot if item.get("chunk_id")}
    return by_label, by_id


def _resolve_chunk(citation_id: str, by_label: dict, by_id: dict) -> dict | None:
    return by_label.get(citation_id) or by_id.get(citation_id)


_MIN_WORD_LENGTH = 3


def _words(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in tokens if w not in _STOPWORDS and len(w) >= _MIN_WORD_LENGTH}


def lexical_overlap(claim_text: str, quote: str) -> float:
    """Fraction of the claim's content words also present in the quote."""
    claim_words = _words(claim_text)
    if not claim_words:
        return 0.0
    return len(claim_words & _words(quote)) / len(claim_words)


def _lexical_entailment(claim_text: str, quote: str) -> str:
    overlap = lexical_overlap(claim_text, quote)
    if overlap >= _LEXICAL_YES_THRESHOLD:
        return "yes"
    if overlap >= _LEXICAL_PARTLY_THRESHOLD:
        return "partly"
    return "no"


def _entailment_verdict(
    claim_text: str, quote: str, *, mode: str, entailment_fn: EntailmentFn | None
) -> str:
    """`mode` is `GROUNDING_ENTAILMENT_MODE` (lexical | model | hybrid).

    `hybrid` trusts a confident lexical read (very high or very low overlap)
    and only spends a model call on the ambiguous middle — the deterministic
    lexical check plus a constrained model call ARCH §8.3 step 3 calls for,
    applied selectively rather than on every claim, to avoid a model round
    trip on unambiguous cases (DEVIATIONS.md #69). Without an `entailment_fn`
    (e.g. an offline caller with no gateway attached), falls back to
    lexical-only regardless of `mode`.
    """
    if entailment_fn is None or mode == "lexical":
        return _lexical_entailment(claim_text, quote)
    if mode == "model":
        return entailment_fn(claim_text, quote)
    overlap = lexical_overlap(claim_text, quote)
    if overlap >= _HYBRID_CONFIDENT_YES_THRESHOLD:
        return "yes"
    if overlap <= _HYBRID_CONFIDENT_NO_THRESHOLD:
        return "no"
    return entailment_fn(claim_text, quote)


def _verify_framing_segment(idx: int, seg: dict) -> SegmentVerdict:
    reasons = scan_segment(seg.get("text", ""), cited_quotes=[])
    if reasons:
        return SegmentVerdict(idx, GroundingVerdict.UNSUPPORTED, "scope_violation")
    return SegmentVerdict(idx, GroundingVerdict.SUPPORTED, None)


def _under_review(chunk: dict | None) -> bool:
    if not chunk:
        return False
    return (chunk.get("meta") or {}).get("review_status") in NOT_RETRIEVABLE


def _weak_support_only(chunk: dict | None) -> bool:
    """A chunk that can't be a claim's sole full support (ARCH §8.3 rule 5):
    a figure with no text layer or OCR text, or a flowchart whose edges are
    not all verified."""
    if not chunk:
        return True
    meta = chunk.get("meta") or {}
    if chunk.get("chunk_type") == "figure":
        return not meta.get("has_embedded_text", False)
    if chunk.get("chunk_type") == "flowchart":
        return (meta.get("flowchart") or {}).get("verification") != "verified"
    return False


def _verify_claim_segment(
    idx: int,
    seg: dict,
    by_label: dict,
    by_id: dict,
    *,
    mode: str,
    entailment_fn: EntailmentFn | None,
) -> SegmentVerdict:
    text = seg.get("text", "")
    quote = seg.get("quote") or ""
    citation_ids = seg.get("citation_ids") or []
    chunks = [_resolve_chunk(cid, by_label, by_id) for cid in citation_ids]

    if not citation_ids or any(c is None for c in chunks):
        return SegmentVerdict(idx, GroundingVerdict.UNSUPPORTED, "citation_not_retrieved")

    verified_ids = [
        cid
        for cid, c in zip(citation_ids, chunks, strict=True)
        if c and quote and find_verbatim_quote(quote, c.get("text", ""))
    ]
    # Defence in depth (ARCH-044): a chunk held for review (OCR'd numbers,
    # low parse quality) or rejected by a reviewer is excluded from retrieval
    # and so should never be here — but if it is, it can't support a claim.
    by_cid = dict(zip(citation_ids, chunks, strict=True))
    held = [cid for cid in verified_ids if _under_review(by_cid[cid])]
    verified_ids = [cid for cid in verified_ids if cid not in held]
    if not verified_ids:
        reason = "chunk_under_review" if held else "quote_mismatch"
        return SegmentVerdict(idx, GroundingVerdict.UNSUPPORTED, reason)

    if scan_segment(text, cited_quotes=[quote]):
        return SegmentVerdict(idx, GroundingVerdict.UNSUPPORTED, "scope_violation")

    verdict = _entailment_verdict(text, quote, mode=mode, entailment_fn=entailment_fn)
    # ARCH §8.3 rule 5 (+ ARCH-044): when every chunk supporting the claim is
    # a caption-only figure or a flowchart whose structure isn't verified,
    # the claim is at most `weak` and never released as fully supported.
    if verdict == "yes" and all(_weak_support_only(by_cid[cid]) for cid in verified_ids):
        verdict = "partly"
    if verdict == "no":
        return SegmentVerdict(idx, GroundingVerdict.UNSUPPORTED, "not_entailed")
    if verdict == "partly":
        return SegmentVerdict(idx, GroundingVerdict.WEAK, None, verified_citation_ids=verified_ids)
    return SegmentVerdict(idx, GroundingVerdict.SUPPORTED, None, verified_citation_ids=verified_ids)


def _decide_action(per_segment: list[SegmentVerdict], claim_indexes: set[int]) -> GroundingReport:
    scope_violation = any(v.reason == "scope_violation" for v in per_segment)
    unsupported_non_scope = [
        v.segment_index
        for v in per_segment
        if v.segment_index in claim_indexes
        and v.verdict == GroundingVerdict.UNSUPPORTED
        and v.reason != "scope_violation"
    ]

    if scope_violation:
        return GroundingReport(per_segment=per_segment, action="escalate")

    if not unsupported_non_scope:
        weak = any(v.verdict == GroundingVerdict.WEAK for v in per_segment)
        action = "release_marked" if weak else "release"
        return GroundingReport(per_segment=per_segment, action=action)

    ok_verdicts = (GroundingVerdict.SUPPORTED, GroundingVerdict.WEAK)
    remaining_ok = any(
        v.segment_index in claim_indexes and v.verdict in ok_verdicts for v in per_segment
    )
    if remaining_ok:
        return GroundingReport(
            per_segment=per_segment,
            action="partial_strip",
            stripped_segment_indexes=unsupported_non_scope,
        )
    return GroundingReport(per_segment=per_segment, action="escalate")


def verify(
    segments: list[dict],
    retrieval_snapshot: list[dict],
    *,
    entailment_fn: EntailmentFn | None = None,
) -> GroundingReport:
    settings = get_settings()
    by_label, by_id = _label_maps(retrieval_snapshot)
    per_segment: list[SegmentVerdict] = []
    claim_indexes: set[int] = set()

    for idx, seg in enumerate(segments):
        if seg.get("type") == SegmentType.FRAMING:
            per_segment.append(_verify_framing_segment(idx, seg))
            continue
        claim_indexes.add(idx)
        per_segment.append(
            _verify_claim_segment(
                idx,
                seg,
                by_label,
                by_id,
                mode=settings.grounding_entailment_mode,
                entailment_fn=entailment_fn,
            )
        )

    return _decide_action(per_segment, claim_indexes)


def citations_for_segments(
    segments: list[dict], retrieval_snapshot: list[dict]
) -> dict[str, Citation]:
    """Build one `Citation` per distinct `(citation_id, quote)` referenced by a
    claim segment, keyed by the citation_id label used in the answer text.
    Only called for segments the grounding gate has already accepted
    (release / release_marked / the retained half of a partial_strip)."""
    by_label, by_id = _label_maps(retrieval_snapshot)
    out: dict[str, Citation] = {}
    for seg in segments:
        if seg.get("type") != SegmentType.CLAIM:
            continue
        quote = seg.get("quote") or ""
        for cid in seg.get("citation_ids") or []:
            if cid in out:
                continue
            chunk = _resolve_chunk(cid, by_label, by_id)
            if chunk is None or not find_verbatim_quote(quote, chunk.get("text", "")):
                continue
            out[cid] = build_citation(cid, chunk, quote)
    return out
