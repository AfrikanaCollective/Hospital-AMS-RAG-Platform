"""Build + re-verify Citation objects (ARCH §8.1; PRD-016).

`build_citation` is a pure function with no agent dependency — it turns a
retrieved `chunk_row` (the dict shape returned by `app.retrieval.hybrid.retrieve`,
carrying document/version identity + section/page/char span) plus a chosen
`quote` into a `Citation`. Pulled into Phase 2 (DEVIATIONS.md #46) rather than
left for Phase 3: unlike the grounding gate (`app.grounding.verifier.verify`)
and answer segmentation (`app.grounding.segments`), which require the Phase-3
synthesis agent's claim segments as input, this only needs retrieval output.
The synthesis agent (Phase 3) is still what decides *which* quote supports a
given claim; this function just assembles the citation object once that quote
is chosen.

`verify_citation` re-checks a citation against stored chunk text:
- the quote is a verbatim substring of the cited chunk,
- quote offsets are consistent with chunk offsets,
- chunk offsets match the normalized document text.
This is deterministic and is part of the grounding gate (step 2).

`find_verbatim_quote` (DEVIATIONS.md #104) is the one shared "is this
actually verbatim" primitive — used here and in `app.grounding.verifier`'s
three other quote-matching call sites, so "verbatim" means the same thing
everywhere rather than four independent implementations drifting apart.
Whitespace-tolerant, nothing else: real PDF-extracted chunk text routinely
has a line-wrap newline in the middle of a sentence (an artifact of the
source PDF's layout, not a semantic break), which a model naturally
normalizes to a single space when it reproduces a quote — a byte-exact
`in`/`.find()` check then wrongly rejects an otherwise word-for-word,
in-order, nothing-added-or-changed quote. This tolerates only that: every
non-whitespace character must still match exactly, in the same order: no
paraphrase, no reordering, no synonym or fuzzy tolerance. Confirmed against
a real ingested PDF chunk with embedded line-wrap newlines and a real
model-produced quote that failed the old byte-exact check purely because of
this (DEVIATIONS.md #104).
"""

from __future__ import annotations

import re

from app.schemas.citation import Citation
from app.schemas.enums import DocumentVersionStatus


def find_verbatim_quote(quote: str, text: str) -> tuple[int, int] | None:
    """Whitespace-tolerant substring search: each run of whitespace in
    `quote` matches any run of whitespace in `text` (including none-vs-some,
    e.g. a hyphenated line-wrap); every other character must match exactly,
    in order. Returns the actual `(start, end)` span in `text`'s own
    coordinates — which may be longer than `len(quote)` if `text` had extra
    whitespace there — or `None` if no match. Empty/whitespace-only `quote`
    never matches (an empty pattern would match everywhere)."""
    words = quote.split()
    if not words:
        return None
    pattern = r"\s+".join(re.escape(w) for w in words)
    match = re.search(pattern, text)
    return (match.start(), match.end()) if match else None


def _normalize_ws(s: str) -> str:
    return " ".join(s.split())


def corrections_for_quote(chunk_row: dict, quote: str) -> list[dict]:
    """The chunk's attested corrections (`meta.corrections`, ARCH-044) whose
    corrected text overlaps `quote`'s span in the chunk text."""
    corrections = (chunk_row.get("meta") or {}).get("corrections") or []
    if not corrections:
        return []
    text = chunk_row["text"]
    q = find_verbatim_quote(quote, text)
    if q is None:
        return []
    out = []
    for corr in corrections:
        c = find_verbatim_quote(corr.get("corrected", ""), text)
        if c is not None and c[0] < q[1] and q[0] < c[1]:
            out.append(dict(corr))
    return out


def build_citation(citation_id: str, chunk_row: dict, quote: str) -> Citation:
    """Build a `Citation` for `quote` found verbatim (whitespace-tolerant,
    `find_verbatim_quote`) in `chunk_row["text"]`.

    `chunk_row` carries: chunk_id, text, char_start, char_end, page_start,
    page_end, section_number, section_path, document_id, document_title,
    document_version_id, version_label, effective_date, version_status.

    Raises `ValueError` if `quote` is not a substring of the chunk's text —
    a citation is never built for an unsupported quote (ARCH §8.1).
    """
    text = chunk_row["text"]
    span = find_verbatim_quote(quote, text)
    if span is None:
        raise ValueError(f"quote is not a substring of chunk {chunk_row.get('chunk_id')!r} text")
    idx, end = span
    char_start = chunk_row["char_start"]
    return Citation(
        citation_id=citation_id,
        document_id=chunk_row["document_id"],
        document_title=chunk_row["document_title"],
        document_version_id=chunk_row["document_version_id"],
        version_label=chunk_row["version_label"],
        effective_date=chunk_row.get("effective_date"),
        version_status=DocumentVersionStatus(chunk_row.get("version_status", "active")),
        chunk_id=chunk_row["chunk_id"],
        section_number=chunk_row.get("section_number"),
        section_path=chunk_row.get("section_path"),
        page_start=chunk_row["page_start"],
        page_end=chunk_row["page_end"],
        char_start=char_start,
        char_end=chunk_row["char_end"],
        quote=quote,
        quote_char_start=char_start + idx,
        quote_char_end=char_start + end,
        corrections=corrections_for_quote(chunk_row, quote),
    )


def verify_citation(citation: Citation, chunk_text: str) -> bool:
    if not (0 <= citation.quote_char_start <= citation.quote_char_end):
        return False
    local_start = citation.quote_char_start - citation.char_start
    local_end = citation.quote_char_end - citation.char_start
    if local_start < 0 or local_end > len(chunk_text):
        return False
    excerpt = chunk_text[local_start:local_end]
    return (
        _normalize_ws(excerpt) == _normalize_ws(citation.quote)
        and find_verbatim_quote(citation.quote, chunk_text) is not None
    )
