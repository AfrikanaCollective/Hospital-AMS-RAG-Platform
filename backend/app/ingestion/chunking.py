"""Structure-aware, format-profile-aware chunking (ARCH §6, ARCH-013; PRD-004).

Rules, in priority order (ARCH §6):
0. `format_profile` (`grade_recommendations` | `clinical_protocol` | `narrative`)
   selects the atomic unit for rules 1/1b.
1. `grade_recommendations`: a numbered clause carrying a GRADE strength/
   certainty marker is one `recommendation` chunk (soft cap 1024 tokens; over
   cap -> split at sentence boundaries, tag `split_group_id`).
1b. `clinical_protocol`: a numbered clause is one `protocol_step` chunk (same
    atomicity, no GRADE marker required).
2. Otherwise: window prose within a section, target 350-600 tokens, ~15%
   overlap between adjacent prose chunks (overlap carries no citation
   authority).
3. A markdown-table block is one `table` chunk (caption/heading prepended).
3b. Figures: `.pdf` only (best-effort — see the module docstring's limitation
    note); one `figure` chunk per detected image.
4. A table whose nearest heading/caption mentions "criteria" becomes a
   `criteria` chunk instead, with best-effort `meta.criteria[]` extraction.
5. `meta.embedding_text` carries the section_path breadcrumb prefix for
   embedding; `text` stays the raw verbatim slice for citation display.
6. `parent_ordinal` (resolved to `parent_chunk_id` at persist time) links each
   chunk to its section's first chunk (or, for a table/figure inside an open
   protocol step, to that step's chunk).

**Limitations, flagged per CLAUDE.md §2 rather than silently worked around**
(DEVIATIONS.md #47):
- `token_count` is a whitespace-word-count proxy, not a real tokenizer count —
  no tokenizer dependency is committed for this. Close enough to bucket into
  the 350-600 target but not exact.
- Figure detection for `.pdf` uses `pypdf`'s embedded-image list, which carries
  no bounding box; `figure_ref.bbox` is always `null` here (ARCH §6 rule 3b
  still wants `{page, bbox, image_sha256}` — `bbox` is the unavailable part).
- Criteria `meta.criteria[]` extraction is a simple per-cell operator regex
  (`>=`, `<=`, `>`, `<`, `=`), not a general parser — "where the text is
  regular enough" (ARCH §6 rule 4) is interpreted narrowly.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from app.ingestion.pdf_parse import Block, ParsedDocument

logger = logging.getLogger(__name__)

RECOMMENDATION_SOFT_CAP_TOKENS = 1024
PROSE_TARGET_MIN_TOKENS = 350
PROSE_TARGET_MAX_TOKENS = 600
PROSE_OVERLAP_FRACTION = 0.15
_MIN_TABLE_ROWS = 2
_MIN_TABLE_ROW_MATCHES = 2

_GRADE_MARKER_RE = re.compile(
    r"\b(strong|weak|conditional)\s+recommendation\b|\b(certainty|quality)\s+of\s+evidence\b",
    re.IGNORECASE,
)
_NUMBERED_START_RE = re.compile(r"^(?P<num>\d+(?:\.\d+){1,4})\.?\s+(?P<rest>.*)$", re.DOTALL)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:|-]+\|[\s:|-]*\|?\s*$")
_FIGURE_CAPTION_RE = re.compile(r"\bfig(?:ure)?\.?\s*\d+\b", re.IGNORECASE)
_CRITERIA_CELL_RE = re.compile(
    r"(?P<field>[A-Za-z_ ]+?)\s*(?P<op>>=|<=|>|<|=)\s*(?P<value>[\d.]+)\s*(?P<unit>[A-Za-z%/]*)"
)


def _approx_tokens(text: str) -> int:
    return max(1, len(text.split()))


@dataclass
class _Chunk:
    ordinal: int
    section_path: str | None
    section_number: str | None
    heading: str | None
    page_start: int
    page_end: int
    char_start: int
    char_end: int
    chunk_type: str
    text: str
    figure_ref: dict | None = None
    parent_ordinal: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        embedding_text = f"{self.section_path}\n\n{self.text}" if self.section_path else self.text
        # Retrieval-only additions (table row renderings, flowchart path
        # summaries — deterministic, never model-written). They help a query
        # find the chunk; they are not citable (ARCH-044 text-provenance rule).
        meta = dict(self.meta)
        extra = meta.pop("embedding_extra", None)
        if extra:
            embedding_text = f"{embedding_text}\n\n{extra}"
        return {
            "ordinal": self.ordinal,
            "section_path": self.section_path,
            "section_number": self.section_number,
            "heading": self.heading,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "chunk_type": self.chunk_type,
            "text": self.text,
            "figure_ref": self.figure_ref,
            "token_count": _approx_tokens(self.text),
            "parent_ordinal": self.parent_ordinal,
            "meta": {**meta, "embedding_text": embedding_text},
        }


def _split_paragraphs(text: str, base_offset: int) -> list[tuple[str, int, int]]:
    """Return [(paragraph_text, char_start, char_end)] for blank-line-separated
    paragraphs, skipping empty ones. Offsets are absolute (relative to
    `base_offset`, i.e. the containing section's char_start)."""
    out: list[tuple[str, int, int]] = []
    pos = 0
    for block in re.split(r"\n\s*\n", text):
        start_in_block = text.index(block, pos) if block else pos
        stripped = block.strip()
        if stripped:
            abs_start = base_offset + start_in_block + (len(block) - len(block.lstrip()))
            out.append((stripped, abs_start, abs_start + len(stripped)))
        pos = start_in_block + len(block)
    return out


def _is_table_block(paragraph: str) -> bool:
    lines = [ln for ln in paragraph.splitlines() if ln.strip()]
    if len(lines) < _MIN_TABLE_ROWS:
        return False
    row_matches = sum(1 for ln in lines if _TABLE_ROW_RE.match(ln))
    return row_matches >= _MIN_TABLE_ROW_MATCHES and any(_TABLE_SEP_RE.match(ln) for ln in lines)


def _extract_criteria(paragraph: str) -> list[dict]:
    criteria = []
    for m in _CRITERIA_CELL_RE.finditer(paragraph):
        criteria.append(
            {
                "field": m.group("field").strip(" |"),
                "operator": m.group("op"),
                "value": float(m.group("value")),
                "unit": m.group("unit") or None,
            }
        )
    return criteria


def _split_oversized_recommendation(text: str, max_tokens: int) -> list[str]:
    if _approx_tokens(text) <= max_tokens:
        return [text]
    sentences = _SENTENCE_SPLIT_RE.split(text)
    parts: list[str] = []
    current: list[str] = []
    count = 0
    for sent in sentences:
        t = _approx_tokens(sent)
        if current and count + t > max_tokens:
            parts.append(" ".join(current))
            current, count = [], 0
        current.append(sent)
        count += t
    if current:
        parts.append(" ".join(current))
    return parts or [text]


def _window_prose(paragraphs: list[tuple[str, int, int]]) -> list[tuple[str, int, int]]:
    """Merge paragraphs into 350-600 "token" windows with ~15% overlap between
    adjacent windows. Returns [(text, char_start, char_end)]."""
    if not paragraphs:
        return []
    windows: list[tuple[str, int, int]] = []
    cur_texts: list[str] = []
    cur_start = paragraphs[0][1]
    cur_tokens = 0
    prev_tail_words: list[str] = []

    def flush(end: int) -> None:
        nonlocal cur_texts, cur_tokens, prev_tail_words
        if not cur_texts:
            return
        body = "\n\n".join(cur_texts)
        text = (" ".join(prev_tail_words) + "\n\n" + body) if prev_tail_words else body
        start = cur_start
        if prev_tail_words:
            overlap_str = " ".join(prev_tail_words)
            start = cur_start - len(overlap_str) - 2  # best-effort backward offset
        windows.append((text, max(start, 0), end))
        words = body.split()
        n_overlap = max(1, round(len(words) * PROSE_OVERLAP_FRACTION))
        prev_tail_words = words[-n_overlap:]
        cur_texts, cur_tokens = [], 0

    for i, (para, pstart, _pend) in enumerate(paragraphs):
        t = _approx_tokens(para)
        if cur_texts and cur_tokens + t > PROSE_TARGET_MAX_TOKENS:
            flush(paragraphs[i - 1][2])
            cur_start = pstart
        cur_texts.append(para)
        cur_tokens += t
    flush(paragraphs[-1][2])
    return windows


class _SectionBuilder:
    """Mutable state for chunking one section: accumulated chunks, the
    running ordinal counter, the section's first chunk (parent anchor for
    everything else in the section), and any currently-open protocol step
    (parent anchor for a table/figure bound to that step)."""

    def __init__(self, section: dict, chunks: list[_Chunk], start_ordinal: int) -> None:
        self.section = section
        self.chunks = chunks
        self.ordinal = start_ordinal
        self.section_first_ordinal: int | None = None
        self.open_step_ordinal: int | None = None
        self.prose_buffer: list[tuple[str, int, int]] = []

    def _register(self, ordinal: int) -> None:
        if self.section_first_ordinal is None:
            self.section_first_ordinal = ordinal

    def _emit(
        self,
        *,
        chunk_type: str,
        text: str,
        char_start: int,
        char_end: int,
        page_start: int,
        page_end: int,
        parent_ordinal: int | None,
        meta: dict | None = None,
        figure_ref: dict | None = None,
    ) -> int:
        ordinal = self.ordinal
        self.chunks.append(
            _Chunk(
                ordinal=ordinal,
                section_path=self.section["path"],
                section_number=self.section["number"],
                heading=self.section["heading"],
                page_start=page_start,
                page_end=page_end,
                char_start=char_start,
                char_end=char_end,
                chunk_type=chunk_type,
                text=text,
                figure_ref=figure_ref,
                parent_ordinal=parent_ordinal,
                meta=meta or {},
            )
        )
        self._register(ordinal)
        self.ordinal += 1
        return ordinal

    def flush_prose(self, doc: ParsedDocument) -> None:
        for text, cstart, cend in _window_prose(self.prose_buffer):
            self._emit(
                chunk_type="prose",
                text=text,
                char_start=cstart,
                char_end=cend,
                page_start=doc.page_for_offset(cstart),
                page_end=doc.page_for_offset(max(cend - 1, cstart)),
                parent_ordinal=self.section_first_ordinal,
            )
        self.prose_buffer = []

    def emit_table_or_criteria(
        self, para: str, pstart: int, pend: int, doc: ParsedDocument
    ) -> None:
        heading = self.section.get("heading") or ""
        heading_mentions_criteria = "criteria" in heading.lower()
        chunk_type = "criteria" if heading_mentions_criteria else "table"
        meta = {"criteria": _extract_criteria(para)} if chunk_type == "criteria" else {}
        text = f"{heading}\n\n{para}" if heading else para
        self._emit(
            chunk_type=chunk_type,
            text=text,
            char_start=pstart,
            char_end=pend,
            page_start=doc.page_for_offset(pstart),
            page_end=doc.page_for_offset(max(pend - 1, pstart)),
            parent_ordinal=self.open_step_ordinal
            if self.open_step_ordinal is not None
            else self.section_first_ordinal,
            meta=meta,
        )

    def emit_atomic_clause(
        self, para: str, pstart: int, pend: int, doc: ParsedDocument, *, chunk_type: str
    ) -> None:
        page_start = doc.page_for_offset(pstart)
        page_end = doc.page_for_offset(max(pend - 1, pstart))
        pieces = _split_oversized_recommendation(para, RECOMMENDATION_SOFT_CAP_TOKENS)
        split_group_id = (
            hashlib.sha256(para.encode("utf-8")).hexdigest()[:16] if len(pieces) > 1 else None
        )
        cursor = pstart
        first_ordinal_in_group: int | None = None
        for piece in pieces:
            piece_start = para.index(piece, cursor - pstart) + pstart
            piece_end = piece_start + len(piece)
            meta = {"split_group_id": split_group_id} if split_group_id else {}
            ordinal = self._emit(
                chunk_type=chunk_type,
                text=piece,
                char_start=piece_start,
                char_end=piece_end,
                page_start=page_start,
                page_end=page_end,
                parent_ordinal=self.section_first_ordinal,
                meta=meta,
            )
            if first_ordinal_in_group is None:
                first_ordinal_in_group = ordinal
            cursor = piece_end
        if chunk_type == "protocol_step":
            self.open_step_ordinal = first_ordinal_in_group


def _atomic_chunk_type(para: str, format_profile: str) -> str | None:
    """Return the atomic chunk type for a numbered clause under this format
    profile, or None if it should fall through to prose (ARCH §6 rules 1/1b)."""
    if not _NUMBERED_START_RE.match(para):
        return None
    if format_profile == "grade_recommendations":
        return "recommendation" if _GRADE_MARKER_RE.search(para) else None
    if format_profile == "clinical_protocol":
        return "protocol_step"
    return None


def _chunk_section(
    section: dict,
    doc: ParsedDocument,
    format_profile: str,
    chunks: list[_Chunk],
    start_ordinal: int,
) -> int:
    sec_start, sec_end = section["char_start"], section["char_end"]
    body = doc.normalized_text[sec_start:sec_end]
    if section.get("heading") and "\n" in body:
        newline_at = body.index("\n")
        body_offset = sec_start + newline_at + 1
        body = body[newline_at + 1 :]
    elif section.get("heading"):
        body, body_offset = "", sec_end
    else:
        body_offset = sec_start

    builder = _SectionBuilder(section, chunks, start_ordinal)
    for para, pstart, pend in _split_paragraphs(body, body_offset):
        if _is_table_block(para):
            builder.flush_prose(doc)
            builder.emit_table_or_criteria(para, pstart, pend, doc)
            continue
        chunk_type = _atomic_chunk_type(para, format_profile)
        if chunk_type is not None:
            builder.flush_prose(doc)
            builder.emit_atomic_clause(para, pstart, pend, doc, chunk_type=chunk_type)
            continue
        builder.prose_buffer.append((para, pstart, pend))
    builder.flush_prose(doc)
    return builder.ordinal


def _block_provenance(blocks: list[Block]) -> dict:
    """Chunk-level provenance summary + the OCR review gate (ARCH-044;
    LAYOUT-INGESTION-PROPOSAL.md §3, §8). A chunk containing OCR text with
    digits is held (`review_status = pending`) until an admin confirms it
    against the page crop: OCR'd numbers (doses) are the highest-risk
    extraction."""
    origins = sorted({b.origin for b in blocks})
    meta: dict[str, Any] = {"text_origins": origins}
    ocr = [b.meta["ocr"] for b in blocks if b.meta.get("ocr")]
    if ocr:
        confs = [o["min_confidence"] for o in ocr if o.get("min_confidence") is not None]
        meta["ocr"] = {
            "min_confidence": min(confs) if confs else None,
            "has_digits": any(o.get("has_digits") for o in ocr),
        }
        if meta["ocr"]["has_digits"]:
            meta["review_status"] = "pending"
            meta.setdefault("review_reasons", []).append("ocr_numeric")
    # D12 (proposal §18.7): a vision-LLM transcription is always held until an
    # admin confirms it against the crop, whether or not it agreed with OCR.
    if any(b.origin == "vlm_transcription" for b in blocks):
        meta["review_status"] = "pending"
        meta.setdefault("review_reasons", []).append("vlm_transcription")
        for b in blocks:
            for key in ("vlm", "cell_diff", "ocr_alternative"):
                if key in b.meta:
                    meta[key] = b.meta[key]
    corrections = [c for b in blocks for c in b.meta.get("corrections", [])]
    if corrections:
        meta["corrections"] = [
            {k: v for k, v in c.items() if k not in ("start", "end")} for c in corrections
        ]
    return meta


def _flowchart_path_summary(graph_meta: dict) -> str:
    """Deterministic, retrieval-only walk of the verified edges, using each
    node's first line ("Has ONE of the following —Yes→ Severe neonatal
    sepsis"). Built from the graph, never by a model."""
    first = {
        n["id"]: n["text"].splitlines()[0] if n["text"] else n["id"] for n in graph_meta["nodes"]
    }
    lines = []
    for e in graph_meta["edges"]:
        if not e["verified"]:
            continue
        arrow = f" —{e['label']}→ " if e.get("label") else " → "
        lines.append(f"{first[e['from']]}{arrow}{first[e['to']]}")
    return "\n".join(lines)


class _LayoutSectionBuilder(_SectionBuilder):
    """Emits atomic table / flowchart / figure chunks from layout blocks
    (ARCH §6 rules 3, 3b; ARCH-044), alongside the unchanged prose and
    atomic-clause handling inherited from `_SectionBuilder`."""

    def __init__(self, section: dict, chunks: list[_Chunk], start_ordinal: int) -> None:
        super().__init__(section, chunks, start_ordinal)
        self.prose_blocks: list[Block] = []

    def flush_prose(self, doc: ParsedDocument) -> None:
        blocks, self.prose_blocks = self.prose_blocks, []
        for text, cstart, cend in _window_prose(self.prose_buffer):
            covered = [b for b in blocks if b.char_start < cend and cstart < b.char_end]
            self._emit(
                chunk_type="prose",
                text=text,
                char_start=cstart,
                char_end=cend,
                page_start=doc.page_for_offset(cstart),
                page_end=doc.page_for_offset(max(cend - 1, cstart)),
                parent_ordinal=self.section_first_ordinal,
                meta=_block_provenance(covered or blocks),
            )
        self.prose_buffer = []

    def _anchor(self) -> int | None:
        return (
            self.open_step_ordinal
            if self.open_step_ordinal is not None
            else self.section_first_ordinal
        )

    def emit_block(self, block: Block, doc: ParsedDocument) -> None:
        text = doc.normalized_text[block.char_start : block.char_end]
        page = doc.page_for_offset(block.char_start)
        base = _block_provenance([block])
        figure_ref = None
        if block.meta.get("figure_ref"):
            figure_ref = {"page": page, **block.meta["figure_ref"]}
        if block.kind == "table":
            heading = self.section.get("heading") or ""
            is_criteria = "criteria" in heading.lower()
            parts = block.meta.get("table_parts") or [text]
            split_group_id = (
                hashlib.sha256(text.encode("utf-8")).hexdigest()[:16] if len(parts) > 1 else None
            )
            cursor = block.char_start
            for i, part in enumerate(parts):
                start = doc.normalized_text.index(part, cursor)
                end = start + len(part)
                meta = dict(base)
                meta["embedding_extra"] = (
                    "\n".join(block.meta.get("row_texts", [])) if i == 0 else None
                )
                if split_group_id:
                    meta.update(split_group_id=split_group_id, table_part=i)
                if is_criteria:
                    meta["criteria"] = _extract_criteria(part)
                if block.meta.get("caption"):
                    meta["caption"] = block.meta["caption"]
                self._emit(
                    chunk_type="criteria" if is_criteria else "table",
                    text=part,
                    char_start=start,
                    char_end=end,
                    page_start=page,
                    page_end=doc.page_for_offset(max(end - 1, start)),
                    parent_ordinal=self._anchor(),
                    meta=meta,
                    figure_ref=figure_ref if i == 0 else None,
                )
                cursor = end
            return
        if block.kind == "flowchart":
            graph = block.meta["flowchart"]
            meta = {
                **base,
                "flowchart": graph,
                "embedding_extra": _flowchart_path_summary(graph),
            }
            if block.meta.get("caption"):
                meta["caption"] = block.meta["caption"]
            self._emit(
                chunk_type="flowchart",
                text=text,
                char_start=block.char_start,
                char_end=block.char_end,
                page_start=page,
                page_end=doc.page_for_offset(max(block.char_end - 1, block.char_start)),
                parent_ordinal=self._anchor(),
                meta=meta,
                figure_ref=figure_ref,
            )
            return
        # figure
        meta = {**base, "has_embedded_text": bool(block.meta.get("has_embedded_text"))}
        self._emit(
            chunk_type="figure",
            text=text,
            char_start=block.char_start,
            char_end=block.char_end,
            page_start=page,
            page_end=page,
            parent_ordinal=self._anchor(),
            meta=meta,
            figure_ref=figure_ref,
        )


_ATOMIC_BLOCK_KINDS = frozenset({"table", "flowchart", "figure"})


def _chunk_section_layout(
    section: dict,
    doc: ParsedDocument,
    format_profile: str,
    chunks: list[_Chunk],
    start_ordinal: int,
) -> int:
    assert doc.blocks is not None
    builder = _LayoutSectionBuilder(section, chunks, start_ordinal)
    for block in doc.blocks:
        if block.char_start < section["char_start"] or block.char_start >= section["char_end"]:
            continue
        if block.kind == "heading":
            continue  # the section's own heading line: carried as section_path/heading
        if block.kind in _ATOMIC_BLOCK_KINDS:
            builder.flush_prose(doc)
            builder.emit_block(block, doc)
            continue
        para = doc.normalized_text[block.char_start : block.char_end]
        chunk_type = _atomic_chunk_type(para, format_profile)
        if chunk_type is not None:
            builder.flush_prose(doc)
            before = len(chunks)
            builder.emit_atomic_clause(
                para, block.char_start, block.char_end, doc, chunk_type=chunk_type
            )
            for c in chunks[before:]:
                c.meta.update(_block_provenance([block]))
            continue
        builder.prose_buffer.append((para, block.char_start, block.char_end))
        builder.prose_blocks.append(block)
    builder.flush_prose(doc)
    return builder.ordinal


def chunk_document(doc: ParsedDocument, *, format_profile: str = "narrative") -> list[dict]:
    chunks: list[_Chunk] = []
    ordinal = 0
    if doc.blocks is not None:
        for section in doc.sections:
            ordinal = _chunk_section_layout(section, doc, format_profile, chunks, ordinal)
        return [c.to_dict() for c in sorted(chunks, key=lambda c: c.ordinal)]
    sections = doc.sections or [
        {
            "number": None,
            "heading": None,
            "path": None,
            "char_start": 0,
            "char_end": len(doc.normalized_text),
            "page_start": 1,
            "page_end": doc.page_count or 1,
        }
    ]

    for section in sections:
        ordinal = _chunk_section(section, doc, format_profile, chunks, ordinal)

    if format_profile != "narrative":
        chunks.extend(_extract_pdf_figures(doc, chunks))

    return [c.to_dict() for c in sorted(chunks, key=lambda c: c.ordinal)]


def _extract_pdf_figures(doc: ParsedDocument, existing: list[_Chunk]) -> list[_Chunk]:
    """Best-effort figure detection for `.pdf` sources (ARCH §6 rule 3b).
    No-op for non-PDF `ParsedDocument`s (`source_path` unset) or when the
    document has no embedded images."""
    path = doc.source_path
    if not path:
        return []
    from pypdf import PdfReader  # noqa: PLC0415 - heavy/optional dep, only needed for .pdf sources

    reader = PdfReader(path)
    figures: list[_Chunk] = []
    next_ordinal = (max((c.ordinal for c in existing), default=-1)) + 1
    for page_idx, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        caption_match = _FIGURE_CAPTION_RE.search(page_text)
        try:
            images = list(getattr(page, "images", []))
        except Exception:
            # Best-effort per the module's own framing (ARCH §6 rule 3b) --
            # a missing optional image-decode dependency or one malformed
            # embedded image must not abort chunking the rest of the
            # document (DEVIATIONS.md #174, found via a real PDF whose
            # image decoding raised ImportError: pillow is required).
            logger.warning(
                "figure extraction failed on page %d of %s, skipping this page's figures",
                page_idx,
                path,
                exc_info=True,
            )
            continue
        for image in images:
            has_caption = caption_match is not None
            caption = caption_match.group(0) if caption_match else "(figure, no caption detected)"
            image_sha256 = hashlib.sha256(image.data).hexdigest()
            enclosing = next((c for c in existing if c.page_start <= page_idx <= c.page_end), None)
            figures.append(
                _Chunk(
                    ordinal=next_ordinal,
                    section_path=enclosing.section_path if enclosing else None,
                    section_number=enclosing.section_number if enclosing else None,
                    heading=enclosing.heading if enclosing else None,
                    page_start=page_idx,
                    page_end=page_idx,
                    char_start=0,
                    char_end=0,
                    chunk_type="figure",
                    text=caption,
                    figure_ref={"page": page_idx, "bbox": None, "image_sha256": image_sha256},
                    parent_ordinal=enclosing.ordinal if enclosing else None,
                    meta={"has_embedded_text": has_caption},
                )
            )
            next_ordinal += 1
    return figures
