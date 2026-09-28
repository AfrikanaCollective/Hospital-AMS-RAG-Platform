"""Structure-aware document parsing (ARCH §5.1, ARCH-013).

Produces: one normalized document text with a stable character index, a section
tree (heading detection from numbering / regex on `^\\d+(\\.\\d+)*\\s`, or
markdown `#` headings), and a per-document parse-quality score. Low-quality
parses get a visible badge and admin review (self-critique §21a mitigation).

Two source formats are supported (ARCH-038 / `scripts/prepare_sample_guidelines.py`
accepts `.pdf` and `.md` in the real corpus dir, and the CI-only synthetic
fixtures under `tests/fixtures/guidelines/` are `.md` so no binary PDF fixtures
need to be committed):

- `.pdf`: primary path — `pypdf` per-page text extraction. **Limitation,
  flagged rather than worked around silently (CLAUDE.md §2):** `pypdf` does not
  expose font size/weight, so heading detection cannot use the font-size signal
  ARCH §5.1 step 2 names alongside numbering/regex — only the numbering/regex
  signal is implemented. A heading is recognised only when it is *alone on its
  line* (`^\\d+(\\.\\d+){0,4}\\.?\\s+<short title>$`); a numbered clause that
  starts a wrapped prose paragraph (e.g. a GRADE recommendation statement) is
  deliberately NOT treated as a heading here — that is chunking's job (§6 rule
  1), not parsing's. OCR is out of scope for MVP (ARCH §5.1 step 2): image-only
  regions contribute no text, which lowers `parse_quality`.
- `.md` / `.markdown` / `.txt`: read directly (already fully extractable text,
  so `parse_quality = 1.0`); `#`/`##`/... lines are headings, with a leading
  `\\d+(\\.\\d+)*` in the heading text captured as `section_number` when present.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# A line, on its own, that looks like a heading: optional numbering, a short
# title, nothing else on the line. Deliberately conservative (whole-line match)
# so wrapped prose paragraphs (recommendation statements, protocol steps) are
# never misdetected as headings.
_HEADING_LINE_RE = re.compile(r"^(?P<num>\d+(?:\.\d+){0,4})\.?\s+(?P<title>[A-Z][^\n]{0,90})$")
_MD_HEADING_RE = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<rest>.+?)\s*$")
_LEADING_NUM_RE = re.compile(r"^(?P<num>\d+(?:\.\d+){0,4})\.?\s+(?P<title>.*)$")

_CHARS_PER_SYNTHETIC_PAGE = 3000  # pagination budget for non-paginated text sources


@dataclass
class Block:
    """One layout element placed in the normalized text (layout parser only,
    ARCH-044). `kind`: heading | text | list_item | caption | footnote |
    table | flowchart | figure. Offsets are absolute in `normalized_text`.
    `meta` carries what chunking needs to emit an atomic chunk: table parts
    and row texts, the flowchart graph, figure crop, origin/OCR statistics,
    applied corrections."""

    kind: str
    char_start: int
    char_end: int
    page_no: int  # physical page in the ingested file
    origin: str = "text_layer"
    meta: dict = field(default_factory=dict)


@dataclass
class ParsedDocument:
    normalized_text: str
    sections: list[dict]  # [{number, heading, path, char_start, char_end, page_start, page_end}]
    page_count: int
    parse_quality: float  # 0..1
    # per-page (page_num, char_start_in_normalized_text), 1-indexed page_num
    page_starts: list[tuple[int, int]]
    source_path: str | None = None  # set for .pdf sources; used for figure extraction (§6 rule 3b)
    # ── layout parser only (ARCH-044); `None`/empty on the pypdf/markdown path ──
    blocks: list[Block] | None = None
    parser_version: str | None = None
    parse_report: dict = field(default_factory=dict)

    def page_for_offset(self, char_offset: int) -> int:
        page = self.page_starts[0][0] if self.page_starts else 1
        for page_num, start in self.page_starts:
            if start <= char_offset:
                page = page_num
            else:
                break
        return page


def _extract_pdf_pages(path: str) -> list[str]:
    # noqa justification: pypdf is cheap, but keeping the import lazy here
    # mirrors the same pattern used for the genuinely heavy optional deps
    # (sentence-transformers/torch) elsewhere in ingestion/retrieval.
    from pypdf import PdfReader  # noqa: PLC0415

    reader = PdfReader(path)
    return [(page.extract_text() or "") for page in reader.pages]


def _paginate_text(text: str, *, chars_per_page: int = _CHARS_PER_SYNTHETIC_PAGE) -> list[str]:
    if not text:
        return [""]
    return [text[i : i + chars_per_page] for i in range(0, len(text), chars_per_page)] or [""]


def _match_markdown_heading(stripped: str) -> tuple[str | None, str, int] | None:
    m = _MD_HEADING_RE.match(stripped)
    if not m:
        return None
    depth = len(m.group("hashes"))
    rest = m.group("rest")
    num_m = _LEADING_NUM_RE.match(rest)
    if num_m:
        return num_m.group("num"), num_m.group("title").strip(), depth
    return None, rest.strip(), depth


def _match_regex_heading(stripped: str) -> tuple[str | None, str, int] | None:
    m = _HEADING_LINE_RE.match(stripped.strip())
    if not m:
        return None
    number, title = m.group("num"), m.group("title").strip()
    return number, title, number.count(".") + 1


def _close_open_sections(sections: list[dict], offset: int) -> None:
    for prev in reversed(sections):
        if prev["char_end"] is None:
            prev["char_end"] = offset


def _build_section_tree(normalized_text: str, *, markdown: bool) -> list[dict]:
    """Walk lines, tracking char offsets, and collect heading lines into a
    breadcrumb-annotated flat list (each entry's `path` includes ancestors)."""
    sections: list[dict] = []
    stack: list[dict] = []  # open headings by depth
    offset = 0
    match_heading = _match_markdown_heading if markdown else _match_regex_heading
    for line in normalized_text.splitlines(keepends=True):
        stripped = line.rstrip("\n")
        match = match_heading(stripped)

        if match is not None:
            number, title, depth = match
            while stack and stack[-1]["_depth"] >= depth:
                stack.pop()
            path_parts = [s["heading"] for s in stack] + [title]
            entry = {
                "number": number,
                "heading": title,
                "path": " › ".join(path_parts),
                "char_start": offset,
                "char_end": None,  # closed when the next section starts, or at EOF
                "_depth": depth,
            }
            _close_open_sections(sections, offset)
            sections.append(entry)
            stack.append(entry)
        offset += len(line)

    for s in sections:
        if s["char_end"] is None:
            s["char_end"] = len(normalized_text)
        del s["_depth"]
    return sections


def _parse_quality_for_pdf(page_texts: list[str], sections: list[dict]) -> float:
    if not page_texts:
        return 0.0
    non_ws_chars = sum(len(re.sub(r"\s+", "", t)) for t in page_texts)
    # ~500 non-whitespace chars/page is a light, real academic-PDF page of body
    # text; below that a page is likely image-only / OCR-needed (out of scope).
    expected = max(len(page_texts) * 500, 1)
    extractable_ratio = min(1.0, non_ws_chars / expected)
    heading_bonus = 0.1 if sections else 0.0
    return round(min(1.0, extractable_ratio + heading_bonus), 3)


def _assemble(page_texts: list[str], *, markdown: bool) -> ParsedDocument:
    page_starts: list[tuple[int, int]] = []
    offset = 0
    parts: list[str] = []
    for i, page_text in enumerate(page_texts, start=1):
        page_starts.append((i, offset))
        parts.append(page_text)
        offset += len(page_text)
        if i < len(page_texts):
            parts.append("\n")
            offset += 1
    normalized_text = "".join(parts)
    sections = _build_section_tree(normalized_text, markdown=markdown)

    doc = ParsedDocument(
        normalized_text=normalized_text,
        sections=[],
        page_count=len(page_texts),
        parse_quality=1.0 if markdown else _parse_quality_for_pdf(page_texts, sections),
        page_starts=page_starts,
    )
    for s in sections:
        s["page_start"] = doc.page_for_offset(s["char_start"])
        s["page_end"] = doc.page_for_offset(max(s["char_end"] - 1, s["char_start"]))
    doc.sections = sections
    return doc


def parse_pdf(path: str) -> ParsedDocument:
    doc = _assemble(_extract_pdf_pages(path), markdown=False)
    doc.source_path = path
    return doc


def parse_markdown(path: str) -> ParsedDocument:
    raw = Path(path).read_text(encoding="utf-8")
    return _assemble(_paginate_text(raw), markdown=True)


def parse_document(path: str) -> ParsedDocument:
    """Dispatch on file extension. `.pdf` -> `parse_pdf`; `.md`/`.markdown`/`.txt`
    -> `parse_markdown`."""
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(path)
    if suffix in (".md", ".markdown", ".txt"):
        return parse_markdown(path)
    raise ValueError(f"unsupported guideline file type: {suffix!r} ({path})")
