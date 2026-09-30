"""Heading-level fusion and the mid-sentence guard (ARCH-044;
LAYOUT-INGESTION-PROPOSAL.md §5.4).

Docling labels section headers but its levels are usually flat, so the level
is fused from, in order:

1. **Numbering depth**: "1.10 Assessing …" -> 2 (same rule as the `pypdf`
   path's `_HEADING_LINE_RE`).
2. **Structure**: an unnumbered heading nests under the most recent
   numbered heading (depth + 1). Before any numbered heading it is level 1.
3. **Font**: an unnumbered "heading" set smaller than body text and not bold
   is demoted to body text; the layout model over-labels emphasised lines.

**Font-size tiers override 1-2 when the document has them (DEVIATIONS.md
#230).** Rules 1-2 inverted NICE NG195's hierarchy: its unnumbered 25.5 pt
chapters ("Risk factors and clinical indicators: …") got "numbered depth +
1" and so ranked *below* the 21 pt "1.10 …" sections they contain, and its
16.5 pt research recommendations "1 …" to "15 …" read as depth-1 headings,
so "Rationale and impact", "Context" etc. nested under "15 Long-term
outcomes of bacterial meningitis". When the headings come in at least two
sizes that each recur, a heading's level is its size rank (largest = 1).
Numbering and structure remain the fallback for documents whose headings
share one size, and for a heading with no font size (OCR).

**Typographic tables (DEVIATIONS.md #234)**, e.g. WHO SBI Tables 1.1/3.1,
which are one-column tables of shaded band rows, each a sub-topic title
followed by its bullets:

- a text or list item that sits *alone* in a filled, near-full-width, short
  rectangle (a band row) is a heading; Docling typed some band rows as list
  items, so their content merged into the previous section (B.4 meningitis
  dosing inside B.3);
- a standalone "Table N.N …" caption directly followed by a heading or band
  is the table's title, so it becomes a heading;
- "Table N.N continued" lines are dropped (they would open empty sections);
- a single-letter part heading ("A. Non-hospital settings") parents the
  headings that follow it at its own level ("1. …", "A.2 …"), until a
  higher-level heading or the next part.

**Mid-sentence guard** (the NICE NG195 split-heading defect): an unnumbered
heading that ends without terminal punctuation and is followed by a text
element starting in lower case is really the start of a sentence, so it is
turned back into text and joined to that element.
"""

from __future__ import annotations

import re
import statistics

from app.ingestion.layout.model import (
    KIND_CAPTION,
    KIND_HEADING,
    KIND_LIST_ITEM,
    KIND_PAGE_FOOTER,
    KIND_PAGE_HEADER,
    KIND_TEXT,
    BBox,
    Element,
    LayoutDocument,
    LayoutPage,
)

_NUMBERED_RE = re.compile(r"^(?P<num>\d+(?:\.\d+){0,4})\.?\s+(?P<title>\S.*)$", re.DOTALL)
_TERMINAL_PUNCT = (".", ":", "?", "!", ")")


def split_number(text: str) -> tuple[str | None, str]:
    """`"1.10 Assessing …"` -> `("1.10", "Assessing …")`; unnumbered -> `(None, text)`."""
    m = _NUMBERED_RE.match(text.strip())
    if not m:
        return None, text.strip()
    return m.group("num"), m.group("title").strip()


def _body_font_size(doc: LayoutDocument) -> float | None:
    sizes = [
        el.font_size
        for page in doc.pages
        for el in page.elements
        if el.kind in (KIND_TEXT, KIND_LIST_ITEM) and el.font_size
    ]
    return statistics.median(sizes) if sizes else None


def _is_sentence_start(heading: Element, nxt: Element | None) -> bool:
    if nxt is None or nxt.kind not in (KIND_TEXT, KIND_LIST_ITEM):
        return False
    text = heading.text.rstrip()
    return bool(text) and not text.endswith(_TERMINAL_PUNCT) and nxt.text[:1].islower()


def fuse_heading_levels(doc: LayoutDocument) -> None:
    """Set `Element.level` on every heading, demoting or merging false ones.
    Mutates `doc` in place; reading order is preserved."""
    _typographic_table_headings(doc)
    body_size = _body_font_size(doc)
    ordered = [el for page in doc.pages for el in page.elements]
    current_numbered_depth = 0

    for i, el in enumerate(ordered):
        if el.kind != KIND_HEADING or el.box_depth is not None:
            continue  # row-labelled box headings are levelled by label_boxes (#238)
        number, _title = split_number(el.text)
        if number is not None:
            el.level = number.count(".") + 1
            current_numbered_depth = el.level
            continue
        nxt = ordered[i + 1] if i + 1 < len(ordered) else None
        if _is_sentence_start(el, nxt):
            assert nxt is not None
            nxt.text = f"{el.text.rstrip()} {nxt.text}"
            nxt.bbox = type(nxt.bbox)(
                min(el.bbox.x0, nxt.bbox.x0),
                min(el.bbox.top, nxt.bbox.top),
                max(el.bbox.x1, nxt.bbox.x1),
                max(el.bbox.bottom, nxt.bbox.bottom),
            )
            el.kind = "_merged"
            continue
        if (
            body_size is not None
            and el.font_size is not None
            and el.font_size < body_size - 0.25
            and not el.is_bold
        ):
            el.kind = KIND_TEXT
            el.level = None
            continue
        el.level = current_numbered_depth + 1

    for page in doc.pages:
        page.elements = [el for el in page.elements if el.kind != "_merged"]

    headings = [el for el in ordered if el.kind == KIND_HEADING]
    _levels_from_font_tiers(headings)
    _nest_under_letter_parts(headings)


# Sizes closer than this are one tier (rounding in the PDF's font matrix).
_TIER_TOLERANCE_PT = 0.25
_MIN_HEADINGS_PER_TIER = 2  # a size used once (a cover title) isn't a tier
_MIN_TIERS = 2  # with one tier, font can't order the headings


def _font_tiers(headings: list[Element]) -> list[float]:
    """Heading font sizes that recur (≥ 2 headings), largest first. A size
    used once (a cover title) isn't a tier of the hierarchy."""
    counts: dict[float, int] = {}
    for el in headings:
        if el.font_size is None:
            continue
        size = next(
            (t for t in counts if abs(t - el.font_size) <= _TIER_TOLERANCE_PT), el.font_size
        )
        counts[size] = counts.get(size, 0) + 1
    return sorted((t for t, n in counts.items() if n >= _MIN_HEADINGS_PER_TIER), reverse=True)


def _levels_from_font_tiers(headings: list[Element]) -> None:
    tiers = _font_tiers(headings)
    if len(tiers) < _MIN_TIERS:
        return  # keep numbering/structure
    for el in headings:
        if el.font_size is None:
            continue
        el.level = 1 + sum(1 for t in tiers if t > el.font_size + _TIER_TOLERANCE_PT)


# ── typographic tables (DEVIATIONS.md #234) ──

_TABLE_TITLE_RE = re.compile(r"^Table\s+\d+(?:\.\d+)*\b")
_TABLE_CONTINUED_RE = re.compile(r"^Table\s+\d+(?:\.\d+)*\s*\(?continued\)?\.?$", re.IGNORECASE)
_LETTER_PART_RE = re.compile(r"^[A-Z]\.\s+\S")
_BAND_MIN_WIDTH_RATIO = 0.6  # of the page width
_BAND_MAX_HEIGHT_PT = 40.0  # a band row holds a one- or two-line title
_BAND_TOLERANCE_PT = 1.0
_SKIP = (KIND_PAGE_HEADER, KIND_PAGE_FOOTER)


def _inside(inner: Element, outer: BBox) -> bool:
    t = _BAND_TOLERANCE_PT
    return (
        inner.bbox.x0 >= outer.x0 - t
        and inner.bbox.x1 <= outer.x1 + t
        and inner.bbox.top >= outer.top - t
        and inner.bbox.bottom <= outer.bottom + t
    )


def _band_rows(page: LayoutPage) -> list[Element]:
    """Text/list elements that sit alone in a shaded band row."""
    bands = [
        d.bbox
        for d in page.drawings
        if d.kind == "rect"
        and d.filled
        and (d.bbox.x1 - d.bbox.x0) >= _BAND_MIN_WIDTH_RATIO * page.width
        and (d.bbox.bottom - d.bbox.top) <= _BAND_MAX_HEIGHT_PT
    ]
    found: list[Element] = []
    for band in bands:
        inside = [el for el in page.elements if el.kind not in _SKIP and _inside(el, band)]
        if len(inside) != 1:
            continue  # a shaded box with several items is a box, not a band row
        el = inside[0]
        text = el.text.strip()
        if el.kind in (KIND_TEXT, KIND_LIST_ITEM) and text and not text.endswith("."):
            found.append(el)
    return found


def _typographic_table_headings(doc: LayoutDocument) -> None:
    for page in doc.pages:
        for el in _band_rows(page):
            el.kind = KIND_HEADING
        page.elements = [
            el
            for el in page.elements
            if not (
                el.kind in (KIND_HEADING, KIND_TEXT, KIND_CAPTION)
                and _TABLE_CONTINUED_RE.match(el.text.strip())
            )
        ]
    ordered = [el for page in doc.pages for el in page.elements if el.kind not in _SKIP]
    for i, el in enumerate(ordered):
        nxt = ordered[i + 1] if i + 1 < len(ordered) else None
        if (
            el.kind == KIND_CAPTION
            and _TABLE_TITLE_RE.match(el.text.strip())
            and nxt is not None
            and nxt.kind == KIND_HEADING
        ):
            el.kind = KIND_HEADING


def _nest_under_letter_parts(headings: list[Element]) -> None:
    """Letter parts ("A. …") parent the headings that follow them, and a table
    title ("Table 3.1 …") parents its parts.

    A part's span runs until a heading ranked above the part, the next part,
    or a table title ranked at or above the part. (A table title ranked below
    it, MoH's "Table 5: …" inside part "B.", is a member.) If any member
    shares the part's level, the whole span moves one level down; members
    already set smaller than the part are left where the font put them.

    A part level with or above the table title it follows is pushed under the
    title first; a single-table document has no separate title size."""
    live = [el for el in headings if el.level is not None]
    orig = {id(el): el.level for el in live}
    spans: list[tuple[Element, list[Element], int]] = []  # part, members, table push
    table_level: int | None = None
    current: tuple[Element, list[Element], int] | None = None
    for el in live:
        level = orig[id(el)]
        assert level is not None
        text = el.text.strip()
        part_level = orig[id(current[0])] if current else None
        if _TABLE_TITLE_RE.match(text) and (part_level is None or level <= part_level):
            table_level, current = level, None
            continue
        if _LETTER_PART_RE.match(text):
            push = (
                table_level + 1 - level if table_level is not None and level <= table_level else 0
            )
            current = (el, [], push)
            spans.append(current)
            continue
        if current is not None and part_level is not None and level < part_level:
            current = None
        if current is not None:
            current[1].append(el)
        elif table_level is not None and level <= table_level:
            table_level = None  # a heading at the title's rank ends the table
    for part, members, push in spans:
        part_level = orig[id(part)]
        assert part_level is not None
        extra = 1 if any(orig[id(m)] == part_level for m in members) else 0
        part.level = part_level + push
        for m in members:
            m.level = orig[id(m)] + push + extra  # type: ignore[operator]
