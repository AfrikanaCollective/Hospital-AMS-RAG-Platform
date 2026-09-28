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

**Mid-sentence guard** (the NICE NG195 split-heading defect): an unnumbered
heading that ends without terminal punctuation and is followed by a text
element starting in lower case is really the start of a sentence, so it is
turned back into text and joined to that element.
"""

from __future__ import annotations

import re
import statistics

from app.ingestion.layout.model import (
    KIND_HEADING,
    KIND_LIST_ITEM,
    KIND_TEXT,
    Element,
    LayoutDocument,
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
    body_size = _body_font_size(doc)
    ordered = [el for page in doc.pages for el in page.elements]
    current_numbered_depth = 0

    for i, el in enumerate(ordered):
        if el.kind != KIND_HEADING:
            continue
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
