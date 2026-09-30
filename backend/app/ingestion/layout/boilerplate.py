"""Page header/footer removal, two layers (ARCH-044; LAYOUT-INGESTION-PROPOSAL.md
§5.3).

1. Drop elements the layout model labels `page_header` / `page_footer`.
2. Fallback for text it misses: an element in the top/bottom margin zone is
   dropped if its digit-normalised text repeats on enough pages, is a bare
   page-number pattern, or matches an operator-supplied manifest pattern.

The repeat rule needs at least `min_pages` pages: on a 2-page excerpt a 50%
ratio would drop any margin text that appears once. Short excerpts rely on
the layout labels, the page-number pattern and the manifest patterns.

Guards: headings, tables and pictures are never dropped, and neither is any
text carrying a dose-like quantity. Every drop is reported so an admin can
audit what was removed (`parse_report.dropped_boilerplate`).

**Running headers typed as headings (DEVIATIONS.md #236).** The MoH
guideline prints "EMPIRIC ANTIBIOTIC USE" at the top of every page of that
part, and Docling typed most of them as headings, so the heading guard kept
them and they became parent sections ("NEONATAL SEPSIS … › EMPIRIC
ANTIBIOTIC USE › Early onset sepsis"). A heading loses the guard only when
the same text sits in the margin zone at the same vertical position
(± `RUNNING_HEADER_POSITION_PT`) on at least `min_pages` pages. That is a
running header by definition, and a real section heading doesn't recur at
one fixed spot. This test uses a page count, not the repeat ratio, so a short
excerpt with a few such pages is covered. Dose text is still never dropped.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from app.ingestion.layout.model import (
    KIND_CAPTION,
    KIND_HEADING,
    KIND_LIST_ITEM,
    KIND_PAGE_FOOTER,
    KIND_PAGE_HEADER,
    KIND_PICTURE,
    KIND_TABLE,
    KIND_TEXT,
    Element,
    LayoutDocument,
    LayoutPage,
)

_PAGE_NUMBER_RE = re.compile(r"^(page\s+)?#+(\s+of\s+#+)?$")
_DOSE_RE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:mg|g|kg|ml|mcg|µg|iu|i\.u|units?|%|mmol)\b", re.IGNORECASE
)
_NEVER_DROP_KINDS = frozenset({KIND_HEADING, KIND_TABLE, KIND_PICTURE})


def normalize(text: str) -> str:
    """Lower-case, digits -> `#`, whitespace collapsed ("Page 3 of 40" ==
    "Page 4 of 40")."""
    return re.sub(r"\s+", " ", re.sub(r"\d", "#", text.lower())).strip()


@dataclass
class BoilerplateReport:
    dropped_count: int = 0
    distinct: list[str] = field(default_factory=list)

    def add(self, text: str) -> None:
        self.dropped_count += 1
        n = normalize(text)
        if n not in self.distinct:
            self.distinct.append(n)


def _in_margin(el: Element, page: LayoutPage, zone: float) -> bool:
    return el.bbox.bottom <= page.height * zone or el.bbox.top >= page.height * (1 - zone)


def _protected(el: Element) -> bool:
    return el.kind in _NEVER_DROP_KINDS or bool(_DOSE_RE.search(el.text))


RUNNING_HEADER_POSITION_PT = 2.0


def _running_header_headings(doc: LayoutDocument, zone: float, min_pages: int) -> set[int]:
    """ids of heading elements that are running headers: the same
    normalised text in the margin zone, at one vertical position, on at least
    `min_pages` pages."""
    by_text: dict[str, list[tuple[int, Element]]] = {}
    for page in doc.pages:
        for el in page.elements:
            if el.kind == KIND_HEADING and el.text and _in_margin(el, page, zone):
                by_text.setdefault(normalize(el.text), []).append((page.page_no, el))
    ids: set[int] = set()
    for hits in by_text.values():
        tops = sorted(el.bbox.top for _, el in hits)
        anchor = tops[len(tops) // 2]  # the median position
        aligned = [
            (n, el) for n, el in hits if abs(el.bbox.top - anchor) <= RUNNING_HEADER_POSITION_PT
        ]
        if len({n for n, _ in aligned}) >= min_pages:
            ids.update(id(el) for _, el in aligned if not _DOSE_RE.search(el.text))
    return ids


def remove_boilerplate(
    doc: LayoutDocument,
    *,
    margin_zone: float,
    repeat_ratio: float,
    min_pages: int,
    manifest_patterns: list[str] | None = None,
) -> BoilerplateReport:
    """Remove boilerplate elements in place and return what was removed."""
    report = BoilerplateReport()
    patterns = [re.compile(p, re.IGNORECASE) for p in (manifest_patterns or [])]
    n_pages = len(doc.pages)

    repeated: set[str] = set()
    if n_pages >= min_pages:
        pages_seen: Counter[str] = Counter()
        for page in doc.pages:
            seen_here = {
                normalize(el.text)
                for el in page.elements
                if el.text and _in_margin(el, page, margin_zone)
            }
            pages_seen.update(seen_here)
        repeated = {t for t, c in pages_seen.items() if c / n_pages >= repeat_ratio}

    running = _running_header_headings(doc, margin_zone, min_pages)
    for page in doc.pages:
        kept: list[Element] = []
        for el in page.elements:
            drop = False
            if el.kind in (KIND_PAGE_HEADER, KIND_PAGE_FOOTER) or id(el) in running:
                drop = True
            elif el.text and not _protected(el) and _in_margin(el, page, margin_zone):
                norm = normalize(el.text)
                drop = (
                    norm in repeated
                    or bool(_PAGE_NUMBER_RE.match(norm))
                    or any(p.search(el.text) for p in patterns)
                )
            if drop:
                if el.text:
                    report.add(el.text)
                continue
            kept.append(el)
        page.elements = kept
    return report


_MID_SENTENCE_START = ")]},;:.–-"
_FRAGMENT_KINDS = frozenset({KIND_TEXT, KIND_LIST_ITEM, KIND_CAPTION})


def drop_leading_excerpt_fragment(doc: LayoutDocument) -> list[str]:
    """For a page-range excerpt only (the caller checks `source_pages`): drop
    the text that opens the first page before its first heading when it
    starts mid-sentence, i.e. the tail of a paragraph from a page the excerpt
    doesn't contain (WHO SBI p.17: "fives) in resource-limited settings
    (26). …", continued from p.16; DEVIATIONS.md #235). It has no section
    and can't be read in context.

    Conservative: nothing is dropped unless the first element starts with a
    lowercase letter or closing punctuation, only text/list/caption elements
    before the first heading are considered, and nothing is dropped if a
    table, figure or flowchart comes first, the page has no heading, or the
    fragment contains a dose.
    Returns the dropped texts for the parse report."""
    if not doc.pages:
        return []
    page = doc.pages[0]
    lead: list[Element] = []
    for el in page.elements:
        if el.kind == KIND_HEADING:
            break
        lead.append(el)
    else:
        return []  # no heading on the page: never drop a whole page
    if not lead or any(el.kind not in _FRAGMENT_KINDS for el in lead):
        return []
    if any(_DOSE_RE.search(el.text) for el in lead):
        return []  # never drop dose text, even a fragment (same rule as boilerplate)
    first = lead[0].text.lstrip()
    if not first or not (first[0].islower() or first[0] in _MID_SENTENCE_START):
        return []
    page.elements = page.elements[len(lead) :]
    return [el.text for el in lead]
