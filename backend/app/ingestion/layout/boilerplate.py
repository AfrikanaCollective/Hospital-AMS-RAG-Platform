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
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from app.ingestion.layout.model import (
    KIND_HEADING,
    KIND_PAGE_FOOTER,
    KIND_PAGE_HEADER,
    KIND_PICTURE,
    KIND_TABLE,
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

    for page in doc.pages:
        kept: list[Element] = []
        for el in page.elements:
            drop = False
            if el.kind in (KIND_PAGE_HEADER, KIND_PAGE_FOOTER):
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
