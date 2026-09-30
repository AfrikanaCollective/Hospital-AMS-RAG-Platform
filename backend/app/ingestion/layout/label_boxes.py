"""Row-labelled boxes (ARCH-044; DEVIATIONS.md #238).

The Kenya MoH National Antibiotic Use Guidelines set each syndrome's
pathogens, treatment and comments in a ruled box whose narrow, shaded left
column holds the row labels ("Common Pathogens", "Empiric Therapy",
"Comments", "Cellulitis …"). Docling either misses the box (pp. 34–35:
labels read as headings and ordered ahead of the page's other text, the
First- and Second-line columns run together across the page break) or
detects a grid whose cells it re-splits (p. 37: seven columns for three).

Operator exemplar (2026-09-30): the narrow left column is the sub-topic
heading and the rest of the row is its content, so that "First line" and
"Second line" become labelled sub-topics under "Empiric Therapy".

Rebuilt from the drawing, not from Docling's reading:

1. **Box:** vertical rules give a narrow label column (≤ `MAX_LABEL_WIDTH_RATIO`
   of the page width) with the box's right edge further out, and horizontal
   rules crossing the label column give the rows. At least half the rows
   must have a *shaded* label cell, which keeps plain lookup grids (CRP
   levels, LRINEC points) out.
2. **Columns** of a row: the vertical rules present in that row.
3. **Text** of each cell: the page's text lines whose centre falls in it
   (word-completed, text layer), joined into paragraphs. A short bold line
   ending ":" (or followed by non-bold text) is a sub-label ("First line:").
4. **Rows:**
   - a *header row* (every cell one short bold line): its cells label the
     columns below it, on later pages too, while the box's geometry holds;
     with a single cell and no label it is the box *title*, dropped when
     the enclosing section heading already says it;
   - a *labelled row*: the label becomes a heading, then each cell under its
     column header or its own sub-label;
   - a *continuation row* (empty label, first on the page, same geometry as
     the box the previous page ended in): its content continues the previous
     row. The previous row label and the column's last sub-label are
     repeated as headings marked `box_continued`; the text stays on its own
     page, so citations keep the right page.

The page's elements inside the box (and any Docling table covering it) are
replaced by the rebuilt sequence, placed where the box sits on the page.
Levels are set after `fuse_heading_levels` by `apply_box_levels`: depth
below the nearest preceding section heading set larger than body text.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field

from app.ingestion.layout.model import (
    KIND_HEADING,
    KIND_PAGE_FOOTER,
    KIND_PAGE_HEADER,
    KIND_TABLE,
    KIND_TEXT,
    ORIGIN_OCR,
    ORIGIN_TEXT_LAYER,
    BBox,
    Element,
    LayoutDocument,
    LayoutPage,
    TextLine,
)

MIN_LABEL_WIDTH_RATIO = 0.06
MAX_LABEL_WIDTH_RATIO = 0.18
MIN_SHADED_ROW_SHARE = 0.5
RULE_THICKNESS_PT = 2.0
SNAP_PT = 2.5  # rules and edges closer than this are the same line
MIN_ROW_HEIGHT_PT = 8.0
SUB_LABEL_MAX_WORDS = 6
SUB_LABEL_MIN_WORDS = 2  # a lone bold word is a drug or connector ("PLUS")
HEADER_CELL_MAX_WORDS = 5
BADGE_MAX_FONT_PT = 5.0  # AWaRe badge captions ("ACCESS", "WATCH") are ~3.9 pt
LOOKUP_MAX_WORDS = 3  # a grid whose every cell is this short is a lookup table
_BULLET_RE = re.compile(r"^[•▪◦●○■□‣⁃∙·\-–*]\s*|^\d+[.)]\s")


@dataclass
class _Rule:
    pos: float  # x of a vertical rule, y of a horizontal one
    lo: float  # extent along the other axis
    hi: float


@dataclass
class _Row:
    top: float
    bottom: float
    label: str
    label_lines: list[TextLine]
    cells: list[tuple[float, float, list[list[TextLine]]]]  # (x0, x1, paragraphs), all sub-rows
    shaded: bool
    # a row split by rules across its content columns only (skin "Traumatic
    # wounds": without infection / with systemic features) has several
    subrows: list[list[tuple[float, float, list[list[TextLine]]]]] = field(default_factory=list)


@dataclass
class _Box:
    page_no: int
    left: float
    mid: float
    right: float
    rows: list[_Row] = field(default_factory=list)

    @property
    def top(self) -> float:
        return self.rows[0].top

    @property
    def bottom(self) -> float:
        return self.rows[-1].bottom

    def same_geometry(self, other: _Box) -> bool:
        return (
            abs(self.left - other.left) <= SNAP_PT
            and abs(self.mid - other.mid) <= SNAP_PT
            and abs(self.right - other.right) <= SNAP_PT
        )


def _rules(page: LayoutPage) -> tuple[list[_Rule], list[_Rule]]:
    vertical: list[_Rule] = []
    horizontal: list[_Rule] = []
    for d in page.drawings:
        b = d.bbox
        if d.kind not in ("rect", "line"):
            continue
        if b.width <= RULE_THICKNESS_PT and b.height > MIN_ROW_HEIGHT_PT:
            vertical.append(_Rule((b.x0 + b.x1) / 2, b.top, b.bottom))
        elif b.height <= RULE_THICKNESS_PT and b.width > MIN_ROW_HEIGHT_PT:
            horizontal.append(_Rule((b.top + b.bottom) / 2, b.x0, b.x1))
    return vertical, horizontal


def _cluster(values: list[float]) -> list[float]:
    out: list[float] = []
    for v in sorted(values):
        if out and v - out[-1] <= SNAP_PT:
            continue
        out.append(v)
    return out


def _v_at(vertical: list[_Rule], x: float, y: float) -> bool:
    return any(
        abs(r.pos - x) <= SNAP_PT and r.lo - SNAP_PT <= y <= r.hi + SNAP_PT for r in vertical
    )


def _find_frames(page: LayoutPage, vertical: list[_Rule]) -> list[tuple[float, float, float]]:
    """(left, mid, right) x-triples: a narrow label column and the box edge."""
    xs = _cluster([r.pos for r in vertical])
    frames: list[tuple[float, float, float]] = []
    for i, left in enumerate(xs):
        for mid in xs[i + 1 :]:
            width = mid - left
            if width < MIN_LABEL_WIDTH_RATIO * page.width:
                continue
            if width > MAX_LABEL_WIDTH_RATIO * page.width:
                break
            rights = [
                x
                for x in xs
                if x > mid + MIN_LABEL_WIDTH_RATIO * page.width
                and any(
                    _v_at(vertical, left, (r.lo + r.hi) / 2)
                    and _v_at(vertical, x, (r.lo + r.hi) / 2)
                    for r in vertical
                    if abs(r.pos - mid) <= SNAP_PT
                )
            ]
            if rights:
                frames.append((left, mid, max(rights)))
                break
    return frames


def _label_shaded(
    page: LayoutPage,
    left: float,
    mid: float,
    *,
    cells_x: list[tuple[float, float]],
    top: float,
    bottom: float,
) -> bool:
    """The label cell is shaded and the content cells are not: a shaded label
    *column*, not a grid shaded throughout (LRINEC's lookup table)."""
    if not _shaded(page, left, mid, top, bottom):
        return False
    return not all(_shaded(page, x0, x1, top, bottom) for x0, x1 in cells_x)


def _shaded(page: LayoutPage, x0: float, x1: float, top: float, bottom: float) -> bool:
    for d in page.drawings:
        b = d.bbox
        if (
            d.kind == "rect"
            and d.filled
            and b.width > RULE_THICKNESS_PT
            and b.x0 <= x0 + SNAP_PT + 4
            and b.x1 >= x1 - SNAP_PT - 4
            and b.top <= top + SNAP_PT + 4
            and b.bottom >= bottom - SNAP_PT - 4
        ):
            return True
    return False


def _split_at(line: TextLine, x: float) -> list[TextLine]:
    """A line crossing the label column's edge ("Comments Duration of
    therapy:" as one text run) is split at the word boundary nearest the
    edge, estimated from the line's width. Lines without a word boundary are
    kept whole."""
    text = line.text.strip()
    b = line.bbox
    if not (b.x0 < x - SNAP_PT and b.x1 > x + SNAP_PT) or " " not in text:
        return [line]
    target = round((x - b.x0) / b.width * len(text))
    spaces = [i for i, ch in enumerate(text) if ch == " "]
    k = min(spaces, key=lambda i: abs(i - target))
    cut = b.x0 + b.width * k / len(text)
    parts = [(text[:k].strip(), b.x0, cut), (text[k:].strip(), cut, b.x1)]
    return [
        TextLine(
            t,
            BBox(x0, b.top, x1, b.bottom),
            line.from_ocr,
            line.confidence,
            line.font_name,
            line.font_size,
            line.is_bold,
        )
        for t, x0, x1 in parts
        if t
    ]


def _lines_in(
    lines: list[TextLine], x0: float, x1: float, top: float, bottom: float
) -> list[TextLine]:
    out = []
    for ln in lines:
        cx, cy = ln.bbox.center
        if x0 <= cx <= x1 and top <= cy <= bottom and ln.text.strip():
            out.append(ln)
    return sorted(out, key=lambda ln: (round(ln.bbox.top), ln.bbox.x0))


def _bold(line: TextLine) -> bool:
    """Bold from the line's font name as well as its character count: in the
    MoH guideline "IU/kg IV 6 hourly" is set in Cambria-Bold but its
    character-level flag came out False, which split dose lines in two."""
    return line.is_bold or "bold" in (line.font_name or "").lower()


def _is_badge(line: TextLine) -> bool:
    return line.font_size is not None and line.font_size < BADGE_MAX_FONT_PT


def _paragraphs(lines: list[TextLine]) -> list[list[TextLine]]:
    """Lines joined into paragraphs. A new paragraph starts after a vertical
    gap, at a bullet, after a sub-label, or where the weight changes (bold to
    regular or back) on a line starting with a capital ("Early onset sepsis"
    / "Group B …"; "… pneumoniae)" / "Late onset sepsis"). AWaRe badge
    captions are set apart and appended to the paragraph they sit beside, so
    they don't split a dose line mid-way."""
    body = [ln for ln in lines if not _is_badge(ln)]
    badges = [ln for ln in lines if _is_badge(ln)]
    paras: list[list[TextLine]] = []
    for ln in body:
        if paras:
            prev = paras[-1][-1]
            gap = ln.bbox.top - prev.bbox.bottom
            t = ln.text.strip()
            new = (
                gap > 0.6 * max(prev.bbox.height, 1.0)
                or bool(_BULLET_RE.match(t))
                or _is_sub_label_text(prev)
                or (_bold(prev) != _bold(ln) and t[:1].isupper())
            )
            if not new:
                paras[-1].append(ln)
                continue
        paras.append([ln])
    for badge in badges:
        if not paras:
            paras.append([badge])
            continue
        cy = badge.bbox.center[1]
        target = min(
            paras,
            key=lambda p: (
                0.0
                if min(ln.bbox.top for ln in p) - 2 <= cy <= max(ln.bbox.bottom for ln in p) + 2
                else min(abs(cy - ln.bbox.center[1]) for ln in p)
            ),
        )
        target.append(badge)
    return paras


def _text(para: list[TextLine]) -> str:
    return " ".join(" ".join(ln.text.split()) for ln in para).strip()


def _is_sub_label_text(line: TextLine) -> bool:
    t = line.text.strip()
    return _bold(line) and t.endswith(":") and len(t.split()) <= SUB_LABEL_MAX_WORDS


def _is_sub_label(para: list[TextLine], nxt: list[TextLine] | None) -> bool:
    if len(para) != 1 or not _bold(para[0]):
        return False
    t = _text(para)
    words = t.split()
    if t.endswith(":") and len(words) <= SUB_LABEL_MAX_WORDS:
        return True
    return (
        SUB_LABEL_MIN_WORDS <= len(words) <= SUB_LABEL_MAX_WORDS
        and not re.search(r"\d", t)
        and nxt is not None
        and not all(_bold(ln) for ln in nxt)
    )


SUB_ROW_MIN_COVERAGE = 0.9


def _sub_row_rules(
    horizontal: list[_Rule], mid: float, right: float, top: float, bottom: float
) -> list[float]:
    """y of rules that divide a row's content into sub-rows: segments at one
    height inside the row that together cover ≥ `SUB_ROW_MIN_COVERAGE` of the
    content width. Underlines under "First line:" (~40 pt) don't qualify."""
    inside = [
        r for r in horizontal if top + SNAP_PT < r.pos < bottom - SNAP_PT and r.lo >= mid - SNAP_PT
    ]
    out = []
    for y in _cluster([r.pos for r in inside]):
        segs = sorted((r.lo, r.hi) for r in inside if abs(r.pos - y) <= SNAP_PT)
        covered, reach = 0.0, mid
        for seg_lo, hi in segs:
            start = max(seg_lo, reach)
            if hi > start:
                covered += hi - start
                reach = hi
        if covered >= SUB_ROW_MIN_COVERAGE * (right - mid):
            out.append(y)
    return out


def _read_rows(
    page: LayoutPage,
    frame: tuple[float, float, float],
    vertical: list[_Rule],
    horizontal: list[_Rule],
) -> list[_Row]:
    left, mid, right = frame
    ys = _cluster(
        [r.pos for r in horizontal if r.lo <= left + SNAP_PT + 1 and r.hi >= mid - SNAP_PT]
    )
    lines = [part for ln in page.lines for part in _split_at(ln, mid)]
    rows: list[_Row] = []
    for top, bottom in zip(ys, ys[1:], strict=False):
        cy = (top + bottom) / 2
        if bottom - top < MIN_ROW_HEIGHT_PT or not _v_at(vertical, mid, cy):
            continue
        inner = _sub_row_rules(horizontal, mid, right, top, bottom)
        subrows = []
        for st, sb in zip([top, *inner], [*inner, bottom], strict=False):
            scy = (st + sb) / 2
            splits = sorted(
                x
                for x in _cluster([r.pos for r in vertical])
                if mid + SNAP_PT < x < right - SNAP_PT and _v_at(vertical, x, scy)
            )
            edges = [mid, *splits, right]
            subrows.append(
                [
                    (x0, x1, _paragraphs(_lines_in(lines, x0, x1, st, sb)))
                    for x0, x1 in zip(edges, edges[1:], strict=False)
                ]
            )
        cells = [c for sub in subrows for c in sub]
        label_lines = _lines_in(lines, left, mid, top, bottom)
        rows.append(
            _Row(
                top=top,
                bottom=bottom,
                label=_text(label_lines),
                label_lines=label_lines,
                cells=cells,
                shaded=_label_shaded(
                    page, left, mid, cells_x=[(c[0], c[1]) for c in cells], top=top, bottom=bottom
                ),
                subrows=subrows,
            )
        )
    return rows


def _is_lookup_grid(rows: list[_Row]) -> bool:
    """Every content cell a short value ("≤5", "<50%"): a lookup table such
    as LRINEC's risk categories, not a box of labelled sub-topics."""
    body = [
        r
        for r in rows
        if not _is_header_row(r)
        and not all(_bold(ln) for _x0, _x1, paras in r.cells for p in paras for ln in p)
    ]  # an all-bold row is a header even with a long heading cell
    cells = [_text(p) for r in body for _x0, _x1, paras in r.cells for p in paras]
    return bool(cells) and all(len(c.split()) <= LOOKUP_MAX_WORDS for c in cells)


def find_boxes(page: LayoutPage) -> list[_Box]:
    vertical, horizontal = _rules(page)
    boxes: list[_Box] = []
    for frame in _find_frames(page, vertical):
        rows = _read_rows(page, frame, vertical, horizontal)
        if not rows:
            continue
        if sum(r.shaded for r in rows) < MIN_SHADED_ROW_SHARE * len(rows):
            continue
        if _is_lookup_grid(rows):
            continue
        boxes.append(_Box(page.page_no, *frame, rows=rows))
    return boxes


def _is_header_row(row: _Row) -> bool:
    filled = [c for c in row.cells if c[2]]
    if not filled:
        return False
    for _x0, _x1, paras in filled:
        if len(paras) != 1 or not all(_bold(ln) for ln in paras[0]):
            return False
        if len(_text(paras[0]).split()) > HEADER_CELL_MAX_WORDS:
            return False
    return not row.label or all(_bold(ln) for ln in row.label_lines)


def _element(
    kind: str,
    text: str,
    lines: list[TextLine],
    page_no: int,
    *,
    depth: int | None = None,
    continued: bool = False,
) -> Element:
    bbox = BBox(
        min(ln.bbox.x0 for ln in lines),
        min(ln.bbox.top for ln in lines),
        max(ln.bbox.x1 for ln in lines),
        max(ln.bbox.bottom for ln in lines),
    )
    sizes = [ln.font_size for ln in lines if ln.font_size]
    return Element(
        kind=kind,
        page_no=page_no,
        bbox=bbox,
        text=text,
        origin=ORIGIN_OCR if any(ln.from_ocr for ln in lines) else ORIGIN_TEXT_LAYER,
        font_size=statistics.median(sizes) if sizes else None,
        is_bold=all(_bold(ln) for ln in lines),
        box_depth=depth,
        box_continued=continued,
    )


@dataclass
class _Context:
    """What a box that continues onto the next page needs from the last one."""

    box: _Box | None = None
    headers: list[tuple[float, float, str]] = field(default_factory=list)
    label: str | None = None
    label_lines: list[TextLine] = field(default_factory=list)
    sub_labels: dict[int, tuple[str, list[TextLine]]] = field(default_factory=dict)
    split_rows: bool = False  # the last labelled row had sub-rows
    # the last element emitted for each content column of the current row,
    # and for the row as a whole: where a cell continued on the next page is
    # joined back in (DEVIATIONS.md #239)
    col_tail: dict[int, Element] = field(default_factory=dict)
    row_tail: Element | None = None


_Move = tuple[Element, list[Element]]  # insert these right after that element


def _header_for(headers: list[tuple[float, float, str]], x0: float, x1: float) -> str | None:
    """The column header over a cell; none for a cell merged across several
    columns ("Wounds": one statement spanning Description and Empiric
    Therapy)."""
    spanned = [
        text for hx0, hx1, text in headers if min(x1, hx1) - max(x0, hx0) > 0.3 * (hx1 - hx0)
    ]
    return spanned[0] if len(spanned) == 1 else None


def _prefixed(header: str | None, paras: list[list[TextLine]]) -> str:
    body = "\n".join(_text(p) for p in paras)
    return f"{header}: {body}" if header else body


def _tail(ctx: _Context, col: int, el: Element) -> None:
    ctx.col_tail[col] = el
    ctx.row_tail = el


def _emit_subrows(row: _Row, ctx: _Context, n: int) -> list[Element]:
    """A row with several sub-rows: each cell of each sub-row as "<column
    header>: <text>" (the table row format, DEVIATIONS.md #220), sub-row by
    sub-row, so a description stays next to its own treatment."""
    out: list[Element] = []
    for sub in row.subrows:
        for col, (x0, x1, paras) in enumerate(sub):
            if not paras:
                continue
            lines = [ln for p in paras for ln in p]
            el = _element(KIND_TEXT, _prefixed(_header_for(ctx.headers, x0, x1), paras), lines, n)
            out.append(el)
            _tail(ctx, col, el)
    return out


def _cell_elements(
    paras: list[list[TextLine]], n: int, depth: int, *, record_sub: dict | None = None, col: int = 0
) -> list[Element]:
    out: list[Element] = []
    for j, para in enumerate(paras):
        nxt = paras[j + 1] if j + 1 < len(paras) else None
        if _is_sub_label(para, nxt):
            out.append(_element(KIND_HEADING, _text(para), para, n, depth=depth))
            if record_sub is not None:
                record_sub[col] = (_text(para), para)
        else:
            out.append(_element(KIND_TEXT, _text(para), para, n))
    return out


def _emit_cells(row: _Row, ctx: _Context, n: int) -> list[Element]:
    """A row's content cells: each under its column header, or its own
    sub-labels ("First line:")."""
    out: list[Element] = []
    for col, (x0, x1, paras) in enumerate(row.cells):
        if not paras:
            continue
        header = _header_for(ctx.headers, x0, x1)
        depth = 2
        if header:
            first = [ln for p in paras for ln in p][:1]
            out.append(_element(KIND_HEADING, header, first, n, depth=2))
            depth = 3
        els = _cell_elements(
            paras, n, depth, record_sub=None if header else ctx.sub_labels, col=col
        )
        out.extend(els)
        if els:
            _tail(ctx, col, els[-1])
    return out


def _continued_cells(row: _Row, ctx: _Context, n: int) -> list[_Move] | None:
    """A row continued from the previous page (empty label, first on the
    page): each cell's content is joined back after the last part of the same
    cell on the previous page, so e.g. "Empiric Therapy › First line:" is one
    section across pp. 34–35. Text keeps its own page. `None` when there is
    nothing to join to (then the caller repeats the labels instead)."""
    if ctx.row_tail is None:
        return None
    moves: list[_Move] = []
    split = ctx.split_rows
    for sub_i, sub in enumerate(row.subrows or [row.cells]):
        for col, (x0, x1, paras) in enumerate(sub):
            if not paras:
                continue
            header = _header_for(ctx.headers, x0, x1)
            if sub_i == 0:
                anchor = ctx.col_tail.get(col, ctx.row_tail)
                if split:  # same cell of a "header: value" block: no second prefix
                    lines = [ln for p in paras for ln in p]
                    els = [_element(KIND_TEXT, _prefixed(None, paras), lines, n)]
                else:
                    els = _cell_elements(paras, n, 3 if header else 2)
            else:  # a further sub-row of the continued row
                anchor = ctx.row_tail
                lines = [ln for p in paras for ln in p]
                els = [_element(KIND_TEXT, _prefixed(header, paras), lines, n)]
            if not els:
                continue
            moves.append((anchor, els))
            _tail(ctx, col, els[-1])
    return moves


def _carry_over(row: _Row, ctx: _Context, n: int) -> list[_Move]:
    """A labelled row that opens a continuing page may begin with the end of
    the previous row's sentence: the PDF prints diabetic foot's "Surgical
    debridement is an important" on p. 48 and "component in management" at
    the top of the *next* row's cell on p. 49. A cell whose first paragraph
    starts in lower case gives that paragraph back to the same cell of the
    previous row."""
    moves: list[_Move] = []
    for col, (_x0, _x1, paras) in enumerate(row.subrows[0] if row.subrows else row.cells):
        anchor = ctx.col_tail.get(col)
        if anchor is None or not paras or not _text(paras[0])[:1].islower():
            continue
        para = paras.pop(0)
        el = _element(KIND_TEXT, _text(para), para, n)
        moves.append((anchor, [el]))
        ctx.col_tail[col] = el
    return moves


def _repeat_labels(row: _Row, ctx: _Context, n: int) -> list[Element]:
    """Fallback when a continued row has nothing on the previous page to join
    to: repeat the row label (and each column's last sub-label) as headings
    marked `box_continued`."""
    assert ctx.label is not None
    out = [_element(KIND_HEADING, ctx.label, ctx.label_lines, n, depth=1, continued=True)]
    for col, (x0, x1, paras) in enumerate(row.cells):
        if not paras:
            continue
        header = _header_for(ctx.headers, x0, x1)
        depth = 2
        if header:
            out.append(_element(KIND_HEADING, header, paras[0][:1], n, depth=2, continued=True))
            depth = 3
        elif col in ctx.sub_labels and not _is_sub_label(
            paras[0], paras[1] if len(paras) > 1 else None
        ):
            text, lines = ctx.sub_labels[col]
            out.append(_element(KIND_HEADING, text, lines, n, depth=2, continued=True))
        out.extend(_cell_elements(paras, n, depth))
    return out


def _group_rows(rows: list[_Row]) -> list[_Row]:
    """An unlabelled row after a labelled one in the same box is another
    sub-row of it (skin "Surgical site infections": the label cell spans only
    the first of its two sub-rows). The first row on the page is left alone:
    unlabelled there, it continues the previous page."""
    out: list[_Row] = []
    for i, row in enumerate(rows):
        if i > 0 and not row.label and out and out[-1].label and not _is_header_row(row):
            prev = out[-1]
            out[-1] = _Row(
                prev.top,
                row.bottom,
                prev.label,
                prev.label_lines,
                prev.cells + row.cells,
                prev.shaded,
                prev.subrows + row.subrows,
            )
            continue
        out.append(row)
    return out


def _take_header_row(row: _Row, ctx: _Context, n: int) -> list[Element]:
    """A one-cell unlabelled header row is the box's title; any other header
    row labels the columns below it."""
    texts = [(x0, x1, _text(p[0])) for x0, x1, p in row.cells if p]
    if not row.label and len(texts) == 1:
        title_lines = next(p[0] for _, _, p in row.cells if p)
        return [_element(KIND_HEADING, texts[0][2], title_lines, n, depth=0)]
    ctx.headers = texts
    return []


def _emit_box(box: _Box, ctx: _Context, first_on_page: bool) -> tuple[list[Element], list[_Move]]:
    """The box's elements in reading order, plus the continuations to join
    back into earlier pages."""
    out: list[Element] = []
    moves: list[_Move] = []
    n = box.page_no
    continues = first_on_page and ctx.box is not None and ctx.box.same_geometry(box)
    if not continues:
        ctx.headers, ctx.label, ctx.sub_labels = [], None, {}
        ctx.col_tail, ctx.row_tail = {}, None
    for i, row in enumerate(_group_rows(box.rows)):
        if _is_header_row(row):
            out.extend(_take_header_row(row, ctx, n))
            continue
        first = i == 0 and continues and ctx.label is not None
        if not row.label and first:
            joined = _continued_cells(row, ctx, n)
            if joined is not None:
                moves.extend(joined)
            else:
                out.extend(_repeat_labels(row, ctx, n))
            continue
        if row.label and first:
            moves.extend(_carry_over(row, ctx, n))
        if row.label:
            out.append(_element(KIND_HEADING, row.label, row.label_lines, n, depth=1))
            ctx.label, ctx.label_lines, ctx.sub_labels = row.label, row.label_lines, {}
            ctx.col_tail, ctx.row_tail = {}, None
        if len(row.subrows) > 1:
            out.extend(_emit_subrows(row, ctx, n))
        else:
            out.extend(_emit_cells(row, ctx, n))
        if row.label:
            ctx.split_rows = len(row.subrows) > 1
    ctx.box = box
    return out, moves


def _covered(el: Element, box: _Box) -> bool:
    cx, cy = el.bbox.center
    if el.kind == KIND_TABLE:
        overlap = min(el.bbox.bottom, box.bottom) - max(el.bbox.top, box.top)
        return overlap > 0.5 * max(el.bbox.height, 1.0)
    return box.left - SNAP_PT <= cx <= box.right + SNAP_PT and box.top <= cy <= box.bottom


def _apply_moves(doc: LayoutDocument, page_no: int, moves: list[_Move]) -> int:
    """Insert each continuation right after its anchor on an earlier page.
    The moved elements keep their own `page_no`, which assembly uses for the
    text's page (DEVIATIONS.md #239)."""
    done = 0
    for anchor, els in moves:
        for page in reversed(doc.pages[: page_no - 1]):
            idx = next((i for i, el in enumerate(page.elements) if el is anchor), None)
            if idx is not None:
                for el in els:
                    el.box_continued = True
                page.elements[idx + 1 : idx + 1] = els
                done += 1
                break
    return done


def restructure_label_boxes(doc: LayoutDocument) -> dict:
    """Rebuild every row-labelled box in reading order; returns a report."""
    ctx = _Context()
    report = {
        "boxes": 0,
        "rows": 0,
        "continued_rows": 0,
        "joined_cells": 0,
        "replaced_tables": 0,
    }
    for page in doc.pages:
        boxes = sorted(find_boxes(page), key=lambda b: b.top)
        if not boxes:
            if any(el.kind == KIND_HEADING for el in page.elements):
                ctx.box = None  # a new section ends any open box
            continue
        keep = [
            el
            for el in page.elements
            if el.kind in (KIND_PAGE_HEADER, KIND_PAGE_FOOTER)
            or not any(_covered(el, b) for b in boxes)
        ]
        report["replaced_tables"] += sum(
            1 for el in page.elements if el.kind == KIND_TABLE and el not in keep
        )
        body = [
            el
            for el in page.elements
            if el in keep and el.kind not in (KIND_PAGE_HEADER, KIND_PAGE_FOOTER)
        ]
        first_body_top = min((el.bbox.top for el in body), default=float("inf"))
        rebuilt: list[Element] = list(keep)
        for box in boxes:
            first_on_page = box.top <= first_body_top
            seq, moves = _emit_box(box, ctx, first_on_page)
            report["boxes"] += 1
            report["rows"] += len(box.rows)
            report["continued_rows"] += sum(
                1 for el in seq if el.box_continued and el.box_depth == 1
            )
            report["joined_cells"] += _apply_moves(doc, page.page_no, moves)
            at = next(
                (
                    i
                    for i, el in enumerate(rebuilt)
                    if el.kind not in (KIND_PAGE_HEADER, KIND_PAGE_FOOTER)
                    and el.bbox.top >= box.top
                ),
                len(rebuilt),
            )
            rebuilt[at:at] = seq
        page.elements = rebuilt
    return report


def apply_box_levels(doc: LayoutDocument) -> None:
    """After `fuse_heading_levels`: a box heading at depth d sits d levels
    below the most recent *top-level* section heading. In this guideline a
    box holds a whole syndrome's pathogens, treatment and comments, so it
    belongs to the syndrome ("9. SKIN AND SOFT TISSUE INFECTIONS"), not to a
    sub-heading that happens to precede it ("LRINEC risk assessment").
    A title (depth 0) that repeats that heading is dropped; a kept title
    sits one level below it and pushes the box's rows one further."""
    top_level: int | None = None
    base_level, base_text, offset = 0, "", 0
    for page in doc.pages:
        kept: list[Element] = []
        for el in page.elements:
            if el.kind == KIND_HEADING and el.box_depth is None:
                if el.level is not None and (top_level is None or el.level <= top_level):
                    top_level = el.level
                    base_level, base_text, offset = el.level, el.text, 0
            elif el.kind == KIND_HEADING and el.box_depth is not None:
                if el.box_depth == 0:
                    if _norm(el.text) in _norm(base_text):
                        offset = 0
                        continue  # the title repeats the section heading
                    el.level, offset = base_level + 1, 1
                else:
                    el.level = base_level + offset + el.box_depth
            kept.append(el)
        page.elements = kept


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
