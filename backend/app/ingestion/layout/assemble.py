"""`LayoutDocument` -> `ParsedDocument` (ARCH-044; LAYOUT-INGESTION-PROPOSAL.md
§5.3-§5.8, §5.10-§5.11).

Order of operations:

1. header/footer removal (`boilerplate`), heading-level fusion (`headings`);
2. flowchart regions the layout model didn't label as pictures are found
   from their geometry (≥ 3 text-bearing boxes joined by connectors);
3. each element becomes a `Block` with **citable text only**: verbatim
   text-layer or OCR text, plus deterministic `structure` (list bullets,
   table markdown, flowchart node ids and verified edges). Model-generated
   text never enters a block;
4. attested text corrections are applied to the blocks of their page
   (exactly one match or ingestion stops);
5. blocks are joined with blank lines into `normalized_text`, which is what
   every chunk offset and citation offset refers to (ARCH §8.1), and the
   section tree is built from the heading blocks.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Callable
from dataclasses import dataclass, field

from app.ingestion.corrections import (
    AppliedCorrection,
    Correction,
    apply_all_to_units,
    apply_to_units,
    correct_copy,
)
from app.ingestion.layout.boilerplate import drop_leading_excerpt_fragment, remove_boilerplate
from app.ingestion.layout.flowchart import (
    MIN_NODE_SIDE,
    UNVERIFIED,
    FlowGraph,
    apply_attestations,
    extract_flowchart,
    serialize,
)
from app.ingestion.layout.headings import fuse_heading_levels, split_number
from app.ingestion.layout.label_boxes import apply_box_levels, restructure_label_boxes
from app.ingestion.layout.model import (
    KIND_CAPTION,
    KIND_FOOTNOTE,
    KIND_HEADING,
    KIND_LIST_ITEM,
    KIND_PICTURE,
    KIND_TABLE,
    KIND_TEXT,
    ORIGIN_OCR,
    ORIGIN_TEXT_LAYER,
    ORIGIN_VLM,
    BBox,
    Element,
    LayoutDocument,
    LayoutPage,
    TableData,
)
from app.ingestion.layout.tables import render_table
from app.ingestion.pdf_parse import Block, ParsedDocument

_DIGIT_RE = re.compile(r"\d")
MIN_FIGURE_PX = 60  # crops smaller than this on either side are logos/icons (reference design)
MIN_UNLABELLED_FLOWCHART_BOXES = 3
_BLOCK_SEP = "\n\n"


@dataclass
class AssemblyOptions:
    margin_zone: float = 0.08
    repeat_ratio: float = 0.5
    boilerplate_min_pages: int = 3
    boilerplate_patterns: list[str] = field(default_factory=list)
    table_max_tokens: int = 700
    # Measures table size for row-group splitting; None = whitespace words.
    token_counter: Callable[[str], int] | None = None
    images_scale: float = 2.0
    low_density_chars: int = 200
    source_pages: list[int] | None = None  # physical page i -> source page source_pages[i-1]
    corrections: list[Correction] = field(default_factory=list)
    flowchart_attestations: list[dict] = field(default_factory=list)
    # Vision-LLM table transcription (D12, proposal §18): called for each OCR
    # table; returns (merged table, meta) or None to keep the OCR table.
    table_transcriber: Callable[[Element], tuple[TableData, dict] | None] | None = None
    # Manifest `table_sources` (proposal §18.7): per-table source choice,
    # {(source page, table index on that page): entry}. `source: "ocr"` keeps
    # a table off the vision path, e.g. after a reviewer rejected its
    # transcription.
    table_sources: dict[tuple[int, int], dict] = field(default_factory=dict)
    # Vision transcription of OCR prose (DEVIATIONS.md #222): returns
    # (text, meta) or None to keep the OCR text.
    prose_transcriber: Callable[[Element], tuple[str, dict] | None] | None = None


@dataclass
class _Unit:
    kind: str
    page_no: int
    text: str
    origin: str = ORIGIN_TEXT_LAYER
    structure: list[tuple[int, int]] = field(default_factory=list)  # relative to text
    meta: dict = field(default_factory=dict)
    heading_level: int | None = None


def _source_page(page_no: int, source_pages: list[int] | None) -> int:
    if source_pages and 1 <= page_no <= len(source_pages):
        return source_pages[page_no - 1]
    return page_no


def _ocr_meta(origin: str, min_conf: float | None, text: str) -> dict:
    if origin != ORIGIN_OCR:
        return {}
    return {
        "ocr": {
            "min_confidence": min_conf,
            "has_digits": bool(_DIGIT_RE.search(text)),
        }
    }


def _find_unlabelled_flowcharts(page: LayoutPage) -> None:
    """Promote a cluster of ≥ 3 text-bearing stroked boxes joined by
    connectors to a picture element when the layout model left it as loose
    text, absorbing the text elements inside it."""
    covered = [el.bbox for el in page.elements if el.kind in (KIND_PICTURE, KIND_TABLE)]
    boxes = [
        d.bbox
        for d in page.drawings
        if d.kind == "rect"
        and d.stroked
        and d.bbox.width >= MIN_NODE_SIDE
        and d.bbox.height >= MIN_NODE_SIDE
        and not any(c.contains(d.bbox, tol=2.0) for c in covered)
        and any(d.bbox.contains_point(*ln.bbox.center) for ln in page.lines)
    ]
    if len(boxes) < MIN_UNLABELLED_FLOWCHART_BOXES:
        return
    region = BBox(
        min(b.x0 for b in boxes) - 4,
        min(b.top for b in boxes) - 4,
        max(b.x1 for b in boxes) + 4,
        max(b.bottom for b in boxes) + 4,
    )
    inner = [ln for ln in page.lines if region.contains_point(*ln.bbox.center)]
    graph = extract_flowchart(region, page.drawings, inner)
    if graph is None or not graph.edges:
        return
    absorbed = [
        el
        for el in page.elements
        if el.kind != KIND_PICTURE and region.contains_point(*el.bbox.center)
    ]
    first = min((page.elements.index(el) for el in absorbed), default=len(page.elements))
    page.elements = [el for el in page.elements if el not in absorbed]
    page.elements.insert(
        min(first, len(page.elements)),
        Element(kind=KIND_PICTURE, page_no=page.page_no, bbox=region, inner_lines=inner),
    )


def _units_for_page(page: LayoutPage, opts: AssemblyOptions, report: dict) -> list[_Unit]:
    units: list[_Unit] = []
    table_index = 0
    for el in page.elements:
        # A row-labelled box cell continued from the next page is placed in
        # this page's list but keeps its own page (DEVIATIONS.md #239).
        if el.kind == KIND_HEADING:
            units.append(
                _Unit(
                    KIND_HEADING,
                    el.page_no,
                    el.text,
                    el.origin,
                    heading_level=el.level or 1,
                    meta=_ocr_meta(el.origin, el.ocr_min_confidence, el.text),
                )
            )
        elif (
            el.kind in _PROSE_KINDS
            and el.origin == ORIGIN_OCR
            and opts.prose_transcriber
            and el.page_no == page.page_no
        ):
            units.append(_prose_unit(el, page, opts, report))
        elif el.kind == KIND_LIST_ITEM:
            units.append(
                _Unit(
                    KIND_LIST_ITEM,
                    el.page_no,
                    f"• {el.text}",
                    el.origin,
                    structure=[(0, 2)],
                    meta=_ocr_meta(el.origin, el.ocr_min_confidence, el.text),
                )
            )
        elif el.kind in (KIND_TEXT, KIND_CAPTION, KIND_FOOTNOTE):
            if el.text:
                units.append(
                    _Unit(
                        el.kind,
                        el.page_no,
                        el.text,
                        el.origin,
                        meta=_ocr_meta(el.origin, el.ocr_min_confidence, el.text),
                    )
                )
        elif el.kind == KIND_TABLE and el.table is not None:
            units.append(_table_unit(el, page, opts, report, table_index))
            table_index += 1
            report["tables"] += 1
        elif el.kind == KIND_PICTURE:
            unit = _picture_unit(el, page, opts, report)
            if unit is not None:
                units.append(unit)
    return units


_PROSE_KINDS = frozenset({KIND_TEXT, KIND_LIST_ITEM, KIND_CAPTION, KIND_FOOTNOTE})


def _prose_unit(el: Element, page: LayoutPage, opts: AssemblyOptions, report: dict) -> _Unit:
    """An OCR paragraph / list item / caption / footnote, re-transcribed by
    the vision model when it passes the checks (otherwise the OCR text)."""
    assert opts.prose_transcriber is not None
    text, origin = el.text, el.origin
    extra: dict = {}
    transcribed = opts.prose_transcriber(el)
    if transcribed is not None:
        text, extra = transcribed
        origin = ORIGIN_VLM
        report["prose_vlm"] = report.get("prose_vlm", 0) + 1
    meta = {**_ocr_meta(ORIGIN_OCR, el.ocr_min_confidence, text), **extra}
    if el.kind == KIND_LIST_ITEM:
        return _Unit(
            KIND_LIST_ITEM, page.page_no, f"• {text}", origin, structure=[(0, 2)], meta=meta
        )
    return _Unit(el.kind, page.page_no, text, origin, meta=meta)


def _table_unit(
    el: Element, page: LayoutPage, opts: AssemblyOptions, report: dict, index: int
) -> _Unit:
    table, origin = el.table, el.origin
    extra: dict = {}
    assert table is not None
    override = opts.table_sources.get((_source_page(page.page_no, opts.source_pages), index))
    if override is not None:
        extra["table_source"] = {k: override.get(k) for k in ("source", "reason", "decided_by")}
        report.setdefault("table_sources_applied", []).append(
            {
                "page": _source_page(page.page_no, opts.source_pages),
                "table_index": index,
                "source": override.get("source"),
            }
        )
    use_vlm = override is None or override.get("source") == "vlm"
    if use_vlm and opts.table_transcriber is not None and el.origin == ORIGIN_OCR:
        transcribed = opts.table_transcriber(el)
        if transcribed is not None:
            ocr_render = render_table(table, max_tokens=10**9)
            table, extra = transcribed
            origin = ORIGIN_VLM
            # The OCR reading stays beside the transcription for the reviewer.
            extra["ocr_alternative"] = ocr_render.markdown
            report["tables_vlm"] = report.get("tables_vlm", 0) + 1
    render = render_table(table, max_tokens=opts.table_max_tokens, count_tokens=opts.token_counter)
    text = _BLOCK_SEP.join(render.parts)
    meta = {
        "table_parts": render.parts,
        "table_grid": render.grid_markdown,
        "header_paths": render.header_paths,
        "caption": el.caption,
        "figure_ref": {"bbox": el.bbox.as_list(), "image_sha256": el.image_sha256},
        # Transcribed cells were OCR'd too: keep the OCR statistics.
        **_ocr_meta(el.origin, el.ocr_min_confidence, text),
        **extra,
    }
    return _Unit("table", page.page_no, text, origin, meta=meta)


def _picture_unit(
    el: Element, page: LayoutPage, opts: AssemblyOptions, report: dict
) -> _Unit | None:
    if min(el.bbox.width, el.bbox.height) * opts.images_scale < MIN_FIGURE_PX:
        report["figures_skipped_small"] += 1
        return None
    figure_ref = {"bbox": el.bbox.as_list(), "image_sha256": el.image_sha256}
    graph: FlowGraph | None = extract_flowchart(el.bbox, page.drawings, el.inner_lines)
    if graph is not None:
        if opts.flowchart_attestations:
            apply_attestations(graph, opts.flowchart_attestations)
        text, structure = serialize(graph)
        origin = (
            ORIGIN_OCR if any(n.origin == ORIGIN_OCR for n in graph.nodes) else ORIGIN_TEXT_LAYER
        )
        conf = [ln.confidence for ln in el.inner_lines if ln.from_ocr]
        meta = {
            "flowchart": graph.to_meta(),
            "figure_ref": figure_ref,
            "caption": el.caption,
            **_ocr_meta(origin, min(conf) if conf else None, text),
        }
        report["flowcharts"][graph.verification] = (
            report["flowcharts"].get(graph.verification, 0) + 1
        )
        return _Unit("flowchart", page.page_no, text, origin, structure=structure, meta=meta)

    inner = "\n".join(
        ln.text for ln in sorted(el.inner_lines, key=lambda ln: (round(ln.bbox.top), ln.bbox.x0))
    )
    parts = [p for p in (el.caption, inner) if p]
    has_text = bool(parts)
    text = "\n".join(parts) if has_text else "(figure, no caption detected)"
    origin = el.origin if inner else ORIGIN_TEXT_LAYER
    conf = [ln.confidence for ln in el.inner_lines if ln.from_ocr]
    meta = {
        "figure_ref": figure_ref,
        "caption": el.caption,
        "has_embedded_text": has_text,
        **_ocr_meta(origin, min(conf) if conf else None, inner),
    }
    report["figures"] += 1
    return _Unit(
        "figure",
        page.page_no,
        text,
        origin,
        structure=[] if has_text else [(0, len(text))],
        meta=meta,
    )


def _apply_corrections(units: list[_Unit], opts: AssemblyOptions) -> list[dict]:
    applied: list[dict] = []
    for corr in opts.corrections:
        idx = [
            i
            for i, u in enumerate(units)
            if _source_page(u.page_no, opts.source_pages) == corr.page
        ]
        page_units = [units[i] for i in idx]
        texts = [u.text for u in page_units]
        protected = [u.structure for u in page_units]
        if corr.occurrences == "all":
            new_texts, done_list = apply_all_to_units(texts, corr, protected=protected)
        else:
            new_texts, done = apply_to_units(texts, corr, protected=protected)
            done_list = [done]
        touched: set[int] = set()
        for done in done_list:
            unit = page_units[done.unit_index]
            unit.structure = _shift(unit.structure, done)
            unit.meta.setdefault("corrections", []).append(
                {**corr.public(), "start": done.start, "end": done.end}
            )
            touched.add(done.unit_index)
        for k in touched:
            unit = page_units[k]
            unit.text = new_texts[k]
            if unit.kind == "table":
                _correct_table_copies(unit, corr)
        applied.append(
            {
                "id": corr.id,
                "page": corr.page,
                "unit_kind": page_units[done_list[0].unit_index].kind,
                "occurrences": len(done_list),
            }
        )
    return applied


def _correct_table_copies(unit: _Unit, corr: Correction) -> None:
    """A table's parts, grid and header paths are copies of its text that
    chunking and the review page read; they must carry the same correction
    (DEVIATIONS.md #243). Chunking locates each part in the document text, so
    a stale part would not be found."""
    meta = unit.meta
    if meta.get("table_parts"):
        meta["table_parts"] = [correct_copy(t, corr) for t in meta["table_parts"]]
    if meta.get("table_grid"):
        meta["table_grid"] = correct_copy(meta["table_grid"], corr)
    if meta.get("header_paths"):
        meta["header_paths"] = [correct_copy(t, corr) for t in meta["header_paths"]]


def _shift(spans: list[tuple[int, int]], done: AppliedCorrection) -> list[tuple[int, int]]:
    return [
        (s + done.delta, e + done.delta) if s >= done.original_end else (s, e) for s, e in spans
    ]


def _sections(blocks: list[Block], doc_len: int) -> list[dict]:
    sections: list[dict] = []
    stack: list[tuple[int, str]] = []
    heading_blocks = [b for b in blocks if b.kind == KIND_HEADING]
    if blocks and (not heading_blocks or blocks[0].kind != KIND_HEADING):
        first_heading_start = heading_blocks[0].char_start if heading_blocks else doc_len
        sections.append(
            {
                "number": None,
                "heading": None,
                "path": None,
                "char_start": 0,
                "char_end": first_heading_start,
                "has_heading_line": False,
            }
        )
    for i, b in enumerate(heading_blocks):
        number, title = split_number(b.meta["heading_text"])
        level = b.meta.get("heading_level") or 1
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        end = heading_blocks[i + 1].char_start if i + 1 < len(heading_blocks) else doc_len
        sections.append(
            {
                "number": number,
                "heading": title,
                "path": " › ".join(t for _, t in stack),
                "char_start": b.char_start,
                "char_end": end,
                "has_heading_line": True,
            }
        )
    return sections


def _parse_quality(units: list[_Unit], doc: LayoutDocument, report: dict) -> float:
    if not units:
        return 0.0
    pages_without_text = sum(
        1
        for p in doc.pages
        if not any(u.page_no == p.page_no and u.kind != "figure" for u in units)
    )
    confs = [
        ln.confidence
        for p in doc.pages
        for ln in p.lines
        if ln.from_ocr and len(ln.text.strip()) > 1
    ]
    mean_conf = statistics.mean(confs) if confs else 1.0
    n_flow = sum(report["flowcharts"].values())
    unverified = report["flowcharts"].get(UNVERIFIED, 0)
    q = 0.6 + 0.4 * mean_conf
    q -= 0.3 * pages_without_text / max(len(doc.pages), 1)
    q -= 0.1 * (unverified / n_flow if n_flow else 0.0)
    return round(max(0.0, min(1.0, q)), 3)


def assemble(doc: LayoutDocument, opts: AssemblyOptions) -> ParsedDocument:
    report: dict = {
        "parser": "layout",
        "parser_version": doc.parser_version,
        "tables": 0,
        "figures": 0,
        "figures_skipped_small": 0,
        "flowcharts": {},
        "text_layer_repairs": doc.text_repairs,
    }
    bp = remove_boilerplate(
        doc,
        margin_zone=opts.margin_zone,
        repeat_ratio=opts.repeat_ratio,
        min_pages=opts.boilerplate_min_pages,
        manifest_patterns=opts.boilerplate_patterns,
    )
    report["dropped_boilerplate"] = {"count": bp.dropped_count, "distinct": bp.distinct}
    report["label_boxes"] = restructure_label_boxes(doc)
    fuse_heading_levels(doc)
    apply_box_levels(doc)
    if opts.source_pages:
        report["excerpt_leading_fragment_dropped"] = drop_leading_excerpt_fragment(doc)
    for page in doc.pages:
        _find_unlabelled_flowcharts(page)

    units: list[_Unit] = []
    for page in doc.pages:
        units.extend(_units_for_page(page, opts, report))
    report["corrections_applied"] = _apply_corrections(units, opts)

    parts: list[str] = []
    blocks: list[Block] = []
    # One entry per *run* of a page, not per page: a box cell continued onto
    # the next page is joined back into its cell, so text can go p.34 → p.35
    # → p.34 (DEVIATIONS.md #239). `page_for_offset` takes the last run
    # starting at or before an offset, so every offset keeps its own page.
    runs: list[tuple[int, int]] = []
    offset = 0
    for i, u in enumerate(units):
        if i:
            parts.append(_BLOCK_SEP)
            offset += len(_BLOCK_SEP)
        if not runs or runs[-1][0] != u.page_no:
            runs.append((u.page_no, offset))
        meta = dict(u.meta)
        meta["structure_spans"] = u.structure
        if u.kind == KIND_HEADING:
            meta["heading_text"] = u.text
            meta["heading_level"] = u.heading_level
        blocks.append(Block(u.kind, offset, offset + len(u.text), u.page_no, u.origin, meta))
        parts.append(u.text)
        offset += len(u.text)
    normalized = "".join(parts)

    # Pages with no blocks start where the next page with content starts.
    with_text = {p for p, _ in runs}
    starts: list[tuple[int, int]] = []
    for page in doc.pages:
        if page.page_no in with_text:
            starts.extend(r for r in runs if r[0] == page.page_no and r not in starts)
            continue
        later = [start for p, start in runs if p > page.page_no]
        starts.append((page.page_no, min(later) if later else len(normalized)))
    starts.sort(key=lambda r: r[1])

    lines = [ln for p in doc.pages for ln in p.lines]
    ocr_lines = [ln for ln in lines if ln.from_ocr and len(ln.text.strip()) > 1]
    report["ocr"] = {
        "lines": len(ocr_lines),
        "pages": sorted(
            {
                p.page_no
                for p in doc.pages
                for ln in p.lines
                if ln.from_ocr and len(ln.text.strip()) > 1
            }
        ),
        "mean_confidence": round(statistics.mean(ln.confidence for ln in ocr_lines), 4)
        if ocr_lines
        else None,
        "min_confidence": round(min(ln.confidence for ln in ocr_lines), 4) if ocr_lines else None,
    }
    report["low_density_pages"] = [
        p.page_no
        for p in doc.pages
        if sum(len(ln.text) for ln in p.lines if not ln.from_ocr) < opts.low_density_chars
    ]
    report["pages_without_text"] = [
        p.page_no for p in doc.pages if not any(u.page_no == p.page_no for u in units)
    ]

    parsed = ParsedDocument(
        normalized_text=normalized,
        sections=[],
        page_count=len(doc.pages),
        parse_quality=_parse_quality(units, doc, report),
        page_starts=starts,
        source_path=doc.source_path,
        blocks=blocks,
        parser_version=doc.parser_version,
        parse_report=report,
    )
    sections = _sections(blocks, len(normalized))
    for s in sections:
        s["page_start"] = parsed.page_for_offset(s["char_start"])
        s["page_end"] = parsed.page_for_offset(max(s["char_end"] - 1, s["char_start"]))
    parsed.sections = sections
    return parsed
