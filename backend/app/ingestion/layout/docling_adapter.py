"""PDF -> `LayoutDocument` via Docling + pdfplumber (ARCH-044, PRD-113;
LAYOUT-INGESTION-PROPOSAL.md §4, §5.1-§5.2).

The **only** module that imports Docling or pdfplumber (both from the
`layout-parse` extra). Everything downstream works on
`app.ingestion.layout.model` dataclasses, so the ingestion logic is testable
offline against recorded JSON fixtures and a parser API change touches this
file only.

- **Docling** gives labelled layout items (section headers, text, list items,
  captions, footnotes, tables, pictures, page headers/footers), reading
  order, per-item bounding boxes, TableFormer cell structure, crops, and
  local OCR. Its per-line text cells carry `from_ocr` + `confidence`, which
  is how every element's origin (`text_layer` vs `ocr`) is decided.
- **pdfplumber** gives per-character font name and size (heading levels,
  boilerplate) and vector drawing primitives — rects, lines, curves — which
  is what vector flowcharts are made of (proposal §5.6 path A).

Nothing here calls a network service. Docling's layout/table model weights
and the OCR engine's weights must already be in the local cache
(`scripts/fetch_layout_models.py`); OCR runs in-process.
"""

from __future__ import annotations

import hashlib
import io
import logging
import re
import statistics
from functools import lru_cache
from importlib.metadata import version as pkg_version
from typing import Any

from app.ingestion.layout.model import (
    KIND_CAPTION,
    KIND_FOOTNOTE,
    KIND_HEADING,
    KIND_LIST_ITEM,
    KIND_PAGE_FOOTER,
    KIND_PAGE_HEADER,
    KIND_PICTURE,
    KIND_TABLE,
    KIND_TEXT,
    ORIGIN_OCR,
    ORIGIN_TEXT_LAYER,
    BBox,
    Drawing,
    Element,
    LayoutDocument,
    LayoutPage,
    TableCell,
    TableData,
    TextLine,
)

logger = logging.getLogger(__name__)

# Docling label value -> our element kind. Labels not listed are kept as
# plain text when they carry text, and dropped otherwise.
_LABEL_TO_KIND = {
    "section_header": KIND_HEADING,
    "title": KIND_HEADING,
    "text": KIND_TEXT,
    "paragraph": KIND_TEXT,
    "list_item": KIND_LIST_ITEM,
    "caption": KIND_CAPTION,
    "footnote": KIND_FOOTNOTE,
    "table": KIND_TABLE,
    "document_index": KIND_TABLE,
    "picture": KIND_PICTURE,
    "chart": KIND_PICTURE,
    "page_header": KIND_PAGE_HEADER,
    "page_footer": KIND_PAGE_FOOTER,
    "code": KIND_TEXT,
    "formula": KIND_TEXT,
}


def parser_version() -> str:
    """Identifies the parser stack a `document_version` was built with
    (`document_version.parser_version`)."""
    try:
        return f"docling-{pkg_version('docling')}+pdfplumber-{pkg_version('pdfplumber')}"
    except Exception:  # pragma: no cover - only when the extra is missing
        return "docling+pdfplumber"


# The corpus language (manifest `language`, BCP-47) in each engine's own
# language codes. Docling's defaults are multi-language (fr/de/es/en), which
# costs accuracy on an English corpus and makes Tesseract fail outright when
# those language packs aren't installed.
_OCR_LANG = {
    "tesseract": {"en": ["eng"]},
    "easyocr": {"en": ["en"]},
    # RapidOCR's default PP-OCR `ch` model reads Latin script well and scored
    # best in the bake-off; it is left at its default rather than remapped.
    "rapidocr": {},
}


@lru_cache(maxsize=8)
def _converter(ocr_engine: str, images_scale: float, lang: str = "en") -> Any:
    from docling.datamodel.base_models import InputFormat  # noqa: PLC0415
    from docling.datamodel.pipeline_options import (  # noqa: PLC0415
        EasyOcrOptions,
        PdfPipelineOptions,
        RapidOcrOptions,
        TesseractCliOcrOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption  # noqa: PLC0415

    engines = {
        "rapidocr": RapidOcrOptions,
        "easyocr": EasyOcrOptions,
        "tesseract": TesseractCliOcrOptions,
    }
    if ocr_engine not in engines:
        raise ValueError(
            f"unknown INGEST_OCR_ENGINE {ocr_engine!r}; expected one of {sorted(engines)}"
        )
    opts = PdfPipelineOptions()
    opts.do_ocr = True
    ocr_options = engines[ocr_engine]()
    if _OCR_LANG[ocr_engine]:
        codes = _OCR_LANG[ocr_engine].get(lang)
        if codes is None:
            raise ValueError(f"no {ocr_engine} language mapping for {lang!r}; add it to _OCR_LANG")
        ocr_options.lang = codes
    opts.ocr_options = ocr_options
    opts.do_table_structure = True
    opts.generate_picture_images = True
    opts.generate_table_images = True
    opts.generate_parsed_pages = True  # keeps per-line cells (from_ocr, confidence, font)
    opts.images_scale = images_scale
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )


def _png_bytes(img: Any) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _to_bbox(docling_bbox: Any, page_height: float) -> BBox:
    b = docling_bbox.to_top_left_origin(page_height=page_height)
    return BBox(float(b.l), float(b.t), float(b.r), float(b.b))


def _plumber_page_data(
    page: Any,
) -> tuple[list[dict], list[Drawing]]:
    chars = [
        {
            "x": (c["x0"] + c["x1"]) / 2,
            "y": (c["top"] + c["bottom"]) / 2,
            "size": float(c["size"]),
            "bold": "bold" in (c.get("fontname") or "").lower(),
            "font": c.get("fontname"),
        }
        for c in page.chars
        if not (c.get("text") or "").isspace()
    ]
    drawings: list[Drawing] = []
    for r in page.rects:
        drawings.append(
            Drawing(
                kind="rect",
                bbox=BBox(r["x0"], r["top"], r["x1"], r["bottom"]),
                filled=bool(r.get("fill")),
                stroked=bool(r.get("stroke")),
            )
        )
    for ln in page.lines:
        drawings.append(
            Drawing(
                kind="line",
                bbox=BBox(ln["x0"], ln["top"], ln["x1"], ln["bottom"]),
                filled=False,
                # pdfplumber `pts` are (x, top) pairs: the line's real end points,
                # which the bbox alone can't give for a diagonal connector.
                points=[(float(x), float(y)) for x, y in (ln.get("pts") or [])],
            )
        )
    for cv in page.curves:
        drawings.append(
            Drawing(
                kind="curve",
                bbox=BBox(cv["x0"], cv["top"], cv["x1"], cv["bottom"]),
                filled=bool(cv.get("fill")),
                stroked=bool(cv.get("stroke")),
            )
        )
    return chars, drawings


def prefer_text_layer(docling_text: str, plumber_text: str) -> tuple[str, bool]:
    """The text-layer string for a region: pdfplumber's reading of the PDF's
    own characters when it has any, otherwise Docling's (OCR regions have no
    characters). Returns `(text, repaired)`, where `repaired` means Docling's
    string differed.

    Why pdfplumber: Docling's own PDF text decoding dropped every "h" in the
    Inter font NICE NG195 uses ("wit out", "W at is t e") while pdfplumber
    and pypdf read the same characters correctly (DEVIATIONS.md #221).
    Docling stays the authority for *layout* (labels, reading order, table
    structure); pdfplumber supplies the characters."""
    plumber = " ".join(plumber_text.split())
    if not plumber:
        return docling_text, False
    # Docling strips a list item's printed bullet (assembly adds "• " back);
    # pdfplumber keeps it. Keep the two consistent.
    if not _LEADING_BULLET_RE.match(docling_text.strip()):
        plumber = _LEADING_BULLET_RE.sub("", plumber, count=1)
    return plumber, " ".join(docling_text.split()) != plumber


MIN_LINE_Y_TOLERANCE = 3.0  # pdfplumber's own default
SINGLE_LINE_Y_TOLERANCE = 1_000.0  # one physical line: never split it
_LEADING_BULLET_RE = re.compile(r"^[•▪◦●○■□‣⁃∙·\-–*]\s*")


def _plumber_text(page_chars: list[dict], bbox: BBox, *, single_line: bool = False) -> str:
    """pdfplumber's reading of the characters whose centre lies in `bbox`.

    Sub/superscripts sit off the baseline, and pdfplumber's default line
    clustering read "SpO₂ below 90%" as "SpO below 90% 2". A Docling text
    line is one physical line, so its characters are simply ordered left to
    right (`single_line`). For multi-line regions the line tolerance scales
    with the font: half the median character size."""
    from pdfplumber.utils import extract_text  # noqa: PLC0415

    inside = [
        c
        for c in page_chars
        if bbox.contains_point((c["x0"] + c["x1"]) / 2, (c["top"] + c["bottom"]) / 2, tol=0.5)
    ]
    if not inside:
        return ""
    if single_line:
        y_tol = SINGLE_LINE_Y_TOLERANCE
    else:
        y_tol = max(MIN_LINE_Y_TOLERANCE, 0.5 * statistics.median(float(c["size"]) for c in inside))
    return extract_text(inside, y_tolerance=y_tol)


def _font_signal(chars: list[dict], bbox: BBox) -> tuple[float | None, bool]:
    inside = [c for c in chars if bbox.contains_point(c["x"], c["y"], tol=0.5)]
    if not inside:
        return None, False
    size = statistics.median(c["size"] for c in inside)
    bold = sum(c["bold"] for c in inside) > len(inside) / 2
    return round(size, 2), bold


def _page_lines(
    parsed_page: Any, page_height: float, chars: list[dict], ppage: Any, repairs: list[int]
) -> list[TextLine]:
    lines: list[TextLine] = []
    if parsed_page is None:
        return lines
    for cell in parsed_page.textline_cells:
        text = (cell.text or "").strip()
        if not text:
            continue
        rect = cell.rect.to_bounding_box().to_top_left_origin(page_height=page_height)
        bbox = BBox(float(rect.l), float(rect.t), float(rect.r), float(rect.b))
        if not cell.from_ocr:
            text, repaired = prefer_text_layer(
                text, _plumber_text(ppage.chars, bbox, single_line=True)
            )
            repairs[0] += repaired
        size, bold = (None, False) if cell.from_ocr else _font_signal(chars, bbox)
        lines.append(
            TextLine(
                text=text,
                bbox=bbox,
                from_ocr=bool(cell.from_ocr),
                confidence=float(cell.confidence if cell.confidence is not None else 1.0),
                font_name=getattr(cell, "font_name", None) or None,
                font_size=size,
                is_bold=bold,
            )
        )
    return lines


def _lines_in(lines: list[TextLine], bbox: BBox) -> list[TextLine]:
    return [ln for ln in lines if bbox.contains_point(*ln.bbox.center, tol=1.0)]


def _origin_of(lines: list[TextLine]) -> tuple[str, float | None]:
    """The element's dominant origin: `ocr` if most of its characters came
    from OCR lines. `ocr_min_confidence` is over its OCR lines only."""
    if not lines:
        return ORIGIN_TEXT_LAYER, None
    ocr_chars = sum(len(ln.text) for ln in lines if ln.from_ocr)
    total = sum(len(ln.text) for ln in lines) or 1
    ocr_conf = [ln.confidence for ln in lines if ln.from_ocr]
    origin = ORIGIN_OCR if ocr_chars * 2 > total else ORIGIN_TEXT_LAYER
    return origin, (round(min(ocr_conf), 4) if ocr_conf else None)


def _table_data(item: Any, ppage: Any, page_height: float, repairs: list[int]) -> TableData:
    data = item.data
    cells = []
    for c in data.table_cells:
        text = (c.text or "").strip()
        if getattr(c, "bbox", None) is not None:
            b = c.bbox.to_top_left_origin(page_height=page_height)
            text, repaired = prefer_text_layer(
                text,
                _plumber_text(ppage.chars, BBox(float(b.l), float(b.t), float(b.r), float(b.b))),
            )
            repairs[0] += repaired
        cells.append(
            TableCell(
                row=c.start_row_offset_idx,
                col=c.start_col_offset_idx,
                row_span=max(1, c.end_row_offset_idx - c.start_row_offset_idx),
                col_span=max(1, c.end_col_offset_idx - c.start_col_offset_idx),
                text=text,
                is_header=bool(c.column_header),
            )
        )
    return TableData(num_rows=data.num_rows, num_cols=data.num_cols, cells=cells)


def render_region(
    path: str,
    page_no: int,
    bbox: BBox,
    *,
    scale: float,
    pad: float,
    max_bytes: int,
    mask: list[BBox] | None = None,
) -> bytes:
    """PNG of one page region (top-left-origin points) for the vision
    transcription (proposal §18.4): rendered at `scale`, padded by `pad`
    points, downscaled in steps until it is under `max_bytes`. `mask` regions
    (other elements overlapping the crop) are painted white, so a prose crop
    shows only its own element — tightly spaced bullets otherwise bled into
    each other's transcriptions (DEVIATIONS.md #222)."""
    import pypdfium2 as pdfium  # noqa: PLC0415 - comes with the layout-parse extra

    pdf = pdfium.PdfDocument(path)
    try:
        page = pdf[page_no - 1]
        width, height = page.get_size()
        box = (
            max(bbox.x0 - pad, 0.0),
            max(bbox.top - pad, 0.0),
            min(bbox.x1 + pad, width),
            min(bbox.bottom + pad, height),
        )
        s = scale
        while True:
            img = page.render(scale=s).to_pil()
            if mask:
                from PIL import ImageDraw  # noqa: PLC0415

                draw = ImageDraw.Draw(img)
                for m in mask:
                    draw.rectangle([m.x0 * s, m.top * s, m.x1 * s, m.bottom * s], fill="white")
            crop = img.crop(tuple(int(v * s) for v in box))
            png = _png_bytes(crop)
            if len(png) <= max_bytes or s <= 1.0:
                return png
            s = max(1.0, s * 0.75)
    finally:
        pdf.close()


def parse_layout(
    path: str, *, ocr_engine: str, images_scale: float = 2.0, lang: str = "en"
) -> LayoutDocument:
    """Convert one PDF. Raises on any parser failure — the caller
    (`app.ingestion.layout.pipeline`) decides whether to fall back to `pypdf`."""
    import pdfplumber  # noqa: PLC0415
    from docling_core.types.doc import ContentLayer  # noqa: PLC0415

    result = _converter(ocr_engine, images_scale, lang).convert(path)
    doc = result.document
    parsed_by_page = {p.page_no: p.parsed_page for p in result.pages}  # 1-based

    crops: dict[str, bytes] = {}
    pages: dict[int, LayoutPage] = {}
    plumber_chars: dict[int, list[dict]] = {}
    repairs = [0]  # text-layer strings where Docling's decoding differed from pdfplumber's
    with pdfplumber.open(path) as pdf:
        ppages = {i: p for i, p in enumerate(pdf.pages, start=1)}
        for i, ppage in ppages.items():
            chars, drawings = _plumber_page_data(ppage)
            plumber_chars[i] = chars
            height = float(ppage.height)
            pages[i] = LayoutPage(
                page_no=i,
                width=float(ppage.width),
                height=height,
                elements=[],
                lines=_page_lines(parsed_by_page.get(i), height, chars, ppage, repairs),
                drawings=drawings,
            )
        _collect_elements(doc, pages, ppages, plumber_chars, crops, repairs, ContentLayer)

    return LayoutDocument(
        source_path=path,
        parser_version=parser_version(),
        pages=[pages[k] for k in sorted(pages)],
        crops=crops,
        text_repairs=repairs[0],
    )


def _collect_elements(  # noqa: PLR0917 - internal helper split out of parse_layout
    doc: Any,
    pages: dict[int, LayoutPage],
    ppages: dict[int, Any],
    plumber_chars: dict[int, list[dict]],
    crops: dict[str, bytes],
    repairs: list[int],
    content_layer: Any,
) -> None:
    for item, _level in doc.iterate_items(included_content_layers=set(content_layer)):
        label = getattr(getattr(item, "label", None), "value", None)
        kind = _LABEL_TO_KIND.get(label or "")
        prov = item.prov[0] if getattr(item, "prov", None) else None
        if prov is None or prov.page_no not in pages:
            continue
        page = pages[prov.page_no]
        text = (getattr(item, "text", "") or "").strip()
        if kind is None:
            if not text:
                continue
            kind = KIND_TEXT
        bbox = _to_bbox(prov.bbox, page.height)
        inside = _lines_in(page.lines, bbox)
        origin, ocr_min_conf = _origin_of(inside)
        if text and origin == ORIGIN_TEXT_LAYER and kind not in (KIND_TABLE, KIND_PICTURE):
            text, repaired = prefer_text_layer(
                text, _plumber_text(ppages[prov.page_no].chars, bbox)
            )
            repairs[0] += repaired
        font_size, bold = _font_signal(plumber_chars[prov.page_no], bbox)
        el = Element(
            kind=kind,
            page_no=prov.page_no,
            bbox=bbox,
            text=text,
            level=getattr(item, "level", None),
            origin=origin,
            ocr_min_confidence=ocr_min_conf,
            font_size=font_size,
            is_bold=bold,
        )
        if kind in (KIND_TABLE, KIND_PICTURE):
            try:
                el.caption = (item.caption_text(doc) or "").strip() or None
            except Exception:
                el.caption = None
            img = item.get_image(doc)
            if img is not None:
                png = _png_bytes(img)
                sha = hashlib.sha256(png).hexdigest()
                crops[sha] = png
                el.image_sha256 = sha
        if kind == KIND_TABLE:
            el.table = _table_data(item, ppages[prov.page_no], page.height, repairs)
        if kind == KIND_PICTURE:
            el.inner_lines = inside
        page.elements.append(el)
