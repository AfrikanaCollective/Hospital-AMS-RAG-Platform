"""Parser-independent layout representation (ARCH-044, PRD-113;
LAYOUT-INGESTION-PROPOSAL.md §4-§5).

`app.ingestion.layout.docling_adapter` is the only module that imports
Docling or pdfplumber. It converts a PDF into these plain dataclasses, and
everything downstream (boilerplate removal, heading fusion, tables,
flowcharts, assembly) works on them alone. That keeps the logic testable
offline from recorded JSON fixtures (CLAUDE.md §5) and confines a parser API
change to one file.

Coordinates are PDF points with a **top-left origin** (y grows downward),
per physical page of the file being ingested (not the `source_pages`
remapped page number — that remap happens after assembly, exactly as for
the `pypdf` path).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# Text origins (LAYOUT-INGESTION-PROPOSAL.md §3). `derived` (model-generated)
# text never appears in these structures' citable fields at all.
ORIGIN_TEXT_LAYER = "text_layer"
ORIGIN_OCR = "ocr"
ORIGIN_STRUCTURE = "structure"
ORIGIN_ATTESTED = "attested"
# A vision model's transcription of printed table content (D12, proposal §18):
# citable only after admin confirmation against the crop.
ORIGIN_VLM = "vlm_transcription"

# Element kinds, normalised from the parser's own labels.
KIND_HEADING = "heading"
KIND_TEXT = "text"
KIND_LIST_ITEM = "list_item"
KIND_CAPTION = "caption"
KIND_FOOTNOTE = "footnote"
KIND_TABLE = "table"
KIND_PICTURE = "picture"
KIND_PAGE_HEADER = "page_header"
KIND_PAGE_FOOTER = "page_footer"

TEXTUAL_KINDS = frozenset({KIND_HEADING, KIND_TEXT, KIND_LIST_ITEM, KIND_CAPTION, KIND_FOOTNOTE})


@dataclass(frozen=True)
class BBox:
    x0: float
    top: float
    x1: float
    bottom: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2, (self.top + self.bottom) / 2)

    def contains_point(self, x: float, y: float, tol: float = 0.0) -> bool:
        return self.x0 - tol <= x <= self.x1 + tol and self.top - tol <= y <= self.bottom + tol

    def contains(self, other: BBox, tol: float = 1.0) -> bool:
        return (
            self.x0 - tol <= other.x0
            and self.top - tol <= other.top
            and other.x1 <= self.x1 + tol
            and other.bottom <= self.bottom + tol
        )

    def as_list(self) -> list[float]:
        return [round(self.x0, 2), round(self.top, 2), round(self.x1, 2), round(self.bottom, 2)]

    @classmethod
    def from_list(cls, v: list[float]) -> BBox:
        return cls(v[0], v[1], v[2], v[3])


@dataclass
class TextLine:
    """One physical line of text as the parser saw it. Used for heading-font
    signals, OCR provenance, and flowchart node/edge-label geometry."""

    text: str
    bbox: BBox
    from_ocr: bool = False
    confidence: float = 1.0
    font_name: str | None = None
    font_size: float | None = None
    is_bold: bool = False


@dataclass
class TableCell:
    row: int
    col: int
    row_span: int
    col_span: int
    text: str
    is_header: bool = False


@dataclass
class TableData:
    num_rows: int
    num_cols: int
    cells: list[TableCell]


@dataclass
class Element:
    kind: str
    page_no: int  # physical 1-based page in the ingested file
    bbox: BBox
    text: str = ""
    level: int | None = None  # parser-reported heading level (often flat; see headings.py)
    origin: str = ORIGIN_TEXT_LAYER  # text_layer | ocr (dominant origin of `text`)
    ocr_min_confidence: float | None = None
    font_size: float | None = None
    is_bold: bool = False
    table: TableData | None = None
    caption: str | None = None
    image_sha256: str | None = None  # picture/table crop, when rendered
    inner_lines: list[TextLine] = field(default_factory=list)  # text inside a picture region


@dataclass
class Drawing:
    """A vector drawing primitive. `kind`: rect | line | curve."""

    kind: str
    bbox: BBox
    filled: bool = False
    stroked: bool = True
    points: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class LayoutPage:
    page_no: int
    width: float
    height: float
    elements: list[Element]
    lines: list[TextLine] = field(default_factory=list)
    drawings: list[Drawing] = field(default_factory=list)


@dataclass
class LayoutDocument:
    source_path: str
    parser_version: str
    pages: list[LayoutPage]
    # Rendered crops, keyed by sha256 of the PNG bytes. Not serialized into
    # fixtures (binary); the adapter writes them to the crop store.
    crops: dict[str, bytes] = field(default_factory=dict)
    # Text-layer strings where Docling's decoding differed from pdfplumber's
    # and pdfplumber's was used (DEVIATIONS.md #221).
    text_repairs: int = 0

    def to_json_dict(self) -> dict[str, Any]:
        raw = asdict(self)
        raw.pop("crops", None)
        # asdict turns each BBox into a dict; store them as compact lists.
        return _bbox_dicts_to_lists(raw)

    @classmethod
    def from_json_dict(cls, d: dict[str, Any]) -> LayoutDocument:
        pages = []
        for p in d["pages"]:
            elements = [_element_from_dict(e) for e in p["elements"]]
            lines = [_line_from_dict(ln) for ln in p.get("lines", [])]
            drawings = [
                Drawing(
                    kind=dr["kind"],
                    bbox=BBox.from_list(dr["bbox"]),
                    filled=dr.get("filled", False),
                    stroked=dr.get("stroked", True),
                    points=[(pt[0], pt[1]) for pt in dr.get("points", [])],
                )
                for dr in p.get("drawings", [])
            ]
            pages.append(
                LayoutPage(
                    page_no=p["page_no"],
                    width=p["width"],
                    height=p["height"],
                    elements=elements,
                    lines=lines,
                    drawings=drawings,
                )
            )
        return cls(source_path=d["source_path"], parser_version=d["parser_version"], pages=pages)


def _bbox_dicts_to_lists(o: Any) -> Any:
    if isinstance(o, dict):
        if set(o) == {"x0", "top", "x1", "bottom"}:
            return [round(o["x0"], 2), round(o["top"], 2), round(o["x1"], 2), round(o["bottom"], 2)]
        return {k: _bbox_dicts_to_lists(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_bbox_dicts_to_lists(x) for x in o]
    return o


def _line_from_dict(ln: dict) -> TextLine:
    return TextLine(
        text=ln["text"],
        bbox=BBox.from_list(ln["bbox"]),
        from_ocr=ln.get("from_ocr", False),
        confidence=ln.get("confidence", 1.0),
        font_name=ln.get("font_name"),
        font_size=ln.get("font_size"),
        is_bold=ln.get("is_bold", False),
    )


def _element_from_dict(e: dict) -> Element:
    table = None
    if e.get("table"):
        t = e["table"]
        table = TableData(
            num_rows=t["num_rows"],
            num_cols=t["num_cols"],
            cells=[TableCell(**c) for c in t["cells"]],
        )
    return Element(
        kind=e["kind"],
        page_no=e["page_no"],
        bbox=BBox.from_list(e["bbox"]),
        text=e.get("text", ""),
        level=e.get("level"),
        origin=e.get("origin", ORIGIN_TEXT_LAYER),
        ocr_min_confidence=e.get("ocr_min_confidence"),
        font_size=e.get("font_size"),
        is_bold=e.get("is_bold", False),
        table=table,
        caption=e.get("caption"),
        image_sha256=e.get("image_sha256"),
        inner_lines=[_line_from_dict(ln) for ln in e.get("inner_lines", [])],
    )
