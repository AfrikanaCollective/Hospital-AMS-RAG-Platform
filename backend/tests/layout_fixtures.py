"""Synthetic `LayoutDocument` fixtures for the layout-ingestion tests (ARCH-044,
PRD-113). Offline, no parser, no real guideline text: the geometry mirrors
the Kenya MoH p. 47 flowchart (six boxes, five connectors with arrowheads,
Yes/No labels) and the p. 48 OCR dose table, but every string is invented.
NOT clinical content."""

from __future__ import annotations

from app.ingestion.layout.model import (
    BBox,
    Drawing,
    Element,
    LayoutDocument,
    LayoutPage,
    TableCell,
    TableData,
    TextLine,
)

W, H = 500.0, 700.0


def _line(
    text: str,
    x0: float,
    top: float,
    *,
    ocr: bool = False,
    conf: float = 1.0,
    size: float = 9.5,
    bold: bool = False,
) -> TextLine:
    return TextLine(
        text=text,
        bbox=BBox(x0, top, x0 + 6 * len(text), top + 8),
        from_ocr=ocr,
        confidence=conf,
        font_size=None if ocr else size,
        is_bold=bold,
    )


def _box_lines(x0: float, top: float, lines: list[str]) -> list[TextLine]:
    return [_line(t, x0 + 8, top + 8 + 11 * i) for i, t in enumerate(lines)]


# Box geometry (x0, top, x1, bottom) — same layout as Kenya p. 47.
BOXES = {
    "A": BBox(67, 97, 223, 206),
    "B": BBox(276, 97, 432, 206),
    "C": BBox(67, 245, 223, 316),
    "D": BBox(276, 239, 432, 348),
    "E": BBox(67, 383, 223, 408),
    "F": BBox(276, 373, 432, 431),
}
BOX_TEXT = {
    "A": ["Has ANY of the following", "• Sign alpha", "• Sign beta"],
    "B": ["Pathway one", "• Action X per protocol"],
    "C": [
        "Has ANY of the following",
        "• Sign gamma",
        "• Reading above or equal to",
        "30 units or below 20 units",
    ],
    "D": ["Pathway two", "• Action Y per protocol"],
    "E": ["Condition unlikely"],
    "F": ["Assess further"],
}


def _flowchart_drawings(*, drop_arrowhead_on: str | None = None) -> list[Drawing]:
    ds = [Drawing(kind="rect", bbox=b, filled=False, stroked=True) for b in BOXES.values()]
    # connectors: (start, end, arrowhead centre, name)
    conns = [
        ((223, 149), (275, 149), (273, 149), "A-B"),
        ((136, 206), (136, 244), (136, 242), "A-C"),
        ((223, 279), (276, 279), (274, 279), "C-D"),
        ((136, 316), (136, 382), (136, 380), "C-E"),
        ((223, 397), (275, 397), (273, 397), "E-F"),
    ]
    for (x0, y0), (x1, y1), (hx, hy), name in conns:
        ds.append(
            Drawing(
                kind="line",
                bbox=BBox(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)),
                points=[(x0, y0), (x1, y1)],
                filled=False,
            )
        )
        if name != drop_arrowhead_on:
            ds.append(Drawing(kind="curve", bbox=BBox(hx - 2, hy - 2, hx + 2, hy + 2), filled=True))
    return ds


FLOW_REGION = BBox(63, 95, 434, 432)


def flowchart_lines() -> list[TextLine]:
    lines: list[TextLine] = []
    for key, box in BOXES.items():
        lines += _box_lines(box.x0, box.top, BOX_TEXT[key])
    lines += [
        _line("Yes", 242, 139, size=8.7),
        _line("No", 143, 218, size=8.7),
        _line("Yes", 243, 269, size=8.7),
        _line("No", 143, 347, size=8.7),
        # OCR noise on an arrowhead — must be ignored
        TextLine(text="→", bbox=BBox(269, 394, 275, 400), from_ocr=True, confidence=0.99),
    ]
    return lines


def dose_table(*, ocr: bool = True) -> TableData:
    cells = [
        TableCell(0, 0, 1, 3, "Synthetic doses for group one", is_header=True),
        TableCell(1, 0, 1, 1, "Weight (kg)", is_header=True),
        TableCell(1, 1, 1, 1, "Agent P (10 u/kg)", is_header=True),
        TableCell(1, 2, 1, 1, "Agent Q (2 u/kg)", is_header=True),
        TableCell(2, 1, 1, 1, "12 hrly", is_header=True),
        TableCell(2, 2, 1, 1, "24 hrly", is_header=True),
        TableCell(3, 0, 1, 1, "1.0"),
        TableCell(3, 1, 1, 1, "10"),
        TableCell(3, 2, 1, 1, "2"),
        TableCell(4, 0, 1, 1, "2.0"),
        TableCell(4, 1, 1, 1, "20"),
        TableCell(4, 2, 1, 1, "4"),
    ]
    return TableData(num_rows=5, num_cols=3, cells=cells)


def synthetic_document(
    *, drop_arrowhead_on: str | None = None, with_mid_sentence_heading: bool = False
) -> LayoutDocument:
    p1_elements = [
        Element(
            kind="page_header",
            page_no=1,
            bbox=BBox(270, 27, 460, 37),
            text="Synthetic Care Protocols",
            font_size=10,
        ),
        Element(
            kind="heading",
            page_no=1,
            bbox=BBox(33, 57, 300, 68),
            text="2.1 Assessment of danger signs",
            level=1,
            font_size=14,
            is_bold=True,
        ),
        Element(
            kind="picture",
            page_no=1,
            bbox=FLOW_REGION,
            inner_lines=flowchart_lines(),
            image_sha256="a" * 64,
        ),
        Element(
            kind="heading",
            page_no=1,
            bbox=BBox(60, 500, 210, 510),
            text="Prophylaxis note",
            level=1,
            font_size=9.5,
            is_bold=True,
        ),
        Element(
            kind="text",
            page_no=1,
            bbox=BBox(60, 513, 425, 532),
            text="Give the synthetic agent at 5 mg/kg to every eligible case.",
            font_size=9.5,
        ),
        Element(
            kind="page_footer", page_no=1, bbox=BBox(470, 684, 483, 694), text="31", font_size=10
        ),
    ]
    if with_mid_sentence_heading:
        p1_elements[3:5] = [
            Element(
                kind="heading",
                page_no=1,
                bbox=BBox(60, 500, 400, 510),
                text="For people in labour, identify and assess any",
                level=1,
                font_size=9.5,
                is_bold=True,
            ),
            Element(
                kind="text",
                page_no=1,
                bbox=BBox(60, 513, 425, 532),
                text="risk factors for the synthetic infection. Monitor throughout.",
                font_size=9.5,
            ),
        ]
    p1 = LayoutPage(
        page_no=1,
        width=W,
        height=H,
        elements=p1_elements,
        lines=flowchart_lines(),
        drawings=_flowchart_drawings(drop_arrowhead_on=drop_arrowhead_on),
    )
    p2 = LayoutPage(
        page_no=2,
        width=W,
        height=H,
        elements=[
            Element(
                kind="page_header",
                page_no=2,
                bbox=BBox(270, 27, 460, 37),
                text="Synthetic Care Protocols",
                font_size=10,
            ),
            Element(
                kind="heading",
                page_no=2,
                bbox=BBox(38, 60, 215, 70),
                text="2.2 Doses",
                level=1,
                font_size=14,
                is_bold=True,
            ),
            Element(
                kind="heading",
                page_no=2,
                bbox=BBox(116, 78, 380, 94),
                text="Dose table for group one",
                level=1,
                origin="ocr",
                ocr_min_confidence=0.99,
            ),
            Element(
                kind="table",
                page_no=2,
                bbox=BBox(57, 98, 438, 266),
                table=dose_table(),
                origin="ocr",
                ocr_min_confidence=0.8,
                image_sha256="b" * 64,
            ),
            Element(
                kind="list_item",
                page_no=2,
                bbox=BBox(83, 274, 278, 284),
                text="Dose already adjusted for weight in days",
                origin="ocr",
                ocr_min_confidence=0.97,
            ),
            Element(
                kind="page_footer",
                page_no=2,
                bbox=BBox(38, 680, 228, 689),
                text="Page 2 of 9",
                font_size=10,
            ),
        ],
        lines=[_line("Dose table for group one", 116, 78, ocr=True, conf=0.99)],
        drawings=[],
    )
    return LayoutDocument(source_path="synthetic.pdf", parser_version="test-parser", pages=[p1, p2])
