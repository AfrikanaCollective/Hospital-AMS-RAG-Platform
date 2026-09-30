"""Row-labelled boxes (ARCH-044; DEVIATIONS.md #238), modelled on the Kenya
MoH National Antibiotic Use Guidelines pp. 34-37: a ruled box whose narrow,
shaded left column holds row labels. Synthetic geometry, no PDF."""

from __future__ import annotations

from app.ingestion.layout.headings import fuse_heading_levels
from app.ingestion.layout.label_boxes import apply_box_levels, restructure_label_boxes
from app.ingestion.layout.model import (
    BBox,
    Drawing,
    Element,
    LayoutDocument,
    LayoutPage,
    TextLine,
)

W, H = 420.0, 595.0
LEFT, MID, RIGHT = 50.0, 101.0, 378.0


def _v(x: float, top: float, bottom: float) -> Drawing:
    return Drawing(kind="rect", bbox=BBox(x - 0.25, top, x + 0.25, bottom), filled=True)


def _h(y: float, x0: float = LEFT, x1: float = RIGHT) -> Drawing:
    return Drawing(kind="rect", bbox=BBox(x0, y - 0.25, x1, y + 0.25), filled=True)


def _fill(x0: float, x1: float, top: float, bottom: float) -> Drawing:
    return Drawing(kind="rect", bbox=BBox(x0, top, x1, bottom), filled=True)


def _line(
    text: str,
    x0: float,
    top: float,
    *,
    bold: bool = False,
    size: float = 8.52,
    font: str | None = None,
    width: float | None = None,
) -> TextLine:
    w = width if width is not None else 4.2 * len(text)
    name = font or ("Cambria-Bold" if bold else "Cambria")
    return TextLine(text, BBox(x0, top, x0 + w, top + 8.2), False, 1.0, name, size, bold)


def _frame(top: float, bottom: float, rows: list[float], splits: dict[float, tuple[float, float]]):
    """Rules for a box: outer and label-column verticals, one horizontal per
    row edge, and extra verticals (x -> (top, bottom)) that split content."""
    d = [_v(LEFT, top, bottom), _v(MID, top, bottom), _v(RIGHT, top, bottom)]
    d += [_h(y) for y in rows]
    d += [_v(x, t, b) for x, (t, b) in splits.items()]
    return d


def _section(text: str, top: float, page: int) -> Element:
    return Element(
        "heading", page, BBox(50, top, 300, top + 11), text=text, font_size=11.04, is_bold=True
    )


def _neonatal_doc() -> LayoutDocument:
    p1_lines = [
        _line("Neonatal Sepsis", 105, 206, bold=True),
        _line("Common Pathogens", 54, 230, bold=True, width=40),
        _line("Early onset sepsis", 105, 230, bold=True),
        _line("Group B streptococcus, Escherichia coli", 105, 242),
        _line("Late onset sepsis", 105, 265, bold=True),
        _line("CoNS, Staphylococcus aureus, Candida", 105, 276),
        _line("Empiric Therapy", 54, 307, bold=True, width=40),
        _line("First line:", 128, 307, bold=True),
        _line("Benzylpenicillin 50,000", 128, 319, bold=True),
        _line("ACCESS", 106, 329.6, bold=True, size=3.94, width=12),
        # set in a bold font, but the character-level flag came out False
        _line("IU/kg IV 6 hourly", 128, 330.8, bold=False, font="Cambria-Bold"),
        _line("Second line:", 262, 307, bold=True),
        _line("Cefepime 50mg/kg 8 hourly", 262, 319, bold=True),
    ]
    p1_draw = _frame(200, 541, [200, 225, 302, 541], {229.0: (302, 541)})
    p1_draw += [
        _fill(50.7, 101.2, 200.6, 224.6),
        _fill(50.7, 101.2, 225.2, 301.9),
        _fill(50.7, 101.2, 302.5, 540.7),
    ]
    p1_els = [
        _section("5. NEONATAL SEPSIS IN INFANTS < 60 DAYS", 40, 1),
        Element(
            "text",
            1,
            BBox(72, 56, 367, 76),
            text="Necrotising enterocolitis: abdominal distension.",
            font_size=8.52,
        ),
        # Docling's own reading of the box: replaced by the rebuilt sequence
        Element(
            "heading",
            1,
            BBox(54, 307, 87, 327),
            text="Empiric Therapy",
            font_size=8.52,
            is_bold=True,
        ),
        Element(
            "text",
            1,
            BBox(128, 319, 220, 339),
            text="Benzylpenicillin 50,000 IU/kg IV 6 hourly",
            font_size=8.52,
        ),
    ]
    p2_lines = [
        _line("Metronidazole 7.5", 128, 56, bold=True),
        _line("mg/kg: < 1month 12 hourly", 128, 67.8),
        # one text run crossing the label column's edge
        _line("Comments Duration of therapy:", 54, 156.5, bold=True, width=131),
        _line("Neonate at risk of sepsis: stop IV antibiotics", 105, 172),
    ]
    p2_draw = _frame(50, 497, [50, 151, 497], {229.0: (50, 151)})
    p2_draw += [_fill(50.7, 101.2, 50.9, 150.5), _fill(50.7, 101.2, 151.1, 497.1)]
    pages = [
        LayoutPage(1, W, H, p1_els, lines=p1_lines, drawings=p1_draw),
        LayoutPage(2, W, H, [], lines=p2_lines, drawings=p2_draw),
    ]
    return LayoutDocument("moh.pdf", "t", pages)


def _run(doc: LayoutDocument) -> dict:
    report = restructure_label_boxes(doc)
    fuse_heading_levels(doc)
    apply_box_levels(doc)
    return report


def _seq(page: LayoutPage) -> list[tuple]:
    return [
        (el.kind, el.text, el.level, el.box_continued)
        for el in page.elements
        if el.kind in ("heading", "text")
    ]


def test_row_labels_become_sub_topics_and_columns_keep_their_sub_labels() -> None:
    doc = _neonatal_doc()
    report = _run(doc)
    assert report["boxes"] == 2 and report["joined_cells"] == 1
    assert _seq(doc.pages[0]) == [
        ("heading", "5. NEONATAL SEPSIS IN INFANTS < 60 DAYS", 1, False),
        ("text", "Necrotising enterocolitis: abdominal distension.", None, False),
        # the "Neonatal Sepsis" title band repeats the section: dropped
        ("heading", "Common Pathogens", 2, False),
        ("heading", "Early onset sepsis", 3, False),
        ("text", "Group B streptococcus, Escherichia coli", None, False),
        ("heading", "Late onset sepsis", 3, False),
        ("text", "CoNS, Staphylococcus aureus, Candida", None, False),
        ("heading", "Empiric Therapy", 2, False),
        ("heading", "First line:", 3, False),
        # the dose line stays whole; the AWaRe badge goes to its end
        ("text", "Benzylpenicillin 50,000 IU/kg IV 6 hourly ACCESS", None, False),
        # p.2's continuation of the same cell, joined back (DEVIATIONS.md #239)
        ("text", "Metronidazole 7.5 mg/kg: < 1month 12 hourly", None, True),
        ("heading", "Second line:", 3, False),
        ("text", "Cefepime 50mg/kg 8 hourly", None, False),
    ]


def test_continued_cell_is_joined_to_its_cell_and_keeps_its_own_page() -> None:
    """MoH p.35: metronidazole continues the First line cell from p.34; it is
    one section with it (not under "Second line:", DEVIATIONS.md #237) and
    not a repeated heading on p.35 (#239)."""
    doc = _neonatal_doc()
    _run(doc)
    joined = next(el for el in doc.pages[0].elements if el.text.startswith("Metronidazole"))
    assert joined.page_no == 2  # its text is still cited on its own page
    assert _seq(doc.pages[1]) == [
        ("heading", "Comments", 2, False),  # split off the merged text run
        ("heading", "Duration of therapy:", 3, False),
        ("text", "Neonate at risk of sepsis: stop IV antibiotics", None, False),
    ]


def test_header_row_labels_the_columns_below_it() -> None:
    """MoH BSI p.36: three column headers, then a labelled row."""
    lines = [
        _line("Community Acquired BSI", 105, 385, bold=True),
        _line("Hospital Acquired BSI", 190, 385, bold=True),
        _line("Common Pathogens", 54, 420, bold=True, width=40),
        _line("Staphylococcus aureus, Escherichia coli", 105, 420, width=78),
        _line("Enterobacterales, Klebsiella", 190, 420, width=78),
    ]
    draw = _frame(378, 495, [378, 414, 495], {187.0: (378, 495)})
    draw += [
        _fill(50.7, 101.2, 378.8, 413.6),
        _fill(50.7, 101.2, 414.2, 494.9),
        _fill(101.7, 186.5, 378.8, 413.6),
        _fill(187, 378, 378.8, 413.6),
    ]
    doc = LayoutDocument(
        "moh.pdf",
        "t",
        [
            LayoutPage(
                1,
                W,
                H,
                [_section("6. BACTERIAL BLOOD STREAM INFECTIONS (BSI)", 52, 1)],
                lines=lines,
                drawings=draw,
            )
        ],
    )
    _run(doc)
    assert _seq(doc.pages[0])[1:] == [
        ("heading", "Common Pathogens", 2, False),
        ("heading", "Community Acquired BSI", 3, False),
        ("text", "Staphylococcus aureus, Escherichia coli", None, False),
        ("heading", "Hospital Acquired BSI", 3, False),
        ("text", "Enterobacterales, Klebsiella", None, False),
    ]


def _grid_doc(*, shade_content: bool, values: tuple[str, str]) -> LayoutDocument:
    lines = [
        _line("Low", 54, 440, bold=False, width=20),
        _line(values[0], 130, 440, width=40),
        _line("Medium", 54, 470, bold=False, width=30),
        _line(values[1], 130, 470, width=40),
    ]
    draw = _frame(430, 490, [430, 460, 490], {})
    draw += [_fill(50.7, 101.2, 430.5, 459.5), _fill(50.7, 101.2, 460.5, 489.5)]
    if shade_content:
        draw += [_fill(101.5, 377.5, 430.5, 459.5), _fill(101.5, 377.5, 460.5, 489.5)]
    return LayoutDocument("x.pdf", "t", [LayoutPage(1, W, H, [], lines=lines, drawings=draw)])


def test_lookup_grid_of_short_values_is_not_a_label_box() -> None:
    """LRINEC risk categories: "≤5", "<50%" are values, not sub-topics."""
    doc = _grid_doc(shade_content=False, values=("≤5 <50%", "6–7 50–75%"))
    assert restructure_label_boxes(doc)["boxes"] == 0


def test_grid_shaded_throughout_is_not_a_label_box() -> None:
    doc = _grid_doc(
        shade_content=True,
        values=("Give amoxicillin by mouth for five days", "Admit and give IV antibiotics now"),
    )
    assert restructure_label_boxes(doc)["boxes"] == 0


def test_page_without_boxes_is_untouched() -> None:
    els = [
        _section("INTRODUCTION", 56, 1),
        Element("text", 1, BBox(54, 78, 360, 90), text="Body.", font_size=9.0),
    ]
    doc = LayoutDocument("x.pdf", "t", [LayoutPage(1, W, H, list(els))])
    assert restructure_label_boxes(doc)["boxes"] == 0
    assert doc.pages[0].elements == els


# --- skin-table cases (MoH pp. 47-51) ---

SKIN_MID, SKIN_SPLIT = 122.0, 250.0


def _skin_page(
    page_no: int,
    rows: list[float],
    lines: list[TextLine],
    *,
    extra: list[Drawing] | None = None,
    label_rows: list[tuple[float, float]],
) -> LayoutPage:
    d = [
        _v(LEFT, rows[0], rows[-1]),
        _v(SKIN_MID, rows[0], rows[-1]),
        _v(RIGHT, rows[0], rows[-1]),
        _v(SKIN_SPLIT, rows[0], rows[-1]),
    ]
    d += [_h(y) for y in rows]
    d += [_fill(50.7, SKIN_MID - 0.3, t + 0.5, b - 0.5) for t, b in label_rows]
    d += extra or []
    return LayoutPage(page_no, W, H, [], lines=lines, drawings=d)


def _skin_header_lines(top: float) -> list[TextLine]:
    return [
        _line("Condition", 54, top, bold=True, width=40),
        _line("Description", 126, top, bold=True, width=50),
        _line("Empiric Therapy", 254, top, bold=True, width=60),
    ]


def test_sub_rows_keep_each_description_with_its_own_treatment() -> None:
    """ "Traumatic wounds": a rule across the content columns only splits it
    into "without infection" and "with systemic features"."""
    lines = [
        *_skin_header_lines(120),
        _line("Traumatic wounds", 54, 150, bold=True, width=60),
        _line("Without signs of infection", 126, 150, width=110),
        _line("Do not need antimicrobial therapy", 254, 150, width=110),
        _line("With systemic features", 126, 190, width=110),
        _line("Amoxicillin+ Clavulanic acid 1.2g IV", 254, 190, width=110, bold=True),
    ]
    sub_rule = [_h(180, SKIN_MID, SKIN_SPLIT), _h(180, SKIN_SPLIT, RIGHT)]  # per-cell segments
    page = _skin_page(
        1, [115, 140, 220], lines, extra=sub_rule, label_rows=[(115, 140), (140, 220)]
    )
    doc = LayoutDocument("moh.pdf", "t", [page])
    _run(doc)
    texts = [el.text for el in doc.pages[0].elements if el.kind == "text"]
    assert texts == [
        "Description: Without signs of infection",
        "Empiric Therapy: Do not need antimicrobial therapy",
        "Description: With systemic features",
        "Empiric Therapy: Amoxicillin+ Clavulanic acid 1.2g IV",
    ]


def test_underline_under_a_sub_label_is_not_a_sub_row_rule() -> None:
    """The ~40 pt underline under "First line:" must not split the row."""
    doc = _neonatal_doc()
    doc.pages[0].drawings += [_h(315.5, 128, 166), _h(315.5, 263, 310)]
    _run(doc)
    seq = [(el.kind, el.text) for el in doc.pages[0].elements if el.kind in ("heading", "text")]
    assert seq[-6:] == [
        ("heading", "Empiric Therapy"),
        ("heading", "First line:"),
        ("text", "Benzylpenicillin 50,000 IU/kg IV 6 hourly ACCESS"),
        ("text", "Metronidazole 7.5 mg/kg: < 1month 12 hourly"),
        ("heading", "Second line:"),
        ("text", "Cefepime 50mg/kg 8 hourly"),
    ]


def test_unlabelled_row_after_a_labelled_one_is_its_sub_row() -> None:
    """ "Surgical site infections": the label cell spans only the first part."""
    lines = [
        *_skin_header_lines(120),
        _line("Surgical site infections", 54, 150, bold=True, width=60),
        _line("Subcutaneous, no systemic response", 126, 150, width=110),
        _line("Not routinely recommended", 254, 150, width=110),
        _line("Deep tissue involvement", 126, 190, width=110),
        _line("Suture removal plus drainage", 254, 190, width=110),
    ]
    page = _skin_page(
        1, [115, 140, 180, 220], lines, label_rows=[(115, 140), (140, 180), (180, 220)]
    )
    doc = LayoutDocument("moh.pdf", "t", [page])
    _run(doc)
    els = [(el.kind, el.text) for el in doc.pages[0].elements]
    assert els == [
        ("heading", "Surgical site infections"),
        ("text", "Description: Subcutaneous, no systemic response"),
        ("text", "Empiric Therapy: Not routinely recommended"),
        ("text", "Description: Deep tissue involvement"),
        ("text", "Empiric Therapy: Suture removal plus drainage"),
    ]


def test_lowercase_spill_goes_back_to_the_previous_row_and_merged_cell_has_no_header() -> None:
    """Diabetic foot's sentence ends on p.49 at the top of the next row's
    cell; "Wounds" is one statement across both columns."""
    p1 = _skin_page(
        1,
        [115, 140, 300],
        [
            *_skin_header_lines(120),
            _line("Diabetic foot infections", 54, 150, bold=True, width=60),
            _line("Most do not require antibiotics", 126, 150, width=110),
            _line("Surgical debridement is an important", 254, 150, width=110),
        ],
        label_rows=[(115, 140), (140, 300)],
    )
    p2 = _skin_page(
        2,
        [51, 120, 160],
        [
            _line("Decubitus wound infection", 54, 56, bold=True, width=60),
            _line("Start antibiotics only if inflamed", 126, 56, width=110),
            _line("component in management", 254, 56, width=110),
            _line("Amoxicillin+ Clavulanic acid 1g PO", 254, 70, width=110, bold=True),
            _line("Wounds", 54, 125, bold=True, width=40),
            _line("Usually polymicrobial from environmental contamination.", 126, 125, width=240),
        ],
        label_rows=[(51, 120), (120, 160)],
    )
    p2.drawings = [
        d
        for d in p2.drawings
        if not (abs(d.bbox.x0 - SKIN_SPLIT) < 1 and d.bbox.top < 121 and d.bbox.bottom > 159)
    ]
    p2.drawings += [_v(SKIN_SPLIT, 51, 120)]  # the split runs only through the first row
    doc = LayoutDocument("moh.pdf", "t", [p1, p2])
    _run(doc)
    p1_texts = [(el.text, el.page_no) for el in doc.pages[0].elements if el.kind == "text"]
    assert p1_texts[-2:] == [
        ("Surgical debridement is an important", 1),
        ("component in management", 2),  # joined to its cell, cited on its own page
    ]
    seq = [(el.kind, el.text, el.box_continued) for el in doc.pages[1].elements]
    assert seq[0] == ("heading", "Decubitus wound infection", False)
    assert all(t != "component in management" for _k, t, _c in seq)
    assert ("heading", "Wounds", False) in seq
    wounds_at = seq.index(("heading", "Wounds", False))
    assert seq[wounds_at + 1] == (
        "text",
        "Usually polymicrobial from environmental contamination.",
        False,
    )


def test_joined_cell_is_one_chunk_spanning_both_pages() -> None:
    """Assembly records one page entry per *run*, so text going p.1 → p.2 →
    p.1 keeps every offset's page, and the "First line:" chunk spans pp.1-2
    (DEVIATIONS.md #239). Excerpt page remapping works on the runs too."""
    from app.ingestion.chunking import chunk_document
    from app.ingestion.layout.assemble import AssemblyOptions, assemble
    from app.ingestion.page_provenance import apply_source_pages

    parsed = assemble(_neonatal_doc(), AssemblyOptions())
    pages = [p for p, _ in parsed.page_starts]
    assert pages[:3] == [1, 2, 1]  # the joined cell makes a second run of page 1
    by_text = {parsed.normalized_text[b.char_start : b.char_end]: b for b in parsed.blocks or []}
    metro = by_text["Metronidazole 7.5 mg/kg: < 1month 12 hourly"]
    second = by_text["Second line:"]
    assert parsed.page_for_offset(metro.char_start) == 2
    assert parsed.page_for_offset(second.char_start) == 1

    first_line = [
        c
        for c in chunk_document(parsed, format_profile="clinical_protocol")
        if (c.get("section_path") or "").endswith("Empiric Therapy › First line:")
    ]
    assert len(first_line) == 1
    assert "Benzylpenicillin" in first_line[0]["text"]
    assert "Metronidazole" in first_line[0]["text"]
    assert (first_line[0]["page_start"], first_line[0]["page_end"]) == (1, 2)

    apply_source_pages(parsed, [34, 35])
    assert parsed.page_for_offset(metro.char_start) == 35
    assert parsed.page_for_offset(second.char_start) == 34
