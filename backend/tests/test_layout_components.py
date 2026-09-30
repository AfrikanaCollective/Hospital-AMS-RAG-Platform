"""Layout ingestion building blocks: header/footer removal, heading fusion,
table serialization, text corrections, chunk lineage (ARCH-044, PRD-113;
LAYOUT-INGESTION-PROPOSAL.md §5.3-§5.5, §5.11, §10). Synthetic inputs only."""

from __future__ import annotations

import pytest

from app.ingestion.corrections import (
    Correction,
    CorrectionError,
    apply_to_units,
    check_scope,
    load_corrections,
)
from app.ingestion.layout.boilerplate import (
    drop_leading_excerpt_fragment,
    normalize,
    remove_boilerplate,
)
from app.ingestion.layout.headings import fuse_heading_levels
from app.ingestion.layout.model import BBox, Drawing, Element, LayoutDocument, LayoutPage
from app.ingestion.layout.tables import render_table
from app.ingestion.lineage import map_chunks, remap_gold
from tests.layout_fixtures import H, W, dose_table, synthetic_document

# ── boilerplate (§5.3) ──


def _page(n: int, elements: list[Element]) -> LayoutPage:
    return LayoutPage(page_no=n, width=W, height=H, elements=elements)


def _el(kind: str, text: str, top: float, n: int = 1) -> Element:
    return Element(kind=kind, page_no=n, bbox=BBox(40, top, 300, top + 9), text=text)


def test_layout_labels_drop_headers_and_footers() -> None:
    doc = synthetic_document()
    report = remove_boilerplate(doc, margin_zone=0.08, repeat_ratio=0.5, min_pages=3)
    kinds = {el.kind for p in doc.pages for el in p.elements}
    assert "page_header" not in kinds and "page_footer" not in kinds
    assert report.dropped_count == 4


def test_repeated_margin_text_dropped_with_digits_normalised() -> None:
    pages = [
        _page(n, [_el("text", f"Page {n} of 40", 690, n), _el("text", "Body text here.", 300, n)])
        for n in range(1, 5)
    ]
    doc = LayoutDocument("x.pdf", "t", pages)
    remove_boilerplate(doc, margin_zone=0.08, repeat_ratio=0.5, min_pages=3)
    assert all([el.text for el in p.elements] == ["Body text here."] for p in doc.pages)
    assert normalize("Page 3 of 40") == normalize("Page 4 of 40")


def test_repeat_rule_needs_min_pages_so_short_excerpts_keep_margin_text() -> None:
    pages = [_page(n, [_el("text", "Margin note in a two-page excerpt", 690, n)]) for n in (1, 2)]
    doc = LayoutDocument("x.pdf", "t", pages)
    remove_boilerplate(doc, margin_zone=0.08, repeat_ratio=0.5, min_pages=3)
    assert all(p.elements for p in doc.pages)


def test_dose_text_and_headings_in_the_margin_are_never_dropped() -> None:
    """A heading in the margin is kept unless it is a running header (#236):
    here each page's margin heading is different."""
    pages = [
        _page(
            n,
            [
                _el("text", "Give 5 mg/kg daily", 690, n),
                _el(
                    "heading", ["Assessment", "Treatment", "Monitoring", "Discharge"][n - 1], 20, n
                ),
            ],
        )
        for n in range(1, 5)
    ]
    doc = LayoutDocument("x.pdf", "t", pages)
    remove_boilerplate(doc, margin_zone=0.08, repeat_ratio=0.5, min_pages=3)
    assert all(len(p.elements) == 2 for p in doc.pages)


def test_running_header_typed_as_heading_is_dropped() -> None:
    """MoH: "EMPIRIC ANTIBIOTIC USE" at the same spot atop every page of the
    part, typed as a heading (DEVIATIONS.md #236)."""
    pages = [
        _page(n, [_el("heading", "EMPIRIC ANTIBIOTIC USE", 20, n), _el("text", "Body.", 300, n)])
        for n in range(1, 4)
    ]
    doc = LayoutDocument("x.pdf", "t", pages)
    report = remove_boilerplate(doc, margin_zone=0.08, repeat_ratio=0.9, min_pages=3)
    assert all([el.text for el in p.elements] == ["Body."] for p in doc.pages)
    assert report.dropped_count == 3


def test_repeated_heading_below_the_margin_or_on_too_few_pages_is_kept() -> None:
    in_body = [
        _page(n, [_el("heading", "Why the committee made the recommendations", 300, n)])
        for n in range(1, 5)
    ]
    two_pages = [_page(n, [_el("heading", "EMPIRIC ANTIBIOTIC USE", 20, n)]) for n in (1, 2)]
    shifted = [_page(n, [_el("heading", "Recurring title", 10 + 15 * n, n)]) for n in range(1, 4)]
    for pages in (in_body, two_pages, shifted):
        doc = LayoutDocument("x.pdf", "t", pages)
        remove_boilerplate(doc, margin_zone=0.08, repeat_ratio=0.5, min_pages=3)
        assert all(p.elements for p in doc.pages)


def test_manifest_patterns_drop_matching_margin_text() -> None:
    doc = LayoutDocument("x.pdf", "t", [_page(1, [_el("text", "TABLE OF CONTENT", 690)])])
    remove_boilerplate(
        doc, margin_zone=0.08, repeat_ratio=0.5, min_pages=3, manifest_patterns=["table of content"]
    )
    assert doc.pages[0].elements == []


# ── headings (§5.4) ──


def test_heading_levels_fused_from_numbering_and_structure() -> None:
    doc = synthetic_document()
    fuse_heading_levels(doc)
    levels = {el.text: el.level for p in doc.pages for el in p.elements if el.kind == "heading"}
    assert levels["2.1 Assessment of danger signs"] == 2
    assert levels["Prophylaxis note"] == 3  # unnumbered -> child of the numbered heading
    assert levels["Dose table for group one"] == 3


def test_mid_sentence_heading_is_merged_back_into_its_sentence() -> None:
    doc = synthetic_document(with_mid_sentence_heading=True)
    fuse_heading_levels(doc)
    texts = [el.text for el in doc.pages[0].elements if el.kind in ("heading", "text")]
    assert "For people in labour, identify and assess any" not in texts
    assert any(
        t.startswith("For people in labour, identify and assess any risk factors") for t in texts
    )


def test_small_non_bold_heading_is_demoted_to_text() -> None:
    body = Element("text", 1, BBox(40, 100, 300, 109), text="Body.", font_size=9.5)
    small = Element("heading", 1, BBox(40, 120, 300, 128), text="Tiny label", font_size=7.0)
    doc = LayoutDocument("x.pdf", "t", [_page(1, [body, small])])
    fuse_heading_levels(doc)
    assert small.kind == "text"


def _heading(text: str, size: float | None, top: float) -> Element:
    return Element(
        "heading",
        1,
        BBox(40, top, 400, top + size if size else top + 12),
        text=text,
        font_size=size,
        is_bold=True,
    )


def test_font_tiers_set_levels_when_numbering_would_invert_them() -> None:
    """NICE NG195's pattern (DEVIATIONS.md #230): unnumbered 25.5 pt chapters
    contain numbered 21 pt sections; research recommendations numbered "1" …
    "15" are 16.5 pt subsections; a 12 pt semibold box label sits below."""
    body = Element("text", 1, BBox(40, 10, 400, 22), text="Body.", font_size=12.0)
    specs = [
        ("Neonatal infection: antibiotics for prevention and treatment", 33.0),  # cover, once
        ("Information and support for parents and carers", 25.5),
        ("1.1 Babies at increased risk of neonatal infection", 21.0),
        ("Risk factors and clinical indicators: early-onset neonatal infection", 25.5),
        ("1.10 Assessment and risk management after birth", 21.0),
        ("Other clinical indicators:", 12.0),
        ("Recommendations for research", 25.5),
        ("Key recommendations for research", 21.0),
        ("1 Switching from intravenous to oral antibiotics", 16.5),
        ("15 Long-term outcomes of bacterial meningitis", 16.5),
        ("Rationale and impact", 25.5),
        ("Why the committee made the recommendations", 16.5),
        ("Scanned heading", None),  # OCR: no font size
    ]
    heads = [_heading(t, sz, 40 + 30 * i) for i, (t, sz) in enumerate(specs)]
    doc = LayoutDocument("x.pdf", "t", [_page(1, [body, *heads])])
    fuse_heading_levels(doc)
    level = {el.text: el.level for el in heads}
    assert level["Neonatal infection: antibiotics for prevention and treatment"] == 1
    for chapter in (
        "Information and support for parents and carers",
        "Risk factors and clinical indicators: early-onset neonatal infection",
        "Recommendations for research",
        "Rationale and impact",
    ):
        assert level[chapter] == 1, chapter
    assert level["1.10 Assessment and risk management after birth"] == 2
    assert level["Key recommendations for research"] == 2
    assert level["15 Long-term outcomes of bacterial meningitis"] == 3  # not 1
    assert level["Why the committee made the recommendations"] == 3
    assert level["Other clinical indicators:"] == 4
    assert level["Scanned heading"] is not None  # keeps its numbering/structure level


def test_one_heading_size_keeps_numbering_and_structure_levels() -> None:
    heads = [
        _heading("2 Management", 12.0, 40),
        _heading("2.1 Assessment of danger signs", 12.0, 70),
        _heading("Prophylaxis note", 12.0, 100),
    ]
    doc = LayoutDocument("x.pdf", "t", [_page(1, heads)])
    fuse_heading_levels(doc)
    assert [el.level for el in heads] == [1, 2, 3]


# --- typographic tables: WHO SBI Tables 1.1 / 3.1 (DEVIATIONS.md #234) ---


def _who_el(kind: str, text: str, top: float, size: float, bold: bool) -> Element:
    return Element(kind, 1, BBox(71, top, 480, top + 8.2), text=text, font_size=size, is_bold=bold)


def _band(top: float) -> Drawing:
    return Drawing(kind="rect", bbox=BBox(65, top - 6.5, 532, top + 16.2), filled=True)


def _who_page() -> LayoutPage:
    """Table 3.1's pattern: shaded one-line bands, Docling typing two of them
    as list items; a shaded box holding several bullets; body text."""
    els = [
        _who_el(
            "caption",
            "Table 3.1 Antibiotic dosing for serious bacterial infections",
            90,
            11.0,
            True,
        ),
        _who_el("heading", "A. Non-hospital settings", 116, 9.5, True),
        _who_el("heading", "A.3 Clinical severe infection in young infants", 139, 9.5, False),
        _who_el("text", "Amoxicillin oral for a total of at least 7 days", 160, 9.5, False),
        _who_el("list_item", "A.4 Fast breathing as the only clinical sign", 210, 9.5, False),
        _who_el("text", "Amoxicillin oral for at least 7 days", 232, 9.5, False),
        _who_el("heading", "B. Hospital settings", 262, 9.5, True),
        _who_el("list_item", "B.4 Suspected meningitis in young infants", 285, 9.5, False),
        _who_el("text", "Ampicillin IM/IV for a total of at least 3 weeks", 306, 9.5, False),
        _who_el("heading", "Table 3.1 continued", 330, 9.5, False),
        _who_el("list_item", "Box: first shaded bullet", 400, 9.5, False),
        _who_el("list_item", "Box: second shaded bullet", 412, 9.5, False),
        _who_el("text", "A one-line shaded note.", 470, 9.5, False),
        _who_el("text", "Body text below.", 520, 9.5, False),
    ]
    drawings = [_band(t) for t in (116, 139, 210, 262, 285, 470)]
    drawings.append(Drawing(kind="rect", bbox=BBox(65, 395, 532, 425), filled=True))  # a box
    return LayoutPage(page_no=1, width=595.0, height=842.0, elements=els, drawings=drawings)


def _levels(doc: LayoutDocument) -> dict[str, int | None]:
    return {el.text: el.level for p in doc.pages for el in p.elements if el.kind == "heading"}


def test_band_rows_become_headings_but_boxes_and_sentences_do_not() -> None:
    doc = LayoutDocument("who.pdf", "t", [_who_page()])
    fuse_heading_levels(doc)
    kinds = {el.text: el.kind for el in doc.pages[0].elements}
    assert kinds["A.4 Fast breathing as the only clinical sign"] == "heading"
    assert kinds["B.4 Suspected meningitis in young infants"] == "heading"
    assert kinds["Box: first shaded bullet"] == "list_item"  # several items: a box
    assert kinds["A one-line shaded note."] == "text"  # ends in a full stop


def test_table_title_caption_is_promoted_and_continued_line_dropped() -> None:
    doc = LayoutDocument("who.pdf", "t", [_who_page()])
    fuse_heading_levels(doc)
    texts = [el.text for el in doc.pages[0].elements]
    assert "Table 3.1 continued" not in texts
    title = next(el for el in doc.pages[0].elements if el.text.startswith("Table 3.1 Antibiotic"))
    assert title.kind == "heading"


def test_letter_parts_parent_their_items() -> None:
    doc = LayoutDocument("who.pdf", "t", [_who_page()])
    fuse_heading_levels(doc)
    level = _levels(doc)
    title = level["Table 3.1 Antibiotic dosing for serious bacterial infections"]
    part_a = level["A. Non-hospital settings"]
    assert title is not None and part_a is not None and title < part_a
    assert level["A.3 Clinical severe infection in young infants"] == part_a + 1
    assert level["A.4 Fast breathing as the only clinical sign"] == part_a + 1
    assert level["B. Hospital settings"] == part_a  # the next part closes A
    assert level["B.4 Suspected meningitis in young infants"] == part_a + 1


def test_table_heading_inside_a_part_does_not_end_the_part() -> None:
    """MoH surgical prophylaxis: "B. GASTROINTESTINAL PROCEDURES" (part) then
    "Table 5: …" and "Post-Operative Care" set one size smaller."""
    heads = [
        _heading("B. GASTROINTESTINAL PROCEDURES", 11.0, 40),
        _heading("Table 5: Endoscopic Gastrointestinal Procedure", 10.0, 70),
        _heading("Post-Operative Care", 10.0, 100),
        _heading("C. Neurosurgery", 11.0, 130),
        _heading("Craniotomy", 10.0, 160),
    ]
    doc = LayoutDocument("x.pdf", "t", [_page(1, heads)])
    fuse_heading_levels(doc)
    assert [el.level for el in heads] == [1, 2, 2, 1, 2]


def test_caption_not_followed_by_a_heading_stays_a_caption() -> None:
    cap = _who_el("caption", "Table 2 Doses by weight", 100, 11.0, True)
    body = _who_el("text", "Weight 2 kg: 100 mg.", 120, 9.5, False)
    doc = LayoutDocument("x.pdf", "t", [LayoutPage(1, 595.0, 842.0, [cap, body])])
    fuse_heading_levels(doc)
    assert cap.kind == "caption"


# --- excerpt boundary fragment (DEVIATIONS.md #235) ---


def _excerpt(first_texts: list[tuple[str, str]]) -> LayoutDocument:
    els = [_el(kind, text, 100 + 12 * i) for i, (kind, text) in enumerate(first_texts)]
    els.append(_el("heading", "Table 1.1 Clinical case definitions", 300))
    els.append(_el("text", "WHO defines PSBI as …", 320))
    page2 = _page(2, [_el("text", "starts lower case but is not page one", 100)])
    return LayoutDocument("who.pdf", "t", [_page(1, els), page2])


def test_mid_sentence_opening_of_an_excerpt_is_dropped() -> None:
    doc = _excerpt(
        [
            ("text", "fives) in resource-limited settings (26). Recognizing …"),
            ("text", "Recognizing the limited resources in community-level care …"),
        ]
    )
    dropped = drop_leading_excerpt_fragment(doc)
    assert len(dropped) == 2 and dropped[0].startswith("fives)")
    assert doc.pages[0].elements[0].text.startswith("Table 1.1")
    assert doc.pages[1].elements[0].text.startswith("starts lower case")  # page 2 untouched


@pytest.mark.parametrize(
    "lead",
    [
        [("text", "Background. Every year about 3 million infants die.")],  # a real start
        [("text", "and gentamicin 5 mg/kg once daily for 7 days")],  # a dose: never dropped
        [("table", "cells"), ("text", "fives) in settings")],  # a table comes first
    ],
)
def test_excerpt_opening_is_kept_when_it_is_not_a_plain_fragment(lead) -> None:
    doc = _excerpt(lead)
    assert drop_leading_excerpt_fragment(doc) == []
    assert len(doc.pages[0].elements) == len(lead) + 2


def test_first_page_without_a_heading_is_never_dropped() -> None:
    doc = LayoutDocument("x.pdf", "t", [_page(1, [_el("text", "continued from before", 100)])])
    assert drop_leading_excerpt_fragment(doc) == []
    assert len(doc.pages[0].elements) == 1


# ── tables (§5.5) ──


def test_multirow_headers_flatten_to_column_paths_and_title_line() -> None:
    render = render_table(dose_table(), max_tokens=700)
    assert render.title_lines == ["Synthetic doses for group one"]
    assert render.header_paths == [
        "Weight (kg)",
        "Agent P (10 u/kg) · 12 hrly",
        "Agent Q (2 u/kg) · 24 hrly",
    ]
    assert "| 1.0 | 10 | 2 |" in render.grid_markdown  # grid kept for display


def test_citable_table_text_is_row_wise_with_header_path_labels() -> None:
    """DEVIATIONS.md #220: each value line names its own column."""
    render = render_table(dose_table(), max_tokens=700)
    assert render.text == (
        "Synthetic doses for group one\n\n"
        "Weight (kg): 1.0\n"
        "  Agent P (10 u/kg) · 12 hrly: 10\n"
        "  Agent Q (2 u/kg) · 24 hrly: 2\n\n"
        "Weight (kg): 2.0\n"
        "  Agent P (10 u/kg) · 12 hrly: 20\n"
        "  Agent Q (2 u/kg) · 24 hrly: 4"
    )


def test_large_table_splits_by_row_group_repeating_the_title() -> None:
    render = render_table(dose_table(), max_tokens=25)
    assert len(render.parts) == 2
    for part in render.parts:
        assert part.startswith("Synthetic doses for group one")
        assert "Weight (kg):" in part


# ── attested corrections (§5.11) ──

_ORIG = "Reading above or equal to 30 units or below 20 units"
_FIXED = "Reading below 20 units or above or equal to 30 units"


def _entry(**kw) -> dict:
    return {
        "id": "c1",
        "page": 47,
        "original": _ORIG,
        "corrected": _FIXED,
        "kind": "erratum",
        "rationale": "inequalities transposed",
        "evidence": "section 3 states it correctly",
        "attested_by": "Dr Example (paediatrics)",
        "attested_on": "2026-09-28",
    } | kw


def test_reordering_the_same_tokens_is_in_scope() -> None:
    assert check_scope(_ORIG, _FIXED) == []


@pytest.mark.parametrize(
    "corrected",
    [
        "Reading below 25 units or above or equal to 30 units",  # changed number
        "Reading below 20 units or above or equal to 30 units or 40 units",  # added number
        "Reading below 20 units or above or equal to 30 units with gentamicin",  # new drug word
        "Reading below 20 mg or above or equal to 30 units",  # unit changed
    ],
)
def test_changing_values_units_or_clinical_words_is_out_of_scope(corrected: str) -> None:
    assert check_scope(_ORIG, corrected)
    with pytest.raises(CorrectionError):
        load_corrections([_entry(corrected=corrected)])


def test_missing_attestation_fields_are_rejected() -> None:
    with pytest.raises(CorrectionError):
        load_corrections([_entry(attested_by="")])


def test_correction_matches_across_a_line_break_exactly_once() -> None:
    corr = load_corrections([_entry()])[0]
    units = ["• Sign gamma\n• Reading above or equal to\n30 units or below 20 units"]
    new, done = apply_to_units(units, corr)
    assert _FIXED in new[0]
    assert new[0][done.start : done.end] == _FIXED


def test_zero_or_several_matches_fail_closed() -> None:
    corr = load_corrections([_entry()])[0]
    with pytest.raises(CorrectionError, match="0 time"):
        apply_to_units(["unrelated text"], corr)
    with pytest.raises(CorrectionError, match="2 time"):
        apply_to_units([_ORIG, _ORIG], corr)


def test_correction_may_not_touch_generated_structure() -> None:
    corr = Correction(**{k: v for k, v in _entry().items()})
    unit = "[n1] " + _ORIG
    with pytest.raises(CorrectionError, match="structure"):
        apply_to_units([unit], corr, protected=[[(0, 12)]])


# ── lineage / gold remap (§10) ──


def test_split_chunk_maps_to_both_new_chunks_and_junk_maps_to_nothing() -> None:
    old = [
        ("o1", "alpha beta gamma delta epsilon zeta eta theta"),
        ("o2", "Synthetic Care Protocols 32 Integrating Technologies"),
    ]
    new = [
        ("n1", "alpha beta gamma delta"),
        ("n2", "epsilon zeta eta theta"),
        ("n3", "unrelated words"),
    ]
    links = map_chunks(old, new)
    assert {(lk.old_id, lk.new_id) for lk in links} == {("o1", "n1"), ("o1", "n2")}
    assert remap_gold(["o1"], links, {"o1", "o2"}) == ["n1", "n2"]
    assert remap_gold(["o2"], links, {"o1", "o2"}) == []  # its content is gone -> no gold
    assert remap_gold(["other-doc"], links, {"o1", "o2"}) == ["other-doc"]  # not re-ingested


def test_shared_function_words_do_not_link_unrelated_chunks() -> None:
    """Regression (DEVIATIONS.md #216): v1 linked old chunks to up to 18 new
    ones through words like "the/with/and"."""
    old = [("o1", "Give the gentamicin dose with the infusion and review the levels daily")]
    new = [
        ("n1", "Give the gentamicin dose with the infusion and review the levels daily"),
        ("n2", "Record the weight with the scale and the date of birth for the chart"),
        ("n3", "Keep the baby warm with the mother and review the feeding plan"),
    ]
    assert [(lk.old_id, lk.new_id) for lk in map_chunks(old, new)] == [("o1", "n1")]


def test_short_old_chunk_links_only_to_its_most_specific_match() -> None:
    old = [("o1", "Neonatal infection")]
    new = [
        ("n1", "Neonatal infection guidance on antibiotics, dosing, monitoring and review"),
        ("n2", "Neonatal infection"),
        ("n3", "Signs of neonatal infection include poor feeding and lethargy"),
    ]
    assert [(lk.old_id, lk.new_id) for lk in map_chunks(old, new)] == [("o1", "n2")]


def test_unchanged_chunk_links_only_to_its_identical_copy() -> None:
    old = [("o1", "• Do not routinely give antibiotic treatment to babies without risk factors")]
    new = [
        ("n1", "•  Do not routinely give antibiotic treatment to babies without risk factors"),
        ("n2", "Do not routinely give antibiotic treatment"),  # a piece-like near match
    ]
    assert [(lk.old_id, lk.new_id, lk.score) for lk in map_chunks(old, new)] == [("o1", "n1", 1.0)]


def test_text_layer_characters_win_over_docling_decoding() -> None:
    """DEVIATIONS.md #221: Docling dropped every "h" in NICE's Inter font."""
    from app.ingestion.layout.docling_adapter import prefer_text_layer

    assert prefer_text_layer("W at is t e impact", "What is the impact") == (
        "What is the impact",
        True,
    )
    assert prefer_text_layer("Gentamycin 5 mg/kg", "") == (
        "Gentamycin 5 mg/kg",
        False,
    )  # OCR region
    assert prefer_text_layer("same  text", "same text") == ("same text", False)
    # a printed bullet Docling already stripped isn't reintroduced
    assert prefer_text_layer("Membrane rupture", "• Membrane rupture") == (
        "Membrane rupture",
        False,
    )


# --- word completion at the element box edge (DEVIATIONS.md #230) ---


def _chars(text: str, x0: float, top: float = 592.3, width: float = 4.5) -> list[dict]:
    """One pdfplumber-style char per letter; spaces are real characters."""
    out, x = [], x0
    for ch in text:
        out.append(
            {
                "text": ch,
                "x0": x,
                "x1": x + width,
                "top": top,
                "bottom": top + 12.0,
                "upright": True,
                "size": 12.0,
                "fontname": "Inter-Regular",
                "doctop": top,
                "height": 12.0,
                "width": width,
            }
        )
        x += width + 0.1
    return out


def _centre_inside(chars: list[dict], box: BBox) -> list[dict]:
    return [
        c
        for c in chars
        if box.contains_point((c["x0"] + c["x1"]) / 2, (c["top"] + c["bottom"]) / 2, tol=0.5)
    ]


def test_word_cut_at_the_box_edge_is_completed() -> None:
    """NICE p.21: Docling's box ended at the "t" of "birth" (#230)."""
    from app.ingestion.layout.docling_adapter import _complete_words

    line = _chars("hours of birth", x0=100.0)
    t = next(c for c in reversed(line) if c["text"] == "t")
    box = BBox(95.0, 590.0, t["x1"], 606.0)  # ends where the "t" ends
    got = _complete_words(line, _centre_inside(line, box))
    assert "".join(c["text"] for c in sorted(got, key=lambda c: c["x0"])) == "hours of birth"


def test_completion_stops_at_a_space_and_at_a_gap() -> None:
    from app.ingestion.layout.docling_adapter import _complete_words

    left = _chars("culture", x0=100.0)
    spaced = _chars(" and", x0=left[-1]["x1"] + 0.1)
    far = _chars("next", x0=spaced[-1]["x1"] + 6.0)  # other column: no space char, wide gap
    line = left + spaced + far
    box = BBox(95.0, 590.0, left[3]["x1"], 606.0)  # "cult|ure and next"
    got = _complete_words(line, _centre_inside(line, box))
    assert "".join(c["text"] for c in sorted(got, key=lambda c: c["x0"])) == "culture"


def test_other_lines_are_never_pulled_in() -> None:
    from app.ingestion.layout.docling_adapter import _complete_words

    first = _chars("treatment", x0=100.0)
    below = _chars("treatment", x0=100.0, top=606.3)  # next line, same x
    box = BBox(95.0, 590.0, first[5]["x1"], 604.5)
    got = _complete_words(first + below, _centre_inside(first + below, box))
    assert len(got) == len(first) and all(c["top"] == first[0]["top"] for c in got)


def test_plumber_text_reads_the_completed_word() -> None:
    pytest.importorskip("pdfplumber")
    from app.ingestion.layout.docling_adapter import _plumber_text

    line = _chars("jaundice within 24 hours of birth", x0=62.8)
    t = next(c for c in reversed(line) if c["text"] == "t")
    assert _plumber_text(line, BBox(62.8, 592.7, t["x1"], 603.6)) == (
        "jaundice within 24 hours of birth"
    )
