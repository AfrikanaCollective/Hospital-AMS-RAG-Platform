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
from app.ingestion.layout.boilerplate import normalize, remove_boilerplate
from app.ingestion.layout.headings import fuse_heading_levels
from app.ingestion.layout.model import BBox, Element, LayoutDocument, LayoutPage
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
    pages = [
        _page(
            n, [_el("text", "Give 5 mg/kg daily", 690, n), _el("heading", "Section title", 20, n)]
        )
        for n in range(1, 5)
    ]
    doc = LayoutDocument("x.pdf", "t", pages)
    remove_boilerplate(doc, margin_zone=0.08, repeat_ratio=0.5, min_pages=3)
    assert all(len(p.elements) == 2 for p in doc.pages)


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
