"""OCR-artefact overrides (ARCH-044; DEVIATIONS.md #243): an `ocr_override`
that only undoes OCR damage ("1.V / 1.M", "Metronid azole") passes the scope
limit and may fix every occurrence on a page; nothing clinical can change."""

from __future__ import annotations

import pytest

from app.ingestion.chunking import chunk_document
from app.ingestion.corrections import (
    CorrectionError,
    apply_all_to_units,
    is_ocr_artefact_fix,
    load_corrections,
)
from app.ingestion.layout.assemble import AssemblyOptions, assemble
from app.ingestion.layout.model import (
    BBox,
    Element,
    LayoutDocument,
    LayoutPage,
    TableCell,
    TableData,
)


@pytest.mark.parametrize(
    ("original", "corrected"),
    [
        ("1.V / 1.M", "I.V / I.M"),
        ("1.V/ 1.M", "I.V/ I.M"),
        ("I.V / 1.M", "I.V / I.M"),
        ("Metronid azole** 7.5 mg/kg · 1.V", "Metronidazole** 7.5 mg/kg · I.V"),
        ("Flucloxacill in", "Flucloxacillin"),
    ],
)
def test_ocr_artefact_fixes_pass(original: str, corrected: str) -> None:
    assert is_ocr_artefact_fix(original, corrected)


@pytest.mark.parametrize(
    ("original", "corrected"),
    [
        ("1 g", "I g"),  # a dose digit turned into a letter
        ("IO mg", "10 mg"),  # letters turned into a dose
        ("10 mg", "IO mg"),
        ("0.5 g", "O.5 g"),  # a decimal is not glued to a letter
        ("Amikacin", "Gentamicin"),  # a different drug
        ("every 8 hours", "every 6 hours"),
        ("same", "same"),
    ],
)
def test_anything_beyond_ocr_artefacts_fails(original: str, corrected: str) -> None:
    assert not is_ocr_artefact_fix(original, corrected)


def _entry(**kw: object) -> dict:
    base = {
        "id": "kenya-p48-route",
        "page": 48,
        "original": "1.V / 1.M",
        "corrected": "I.V / I.M",
        "kind": "ocr_override",
        "rationale": "OCR read the route abbreviation I.V/I.M as 1.V/1.M",
        "evidence": "Page image shows I.V / I.M",
        "attested_by": "Dr Example",
        "attested_on": "2026-10-01",
        "occurrences": "all",
    }
    base.update(kw)
    return base


def test_all_occurrences_requires_an_ocr_artefact_override() -> None:
    assert load_corrections([_entry()])[0].occurrences == "all"
    with pytest.raises(CorrectionError, match="out of scope"):
        load_corrections([_entry(kind="erratum")])  # an erratum gets no artefact pass
    with pytest.raises(CorrectionError, match="occurrences 'all'"):
        # in scope as an erratum (relational words only), but not an OCR artefact
        load_corrections(
            [_entry(kind="erratum", original="less than 5 days", corrected="more than 5 days")]
        )
    with pytest.raises(CorrectionError, match="out of scope"):
        load_corrections([_entry(original="Amikacin", corrected="Gentamicin")])
    with pytest.raises(CorrectionError, match="occurrences must be"):
        load_corrections([_entry(occurrences="some")])


def test_apply_all_replaces_every_match_with_valid_offsets() -> None:
    corr = load_corrections([_entry()])[0]
    units = ["A · 1.V / 1.M · 8 hrly: 125\n  B · 1.V / 1.M · 24 hrly: 20", "no match here"]
    new, done = apply_all_to_units(units, corr)
    assert new[0] == "A · I.V / I.M · 8 hrly: 125\n  B · I.V / I.M · 24 hrly: 20"
    assert [new[d.unit_index][d.start : d.end] for d in done] == ["I.V / I.M", "I.V / I.M"]
    with pytest.raises(CorrectionError, match="not found"):
        apply_all_to_units(["nothing"], corr)


def test_corrected_table_still_chunks_and_carries_the_correction() -> None:
    """A table's parts are copies of its text that chunking looks up in the
    document; they must be corrected too, or ingestion would fail."""
    cells = [
        TableCell(0, 0, 1, 1, "Weight (kg)", is_header=True),
        TableCell(0, 1, 1, 1, "Gentamycin 7.5 mg/kg · 1.V / 1.M · 24 hrly", is_header=True),
        TableCell(0, 2, 1, 1, "Ceftazidime 50mg/kg · 1.V / 1.M · 8 hrly", is_header=True),
        TableCell(1, 0, 1, 1, "2.5"),
        TableCell(1, 1, 1, 1, "20"),
        TableCell(1, 2, 1, 1, "125"),
        TableCell(2, 0, 1, 1, "3.0"),
        TableCell(2, 1, 1, 1, "20"),
        TableCell(2, 2, 1, 1, "150"),
    ]
    table = Element(
        "table", 1, BBox(50, 100, 380, 300), table=TableData(num_rows=3, num_cols=3, cells=cells)
    )
    heading = Element("heading", 1, BBox(50, 60, 300, 72), text="Antibiotic doses", level=1)
    doc = LayoutDocument("kenya.pdf", "t", [LayoutPage(1, 420, 595, [heading, table])])
    corr = load_corrections([_entry(page=1)])[0]
    parsed = assemble(doc, AssemblyOptions(corrections=[corr]))
    assert parsed.parse_report["corrections_applied"][0]["occurrences"] == 4
    chunks = chunk_document(parsed, format_profile="clinical_protocol")
    text = next(c["text"] for c in chunks if c["chunk_type"] == "table")
    assert "1.V" not in text and text.count("I.V / I.M") == 4
    assert "20" in text and "125" in text and "150" in text  # values untouched
