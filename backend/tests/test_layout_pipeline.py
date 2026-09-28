"""Layout ingestion end to end, offline (ARCH-044, PRD-113;
LAYOUT-INGESTION-PROPOSAL.md §3, §5.8-§5.11, §6, §8): synthetic layout ->
normalized text -> chunks -> citations -> grounding, plus the review gate in
an in-memory Qdrant and the parser fallback. No parser, network or DB."""

from __future__ import annotations

import pytest

from app.citations.model import build_citation
from app.grounding.verifier import verify
from app.ingestion.chunking import chunk_document
from app.ingestion.corrections import CorrectionError, load_corrections
from app.ingestion.layout import pipeline
from app.ingestion.layout.assemble import AssemblyOptions, assemble
from app.ingestion.page_provenance import apply_source_pages
from app.ingestion.review import hold_low_quality_chunks
from app.schemas.enums import GroundingVerdict
from app.schemas.query import QueryResponse
from tests.layout_fixtures import synthetic_document

ORIG = "Reading above or equal to 30 units or below 20 units"
FIXED = "Reading below 20 units or above or equal to 30 units"
CORRECTION = {
    "id": "syn-p47-reading",
    "page": 47,
    "original": ORIG,
    "corrected": FIXED,
    "kind": "erratum",
    "rationale": "inequalities transposed in source",
    "evidence": "synthetic evidence",
    "attested_by": "Dr Example (paediatrics)",
    "attested_on": "2026-09-28",
}


def _parsed(**opts):
    doc = synthetic_document(**{k: v for k, v in opts.items() if k == "drop_arrowhead_on"})
    options = AssemblyOptions(
        source_pages=[47, 48],
        corrections=load_corrections(opts.get("corrections", [CORRECTION])),
    )
    parsed = assemble(doc, options)
    apply_source_pages(parsed, [47, 48])
    return parsed


def _chunks(**opts) -> list[dict]:
    return chunk_document(_parsed(**opts), format_profile="clinical_protocol")


def _by_type(chunks: list[dict], t: str) -> list[dict]:
    return [c for c in chunks if c["chunk_type"] == t]


def _as_retrieved(c: dict, chunk_id: str = "k1") -> dict:
    meta = {k: v for k, v in c["meta"].items() if k != "embedding_text"}
    return {
        "chunk_id": chunk_id,
        "text": c["text"],
        "chunk_type": c["chunk_type"],
        "meta": meta,
        "document_id": "d1",
        "document_title": "Synthetic Care Protocols",
        "document_version_id": "v2",
        "version_label": "2022",
        "version_status": "active",
        "effective_date": None,
        "section_number": c["section_number"],
        "section_path": c["section_path"],
        "page_start": c["page_start"],
        "page_end": c["page_end"],
        "char_start": c["char_start"],
        "char_end": c["char_end"],
    }


def test_every_chunk_is_an_exact_slice_of_the_normalized_text() -> None:
    parsed = _parsed()
    for c in chunk_document(parsed, format_profile="clinical_protocol"):
        if c["meta"].get("split_group_id") is None and c["chunk_type"] != "prose":
            assert parsed.normalized_text[c["char_start"] : c["char_end"]] == c["text"]


def test_no_running_header_or_footer_in_any_chunk() -> None:
    for c in _chunks():
        assert "Synthetic Care Protocols" not in c["text"]
        assert "Page 2 of 9" not in c["text"]


def test_flowchart_is_one_atomic_verified_chunk_on_the_source_page() -> None:
    [flow] = _by_type(_chunks(), "flowchart")
    graph = flow["meta"]["flowchart"]
    assert graph["verification"] == "verified"
    assert len(graph["nodes"]) == 6 and len(graph["edges"]) == 5
    assert flow["page_start"] == 47  # source_pages remap applied
    assert flow["figure_ref"]["page"] == 47 and flow["figure_ref"]["bbox"]
    assert "→ Yes →" in flow["text"]
    # retrieval-only path summary is in the embedding text, not the citable text
    assert "—Yes→" in flow["meta"]["embedding_text"]
    assert "—Yes→" not in flow["text"]


def test_attested_correction_is_applied_and_recorded() -> None:
    [flow] = _by_type(_chunks(), "flowchart")
    assert FIXED in flow["text"] and ORIG not in " ".join(flow["text"].split())
    [corr] = flow["meta"]["corrections"]
    assert corr["id"] == "syn-p47-reading" and corr["original"] == ORIG


def test_correction_that_matches_nothing_stops_the_document() -> None:
    with pytest.raises(CorrectionError):
        _parsed(corrections=[CORRECTION | {"original": "text that is not on the page at all"}])


def test_ocr_table_with_numbers_is_held_for_review() -> None:
    [table] = _by_type(_chunks(), "table")
    assert table["meta"]["review_status"] == "pending"
    assert table["meta"]["ocr"]["has_digits"] is True
    assert "Weight (kg) 1.0 — Agent P" in table["meta"]["embedding_text"]  # row rendering
    [prose] = [c for c in _chunks() if "synthetic agent" in c["text"]]
    assert "review_status" not in prose["meta"]  # text-layer prose isn't held


def test_low_parse_quality_holds_every_chunk() -> None:
    chunks = _chunks()
    assert hold_low_quality_chunks(chunks, 0.4, 0.6) == len(chunks) - 1  # table already held
    assert all(c["meta"]["review_status"] == "pending" for c in chunks)
    assert hold_low_quality_chunks(_chunks(), 0.9, 0.6) == 0


# ── citations + grounding (§5.11, §6) ──


def _claim(quote: str) -> dict:
    return {"type": "claim", "text": quote, "citation_ids": ["c1"], "quote": quote}


def test_citation_of_corrected_text_carries_the_correction_and_the_notice() -> None:
    [flow] = _by_type(_chunks(), "flowchart")
    row = _as_retrieved(flow)
    cit = build_citation("c1", row, FIXED)
    assert [c["id"] for c in cit.corrections] == ["syn-p47-reading"]
    resp = QueryResponse(
        conversation_id="x",
        message_id="y",
        observed_outcome="well_supported",
        scope_label="scope_1",
        citations=[cit],
    )
    assert len(resp.correction_notices) == 1
    assert ORIG in resp.correction_notices[0] and "Dr Example" in resp.correction_notices[0]
    # a quote elsewhere in the same chunk carries no correction
    assert build_citation("c1", row, "Pathway one").corrections == []


def test_verified_flowchart_can_fully_support_a_claim() -> None:
    [flow] = _by_type(_chunks(), "flowchart")
    report = verify([_claim("Pathway one")], [_as_retrieved(flow)])
    assert report.per_segment[0].verdict == GroundingVerdict.SUPPORTED


def test_unverified_flowchart_is_capped_at_weak() -> None:
    [flow] = _by_type(_chunks(drop_arrowhead_on="E-F"), "flowchart")
    assert flow["meta"]["flowchart"]["verification"] == "partial"
    report = verify([_claim("Pathway one")], [_as_retrieved(flow)])
    assert report.per_segment[0].verdict == GroundingVerdict.WEAK
    assert report.action == "release_marked"


def test_caption_only_figure_is_capped_at_weak() -> None:
    fig = {
        "chunk_type": "figure",
        "text": "Figure 3 Management algorithm",
        "meta": {"has_embedded_text": False},
        "section_number": None,
        "section_path": None,
        "page_start": 1,
        "page_end": 1,
        "char_start": 0,
        "char_end": 29,
    }
    report = verify([_claim("Figure 3 Management algorithm")], [_as_retrieved(fig)])
    assert report.per_segment[0].verdict == GroundingVerdict.WEAK


def test_chunk_under_review_cannot_support_a_claim() -> None:
    [table] = _by_type(_chunks(), "table")
    report = verify([_claim("Agent P (10 u/kg)")], [_as_retrieved(table)])
    assert report.per_segment[0].verdict == GroundingVerdict.UNSUPPORTED
    assert report.per_segment[0].reason == "chunk_under_review"


# ── pipeline fallback (§4) ──


def test_parser_failure_falls_back_to_pypdf_and_is_held(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from app.ingestion.layout import docling_adapter
    from app.ingestion.pdf_parse import ParsedDocument

    def boom(*a, **k):
        raise RuntimeError("layout model unavailable")

    monkeypatch.setattr(docling_adapter, "parse_layout", boom)
    monkeypatch.setattr(
        pipeline,
        "parse_document",
        lambda path: ParsedDocument("text", [], 1, 0.9, [(1, 0)]),
    )
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    parsed = pipeline.parse_with_layout(str(pdf), {})
    assert parsed.parser_version == "pypdf-fallback"
    assert parsed.parse_quality <= pipeline.FALLBACK_QUALITY_CAP
    assert "layout model unavailable" in parsed.parse_report["error"]


def test_bad_correction_never_falls_back(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    with pytest.raises(CorrectionError):
        pipeline.parse_with_layout(
            str(pdf), {"text_corrections": [CORRECTION | {"corrected": "Reading 99"}]}
        )
