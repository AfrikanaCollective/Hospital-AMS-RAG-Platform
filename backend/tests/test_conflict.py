"""Conflict detection (ARCH §7.6; PRD-014)."""

from __future__ import annotations

from app.retrieval.conflict import detect_conflicts


def _item(**kw: object) -> dict:
    base = {
        "chunk_id": "c1",
        "score": 0.9,
        "section_path": None,
        "section_number": None,
        "page_start": 1,
        "page_end": 1,
        "char_start": 0,
        "char_end": 10,
        "document_id": "doc-1",
        "document_title": "T",
        "document_version_id": "v1",
        "version_label": "1.0",
        "effective_date": None,
        "version_status": "active",
        "chunk_type": "prose",
        "text": "",
    }
    base.update(kw)
    return base  # type: ignore[return-value]


def test_same_section_different_versions_with_different_text_flagged() -> None:
    items = [
        _item(
            chunk_id="a",
            document_version_id="v1",
            section_number="3.2",
            text="Give amoxicillin 500mg twice daily for 7 days.",
        ),
        _item(
            chunk_id="b",
            document_version_id="v2",
            section_number="3.2",
            text="Give ceftriaxone 50mg/kg once daily for 10 days.",
        ),
    ]
    flags = detect_conflicts(items)
    assert any(f["reason"] == "same_section_diff_versions" for f in flags)


def test_same_section_same_version_not_flagged() -> None:
    items = [
        _item(
            chunk_id="a",
            document_version_id="v1",
            section_number="3.2",
            text="Give amoxicillin 500mg twice daily.",
        ),
        _item(
            chunk_id="b",
            document_version_id="v1",
            section_number="3.2",
            text="A different sentence entirely about something else.",
        ),
    ]
    assert detect_conflicts(items) == []


def test_similar_text_across_versions_not_flagged() -> None:
    items = [
        _item(
            chunk_id="a",
            document_version_id="v1",
            section_number="3.2",
            text="Give amoxicillin 500mg twice daily for 7 days.",
        ),
        _item(
            chunk_id="b",
            document_version_id="v2",
            section_number="3.2",
            text="Give amoxicillin 500 mg twice daily for 7 days.",
        ),
    ]
    assert detect_conflicts(items) == []


def test_withdrawn_version_excluded_from_section_check() -> None:
    items = [
        _item(
            chunk_id="a",
            document_version_id="v1",
            section_number="3.2",
            version_status="active",
            text="Give amoxicillin.",
        ),
        _item(
            chunk_id="b",
            document_version_id="v2",
            section_number="3.2",
            version_status="withdrawn",
            text="Give a totally different drug entirely.",
        ),
    ]
    assert detect_conflicts(items) == []


def test_recommendation_negation_contradiction_flagged() -> None:
    recommended = "Prophylactic antibiotics are recommended before the procedure."
    not_recommended = "Prophylactic antibiotics are not recommended before the procedure."
    items = [
        _item(chunk_id="a", chunk_type="recommendation", text=recommended),
        _item(chunk_id="b", chunk_type="recommendation", text=not_recommended),
    ]
    flags = detect_conflicts(items)
    assert any(f["reason"] == "lexical_contradiction" for f in flags)


def test_unrelated_recommendations_not_flagged() -> None:
    recommended = "Prophylactic antibiotics are recommended before the procedure."
    unrelated = "Vitamin K is given to all newborns shortly after birth."
    items = [
        _item(chunk_id="a", chunk_type="recommendation", text=recommended),
        _item(chunk_id="b", chunk_type="recommendation", text=unrelated),
    ]
    assert detect_conflicts(items) == []


def test_different_guidelines_sharing_a_section_number_not_flagged() -> None:
    """Rule (a) compares versions of ONE document (DEVIATIONS.md #261): two
    guidelines that both number a section "1.10" are not two versions."""
    items = [
        _item(
            chunk_id="a",
            document_id="kenya",
            document_version_id="vk",
            section_number="1.10",
            text="Admit in category A and do blood cultures.",
        ),
        _item(
            chunk_id="b",
            document_id="nice",
            document_version_id="vn",
            section_number="1.10",
            text="Perform an immediate clinical assessment and review the history.",
        ),
    ]
    assert detect_conflicts(items) == []
