"""Curated per-area reference sets (PRD-112, ARCH-043; DEVIATIONS.md #307)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from app.eval.area_reference import (
    AreaReferenceNotAttested,
    AreaReferenceUnresolved,
    chunk_text_sha256,
    load_area_reference_sets,
    resolve_reference_sets,
    score_rows,
)
from app.eval.question_gen.areas import AREA_ORDER
from scripts.area_reference_review import ReviewError, build_document, write_document

H = {area: chunk_text_sha256(f"passage for {area}") for area in AREA_ORDER}


def _write(tmp_path: Path, **overrides) -> Path:
    body = {
        "attested_by": "Timothy Tuti",
        "attested_role": "Primary Investigator",
        "attested_date": "2026-10-21",
        "areas": {
            area: {
                "passages": [
                    {"include": True, "text_sha256": H[area]},
                    {"include": False, "text_sha256": chunk_text_sha256("noise")},
                ]
            }
            for area in AREA_ORDER
        },
    }
    body.update(overrides)
    path = tmp_path / "area_reference_sets.yaml"
    path.write_text(yaml.safe_dump(body))
    return path


def test_an_attested_file_loads_only_included_passages(tmp_path: Path) -> None:
    sets = load_area_reference_sets(_write(tmp_path))
    assert sets.passages == {area: frozenset({H[area]}) for area in AREA_ORDER}
    assert len(sets.sha256) == 64


@pytest.mark.parametrize("field", ["attested_by", "attested_role", "attested_date"])
def test_a_placeholder_attestation_fails_closed(tmp_path: Path, field: str) -> None:
    with pytest.raises(AreaReferenceNotAttested):
        load_area_reference_sets(_write(tmp_path, **{field: "TODO_CONFIRM"}))


def test_missing_file_area_or_included_passage_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(AreaReferenceNotAttested):
        load_area_reference_sets(tmp_path / "absent.yaml")
    with pytest.raises(AreaReferenceNotAttested):
        load_area_reference_sets(_write(tmp_path, areas={"assessment": {"passages": []}}))
    areas = {a: {"passages": [{"include": True, "text_sha256": H[a]}]} for a in AREA_ORDER}
    areas["investigations"] = {"passages": [{"include": False, "text_sha256": H["assessment"]}]}
    with pytest.raises(AreaReferenceNotAttested):
        load_area_reference_sets(_write(tmp_path, areas=areas))


def test_include_must_be_an_explicit_boolean(tmp_path: Path) -> None:
    areas = {a: {"passages": [{"include": "yes", "text_sha256": H[a]}]} for a in AREA_ORDER}
    with pytest.raises(AreaReferenceNotAttested):
        load_area_reference_sets(_write(tmp_path, areas=areas))


def test_passages_resolve_by_text_hash_and_fail_when_missing(tmp_path: Path) -> None:
    sets = load_area_reference_sets(_write(tmp_path))
    by_hash = {H[a]: [f"chunk-{a}"] for a in AREA_ORDER}
    assert resolve_reference_sets(sets, by_hash)["assessment"] == frozenset({"chunk-assessment"})
    del by_hash[H["investigations"]]
    with pytest.raises(AreaReferenceUnresolved):
        resolve_reference_sets(sets, by_hash)


def _row(area: str, retrieved: list[str], k: int = 2) -> dict:
    return {
        "level1_condition": "present_only",
        "level2_condition": "raw",
        "bm25_weight": 1.0,
        "k": k,
        "question_area": area,
        "retrieved_ids": retrieved,
        "recall_at_k": 0.0,
        "reciprocal_rank_at_k": 0.0,
        "hit_at_k": 0.0,
    }


def test_scores_use_the_area_set_not_the_run_gold() -> None:
    reference = {"assessment": frozenset({"a", "b"})}
    rows = [_row("assessment", ["x", "a"]), _row("assessment", ["a", "b"]), _row("other", ["a"])]
    scores = {s.metric: s for s in score_rows(rows, reference)}
    assert scores["recall"].n == 2  # the area without a set is skipped
    assert scores["recall"].mean == pytest.approx(0.75)
    assert scores["mrr"].mean == pytest.approx(0.75)
    assert scores["hit"].mean == pytest.approx(1.0)
    assert scores["recall"].current_gold_mean == 0.0


# --- local review page (scripts/area_reference_review.py, DEVIATIONS.md #308) ---

CORPUS = [
    {
        "document": "Doc",
        "page": 1,
        "heading": f"h-{a}",
        "text_sha256": H[a],
        "text": f"passage for {a}",
        "cited_by": {b: [1, 4] for b in AREA_ORDER},
    }
    for a in AREA_ORDER
]


def _payload(reviewed: bool = True, include: bool = True, **attestation) -> dict:
    return {
        "areas": {
            a: {"reviewed": reviewed, "passages": [{"include": include, "text_sha256": H[a]}]}
            for a in AREA_ORDER
        },
        "attestation": {"by": "Timothy Tuti", "role": "Primary Investigator", "date": "2026-10-21"}
        | attestation,
    }


def test_an_attesting_save_writes_a_file_the_loader_accepts(tmp_path: Path) -> None:
    body = build_document(_payload(), CORPUS, attest=True)
    path = tmp_path / "sets.yaml"
    path.write_text(yaml.safe_dump(body))
    sets = load_area_reference_sets(path)
    assert sets.attested_by == "Timothy Tuti"
    assert body["areas"]["assessment"]["passages"][0]["cited_by"] == "1 of 4 answers (25%)"


def test_a_draft_save_never_leaves_the_file_signed(tmp_path: Path) -> None:
    body = build_document(_payload(), CORPUS, attest=False)
    assert body["attested_by"] == body["attested_role"] == body["attested_date"] == "TODO_CONFIRM"
    path = tmp_path / "sets.yaml"
    path.write_text(yaml.safe_dump(body))
    with pytest.raises(AreaReferenceNotAttested):
        load_area_reference_sets(path)


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        (_payload(reviewed=False), "not marked reviewed"),
        (_payload(include=False), "no included passage"),
        (_payload(by=""), "name and role"),
        (_payload(date="21/10/2026"), "YYYY-MM-DD"),
    ],
)
def test_attesting_requires_review_inclusion_and_signature(payload: dict, reason: str) -> None:
    with pytest.raises(ReviewError, match=reason):
        build_document(payload, CORPUS, attest=True)


def test_a_save_keeps_the_file_inode_so_the_api_mount_sees_it(tmp_path: Path) -> None:
    # `api` bind-mounts the single file; replacing it would leave the
    # container reading the old inode (DEVIATIONS.md #309).
    path = tmp_path / "sets.yaml"
    path.write_text("# header\n" + "x" * 5000)
    inode = path.stat().st_ino
    write_document(path, "# header\n", build_document(_payload(), CORPUS, attest=True))
    assert path.stat().st_ino == inode
    assert load_area_reference_sets(path).attested_by == "Timothy Tuti"
