"""Provenance pin + check: the pure comparison rules (PRD-072; DEVIATIONS.md #295).

The database half (`digest_tables`, `known_gaps`, `broken_links`) is plain
SQL run against the live stack by `make provenance-check`."""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.provenance import (
    canonical,
    check_attestations,
    check_version_files,
    compare_files,
    compare_known_gaps,
    compare_tables,
    digest_rows,
    hash_files,
    resolve,
)


def levels(findings):  # noqa: ANN001, ANN201
    return [f.level for f in findings]


def test_digest_ignores_row_order_and_timezone_of_equal_instants() -> None:
    t = datetime(2026, 10, 13, 9, 0, tzinfo=UTC)
    eat = t.astimezone(timezone(timedelta(hours=3)))
    rid = uuid.uuid4()
    a = digest_rows([(rid, t, {"b": 1, "a": 2}), ("x", None, b"\x00")])
    b = digest_rows([("x", None, b"\x00"), (rid, eat, {"a": 2, "b": 1})])
    assert a == b and a["count"] == 2
    assert digest_rows([(rid, t, {"a": 3})]) != digest_rows([(rid, t, {"a": 2})])


def test_bytes_are_hashed_never_written() -> None:
    secret = b"ciphertext that must not appear"
    out = canonical([secret])
    assert secret not in out and hashlib.sha256(secret).hexdigest().encode() in out


def test_files_missing_or_changed_fail_and_new_files_are_reported(tmp_path: Path) -> None:
    (tmp_path / "data" / "crops").mkdir(parents=True)
    (tmp_path / "data" / "a.yaml").write_text("a")
    (tmp_path / "data" / "crops" / "x.png").write_text("cache")
    (tmp_path / "data" / ".gitkeep").write_text("")
    roots = {"data": tmp_path / "data"}
    pinned = hash_files(roots)
    assert list(pinned) == ["data/a.yaml"]  # caches and .gitkeep skipped

    (tmp_path / "data" / "a.yaml").write_text("edited")
    (tmp_path / "data" / "b.pdf").write_text("new")
    findings = compare_files({**pinned, "data/gone.csv": "0" * 64}, hash_files(roots))
    assert [(f.level, f.message) for f in findings] == [
        ("fail", "changed: data/a.yaml"),
        ("fail", "missing: data/gone.csv"),
        ("info", "1 file(s) not pinned: data/b.pdf"),
    ]


def test_table_changes_fail_but_added_rows_are_only_reported() -> None:
    pinned = {"eval.rating_round": {"count": 2, "sha256": "aa"}}
    same = {"eval.rating_round": {"count": 2, "sha256": "aa", "added_since": 5}}
    assert levels(compare_tables(pinned, same)) == ["info"]
    edited = {"eval.rating_round": {"count": 2, "sha256": "bb", "added_since": 0}}
    assert levels(compare_tables(pinned, edited)) == ["fail"]
    deleted = {"eval.rating_round": {"count": 1, "sha256": "cc", "added_since": 0}}
    assert "deleted" in compare_tables(pinned, deleted)[0].message
    assert levels(compare_tables(pinned, {})) == ["fail"]


def test_known_gaps_warn_until_they_grow() -> None:
    pinned = {"questions_missing_source_record": 12}
    assert levels(compare_known_gaps(pinned, {"questions_missing_source_record": 12})) == ["warn"]
    assert levels(compare_known_gaps(pinned, {"questions_missing_source_record": 13})) == ["fail"]
    assert compare_known_gaps(pinned, {"questions_missing_source_record": 0}) == []


def test_active_guideline_files_must_match_their_ingest_hash(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    pdf = tmp_path / "data" / "g.pdf"
    pdf.write_bytes(b"%PDF guideline")
    sha = hashlib.sha256(b"%PDF guideline").hexdigest()
    roots = {"data": tmp_path / "data"}

    def locate(u: str) -> Path | None:
        return resolve(u, roots)

    assert check_version_files([("v1", "data/g.pdf", sha)], locate) == []
    pdf.write_bytes(b"%PDF edited")
    assert "differs" in check_version_files([("v1", "data/g.pdf", sha)], locate)[0].message
    assert "missing" in check_version_files([("v2", "data/none.pdf", sha)], locate)[0].message


def test_resolve_maps_repo_paths_to_mounts() -> None:
    roots = {"data": Path("/app/data"), "backend/results": Path("/app/results")}
    assert resolve("backend/results/ablation/r1/x.json", roots) == Path(
        "/app/results/ablation/r1/x.json"
    )
    assert resolve("data/query_areas.yaml", roots) == Path("/app/data/query_areas.yaml")
    assert resolve("frontend/x", roots) is None


@pytest.mark.parametrize("fence", ["", "---\n# comment\n"])
def test_unsettled_attestation_field_is_flagged(tmp_path: Path, fence: str) -> None:
    ds = tmp_path / "data" / "patient_records" / "deidentified" / "set1"
    ds.mkdir(parents=True)
    (ds / "DATASET.md").write_text(
        fence
        + 'source: "CIN"\nconsent_basis: "Waiver / broad consent / other"\n'
        + '---\nbody: "a / b"\n'
    )
    findings = check_attestations({"data": tmp_path / "data"})
    assert [(f.level, f.message) for f in findings] == [
        (
            "warn",
            "data/patient_records/deidentified/set1/DATASET.md: consent_basis lists alternatives",
        )
    ]
