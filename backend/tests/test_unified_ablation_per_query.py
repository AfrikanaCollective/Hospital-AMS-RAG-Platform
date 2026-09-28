"""Unified-ablation run-directory helpers (PRD-112 / ARCH-043;
DEVIATIONS.md #211: vocabulary content hash in `configuration.json`)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app.eval.unified_ablation.per_query import file_sha256, write_configuration


def test_file_sha256_matches_hashlib_and_changes_with_content(tmp_path: Path) -> None:
    vocab = tmp_path / "clinical_concepts.yaml"
    vocab.write_bytes(b"concepts: []\n")
    first = file_sha256(vocab)
    assert first == hashlib.sha256(b"concepts: []\n").hexdigest()

    vocab.write_bytes(b"concepts: [grunting]\n")
    assert file_sha256(vocab) != first


def test_file_sha256_is_none_for_missing_file(tmp_path: Path) -> None:
    assert file_sha256(tmp_path / "absent.yaml") is None


def test_concepts_hash_round_trips_through_configuration_json(tmp_path: Path) -> None:
    vocab = tmp_path / "clinical_concepts.yaml"
    vocab.write_bytes(b"concepts: []\n")
    path = write_configuration(tmp_path, {"concepts_sha256": file_sha256(vocab)})
    assert json.loads(path.read_text())["concepts_sha256"] == file_sha256(vocab)
