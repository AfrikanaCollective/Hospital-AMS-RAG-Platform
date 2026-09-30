"""Every unified-ablation run renders its three figures (PRD-112 / ARCH-043;
DEVIATIONS.md #229). Synthetic per-query rows, no Postgres/Qdrant."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

pytest.importorskip("seaborn")

from scripts.plot_ablation_figures import render_all  # noqa: E402
from scripts.plot_recall_by_bm25_weight import read_per_query  # noqa: E402

_WEIGHTS = (0.0, 0.5, 1.0)


def _write_run(run_dir: Path, ks: tuple[int, ...]) -> None:
    run_dir.mkdir()
    with (run_dir / "per_query_results.jsonl").open("w") as fh:
        for q, l1, l2, w, k in itertools.product(
            ("q1", "q2"), ("present_only", "all_assessed"), ("raw", "enriched"), _WEIGHTS, ks
        ):
            fh.write(
                json.dumps(
                    {
                        "query_id": q,
                        "level1_condition": l1,
                        "level2_condition": l2,
                        "bm25_weight": w,
                        "k": k,
                        "query_text": "not needed by the figures",
                        "retrieved_ids": ["a", "b"],
                        "recall_at_k": 1.0 - w / 2,
                        "reciprocal_rank_at_k": 0.5 + w / 4,
                    }
                )
                + "\n"
            )


def test_render_all_writes_the_three_figures(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir, ks=(8, 10, 12, 14))
    written = render_all(run_dir, weights=_WEIGHTS, mrr_k=12)
    assert {p.name for p in written} == {
        "recall_at_k_by_bm25_weight.png",
        "recall_at_k_vs_bm25_weight_by_k.png",
        "mrr_at_k_vs_bm25_weight_by_arm.png",
    }
    assert all(p.is_file() and p.stat().st_size > 0 for p in written)


def test_figures_needing_missing_k_values_are_skipped_not_fatal(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir, ks=(2, 4))  # none of 8/10/12/14, no k=12 for MRR
    written = render_all(run_dir, weights=_WEIGHTS, mrr_k=12)
    assert [p.name for p in written] == ["recall_at_k_by_bm25_weight.png"]


def test_read_per_query_keeps_only_the_figure_columns(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir, ks=(12,))
    df = read_per_query(run_dir / "per_query_results.jsonl")
    assert "query_text" not in df.columns and "retrieved_ids" not in df.columns
    assert df.bm25_weight.dtype == float
