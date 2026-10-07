"""Every unified-ablation run renders its figures, Recall@K and Hit@K
alongside each other (PRD-112 / ARCH-043; DEVIATIONS.md #229, #269).
Synthetic per-query rows, no Postgres/Qdrant."""

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
                        "hit_at_k": 1.0,
                    }
                )
                + "\n"
            )


def test_render_all_writes_recall_and_hit_figures(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir, ks=(8, 10, 12, 14))
    written = render_all(run_dir, weights=_WEIGHTS)
    assert {p.name for p in written} == {
        "recall_at_k_by_bm25_weight.png",
        "recall_at_k_vs_bm25_weight_by_k.png",
        "hit_at_k_by_bm25_weight.png",
        "hit_at_k_vs_bm25_weight_by_k.png",
        "mrr_at_k_vs_bm25_weight_by_arm.png",
    }
    assert all(p.is_file() and p.stat().st_size > 0 for p in written)


def test_figures_needing_missing_k_values_are_skipped_not_fatal(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir, ks=(2, 3))  # none of the by-K lines or MRR panels
    written = render_all(run_dir, weights=_WEIGHTS)
    assert [p.name for p in written] == [
        "recall_at_k_by_bm25_weight.png",
        "hit_at_k_by_bm25_weight.png",
    ]


def test_read_per_query_keeps_only_the_figure_columns(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir, ks=(12,))
    df = read_per_query(run_dir / "per_query_results.jsonl")
    assert "query_text" not in df.columns and "retrieved_ids" not in df.columns
    assert df.bm25_weight.dtype == float


def test_hit_at_k_is_derived_from_first_relevant_rank_for_older_rows(tmp_path: Path) -> None:
    """Runs written before `hit_at_k` existed (DEVIATIONS.md #269) still plot:
    a hit at K iff the first gold chunk's full-ranking rank is <= K."""
    path = tmp_path / "per_query_results.jsonl"
    base = {
        "level1_condition": "present_only",
        "level2_condition": "raw",
        "bm25_weight": 0.0,
        "recall_at_k": 0.0,
        "reciprocal_rank_at_k": 0.0,
    }
    rows = [
        {**base, "k": 4, "first_relevant_rank": 3},
        {**base, "k": 2, "first_relevant_rank": 3},
        {**base, "k": 2, "first_relevant_rank": None},
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert read_per_query(path).hit_at_k.tolist() == [1.0, 0.0, 0.0]


def test_by_k_figures_draw_k_lines_4_to_14(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    """DEVIATIONS.md #274: both by-K figures draw K = 4, 6, ..., 14."""
    from scripts import plot_recall_vs_bm25_weight_by_k as by_k  # noqa: PLC0415

    drawn: dict[str, tuple[int, ...]] = {}
    monkeypatch.setattr(by_k, "plot", lambda agg, ks, out, metric: drawn.setdefault(metric, ks))
    run_dir = tmp_path / "run"
    _write_run(run_dir, ks=(2, 4, 6, 8, 10, 12, 14))
    render_all(run_dir, weights=_WEIGHTS)
    assert drawn == {"recall_at_k": (4, 6, 8, 10, 12, 14), "hit_at_k": (4, 6, 8, 10, 12, 14)}


def test_by_weight_figures_plot_every_k(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    """DEVIATIONS.md #274 (reverting #273): the Recall@K / Hit@K by-weight
    figures' K axis covers every K in the run, from K=2."""
    from scripts import plot_recall_by_bm25_weight as by_weight  # noqa: PLC0415

    plotted: dict[str, list[int]] = {}
    real_relplot = by_weight.sns.relplot

    def _spy(*, data, x, y, **kw):  # noqa: ANN001, ANN202
        if x == "k":  # the by-weight figures; the by-K ones put the weight on x
            plotted[y] = sorted(data.k.unique())
        return real_relplot(data=data, x=x, y=y, **kw)

    monkeypatch.setattr(by_weight.sns, "relplot", _spy)
    run_dir = tmp_path / "run"
    _write_run(run_dir, ks=(2, 4, 6))
    render_all(run_dir, weights=_WEIGHTS)
    assert plotted == {"recall_at_k": [2, 4, 6], "hit_at_k": [2, 4, 6]}


def test_mrr_figure_has_panels_a_to_d_for_k_8_to_14(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    """DEVIATIONS.md #275: one MRR figure, panels A-D = MRR@8/10/12/14."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    from scripts import plot_mrr_vs_bm25_weight_by_arm as mrr_by_arm  # noqa: PLC0415

    figures = []
    real_savefig = plt.Figure.savefig

    def _keep(fig, *a, **kw):  # noqa: ANN001, ANN202
        figures.append(fig)
        return real_savefig(fig, *a, **kw)

    monkeypatch.setattr(plt.Figure, "savefig", _keep)
    run_dir = tmp_path / "run"
    _write_run(run_dir, ks=(2, 8, 10, 12, 14, 16))
    agg = mrr_by_arm.load_mean_recall(
        run_dir / "per_query_results.jsonl", _WEIGHTS, metric="reciprocal_rank_at_k"
    )
    mrr_by_arm.plot(agg, (8, 10, 12, 14), tmp_path / "mrr.png")
    axes = figures[-1].axes
    assert [ax.get_title() for ax in axes] == ["K=8", "K=10", "K=12", "K=14"]
    letters = [t.get_text() for ax in axes for t in ax.texts]
    assert letters == ["A", "B", "C", "D"]
    assert axes[0].get_ylabel() == "Mean Reciprocal Rank (MRR)"
