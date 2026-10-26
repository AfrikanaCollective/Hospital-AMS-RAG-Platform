"""CLI: python -m scripts.plot_mrr_vs_bm25_weight_by_arm RUN_DIR
[--out FILE] [--k-values 4,6,8,10] [--bm25-weight-values 0.0,0.2,...]
(PRD-112 / ARCH-043 — supplementary figure for a unified ablation run).

Companion to `scripts/plot_recall_by_bm25_weight.py`: mean reciprocal rank
(MRR@K, the per-query `reciprocal_rank_at_k` — the ablation's secondary
metric) vs. BM25 score weight, one line per Level-1 x Level-2 arm
(Present-only/All-assessed x Raw/Enriched, plus the dashed "Area question
only / Raw" line since DEVIATIONS.md #311). One panel per K, lettered A, B,
C, D in K order, titled K=4, K=6, K=8, K=10 by default (operator request,
DEVIATIONS.md #275/#277, K=2 -> K=10 in #319; it was a single panel at `ABLATION_MRR_K` before).
Output is an 18 x 18 cm, 300 dpi PNG written next to the input
(`mrr_at_k_vs_bm25_weight_by_arm.png` by default).

The weight grid is the configured one (`bm25_weight_values()`). Fails if a
requested K has no rows in the run.

Arms are categorical, so they use four fixed-order categorical hues plus a
distinct marker per arm: two of the hues sit below 3:1 contrast on white, so
identity never rests on color alone.

Purely a read of an existing run's file output — no DB, Qdrant, or model
calls. Requires the `retrieval-tuning` optional extra
(seaborn/pandas/matplotlib).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from app.eval.ablation_config import bm25_weight_values  # noqa: E402
from scripts.plot_recall_by_bm25_weight import (  # noqa: E402
    _CM,
    _DPI,
    _FIGSIZE_CM,
    SCORER_SUFFIX,
    facet_order,
    load_mean_recall,
    select_scorer,
    with_suffix,
)

_DEFAULT_OUT_NAME = "mrr_at_k_vs_bm25_weight_by_arm.png"
# One panel per K, lettered A, B, C, D in this order (DEVIATIONS.md #275).
_DEFAULT_K_VALUES = "4,6,8,10"
_PANEL_LETTERS = "ABCD"
# Categorical slots 1-4 (blue, orange, aqua, yellow), validated as a set,
# plus a dark purple for the area-question arm (DEVIATIONS.md #311), which
# is also the only arm drawn dashed.
_ARM_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#5b3a8c"]
_ARM_MARKERS = ["o", "s", "^", "D", "v"]


def plot(agg: pd.DataFrame, k_values: tuple[int, ...], out_path: Path) -> None:
    """One panel per K in `k_values` (at most four, lettered A-D), sharing
    axes and one legend."""
    if not 1 <= len(k_values) <= len(_PANEL_LETTERS):
        raise ValueError(f"need 1-{len(_PANEL_LETTERS)} K values, got {k_values}")
    agg = agg[agg.k.isin(k_values)].copy()
    agg["panel"] = agg.k.map(lambda k: f"K={k}")
    panel_order = [f"K={k}" for k in k_values]
    weights = sorted(agg.bm25_weight.unique())
    facets = facet_order(agg)
    palette = _ARM_PALETTE[: len(facets)]
    markers = _ARM_MARKERS[: len(facets)]
    dashes = [(1, 0)] * min(len(facets), 4) + [(4, 2)] * (len(facets) - 4)

    sns.set_theme(style="whitegrid", context="paper")
    g = sns.relplot(
        data=agg,
        x="bm25_weight",
        y="reciprocal_rank_at_k",
        hue="facet",
        hue_order=facets,
        palette=palette,
        style="facet",
        style_order=facets,
        markers=markers,
        dashes=dashes,
        col="panel",
        col_order=panel_order,
        col_wrap=min(2, len(k_values)),
        kind="line",
        markersize=5,
        linewidth=1.5,
        facet_kws={"sharex": True, "sharey": True},
    )
    g.figure.set_size_inches(_FIGSIZE_CM * _CM, _FIGSIZE_CM * _CM)
    g.set(xlim=(-0.05, 1.05), ylim=(0, 1), xticks=weights)
    g.set_titles("{col_name}")
    for letter, ax in zip(_PANEL_LETTERS, g.axes.flat, strict=False):
        ax.set_xticklabels([f"{w:.1f}" for w in weights])
        ax.text(-0.12, 1.06, letter, transform=ax.transAxes, fontsize=12, fontweight="bold")
    g.set_axis_labels("BM25 score weight", "Mean Reciprocal Rank (MRR)")
    sns.move_legend(
        g,
        "lower center",
        ncol=2,
        title="Level 1 / Level 2 arm",
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
    )
    g.figure.tight_layout(rect=(0, 0.1, 1, 1))
    g.figure.savefig(out_path, dpi=_DPI)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="results/ablation/<run_id> directory")
    parser.add_argument("--out", type=Path, default=None, help="output PNG path")
    parser.add_argument(
        "--k-values",
        type=str,
        default=_DEFAULT_K_VALUES,
        help="one panel per K, lettered A-D in this order (default: 4,6,8,10)",
    )
    parser.add_argument(
        "--bm25-weight-values", type=str, default=None, help="comma-separated, e.g. 0.0,0.5,1.0"
    )
    parser.add_argument(
        "--scorer",
        choices=tuple(SCORER_SUFFIX),
        default=None,
        help="lexical scorer rows to plot when a run has two (DEVIATIONS.md #311); "
        "default: the run's only scorer, or TF x IDF",
    )
    args = parser.parse_args(argv)
    if args.bm25_weight_values is not None:
        os.environ["ABLATION_BM25_WEIGHT_VALUES"] = args.bm25_weight_values

    k_values = tuple(int(x) for x in args.k_values.split(",") if x.strip())
    agg = load_mean_recall(
        args.run_dir / "per_query_results.jsonl",
        bm25_weight_values(),
        metric="reciprocal_rank_at_k",
    )
    missing = sorted(set(k_values) - set(agg.k))
    if missing:
        raise SystemExit(f"no rows in {args.run_dir} for k value(s) {missing}")

    agg, suffix_as = select_scorer(agg, args.scorer)
    out_path = args.out or args.run_dir / with_suffix(_DEFAULT_OUT_NAME, suffix_as)
    plot(agg, k_values, out_path)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
