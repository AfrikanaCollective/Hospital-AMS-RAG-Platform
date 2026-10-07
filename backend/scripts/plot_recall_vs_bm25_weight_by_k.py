"""CLI: python -m scripts.plot_recall_vs_bm25_weight_by_k RUN_DIR
[--out FILE] [--k-values 4,6,8,10,12,14] [--bm25-weight-values 0.0,0.2,...]
[--metric recall|hit]
(PRD-112 / ARCH-043 — supplementary figure for a unified ablation run).

Companion to `scripts/plot_recall_by_bm25_weight.py` with the axes swapped:
mean Recall@K vs. BM25 score weight on x, one line per K, faceted 2x2 over
the Level-1 x Level-2 arms (Present-only/All-assessed x Raw/Enriched).
Output is an 18 x 18 cm, 300 dpi PNG written next to the input
(`recall_at_k_vs_bm25_weight_by_k.png` by default). `--metric hit` plots
Hit@K instead (`hit_at_k_vs_bm25_weight_by_k.png`, DEVIATIONS.md #269).
Both metrics draw K lines 4,6,8,10,12,14 by default (operator request,
DEVIATIONS.md #274).

The weight grid is the configured one (`bm25_weight_values()`, same as the
companion script). `--k-values` picks which K lines to draw; its default
(4,6,8,10,12,14) is a presentation choice for
this figure, not the sweep's K grid, which stays config-driven
(`ABLATION_K_VALUES`). Fails if a requested K has no rows in the run.

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
    _FACET_ORDER,
    _FIGSIZE_CM,
    METRIC_LABELS,
    load_mean_recall,
)

_DEFAULT_OUT_NAME = "recall_at_k_vs_bm25_weight_by_k.png"
_HIT_OUT_NAME = "hit_at_k_vs_bm25_weight_by_k.png"
_DEFAULT_K_VALUES = "4,6,8,10,12,14"


def plot(
    agg: pd.DataFrame, k_values: tuple[int, ...], out_path: Path, metric: str = "recall_at_k"
) -> None:
    agg = agg[agg.k.isin(k_values)].copy()
    agg["k_label"] = agg.k.astype(str)
    hue_order = [str(k) for k in k_values]
    weights = sorted(agg.bm25_weight.unique())

    sns.set_theme(style="whitegrid", context="paper")
    g = sns.relplot(
        data=agg,
        x="bm25_weight",
        y=metric,
        hue="k_label",
        hue_order=hue_order,
        palette=sns.color_palette("flare", len(hue_order)),
        col="facet",
        col_order=_FACET_ORDER,
        col_wrap=2,
        kind="line",
        marker="o",
        markersize=5,
        linewidth=1.5,
        facet_kws={"sharex": True, "sharey": True},
    )
    g.figure.set_size_inches(_FIGSIZE_CM * _CM, _FIGSIZE_CM * _CM)
    g.set(xlim=(-0.05, 1.05), ylim=(0, 1), xticks=weights)
    for ax in g.axes.flat:
        ax.set_xticklabels([f"{w:.1f}" for w in weights])
    g.set_titles("{col_name}")
    g.set_axis_labels("BM25 score weight", METRIC_LABELS[metric])
    sns.move_legend(
        g,
        "lower center",
        ncol=len(hue_order),
        title="Number of context hits, K",
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
    )
    g.figure.tight_layout(rect=(0, 0.07, 1, 1))
    g.figure.savefig(out_path, dpi=_DPI)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="results/ablation/<run_id> directory")
    parser.add_argument("--out", type=Path, default=None, help="output PNG path")
    parser.add_argument(
        "--k-values",
        type=str,
        default="",
        help="K lines (default: 4,6,8,10,12,14)",
    )
    parser.add_argument(
        "--bm25-weight-values", type=str, default=None, help="comma-separated, e.g. 0.0,0.5,1.0"
    )
    parser.add_argument(
        "--metric",
        choices=("recall", "hit"),
        default="recall",
        help="recall: Recall@K (fraction of gold); hit: Hit@K (any gold in top K)",
    )
    args = parser.parse_args(argv)
    if args.bm25_weight_values is not None:
        os.environ["ABLATION_BM25_WEIGHT_VALUES"] = args.bm25_weight_values

    k_values = tuple(int(x) for x in (args.k_values or _DEFAULT_K_VALUES).split(",") if x.strip())
    metric = f"{args.metric}_at_k"
    agg = load_mean_recall(args.run_dir / "per_query_results.jsonl", bm25_weight_values(), metric)
    missing = sorted(set(k_values) - set(agg.k))
    if missing:
        raise SystemExit(f"no rows in {args.run_dir} for k value(s) {missing}")

    out_name = _HIT_OUT_NAME if args.metric == "hit" else _DEFAULT_OUT_NAME
    out_path = args.out or args.run_dir / out_name
    plot(agg, k_values, out_path, metric)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
