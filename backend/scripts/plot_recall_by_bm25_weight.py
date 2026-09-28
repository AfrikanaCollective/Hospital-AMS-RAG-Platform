"""CLI: python -m scripts.plot_recall_by_bm25_weight RUN_DIR
[--out FILE] [--bm25-weight-values 0.0,0.2,...]
(PRD-112 / ARCH-043 — supplementary figure for a unified ablation run).

Reads `RUN_DIR/per_query_results.jsonl` written by
`scripts/run_unified_ablation.py` and renders one seaborn figure of mean
Recall@K vs K, one line per Level-3 BM25 score weight, faceted 2x2 over the
Level-1 x Level-2 arms (Present-only/All-assessed x Raw/Enriched). Output is
an 18 x 18 cm, 300 dpi PNG written next to the input
(`recall_at_k_by_bm25_weight.png` by default).

Only the configured `bm25_weight` grid is plotted
(`app.eval.ablation_config.bm25_weight_values()`, i.e.
`ABLATION_BM25_WEIGHT_VALUES`, or `--bm25-weight-values`) — rows at any
other weight are dropped, so a run made under an older, finer grid can be
re-plotted at the current one (proposal §14, DEVIATIONS.md #207). Fails if
a requested weight has no rows in the run.

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

_CM = 1 / 2.54
_FIGSIZE_CM = 18.0
_DPI = 300
_DEFAULT_OUT_NAME = "recall_at_k_by_bm25_weight.png"

_LEVEL1_LABELS = {"present_only": "Present-only", "all_assessed": "All-assessed"}
_LEVEL2_LABELS = {"raw": "Raw", "enriched": "Enriched"}
_FACET_ORDER = [
    "Present-only / Raw",
    "Present-only / Enriched",
    "All-assessed / Raw",
    "All-assessed / Enriched",
]


def load_mean_recall(
    per_query_path: Path, weights: tuple[float, ...], metric: str = "recall_at_k"
) -> pd.DataFrame:
    """Mean `metric` (a per-query column: `recall_at_k` or
    `reciprocal_rank_at_k`) per (level1, level2, k, bm25_weight) cell,
    restricted to `weights`."""
    df = pd.read_json(per_query_path, lines=True, dtype={"bm25_weight": float})
    df = df[df.bm25_weight.round(6).isin([round(w, 6) for w in weights])]
    missing = sorted(set(round(w, 6) for w in weights) - set(df.bm25_weight.round(6)))
    if missing:
        raise SystemExit(f"no rows in {per_query_path} for bm25_weight(s) {missing}")
    agg = (
        df.groupby(["level1_condition", "level2_condition", "k", "bm25_weight"])[metric]
        .mean()
        .reset_index()
    )
    agg["facet"] = (
        agg.level1_condition.map(_LEVEL1_LABELS) + " / " + agg.level2_condition.map(_LEVEL2_LABELS)
    )
    agg["bm25_weight_label"] = agg.bm25_weight.map(lambda w: f"{w:.1f}")
    return agg


def plot(agg: pd.DataFrame, out_path: Path) -> None:
    weights = sorted(agg.bm25_weight.unique())
    hue_order = [f"{w:.1f}" for w in weights]
    k_values = sorted(agg.k.unique())

    sns.set_theme(style="whitegrid", context="paper")
    g = sns.relplot(
        data=agg,
        x="k",
        y="recall_at_k",
        hue="bm25_weight_label",
        hue_order=hue_order,
        palette=sns.color_palette("crest", len(hue_order)),
        col="facet",
        col_order=_FACET_ORDER,
        col_wrap=2,
        kind="line",
        marker="o",
        markersize=4,
        linewidth=1.5,
        facet_kws={"sharex": True, "sharey": True},
    )
    g.figure.set_size_inches(_FIGSIZE_CM * _CM, _FIGSIZE_CM * _CM)
    g.set(
        xlim=(min(k_values) - 1, max(k_values) + 1),
        ylim=(0, 1),
        xticks=k_values,
    )
    g.set_titles("{col_name}")
    g.set_axis_labels("Number of context hits, K", "Recall@K")
    sns.move_legend(
        g,
        "lower center",
        ncol=len(hue_order),
        title="BM25 score weight",
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
        columnspacing=0.8,
        handlelength=1.2,
    )
    g.figure.tight_layout(rect=(0, 0.07, 1, 1))
    g.figure.savefig(out_path, dpi=_DPI)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="results/ablation/<run_id> directory")
    parser.add_argument("--out", type=Path, default=None, help="output PNG path")
    parser.add_argument(
        "--bm25-weight-values", type=str, default=None, help="comma-separated, e.g. 0.0,0.5,1.0"
    )
    args = parser.parse_args(argv)
    if args.bm25_weight_values is not None:
        # Same env var `app.config.Settings` reads -- one source of truth,
        # as in `run_unified_ablation._parse_values`.
        os.environ["ABLATION_BM25_WEIGHT_VALUES"] = args.bm25_weight_values

    out_path = args.out or args.run_dir / _DEFAULT_OUT_NAME
    weights = bm25_weight_values()
    plot(load_mean_recall(args.run_dir / "per_query_results.jsonl", weights), out_path)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
