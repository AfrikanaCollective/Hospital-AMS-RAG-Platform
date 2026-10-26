"""CLI: python -m scripts.plot_recall_by_bm25_weight RUN_DIR
[--out FILE] [--bm25-weight-values 0.0,0.2,...] [--metric recall|hit]
(PRD-112 / ARCH-043 — supplementary figure for a unified ablation run).

Reads `RUN_DIR/per_query_results.jsonl` written by
`scripts/run_unified_ablation.py` and renders one seaborn figure of mean
Recall@K vs K, one line per Level-3 BM25 score weight, faceted 2x2 over the
Level-1 x Level-2 arms (Present-only/All-assessed x Raw/Enriched, plus
"Area question only / Raw" for runs since DEVIATIONS.md #311). Output is
an 18 x 18 cm (18 x 27 cm with the fifth arm), 300 dpi PNG written next to the input
(`recall_at_k_by_bm25_weight.png` by default).

`--metric hit` plots Hit@K instead (any gold chunk in the top K;
`hit_at_k_by_bm25_weight.png`, DEVIATIONS.md #269). Recall@K is the
fraction of a multi-chunk gold set retrieved, so it's capped below 1.0
while K < gold-set size; Hit@K is the metric single-gold-chunk "recall at
k" curves in the literature report.

Only the configured `bm25_weight` grid is plotted
(`app.eval.ablation_config.bm25_weight_values()`, i.e.
`ABLATION_BM25_WEIGHT_VALUES`, or `--bm25-weight-values`) — rows at any
other weight are dropped, so a run made under an older, finer grid can be
re-plotted at the current one (proposal §14, DEVIATIONS.md #207). Fails if
a requested weight has no rows in the run.

`--scorer okapi_bm25_sapbert` plots the Okapi BM25 rows (#311) to a
`_okapi_bm25`-suffixed file; the default is the original TF x IDF scorer.

Purely a read of an existing run's file output — no DB, Qdrant, or model
calls. Requires the `retrieval-tuning` optional extra
(seaborn/pandas/matplotlib).
"""

from __future__ import annotations

import argparse
import json
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
_HIT_OUT_NAME = "hit_at_k_by_bm25_weight.png"
# Per-query column -> axis label, for the metrics these figures plot.
METRIC_LABELS = {
    "recall_at_k": "Recall at K",  # y-axis label, DEVIATIONS.md #321
    "hit_at_k": "Hit@K (any gold chunk in top K)",
}

_LEVEL1_LABELS = {
    "present_only": "Present-only",
    "all_assessed": "All-assessed",
    "area_question": "Area question only",  # DEVIATIONS.md #311
}
_LEVEL2_LABELS = {"raw": "Raw", "enriched": "Enriched"}
_FACET_ORDER = [
    "Present-only / Raw",
    "Present-only / Enriched",
    "All-assessed / Raw",
    "All-assessed / Enriched",
    "Area question only / Raw",
]
# Lexical scorer per `level3_condition` (DEVIATIONS.md #311). Rows written
# before #311 have only the original scorer.
DEFAULT_SCORER = "bm25_sapbert"
SCORER_SUFFIX = {"bm25_sapbert": "", "okapi_bm25_sapbert": "_okapi_bm25"}
SCORER_TITLE = {"bm25_sapbert": "TF x IDF lexical scorer", "okapi_bm25_sapbert": "Okapi BM25"}


def facet_order(agg: pd.DataFrame) -> list[str]:
    """The arms present in `agg`, in the fixed order: a run written before
    DEVIATIONS.md #311 has no area-question arm."""
    present = set(agg.facet)
    return [f for f in _FACET_ORDER if f in present]


def figure_height_cm(n_facets: int, ncols: int = 2) -> float:
    """18 cm for up to two rows of panels; 9 cm more per extra row, so the
    five-arm figures (#311) keep the panel size of the four-arm ones."""
    rows = -(-n_facets // ncols)
    return _FIGSIZE_CM * max(rows, 2) / 2


def select_scorer(agg: pd.DataFrame, requested: str | None) -> tuple[pd.DataFrame, str]:
    """The rows of one lexical scorer, and the scorer whose file-name suffix
    to use: `requested`, else the run's only scorer, else TF x IDF. A
    one-scorer run keeps the plain file names (DEVIATIONS.md #312)."""
    present = sorted(set(agg.level3_condition))
    scorer = requested or (present[0] if len(present) == 1 else DEFAULT_SCORER)
    if scorer not in present:
        raise SystemExit(f"no rows for lexical scorer {scorer!r}; the run has {present}")
    return agg[agg.level3_condition == scorer], scorer if len(present) > 1 else DEFAULT_SCORER


def with_suffix(name: str, scorer: str) -> str:
    """`recall_at_k_by_bm25_weight.png` -> `..._okapi_bm25.png` for the
    Okapi scorer; unchanged for the original one."""
    stem, dot, ext = name.rpartition(".")
    return f"{stem}{SCORER_SUFFIX[scorer]}{dot}{ext}"


def load_mean_recall(
    per_query_path: Path, weights: tuple[float, ...], metric: str = "recall_at_k"
) -> pd.DataFrame:
    """Mean `metric` (a per-query column: `recall_at_k`, `hit_at_k` or
    `reciprocal_rank_at_k`) per (level1, level2, k, bm25_weight) cell,
    restricted to `weights`."""
    return aggregate(read_per_query(per_query_path), weights, metric, source=per_query_path)


def read_per_query(per_query_path: Path) -> pd.DataFrame:
    """Only the columns the figures use: a full run's jsonl is ~1 GB, and
    `plot_ablation_figures.render_all` aggregates it several times from one
    read (DEVIATIONS.md #229)."""
    columns = (
        "level1_condition",
        "level2_condition",
        "k",
        "bm25_weight",
        "recall_at_k",
        "reciprocal_rank_at_k",
    )
    # Streamed line by line, keeping only these columns: `pd.read_json`
    # would first materialize every row's query text and id lists.
    data: dict[str, list] = {c: [] for c in (*columns, "hit_at_k", "level3_condition")}
    with per_query_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            for c in columns:
                data[c].append(row[c])
            data["hit_at_k"].append(_row_hit_at_k(row))
            data["level3_condition"].append(row.get("level3_condition", DEFAULT_SCORER))
    df = pd.DataFrame(data)
    df["bm25_weight"] = df["bm25_weight"].astype(float)
    return df


def _row_hit_at_k(row: dict) -> float:
    """The row's `hit_at_k`; rows written before that field existed
    (DEVIATIONS.md #269) derive it from `first_relevant_rank`, the rank of
    the first gold chunk in the full ranking — a hit at K iff it is <= K."""
    if row.get("hit_at_k") is not None:
        return float(row["hit_at_k"])
    rank = row.get("first_relevant_rank")
    return 1.0 if rank is not None and rank <= row["k"] else 0.0


def aggregate(
    df: pd.DataFrame, weights: tuple[float, ...], metric: str, *, source: Path | str = "rows"
) -> pd.DataFrame:
    df = df[df.bm25_weight.round(6).isin([round(w, 6) for w in weights])]
    missing = sorted(set(round(w, 6) for w in weights) - set(df.bm25_weight.round(6)))
    if missing:
        raise SystemExit(f"no rows in {source} for bm25_weight(s) {missing}")
    agg = (
        df.groupby(
            ["level3_condition", "level1_condition", "level2_condition", "k", "bm25_weight"]
        )[metric]
        .mean()
        .reset_index()
    )
    agg["facet"] = (
        agg.level1_condition.map(_LEVEL1_LABELS) + " / " + agg.level2_condition.map(_LEVEL2_LABELS)
    )
    agg["bm25_weight_label"] = agg.bm25_weight.map(lambda w: f"{w:.1f}")
    return agg


def plot(agg: pd.DataFrame, out_path: Path, metric: str = "recall_at_k") -> None:
    weights = sorted(agg.bm25_weight.unique())
    hue_order = [f"{w:.1f}" for w in weights]
    k_values = sorted(agg.k.unique())
    facets = facet_order(agg)

    sns.set_theme(style="whitegrid", context="paper")
    g = sns.relplot(
        data=agg,
        x="k",
        y=metric,
        hue="bm25_weight_label",
        hue_order=hue_order,
        palette=sns.color_palette("crest", len(hue_order)),
        col="facet",
        col_order=facets,
        col_wrap=2,
        kind="line",
        marker="o",
        markersize=4,
        linewidth=1.5,
        facet_kws={"sharex": True, "sharey": True},
    )
    height = figure_height_cm(len(facets))
    g.figure.set_size_inches(_FIGSIZE_CM * _CM, height * _CM)
    g.set(
        xlim=(min(k_values) - 1, max(k_values) + 1),
        ylim=(0, 1),
        xticks=k_values,
    )
    g.set_titles("{col_name}")
    g.set_axis_labels("Number of context hits, K", METRIC_LABELS[metric])
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
    g.figure.tight_layout(rect=(0, 0.07 * _FIGSIZE_CM / height, 1, 1))
    g.figure.savefig(out_path, dpi=_DPI)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="results/ablation/<run_id> directory")
    parser.add_argument("--out", type=Path, default=None, help="output PNG path")
    parser.add_argument(
        "--bm25-weight-values", type=str, default=None, help="comma-separated, e.g. 0.0,0.5,1.0"
    )
    parser.add_argument(
        "--metric",
        choices=("recall", "hit"),
        default="recall",
        help="recall: Recall@K (fraction of gold); hit: Hit@K (any gold in top K)",
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
        # Same env var `app.config.Settings` reads -- one source of truth,
        # as in `run_unified_ablation._parse_values`.
        os.environ["ABLATION_BM25_WEIGHT_VALUES"] = args.bm25_weight_values

    metric = f"{args.metric}_at_k"
    weights = bm25_weight_values()
    agg = load_mean_recall(args.run_dir / "per_query_results.jsonl", weights, metric)
    agg, suffix_as = select_scorer(agg, args.scorer)
    out_name = with_suffix(_HIT_OUT_NAME if args.metric == "hit" else _DEFAULT_OUT_NAME, suffix_as)
    out_path = args.out or args.run_dir / out_name
    plot(agg, out_path, metric)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
