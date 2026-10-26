"""Render the unified-ablation figures for one run directory
(PRD-112 / ARCH-043; DEVIATIONS.md #229, #269, #320):

- `recall_at_k_by_bm25_weight.png` (`scripts.plot_recall_by_bm25_weight`)
- `recall_at_k_vs_bm25_weight_by_k.png` (`scripts.plot_recall_vs_bm25_weight_by_k`)
- `mrr_at_k_vs_bm25_weight_by_arm.png` (`scripts.plot_mrr_vs_bm25_weight_by_arm`):
  panels A-D for K=4, 6, 8, 10 (DEVIATIONS.md #275/#277/#319)

The Hit@K versions of the first two (#269) are no longer drawn at the end of
a run (operator request, #320); `--metric hit` on either script still draws
them.

`scripts.run_unified_ablation` calls `render_all` after writing a run's
results, so every run gets its figures. `per_query_results.jsonl` (~1 GB for
a full run) is read once and aggregated per metric, rather than once per
figure. Each figure's own script still works standalone for re-rendering.

A #311 run has two lexical scorers; each figure is then drawn once per
scorer, the Okapi BM25 set with an `_okapi_bm25` file-name suffix. Since #312
a run has one scorer (Okapi by default) and the figures keep their plain
names. The area-question arm, when run, is a fifth panel / line.

Usage: python -m scripts.plot_ablation_figures results/ablation/<run_id>
"""

from __future__ import annotations

import argparse
import functools
from pathlib import Path

import pandas as pd

from app.eval import ablation_config
from scripts import plot_mrr_vs_bm25_weight_by_arm as mrr_by_arm
from scripts import plot_recall_by_bm25_weight as recall_by_weight
from scripts import plot_recall_vs_bm25_weight_by_k as recall_by_k


def render_all(run_dir: Path, *, weights: tuple[float, ...]) -> list[Path]:
    """Writes the PNGs into `run_dir` and returns their paths.

    Each by-K figure draws the default K lines 4,6,...,14 (DEVIATIONS.md
    #274), restricted to the K values this run actually has; if it has none
    of them, that figure is skipped with a message instead of failing the
    run.
    """
    df = recall_by_weight.read_per_query(run_dir / "per_query_results.jsonl")
    written: list[Path] = []
    # One set per lexical scorer, the original first; the Okapi set's file
    # names carry `_okapi_bm25` (DEVIATIONS.md #311).
    scorers = sorted(
        set(df.level3_condition), key=lambda s: (s != recall_by_weight.DEFAULT_SCORER, s)
    )
    for scorer in scorers:
        # A one-scorer run (the default since #312) keeps the plain names.
        suffix_as = scorer if len(scorers) > 1 else recall_by_weight.DEFAULT_SCORER
        written += _render_scorer(df[df.level3_condition == scorer], run_dir, weights, suffix_as)
    return written


def _render_scorer(
    df: pd.DataFrame, run_dir: Path, weights: tuple[float, ...], scorer: str
) -> list[Path]:
    name = functools.partial(recall_by_weight.with_suffix, scorer=scorer)
    rr = recall_by_weight.aggregate(df, weights, "reciprocal_rank_at_k", source=run_dir)
    written: list[Path] = []
    wanted = tuple(int(x) for x in recall_by_k._DEFAULT_K_VALUES.split(","))

    for metric, by_weight_name, by_k_name in (
        ("recall_at_k", recall_by_weight._DEFAULT_OUT_NAME, recall_by_k._DEFAULT_OUT_NAME),
    ):
        agg = recall_by_weight.aggregate(df, weights, metric, source=run_dir)
        out = run_dir / name(by_weight_name)
        recall_by_weight.plot(agg, out, metric)
        written.append(out)

        ks = tuple(k for k in wanted if k in set(agg.k))
        if ks:
            out = run_dir / name(by_k_name)
            recall_by_k.plot(agg, ks, out, metric)
            written.append(out)
        else:
            print(f"[ablation-figures] skipped {by_k_name}: the run has none of K={wanted}")

    mrr_wanted = tuple(int(x) for x in mrr_by_arm._DEFAULT_K_VALUES.split(","))
    mrr_ks = tuple(k for k in mrr_wanted if k in set(rr.k))
    if mrr_ks:
        out = run_dir / name(mrr_by_arm._DEFAULT_OUT_NAME)
        mrr_by_arm.plot(rr, mrr_ks, out)
        written.append(out)
    else:
        print(
            f"[ablation-figures] skipped {mrr_by_arm._DEFAULT_OUT_NAME}: the run has none "
            f"of K={mrr_wanted}"
        )
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="results/ablation/<run_id> directory")
    args = parser.parse_args(argv)
    for path in render_all(args.run_dir, weights=ablation_config.bm25_weight_values()):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
