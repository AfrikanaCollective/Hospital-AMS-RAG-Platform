"""Render the three unified-ablation figures for one run directory
(PRD-112 / ARCH-043; DEVIATIONS.md #229):

- `recall_at_k_by_bm25_weight.png` (`scripts.plot_recall_by_bm25_weight`)
- `recall_at_k_vs_bm25_weight_by_k.png` (`scripts.plot_recall_vs_bm25_weight_by_k`)
- `mrr_at_k_vs_bm25_weight_by_arm.png` (`scripts.plot_mrr_vs_bm25_weight_by_arm`)

`scripts.run_unified_ablation` calls `render_all` after writing a run's
results, so every run gets its figures. `per_query_results.jsonl` (~1 GB for
a full run) is read once and aggregated per metric, rather than once per
figure. Each figure's own script still works standalone for re-rendering.

Usage: python -m scripts.plot_ablation_figures results/ablation/<run_id>
"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.eval import ablation_config
from scripts import plot_mrr_vs_bm25_weight_by_arm as mrr_by_arm
from scripts import plot_recall_by_bm25_weight as recall_by_weight
from scripts import plot_recall_vs_bm25_weight_by_k as recall_by_k


def render_all(run_dir: Path, *, weights: tuple[float, ...], mrr_k: int) -> list[Path]:
    """Writes the three PNGs into `run_dir` and returns their paths.

    The by-K figure draws the `8,10,12,14` lines it has always defaulted to,
    restricted to the K values this run actually has; if it has none of
    them, that figure is skipped with a message instead of failing the run.
    """
    df = recall_by_weight.read_per_query(run_dir / "per_query_results.jsonl")
    recall = recall_by_weight.aggregate(df, weights, "recall_at_k", source=run_dir)
    rr = recall_by_weight.aggregate(df, weights, "reciprocal_rank_at_k", source=run_dir)
    written: list[Path] = []

    out = run_dir / recall_by_weight._DEFAULT_OUT_NAME
    recall_by_weight.plot(recall, out)
    written.append(out)

    wanted = tuple(int(x) for x in recall_by_k._DEFAULT_K_VALUES.split(","))
    ks = tuple(k for k in wanted if k in set(recall.k))
    if ks:
        out = run_dir / recall_by_k._DEFAULT_OUT_NAME
        recall_by_k.plot(recall, ks, out)
        written.append(out)
    else:
        print(
            f"[ablation-figures] skipped {recall_by_k._DEFAULT_OUT_NAME}: the run has none "
            f"of K={wanted}"
        )

    if mrr_k in set(rr.k):
        out = run_dir / mrr_by_arm._DEFAULT_OUT_NAME
        mrr_by_arm.plot(rr, mrr_k, out)
        written.append(out)
    else:
        print(f"[ablation-figures] skipped {mrr_by_arm._DEFAULT_OUT_NAME}: no rows for k={mrr_k}")
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="results/ablation/<run_id> directory")
    args = parser.parse_args(argv)
    for path in render_all(
        args.run_dir, weights=ablation_config.bm25_weight_values(), mrr_k=ablation_config.mrr_k()
    ):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
