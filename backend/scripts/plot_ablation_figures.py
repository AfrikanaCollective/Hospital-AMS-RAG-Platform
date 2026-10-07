"""Render the unified-ablation figures for one run directory
(PRD-112 / ARCH-043; DEVIATIONS.md #229, #269):

- `recall_at_k_by_bm25_weight.png` (`scripts.plot_recall_by_bm25_weight`)
- `recall_at_k_vs_bm25_weight_by_k.png` (`scripts.plot_recall_vs_bm25_weight_by_k`)
- `hit_at_k_by_bm25_weight.png` and `hit_at_k_vs_bm25_weight_by_k.png`: the
  same two figures for Hit@K (any gold chunk in the top K), drawn alongside
  Recall@K because Recall@K is capped below 1.0 while K < gold-set size
- `mrr_at_k_vs_bm25_weight_by_arm.png` (`scripts.plot_mrr_vs_bm25_weight_by_arm`):
  panels A-D for K=2, 4, 6, 8 (DEVIATIONS.md #275/#277)

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


def render_all(run_dir: Path, *, weights: tuple[float, ...]) -> list[Path]:
    """Writes the PNGs into `run_dir` and returns their paths.

    Each by-K figure draws the default K lines 4,6,...,14 (DEVIATIONS.md
    #274), restricted to the K values this run actually has; if it has none
    of them, that figure is skipped with a message instead of failing the
    run.
    """
    df = recall_by_weight.read_per_query(run_dir / "per_query_results.jsonl")
    rr = recall_by_weight.aggregate(df, weights, "reciprocal_rank_at_k", source=run_dir)
    written: list[Path] = []
    wanted = tuple(int(x) for x in recall_by_k._DEFAULT_K_VALUES.split(","))

    for metric, by_weight_name, by_k_name in (
        ("recall_at_k", recall_by_weight._DEFAULT_OUT_NAME, recall_by_k._DEFAULT_OUT_NAME),
        ("hit_at_k", recall_by_weight._HIT_OUT_NAME, recall_by_k._HIT_OUT_NAME),
    ):
        agg = recall_by_weight.aggregate(df, weights, metric, source=run_dir)
        out = run_dir / by_weight_name
        recall_by_weight.plot(agg, out, metric)
        written.append(out)

        ks = tuple(k for k in wanted if k in set(agg.k))
        if ks:
            out = run_dir / by_k_name
            recall_by_k.plot(agg, ks, out, metric)
            written.append(out)
        else:
            print(f"[ablation-figures] skipped {by_k_name}: the run has none of K={wanted}")

    mrr_wanted = tuple(int(x) for x in mrr_by_arm._DEFAULT_K_VALUES.split(","))
    mrr_ks = tuple(k for k in mrr_wanted if k in set(rr.k))
    if mrr_ks:
        out = run_dir / mrr_by_arm._DEFAULT_OUT_NAME
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
