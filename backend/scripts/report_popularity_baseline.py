"""CLI: python -m scripts.report_popularity_baseline RUN_DIR
(PRD-112 / ARCH-043; DEVIATIONS.md #272, #278).

Reports every Level 1 x Level 2 x bm25_weight arm against the question-blind
popularity baseline (`app.eval.unified_ablation.summary`) at every K in the
run, on Recall@K, MRR@K and Hit@K:

- `arm_vs_popularity_baseline.csv` in the run directory: one row per
  K x metric x arm, with arm mean, baseline mean, paired delta, 95% CI,
  p-value and an above / below / not significant verdict.
- `statistical_summary.json["popularity_baseline_by_k"]`: the same, as the
  `PopularityBaseline` records. No other key is changed.

`scripts.run_unified_ablation` calls `write_popularity_report` at the end of
every run; this CLI backfills runs written before it existed. Purely a read
of the run's own files: no DB, Qdrant or model calls.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from app.eval.unified_ablation.per_query import PerQueryResult, read_per_query_results
from app.eval.unified_ablation.summary import (
    PopularityBaseline,
    popularity_report_rows,
    summarize_popularity_baseline_by_k,
)

REPORT_NAME = "arm_vs_popularity_baseline.csv"


def write_popularity_report(
    run_dir: Path,
    rows: list[PerQueryResult],
    *,
    k_values: tuple[int, ...],
    weights: tuple[float, ...],
) -> Path:
    """Writes the CSV and adds `popularity_baseline_by_k` to the run's
    `statistical_summary.json` (created if missing). Returns the CSV path."""
    baselines = summarize_popularity_baseline_by_k(rows, k_values=k_values, weight_values=weights)
    records = popularity_report_rows(baselines)
    out = run_dir / REPORT_NAME
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    summary_path = run_dir / "statistical_summary.json"
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    )
    summary["popularity_baseline_by_k"] = [asdict(b) for b in baselines]
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _print_report(baselines)
    return out


def _print_report(baselines: list[PopularityBaseline]) -> None:
    """Per K: the baseline, and how many arms are above / below it."""
    for pb in baselines:
        b = pb.baseline
        counts = []
        for metric, deltas in pb.arm_vs_baseline.items():
            above = sum(d.ci_low > 0 for d in deltas)
            below = sum(d.ci_high < 0 for d in deltas)
            counts.append(f"{metric} {above} above / {below} below")
        print(
            f"[popularity-baseline] K={pb.k}: baseline Recall {b['recall'].mean:.3f}, "
            f"MRR {b['mrr'].mean:.3f}, Hit {b['hit'].mean:.3f}; arms: " + ", ".join(counts)
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="results/ablation/<run_id> directory")
    args = parser.parse_args(argv)
    config = json.loads((args.run_dir / "configuration.json").read_text(encoding="utf-8"))
    rows = read_per_query_results(args.run_dir / "per_query_results.jsonl")
    out = write_popularity_report(
        args.run_dir,
        rows,
        k_values=tuple(config["k_values"]),
        weights=tuple(float(w) for w in config["bm25_weight_values"]),
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
