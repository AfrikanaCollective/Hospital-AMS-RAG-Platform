"""Manuscript numbers for the retrieval ablation (PRD-112 / ARCH-043).

Re-summarises an existing unified-ablation run at the manuscript's headline
metric, MRR@10, using the ablation's own summary functions
(`app.eval.unified_ablation.summary`, same paired bootstrap, 10,000
resamples, seed 1234). Nothing is re-retrieved: the input is the run's
`per_query_results.jsonl`. Output is aggregates only (no query text, no
record ids), as JSON on stdout.

Also reports, from the same rows:
- a check that the run's own Recall@12 Level-1 summary is reproduced;
- gold-set descriptives (size, distinct chunks, concentration by area);
- Level 1/2/3 at MRR@10 within each query area (exploratory).

Run in the api container (needs `app` on the path):

    docker compose run --rm --no-deps -T -e PYTHONPATH=/app \
      -v ./backend/app:/app/app:ro -v ./backend/results:/app/results:ro \
      -v ./manuscript/scripts:/app/manuscript_scripts:ro \
      api python manuscript_scripts/ablation_numbers.py \
      results/ablation/<run_id> > manuscript/derived/ablation_numbers.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean, median

from app.eval.unified_ablation.per_query import read_per_query_results
from app.eval.unified_ablation.summary import (
    core_rows,
    summarize_best_weight_vs_bm25,
    summarize_level1,
    summarize_level2,
    summarize_level3_by_weight_and_k,
    summarize_level3_curve,
    summarize_popularity_baseline,
)

HEADLINE_K = 10
METRIC = "mrr"


def _gold_descriptives(rows: list) -> dict:
    gold: dict[str, frozenset[str]] = {}
    area_of: dict[str, str] = {}
    for r in rows:
        gold.setdefault(r.query_id, frozenset(r.relevant_ids))
        area_of.setdefault(r.query_id, r.question_area)
    sizes = [len(g) for g in gold.values()]
    by_area: dict[str, dict] = {}
    for area in sorted(set(area_of.values())):
        sets = [g for q, g in gold.items() if area_of[q] == area]
        counts = Counter(c for g in sets for c in g)
        top = counts.most_common(1)[0][1]
        by_area[area] = {
            "n_queries": len(sets),
            "distinct_chunks": len(counts),
            "gold_size_mean": mean(len(g) for g in sets),
            "gold_size_median": median(len(g) for g in sets),
            "top_chunk_share": top / len(sets),
        }
    return {
        "n_queries": len(gold),
        "gold_size_mean": mean(sizes),
        "gold_size_median": median(sizes),
        "gold_size_min": min(sizes),
        "gold_size_max": max(sizes),
        "distinct_chunks": len({c for g in gold.values() for c in g}),
        "distinct_gold_sets": len(set(gold.values())),
        "by_area": by_area,
    }


def _levels(rows: list, *, k: int, weights: tuple[float, ...]) -> dict:
    return {
        "level1": asdict(summarize_level1(rows, k=k, metric=METRIC)),
        "level2": {
            l1: asdict(s)
            for l1, s in summarize_level2(rows, k=k, metric=METRIC).items()
        },
        "level3_curve": [
            asdict(c)
            for c in summarize_level3_curve(
                rows, k=k, metric=METRIC, weight_values=weights
            )
        ],
        "level3_pooled_by_weight": [
            asdict(p)
            for p in summarize_level3_by_weight_and_k(
                rows, k_values=(k,), metric=METRIC, weight_values=weights
            )
        ],
    }


def main(argv: list[str]) -> int:
    run_dir = Path(argv[1])
    config = json.loads((run_dir / "configuration.json").read_text(encoding="utf-8"))
    published = json.loads(
        (run_dir / "statistical_summary.json").read_text(encoding="utf-8")
    )
    weights = tuple(config["bm25_weight_values"])
    rows = core_rows(
        read_per_query_results(run_dir / "per_query_results.jsonl"),
        config["lexical_scorer"],
    )

    check = summarize_level1(rows, k=published["k"])
    reproduced = {
        "k": published["k"],
        "metric": "recall",
        "published_delta": published["level1"]["delta"]["mean_delta"],
        "recomputed_delta": check.delta.mean_delta,
        "match": abs(
            check.delta.mean_delta - published["level1"]["delta"]["mean_delta"]
        )
        < 1e-12,
    }

    at_k = [r for r in rows if r.k == HEADLINE_K]
    out = {
        "run_id": config["run_id"],
        "headline": {"metric": "MRR", "k": HEADLINE_K},
        "reproduction_check": reproduced,
        "gold": _gold_descriptives(at_k),
        "pooled": _levels(at_k, k=HEADLINE_K, weights=weights),
        "best_weight_vs_bm25": asdict(
            summarize_best_weight_vs_bm25(
                at_k, k=HEADLINE_K, metric=METRIC, weight_values=weights
            )
        ),
        "popularity_baseline": asdict(
            summarize_popularity_baseline(at_k, k=HEADLINE_K, weight_values=weights)
        ),
        "secondary_recall_curve": [
            asdict(c)
            for c in summarize_level3_curve(
                at_k, k=HEADLINE_K, metric="recall", weight_values=weights
            )
        ],
        "by_area": {},
    }
    by_area: dict[str, list] = defaultdict(list)
    for r in at_k:
        by_area[r.question_area].append(r)
    for area, area_rows in sorted(by_area.items()):
        out["by_area"][area] = _levels(area_rows, k=HEADLINE_K, weights=weights)
    json.dump(out, sys.stdout, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
