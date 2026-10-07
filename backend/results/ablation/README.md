# `results/ablation/` — unified hierarchical ablation run output

Written by `python -m scripts.run_unified_ablation` (PRD-112 / ARCH-043,
`UNIFIED-ABLATION-PROPOSAL.md` §4 point 2). One subdirectory per run:

```
results/ablation/<run_id>/
├── configuration.json       # reproducibility snapshot (model ids/versions,
│                             K/alpha grids, seed, timestamp, template
│                             version, vocabulary-attestation status,
│                             `concepts_sha256` of the vocabulary file
│                             as loaded)
├── per_query_results.jsonl  # one JSON line per (query, level1, level2,
│                             level3, alpha, k) row — see
│                             app.eval.unified_ablation.per_query.PerQueryResult
│                             for the exact field list
└── statistical_summary.json # Level 1/2/3 deltas with bootstrap CIs/p-values,
                              full Recall@k x bm25_weight grid (no combined
                              PNG report since DEVIATIONS.md #212; runs
                              before that also contain
                              unified_ablation_report.png)
```

Optional supplementary figure (reads only `per_query_results.jsonl`;
`make recall-by-bm25-weight-plot [RUN_ID=<run_id>]` from the repo root, or):
`python -m scripts.plot_recall_by_bm25_weight results/ablation/<run_id>`
writes `recall_at_k_by_bm25_weight.png` (18 x 18 cm, 300 dpi) — mean
Recall@K vs K, one line per BM25 weight, faceted over the four
Level-1 x Level-2 arms.
`python -m scripts.plot_recall_vs_bm25_weight_by_k results/ablation/<run_id>
[--k-values 8,10,12,14]` (or `make recall-vs-bm25-weight-by-k-plot
[RUN_ID=<run_id>] [K_VALUES=8,10,12,14]` from the repo root) writes `recall_at_k_vs_bm25_weight_by_k.png` (same
size/dpi/facets): Recall@K vs. BM25 weight, one line per K.
`python -m scripts.plot_mrr_vs_bm25_weight_by_arm results/ablation/<run_id>
[--k-values 8,10,12,14]` writes `mrr_at_k_vs_bm25_weight_by_arm.png`
(18 x 18 cm, 300 dpi, panels A-D = MRR@8, @10, @12, @14): mean reciprocal
rank vs. BM25 weight, one line per Level-1 x Level-2 arm (DEVIATIONS.md #275).

**Never committed** — `per_query_results.jsonl` can run into the hundreds
of MB per run (a single 238-question run against this deployment's real
16-arm × 10-k × 6-alpha grid produced 180,880 rows / 472MB — DEVIATIONS.md
#197). `.gitignore` excludes everything under this directory except this
file and `.gitkeep`.
