"""Render the ablation number tables for `manuscript/numbers.md`.

Reads only aggregate outputs: `manuscript/derived/ablation_numbers.json`
(from `ablation_numbers.py`) and the run's `area_reference_scores.csv` and
`arm_vs_popularity_baseline.csv`. Writes Markdown to stdout. Standard
library only, so it runs on the host:

    python3 -I manuscript/scripts/render_ablation_tables.py \
      manuscript/derived/ablation_numbers.json backend/results/ablation/<run_id> \
      > manuscript/derived/ablation_tables.md

p-values come from a 10,000-resample paired bootstrap, so an estimated
p of 0 is reported as p<0.001.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ARMS = [
    ("present_only", "raw"),
    ("present_only", "enriched"),
    ("all_assessed", "raw"),
    ("all_assessed", "enriched"),
]
ARM_LABEL = {
    ("present_only", "raw"): "Present only, raw",
    ("present_only", "enriched"): "Present only, enriched",
    ("all_assessed", "raw"): "All assessed, raw",
    ("all_assessed", "enriched"): "All assessed, enriched",
}
AREAS = ["assessment", "investigations", "severity_classification", "antibiotic_course"]
AREA_LABEL = {
    "assessment": "Assessment",
    "investigations": "Investigations",
    "severity_classification": "Severity classification",
    "antibiotic_course": "Antibiotic course",
}


def est(s: dict) -> str:
    return f"{s['mean']:.3f} ({s['ci_low']:.3f} to {s['ci_high']:.3f})"


def p_fmt(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def delta(s: dict) -> str:
    return (
        f"{s['mean_delta']:+.3f} ({s['ci_low']:+.3f} to {s['ci_high']:+.3f}) | "
        f"{p_fmt(s['p_value'])}"
    )


def weight_header(weights: list[float]) -> list[str]:
    head = "| Arm | " + " | ".join(f"w = {w:.1f}" for w in weights) + " |"
    return [head, "|---" * (len(weights) + 1) + "|"]


def curve_table(curves: list[dict], weights: list[float]) -> list[str]:
    out = weight_header(weights)
    for l1, l2 in ARMS:
        c = next(c for c in curves if c["level1"] == l1 and c["level2"] == l2)
        by_w = {p["bm25_weight"]: p["score"] for p in c["points"]}
        out.append(
            f"| {ARM_LABEL[(l1, l2)]} | "
            + " | ".join(est(by_w[w]) for w in weights)
            + " |"
        )
    return out


def main(argv: list[str]) -> int:
    d = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    run_dir = Path(argv[2])
    pooled = d["pooled"]
    weights = [p["bm25_weight"] for p in pooled["level3_pooled_by_weight"]]
    k = d["headline"]["k"]
    o: list[str] = [
        f"<!-- generated from run {d['run_id']}; do not edit by hand -->",
        "",
    ]

    g = d["gold"]
    o += [
        "### G. Gold sets (own-citation gold, one per question)",
        "",
        f"- Questions: {g['n_queries']}; gold-set size mean {g['gold_size_mean']:.2f}, "
        f"median {g['gold_size_median']:g}, range {g['gold_size_min']}–{g['gold_size_max']}",
        f"- Distinct chunks in any gold set: {g['distinct_chunks']}; distinct gold sets: "
        f"{g['distinct_gold_sets']}",
        "",
        "| Area | n | Distinct chunks | Gold size mean | Gold size median | "
        "Most-cited chunk in % of gold sets |",
        "|---|---|---|---|---|---|",
    ]
    for a in AREAS:
        x = g["by_area"][a]
        o.append(
            f"| {AREA_LABEL[a]} | {x['n_queries']} | {x['distinct_chunks']} | "
            f"{x['gold_size_mean']:.2f} | {x['gold_size_median']:g} | "
            f"{100 * x['top_chunk_share']:.2f} |"
        )

    o += ["", f"### A. MRR@{k} by arm and BM25 weight w (mean, 95% CI), n = 933", ""]
    o += curve_table(pooled["level3_curve"], weights)
    o += ["", "Pooled over the four arms (each question averaged over arms first):", ""]
    o += ["| | " + " | ".join(f"w = {w:.1f}" for w in weights) + " |"]
    o += ["|---" * (len(weights) + 1) + "|"]
    o += [
        "| All arms | "
        + " | ".join(est(p["score"]) for p in pooled["level3_pooled_by_weight"])
        + " |"
    ]

    o += [
        "",
        f"### B. Paired contrasts, MRR@{k} (mean difference, 95% CI | p)",
        "",
        "| Contrast | Pooled over | Difference (95% CI) | p |",
        "|---|---|---|---|",
        f"| Level 1: present only − all assessed | L2 × w | {delta(pooled['level1']['delta'])} |",
    ]
    for l1 in ("present_only", "all_assessed"):
        o.append(
            f"| Level 2: enriched − raw, {ARM_LABEL[(l1, 'raw')].split(',')[0].lower()} | w | "
            f"{delta(pooled['level2'][l1]['delta'])} |"
        )
    for c in pooled["level3_curve"]:
        o.append(
            f"| Level 3: w = 1.0 − w = 0.0, {ARM_LABEL[(c['level1'], c['level2'])]} | — | "
            f"{delta(c['endpoints_delta'])} |"
        )
    b = d["best_weight_vs_bm25"]
    o.append(
        f"| Post hoc: best w (= {b['selected_weight']}, chosen on these data) − w = 1.0 | "
        f"L1 × L2 | {delta(b['delta'])} |"
    )

    pop = d["popularity_baseline"]
    o += [
        "",
        f"### C. Arms vs the popularity baseline, MRR@{k}",
        "",
        f"Baseline (query-blind, leave-one-out): {est(pop['baseline']['mrr'])}",
        "",
        "| Arm | w | Arm − baseline (95% CI) | p |",
        "|---|---|---|---|",
    ]
    for x in pop["arm_vs_baseline"]["mrr"]:
        l1, l2, w = x["label_a"].split("/")
        o.append(f"| {ARM_LABEL[(l1, l2)]} | {w.split('=')[1]} | {delta(x)} |")

    o += [
        "",
        f"### D. By query area, MRR@{k} (exploratory)",
        "",
        "Pooled over the four arms, by w (mean, 95% CI):",
        "",
    ]
    o += ["| Area | " + " | ".join(f"w = {w:.1f}" for w in weights) + " |"]
    o += ["|---" * (len(weights) + 1) + "|"]
    for a in AREAS:
        pts = d["by_area"][a]["level3_pooled_by_weight"]
        o.append(
            f"| {AREA_LABEL[a]} | " + " | ".join(est(p["score"]) for p in pts) + " |"
        )
    o += [
        "",
        "| Area | L1: present − all assessed | p | L2 enriched − raw (present only) | p | "
        "L2 enriched − raw (all assessed) | p |",
        "|---|---|---|---|---|---|---|",
    ]
    for a in AREAS:
        A = d["by_area"][a]
        o.append(
            f"| {AREA_LABEL[a]} | {delta(A['level1']['delta'])} | "
            f"{delta(A['level2']['present_only']['delta'])} | "
            f"{delta(A['level2']['all_assessed']['delta'])} |"
        )

    ref = defaultdict(dict)
    with (run_dir / "area_reference_scores.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if int(r["k"]) == k and r["metric"] == "mrr":
                ref[(r["level1"], r["level2"])][float(r["bm25_weight"])] = r
    o += [
        "",
        f"### E. Sensitivity: PI-curated area reference sets as gold, MRR@{k} (mean, 95% CI)",
        "",
    ]
    o += weight_header(weights)
    for arm in ARMS:
        o.append(
            f"| {ARM_LABEL[arm]} | "
            + " | ".join(
                f"{float(ref[arm][w]['mean']):.3f} ({float(ref[arm][w]['ci_low']):.3f} to "
                f"{float(ref[arm][w]['ci_high']):.3f})"
                for w in weights
            )
            + " |"
        )

    pop_rows = defaultdict(list)
    with (run_dir / "arm_vs_popularity_baseline.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["metric"] == "mrr":
                pop_rows[int(r["k"])].append(r)
    o += [
        "",
        "### F. Sensitivity: MRR across K (from `arm_vs_popularity_baseline.csv`)",
        "",
        "| K | Baseline MRR | Arm MRR range, w ≥ 0.6 | Best arm (MRR) | "
        "Arms above / below / not different from baseline (of 24) |",
        "|---|---|---|---|---|",
    ]
    for kk in sorted(pop_rows):
        rs = pop_rows[kk]
        hi = [float(r["arm_mean"]) for r in rs if float(r["bm25_weight"]) >= 0.6]
        best = max(rs, key=lambda r: float(r["arm_mean"]))
        count = {v: sum(r["vs_baseline"] == v for r in rs) for v in ("above", "below")}
        ns = len(rs) - count["above"] - count["below"]
        o.append(
            f"| {kk} | {float(rs[0]['baseline_mean']):.3f} | {min(hi):.3f}–{max(hi):.3f} | "
            f"{ARM_LABEL[(best['level1'], best['level2'])]}, w = {float(best['bm25_weight']):.1f} "
            f"({float(best['arm_mean']):.3f}) | {count['above']} / {count['below']} / {ns} |"
        )

    o += ["", f"### H. Secondary metric: Recall@{k} by arm and w (mean, 95% CI)", ""]
    o += curve_table(d["secondary_recall_curve"], weights)
    print("\n".join(o))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
