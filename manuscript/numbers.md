# numbers.md — every number for the manuscript, with its source

Status 2026-10-26. Scope: **Option A** (`manuscript.txt`). The clinician
evaluation is primary and has **no data yet**. The retrieval ablation is a
secondary, exploratory analysis of retrieval query construction; its headline
metric is **MRR@10**.

Conventions:
- Proportions and metrics to 3 decimals; percentages to 1 decimal.
- Intervals are 95% percentile bootstrap CIs: 10,000 resamples, seed 1234.
  Contrasts are paired, with the same questions resampled on both sides.
- p-values come from the same bootstrap. The smallest p it can resolve is
  0.0001, so an estimate of 0 is reported as p<0.001.
- No multiple-comparison correction.
- Percentages were computed by script (checks in §1), not by hand.

Ablation run: `backend/results/ablation/20261026T075410Z-e78b8cf6/` (below,
`RUN/`).

How the MRR@10 numbers were produced:
- `manuscript/scripts/ablation_numbers.py` re-summarises `RUN/per_query_results.jsonl`
  at MRR@10 with the ablation's own summary functions
  (`app/eval/unified_ablation/summary.py`). Nothing was re-retrieved and no
  analysis code was changed. Output: `manuscript/derived/ablation_numbers.json`
  (aggregates only).
- Check: the script reproduces the run's published Recall@12 Level-1 contrast
  exactly (`reproduction_check.match = true`).
- Check: its 24 arm-vs-popularity MRR@10 contrasts match
  `RUN/arm_vs_popularity_baseline.csv` (0 mismatches).
- Check: its per-arm MRR@10 means match the `current_gold_mean` column of
  `RUN/area_reference_scores.csv`.
- Tables in §5 are rendered from those files by
  `manuscript/scripts/render_ablation_tables.py` into
  `manuscript/derived/ablation_tables.md`, and are copied below. Regenerate
  them rather than editing them.

---

## 1. Data, cohort and flow

| # | Number | Meaning | Source |
|---|---|---|---|
| 1.1 | 2021-01-01 to 2024-12-31 | Collection period of the de-identified extract | `data/patient_records/deidentified/newborn_nbu_2021/DATASET.md` |
| 1.2 | 8 hospitals | CIN newborn units contributing to the extract = the development-cohort hospitals of Tuti et al. 2026 (CIN overall: 21 public county hospitals in 14 of 47 counties) | PI 2026-10-26; Tuti et al. PLoS One 2026;21(9):e0359212 |
| 1.2b | ~68 / ~42 / ~10 / ~3 / ~1% | Mortality shown to raters, Critical → Minimal: development-cohort in-hospital mortality 67.80 / 42.47 / 9.56 / 2.86 / 0.98% | `data/triage_mortality.yaml`; Tuti et al. 2026 Table 3; DEV #322 |
| 1.2c | 44,626 | Newborn-unit admissions with ≥1 pSBI sign, 8 hospitals, Jan 2021–Dec 2024 (source) | PI 2026-10-27 |
| 1.2d | 237 (0.53% of 44,626) | Excluded: >30% missing across the 14 features; 44,389 remained | PI 2026-10-27 |
| 1.2e | 7,994 (18.01% of 44,389) | One admission kept per unique combination of the 14 features | PI 2026-10-27 |
| 1.2f | 6,494 | Unique-combination admissions not sampled (7,994 − 1,500) | arithmetic |
| 1.3 | 1,500 | Records ingested, one admission snapshot each (18.76% of 7,994) | DEVIATIONS #298, #302 |
| 1.4 | 300 per category | Sampled per triage category, seed 20261013 (Critical, Substantial, Moderate, Low, Minimal) — a property of the extract, not of the source population | DATASET.md; DEV #302 |
| 1.5 | 49,850; 39 | EAV rows; variables | DEV #298 |
| 1.6 | 200 (13.33% of 1,500) | Records allocated to the clinician review queue (40 per triage category) | `auto_seed.py:1431`; `backend/results/triage_review_queue_20261014.log`; seed 20261014 |
| 1.7 | 800 | Review items (200 records × 4 query areas) | DEV #300, #303 |
| 1.8 | 1,300 (86.67%) | Records eligible for the ablation holdout | `auto_seed.py:1154` |
| 1.9 | 844 (64.92% of 1,300) | Holdout questions, first generation run, seed 20261015 | `backend/results/ablation_holdout_20261015.log`; DEV #303 |
| 1.10 | 456 | Records left after the first run | DEV #304 |
| 1.11 | 320 (70.18% of 456) | Rejected as clinical near-duplicates in the top-up | DEV #305; `diversity.py:56-130`. Near-duplicate rejections in the first run were not counted. |
| 1.12 | 136 (29.82% of 456) | Holdout questions, top-up run, seed 20261016 | `ablation_holdout_topup_20261016.log`; DEV #305 |
| 1.13 | 980 (75.38% of 1,300) | Holdout questions, one per record; no overlap with the review queue | DEV #305 |
| 1.14 | 178 / 214 / 198 / 209 / 181 (18.16 / 21.84 / 20.20 / 21.33 / 18.47%) | Triage category of the 980: Critical / Substantial / Moderate / Low / Minimal | DEV #305 (not computed for the 933) |
| 1.15 | 47 (4.80% of 980) | Excluded: the platform's answer cited no chunk, so there is no gold set; all 47 antibiotic course | `auto_seed.py:1119`; `runner.py:142` |
| 1.16 | **933** (95.20% of 980) | Analysed questions | `RUN/configuration.json` `distinct_queries_evaluated` |
| 1.17 | 234 / 233 / 233 / 233 (25.08 / 24.97 / 24.97 / 24.97%) | Analysed by area: assessment / investigations / severity classification / antibiotic course | `RUN/configuration.json` `questions_per_area` |

Totals checked by script: 44,626 − 237 = 44,389; 5 × 300 = 1,500; 7,994 − 1,500 = 6,494; 1,500 − 200 = 1,300; 1,300 − 844 = 456;
136 + 320 = 456; 844 + 136 = 980; 980 − 47 = 933; 234 + 3 × 233 = 933;
triage 178 + 214 + 198 + 209 + 181 = 980. All pass.

## 2. Guideline corpus and vocabularies

| # | Number | Meaning | Source |
|---|---|---|---|
| 2.1 | 4 | Guideline documents: WHO SBI 0–59 days (2024, excerpt); Kenya MoH Comprehensive Newborn Care Protocols (2022, excerpt); Kenya MoH National Antibiotic Use Guidelines (2024, excerpt); NICE NG195 (2021, updated 13 May 2026) | `data/excerpt_guidelines/manifest.json` |
| 2.2 | 297 | Chunks ingested | `backend/results/corpus_ingest_20261014.log`; DEV #302 |
| 2.3 | 144 | Chunks excluded by section (navigation, front matter, rationale) | manifest `exclude_sections` |
| 2.4 | 153 | Retrievable chunks = the ablation corpus | `app/eval/model_ablation/ablation.py:98`; counted live 2026-10-26 |
| 2.5 | 22 | Concepts in the operator-attested enrichment vocabulary | `data/clinical_concepts.yaml` (sha256 `ad4d0033…`) |
| 2.6 | 18 / 13 / 9 / 37 chunks (35 passages for antibiotic course) | PI-curated reference sets: assessment / investigations / severity classification / antibiotic course | `RUN/area_reference_summary.json`; attested 2026-10-21; DEV #309 |
| 2.7 | 4 | Query areas (facets of suspected neonatal infection / sepsis) | `data/query_areas.yaml` (sha256 `55793e6e…`) |

## 3. Ablation design parameters

| # | Number | Meaning | Source |
|---|---|---|---|
| 3.1 | 2 × 2 × 6 × 10 | Level 1 (present only / all assessed) × Level 2 (raw / enriched) × w × K | `RUN/configuration.json` |
| 3.2 | w ∈ {0, 0.2, 0.4, 0.6, 0.8, 1.0} | BM25 weight in min–max fusion: w × Okapi BM25 + (1 − w) × SapBERT cosine | `bm25_weight_values`, `fusion = minmax` |
| 3.3 | K ∈ {2, 4, …, 20} | Cut-offs | `k_values` |
| 3.4 | 10 | **Headline K for the manuscript (MRR@10).** The run itself was configured with Recall as the primary metric and K = 12 — see flag F1. | PI decision 2026-10-26; `configuration.json` `primary_metric`, `mrr_k` |
| 3.5 | k1 = 1.2, b = 0.75 | Okapi BM25 parameters, statistics over the 153 retrievable chunks | `lexical_scorer_detail` |
| 3.6 | 3 | Area-question repeat/weight N | `area_question_repeats`; DEV #313, #315 |
| 3.7 | 223,920 | Per-query rows (933 × 4 arms × 6 w × 10 K; checked) | `n_rows` |
| 3.8 | 10,000; 1234 | Bootstrap resamples; seed | `app/eval/bootstrap.py:36-37` |

## 4. Software, models and hardware (Methods)

| # | Value | Source |
|---|---|---|
| 4.1 | Python 3.11.16; numpy 2.4.6; scipy 1.17.1; pandas 3.0.6; matplotlib 3.11.2; torch 2.14.0; transformers 5.17.0; sentence-transformers 6.0.1; qdrant-client 1.19.0 | `pip freeze` in the `api` image, 00_inventory §6 |
| 4.2 | SapBERT `cambridgeltl/SapBERT-from-PubMedBERT-fulltext`, on CPU (Intel Core i9-14900KF) | `configuration.json`; container check 2026-10-26 |
| 4.3 | Platform retrieval (gold generation and the rated answers): abbreviation expansion; per-guideline hybrid search (bge-large-en-v1.5 dense + sparse, 40 prefetch each, RRF k = 60, 24 candidates per guideline); one bge-reranker-v2-m3 pass; keep up to the area cap per guideline (3 assessment / 4 investigations / 3 severity / 5 antibiotic course) with score ≥ 0.30; groups in priority order CNCP, MoH antibiotic, WHO, NICE; low confidence if, among the top 8 scores, max < 0.30 or < 2 scores ≥ 0.20 | `app/retrieval/hybrid.py:184-270`, `confidence.py:29`; `config.py:124-140`; `data/query_areas.yaml` `per_guideline_cap`; manifest `retrieval_priority` |
| 4.4 | qwen3.6:35b (= Qwen3.6-35B-A3B, PI 2026-10-27; label kept as qwen3.6:35b; citation: Qwen Team, April 2026, qwen.ai/blog?id=qwen3.6-35b-a3b), Q4_K_M, Ollama 0.32.14, BRIDGE LLM Gateway release/1.0, NVIDIA H200; num_ctx 32768, think = false, stream = false | PI, 2026-10-26 |
| 4.5 | temperature 0.0; top-p 1.0; top-k 1; seed 42; num_predict 4,096 | Gateway defaults (PI, 2026-10-27); platform sends no sampling params (`gateway.py:121`, agents pass none) |
| 4.6 | Generation dates 2026-10-15/16; seeds 20261014, 20261015, 20261016 | logs; DEV #303–#305 |

## 5. Ablation results (generated tables)

Arms are labelled by Level 1 (present only / all assessed) and Level 2
(raw / enriched). w = 1.0 is pure Okapi BM25; w = 0.0 is pure SapBERT. The
gold standard is each question's own-citation gold (the chunks the platform's
answer cited), except in table E.

<!-- BEGIN GENERATED: manuscript/derived/ablation_tables.md -->
<!-- generated from run 20261026T075410Z-e78b8cf6; do not edit by hand -->

### G. Gold sets (own-citation gold, one per question)

- Questions: 933; gold-set size mean 6.32, median 6, range 2–14
- Distinct chunks in any gold set: 51; distinct gold sets: 488

| Area | n | Distinct chunks | Gold size mean | Gold size median | Most-cited chunk in % of gold sets |
|---|---|---|---|---|---|
| Assessment | 234 | 18 | 6.69 | 7 | 89.32 |
| Investigations | 233 | 14 | 5.75 | 6 | 99.14 |
| Severity classification | 233 | 17 | 5.11 | 5 | 96.57 |
| Antibiotic course | 233 | 28 | 7.73 | 8 | 93.99 |

### A. MRR@10 by arm and BM25 weight w (mean, 95% CI), n = 933

| Arm | w = 0.0 | w = 0.2 | w = 0.4 | w = 0.6 | w = 0.8 | w = 1.0 |
|---|---|---|---|---|---|---|
| Present only, raw | 0.365 (0.346 to 0.384) | 0.467 (0.452 to 0.484) | 0.703 (0.685 to 0.721) | 0.809 (0.792 to 0.826) | 0.870 (0.855 to 0.886) | 0.872 (0.857 to 0.888) |
| Present only, enriched | 0.377 (0.359 to 0.395) | 0.481 (0.465 to 0.496) | 0.696 (0.677 to 0.714) | 0.864 (0.848 to 0.880) | 0.865 (0.849 to 0.881) | 0.862 (0.846 to 0.878) |
| All assessed, raw | 0.368 (0.350 to 0.386) | 0.492 (0.477 to 0.507) | 0.697 (0.679 to 0.715) | 0.810 (0.792 to 0.827) | 0.868 (0.852 to 0.884) | 0.863 (0.848 to 0.879) |
| All assessed, enriched | 0.396 (0.378 to 0.414) | 0.496 (0.481 to 0.511) | 0.697 (0.679 to 0.716) | 0.856 (0.840 to 0.872) | 0.860 (0.844 to 0.876) | 0.859 (0.842 to 0.875) |

Pooled over the four arms (each question averaged over arms first):

| | w = 0.0 | w = 0.2 | w = 0.4 | w = 0.6 | w = 0.8 | w = 1.0 |
|---|---|---|---|---|---|---|
| All arms | 0.376 (0.358 to 0.395) | 0.484 (0.469 to 0.500) | 0.698 (0.683 to 0.714) | 0.835 (0.820 to 0.849) | 0.866 (0.851 to 0.881) | 0.864 (0.849 to 0.880) |

### B. Paired contrasts, MRR@10 (mean difference, 95% CI | p)

| Contrast | Pooled over | Difference (95% CI) | p |
|---|---|---|---|
| Level 1: present only − all assessed | L2 × w | -0.003 (-0.005 to +0.000) | 0.089 |
| Level 2: enriched − raw, present only | w | +0.010 (+0.005 to +0.014) | <0.001 |
| Level 2: enriched − raw, all assessed | w | +0.011 (+0.005 to +0.017) | <0.001 |
| Level 3: w = 1.0 − w = 0.0, Present only, raw | — | +0.508 (+0.483 to +0.532) | <0.001 |
| Level 3: w = 1.0 − w = 0.0, Present only, enriched | — | +0.485 (+0.461 to +0.509) | <0.001 |
| Level 3: w = 1.0 − w = 0.0, All assessed, raw | — | +0.496 (+0.471 to +0.520) | <0.001 |
| Level 3: w = 1.0 − w = 0.0, All assessed, enriched | — | +0.463 (+0.439 to +0.487) | <0.001 |
| Post hoc: best w (= 0.8, chosen on these data) − w = 1.0 | L1 × L2 | +0.002 (-0.001 to +0.004) | 0.190 |

### C. Arms vs the popularity baseline, MRR@10

Baseline (query-blind, leave-one-out): 0.803 (0.785 to 0.820)

| Arm | w | Arm − baseline (95% CI) | p |
|---|---|---|---|
| Present only, raw | 0.0 | -0.438 (-0.462 to -0.414) | <0.001 |
| Present only, raw | 0.2 | -0.335 (-0.355 to -0.316) | <0.001 |
| Present only, raw | 0.4 | -0.100 (-0.120 to -0.079) | <0.001 |
| Present only, raw | 0.6 | +0.007 (-0.013 to +0.027) | 0.512 |
| Present only, raw | 0.8 | +0.068 (+0.045 to +0.090) | <0.001 |
| Present only, raw | 1.0 | +0.070 (+0.048 to +0.092) | <0.001 |
| Present only, enriched | 0.0 | -0.426 (-0.447 to -0.404) | <0.001 |
| Present only, enriched | 0.2 | -0.322 (-0.340 to -0.303) | <0.001 |
| Present only, enriched | 0.4 | -0.107 (-0.130 to -0.085) | <0.001 |
| Present only, enriched | 0.6 | +0.062 (+0.039 to +0.085) | <0.001 |
| Present only, enriched | 0.8 | +0.062 (+0.039 to +0.085) | <0.001 |
| Present only, enriched | 1.0 | +0.059 (+0.036 to +0.083) | <0.001 |
| All assessed, raw | 0.0 | -0.435 (-0.458 to -0.412) | <0.001 |
| All assessed, raw | 0.2 | -0.311 (-0.328 to -0.294) | <0.001 |
| All assessed, raw | 0.4 | -0.105 (-0.125 to -0.085) | <0.001 |
| All assessed, raw | 0.6 | +0.007 (-0.013 to +0.027) | 0.507 |
| All assessed, raw | 0.8 | +0.066 (+0.043 to +0.089) | <0.001 |
| All assessed, raw | 1.0 | +0.061 (+0.038 to +0.084) | <0.001 |
| All assessed, enriched | 0.0 | -0.407 (-0.427 to -0.385) | <0.001 |
| All assessed, enriched | 0.2 | -0.307 (-0.324 to -0.289) | <0.001 |
| All assessed, enriched | 0.4 | -0.105 (-0.125 to -0.085) | <0.001 |
| All assessed, enriched | 0.6 | +0.053 (+0.032 to +0.075) | <0.001 |
| All assessed, enriched | 0.8 | +0.057 (+0.034 to +0.080) | <0.001 |
| All assessed, enriched | 1.0 | +0.056 (+0.033 to +0.080) | <0.001 |

### D. By query area, MRR@10 (exploratory)

Pooled over the four arms, by w (mean, 95% CI):

| Area | w = 0.0 | w = 0.2 | w = 0.4 | w = 0.6 | w = 0.8 | w = 1.0 |
|---|---|---|---|---|---|---|
| Assessment | 0.733 (0.698 to 0.766) | 0.732 (0.699 to 0.766) | 0.732 (0.699 to 0.766) | 0.731 (0.697 to 0.764) | 0.725 (0.691 to 0.760) | 0.716 (0.681 to 0.751) |
| Investigations | 0.244 (0.230 to 0.259) | 0.437 (0.425 to 0.449) | 0.860 (0.836 to 0.882) | 0.933 (0.910 to 0.954) | 0.942 (0.919 to 0.963) | 0.945 (0.921 to 0.966) |
| Severity classification | 0.327 (0.303 to 0.353) | 0.502 (0.485 to 0.519) | 0.718 (0.690 to 0.745) | 0.937 (0.916 to 0.957) | 0.937 (0.917 to 0.957) | 0.931 (0.910 to 0.951) |
| Antibiotic course | 0.199 (0.183 to 0.215) | 0.264 (0.249 to 0.279) | 0.483 (0.460 to 0.506) | 0.738 (0.709 to 0.767) | 0.860 (0.826 to 0.892) | 0.866 (0.834 to 0.895) |

| Area | L1: present − all assessed | p | L2 enriched − raw (present only) | p | L2 enriched − raw (all assessed) | p |
|---|---|---|---|---|---|---|
| Assessment | +0.003 (+0.000 to +0.006) | 0.027 | -0.008 (-0.011 to -0.005) | <0.001 | -0.008 (-0.013 to -0.003) | 0.001 |
| Investigations | -0.015 (-0.024 to -0.007) | <0.001 | -0.023 (-0.033 to -0.014) | <0.001 | -0.009 (-0.021 to +0.003) | 0.145 |
| Severity classification | +0.002 (-0.005 to +0.009) | 0.564 | +0.013 (-0.001 to +0.027) | 0.073 | +0.004 (-0.012 to +0.019) | 0.624 |
| Antibiotic course | +0.000 (+0.000 to +0.000) | 1.000 | +0.057 (+0.050 to +0.064) | <0.001 | +0.057 (+0.050 to +0.064) | <0.001 |

### E. Sensitivity: PI-curated area reference sets as gold, MRR@10 (mean, 95% CI)

| Arm | w = 0.0 | w = 0.2 | w = 0.4 | w = 0.6 | w = 0.8 | w = 1.0 |
|---|---|---|---|---|---|---|
| Present only, raw | 0.666 (0.644 to 0.688) | 0.714 (0.695 to 0.733) | 0.878 (0.864 to 0.892) | 0.919 (0.907 to 0.931) | 0.999 (0.998 to 1.000) | 0.999 (0.998 to 1.000) |
| Present only, enriched | 0.692 (0.672 to 0.713) | 0.762 (0.745 to 0.780) | 0.879 (0.865 to 0.893) | 0.995 (0.992 to 0.998) | 0.993 (0.989 to 0.997) | 0.991 (0.987 to 0.995) |
| All assessed, raw | 0.656 (0.634 to 0.679) | 0.727 (0.709 to 0.744) | 0.874 (0.860 to 0.888) | 0.920 (0.908 to 0.931) | 0.998 (0.996 to 0.999) | 0.994 (0.990 to 0.997) |
| All assessed, enriched | 0.699 (0.679 to 0.720) | 0.767 (0.749 to 0.784) | 0.889 (0.875 to 0.902) | 0.989 (0.984 to 0.993) | 0.988 (0.983 to 0.993) | 0.987 (0.982 to 0.992) |

### F. Sensitivity: MRR across K (from `arm_vs_popularity_baseline.csv`)

| K | Baseline MRR | Arm MRR range, w ≥ 0.6 | Best arm (MRR) | Arms above / below / not different from baseline (of 24) |
|---|---|---|---|---|
| 2 | 0.787 | 0.783–0.859 | Present only, raw, w = 1.0 (0.859) | 10 / 12 / 2 |
| 4 | 0.797 | 0.807–0.869 | Present only, raw, w = 1.0 (0.869) | 10 / 12 / 2 |
| 6 | 0.801 | 0.809–0.872 | Present only, raw, w = 1.0 (0.872) | 10 / 12 / 2 |
| 8 | 0.802 | 0.809–0.872 | Present only, raw, w = 1.0 (0.872) | 10 / 12 / 2 |
| 10 | 0.803 | 0.809–0.872 | Present only, raw, w = 1.0 (0.872) | 10 / 12 / 2 |
| 12 | 0.803 | 0.809–0.873 | Present only, raw, w = 1.0 (0.873) | 10 / 12 / 2 |
| 14 | 0.803 | 0.809–0.873 | Present only, raw, w = 1.0 (0.873) | 10 / 12 / 2 |
| 16 | 0.803 | 0.809–0.873 | Present only, raw, w = 1.0 (0.873) | 10 / 12 / 2 |
| 18 | 0.803 | 0.809–0.873 | Present only, raw, w = 1.0 (0.873) | 10 / 12 / 2 |
| 20 | 0.803 | 0.809–0.873 | Present only, raw, w = 1.0 (0.873) | 10 / 12 / 2 |

### H. Secondary metric: Recall@10 by arm and w (mean, 95% CI)

| Arm | w = 0.0 | w = 0.2 | w = 0.4 | w = 0.6 | w = 0.8 | w = 1.0 |
|---|---|---|---|---|---|---|
| Present only, raw | 0.280 (0.270 to 0.290) | 0.548 (0.536 to 0.559) | 0.602 (0.591 to 0.614) | 0.612 (0.601 to 0.623) | 0.594 (0.583 to 0.604) | 0.590 (0.579 to 0.601) |
| Present only, enriched | 0.320 (0.309 to 0.332) | 0.578 (0.566 to 0.590) | 0.619 (0.608 to 0.631) | 0.635 (0.624 to 0.646) | 0.626 (0.615 to 0.638) | 0.613 (0.601 to 0.624) |
| All assessed, raw | 0.293 (0.282 to 0.303) | 0.535 (0.524 to 0.545) | 0.625 (0.614 to 0.637) | 0.641 (0.631 to 0.652) | 0.632 (0.622 to 0.643) | 0.621 (0.610 to 0.633) |
| All assessed, enriched | 0.329 (0.317 to 0.341) | 0.566 (0.554 to 0.577) | 0.645 (0.633 to 0.657) | 0.663 (0.652 to 0.674) | 0.649 (0.637 to 0.660) | 0.638 (0.626 to 0.649) |
<!-- END GENERATED -->

## 6. Numbers not yet available (placeholders)

| # | What | Status |
|---|---|---|
| 6.1 | **Primary outcome**: clinician ratings, 11 PATH domains × 4 query areas; P(rating ≥ 4) from the brms model; score distributions | [PLACEHOLDER: rating not started] |
| 6.2 | Number of raters, their qualifications, ratings per item (configured: 3, `IRR_MIN_RATERS`), rating dates | [PLACEHOLDER] |
| 6.3 | Krippendorff's α (ordinal) per domain | [PLACEHOLDER] |
| 6.4 | Triage-agreement results | [PLACEHOLDER] |
| 6.5 | Table 1 baseline characteristics | Done 2026-10-26 — see §8 |
| 6.6 | brms priors, chains, iterations, convergence criteria | [TODO: prespecify before rating starts] |
| 6.7 | Number of CIN hospitals; country/county wording | [TODO: PI] |

## 7. Flags for the drafting

- **F1. Headline metric chosen after results were seen.** The run's code
  sets Recall as the primary metric at K = 12 (`configuration.json`
  `primary_metric = recall_at_k`, `mrr_k = 12`). The manuscript headlines
  MRR@10, chosen on 2026-10-26 after the run's outputs and figures had been
  inspected. Under the rule "code wins, flag the conflict", Methods should
  state this plainly. Recall@10 (table H) and the K range (table F) are
  reported alongside. Since the analysis is exploratory, this is
  disclosable, not disqualifying.
- **F2. Level 1 is identical by construction for antibiotic course.** That
  area's question states age, gestation, weight and care setting but no
  vitals or findings (`app/eval/question_gen/areas.py` `_FACT_FIELDS`, DEV
  #313). Present-only and all-assessed queries are therefore the same text,
  giving a difference of exactly 0 (table D). Report it as structural, not
  as a finding.
- **F3. Post hoc weight.** The "best w (0.8) vs w = 1.0" contrast picks w
  from the same data, so its CI and p are optimistic. It is labelled as post
  hoc in table B.
- **F4. The popularity baseline is strong** (MRR@10 = 0.803), because gold
  is concentrated: the most-cited chunk is in 89.32–99.14% of each area's gold
  sets (table G). Arms only beat it at w ≥ 0.6, and not every arm at
  w = 0.6 does (table C).
- **F5. Configuration choices were made on the same 933 questions** (Okapi
  scorer, N = 3, area weighting, area-scoped enrichment). Only the N tuning
  used a held-out split, and its outputs were deleted (DEV #317). No re-runs
  (PI, 2026-10-26), so those analyses appear in Methods as decisions only,
  with no numbers.
- **F6. The ablation's retriever is not the platform's.** The ablation scores
  offline Okapi BM25 + SapBERT; the rated answers come from the platform's
  hybrid + reranker pipeline (§4.3). The ablation describes how sensitive
  retrieval is to query construction; it does not test the rated answers.
- **F7. Own-citation gold is circular by design.** It measures recovery of
  the passages the platform used, not clinical correctness. Table E (the
  PI-curated sets) is the external-gold check. Reference-set MRR@10 at
  w ≥ 0.8 is 0.987–0.999, close to the ceiling.

## 8. Table 1 — record characteristics by study set (generated)

Source: `manuscript/scripts/table1.py` reads the de-identified records
(`newborn_nbu_2021`) and the question tables in the platform database,
decrypting them in memory and writing aggregates only, to
`manuscript/derived/table1.json`. `render_table1.py` renders that file to
`manuscript/derived/table1.md`, copied below.

Checks run by the script:
- Every record ID in every set resolves to a record.
- The review set and the holdout share no records (0).
- The analysed set is a subset of the holdout.
- The set sizes (1,500 / 200 / 980 / 933 / 47) match §1.
- The holdout triage split matches DEV #305.

<!-- BEGIN GENERATED: manuscript/derived/table1.md -->
<!-- generated by manuscript/scripts/render_table1.py; do not edit by hand -->

| Characteristic | All records (n = 1500) | Clinician review set (n = 200) | Ablation holdout (n = 980) | Ablation analysed (n = 933) | Excluded, no gold set (n = 47) |
|---|---|---|---|---|---|
| Day of life, median (IQR) | 1 (1–1) | 1 (1–1) | 1 (1–1) | 1 (1–1) | 1 (1–1) |
| Gestational age, weeks, median (IQR) | 38.0 (32.0–39.0) [132 missing] | 38.0 (32.0–39.0) [18 missing] | 38.0 (32.0–39.0) [83 missing] | 38.0 (32.0–39.0) [82 missing] | 37.5 (32.0–38.8) [1 missing] |
| Birth weight, g, median (IQR) | 2675 (1630–3200) | 2700 (1598–3200) | 2675 (1754–3200) | 2680 (1755–3200) | 2600 (1610–3072) |
| Temperature, °C, median (IQR) | 36.5 (35.4–37.4) [219 missing] | 36.5 (35.7–37.2) [24 missing] | 36.4 (35.4–37.1) [107 missing] | 36.4 (35.4–37.2) [99 missing] | 36.3 (35.3–36.8) [8 missing] |
| Heart rate, /min, median (IQR) | 137 (123–150) [214 missing] | 136 (124–148) [23 missing] | 138 (122–152) [90 missing] | 138 (122–152) [82 missing] | 136 (120–148) [8 missing] |
| Respiratory rate, /min, median (IQR) | 58 (48–68) [302 missing] | 60 (50–68) [33 missing] | 60 (48–68) [156 missing] | 59 (48–68) [146 missing] | 60 (43–66) [10 missing] |
| SpO2, %, median (IQR) | 93 (87–96) [329 missing] | 93 (88–96) [30 missing] | 93 (86–96) [179 missing] | 93 (86–96) [171 missing] | 94 (90–98) [8 missing] |
| **Sex, n (%)** | | | | | |
| Female | 613 (40.87) | 83 (41.50) | 398 (40.61) | 380 (40.73) | 18 (38.30) |
| Male | 883 (58.87) | 117 (58.50) | 578 (58.98) | 550 (58.95) | 28 (59.57) |
| Not recorded | 4 (0.27) | 0 | 4 (0.41) | 3 (0.32) | 1 (2.13) |
| **Triage category, n (%)** | | | | | |
| Critical | 300 (20.00) | 40 (20.00) | 178 (18.16) | 170 (18.22) | 8 (17.02) |
| Substantial | 300 (20.00) | 40 (20.00) | 214 (21.84) | 206 (22.08) | 8 (17.02) |
| Moderate | 300 (20.00) | 40 (20.00) | 198 (20.20) | 184 (19.72) | 14 (29.79) |
| Low | 300 (20.00) | 40 (20.00) | 209 (21.33) | 196 (21.01) | 13 (27.66) |
| Minimal | 300 (20.00) | 40 (20.00) | 181 (18.47) | 177 (18.97) | 4 (8.51) |
| **Care setting, n (%)** | | | | | |
| NBU | 1500 (100.00) | 200 (100.00) | 980 (100.00) | 933 (100.00) | 47 (100.00) |

Exact counts, no small-cell suppression. Vital signs are the first recorded set.
<!-- END GENERATED -->

Notes for Table 1:
- **Day of life** is 1 (IQR 1–1) in every set. The row is kept (PI,
  2026-10-26).
- **Exact counts, no small-cell suppression** (PI, 2026-10-26). One-cell
  suppression could be undone from the column total, and "sex not recorded"
  (4 of 1,500) is a missing-data marker, not an identifier.
- **Gestational age is missing** for 132 of the 1,500 records (8.80%) and
  82 of the 933 (8.79%). Vital signs are missing for 8.79–21.93% of records,
  depending on the sign and the set. Missing values are omitted from the
  question text and not imputed (00_inventory §4).
- **The 47 excluded records** (all antibiotic course) lean towards Moderate
  and Low triage: 14 (29.79%) and 13 (27.66%); Minimal 4 (8.51%).
- **One ingested record has no vital signs.** It is not in any analysed set.

## 9. Pipeline outcomes on the review items (generated)

Source: `manuscript/scripts/review_outcomes.py` → `manuscript/derived/review_outcomes.json`. It reads unencrypted columns only and writes aggregates. Percentages were computed by script and rounded half up.

| # | Number | Meaning |
|---|---|---|
| 9.1 | 800 | Review items with a result; all `open`; 0 rating rounds (checked 2026-10-26) |
| 9.2 | 760 (95.00%) / 40 (5.00%) / 0 | Released (well supported) / escalated / "no guideline found" |
| 9.3 | 39 of 200 (19.50%); 1 of 200 (0.50%); 0; 0 | Escalated by area: antibiotic course; assessment; investigations; severity classification |
| 9.4 | 5 (3.13%), 8 (5.00%), 6 (3.75%), 11 (6.88%), 10 (6.25%) of 160 each | Escalated by triage category: Critical, Substantial, Moderate, Low, Minimal (matches DEV #303) |
| 9.5 | 40 (39 antibiotic course, 1 assessment) | Answers citing no passage — the same counts as the escalations |
| 9.6 | qwen3.6:35b, 800/800 | Model id in every result's config snapshot |
