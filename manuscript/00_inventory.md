# Manuscript inventory (Step 1)

Branch `manuscript`, analysis code at commit `6a3693c` (2026-10-26). Every
claim below points to code, config or an output file. "DEV #n" = an entry in
`DEVIATIONS.md` (the project's decision log).

**The only analysis with results is the offline retrieval ablation**
(PRD-112 / ARCH-043, "unified ablation"). The platform also has a
multi-rater clinician review workflow (800 queued items), but it has **0
ratings** (`eval.rating_round` = 0, `eval.rubric_rating` = 0, checked
2026-10-26), so nothing from it can be reported yet.

---

## 1. Pipeline map

The order things were run, from raw data to the current results. Each step is
a `python -m scripts.<name>` inside the `api` container unless noted.

| # | Step | Code | Run record |
|---|---|---|---|
| 1 | Ingest the guideline corpus (4 PDF excerpts → 297 chunks, 153 retrievable) | `scripts/ingest_manifest_documents.py`, `app/ingestion/`, manifest `data/excerpt_guidelines/manifest.json` | `backend/results/corpus_ingest_20261014.log`; DEV #302 |
| 2 | Re-confirm the 14 held dose-table chunks from the pre-reset export | `scripts/carry_over_review_confirmations.py` | DEV #301, #302 |
| 3 | Ingest the de-identified newborn records (EAV CSV → `PatientRecord`) | `scripts/ingest_deidentified_records.py`, `app/ingestion/eav.py`, `data/patient_records/deidentified/newborn_nbu_2021/field_mapping.yaml` | DEV #298, #302 |
| 4 | Build the clinician review queue (200 records, 40 per triage category × 4 areas = 800 items); these records are removed from the ablation pool | `scripts/seed_triage_review_queue.py` → `app/eval/auto_seed.py:run_triage_review_queue`, seed 20261014 | `backend/results/triage_review_queue_20261014.log`; DEV #300, #303 |
| 5 | Generate the ablation holdout questions: one record → one area question → one full RAG pipeline call → gold set = the answer's cited chunks | `scripts/generate_ablation_holdout_questions.py` → `app/eval/auto_seed.py:run_four_area_holdout_generation` (1125), `_generate_one_area_question` (1057) | seed 20261015: `backend/results/ablation_holdout_20261015.log` (844 questions); top-up seed 20261016: `ablation_holdout_topup_20261016.log` (136 questions); DEV #303–#305 |
| 6 | Pin provenance (row counts + hashes of corpus, questions, records, files) | `make provenance` | `data/PROVENANCE.json` (pinned at `b4fe4d5`, 2026-10-16) |
| 7 | Curate fixed per-area reference passage sets (draft from the holdout answers' citations, then PI review and attestation) | `scripts/draft_area_reference_sets.py`, `scripts/area_reference_review.py` | `data/area_reference_sets.yaml`; DEV #307–#309 |
| 8 | Run the unified ablation sweep + statistics + popularity baseline + figures | `scripts/run_unified_ablation.py` → `app/eval/unified_ablation/runner.py`, `summary.py`, `blend.py`; `scripts/report_popularity_baseline.py`; `scripts/plot_ablation_figures.py` | run `20261026T075410Z-e78b8cf6`, log `backend/results/unified_ablation_20261026T075410Z-e78b8cf6.log`; DEV #317 |
| 9 | Rescore the run against the per-area reference sets | `scripts/score_area_reference.py` → `app/eval/area_reference.py` | same run directory; DEV #317 |
| 10 | Re-render figures after presentation changes | `scripts/plot_ablation_figures.py` | DEV #318–#321 |

Steps 1–6 ran 2026-10-14 to 2026-10-16; steps 8–10 on 2026-10-26.

---

## 2. Data and cohort

### Data sources
- **Patient records:** `newborn_nbu_2021`, de-identified newborn-unit
  admission snapshots. Attestation (`data/patient_records/deidentified/newborn_nbu_2021/DATASET.md`):
  source "The Clinical Information Network (CIN)"; collection period
  2021-01-01 to 2024-12-31; site "Newborn units of the CIN hospitals";
  de-identification "removal of direct identifiers, integer surrogate key";
  approval KEMRI/SERU/CGMR-C/307/4901; attested by the PI 2026-10-13.
  - 1,500 patients, one assessment snapshot each, 49,850 EAV rows, 39
    variables (DEV #298).
  - **Sampled 300 per triage category** (Critical/Substantial/Moderate/Low/
    Minimal). DATASET.md: "a property of this extract, not of the source
    population".
- **Guideline corpus** (`manifest.json`; ingested 2026-10-14):
  1. WHO, *Recommendations for management of serious bacterial infections in
     infants aged 0–59 days* (2024), excerpt pp. 17–19, 76–77.
  2. Kenya MoH, *Comprehensive Newborn Care Protocols* (Nov 2022), excerpt pp. 47–48.
  3. Kenya MoH, *National Antibiotic Use Guidelines* (Nov 2024), excerpt pp. 11–20, 33–38, 45–52 (2 sections excluded).
  4. NICE NG195, *Neonatal infection: antibiotics for prevention and treatment* (2021, updated 13 May 2026), full (11 sections excluded).
  - 297 chunks; 144 excluded by the manifest's `exclude_sections` (navigation,
    front matter, rationale); **153 retrievable** = the ablation corpus
    (`app/eval/model_ablation/ablation.py:98` `CORPUS_FILTER`; counted live
    2026-10-26).

### Flow (for a flow diagram)

| Step | n | Implemented at |
|---|---|---|
| Records ingested | 1,500 (300 per triage category) | DEV #302 |
| Allocated to the clinician review queue (40 per category) | −200 | `auto_seed.py:1431` `plan_triage_review_records` |
| Eligible for the holdout (not used by any generator) | 1,300 | `auto_seed.py:1154` (`unused_records`) |
| Questions generated, first run (seed 20261015, target 200 usable per area) | 844 records | log 20261015; DEV #303 |
| Records left after the first run | 456 | DEV #304 |
| Rejected as clinical near-duplicates in the top-up | −320 | `auto_seed.py:1076`; `app/eval/question_gen/diversity.py:102` |
| Questions generated, top-up (seed 20261016) | 136 | DEV #305 |
| **Holdout questions** (one per record, no overlap with the review queue) | **980** | DEV #305 |
| Excluded: the answer cited no chunk (no gold set) | −47 (all antibiotic course) | `auto_seed.py:1119`; `runner.py:142` |
| **Analysed** | **933** (assessment 234, investigations 233, severity classification 233, antibiotic course 233) | `configuration.json` `questions_per_area`; `runner.py:114-142` |

- Holdout selection criteria (`runner.py:120-128`): `expected_outcome =
  well_supported`, `provenance = auto_generated`, template `four-area-v2`,
  `purpose = ablation_holdout`, non-empty gold set. Questions whose record
  can't be resolved, or whose area isn't attested, are skipped
  (`runner.py:298`, `:308`); none were (933 records used).
- **Area allocation:** each new record goes to the area with the fewest usable
  questions (`auto_seed.py:1050` `_next_area`). Records are tried in a seeded
  shuffle (`auto_seed.py:398-428`).
- **Near-duplicate rule** (`diversity.py:56-130`): a candidate is rejected
  if, against any record already accepted, both hold: Jaccard similarity of
  the present-findings sets ≥ 0.8 (`QGEN_DEDUP_THRESHOLD`) AND every shared
  vital within tolerance (HR 15 bpm, RR 10/min, temperature 0.5 °C, SpO2 5%,
  capillary refill 1 s, weight 200 g). Near-duplicate rejections during the
  first run were not counted. The 456 → 136 + 320 split is from the top-up.
- **Triage category of the 980 holdout records** (DEV #305): Critical 178,
  Substantial 214, Moderate 198, Low 209, Minimal 181. Not yet computed for
  the 933.

---

## 3. Variables

The unit of analysis is a **query** (one per record); the outcome is
retrieval quality against a gold passage set.

### Query construction (the "exposures")
- **Query area:** one of four fixed questions in `data/query_areas.yaml`
  (attested by the PI, 2026-10-02): assessment, investigations, severity
  classification, antibiotic course. Each is phrased "What does the guideline
  recommend about <area opening> based only on the content provided below:"
  (`app/eval/question_gen/areas.py`).
- **Patient details** rendered from record fields, only those the area lists
  (`record_facts`): age in days (antibiotic course adds an age-band label),
  gestational age, presenting complaint, care setting, weight in kg
  (antibiotic course only, several roundings), maternal risk factors,
  admission vitals, examination findings. Medications and interventions never
  appear (DEV #155). `triage_category` never enters a query (ARCH-046).
- **Level 1, finding handling:** `all_assessed` states assessed-present and
  assessed-absent findings ("did NOT have …"); `present_only` states only
  assessed-present findings. Not-assessed findings are never stated. For
  antibiotic-course questions (no findings listed) the two are identical.
- **Level 2, concept enrichment:** `raw`, or `enriched` = the query plus
  matched terms from an operator-curated vocabulary of 22 concepts
  (`data/clinical_concepts.yaml`: thresholds such as tachypnoea, fever,
  preterm, low birth weight, age bands, maternal infection, PROM, signs).
  Only concepts on a field the area's question states are added
  (`areas.py` `area_vocabulary`, DEV #313).
- **Level 3, retriever:** minmax fusion `w × Okapi BM25 + (1 − w) × SapBERT
  cosine`, w ∈ {0.0, 0.2, …, 1.0}; w = 1.0 is pure BM25, w = 0.0 pure
  SapBERT.
  - Okapi BM25 k1 = 1.2, b = 0.75, IDF `ln((N − n + 0.5)/(n + 0.5) + 1)`,
    statistics over the 153 retrievable chunks (`blend.py:92-135`).
  - SapBERT `cambridgeltl/SapBERT-from-PubMedBERT-fulltext`, [CLS] pooling,
    L2-normalised, brute-force cosine over the full corpus
    (`app/eval/model_ablation/encoders.py`).
  - Each channel min-max normalised per query (`blend.py:163-188`).
- **Area-question weight N = 3:** the area question is written 3 times
  before the patient details (SapBERT input), and its terms are weighted 3×
  in the lexical query (`blend.lexical_query_vector`; DEV #312).

### Outcomes
- **Gold set (per query):** the chunk ids cited by the platform's own
  answer to the N = 1 version of the question, generated 2026-10-15/16 by
  the full pipeline: hybrid retrieval (bge-large-en-v1.5 dense + Qdrant
  TF×IDF sparse, RRF k = 60, CANDIDATE_K 40 / FUSED_K 24 / TOP_K 8,
  per-guideline cap 3–5 by area), bge-reranker-v2-m3, synthesis and citation
  verification by `qwen3.6:35b` through a self-hosted gateway
  (`auto_seed.py:1094-1120`; `.env`).
  - Gold size mean 6.32, median 6, range 2–14.
  - 51 distinct chunks across 933 queries; 488 distinct gold sets.
  - Within each area the most-cited chunk is in 89.3% (assessment) to 99.1%
    (investigations) of gold sets.
- **Metrics at K ∈ {2, 4, …, 20}** (`app/eval/metrics.py`):
  - Recall@K = |top-K ∩ gold| / |gold| (primary metric in code,
    `configuration.json` `primary_metric`).
  - MRR@K = 1/rank of the first gold chunk within the top K, else 0.
  - Hit@K = any gold chunk in the top K.
  - Headline K = 12 (`ABLATION_MRR_K`).
- **Secondary outcome, area reference sets:** the same rankings scored
  against fixed, PI-curated per-area passage sets (assessment 18,
  investigations 13, severity classification 9, antibiotic course 37 chunks;
  DEV #309). Recall@12 is capped at about 0.73 on average by set size.

### Covariates
None are adjusted for. Area is reported in the reference-set scoring.
Triage category and demographics are not used in any analysis.

---

## 4. Missing data
- **Not-assessed fields** (EAV "NA" or an absent row) are omitted from the
  question text; nothing is imputed (`eav.py` null literals; DEV #138/#139).
- **No gold set:** 47 holdout questions whose answer cited nothing were
  excluded (complete-case on the outcome). All 47 were antibiotic course.
- **Near-duplicates:** 320 records excluded by design (diversity filter),
  not for missingness.
- No imputation anywhere.

---

## 5. Statistical / analytic methods
All in `app/eval/unified_ablation/summary.py` and `app/eval/bootstrap.py`.
No models are fitted or trained, and no regression is run.

- **Design:** a full factorial of 2 (Level 1) × 2 (Level 2) × 6 (w) × 10 (K)
  over 933 queries = 223,920 per-query rows. One encode/score per
  (query, L1, L2).
- **Pooling:** for each comparison, a query's metric is averaged over every
  dimension not being compared, so each query contributes one score per side.
  K is never pooled.
- **Comparisons (Recall@12 primary; MRR also computed):**
  - Level 1: present-only − all-assessed, pooled over L2 and w.
  - Level 2: enriched − raw within each L1, pooled over w.
  - Level 3 endpoints: w = 1.0 − w = 0.0 per L1 × L2 arm.
  - Full grid: Recall@K for every w × K, pooled over L1 × L2 (60 points, no
    test).
  - **Post-hoc** best-weight test: the empirically best w vs w = 1.0, flagged
    in output as winner's-curse biased.
  - **Popularity baseline:** a query-blind ranking of chunks by how many
    *other* queries' gold sets contain them (leave-one-out, **pooled across
    areas**, ties by chunk id). Every arm × w is compared with it at every K
    on Recall, MRR and Hit (`summary.py:429-524`).
- **Inference:** paired percentile bootstrap, 10,000 resamples (the same
  query indices resampled for both sides), 95% CI, two-sided bootstrap
  p-value from the same resample, seed 1234 (`bootstrap.py:36-130`).
  "Significant" = CI excludes 0.
- **No multiple-comparison correction.** At each K, 24 arm × w cells are
  compared with the baseline on 3 metrics, across 10 K values.
- **Reference-set scoring:** unpaired bootstrap 95% CI on means
  (`area_reference.py:156-178`), with own-gold scores alongside.
- **Sensitivity / tuning analyses run earlier, outputs since deleted
  (DEV #317), recorded only in DEV #311–#315:**
  - TF×IDF vs Okapi lexical scorer (#311);
  - area-question-only arm (#311);
  - N = 1 vs N = 3 full runs (#315);
  - N tuned on a hash split by source record (465 tune / 468 held-out), grid
    {1, 2, 3, 4, 5, 6, 8} (#315).
- **Leakage / overfitting:** no model is trained. Configuration choices
  (Okapi, N, area weighting, enrichment scope) were made iteratively on the
  same 933-question pool; only the N tuning used a held-out split.

---

## 6. Software and seeds
- Python 3.11.16 in the `api` container image (built 2026-10-21); lockfile
  `backend/requirements-lock.txt`.
- Packages (live `pip freeze`): numpy 2.4.6, scipy 1.17.1, pandas 3.0.6,
  matplotlib 3.11.2, seaborn 0.13.2, torch 2.14.0, transformers 5.17.0,
  sentence-transformers 6.0.1, qdrant-client 1.19.0, scikit-learn 1.9.1,
  SQLAlchemy 2.0.52, langgraph 1.2.11, pydantic 2.13.5, docling 2.130.0,
  pdfplumber 0.11.10, rapidocr 3.9.2. Okapi BM25 is implemented in-repo
  (rank-bm25 is installed but not used here).
- Infrastructure: PostgreSQL, Qdrant (versions [TODO: from images]), Redis.
- Models:
  - SapBERT `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` (flagged
    `SAPBERT_MODEL_VERIFIED=false`);
  - bge-large-en-v1.5 and bge-reranker-v2-m3 (local, gold generation only);
  - LLM `qwen3.6:35b` via a self-hosted gateway (`MODEL_ID_VERIFIED=true`).
    Per the PI (2026-10-26): Qwen3.6:35b, Q4_K_M quantisation, served
    locally by Ollama behind the BRIDGE LLM Gateway with fallback routing
    (https://github.com/AfrikanaCollective/BRIDGE_LLM_Gateway). This
    deployment configures no fallback model (`MODEL_ID_FALLBACKS` is empty),
    and the PI confirmed every gold-generating answer came from
    Qwen3.6:35b. Ollama 0.32.14; gateway release/1.0; NVIDIA H200 GPU.
    Gateway-fixed settings (PI): num_ctx 32768, think = false, stream =
    false. Temperature, top-p, max tokens and seed: not set by this
    repository; whether the gateway sets them is unconfirmed.
- Seeds: bootstrap 1234; review-queue sampling 20261014; holdout 20261015
  and 20261016.
- LLM sampling parameters (temperature, top-p, max tokens, seed) are **not
  set anywhere in the code** (no matches in `app/llm/`, `app/agents/`), so
  gateway defaults applied.
- Prompts: `backend/app/agents/prompts/*.md`.
- Hardware: LLM on an NVIDIA H200 GPU (PI). SapBERT encoding ran on CPU
  (Intel Core i9-14900KF) in the `api` container: `torch.cuda.is_available()`
  is False there and no GPU is passed to the container (checked 2026-10-26).

---

## 7. Outputs (current run `backend/results/ablation/20261026T075410Z-e78b8cf6/`)

| File | Contents |
|---|---|
| `configuration.json` | full run configuration (grid, scorer, N, scope, seeds, hashes of query areas and concepts, questions per area, n rows) |
| `per_query_results.jsonl` | 223,920 rows: query id, L1, L2, L3, K, w, query text, enriched text, top-K ids, gold ids, first relevant rank, Recall/RR/Hit@K, area |
| `statistical_summary.json` | Level 1, Level 2, Level 3 curve and endpoints, w × K grid, post-hoc best weight, popularity baseline at K = 12 and every K |
| `arm_vs_popularity_baseline.csv` | one row per K × metric × arm × w: delta, CI, p, verdict |
| `area_reference_scores.csv`, `area_reference_summary.json` | reference-set Recall/MRR/Hit@K with CIs, own-gold alongside |
| `recall_at_k_by_bm25_weight.png` | Recall at K vs K, one line per w, 2 × 2 panels by arm |
| `recall_at_k_vs_bm25_weight_by_k.png` | Recall at K vs w, lines K = 4, 6, 8, 10, 2 × 2 by arm |
| `mrr_at_k_vs_bm25_weight_by_arm.png` | MRR vs w, one line per arm, panels A–D K = 4, 6, 8, 10 |

Plus the run log, `data/PROVENANCE.json`, and the generation and ingestion
logs listed in §1.

---

## 8. Discrepancies and gaps

**Would change how results can be stated**
1. **The gold standard is the system's own output.** Gold sets are the
   chunks cited by the platform's LLM answer to the same question, retrieved
   by a pipeline with a lexical (BM25-type) channel. This is not clinician
   adjudication, and it may favour lexical retrieval (the high-w arms).
   Reviewers will see this as circular.
2. **Gold sets are concentrated** (51 chunks; the top chunk is in 89–99% of
   each area's sets), so a query-blind popularity ranking scores Recall@12
   0.667 / MRR@12 0.803. Only some arm × w cells beat it at N = 3 (8/24 on
   Recall@12, 10/24 on MRR@12). At N = 1, none did (DEV #315; that run has
   been deleted).
3. **Configuration chosen on the evaluation pool.** Okapi, the area-question
   weighting, N = 3 and area-scoped enrichment were adopted after looking at
   results on these same 933 queries. The analysis is exploratory, not a
   confirmatory test. The held-out N tuning picked N = 8 on its own
   criterion; N = 3 was the operator's choice (DEV #316).
4. **Reference sets were drafted from the same answers' citations** and
   curated by a single reviewer (the PI), with no second annotator or
   agreement statistic.
5. **The query evaluated is not the question that produced the gold.** Gold
   came from the N = 1 question; the ablation queries use N = 3, and the
   enriched/present-only variants.
6. **No multiple-comparison correction** across arms, weights, K and metrics.
7. **Primary outcome ambiguity:** the code's primary metric is Recall@K
   (K = 12), but recent reporting (DEV #312–#317) led with MRR@12. This
   needs a decision before writing.

**Reproducibility**
8. `configuration.json` does not record a git commit, and the run used
   uncommitted code. That code is now committed as `6a3693c`.
9. SapBERT id and pooling flagged unverified in config.
10. LLM sampling parameters are unset in this repository (gateway/Ollama
    defaults apply) and unrecorded. The model build and quantisation are now
    known from the PI (Qwen3.6:35b Q4_K_M, Ollama 0.32.14, BRIDGE LLM
    Gateway release/1.0, H200, no fallback used, num_ctx 32768, think off).
    Temperature / top-p / seed still unconfirmed.
11. Sensitivity analyses cited in DEV #311–#315 (TF×IDF, area-question-only,
    N = 1, N tuning) have no outputs on disk after the cleanup (#317). They
    would need re-running to be reported.

**Missing for a manuscript**
12. **No baseline-characteristics table:** no code summarises the 933 (or
    980) holdout records (age, gestation, birth weight, sex, triage category,
    vitals and finding prevalence, missingness). This needs a new
    aggregate-only script.
13. **No per-area breakdown** of the own-gold metrics (only the reference-set
    scoring is per area).
14. **No human evaluation results** (0 ratings). The answer-quality, safety
    and IRR parts of the platform cannot be reported.
15. No error analysis (which queries fail, and why).
16. Corpus is small and partial (4 documents, mostly excerpts, 153
    retrievable chunks). Generalisability is limited.
17. The triage-balanced extract is not representative of admissions.
18. Dates: DATASET.md says collection 2021–2024, while the dataset id says
    "2021". Confirm the actual admission dates.
19. Data are handled under a development encryption key
    (`SECRETS_BACKEND=file`, DEV #302). Probably not for the manuscript, but
    it bears on the data-protection statement.
