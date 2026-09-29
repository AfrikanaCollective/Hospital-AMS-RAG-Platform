# ARCHITECTURE-ESSENTIALS.md

Short-form, coding-agent-facing outline of the **critical decisions only**.
Full detail and rationale: [ARCHITECTURE.md](ARCHITECTURE.md). Requirements:
[PRD.md](PRD.md). Excluded capabilities: [CDS-FUTURE.md](CDS-FUTURE.md).
Judgment calls: [DEVIATIONS.md](DEVIATIONS.md).

**Keep this file in sync whenever ARCHITECTURE.md changes.**
**Status:** Phase 3 checklist complete, Checkpoint 3 pending. Last updated 2026-09-14.

---

## 0. Hard rules (never violate)

- **No real PHI. Patient data is `synthetic` OR operator-attested
  `deidentified`** (ARCH-039). A `deidentified` dataset is admitted only with a
  complete `DATASET.md` attestation + explicit intent flag, then handled
  **exactly as PHI** (encryption/RBAC/RLS/audit/no-egress/no-training). Any
  real-looking batch with neither marker nor attestation is hard-rejected. All
  patient fields are PHI by default. Dataset files are never committed to VCS.
- **No independent clinical advice.** The system *reports and cites* retrieved
  source text. Framing is "Guideline X recommends…", never "You should…".
  Every response carries a non-removable disclaimer.
- **Grounding is enforced.** No answer segment ships unless a citation resolves
  to a retrieved chunk and its verbatim quote supports the claim. Low
  confidence / conflict / no match ⇒ "no guideline found" or HITL, never a
  general-knowledge answer.
- **No hardcoded model names.** `MODEL_ID`, `EMBEDDING_MODEL_ID`,
  `RERANKER_MODEL_ID` all from config with placeholder defaults. Unverifiable
  names are flagged at startup, never guessed.
- **CDS boundary (CDS-FUTURE.md) is hard.** SCOPE-2.3 and SCOPE-2.4 are not
  implemented, even partially. If a task seems to need them, STOP and flag.
- **Phase checkpoints are hard stops.** Do not chain phases.

---

## 1. Stack (with the one-line why)

| Layer | Choice | Why |
|---|---|---|
| Backend | Python + FastAPI | async, Pydantic schemas, ML ecosystem |
| Vector store | **Qdrant** (self-hosted) | native dense+sparse hybrid + server-side RRF + fast payload filtering for access scoping |
| Retrieval | dense + BM25 (sparse) → RRF → cross-encoder rerank | precision for citation-grade grounding |
| RDBMS | **PostgreSQL** (schemas: corpus, records, memory, hitl, eval, audit, iam) | one transactional system of record; RLS; pgcrypto; LangGraph checkpointer |
| Orchestration | LangGraph (Postgres checkpointer) | typed state, deterministic routing, HITL interrupts |
| Async | Redis + Celery | ingestion fan-out, long agent runs, eval/IRR jobs |
| LLM | `LLMGateway` → self-hosted gateway, `MODEL_ID` + `MODEL_ID_FALLBACKS` | config-driven, fallback routing |
| Embeddings | config model id; `local` \| `gateway` \| `stub` (config-selectable) | no hardcoded model |
| Reranker | config model id; **decided: local**, in the `api` process (not gateway-routed) | no hardcoded model; DEVIATIONS.md #44 |
| Auth | `AuthProvider`: dev-JWT (roles: clinician/reviewer/admin/service) + OIDC stub | full IdP deferred |
| Frontend | React (Vite + TS), static behind nginx | typed API client |
| Deploy | docker-compose; `stub` LLM gateway profile for offline core flows | single host, no external net for core |

Placeholder model ids (UNVERIFIED, operator must confirm): embeddings
`BAAI/bge-large-en-v1.5`, reranker `BAAI/bge-reranker-v2-m3`, `MODEL_ID` =
`"<set-me>"` (answer path refuses to start on placeholder).

---

## 1a. Patient record schema (`app/schemas/record.py`) — currently v1.4.0

One canonical Pydantic model, `PatientRecord` — a flat, **source-agnostic**
clinical snapshot (not an EHR). Every source maps *onto* it via its own
adapter / `field_mapping.yaml`; no source-specific field enters the schema.

- **Temporal:** `Medication` / `Intervention` carry `started_at` **+
  `stopped_at`** (interval; null end = ongoing/unknown). `Vitals` / `LabResult`
  / `ExamFinding` carry one `*_at` (point-in-time).
- Repeated data = `list[TypedSubModel]`, never a `dict` bag.
- Evolution is **additive-only**, gated by `schema_version`; older versions
  still validate. `given_name`/`family_name` optional (de-identified data has
  no names).
- Version history (module docstring): 1.0 adult · 1.1 neonatal fields · 1.2
  exam findings / interventions / capillary refill / names optional · 1.3
  `Medication.stopped_at` + `Intervention.stopped_at` · **1.4
  `maternal_risk_factors[]`** (reuses `ExamFinding`'s shape, kept distinct from
  the newborn's own `examination_findings[]`) (DEVIATIONS #23, #32, #35, #38,
  #39, #139).
- **Repeating-group presence in `field_index`** (`compute_field_index`,
  `app/ingestion/records.py`): a "wide" group (`Vitals`, one field per concept)
  indexes as `{field}.{idx}.{concept}` by recursing its dict directly. A
  "name+value" group (`ExamFinding`/`Intervention`/`Medication`/`LabResult` —
  concept identity lives in a `name`/`analyte` *value*, not a key) is
  special-cased so the concept name becomes part of the index *key*
  (`examination_findings.0.apnoea`), never its value — an item existing at all
  means *assessed* (its bool field is non-nullable), regardless of
  True/False/present-or-not, distinct from no entry at all (never assessed)
  (DEVIATIONS #138).
- For `newborn_nbu_2021`: every med/intervention `started_at` == the record's
  `encounter.admitted_at`; `stopped_at` null (no source data).

---

## 2. Chunking (structure-aware; atomic unit depends on the document's `format_profile`)

0. **`format_profile`** per `document_version` (manifest or detected):
   `grade_recommendations` | `clinical_protocol` | `narrative`. Selects the
   atomic unit (rules 1 / 1b). `chunk_type ∈ {prose, recommendation,
   protocol_step, table, figure, list, criteria}`.
1. `grade_recommendations`: never split a recommendation + its qualifiers
   (strength, evidence grade, preconditions). `chunk_type = recommendation`.
1b. `clinical_protocol`: never split a numbered protocol step + sub-bullets +
   its bound dose table. `chunk_type = protocol_step`. Flowchart → `figure`
   chunk linked via `parent_chunk_id`.
2. Chunk within deepest heading; target 350–600 tokens, ~15% prose overlap
   (overlap has no citation authority).
3. Tables = one chunk (`table`), serialized Markdown + caption + heading.
3b. Figures/algorithms = one `figure` chunk: caption + heading + embedded
   text layer + `figure_ref {page, bbox, image_sha256}`. **OCR is permitted in
   the MVP** (ARCH-044, DEVIATIONS #213) and implemented behind
   `INGEST_PARSER=layout` (#215; Docling + pdfplumber + RapidOCR, `pypdf`
   fallback). Only text-layer, OCR, attested-correction, or verified-structure
   text is citable, never model-generated text. OCR'd digits and low
   parse-quality documents are held (`review_status = pending`) until an admin
   confirms them. D12: OCR tables may be re-transcribed by the gateway's vision
   model (`vlm_transcription`), which is always held and cross-checked against
   OCR per cell and header; the manifest's `table_sources` can pin a table to
   OCR. Flowcharts are one atomic `flowchart` chunk with their
   verified edges; unverified structure is capped at `weak`. A
   caption-only figure (`has_embedded_text=false`) is down-weighted and, per
   §3, capped at `weak` support — never the sole support for a claim.
4. Criteria lists = `criteria` chunks with structured `meta.criteria[]`
   (field/op/value/unit) → feeds SCOPE-2.1 / SCOPE-2.2. Extractable under any
   profile.
5. Store `section_path`, `section_number`, `page_start/end`, `char_start/end`,
   `parent_chunk_id`, `format_profile`, `topic_tags`. Embed with `section_path`
   prefix; cite the raw slice.

Document ingest metadata (title/publisher/version/effective_date/**licence**/
`format_profile`/topic_tags) is **operator-supplied** via a per-file manifest
(`data/excerpt_guidelines/manifest.json`), never inferred from PDF metadata
(**ARCH-038**). Dev corpus = real PDFs in `SAMPLE_GUIDELINES_DIR`; the
synthetic 3-doc set is a CI-only opt-in fallback (`GUIDELINES_ALLOW_SYNTHETIC`).

Changing `EMBEDDING_MODEL_ID` ⇒ full re-embed into a new Qdrant collection;
eval snapshots pin `embedding_collection`.

---

## 3. Citation format (minimum) & grounding gate

**Citation object:** `document_id` + `version_label`/`document_version_id` +
`version_status` + `section_number`/`section_path` + `page_start/end` +
`chunk_id` + `char_start/end` + verbatim `quote` + `quote_char_start/end`.

**Answer = ordered segments.** Each is a *claim segment* (statement about
guideline content, ≥1 citation, verbatim quote) or a *framing segment*
(non-claim connective text, no directive phrasing). Free-form prose is
rejected + regenerated once, then escalated.

**Grounding gate (citation-verifier agent), per claim segment:**
1. citation resolves to a chunk in *this turn's* retrieval snapshot
2. quote is a verbatim substring of that chunk; offsets check out
3. entailment: lexical overlap + constrained model call ("supported?
   yes/no/partly")
4. scope/wording: no directive phrasing, no dosing/therapy/population claims
   absent from the quote

**Verdict:** all supported → release. Only `weak` → release marked + queue for
review. Any `unsupported` and stripping keeps it coherent → partial-strip +
re-check + log stripped spans. Unsupported and stripping breaks it, or a
directive/CDS `scope_violation` → **escalate, hold**. Low-confidence/empty
retrieval never reaches synthesis with claims.

---

## 4. Capability scope (SCOPE-*)

**In scope**
- **SCOPE-1.1/1.2/1.3/1.4** — grounded guideline reporting/synthesis for
  hypotheticals; reported-content framing enforced in every guideline-touching
  agent's *prompt template* + wording filter; explicit "no guideline found"
  when nothing relevant retrieved; auto-generated hypothetical set (§7).
- **SCOPE-2.1** — patient **stage-of-care classification** from *extractable
  guideline criteria*, cited, label + confidence only, no next-step. Uncertain
  ⇒ escalate.
- **SCOPE-2.2** — **missing-information identification**: specific missing
  fields vs. what the matched guideline requires, each cited. Clarification
  only.

**Out of scope — walled off, name/interface stubs only**
- **SCOPE-2.3** — autonomous next-step recommendation from patient data.
- **SCOPE-2.4** — guideline adjustment for local operational constraints not in
  source text.
- Enforcement: orchestrator **scope-classifier** routes SCOPE-2.3/2.4 intent
  straight to `scope_boundary` escalation (never answered). No agent has a tool
  that composes patient data + guidelines into a recommendation. Eval harness
  negative tests must be 0 violations or **build fails**.
- Extension seam (ARCH-026): `local-adaptation agent` = stub returning
  `capability_not_enabled`; `next-step-recommender` = reserved name / interface
  stub. `LOCAL_ADAPTATION_ENABLED=false` hard-wired.

**SCOPE-2.5 (narrow exception):** if a hospital constraint coincides with an
alternative **already in the retrieved guideline text**, surface it as reported
content + citation. Never reason to a substitution not in a retrieved source;
no documented alternative ⇒ `local_constraint_no_source_alt` escalation.

---

## 5. Agent topology (LangGraph)

| Agent | Access | Key tools | Never |
|---|---|---|---|
| **Orchestrator** | conversation, escalation (write), `patient_id` handle | `classify_scope`, `dispatch`, `assemble_response`, `apply_disclaimer`, `open_escalation` | retrieve, read PHI values, generate claims |
| **Retrieval** | Qdrant (guideline), `corpus` read. **No PHI.** | `hybrid_search`, `rerank`, `expand_context`, `get_chunk`, `get_version_status` | synthesize prose, decide final answer |
| **Patient-record** | one `patient_id`, field-filtered by `record_field_policy(role,purpose)`; every read audited | `list_record_fields`, `get_patient_fields` | return unrequested fields, retrieve guidelines, recommend |
| **Stage-classifier** (2.1) | Qdrant criteria + features from patient-record agent | `hybrid_search`, `evaluate_criteria`, `emit_classification` | recommend, infer absent features |
| **Missing-info** (2.2) | `field_index` (names) + authorized values + matched guideline | `list_record_fields`, `get_patient_fields`, `emit_missing_info` | guess values, recommend |
| **Guideline-synthesis** (1) | only this turn's chunk texts + the question. **No store access.** | `get_chunk` (restricted), `get_citation_metadata` | see raw PHI, use outside knowledge, directive text |
| **Citation-verifier** | this turn's snapshot + candidate segments | `nli_support_check`, `resolve_citation`, `lexical_overlap`, `wording_scan` | rewrite content to pass, add citations |
| **Escalation** | `hitl` write, conversation read | `create_escalation`, `enqueue_review`, `notify_reviewers` | answer the question, alter candidate |
| **local-adaptation** | none | none | STUB → `capability_not_enabled` |

Cross-cutting: chunk text is untrusted (data, not instructions); all model
calls via `LLMGateway`; only record/stage/missing-info agents may put patient
values in a prompt (authorized fields only); synthesis agent gets at most an
orchestrator-built feature summary. Scope classification, thresholds, conflict
flags, citation/quote checks are **deterministic code**.

---

## 6. Persistent memory

| Memory | Scope | Store | ACL |
|---|---|---|---|
| Session conversation (messages, retrieved chunk ids+scores, citations, grounding, HITL) | per `conversation_id` (1 user, ≤1 patient) | Postgres `memory.conversation`/`message` + Redis hot window | owner + reviewers of its escalations + admin(audit) |
| Per-patient context (cross-session) — structured, **non-diagnostic**: `guideline_match`, `stage_classification`, `missing_info`, `note`; provenance + citations + validity window | per `patient_id` | Postgres `memory.patient_context` | **same as the patient record** + audited |
| Rating history / IRR | per `result_id`, per `rater_id` | Postgres `eval.*` | reviewers, admin, compliance |
| Agent run checkpoints | per LangGraph thread | Postgres `memory.langgraph_checkpoint` | system/admin |
| Hot state (window, streaming, counters) | per `conversation_id` | Redis (TTL) | system |

Rules: recommendation-shaped `patient_context` writes are **rejected** at the
repo layer. No cross-patient reads (no API for it). Provisional entries
(`model_provisional`) become `reviewer_accepted`/`reviewer_edited` only via a
HITL accept; a reject rolls them back (`valid_to = now`). No free-form
long-term semantic memory in MVP.

---

## 7. HITL

### Escalation trigger codes
`low_confidence` · `no_guideline` (terminal, not held) · `grounding_failure` ·
`weak_support` (soft, released+queued) · `conflicting_sources` ·
`user_requested` · `phi_ambiguity` · `scope_boundary` (SCOPE-2.3/2.4 or
directive wording — never answered) · `local_constraint_no_source_alt` ·
`capability_not_enabled` · `stage_classification_uncertain` ·
`missing_critical_info` · `safety_filter` · `review_sampling` (released+queued).

No reviewer within `ESCALATION_SLA_MINUTES` ⇒ held + safe templated message,
**no auto-release**.

### Two independent axes, captured together in one `rating_round`

**Rank mode** — 11-domain rubric, 5-pt Likert, structured rows
(`rubric_rating`: result_id, rater_id, domain_code, score, rated_at). Ranking
alone does **not** change the shown answer or memory; it keeps the result in
the open queue until ≥3 distinct raters, then IRR per domain → archive.

**Accept axis** — effect on state:
| Action | Shown answer | conversation memory | patient_context | escalation/eval |
|---|---|---|---|---|
| **full_accept** | released as-is, `validated` | assistant turn committed `accepted` | provisional → `reviewer_accepted` | `resolution=accepted` |
| **partial_accept** | reviewer-edited version canonical; original + diff kept | edited turn committed, linked to original; removed spans logged as failures | only retained entries → `reviewer_edited`; rest expired | `resolution=partial`, `reason_code` required |
| **reject** | not released / retracted; safe fallback shown | `rejected` turn (question kept, body = rejection notice) | **all** provisional entries rolled back | `resolution=rejected`, `reason_code`; flagged as eval failure |
| **out_of_scope** | not released / retracted; out-of-scope notice shown | `out_of_scope` turn (question kept, body = out-of-scope notice) | **all** provisional entries rolled back (same mechanics as reject) | `resolution=out_of_scope`, `reason_code`; flagged as a **routing** failure, not a grounding one |

`out_of_scope` (DEVIATIONS #84) judges the *request* — this should never have
reached synthesis/escalation — distinct from `reject`, which judges an
*attempted answer*.

Every HITL action → immutable `audit_event`.

---

## 8. Multi-rater rubric evaluation

**Purpose (state it in-product + reports):** evidence-gathering toward
conformity for **in-scope capabilities only** (SCOPE-1.*, 2.1, 2.2). Shows a
measurement process exists. **Not** evidence for and **not** justification for
SCOPE-2.3/2.4 — those need a separate validation pathway.

**11 domains**, operator-supplied rubric (DEVIATIONS #112; each 5-pt Likert,
anchors stored in `rubric_domain`): `medical_consensus_alignment`*,
`question_comprehension`, `knowledge_recall`, `logical_reasoning`,
`irrelevant_content`, `information_omission`, `extent_of_harm`*,
`likelihood_of_harm`, `clear_communication`*, `local_context_understanding`*,
`demographic_bias`. (* = required minimum.)

**Workflow:** clinician rates → result enters **open review queue** → any
*other* distinct clinician rates independently → at **≥3 distinct raters**,
compute IRR **per domain** → **archive** with full rating history + IRR
snapshot. `<3` raters ⇒ stays visible in the open queue to any clinician.
Distinctness enforced by `rating_round` UNIQUE `(result_id, rater_id)` +
duplicate-account detection. Hard/adversarial cases go in the **same** queue.

**IRR metric: Krippendorff's alpha, ordinal difference function, per domain.**
Why: any number of raters + *variable* raters per item + ordinal 5-pt scale +
missing-data tolerant + chance-corrected. Secondary reported stats: Gwet's AC2
(skew robustness), ICC(2,k) (descriptive only). Per-result alpha is indicative
(small n); the evidentiary statistic is per-slice/corpus-level alpha over
archived results, **never pooling `auto_generated` + `clinician_submitted`**.
Bootstrapped CIs deferred.

---

## 9. Auto-generated hypothetical question set

Synthetic record → narrative **guideline-lookup hypothetical** ("what does the
guideline recommend for a patient presenting with X, Y, Z?" — never
"what should happen next").

- **No fabrication:** every clinical entity in the question must map to a field
  **value present** in the source record; deterministic validator rejects
  unmapped entities; `validator_report` stored.
- **Labels:** `provenance = auto_generated` (inherited by results, shown
  everywhere in the rubric workflow) **and**, separately, `expected_outcome ∈
  {well_supported, missing_info_expected, no_guideline_expected}`.
- **Hard cases on purpose:** sparse records (null out required fields) →
  `missing_info_expected`; scenarios absent from the corpus →
  `no_guideline_expected`. Gold re-check drops accidentally-answerable
  `no_guideline_expected` items.
- **Composition (documented decision): 60/20/20**
  well_supported / missing_info_expected / no_guideline_expected. Hard cases
  (missing + no_guideline combined = 40%) must not exceed 50%. Enforced by the
  set planner.
- **Diversity:** dedupe by embedding similarity (`QGEN_DEDUP_THRESHOLD`),
  minimum distinct topics/sections.
- **Uses:** (a) eval harness fixed test set; (b) seed the rubric/IRR queue when
  clinician volume is low. Auto vs clinician analysed **separately**.

---

## 10. Eval harness

Fixed synthetic test set + pinned config (model ids, thresholds,
`embedding_collection`, corpus snapshot). Metrics:
- **Retrieval:** precision@k / recall@k (k∈{5,8,24}); MRR/nDCG reported only.
- **Citation:** `citation_resolves_rate`, `citation_support_rate`,
  `citation_locus_accuracy` (±1 page, section prefix).
- **Expected-outcome pass/fail vs label:** `well_supported` (released +
  grounded + framed), `missing_info_expected` (asks for the specific fields,
  cited, no guess), `no_guideline_expected` (explicit "no guideline found",
  zero recommendation).
- **Scope safety (GATING):** `scope_boundary_violations` **must be 0**;
  `disclaimer_present_rate` **must be 100%**; `no_guideline_expected` pass
  **100%**.
- **Stage (2.1):** `stage_accuracy`, `stage_escalation_rate`.

Report broken out by `expected_outcome` and **separately** for
`auto_generated` vs `clinician_submitted`. CI fails on any gating breach or
sub-threshold retrieval/citation metric.

**Offline ablation studies (ARCH-040/041/043, PRD-109/110/111/112)** —
independent, additive, never part of `/query`, never CI-gated, run on
demand (`make <name>-ablation-report`) against the real corpus + the
harness's own `gold_relevant_chunks` pool:
- **ARCH-040** (`retrieval_tuning`): BM25/vector `alpha`×`k` sweep vs. gold
  chunks. Real result across live re-runs: no alpha robustly beats
  production RRF.
- **ARCH-041** (`model_ablation`): SapBERT/MedCPT+BM25 vs. RRF, brute-force
  in-memory ranking (these embeddings are never written to Qdrant). Real
  result: no candidate arm's bootstrap CI clears "robustly beats RRF."
- **ARCH-043** (`unified_ablation`, this phase): unifies 040/041's
  retrieval/embedding axis with a new Level 1 (present-only vs.
  all-assessed clinical-sign query construction) and Level 2 (vocabulary
  enrichment, reuses ARCH-042's mechanism unchanged) into one hierarchical,
  16-leaf-arm sweep; alpha is a sub-sweep inside Level 3's 3 dense-bearing
  arms, not its own separate tool. MRR@K primary metric, K/alpha
  config-driven, never hardcoded. Adds a **paired** bootstrap CI on deltas
  (same query indices resampled for both arms — the existing `bootstrap_ci`
  is unpaired). Per-query results persist to
  `results/ablation/<run_id>/{configuration.json,per_query_results.jsonl}`
  (file-based, not Postgres — a deliberate exception to the other two
  modules' PNG-only convention, since per-query row count is much larger).
  **Real result (2026-09-22, 238 questions, real SapBERT/MedCPT)** — the
  first ablation module to find CIs excluding zero rather than
  overlapping: BM25+SapBERT robustly beats BM25 in all 4 L1×L2 slices;
  BM25+MedCPT robustly underperforms it in 3 of 4; L1/L2 deltas also both
  distinguishable from zero. Not acted on — production retrieval
  untouched (DEVIATIONS.md #197). **Extended 2026-09-23** (operator-chosen
  Option B): an RRF-fusion counterpart for every dense-bearing arm
  (28 arms total) + a same-channel mechanism comparison — RRF modestly
  *amplifies* the SapBERT effect rather than just preserving it; MedCPT's
  mechanism comparison stays mixed. **Restructured 2026-09-23, superseding
  the above** (operator-supplied hierarchy) — Level 3 collapses to one
  continuous BM25/SapBERT weighted-rank-fusion sweep (`w_BM25` 0.0→1.0,
  11 points); MedCPT and RRF both dropped entirely (code removed, not
  unused); `ALL_ARMS` 28 → 4. Primary metric MRR@K → **Recall@K** (K still
  config-driven; MRR@K kept secondary). `alpha` renamed `bm25_weight`.
  Report back to 3 panels, Level 3 now a real line chart. Still not acted
  on — production retrieval untouched (DEVIATIONS.md #201). **Extended
  2026-09-23, statistical rigor** — every Level 1/2/3-endpoints delta now
  carries a two-sided bootstrap p-value (`paired_bootstrap_test`, same
  resample pass as the CI); new full Recall@k×`bm25_weight` grid
  (110 points, pooled across Level 1×Level 2) and an explicit **post-hoc**
  best-weight-vs-BM25 test (selects the empirically-best weight after
  seeing the data, compares it to `bm25_weight=1.0`) with a flagged
  winner's-curse caveat on its own CI/p-value. Persisted to a new
  `statistical_summary.json`. Report panels unchanged (DEVIATIONS.md #202).
  **Weight grid reduced 2026-09-25** to `w_BM25 ∈ {0.0, 0.2, …, 1.0}`
  (6 points, `ABLATION_BM25_WEIGHT_VALUES` default); grid 110 → 60 points;
  the 1,589-question run re-derived at the new grid, not re-run
  (DEVIATIONS.md #207, proposal §14).

---

## 11. Security / compliance essentials

- **In transit:** TLS at proxy; private compose net; Postgres/Redis/Qdrant
  authenticated. (mTLS everywhere = deferred.)
- **At rest:** host volume encryption (deployment prereq) + app-level envelope
  encryption (`CryptoProvider`, AES-256-GCM, KEK from `SECRETS_BACKEND`) for:
  `patient_record.payload_enc`, `patient.mrn_enc`, `patient_context` payloads,
  `message.content_enc`, `escalation.candidate_answer_enc`,
  `hitl_decision.edited_answer_enc`, `result.answer_enc`, audit text columns.
  (Full per-column field encryption = later hardening.)
- **RBAC:** API-layer route guards + data-layer RLS on `records.*` /
  `memory.patient_context` + `record_field_policy(role, purpose, field_path)` +
  per-agent tool allow-lists. `purpose` is required on every patient-scoped
  call and recorded in audit.
- **Audit:** append-only (`INSERT, SELECT` only; no UPDATE/DELETE grant),
  `prev_hash`/`row_hash` chain, encrypted query/response text + plaintext
  hashes for correlation. One event per query / retrieval (chunks + scores) /
  record access (field list) / answer (model id, response hash, grounding,
  outcome) / HITL action / ingest / config change / login. (External anchoring
  = deferred.)
- **Disclaimer layer:** non-removable `disclaimer` field on every response;
  API refuses to emit without it; deterministic output filter blocks directive
  phrasing / out-of-source dosing / patient-specific next-step plans →
  `safety_filter`.
- **PHI:** never leaves the deployment; only the self-hosted gateway is
  reachable for model calls; no training on PHI; logs are PHI-redacted;
  patient-record vectors OFF by default (`PATIENT_RECORD_VECTORS_ENABLED`).
- **Patient-data classes (ARCH-039):** `patient.data_class ∈ {synthetic,
  deidentified}` + `patient_record.dataset_id`. `deidentified` = attested (a
  complete `DATASET.md`) real de-identified dataset, handled identically to
  PHI. `guard_batch()` accepts `synthetic` (marker) / attested `deidentified`,
  else raises. EAV/long inputs are pivoted + mapped by `field_mapping.yaml`
  (`app/ingestion/eav.py`); `PatientDataSource` seam = `FileEavSource` now,
  `RestApiPullSource` stub. Dataset files are `.gitignore`d.

---

## 12. Key config vars

`LLM_GATEWAY_URL`, `MODEL_ID` (placeholder blocks answer path), `MODEL_ID_FALLBACKS`,
`MODEL_ID_VERIFIED`, `EMBEDDING_MODEL_ID`, `RERANKER_MODEL_ID`,
`EMBEDDING_BACKEND`, `CANDIDATE_K`/`FUSED_K`/`TOP_K` (40/24/8), `RRF_K` (60),
`RETRIEVAL_MIN_SCORE`/`SUPPORT_SCORE_FLOOR`/`MIN_SUPPORTING_CHUNKS`,
`GROUNDING_ENTAILMENT_MODE` (hybrid), `PATIENT_RECORD_VECTORS_ENABLED` (false),
`SAMPLE_GUIDELINES_DIR` (data/excerpt_guidelines), `GUIDELINES_ALLOW_SYNTHETIC` (false),
`INGEST_MIN_PARSE_QUALITY` (0.60), `PATIENT_RECORDS_DIR` (data/patient_records),
`RECORD_DOMAIN` (neonatal), `DEIDENTIFIED_ATTESTATION_REQUIRED` (true),
`LOCAL_ADAPTATION_ENABLED` (false, inert), `ESCALATION_SLA_MINUTES` (60),
`IRR_MIN_RATERS` (3), `QGEN_COMPOSITION` (60,20,20), `QGEN_DEDUP_THRESHOLD`,
`SECRETS_BACKEND` (file), `RETENTION_*`, `EVAL_MIN_*`, `AUTH_PROVIDER` (devjwt).

---

## 13. Where things live

- Requirements & non-goals → [PRD.md](PRD.md) (`PRD-###`, non-goals `PRD-NG-###`)
- Full design & rationale → [ARCHITECTURE.md](ARCHITECTURE.md) (`ARCH-###`, `SCOPE-#.#`)
- Excluded capabilities + what must be true first → [CDS-FUTURE.md](CDS-FUTURE.md)
- Every judgment call → [DEVIATIONS.md](DEVIATIONS.md) (append-only)
- Requirement → code/test/status matrix → `TRACEABILITY.md` (created Phase 1)
- Agent working rules → `CLAUDE.md` (created Phase 1)
