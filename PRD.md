# PRD.md — Hospital RAG Platform

**Status:** Phase 0 draft (Checkpoint 0 pending review)
**Last updated:** 2026-08-27
**Owner:** Engineering (senior full-stack/ML)
**Related docs:** [ARCHITECTURE.md](ARCHITECTURE.md) · [ARCHITECTURE-ESSENTIALS.md](ARCHITECTURE-ESSENTIALS.md) · [CDS-FUTURE.md](CDS-FUTURE.md) · [DEVIATIONS.md](DEVIATIONS.md)

---

## 1. Summary

A Retrieval-Augmented Generation (RAG) platform for hospital use that answers
clinician questions with **grounded, citable** content drawn from two source
classes:

1. **External clinical/government documents** — clinical guidelines, protocols,
   formularies (PDF and similar).
2. **Internal patient-level records** — structured records supplied via flat
   file (CSV/JSON or EAV/long) or API. **Treated as PHI by default, everywhere,
   including development.** Patient data is either **synthetic** (project-
   generated) or an **operator-attested de-identified dataset** (admitted only
   with a complete `DATASET.md` attestation and then handled exactly as PHI —
   see ARCHITECTURE.md ARCH-039 / DEVIATIONS.md #33). Real, non-de-identified
   patient data is never ingested, requested, or accepted.

The system pairs hybrid retrieval (dense + BM25 + reranking) with a
**multi-agent orchestration layer** (tool use, persistent memory, human-in-the-
loop escalation). It **reports and cites source material**; it does **not**
produce independent clinical recommendations.

A secondary purpose of the MVP is to **begin generating structured evidence**
(a multi-domain rubric plus inter-rater reliability statistics) toward future
regulatory conformity work — but **only for the in-scope capabilities** defined
in §6 and ARCHITECTURE.md. See non-goals (§7) and [CDS-FUTURE.md](CDS-FUTURE.md).

---

## 2. Users & personas

| Persona | Description | Primary needs |
|---|---|---|
| **Clinician (query user)** | Doctor/nurse/pharmacist asking guideline questions, optionally in the context of one synthetic patient record. | Fast, grounded answers; visible citations; clear "no guideline found" when applicable; ability to escalate. |
| **Reviewer (clinician-rater)** | Clinician performing structured evaluation of system outputs via the rubric and the accept/partial/reject/out-of-scope axis. | Efficient review queue; unambiguous rubric; sees provenance and expected-outcome labels; independent rating. |
| **Admin** | Operates the deployment: ingestion, corpus/version management, user/role management, config. | Ingestion tooling; audit visibility; RBAC management; safe config of models/thresholds. |
| **(Implicit) Compliance/quality lead** | Consumes audit logs and rubric/IRR reports. | Immutable audit trail; separable auto-generated vs clinician-submitted evidence. |

Out of persona scope for MVP: patients, external API consumers, multi-hospital
tenants.

---

## 3. Goals & success criteria

| ID | Goal | Success signal (MVP) |
|---|---|---|
| PRD-G1 | Grounded answers with verifiable citations | ≥ 95% of emitted citations resolve to a real chunk and support the adjacent claim on the fixed synthetic eval set. |
| PRD-G2 | No hallucinated guidance | 100% of `no_guideline_expected` eval cases return an explicit "no guideline found" with zero recommendation content. |
| PRD-G3 | Correct missing-information behaviour | ≥ 90% of `missing_info_expected` eval cases produce a specific request for the missing field(s) rather than a guess. |
| PRD-G4 | Enforced scope boundary | 0 outputs in evaluation cross into excluded CDS capabilities (§7, CDS-FUTURE.md). |
| PRD-G5 | Auditability | 100% of retrievals and model responses produce an immutable audit record. |
| PRD-G6 | Evidence generation works | The rubric + multi-rater + IRR workflow runs end-to-end on ≥ 1 batch of outputs, auto-generated and clinician-submitted analysed separately. |
| PRD-G7 | Self-hosted, reproducible deploy | `docker compose up` brings up the full stack against synthetic data with no external network dependency for core flows. |

Non-goals for success measurement: answer "helpfulness" as an absolute,
latency SLAs beyond a soft target (§5), production-scale load.

---

## 4. Functional requirements

IDs are permanent. New requirements append; existing IDs are never renumbered.

### 4.1 Ingestion & corpus

| ID | Requirement |
|---|---|
| PRD-001 | Ingest external clinical/government documents (PDF minimum) into a versioned guideline corpus, preserving document identity, version, and effective date. |
| PRD-002 | Ingest internal patient records via **flat file** (CSV/JSON) against a fixed, documented schema. |
| PRD-003 | Ingest internal patient records via **API** against the same schema. |
| PRD-004 | Every ingested guideline document is chunked per the strategy in ARCHITECTURE.md, with each chunk retaining: document ID, version, section path, page number(s), and character offset span within the source document. |
| PRD-005 | Re-ingesting a new version of a guideline document supersedes the prior version without deleting it; superseded content remains resolvable for citations already issued. |
| PRD-006 | The system ships a **synthetic patient-record generator** (CSV/JSON output) and a set of sample public guideline documents (or a fetch script) for development. No real PHI is ever ingested. |
| PRD-113 | *(Checkpoint 9 — `LAYOUT-INGESTION-PROPOSAL.md`, sub-phases 9a + 9b approved 2026-09-28; DEVIATIONS.md #213-#216)* Guideline PDFs are parsed **layout-aware**: layout labels, reading order, fonts and positions, table cell structure, and **local OCR** for content without a text layer. Running headers/footers are removed before chunking; heading levels come from numbering, structure and font; tables stay atomic with flattened headers; **flowchart decision logic** (boxes, arrows, Yes/No labels) is preserved as a node/edge graph whose verified edges are citable. Only source text (text layer, OCR, operator-attested corrections) and deterministic serializations of verified structure are citable; model-generated text never is. OCR'd numeric content and chunks of low-parse-quality documents are held from retrieval until an admin confirms them against the page image. Evident source errors can be corrected only by an operator-attested manifest entry that keeps every number and unit, and every citation of corrected text shows the correction. |

### 4.2 Retrieval, citation & grounding

| ID | Requirement |
|---|---|
| PRD-010 | Hybrid retrieval over the guideline corpus: dense vector search + BM25 (lexical) + a reranking stage. |
| PRD-011 | Every answer segment that makes a claim about guideline content carries a citation of the form: **document ID + version + section/page + chunk offset** (minimum). |
| PRD-012 | A **grounding check** runs on every generated answer: each claim-bearing segment is verified against the retrieved chunk(s) it cites; unsupported segments are flagged and either removed or the whole answer is escalated (per policy in ARCHITECTURE.md). |
| PRD-013 | On **low-confidence retrieval** (score/þreshold or too-few supporting chunks) the system does not answer from general knowledge; it escalates (HITL) or returns "no sufficiently grounded guideline content found". |
| PRD-014 | On **conflicting sources** (materially different recommendations retrieved for the same question) the system does not silently pick one; it escalates (HITL) and surfaces both with citations. |
| PRD-015 | If **no relevant guideline** is retrieved for the question asked, the system says so explicitly and generates no recommendation. This applies to real and hypothetical/synthetic queries alike. |
| PRD-016 | Citations are machine-verifiable: the stored chunk text and offsets are re-checkable against the source document. |

### 4.3 Multi-agent orchestration, memory & tools

| ID | Requirement |
|---|---|
| PRD-020 | A multi-agent orchestration layer (LangGraph) coordinates named agent roles with tool use, per the topology in ARCHITECTURE.md. |
| PRD-021 | Each agent has an explicitly bounded tool set and data-access scope; the patient-record-reading agent is the only agent with PHI field access, and only within the requester's authorization. |
| PRD-022 | **Persistent memory** stores: (a) per-session conversation history including retrieved chunk IDs and HITL actions; (b) per-patient structured, non-diagnostic context, cross-session, access-controlled identically to the patient record; (c) reviewer rating history. Storage locations per ARCHITECTURE.md. |
| PRD-023 | Long agent runs are checkpointed so they survive worker restarts and can be inspected/resumed. |
| PRD-024 | Memory writes that would constitute an independent clinical recommendation are rejected. No cross-patient memory blending. |

### 4.4 Human-in-the-loop (HITL)

| ID | Requirement |
|---|---|
| PRD-030 | The system supports **HITL escalation** with the concrete triggers enumerated in ARCHITECTURE.md (low confidence, grounding failure, conflicting sources, explicit user request, PHI ambiguity, CDS-boundary queries, hospital-constraint-without-documented-alternative, and others). |
| PRD-031 | HITL supports **rank mode**: a structured multi-domain rubric evaluation (see 4.5). |
| PRD-032 | HITL supports an **accept axis** with four actions — **full accept**, **partial accept**, **reject**, **out of scope** — each with a defined effect on the answer shown, conversation memory, and per-patient context (defined in ARCHITECTURE.md §"HITL modes"). `reject` judges an attempted answer as wrong/ungrounded/unsafe; `out of scope` judges the request itself as one the system should never have attempted to answer (DEVIATIONS.md #84). Rank mode and the accept axis are independent and both are captured. |
| PRD-033 | Every HITL action writes an immutable audit record and updates escalation/queue state. |

### 4.5 Structured multi-rater evaluation (regulatory-evidence oriented)

| ID | Requirement |
|---|---|
| PRD-040 | Rank mode uses an **11-domain rubric**, each domain scored on a **5-point Likert scale**. Domains and definitions are the operator-supplied rubric fixed in ARCHITECTURE.md §14.1 (DEVIATIONS.md #112) and include at minimum: alignment with medical consensus, possible extent of harm, clear communication, understanding of local context. |
| PRD-041 | Rubric scores are stored as **structured data** (domain, score, rater ID, timestamp, result ID), queryable and auditable — not free text. Free-text comments are an optional adjunct field. |
| PRD-042 | Multi-rater workflow: (1) a clinician rates an output; (2) the rated-but-unarchived output enters an **open review queue**; (3) any *other* clinician can pull and rate it independently; (4) once **≥ 3 distinct clinicians** total have rated it, an **inter-rater reliability** statistic is computed **per rubric domain**; (5) the result is then **archived** with full rating history and IRR scores; (6) until the minimum rater count is met, the result stays visible in the open queue. |
| PRD-043 | "Distinct clinician" is enforced (no self-re-rating, no duplicate-account rating counting twice toward the minimum). |
| PRD-044 | The IRR statistic is a single named metric suitable for ordinal multi-rater data with variable raters per item (chosen and justified in ARCHITECTURE.md). |
| PRD-045 | Every question/result carries a **provenance tag**: `auto_generated` or `clinician_submitted`. This tag is visible everywhere the result appears in the rubric workflow and in any downstream evidence report. |
| PRD-046 | Rubric/IRR results for `auto_generated` and `clinician_submitted` items must be **analysable separately** and are not pooled by default in conformity-evidence reporting. |
| PRD-047 | Hard/adversarial cases (`missing_info_expected`, `no_guideline_expected`) are **included** in the review queue, not excluded. |
| PRD-048 | Purpose statement, surfaced in-product and in reports: this rubric/IRR data is an evidence-gathering step toward conformity for **in-scope capabilities only**. It is not evidence for, and must not be used to justify, the capabilities excluded in [CDS-FUTURE.md](CDS-FUTURE.md). |

### 4.6 Capability scope (see ARCHITECTURE.md §"Capability scoping" for full detail and SCOPE-* IDs)

| ID | Requirement |
|---|---|
| PRD-050 | **In scope — grounded guideline reporting/synthesis:** a clinician may ask a hypothetical ("what does the guideline say for a patient presenting with X, Y, Z?") and receive a synthesized answer drawn from retrieved guideline chunks, with citations, framed as **reported guideline content** ("Guideline X recommends…"), never as directive advice ("You should…"). This framing is enforced in every guideline-touching agent's prompt template, not just the UI. |
| PRD-051 | **In scope — patient stage-of-care classification (non-directive):** given a patient record, infer the patient's current stage of in-hospital care **where guideline documents define clear, extractable stage criteria**, as a classification task grounded in retrieved criteria, with citations to the criteria used. |
| PRD-052 | **In scope — missing-information identification (non-directive):** identify and request specific pertinent information missing from the patient record relative to what the matched guideline(s) require. Clarification-seeking only. |
| PRD-053 | **Out of scope — walled off:** autonomous next-step recommendation from patient data (synthesizing patient data + guidelines into "what should happen next" beyond reporting matched text). Not implemented, even partially. See [CDS-FUTURE.md](CDS-FUTURE.md). |
| PRD-054 | **Out of scope — walled off:** guideline adjustment based on local operational constraints (e.g. substituting a recommendation because a service/drug is unavailable) when the substitution is not itself present in the source text. Not implemented, even partially. See [CDS-FUTURE.md](CDS-FUTURE.md). |
| PRD-055 | **Narrow exception:** if a hospital-specific constraint coincides with an alternative **already stated in the retrieved guideline text** (e.g. a documented second-line option), the system may surface that alternative as reported guideline content with citation. It must not reason to a substitution not present in a retrieved source. If no documented alternative exists, route to HITL escalation. |
| PRD-056 | ARCHITECTURE.md defines an **extension seam** (a named but unimplemented agent role / marked interface stub) for the excluded capabilities so they could be added later without redesign. No logic for them is implemented in this build. |
| PRD-057 | **Implemented, bounded to eval-only use (Checkpoint 7 approved 2026-09-19 — see [DEVIATIONS.md](DEVIATIONS.md) #143/#144/#146/#148/#149; mechanism detailed in [PHASE7-PROPOSAL.md](PHASE7-PROPOSAL.md)):** deterministic clinical-concept labeling from patient-record field values against an **operator-authored, attestation-gated vocabulary file** (a threshold/rule is a clinical judgment call — it must come from the operator, never be invented by a coding session, same governance pattern as `DATASET.md`'s attestation). A recorded vital crossing an operator-defined threshold is labeled with its named concept (e.g. "tachypnoea") — the same vocabulary entry also carries operator-curated **synonyms** (spelling variants and phrase-level equivalents, e.g. "tachypnea", "fast breathing") for query-expansion use — for use as an **internal retrieval-signal only** — it is never quoted or framed as reported guideline content in any answer (only a retrieved chunk's own quote can support an answer segment, per CLAUDE.md §3 rule 3), and never a recommendation of what to do about it. Must never label a field that was not actually assessed/documented for that patient (depends on the assessed/not-given/never-documented distinction in ARCH-039's `field_index`/`extract_features`). Labeling logic is deterministic/curated (no LLM judgement of what counts as a concept), matching the existing `app.records.criteria` pattern. **Its only caller is the Phase 7 orchestration ablation** (`app.eval.orchestration_ablation`, eval-only) — not wired into any agent, route, or the live orchestrator graph; production use remains a separate, unapproved, later decision. |

### 4.7 Auto-generated hypothetical question set

| ID | Requirement |
|---|---|
| PRD-060 | A utility converts synthetic patient records into **narrative-form hypothetical questions** phrased as **guideline-lookup hypotheticals** (scope-1 framing), suitable for feeding the RAG pipeline. |
| PRD-061 | The narrative generation follows the same no-fabrication discipline as production output: every clinical detail in a generated question traces back to a field actually present in the source synthetic record. No embellished or inferred symptoms, timelines, or history. |
| PRD-062 | Generated questions are diverse: coverage across distinct guideline topics/sections, deduplicated, not repeated variations of one case. |
| PRD-063 | Every auto-generated question/result is tagged `auto_generated` (per PRD-045) and additionally labelled with an **expected-outcome type**: `well_supported`, `missing_info_expected`, or `no_guideline_expected`. The expected-outcome label is separate from and additional to the provenance tag. |
| PRD-064 | The generator deliberately produces adversarial/edge cases: (a) questions built from **sparse/incomplete** synthetic records (missing guideline-relevant fields) to test missing-info behaviour; (b) questions that **target gaps in the ingested corpus** to test "no guideline found" behaviour. |
| PRD-065 | Generated-set composition target is **60/20/20** across `well_supported` / `missing_info_expected` / `no_guideline_expected`. Hard cases (`missing_info_expected` + `no_guideline_expected` combined) must not exceed 50% of the set. This composition is a documented decision in ARCHITECTURE.md. |
| PRD-066 | Auto-generated Q/A pairs may be used to (a) populate the eval harness fixed test set and (b) seed the rubric/IRR review queue when clinician-submitted volume is low — but auto-generated and clinician-submitted results remain separately analysable (PRD-046) and are not pooled by default when reported as conformity evidence. |

### 4.8 Evaluation harness

| ID | Requirement |
|---|---|
| PRD-070 | An eval harness scores **retrieval precision/recall** and **citation accuracy** against a **fixed synthetic test set** (which includes the auto-generated hypothetical set). |
| PRD-071 | The harness scores each case against its **expected-outcome label** (pass/fail against expectation), not only abstract output quality. |
| PRD-072 | Harness runs are reproducible: pinned corpus snapshot, pinned config (model IDs, thresholds). |
| PRD-073 | Harness reports break results out by expected-outcome type and separately for `auto_generated` vs `clinician_submitted` subsets. |
| PRD-109 | *(backfilled — Checkpoint 6 approved 2026-09-17, `PHASE6-PROPOSAL.md`; see DEVIATIONS.md #192)* Empirical calibration of hybrid retrieval fusion (BM25 weight × depth-k sweep) against the auto-generated eval-question set's known gold chunks; reproducible offline PNG report, no production fusion change. |
| PRD-110 | *(backfilled — approved 2026-09-17, `PHASE2-EMBEDDING-ABLATION-PROPOSAL.md`; see DEVIATIONS.md #192)* Offline ablation of biomedical embedding models (SapBERT, MedCPT) combined with BM25 via client-side RRF, against the same gold-chunk eval-question set; evidence-gathering only, no production fusion/ingestion change. |
| PRD-111 | *(backfilled — Checkpoint 7 approved 2026-09-19, `PHASE7-PROPOSAL.md`; see DEVIATIONS.md #192)* Single-stage vs. two multi-step orchestration variants (criteria-reuse query augmentation, always grounded; operator-vocabulary query augmentation, attestation-gated) against the same gold-chunk eval-question set; evidence-gathering only, no orchestrator/graph change. |
| PRD-112 | *(this phase — `UNIFIED-ABLATION-PROPOSAL.md`; DEVIATIONS.md #192-#202; restructured 2026-09-23, operator-supplied hierarchy; statistical rigor extended same day)* A single hierarchical offline ablation unifying PRD-109/110/111's separate mechanisms into one sweep: **Level 1** (present-only vs. all-assessed clinical-sign query construction), **Level 2** (vocabulary/concept enrichment, reusing PRD-111's mechanism), **Level 3** (a single continuous BM25/SapBERT weighted-rank-fusion sweep, `w_BM25 ∈ {0.0, 0.1, ..., 1.0}` — MedCPT and RRF fusion considered and dropped from scope). **Recall@K is the primary metric** (MRR@K kept secondary); K and the BM25 weight are config-driven, never hardcoded. Per-query results and a reproducibility snapshot (model id/version, K/weight grids, seed, timestamp) persist to disk; comparisons use a **paired** bootstrap CI **and a two-sided bootstrap p-value** on the delta (same query indices resampled for both sides, both statistics from the same resample pass — not independent-sample CIs). Statistical comparisons required at minimum: Level 1 and Level 2 paired deltas (Δ Recall@K, 95% CI, p-value); Level 3's full Recall@k-by-`bm25_weight` grid (k ∈ {2,4,...,20}, pooled across Level 1 × Level 2); and an explicit **post-hoc** test of whether BM25 weighting helps at all — the empirically-best `bm25_weight` (selected after seeing the data) vs. plain BM25 (`bm25_weight=1.0`), with its CI/p-value explicitly documented as a data-dependent selection (winner's-curse caveat), not a pre-registered comparison. Evidence-gathering only — no change to the production `/query` path. |

### 4.9 Security, privacy & safety

| ID | Requirement |
|---|---|
| PRD-080 | Every patient-record field is treated as **PHI by default**, in all environments. |
| PRD-081 | **No real, non-de-identified PHI** is ever ingested, requested, or accepted. Patient data is either project-generated **synthetic** data (carrying the `synthetic-generator-v1` marker) or an **operator-attested de-identified dataset** (complete `DATASET.md` attestation + explicit intent, then handled exactly as PHI — ARCH-039 / DEVIATIONS.md #33). Any real-looking batch with neither marker nor attestation is hard-rejected. Dataset files are never committed to version control. |
| PRD-082 | **Encryption in transit** for all service-to-service and client-to-service communication. |
| PRD-083 | **Encryption at rest** for PHI, including database storage and backups; application-level encryption for PHI free-text fields and stored prompt/response text. |
| PRD-084 | **Field-level access control**: agents and users receive least-privilege field subsets of patient records; access is checked at the API layer and the data layer. |
| PRD-085 | **Immutable audit logging**: append-only records of who queried what, when, which records/chunks were retrieved (with scores), which model/version answered, and what was returned. No update/delete of audit rows. |
| PRD-086 | **RBAC** with at least the roles **clinician**, **reviewer**, **admin** (plus service accounts). |
| PRD-087 | **Disclaimer layer**: every response is wrapped with a non-removable disclaimer stating the content is reported guideline material for clinician reference only, not medical advice, and that clinical judgement is required. All guideline-touching agent prompts instruct deference to the human clinician. |
| PRD-088 | The system generates **no independent diagnostic or treatment content**. An output filter blocks directive phrasing and any dosing/therapy specifics not present in a cited source. |
| PRD-089 | **No training/fine-tuning on PHI**; no PHI sent to any model or service outside the configured self-hosted LLM gateway; no telemetry containing PHI. |
| PRD-090 | Document text ingested from PDFs is treated as **untrusted content**: agents must not follow instructions embedded in retrieved chunks (prompt-injection hardening). |

### 4.10 Platform, config & deployment

| ID | Requirement |
|---|---|
| PRD-100 | Backend in Python (FastAPI). |
| PRD-101 | The LLM model identifier is read from an **environment variable / config file** with a placeholder default. **No model name/version is hardcoded.** If a provided model name cannot be verified against current documentation, the build flags it rather than guessing. |
| PRD-102 | The LLM integration targets a **self-hosted LLM gateway** with **fallback routing** between configured models. |
| PRD-103 | Embedding model and reranker model identifiers are likewise config-driven with placeholder defaults (extension of PRD-101; see [DEVIATIONS.md](DEVIATIONS.md) #10). The reranker runs **locally** (`RERANKER_BACKEND=local`), decided rather than routed through the LLM gateway (DEVIATIONS.md #44); the embedding backend remains config-selectable (`local` \| `gateway` \| `stub`). |
| PRD-104 | Vector store: one self-hosted store, chosen and justified in ARCHITECTURE.md. |
| PRD-105 | Redis + Celery (self-hosted) for async ingestion and long-running agent tasks. |
| PRD-106 | Docker / docker-compose for local self-hosted deployment; core flows work with no external network dependency. |
| PRD-107 | React web frontend implementing the query interface, citation display, and all three HITL interaction modes. |
| PRD-108 | Configuration is via env/config files, never hardcoded secrets; secrets are supplied through a configurable secrets backend. |

---

## 5. Non-functional requirements

| ID | Requirement |
|---|---|
| PRD-NFR-1 | **Soft** latency target: a grounded answer for a typical guideline question returns in ≤ ~15 s p50 on the reference dev machine (documented, not contractually enforced in MVP). |
| PRD-NFR-2 | The system degrades safely: on any component failure in the answer path (retrieval, rerank, grounding check, gateway) it returns "cannot produce a grounded answer" or escalates — never an ungrounded answer. |
| PRD-NFR-3 | Reproducibility: pinned dependency versions; pinned config for eval runs. |
| PRD-NFR-4 | Observability sufficient for debugging: structured logs, basic metrics, request/trace IDs propagated across API → workers → agents. (Full tracing stack is out of scope — see ARCHITECTURE.md self-critique.) |
| PRD-NFR-5 | Data minimisation: agents receive the minimum patient fields required for the current task; patient-record vectorization is off by default. |
| PRD-NFR-6 | Portability: runs on a single host via docker-compose; no managed-cloud service is required for core functionality. |

---

## 6. Constraints (carried from the build brief, apply to every phase)

| ID | Constraint |
|---|---|
| PRD-C1 | No real, non-de-identified PHI, ever. Patient data is **synthetic or operator-attested de-identified** (DEVIATIONS.md #33); de-identified data is handled exactly as PHI end to end. |
| PRD-C2 | PHI-aware architecture: encryption at rest & in transit, field-level access control, immutable audit logging are core, not stretch. |
| PRD-C3 | No diagnostic/treatment generation. Surface and cite retrieved material only. Disclaimer layer required. Agents defer clinical judgement to the human user. |
| PRD-C4 | Grounding is enforced, not assumed (citation format, grounding check, low-confidence/conflict handling as HITL triggers). |
| PRD-C5 | Production-grade patterns, minimal feature surface. When "production-ready" and "MVP" conflict, prefer fewer features built correctly. No silent scope expansion. |
| PRD-C6 | Config, not hardcoding, for the LLM (and, by extension, embedding/reranker) model identifiers. Flag unverifiable model names. |
| PRD-C7 | Phase checkpoints are hard stops. No chaining phases without explicit confirmation. |
| PRD-C8 | The boundary in [CDS-FUTURE.md](CDS-FUTURE.md) is hard. If any later phase would require an excluded capability to function, stop and flag — do not work around it. |

---

## 7. Non-goals (explicit)

Each non-goal is a tracked decision.

| ID | Non-goal | Rationale |
|---|---|---|
| PRD-NG-001 | The system does **not** generate diagnoses, treatment plans, or independent clinical recommendations. | Safety; regulatory posture; brief constraint #3. It reports and cites source material only. |
| PRD-NG-002 | The system does **not** perform autonomous next-step recommendation from patient data (SCOPE-2.3). | Crosses into clinical decision support proper; unresolved regulatory/liability questions. See CDS-FUTURE.md. |
| PRD-NG-003 | The system does **not** adjust guidelines for local operational constraints beyond surfacing alternatives already in the source text (SCOPE-2.4 / narrow exception SCOPE-2.5). | Clinical substitution judgement; hospitals route this through formal processes. See CDS-FUTURE.md. |
| PRD-NG-004 | The system does **not** replace clinician judgement or act as an authority of record. | It is a reference/retrieval aid with a mandatory disclaimer layer. |
| PRD-NG-005 | **No Android / native mobile app** in the MVP. Web (React) only. Android is revisited only on explicit request after the web app is reviewed (Phase 5). | Scope control. |
| PRD-NG-006 | **No EHR write-back, order entry, or any action on hospital systems.** Read-only with respect to patient data. | Safety; scope. |
| PRD-NG-007 | **No multi-tenant / multi-hospital SaaS**, no per-tenant isolation model. Single-deployment, single-institution MVP. | Scope control. |
| PRD-NG-008 | **Not a general-purpose medical chatbot.** No open-domain answers; if it is not in the retrieved corpus, the system says "no guideline found". | Grounding constraint #4. |
| PRD-NG-009 | **No model training or fine-tuning**, and certainly none on PHI. | Privacy; scope. |
| PRD-NG-010 | The rubric/IRR evidence is **not** a regulatory submission, clearance, or validation. It is a measurement process only, for in-scope capabilities only. | See PRD-048, CDS-FUTURE.md. |
| PRD-NG-011 | **No cohort / population / cross-patient analytics.** One patient record per session; cohort queries are rejected. | Scope; PHI minimisation. (See DEVIATIONS.md #15.) |
| PRD-NG-012 | **No real-time streaming ingestion / HL7 / FHIR feed** in MVP. File and simple API ingestion only. | Scope control. |

---

## 8. Assumptions & dependencies

| ID | Assumption / dependency |
|---|---|
| PRD-A1 | A self-hosted LLM gateway endpoint is available (or stubbed) and reachable from the backend; its model catalogue is provided via config. The build does not depend on any specific hosted model provider. |
| PRD-A2 | Sample public guideline PDFs are obtainable for development (bundled or fetched by script). Their licences permit local development use. |
| PRD-A3 | The internal patient-record schema is stable enough to fix in Phase 1 (currently v1.3.0); synthetic data and any operator-attested de-identified dataset are mapped onto that schema. Real, non-de-identified records are never used. |
| PRD-A4 | Reviewers (clinician-raters) are available in sufficient number to reach the 3-distinct-rater minimum for at least a sample of results; where they are not, the auto-generated set seeds the queue (PRD-066). |
| PRD-A5 | The reference deployment is a single host with a modern GPU or an acceptable CPU fallback for embeddings/reranking (documented in README). This is load-bearing for the reranker specifically, since it is decided to run locally on this host rather than via the gateway (DEVIATIONS.md #44). |
| PRD-A6 | Regulatory classification, clinical governance sign-off, and formal clinical validation are **out of engineering scope** and are prerequisites for anything in CDS-FUTURE.md. |

---

## 9. Open product questions (tracked; not blockers for Phase 0)

| ID | Question |
|---|---|
| PRD-Q1 | Guideline versioning policy: when two versions are both retrievable, is "latest effective version" always authoritative, or is that itself a HITL decision? (Leaning: prefer latest, flag the older as superseded, escalate on material difference.) |
| PRD-Q2 | Reviewer SLA and after-hours behaviour when no reviewer is available for an escalation. (Current interim decision: hold + safe templated message; see DEVIATIONS.md #14.) |
| PRD-Q3 | Retention period for audit logs, per-patient context memory, and rubric data. |
| PRD-Q4 | Whether partial-accept edits by a reviewer feed back into any future prompt/example set (risk of drift). Current stance: logged for eval only, not used as few-shot examples. |
| PRD-Q5 | Minimum corpus size / coverage before the system is considered usable for a given clinical area. |
