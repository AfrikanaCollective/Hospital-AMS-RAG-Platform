# TRIPOD-LLM checklist

<!-- Draft 2026-10-27. Items and wording follow the official fillable checklist ("The TRIPOD-LLM Statement", Nat Med, doi:10.1038/s41591-024-03425-5, Supplementary Table 2: https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41591-024-03425-5/MediaObjects/41591_2024_3425_MOESM3_ESM.pdf), read 2026-10-27. Item text is paraphrased in the table; use the official PDF for submission. Page numbers are filled in at assembly; until then "Where" gives the section. -->

**Study classification.** Research design: E (LLM evaluation) and H (LLM evaluation in healthcare settings). LLM tasks: QA (long-form question answering) and IR (information retrieval). No LLM development, fine-tuning or updating.

**Status key:**
- **Addressed:** reported in the current drafts.
- **Partly:** reported, with a gap noted.
- **Pending ratings:** structure and placeholders in place; needs the clinician ratings.
- **Not written:** the section does not exist yet.
- **N/A:** the item does not apply to this design or task; say so where the journal expects it.

Files: Title = `01_title.md`; Background = `02_background.md`; Discussion = `07_discussion.md`; Methods = `03_methods.md`; Results = `04_results.md`; AF1 = Additional file 1 (`05_supplement_S1_methods.md`); AF2 = Additional file 2 (`06_supplement_S2_prompts.md`); AF3 = Additional file 3 (not yet built).

## Summary

| Status | Items |
|---|---|
| Addressed | 1, 3a, 3b, 4, 5a, 5b, 5c, 5d, 5e, 6a, 6c, 6d, 7a, 7c, 7e, 12, 13, 14a, 14b, 14c, 14d, 14e, 15, 16a, 16d |
| Partly | 7b, 7d, 8a, 8b, 8c, 9a, 9b, 14f, 16b, 16c |
| Pending ratings | 17 (clinician part); 19a and 19b drafted with placeholders for the ratings |
| Addressed in draft Discussion | 19c, 19d, 19e, 19f, 19g |
| Not written | 2 |
| N/A | 6b, 6e, 10, 11, 18 |

## Title and abstract

| Item | Requirement | Status | Where | Notes |
|---|---|---|---|---|
| 1 | Identify the study as developing, fine-tuning and/or evaluating an LLM; specify the task, target population and outcome | Addressed | Title (`01_title.md`) | Title chosen by the PI: names the evaluation, a retrieval-augmented large language model approach, guideline-grounded answers, newborns with possible serious bacterial infection, the timing (hospital admission), and Kenya. |
| 2 | Abstract: see TRIPOD-LLM for Abstracts | Not written | — | Write last, within 350 words, against the separate TRIPOD-LLM for Abstracts checklist (not yet retrieved). Depends on the clinician results. |

## Introduction

| Item | Requirement | Status | Where | Notes |
|---|---|---|---|---|
| 3a | Healthcare context, use case and rationale, with existing approaches and models | Addressed | Background ¶1–4 | Many [CITE] markers to fill. |
| 3b | Target population, intended use in the care pathway and intended users | Addressed | Background ¶5; Methods, Study design | Newborns with suspected infection; clinicians in newborn units; reports and cites, no independent advice. |
| 4 | Objectives, including whether development, fine-tuning or validation | Addressed | Background ¶5 | Evaluation only; primary and four secondary objectives, including triage agreement; no hypothesis. |

## Methods

| Item | Requirement | Status | Where | Notes |
|---|---|---|---|---|
| 5a | Data sources for training, tuning and evaluation, with rationale | Addressed | Methods, Data sources; AF1 S1.1 | Evaluation data only (CIN records). The model's training data are described by the developer (link in AF1 S1.4). |
| 5b | Relevant data points; quantitative and qualitative description; source, language, country | Addressed | Methods, Data sources; Table 1; AF1 S1.1–S1.2 | Record variables, guideline corpus (4 documents, 153 retrievable chunks), Kenya, English. |
| 5c | Dates of oldest and newest text in development and evaluation data | Addressed | Methods, Data sources and Pipeline | Records January 2021–December 2024; guideline editions 2021–2026; training cut-off stated as not published. |
| 5d | Pre-processing and quality checking, and whether similar across corpora, institutions and groups | Addressed | Methods, Data sources and Questions; AF1 S1.1, S1.2, S1.5 | Extract selection, chunking, record-to-question templates. Same processing across hospitals; not examined by socio-demographic group (Discussion). |
| 5e | Missing and imbalanced data; reasons for omitting data | Addressed | Methods, Missing data; AF1 S1.1, S1.6 | 237 excluded for missingness; not-assessed fields omitted; triage-stratified sampling; 47 questions without a gold set. |
| 6a | LLM name, version and last date of training | Addressed | Methods, Pipeline; AF1 S1.4 | qwen3.6:35b, Q4_K_M, released 16 April 2026; training cut-off not published by the developers. |
| 6b | LLM development: architecture, training, fine-tuning, alignment | N/A | Methods, Pipeline | No development or fine-tuning ("used without fine-tuning"). The checklist note asks for a reference to developer information: cited (Qwen Team 2026). |
| 6c | How text was generated: prompt engineering, consistency, inference settings | Addressed | Methods, Pipeline; AF1 S1.4; AF2 | Prompts verbatim; temperature 0.0, top-p 1.0, top-k 1, seed 42, 4,096-token cap, context 32,768, thinking off. Output consistency across repeated runs was not tested; say so in Discussion. |
| 6d | Initial and post-processed output | Addressed | Methods, Pipeline; AF1 S1.4; Results, Pipeline outcomes | Unstructured text with citations and quotes; grounding gate; released, escalated or "no guideline found". |
| 6e | Classification: rationale, probabilities and thresholds | N/A | — | Task is not classification. Retrieval thresholds (0.30 score; low-confidence rule) are nevertheless reported (Methods; AF1 S1.3). |
| 7a | Metrics for quality of generative outputs against gold standards | Addressed | Methods, Clinician evaluation, Retrieval ablation and Outcomes | 11-domain PATH rubric; MRR@10 and Recall@K against platform-citation gold sets and PI-curated reference sets. |
| 7b | Relevance of metrics to the downstream task; correlation with human evaluation | Partly | Methods, Retrieval ablation | Gold sets are the platform's own citations, which "measures recovery of the evidence the platform used, not clinical correctness" (stated). Retrieval metrics were not correlated with clinician ratings (different records); state in Discussion. |
| 7c | Outcome definition, how predictions were calculated, inference date for closed-source LLMs, metrics | Addressed | Methods, Outcomes, Statistical analysis; AF1 S1.8, S1.11 | Open-weights model; generation dates 14–16 October 2026 given anyway. |
| 7d | Assessors' qualifications, instructions, demographics and inter-assessor agreement | Partly | Methods, Clinician evaluation and Statistical analysis; AF1 S1.7, Table S1 | Instructions (rubric, anchors, triage text) and Krippendorff's α planned. **[PLACEHOLDER]** rater number, qualifications, experience, demographics, training. |
| 7e | Comparison with other LLMs, humans or benchmarks | Addressed | Methods, Clinician evaluation and Retrieval ablation | Clinician evaluation: no comparator (stated). Ablation: query-blind popularity baseline. |
| 8a | Annotation: how text was labelled, with guidelines and examples | Partly | Methods, Retrieval ablation | Primary gold sets were derived automatically from citations (no annotation). The PI-curated reference sets were annotated, but how they were built is not described. **[TODO: curation procedure for the reference sets, in AF1 S1.8.]** |
| 8b | Number of annotators, proportion with more than one annotator, agreement | Partly | — | Reference sets: one annotator (PI), so no agreement statistic; state this. Clinician ratings are covered under 7d. |
| 8c | Annotators' background and experience | Partly | — | **[TODO: PI's clinical/informatics background, one sentence.]** |
| 9a | Prompt design, curation and selection | Partly | Methods, Pipeline; AF1 S1.4; AF2 | Says fixed templates written during platform development, not tuned on study records. **[TODO: how they were designed and revised, and by whom.]** |
| 9b | Data used to develop the prompts | Partly | Methods, Pipeline | Says not the study records. **[TODO: what was used instead, e.g., synthetic records.]** |
| 10 | Pre-processing before summarisation | N/A | — | Task is not summarisation. |
| 11 | Instruction tuning or alignment | N/A | — | No instruction tuning or alignment. |
| 12 | Compute | Addressed | Methods, Software and compute; AF1 S1.9 | H200 GPU; about 16 hours for 844 generations; encoders on CPU. |
| 13 | Ethics approval and consent | Addressed | Methods, Ethics | KEMRI SERU (KEMRI/SERU/CGMR-C/307/4901); consent waived for records; raters consented. |
| 14a | Funding and role of funders | Addressed | Methods, Funding | Move to Declarations at assembly. |
| 14b | Conflicts of interest | Addressed | Methods, Competing interests | Move to Declarations at assembly. |
| 14c | Protocol access or statement that none was prepared | Addressed | Methods, Study design | "No study protocol was published." |
| 14d | Registration | Addressed | Methods, Study design | "The study was not registered." |
| 14e | Data availability | Addressed | Methods, Data and code availability | Restricted; requests to the KEMRI-Wellcome DGC. |
| 14f | Code availability | Partly | Methods, Data and code availability | Repository URLs given. **[TODO: release tag or commit.]** Manuscript summary scripts are not stated as available (PI decision). |
| 15 | Patient and public involvement | Addressed | Methods, Patient and public involvement | None. |

## Results

| Item | Requirement | Status | Where | Notes |
|---|---|---|---|---|
| 16a | Flow of data through the study, with numbers with and without the outcome or label | Addressed | Results, Study flow; Figure 1 | 44,626 → 1,500 → 200 / 1,300 → 933. Rated-item counts are a **[PLACEHOLDER]** in Figure 1. |
| 16b | Characteristics overall, by data source or setting and by split, with key dates and sample size | Partly | Results, Record characteristics; Table 1 | Reported overall and by study set. Not reported by hospital: the ingested records carry no hospital field (`data/record_schema.json`), and no explanatory sentence is to be added (PI, 2026-10-27). |
| 16c | Distribution of important clinical variables between development and evaluation data | Partly | Table 1 | No development data (no fine-tuning). Table 1 compares the clinician and ablation sets; triage category is the only outcome-related variable. |
| 16d | Number of participants and outcome events in each analysis | Addressed | Results, Study flow and Pipeline outcomes; Figure 1 | 800 items, 933 questions; ratings per analysis to add with results. |
| 17 | Performance on prespecified metrics and human evaluation | Pending ratings | Results; Tables 2–3; Figure 3 | Ablation reported in full. Clinician ratings, agreement, accept decision and triage agreement are **[PLACEHOLDER]**. Note that MRR@10 as the headline was chosen after inspecting results (stated). |
| 18 | LLM updating | N/A | — | No updating. Optionally state that no model was updated during the study. |

## Discussion

| Item | Requirement | Status | Where | Notes |
|---|---|---|---|---|
| 19a | Interpretation, including fairness, against objectives and previous studies | Pending ratings | Discussion, Principal findings; Comparison with previous studies; Fairness | Include fairness: the rubric's demographic-bias domain; records from 8 Kenyan hospitals only. |
| 19b | Limitations: biases, statistical uncertainty, generalisability | Pending ratings | Discussion, Strengths and limitations | Points already flagged: gold sets from the platform's own citations; post hoc headline metric and configuration choices; single model; no repeat-run consistency; raters saw the triage label and mortality before rating; raters saw only assessed findings, not all 14 features; training cut-off unknown, so prior exposure to WHO and NICE text cannot be ruled out; one curator for reference sets. |
| 19c | Known challenges with the data: representation, missingness, harmonisation, bias | Addressed | Discussion, Strengths and limitations (Data) | Missingness of vital signs (8.79–21.93%); deduplication to unique feature combinations changes the case mix; equal sampling per triage category is not the population distribution. |
| 19d | Intended use: input, end user, autonomy and human oversight | Addressed | Discussion, Intended use and implementation | Partly in Background ¶5; restate in Discussion (clinician-facing, reports and cites, escalation, no autonomy). |
| 19e | Handling of poor-quality or unavailable input in implementation | Addressed | Discussion, Intended use and implementation | "No guideline found", escalation, omitted missing fields. |
| 19f | User interaction with input data and expertise required | Addressed | Discussion, Intended use and implementation | Users are clinicians; who builds the question from the record. |
| 19g | Next steps for research | Addressed | Discussion, Future research | E.g., prospective evaluation, other models, other sites, linking retrieval metrics to clinician ratings. |
