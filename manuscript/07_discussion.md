# Discussion

<!-- Draft 2026-10-27. TRIPOD-LLM 19a–19g (item numbers in comments). Every number matches 04_results.md / numbers.md. [PLACEHOLDER] = depends on the clinician ratings; written so the surrounding text stands whichever way they fall. -->

## Principal findings

<!-- 19a -->

[PLACEHOLDER: one or two sentences on the primary outcome: which rubric domains and query areas clinicians rated highly or poorly, and how consistent the ratings were.]

The pipeline released a cited answer for 760 of 800 review items (95.00%). Its escalations were concentrated in one area: 39 of the 40 escalated items were antibiotic-course questions, and in the ablation all 47 questions whose answer cited no passage were also antibiotic-course questions. Antibiotic-course questions state age, gestation, weight and care setting rather than clinical findings, and the passages they need are often dosing tables. We did not analyse why answers were escalated, so we cannot say which step failed.

[PLACEHOLDER: triage agreement: how far clinicians agreed with the latent-class triage category, overall and across categories, and how consistent they were.]

In the exploratory retrieval analysis, lexical scoring retrieved the platform's cited passages far better than semantic scoring with a biomedical encoder (MRR@10 0.864 against 0.376). How the query was built mattered much less: concept enrichment added about 0.01, and stating present or all assessed findings made no clear difference. A query-blind popularity ranking reached 0.803, and only lexically weighted queries beat it, by 0.053 to 0.070.

## Comparison with previous studies

<!-- 19a -->

Mwaniki et al. used the same 11-domain rubric to compare general-purpose LLMs with clinicians on primary-care vignettes written for the study in Kenya [REF: Mwaniki et al. 2026]. Our study differs in three ways: the questions came from routine newborn-unit records, the model could answer only from four indexed guidelines, and every statement had to quote a retrieved passage. [PLACEHOLDER: how the domain ratings compare with theirs, particularly for alignment with medical consensus, harm and local context.] The strength of lexical retrieval is consistent with benchmarks in which BM25 remains a strong baseline for specialised text [CITE: BEIR or similar zero-shot retrieval benchmark; clinical retrieval comparisons of BM25 and dense encoders].

## Interpretation of the retrieval analysis

The gold sets were the passages the platform itself cited, and the platform's retriever combines lexical and dense search. The analysis therefore measures how well each query construction recovers the evidence the platform used, not whether that evidence was clinically correct. The gold sets were also concentrated: 51 distinct passages across 933 questions, with the most-cited passage in 89.32% to 99.14% of each area's sets. This explains why the popularity ranking scored highly and why assessment results barely changed with w (0.716–0.733). Against the reference sets curated by the investigators, lexically weighted queries ranked a relevant passage first almost every time (MRR@10 0.987–0.999), which suggests the main result does not depend on the platform's own citations. Recall@10 was at most 0.663, so even in the best condition a top-10 cut-off missed about a third of the passages the platform cited.

## Strengths and limitations

<!-- 19b, 19c -->

The study used real routine records from eight hospitals, a published 11-domain rubric already used in Kenya, three independent ratings per item, a pipeline that runs entirely on local hardware, and an analysis of retrieval that varied one component at a time. All items were kept, whatever the pipeline's outcome.

It also has limitations.
- **Design.** It was offline, with one model, one quantisation and greedy decoding. We did not regenerate answers to test consistency, and there was no comparator for the clinician ratings.
- **Questions.** Questions were templated from records rather than asked by clinicians, and the four areas cover only part of the management of suspected infection.
- **Model training data.** The developers have not published a training-data cut-off, so we cannot rule out that the model saw the WHO and NICE guidance during training. The grounding gate limits the effect on released statements, which must quote a retrieved passage.
- **Retrieval analysis.** The headline metric and three configuration choices were made after inspecting results on the same questions, and one person curated the reference sets. These results are exploratory.
- **Triage agreement.** Raters saw the assigned category and its mortality before judging it, which may push scores towards agreement. They saw only the findings that had been assessed, not all 14 features the model used.
- **Data.** Keeping one admission per unique combination of the 14 features, and sampling 300 per triage category, gives a case mix that differs from routine admissions: rarer combinations of findings are over-represented, and the five triage categories are equally represented. First vital signs were missing for 8.79% to 21.93% of records, and missing fields were left out of the questions. The records came from eight hospitals in one national network, and the guidelines are in English.

[PLACEHOLDER: limitations from the ratings themselves, e.g., number of raters, completeness, agreement.]

**Fairness.** The rubric's demographic-bias domain gives one view of fairness [PLACEHOLDER: its ratings]. We did not compare answers across sex or other groups, and the sample was not designed to do so.

## Intended use and implementation

<!-- 19d, 19e, 19f -->

The pipeline is intended as a reference tool for clinicians in newborn units. It reports and cites what local and international guidelines say for a given newborn, carries a fixed disclaimer, and does not act on its own: decisions stay with the clinician. When retrieval confidence is low, or no statement can be grounded, it returns "no guideline found" or escalates the question rather than answering from general knowledge. Missing or not-assessed fields are left out of the question, so a sparse record produces a less specific question rather than an assumed finding. In this study, questions were built automatically from the record. In practice, a clinician would need to check that the question reflects the newborn's findings, and to read the cited passages, which requires the clinical training to interpret guideline text.

## Future research

<!-- 19g -->

Next steps are:
- a prospective evaluation with questions asked by clinicians during care;
- comparison with other models and with clinicians answering the same questions;
- a retrieval reference standard annotated independently of the pipeline, by more than one clinician, so that retrieval metrics can be linked to clinician ratings;
- analysing the reasons for escalation, to improve antibiotic-course retrieval;
- evaluation in other hospitals and for other conditions.

[PLACEHOLDER: next steps for the triage categories, depending on agreement.]

## Conclusions

[PLACEHOLDER: two or three sentences on the primary result.] In an exploratory analysis, lexical scoring retrieved the cited guideline passages much better than semantic scoring, while other changes to query construction made little difference.
