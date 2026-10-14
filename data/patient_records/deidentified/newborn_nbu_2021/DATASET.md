---
# Operator attestation (ARCH-039 / DEVIATIONS #33). Every field below must be
# filled with real values before `scripts.ingest_deidentified_records` will run.
# Replace every TODO_CONFIRM.
source: "The Clinical Information Network (CIN)"
collection_period: "2021-01-01 to 2024-12-31"
site: "Newborn units of the CIN hospitals"
deidentification_method: "Removal of direct identifiers, inclusion of integer surrogate key"
deidentification_standard: "Local IRB-approved de-identification"
consent_basis: "KEMRI Scientific and Ethics Review Unit (SERU) approval KEMRI/SERU/CGMR-C/307/4901"
licence: "CC BY-NC-SA 4.0"
attested_by: "Timothy Tuti - Primary Investigator"
attested_date: "2026-10-13" 
---

# Dataset: newborn_nbu_2021

De-identified / anonymised newborn dataset supplied by the operator for
development of the Hospital RAG Platform. Ingested via
`scripts/ingest_deidentified_records.py` using `field_mapping.yaml`.

**This is real patient-derived data.** It is handled as PHI end to end:
envelope-encrypted at rest, field-level RBAC + RLS, purpose-of-use, full audit,
never leaves the deployment, never used for training. It is **not** committed to
version control (`.gitignore`).

## Shape

- Format: EAV / long — columns `key, field_name, field_value, context`.
- **1,500 patients** (2026-10-14 export, replacing the earlier ~40,871-patient
  extract), one assessment snapshot each; `(key, field_name)` is unique.
  300 patients in each of the five `triage_category` values.
- `key` is an integer surrogate id (no name / MRN in the source).
- **2026-09-18 export update:** rows with `field_value == "NA"` are now
  included explicitly for a field that was **not assessed** for a patient
  (previously such fields were simply absent from that `key`'s rows — both
  forms mean the same thing and are handled identically by
  `app/ingestion/eav.py`'s existing null-literal handling; no mapping change
  needed). This is distinct from `field_value == "FALSE"`, which means the
  sign/intervention/medication **was assessed and found absent/not given**.
  See DEVIATIONS #138/#139 for how this three-way distinction (assessed-
  positive / assessed-negative / not-assessed) is preserved end to end.

## Known residual identifiers

- `admission_date_time` is retained as a full timestamp (C3, DEVIATIONS #36).
  It is kept for `day_of_life` reconstruction, vitals timestamping, and
  ordering, and is protected like every other field. If your de-identification
  basis requires date shifting or generalisation, add a `date_shift` /
  `generalise_month` transform for `admission_date_time` in `field_mapping.yaml`
  and re-ingest.

## Source variables (39) by context

| context | variables                                                                                                                                        |
|---|--------------------------------------------------------------------------------------------------------------------------------------------------|
| demographics | age_days, birth_weight (kg), gestational_age (weeks), sex, weight_now (kg)                                                                       |
| encounter_details | admission_date_time, birth_date_time *(added 2026-10-14; not mapped, never stored)*, care_setting (all "NBU"), triage_category (Critical / Substantial / Moderate / Low / Minimal) |
| vitals | capillary_refill (1–7 s), heart_rate, oxygen_saturation, respiratory_rate, temparature *(sic — °C)*                                              |
| history_examination | apnoea, bulging_fontanelle, central_cyanosis, convulsions, crackles, difficulty_breathing *(added 2026-10-14)*, difficulty_feeding, floppy, grunting, indrawing, irritable (all TRUE/FALSE) |
| interventions | antibiotics, cpap, feeds, fluids, kmc, oxygen, phenobarbital (all TRUE/FALSE)                                                                    |
| medication | ampicillin, ceftazidime, ceftriaxone, gentamicin, penicillin (all TRUE/FALSE)                                                                    |
| **maternal_risk_factors** *(added 2026-09-18)* | maternal_infection, prom (all TRUE/FALSE) — `prom` = premature/prolonged rupture of membranes (confirmed by operator, 2026-09-18) |

## Mapping

See `field_mapping.yaml`. Transforms of note: `birth_weight`/`weight_now`
kg→g; `sex` normalised to `male`/`female`; `triage_category` kept as given
(a literal `"None"` would still become null) and used only for the reviewers'
triage-agreement question, never for guideline retrieval (ARCH-046);
`birth_date_time` is deliberately unmapped, so it is dropped at ingestion; `temparature` misspelling mapped to `temperature_c`; the 23 boolean
sign/intervention/medication variables expand into
`examination_findings[]` / `interventions[]` / `medications[]`.
`date_of_birth` is derived (`admitted_at − age_days`, approximate).
`maternal_infection`/`prom` expand into `maternal_risk_factors[]` (schema
v1.4.0) — a maternal-history sign list, kept separate from the newborn's own
`examination_findings[]`.

## Limitations

- Single snapshot per patient (no trends).
- `care_setting` is constant in this extract.
- The 300-per-category triage balance is a property of this extract, not of
  the source population.
- 6 `admission_date_time` values carry a date with no time (parsed as
  midnight UTC).
- Many fields are partially populated — this is expected and exercises the
  SCOPE-2.2 missing-information path.
