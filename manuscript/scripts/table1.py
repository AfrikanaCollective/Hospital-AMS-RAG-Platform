"""Table 1: characteristics of the records in each study set (aggregates only).

Sets:
- all ingested de-identified records (`newborn_nbu_2021`);
- the clinician review queue (records with a triage-stratified
  `review_queue` item);
- the ablation holdout (one `four-area-v2` `ablation_holdout` question per
  record) and, within it, the analysed records (non-empty gold set, the
  ablation's own selection `fetch_four_area_questions`).

Variables: sex, day of life, gestational age, birth weight, triage category,
care setting, and the first recorded vital signs. Continuous variables are
reported as median (IQR) with the number missing; categorical variables as
n (%), with "not recorded" counted as its own level.

Records are decrypted in memory only. Output is aggregates only (no record
ids, no free text), as JSON on stdout. Small cells are reported as exact
counts (PI decision 2026-10-26: one-cell suppression could be undone from
the column total, and "not recorded" is a missing-data marker, not an
identifier).

Run in the api container (needs the database and the KEK):

    docker compose run --rm --no-deps -T -e PYTHONPATH=/app \
      -v ./backend/app:/app/app:ro -v ./manuscript/scripts:/app/manuscript_scripts:ro \
      api python manuscript_scripts/table1.py \
      | sed -n '/^{/,$p' > manuscript/derived/table1.json

(the sed drops the SECRETS_BACKEND dev-key warning the app logs to stdout)
"""

from __future__ import annotations

import json
import sys
import uuid
from collections import Counter

import numpy as np
from sqlalchemy import select

from app.db.models.eval import EvalQuestion, Result
from app.db.session import session_scope
from app.eval.auto_seed import (
    FOUR_AREA_REVIEW_PURPOSE,
    FOUR_AREA_TEMPLATE,
    TRIAGE_REVIEW_QUEUE_DESIGN,
)
from app.eval.deidentified_source import load_deidentified_records
from app.eval.triage import TRIAGE_CATEGORIES
from app.eval.unified_ablation.runner import fetch_four_area_questions
from app.schemas.enums import ExpectedOutcome, Provenance

DATASET_ID = "newborn_nbu_2021"
NOT_RECORDED = "Not recorded"

CONTINUOUS = {
    "day_of_life": ("encounter", "day_of_life"),
    "gestational_age_weeks": ("encounter", "gestational_age_weeks"),
    "birth_weight_g": ("encounter", "birth_weight_g"),
    "temperature_c": ("vitals", "temperature_c"),
    "heart_rate_bpm": ("vitals", "heart_rate_bpm"),
    "resp_rate_bpm": ("vitals", "resp_rate_bpm"),
    "spo2_percent": ("vitals", "spo2_percent"),
}


def _first_vitals(record: dict) -> dict:
    vitals = record.get("vitals") or []
    if not vitals:
        return {}
    dated = [v for v in vitals if v.get("recorded_at")]
    return min(dated, key=lambda v: v["recorded_at"]) if dated else vitals[0]


def _value(record: dict, section: str, field: str) -> float | None:
    source = (
        _first_vitals(record) if section == "vitals" else (record.get(section) or {})
    )
    value = source.get(field)
    return float(value) if value is not None else None


def _continuous(records: list[dict], section: str, field: str) -> dict:
    values = [v for r in records if (v := _value(r, section, field)) is not None]
    if not values:
        return {"n": 0, "missing": len(records)}
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    return {
        "n": len(values),
        "missing": len(records) - len(values),
        "median": float(med),
        "q1": float(q1),
        "q3": float(q3),
    }


def _categorical(labels: list[str], order: list[str] | None = None) -> dict:
    counts = Counter(labels)
    levels = order or sorted(k for k in counts if k != NOT_RECORDED)
    if counts.get(NOT_RECORDED):
        levels = [*levels, NOT_RECORDED]
    total = len(labels)
    out = {}
    for level in levels:
        n = counts.get(level, 0)
        out[level] = {"n": n, "pct": 100 * n / total if total else None}
    return out


def _clean(value: object) -> str:
    return str(value).strip() if value not in (None, "") else NOT_RECORDED


def _describe(records: list[dict]) -> dict:
    out: dict = {"n_records": len(records)}
    out["sex"] = _categorical([_clean(r.get("sex")).capitalize() for r in records])
    out["triage_category"] = _categorical(
        [_clean((r.get("encounter") or {}).get("triage_category")) for r in records],
        order=list(TRIAGE_CATEGORIES),
    )
    out["care_setting"] = _categorical(
        [_clean((r.get("encounter") or {}).get("care_setting")) for r in records]
    )
    out["vitals_snapshots_per_record"] = dict(
        sorted(Counter(len(r.get("vitals") or []) for r in records).items())
    )
    for name, (section, field) in CONTINUOUS.items():
        out[name] = _continuous(records, section, field)
    return out


def _holdout_record_ids(session) -> set[uuid.UUID]:
    rows = session.execute(
        select(EvalQuestion.source_record_id).where(
            EvalQuestion.expected_outcome == ExpectedOutcome.WELL_SUPPORTED,
            EvalQuestion.provenance == Provenance.AUTO_GENERATED,
            EvalQuestion.generator_meta["template_version"].astext
            == FOUR_AREA_TEMPLATE,
            EvalQuestion.generator_meta["purpose"].astext == "ablation_holdout",
        )
    ).scalars()
    return {r for r in rows if r is not None}


def _review_record_ids(session) -> set[uuid.UUID]:
    meta = EvalQuestion.generator_meta
    rows = session.execute(
        select(EvalQuestion.source_record_id)
        .join(Result, Result.eval_question_id == EvalQuestion.id)
        .where(
            meta["purpose"].astext == FOUR_AREA_REVIEW_PURPOSE,
            meta["queue_design"].astext == TRIAGE_REVIEW_QUEUE_DESIGN,
        )
    ).scalars()
    return {r for r in rows if r is not None}


def main() -> int:
    with session_scope() as session:
        records = dict(load_deidentified_records(session, dataset_id=DATASET_ID))
        holdout = _holdout_record_ids(session)
        review = _review_record_ids(session)
        analysed = {
            q.source_record_id
            for q in fetch_four_area_questions(session)
            if q.source_record_id
        }

    sets = {
        "all_ingested": set(records),
        "review_queue": review,
        "holdout": holdout,
        "holdout_analysed": analysed,
        "holdout_excluded_no_gold": holdout - analysed,
    }
    checks = {
        "unresolved_ids": {k: len(v - set(records)) for k, v in sets.items()},
        "review_holdout_overlap": len(review & holdout),
        "analysed_subset_of_holdout": analysed <= holdout,
    }
    out = {
        "dataset_id": DATASET_ID,
        "checks": checks,
        "sets": {
            name: _describe([records[i] for i in sorted(ids, key=str) if i in records])
            for name, ids in sets.items()
        },
    }
    json.dump(out, sys.stdout, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
