"""Pipeline outcomes on the clinician-review items (aggregates only).

For every triage-stratified review item (`review_queue` purpose, the 800
items of DEVIATIONS.md #300/#303), counts the pipeline's observed outcome
and whether the answer cited any passage, by query area and by triage
category. Also reports the number of rating rounds recorded so far and the
model id(s) in the results' config snapshots. Reads no encrypted columns.
JSON on stdout.

    docker compose run --rm --no-deps -T -e PYTHONPATH=/app \
      -v ./backend/app:/app/app:ro -v ./manuscript/scripts:/app/manuscript_scripts:ro \
      api python manuscript_scripts/review_outcomes.py \
      | sed -n '/^{/,$p' > manuscript/derived/review_outcomes.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict

from sqlalchemy import func, select

from app.db.models.eval import EvalQuestion, RatingRound, Result
from app.db.session import session_scope
from app.eval.auto_seed import FOUR_AREA_REVIEW_PURPOSE, TRIAGE_REVIEW_QUEUE_DESIGN


def main() -> int:
    meta = EvalQuestion.generator_meta
    with session_scope() as session:
        rows = session.execute(
            select(
                meta["area"].astext,
                meta["triage_category"].astext,
                Result.observed_outcome,
                Result.citations,
                Result.config_snapshot,
                Result.queue_state,
                Result.id,
            )
            .join(Result, Result.eval_question_id == EvalQuestion.id)
            .where(
                meta["purpose"].astext == FOUR_AREA_REVIEW_PURPOSE,
                meta["queue_design"].astext == TRIAGE_REVIEW_QUEUE_DESIGN,
            )
        ).all()
        result_ids = [r[6] for r in rows]
        n_rounds = session.execute(
            select(func.count())
            .select_from(RatingRound)
            .where(RatingRound.result_id.in_(result_ids))
        ).scalar_one()

    by_area: dict[str, Counter] = defaultdict(Counter)
    by_triage: dict[str, Counter] = defaultdict(Counter)
    no_citation: Counter = Counter()
    models: Counter = Counter()
    states: Counter = Counter()
    for area, triage, outcome, citations, config, state, _ in rows:
        by_area[area][outcome] += 1
        by_triage[triage][outcome] += 1
        if not citations:
            no_citation[area] += 1
        models[(config or {}).get("model_id") or "not recorded"] += 1
        states[state] += 1

    out = {
        "n_items": len(rows),
        "outcome_by_area": {a: dict(c) for a, c in sorted(by_area.items())},
        "outcome_by_triage": {t: dict(c) for t, c in sorted(by_triage.items())},
        "outcome_total": dict(sum(by_area.values(), Counter())),
        "no_citation_by_area": dict(no_citation),
        "model_ids": dict(models),
        "queue_states": dict(states),
        "rating_rounds": n_rounds,
    }
    json.dump(out, sys.stdout, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
