"""CLI: python -m scripts.seed_triage_review_queue
[--per-category N] [--dataset-id ID] [--seed N] [--areas-path PATH]
[--concepts-path PATH] [--mortality-path PATH]
(ARCH-046, PRD-114; DEVIATIONS.md #300).

Builds the clinician review queue from `--per-category` fresh de-identified
records per triage category (default 40; Critical, Substantial, Moderate,
Low, Minimal), each asked all four area questions in
`data/query_areas.yaml`: 5 x 40 x 4 = 800 items. The assessment item of each
record also carries the triage-agreement question, using the attested
`data/triage_mortality.yaml`.

Run it before regenerating the ablation holdout, so the holdout draws only
on records the queue did not use.

Needs a real Postgres, Qdrant with the ingested corpus, and the LLM gateway:
one or two real pipeline calls per item. Commits per item, so an interrupted
run keeps what it made and a re-run finishes unfinished records first.
"""

from __future__ import annotations

import argparse
import sys

from app.db.session import session_scope
from app.eval.auto_seed import TRIAGE_REVIEW_PER_CATEGORY, run_triage_review_queue


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Triage-stratified review queue (ARCH-046)")
    parser.add_argument(
        "--per-category",
        type=int,
        default=TRIAGE_REVIEW_PER_CATEGORY,
        help="records per triage category; each record yields one item per area",
    )
    parser.add_argument("--areas-path", type=str, default="data/query_areas.yaml")
    parser.add_argument("--concepts-path", type=str, default="data/clinical_concepts.yaml")
    parser.add_argument("--mortality-path", type=str, default=None)
    parser.add_argument("--dataset-id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    print(
        f"[triage-review] building {args.per_category} record(s) per triage category, four "
        "items per record, one or two real pipeline calls per item -- this will take a while.",
        file=sys.stderr,
    )
    with session_scope() as session:
        created = run_triage_review_queue(
            session,
            per_category=args.per_category,
            dataset_id=args.dataset_id,
            seed=args.seed,
            areas_path=args.areas_path,
            concepts_path=args.concepts_path,
            mortality_path=args.mortality_path,
        )
    print(f"[triage-review] created {len(created)} new review item(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
