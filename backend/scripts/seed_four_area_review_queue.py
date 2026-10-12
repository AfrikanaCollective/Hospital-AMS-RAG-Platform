"""CLI: python -m scripts.seed_four_area_review_queue
[--per-area N] [--dataset-id ID] [--seed N] [--areas-path PATH]
[--concepts-path PATH] (ARCH §14.2 review queue, PRD-047; DEVIATIONS.md #280).

Tops up the clinician review queue until every attested area in
`data/query_areas.yaml` (assessment, investigations, severity / risk
classification, antibiotic course) has `--per-area` items (default 10).
Each item is one fresh de-identified record, asked the same four-area
question the ablation holdout uses, with that area's per-guideline cap
(`app.eval.auto_seed.run_four_area_review_queue`):

- an `EvalQuestion` tagged `purpose: "review_queue"`, so the unified
  ablation never selects it, and an open `Result` the review wizard shows;
- the multi-stage (vocabulary-enriched) answer alongside, when
  `data/clinical_concepts.yaml` is attested and enrichment changes the query;
- escalations stay on the item; nothing is filed in the HITL escalation queue.

A record used here is never reused by the holdout or the single-topic
review queue, and vice versa.

Needs a real Postgres, Qdrant with the ingested corpus, and the LLM gateway:
one or two real pipeline calls per item. Commits per item, so an interrupted
run keeps what it made and a re-run resumes the top-up.
"""

from __future__ import annotations

import argparse
import sys

from app.db.session import session_scope
from app.eval.auto_seed import FOUR_AREA_REVIEW_PER_AREA_TARGET, run_four_area_review_queue


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Four-area review-queue items (DEVIATIONS #280)")
    parser.add_argument(
        "--per-area",
        type=int,
        default=FOUR_AREA_REVIEW_PER_AREA_TARGET,
        help="review items per area to top up to",
    )
    parser.add_argument("--areas-path", type=str, default="data/query_areas.yaml")
    parser.add_argument("--concepts-path", type=str, default="data/clinical_concepts.yaml")
    parser.add_argument("--dataset-id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    print(
        f"[four-area-review] topping up to {args.per_area} review item(s) per area, one fresh "
        "record and one or two real pipeline calls per item -- this may take a while.",
        file=sys.stderr,
    )
    with session_scope() as session:
        created = run_four_area_review_queue(
            session,
            per_area_target=args.per_area,
            dataset_id=args.dataset_id,
            seed=args.seed,
            areas_path=args.areas_path,
            concepts_path=args.concepts_path,
        )
    print(f"[four-area-review] created {len(created)} new review item(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
