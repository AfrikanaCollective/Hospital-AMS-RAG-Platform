"""CLI: python -m scripts.generate_ablation_holdout_questions
[--question-mode four_area|single] [--per-area N | --target-count N]
[--dataset-id ID] [--seed N] [--areas-path PATH] (PRD-112 / ARCH-043;
DEVIATIONS.md #199, #258; operator request 2026-09-23).

`--question-mode four_area` (default, DEVIATIONS.md #258, #264) tops up
until every attested area in `data/query_areas.yaml` (assessment,
investigations, severity / risk classification, antibiotic course) has
`--per-area` usable questions (default 200). Each record is used for ONE
area only, with one real pipeline call and that area's per-guideline cap.
`--question-mode single` keeps the original single-topic generation below.

Tops up the ablation-only `eval_question` pool to `--target-count`
(default: `settings.ablation_holdout_target_count`, env
`ABLATION_HOLDOUT_TARGET_COUNT`) via
`app.eval.auto_seed.run_ablation_holdout_generation` — writes ONLY
`EvalQuestion` rows, never a `Result`/review-queue item. A de-identified
record consumed here is excluded from the rubric review queue's own
`run_auto_seed_review_queue` (and vice versa) automatically, via the
shared `EvalQuestion.provenance`+`source_record_id` exclusion check both
pipelines read.

Needs a real Postgres + a real LLM gateway (one real pipeline call per new
question — this is not free or instant) and already-ingested de-identified
records covering at least `--target-count` records. If the configured
dataset doesn't have enough unused records, ingest more first:

    python -m scripts.ingest_deidentified_records \\
      --dataset-dir data/patient_records/deidentified/<dataset> \\
      --attest-deidentified --persist --limit <n>

Long-running at scale: a target in the thousands means thousands of real
gateway calls, likely hours, not minutes. Commits per scenario (same
crash-resilience convention as `run_auto_seed_review_queue`,
DEVIATIONS.md #115) — an interrupted run keeps whatever it already
generated; re-running this script simply resumes the top-up.
"""

from __future__ import annotations

import argparse
import sys

from app.config import get_settings
from app.db.session import session_scope
from app.eval.auto_seed import (
    FOUR_AREA_PER_AREA_TARGET,
    run_ablation_holdout_generation,
    run_four_area_holdout_generation,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ablation-only calibration question generation (PRD-112/ARCH-043)"
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=None,
        help="default: settings.ablation_holdout_target_count",
    )
    parser.add_argument("--question-mode", choices=("four_area", "single"), default="four_area")
    parser.add_argument(
        "--per-area",
        type=int,
        default=FOUR_AREA_PER_AREA_TARGET,
        help="four_area mode: usable questions per area to top up to",
    )
    parser.add_argument("--areas-path", type=str, default="data/query_areas.yaml")
    parser.add_argument("--dataset-id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    settings = get_settings()
    target_count = (
        args.target_count
        if args.target_count is not None
        else settings.ablation_holdout_target_count
    )

    if args.question_mode == "four_area":
        print(
            f"[ablation-holdout] four-area: topping up to {args.per_area} usable question(s) "
            "per area, one record and one real pipeline call per question -- this may take "
            "a long time.",
            file=sys.stderr,
        )
        with session_scope() as session:
            created = run_four_area_holdout_generation(
                session,
                per_area_target=args.per_area,
                dataset_id=args.dataset_id,
                seed=args.seed,
                areas_path=args.areas_path,
            )
        print(f"[ablation-holdout] four-area: created {len(created)} new question(s)")
        return 0

    print(
        f"[ablation-holdout] topping up to {target_count} ablation-only question(s) "
        f"(dataset_id={args.dataset_id!r}) -- this may take a long time (one real gateway "
        "call per new question).",
        file=sys.stderr,
    )

    with session_scope() as session:
        created = run_ablation_holdout_generation(
            session,
            target_count=target_count,
            dataset_id=args.dataset_id,
            seed=args.seed,
        )

    print(
        f"[ablation-holdout] created {len(created)} new ablation-only question(s) "
        f"(target {target_count})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
