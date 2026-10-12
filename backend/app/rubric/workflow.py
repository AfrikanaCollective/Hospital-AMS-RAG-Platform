"""Multi-rater workflow state machine (ARCH §14.2, §14.3; PRD-042, PRD-043, PRD-047).

UNRATED -> OPEN_QUEUE -> (>= IRR_MIN_RATERS distinct raters) -> compute IRR -> ARCHIVED.
Until the minimum is met the result stays visible in the open queue to any
clinician (except its producer / raters who already rated it).

Distinctness is enforced twice: `is_eligible_rater` here (checked before
insert, so a duplicate submission gets a clear application error) AND the DB's
own `UNIQUE (result_id, rater_id)` constraint on `rating_round` (ARCH §14.2,
the actual backstop). "All domains scored" is enforced upstream by
`app.schemas.rubric.RatingRoundRequest` (exactly 11 scores, one per domain,
required on every submission) — so every successfully-inserted round already
satisfies it by construction; `can_archive`'s `all_domains_scored_by_each` is
always `True` here.

`result_producer_id` (the "did not produce the result" half of
`is_eligible_rater`) is always `None` in this pipeline: every rated result is
system-produced (an agent-generated answer or a question-generator output),
never authored by a clinician, so there is no producer to exclude.

**Queue visibility + priority (`select_queue_items`, DEVIATIONS.md #113):**
ARCH §14.2 documents only the `< 3` visibility cutoff, not an ordering
between an unrated result and a partially-rated one. Reviewer-stated rule:
the queue only ever shows results with 0-2 distinct rating rounds from other
clinicians; if ANY of those has 1-2 rounds already, ONLY those are offered
(finishing an in-progress result toward the 3-rater minimum takes priority
over starting a fresh one) — once none remain, the 0-round results become
selectable. `list_queue_candidates` does the DB read (open, excluding
whatever the caller has already rated); `select_queue_items` is the pure
priority rule over that list, unit-tested without a database, same style as
`is_eligible_rater`/`can_archive` above.
"""

from __future__ import annotations

import uuid
from collections.abc import Collection
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING

from sqlalchemy import func, select

from app.config import get_settings
from app.db.models.eval import IRRScore, RatingRound, Result, ResultArchive, RubricRating
from app.hitl.decisions import apply_rating_accept_action
from app.rubric.irr import per_domain_irr

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from app.schemas.enums import HitlAcceptAction
    from app.schemas.rubric import DomainScore


class QueueState(StrEnum):
    """Matches `eval.result.queue_state`'s persisted vocabulary exactly
    (`app/db/models/eval.py`'s `Result.queue_state` comment: not_queued | open
    | archived) — this enum is not an independent vocabulary."""

    NOT_QUEUED = "not_queued"
    OPEN_QUEUE = "open"
    ARCHIVED = "archived"


# A result archives once it reaches get_settings().irr_min_raters (3)
# distinct rounds, so "in progress" (visible, but not fresh) tops out at 2 —
# a literal, not the configurable min-raters value itself (DEVIATIONS.md #113).
_MAX_IN_PROGRESS_ROUNDS = 2


class ResultNotFoundError(LookupError):
    pass


class DuplicateRaterError(ValueError):
    pass


class ResultArchivedError(ValueError):
    """The result already has its ratings and was archived; a late
    submission (e.g. a draft finished after others) must not add a 4th
    (DEVIATIONS.md #294)."""


def is_eligible_rater(
    *, rater_id: object, result_producer_id: object, already_rated_by: Collection[object]
) -> bool:
    """A distinct clinician who did not produce the result and has not rated
    it. `rater_id`/`result_producer_id` are opaque identifiers to this check
    (a plain `str` in the gating test fixtures, a `uuid.UUID` for real
    raters) — only equality/membership matters, never their type."""
    return rater_id != result_producer_id and rater_id not in already_rated_by


def can_archive(
    *, distinct_rater_count: int, min_raters: int, all_domains_scored_by_each: bool
) -> bool:
    return distinct_rater_count >= min_raters and all_domains_scored_by_each


@dataclass(frozen=True)
class QueueCandidate:
    """One open result eligible for a given rater (already excludes results
    that rater has rated and anything not `queue_state == 'open'`) —
    `list_queue_candidates`'s output, `select_queue_items`'s input."""

    result_id: uuid.UUID
    distinct_rater_count: int
    created_at: datetime


def select_queue_items(candidates: list[QueueCandidate]) -> list[QueueCandidate]:
    """The queue visibility + priority rule (see module docstring,
    DEVIATIONS.md #113). `candidates` must already be limited to `open`
    results with 0-2 distinct rounds, excluding the calling rater's own —
    this function only decides which of *those* to actually offer:

    - if any candidate has 1 or 2 rounds ("in progress"), return ONLY those
      (finishing an in-progress result takes priority over a fresh one);
    - otherwise return every 0-round candidate.

    Oldest-first within the returned bucket — a stable, deterministic
    tie-break, not a second priority tier."""
    in_progress = [c for c in candidates if 1 <= c.distinct_rater_count <= _MAX_IN_PROGRESS_ROUNDS]
    pool = in_progress if in_progress else [c for c in candidates if c.distinct_rater_count == 0]
    return sorted(pool, key=lambda c: c.created_at)


def list_queue_candidates(session: Session, rater_id: uuid.UUID) -> list[QueueCandidate]:
    """`open` results with 0-2 distinct rating rounds, excluding any result
    `rater_id` has already rated (they couldn't rate it again anyway —
    `rating_round`'s own `UNIQUE (result_id, rater_id)` — so it's excluded
    from their queue view entirely, not just left unselectable)."""
    already_rated = select(RatingRound.result_id).where(RatingRound.rater_id == rater_id)
    rater_counts = (
        select(
            RatingRound.result_id.label("result_id"),
            func.count(func.distinct(RatingRound.rater_id)).label("n"),
        )
        .group_by(RatingRound.result_id)
        .subquery()
    )
    stmt = (
        select(Result.id, Result.created_at, func.coalesce(rater_counts.c.n, 0))
        .outerjoin(rater_counts, rater_counts.c.result_id == Result.id)
        .where(Result.queue_state == QueueState.OPEN_QUEUE)
        .where(Result.id.not_in(already_rated))
    )
    return [
        QueueCandidate(result_id=result_id, created_at=created_at, distinct_rater_count=count)
        for result_id, created_at, count in session.execute(stmt).all()
    ]


def _fetch_result(session: Session, result_id: uuid.UUID) -> Result | None:
    return session.get(Result, result_id)


def _fetch_rating_rounds(session: Session, result_id: uuid.UUID) -> list[RatingRound]:
    stmt = select(RatingRound).where(RatingRound.result_id == result_id)
    return list(session.execute(stmt).scalars().all())


def _fetch_ratings(session: Session, result_id: uuid.UUID) -> list[RubricRating]:
    stmt = select(RubricRating).where(RubricRating.result_id == result_id)
    return list(session.execute(stmt).scalars().all())


def submit_rating(
    session: Session,
    *,
    result_id: uuid.UUID,
    rater_id: uuid.UUID,
    scores: list[DomainScore],
    accept_action: HitlAcceptAction,
    comment: str | None = None,
    accept_edited_answer: str | None = None,
    accept_accepted_context_ids: list[str] | None = None,
    is_original_rater: bool = False,
) -> RatingRound:
    """Both HITL axes, one sitting (ARCH §13.2 "Both axes together"):
    `accept_action` is required on every call — a rater completing the
    rubric for a case always also records an accept-axis decision for it.
    That's the whole task (DEVIATIONS.md #100) — no reason code is collected
    or passed to `apply_rating_accept_action` here, and no escalation is
    ever created by this path. `RatingRound.accept_action_id` links the
    resulting `HitlDecision` back to this round."""
    result = _fetch_result(session, result_id)
    if result is None:
        raise ResultNotFoundError(f"no result {result_id}")
    if result.queue_state == QueueState.ARCHIVED:
        raise ResultArchivedError(f"result {result_id} is already archived")

    already_rated_by = {r.rater_id for r in _fetch_rating_rounds(session, result_id)}
    eligible = is_eligible_rater(
        rater_id=rater_id, result_producer_id=None, already_rated_by=already_rated_by
    )
    if not eligible:
        raise DuplicateRaterError(f"rater {rater_id} has already rated result {result_id}")

    now = datetime.now(UTC)
    round_row = RatingRound(
        result_id=result_id,
        rater_id=rater_id,
        is_original_rater=is_original_rater,
        submitted_at=now,
    )
    session.add(round_row)
    session.flush()

    for s in scores:
        session.add(
            RubricRating(
                result_id=result_id,
                rater_id=rater_id,
                domain_code=s.domain_code,
                score=s.score,
                rated_at=now,
                rating_round_id=round_row.id,
                comment=comment,
            )
        )

    decision = apply_rating_accept_action(
        session,
        result_id=result_id,
        message_id=result.message_id,
        reviewer_id=rater_id,
        action=accept_action,
        edited_answer=accept_edited_answer,
        accepted_context_ids=accept_accepted_context_ids,
    )
    round_row.accept_action_id = decision.id

    if result.queue_state == QueueState.NOT_QUEUED:
        result.queue_state = QueueState.OPEN_QUEUE

    distinct_count = len(already_rated_by) + 1
    if can_archive(
        distinct_rater_count=distinct_count,
        min_raters=get_settings().irr_min_raters,
        all_domains_scored_by_each=True,
    ):
        compute_result_irr_and_archive(session, result_id)

    return round_row


def compute_result_irr_and_archive(session: Session, result_id: uuid.UUID) -> None:
    """Compute per-domain IRR at the current rater count and archive
    (ARCH §14.4, §14.5). Called once the minimum rater count is reached —
    either inline from `submit_rating` or via the Celery task of the same
    name (`app.rubric.tasks.compute_result_irr_and_archive`)."""
    ratings = _fetch_ratings(session, result_id)
    by_domain: dict[str, dict[uuid.UUID, int]] = {}
    for r in ratings:
        by_domain.setdefault(r.domain_code, {})[r.rater_id] = r.score
    rater_ids = sorted({r.rater_id for r in ratings}, key=str)
    domain_ratings = {
        domain: [[scores.get(rid) for rid in rater_ids]] for domain, scores in by_domain.items()
    }
    irr_results = per_domain_irr(domain_ratings)

    now = datetime.now(UTC)
    irr_rows = []
    for dr in irr_results:
        row = IRRScore(
            result_id=result_id,
            domain_code=dr.domain_code,
            metric=dr.metric,
            value=dr.value,
            n_raters=dr.n_raters,
            n_items=dr.n_items,
            computed_at=now,
        )
        session.add(row)
        irr_rows.append(row)

    result = _fetch_result(session, result_id)
    if result is None:
        raise ResultNotFoundError(f"no result {result_id}")
    result.queue_state = QueueState.ARCHIVED

    rounds = _fetch_rating_rounds(session, result_id)
    rounds_history = [
        {"rater_id": str(r.rater_id), "submitted_at": r.submitted_at.isoformat()} for r in rounds
    ]
    ratings_history = [
        {"rater_id": str(r.rater_id), "domain_code": r.domain_code, "score": r.score}
        for r in ratings
    ]
    archive = ResultArchive(
        result_id=result_id,
        archived_at=now,
        rating_history={"rounds": rounds_history, "ratings": ratings_history},
        irr_snapshot={row.domain_code: row.value for row in irr_rows},
        provenance=result.provenance,
    )
    session.add(archive)
