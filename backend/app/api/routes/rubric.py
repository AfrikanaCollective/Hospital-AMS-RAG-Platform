"""Rubric (rank mode) endpoints (PRD-040, PRD-041, PRD-042, PRD-044; ARCH §14).

GET  /rubric/domains                          -> the 11 domains + Likert anchors (reference)
POST /rubric/results/{result_id}/ratings      (reviewer) -> one rating_round: 11 domain scores
      (1-5) + optional comment + a required accept-axis decision, together in
      one call (ARCH §13.2 "Both axes together"; DEVIATIONS.md #100 — a
      ranker's task is exactly these two things, no reason code collected,
      no escalation ever created). Enforces distinct-rater
      (rating_round UNIQUE (result_id, rater_id)).
GET  /rubric/results/{result_id}              -> rating history + IRR (once archived)
IRR per domain computed at >= IRR_MIN_RATERS distinct raters, then archived.

The queue LISTING itself is `GET /review-queue`
(`app.api.routes.review_queue`), not here — it predates this module's own
`/rubric/results/{id}` detail/rating endpoints and the frontend already
calls it (`ReviewPage.tsx`); DEVIATIONS.md #113 extended it with the
priority rule rather than adding a second, competing queue endpoint.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import Principal, get_db, principal_uuid, require_role
from app.config import get_settings
from app.db.models.eval import IRRScore, RatingRound, Result, RubricRating
from app.rubric.assignment import (
    PreviewAccountError,
    QueueRuleError,
    QueueStore,
    SqlQueueStore,
    check_may_rate,
    clear_draft_after_submit,
)
from app.rubric.domains import RUBRIC_DOMAINS
from app.rubric.workflow import (
    DuplicateRaterError,
    ResultArchivedError,
    ResultNotFoundError,
    TriageAgreementError,
    submit_rating,
)
from app.schemas.rubric import RatingRoundRequest

router = APIRouter()


def get_queue_store(session: Session = Depends(get_db)) -> QueueStore:
    """Overridden in tests with an in-memory store."""
    return SqlQueueStore(session)


@router.get("/domains")
async def list_domains() -> list[dict[str, object]]:
    return [d.as_reference() for d in RUBRIC_DOMAINS]


@router.post("/results/{result_id}/ratings")
async def submit_rating_route(
    result_id: str,
    body: RatingRoundRequest,
    session: Session = Depends(get_db),
    principal: Principal = Depends(require_role("reviewer")),
    store: QueueStore = Depends(get_queue_store),
) -> dict:
    rater_id = principal_uuid(principal)
    rid = uuid.UUID(result_id)
    # Only an item the reviewer's queue offers now; with a draft, only the
    # draft's item (DEVIATIONS.md #294).
    try:
        check_may_rate(store, rater_id, rid, batch_size=get_settings().review_batch_size)
    except PreviewAccountError as exc:
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(exc)) from exc
    except QueueRuleError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    try:
        round_row = submit_rating(
            session,
            result_id=rid,
            rater_id=rater_id,
            scores=body.scores,
            accept_action=body.accept_action,
            comment=body.comment,
            accept_edited_answer=body.accept_edited_answer,
            accept_accepted_context_ids=body.accept_accepted_context_ids,
            triage_agreement=body.triage_agreement,
        )
    except ResultNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except (DuplicateRaterError, ResultArchivedError) as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    except TriageAgreementError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    clear_draft_after_submit(store, rater_id, rid)
    return {
        "rating_round_id": str(round_row.id),
        "accept_action_id": str(round_row.accept_action_id),
    }


@router.get("/results/{result_id}")
async def get_result_ratings(
    result_id: str,
    session: Session = Depends(get_db),
    _principal: Principal = Depends(require_role("reviewer", "admin")),
) -> dict:
    rid = uuid.UUID(result_id)
    result = session.get(Result, rid)
    if result is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no result {result_id}")

    rounds_stmt = select(RatingRound).where(RatingRound.result_id == rid)
    rounds = list(session.execute(rounds_stmt).scalars().all())
    irr_stmt = select(IRRScore).where(IRRScore.result_id == rid)
    irr_rows = list(session.execute(irr_stmt).scalars().all())
    ratings_stmt = select(RubricRating).where(RubricRating.result_id == rid)
    ratings = list(session.execute(ratings_stmt).scalars().all())

    return {
        "result_id": result_id,
        "provenance": result.provenance,
        "expected_outcome": result.expected_outcome,
        "distinct_rater_count": len({r.rater_id for r in rounds}),
        "archived": result.queue_state == "archived",
        "rounds": [
            {
                "rater_id": str(r.rater_id),
                "submitted_at": r.submitted_at.isoformat(),
                "scores": {
                    rating.domain_code: rating.score
                    for rating in ratings
                    if rating.rater_id == r.rater_id
                },
            }
            for r in rounds
        ],
        "irr": [
            {
                "domain_code": row.domain_code,
                "metric": row.metric,
                "value": row.value,
                "n_raters": row.n_raters,
                "n_items": row.n_items,
            }
            for row in irr_rows
        ],
    }
