"""Open review queue (PRD-042, PRD-047; ARCH §14.2).

GET  /review-queue                 (reviewer) -> results this reviewer may currently pick to
                                                 rate. Visibility + priority rule
                                                 (DEVIATIONS.md #113): only 0-2-round `open`
                                                 results, excluding anything this reviewer
                                                 already rated; if ANY of those has 1-2 rounds
                                                 ("in progress"), ONLY those are returned —
                                                 finishing an in-progress result takes priority
                                                 over starting a fresh (0-round) one, which only
                                                 becomes selectable once no in-progress result
                                                 remains (`app.rubric.workflow.select_queue_items`,
                                                 the actual rule; `list_queue_candidates`, the
                                                 DB read).
GET  /review-queue/{result_id}     (reviewer) -> result + citations for rating (no other raters'
                                                 scores)
GET  /review-queue/draft           (reviewer) -> the reviewer's one half-finished rating, or 404
PUT  /review-queue/draft           (reviewer) -> save it (only for an item the queue offers)

**Since DEVIATIONS.md #294** the list is `app.rubric.assignment`'s: items
other reviewers started first, then the reviewer's own batch of fresh
items, and while a draft exists only the draft's item. The response is an
object (`mode`, `items`, `draft_result_id`, `draft_dropped`, `batch`).
Hard/adversarial cases are in the SAME queue (PRD-047).

**`eval.result` row creation** (DEVIATIONS.md #79, #113): this router only
reads existing `Result` rows. `app.eval.auto_seed` is the first writer
(auto-generated, de-identified-record-sourced hypotheticals, run through the
real query pipeline and queued `open` immediately); a live `/query` result
or a clinician-submitted one may write here too in the future, under the
same `Result.answer_enc` AAD contract (`app.db.models.eval.result_answer_aad`).
"""

from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import Principal, get_db, principal_uuid, require_role
from app.api.routes.rubric import get_queue_store
from app.config import get_settings
from app.crypto.provider import get_crypto
from app.db.models.eval import (
    EvalQuestion,
    Result,
    result_answer_aad,
    result_multi_stage_answer_aad,
    result_segments_aad,
)
from app.eval.question_gen.areas import reviewer_question_text
from app.rubric.assignment import (
    QueueRuleError,
    QueueStore,
    build_queue_view,
    load_draft,
    save_draft,
)

router = APIRouter()


def _query_area(session: Session, result: Result) -> str | None:
    """The four-area query area (`generator_meta.area`) of the question this
    result answers, shown as a queue column (DEVIATIONS.md #290); None when
    the result has no generated question or the question has no area."""
    if result.eval_question_id is None:
        return None
    question = session.get(EvalQuestion, result.eval_question_id)
    area = (question.generator_meta or {}).get("area") if question is not None else None
    return area if isinstance(area, str) else None


@router.get("")
async def list_queue(
    session: Session = Depends(get_db),
    principal: Principal = Depends(require_role("reviewer")),
    store: QueueStore = Depends(get_queue_store),
) -> dict:
    view = build_queue_view(
        store, principal_uuid(principal), batch_size=get_settings().review_batch_size
    )
    results = {
        r.id: r
        for r in session.execute(select(Result).where(Result.id.in_(view.result_ids)))
        .scalars()
        .all()
    }
    items = [
        {
            "result_id": str(rid),
            "provenance": results[rid].provenance,
            "expected_outcome": results[rid].expected_outcome,
            "observed_outcome": results[rid].observed_outcome,
            "query_area": _query_area(session, results[rid]),
        }
        for rid in view.result_ids
        if rid in results
    ]
    return {
        "mode": view.mode,
        "items": items,
        "draft_result_id": str(view.draft_result_id) if view.draft_result_id else None,
        "draft_dropped": view.draft_dropped,
        "batch": (
            {"done": view.batch_done, "total": view.batch_total} if view.mode == "batch" else None
        ),
    }


class DraftIn(BaseModel):
    result_id: uuid.UUID
    # The rubric form's own state (scores, comment, accept, step).
    payload: dict = Field(default_factory=dict)


@router.get("/draft")
async def get_draft(
    principal: Principal = Depends(require_role("reviewer")),
    store: QueueStore = Depends(get_queue_store),
) -> dict:
    found = load_draft(store, principal_uuid(principal))
    if found is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no saved rating")
    result_id, payload = found
    return {"result_id": str(result_id), "payload": payload}


@router.put("/draft")
async def put_draft(
    body: DraftIn,
    principal: Principal = Depends(require_role("reviewer")),
    store: QueueStore = Depends(get_queue_store),
) -> dict:
    try:
        save_draft(
            store,
            principal_uuid(principal),
            body.result_id,
            body.payload,
            batch_size=get_settings().review_batch_size,
        )
    except QueueRuleError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    return {"result_id": str(body.result_id)}


@router.get("/{result_id}")
async def get_queue_item(
    result_id: str,
    session: Session = Depends(get_db),
    _principal: Principal = Depends(require_role("reviewer")),
) -> dict:
    result = session.get(Result, uuid.UUID(result_id))
    if result is None or result.queue_state == "not_queued":
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no queued result {result_id}")

    crypto = get_crypto()
    answer = None
    if result.answer_enc is not None:
        answer = crypto.decrypt(result.answer_enc, aad=result_answer_aad(result.id)).decode("utf-8")

    # Per-segment citation_ids, so the UI can link a citation to the specific
    # claim it supports instead of only listing citations below a flat answer
    # paragraph (DEVIATIONS.md #120). Absent for a result written before this
    # column existed, or for an escalated result with no released answer —
    # the caller falls back to the flat `answer` text in that case.
    segments = None
    if result.answer_segments_enc is not None:
        segments = json.loads(
            crypto.decrypt(result.answer_segments_enc, aad=result_segments_aad(result.id)).decode(
                "utf-8"
            )
        )

    # The generated question this result answers — an auto-generated
    # scenario had no way to show a reviewer what was actually asked
    # (DEVIATIONS.md #120); `None` for a `clinician_submitted` result with no
    # linked `eval_question` row.
    question = None
    if result.eval_question_id is not None:
        eval_question = session.get(EvalQuestion, result.eval_question_id)
        # Without the antibiotic-course retrieval expansions (#291).
        question = reviewer_question_text(eval_question.text) if eval_question is not None else None

    # Multi-stage (Phase 7 vocabulary-augmented, PRD-111 Arm C) audit-only
    # counterpart (DEVIATIONS.md #186) — read-only, for traceability/
    # comparison. NEVER fed into rating: `app.rubric.workflow` only ever
    # reads/writes `rating_round`/`rubric_rating` rows keyed by `result_id`,
    # never touches these fields, so exposing them here cannot influence
    # what a reviewer is scored on. `None` when this run had no attested
    # vocabulary; identical to the single-stage query (with no answer) when
    # the augmentation didn't fire for this record.
    multi_stage_answer = None
    if result.multi_stage_answer_enc is not None:
        multi_stage_answer = crypto.decrypt(
            result.multi_stage_answer_enc, aad=result_multi_stage_answer_aad(result.id)
        ).decode("utf-8")

    # No other raters' scores exposed here — independence of the rank-mode
    # rating (ARCH §14.2): a reviewer's own score must not be anchored on a
    # prior rater's judgement.
    return {
        "result_id": result_id,
        "provenance": result.provenance,
        "expected_outcome": result.expected_outcome,
        "query_area": _query_area(session, result),
        "question": question,
        "answer": answer,
        "segments": segments,
        "citations": result.citations,
        "grounding_report": result.grounding_report,
        "multi_stage": {
            "query": result.multi_stage_query,
            "fired": result.multi_stage_fired,
            "answer": multi_stage_answer,
            "citations": result.multi_stage_citations,
            "grounding_report": result.multi_stage_grounding_report,
        },
    }
