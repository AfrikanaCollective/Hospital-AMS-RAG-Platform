"""What a reviewer may rate next: items others started, then their own batch,
plus one half-finished rating (DEVIATIONS.md #294; PRD-042, ARCH §14.2).

Operator rules, 2026-10-12, replacing the plain #113 queue for reviewers:

1. **Started by others first, always.** Open results with 1-2 ratings from
   other reviewers and none from this one are offered first. The list is
   live: new ones appear even while the reviewer is part-way through a
   batch, and the batch is hidden until they are rated.
2. **Then a batch.** Only when (1) is empty is the reviewer given up to
   `REVIEW_BATCH_SIZE` (12) *fresh* results: open, rated by nobody, never
   assigned to this reviewer before. The batch is kept until the reviewer
   has rated every item in it. Batches of different reviewers may overlap,
   as long as each item was fresh when it was assigned; items in the
   fewest other reviewers' batches are picked first, then oldest.
3. **Archived before rated.** A batch item that others take to 3 ratings
   (archived) before this reviewer rates it is dropped and replaced by
   another fresh result, if one exists. With fewer fresh results left the
   batch is smaller; with none, "Nothing to review right now".
4. **One draft.** A reviewer has at most one half-finished rating. While it
   exists only that item is offered, and it cannot be discarded: it is
   removed only by submitting it, or (forced, with a notice) if its result
   is archived by others first.

`plan_queue` is the pure rule (unit-tested without a database);
`build_queue_view` applies it through a `QueueStore`.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal, Protocol

from sqlalchemy import func, select, text

from app.crypto.provider import get_crypto
from app.db.models.eval import (
    RatingDraft,
    RatingRound,
    Result,
    ReviewBatchItem,
    rating_draft_aad,
)
from app.rubric.workflow import (
    _MAX_IN_PROGRESS_ROUNDS,
    QueueCandidate,
    QueueState,
    list_queue_candidates,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

Mode = Literal["draft", "in_progress", "batch", "empty"]

# The draft payload is the rubric form's own state; capped so a client
# cannot park arbitrary data in it.
MAX_DRAFT_BYTES = 20_000


class QueueRuleError(ValueError):
    """The reviewer may not rate or draft this result now."""


@dataclass(frozen=True)
class QueuePlan:
    mode: Mode
    offer: list[uuid.UUID]
    drop: list[uuid.UUID] = field(default_factory=list)
    add: list[uuid.UUID] = field(default_factory=list)
    new_batch: bool = False
    batch_done: int = 0
    batch_total: int = 0


def plan_queue(
    *,
    candidates: Sequence[QueueCandidate],
    batch: Sequence[uuid.UUID],
    rated_by_me: Collection[uuid.UUID],
    open_ids: Collection[uuid.UUID],
    ever_assigned: Collection[uuid.UUID],
    assignment_counts: dict[uuid.UUID, int],
    batch_size: int,
) -> QueuePlan:
    """The rule, rules 1-3 above (the draft, rule 4, is checked before).

    `candidates`: open results not rated by this reviewer, with their rater
    counts (`list_queue_candidates`). `batch`: the current batch's
    not-dropped items, in assignment order. `rated_by_me` / `open_ids`: of
    those batch items, which this reviewer rated / which are still open.
    `ever_assigned`: every result ever in this reviewer's batches.
    `assignment_counts`: how many reviewers' batches hold each result."""
    in_progress = sorted(
        (c for c in candidates if 1 <= c.distinct_rater_count <= _MAX_IN_PROGRESS_ROUNDS),
        key=lambda c: c.created_at,
    )
    if in_progress:
        return QueuePlan(mode="in_progress", offer=[c.result_id for c in in_progress])

    fresh = sorted(
        (c for c in candidates if c.distinct_rater_count == 0 and c.result_id not in ever_assigned),
        key=lambda c: (assignment_counts.get(c.result_id, 0), c.created_at),
    )
    drop = [r for r in batch if r not in rated_by_me and r not in open_ids]
    kept = [r for r in batch if r not in drop]
    remaining = [r for r in kept if r not in rated_by_me]
    if remaining or drop:
        add = [c.result_id for c in fresh[: len(drop)]]
        offer = remaining + add
        if offer:
            return QueuePlan(
                mode="batch",
                offer=offer,
                drop=drop,
                add=add,
                batch_done=len(kept) - len(remaining),
                batch_total=len(kept) + len(add),
            )
        # Every unrated item was archived and nothing fresh replaces them:
        # the batch is finished; fall through to a new one (also empty).
    # Shown oldest first, the order `current_batch` reads them back in.
    picks = [c.result_id for c in sorted(fresh[:batch_size], key=lambda c: c.created_at)]
    if not picks:
        return QueuePlan(mode="empty", offer=[], drop=drop)
    return QueuePlan(
        mode="batch", offer=picks, drop=drop, add=picks, new_batch=True, batch_total=len(picks)
    )


# ── storage ──


class QueueStore(Protocol):
    def lock(self, rater_id: uuid.UUID) -> None: ...
    def candidates(self, rater_id: uuid.UUID) -> list[QueueCandidate]: ...
    def current_batch(self, rater_id: uuid.UUID) -> tuple[int, list[uuid.UUID]]: ...
    def ever_assigned(self, rater_id: uuid.UUID) -> set[uuid.UUID]: ...
    def assignment_counts(self, result_ids: Collection[uuid.UUID]) -> dict[uuid.UUID, int]: ...
    def rated_by(
        self, rater_id: uuid.UUID, result_ids: Collection[uuid.UUID]
    ) -> set[uuid.UUID]: ...
    def open_ids(self, result_ids: Collection[uuid.UUID]) -> set[uuid.UUID]: ...
    def add_batch_items(
        self, rater_id: uuid.UUID, batch_no: int, result_ids: Sequence[uuid.UUID], now: datetime
    ) -> None: ...
    def drop_batch_items(
        self, rater_id: uuid.UUID, result_ids: Collection[uuid.UUID], now: datetime
    ) -> None: ...
    def get_draft(self, rater_id: uuid.UUID) -> RatingDraft | None: ...
    def put_draft(self, draft: RatingDraft) -> None: ...
    def delete_draft(self, rater_id: uuid.UUID) -> None: ...


class SqlQueueStore:
    def __init__(self, session: Session) -> None:
        self.session = session

    def lock(self, rater_id: uuid.UUID) -> None:
        # Two tabs refreshing at once must not both create a batch.
        self.session.execute(
            text("SELECT pg_advisory_xact_lock(hashtext(:k))"), {"k": f"review-queue:{rater_id}"}
        )

    def candidates(self, rater_id: uuid.UUID) -> list[QueueCandidate]:
        return list_queue_candidates(self.session, rater_id)

    def current_batch(self, rater_id: uuid.UUID) -> tuple[int, list[uuid.UUID]]:
        batch_no = self.session.execute(
            select(func.max(ReviewBatchItem.batch_no)).where(ReviewBatchItem.rater_id == rater_id)
        ).scalar()
        if batch_no is None:
            return 0, []
        # Stable order: assignment time (replacements last), then the
        # result's age, as `plan_queue` offered them.
        rows = self.session.execute(
            select(ReviewBatchItem.result_id)
            .join(Result, Result.id == ReviewBatchItem.result_id)
            .where(ReviewBatchItem.rater_id == rater_id)
            .where(ReviewBatchItem.batch_no == batch_no)
            .where(ReviewBatchItem.dropped_at.is_(None))
            .order_by(ReviewBatchItem.assigned_at, Result.created_at, Result.id)
        ).scalars()
        return batch_no, list(rows)

    def ever_assigned(self, rater_id: uuid.UUID) -> set[uuid.UUID]:
        return set(
            self.session.execute(
                select(ReviewBatchItem.result_id).where(ReviewBatchItem.rater_id == rater_id)
            ).scalars()
        )

    def assignment_counts(self, result_ids: Collection[uuid.UUID]) -> dict[uuid.UUID, int]:
        if not result_ids:
            return {}
        rows = self.session.execute(
            select(ReviewBatchItem.result_id, func.count())
            .where(ReviewBatchItem.result_id.in_(list(result_ids)))
            .where(ReviewBatchItem.dropped_at.is_(None))
            .group_by(ReviewBatchItem.result_id)
        ).all()
        return {r: int(n) for r, n in rows}

    def rated_by(self, rater_id: uuid.UUID, result_ids: Collection[uuid.UUID]) -> set[uuid.UUID]:
        if not result_ids:
            return set()
        return set(
            self.session.execute(
                select(RatingRound.result_id)
                .where(RatingRound.rater_id == rater_id)
                .where(RatingRound.result_id.in_(list(result_ids)))
            ).scalars()
        )

    def open_ids(self, result_ids: Collection[uuid.UUID]) -> set[uuid.UUID]:
        if not result_ids:
            return set()
        return set(
            self.session.execute(
                select(Result.id)
                .where(Result.id.in_(list(result_ids)))
                .where(Result.queue_state == QueueState.OPEN_QUEUE)
            ).scalars()
        )

    def add_batch_items(
        self, rater_id: uuid.UUID, batch_no: int, result_ids: Sequence[uuid.UUID], now: datetime
    ) -> None:
        for result_id in result_ids:
            self.session.add(
                ReviewBatchItem(
                    rater_id=rater_id, batch_no=batch_no, result_id=result_id, assigned_at=now
                )
            )
        self.session.flush()

    def drop_batch_items(
        self, rater_id: uuid.UUID, result_ids: Collection[uuid.UUID], now: datetime
    ) -> None:
        if not result_ids:
            return
        for item in self.session.execute(
            select(ReviewBatchItem)
            .where(ReviewBatchItem.rater_id == rater_id)
            .where(ReviewBatchItem.result_id.in_(list(result_ids)))
        ).scalars():
            item.dropped_at = now
        self.session.flush()

    def get_draft(self, rater_id: uuid.UUID) -> RatingDraft | None:
        return self.session.get(RatingDraft, rater_id)

    def put_draft(self, draft: RatingDraft) -> None:
        self.session.merge(draft)
        self.session.flush()

    def delete_draft(self, rater_id: uuid.UUID) -> None:
        draft = self.session.get(RatingDraft, rater_id)
        if draft is not None:
            self.session.delete(draft)
            self.session.flush()


# ── the queue a reviewer sees ──


@dataclass(frozen=True)
class QueueView:
    mode: Mode
    result_ids: list[uuid.UUID]
    draft_result_id: uuid.UUID | None = None
    # The draft's result was archived by others before it was submitted.
    draft_dropped: bool = False
    batch_done: int = 0
    batch_total: int = 0


def _now() -> datetime:
    return datetime.now(UTC)


def _check_draft(store: QueueStore, rater_id: uuid.UUID) -> tuple[RatingDraft | None, bool]:
    """The reviewer's draft if it can still be submitted; a draft whose
    result is no longer open (archived by others) is removed (rule 4)."""
    draft = store.get_draft(rater_id)
    if draft is None:
        return None, False
    if draft.result_id in store.open_ids([draft.result_id]) and not store.rated_by(
        rater_id, [draft.result_id]
    ):
        return draft, False
    store.delete_draft(rater_id)
    return None, True


def build_queue_view(store: QueueStore, rater_id: uuid.UUID, *, batch_size: int) -> QueueView:
    """Apply the rules, writing any batch changes they call for."""
    store.lock(rater_id)
    draft, dropped = _check_draft(store, rater_id)
    if draft is not None:
        return QueueView(
            mode="draft", result_ids=[draft.result_id], draft_result_id=draft.result_id
        )

    batch_no, batch = store.current_batch(rater_id)
    candidates = store.candidates(rater_id)
    plan = plan_queue(
        candidates=candidates,
        batch=batch,
        rated_by_me=store.rated_by(rater_id, batch),
        open_ids=store.open_ids(batch),
        ever_assigned=store.ever_assigned(rater_id),
        assignment_counts=store.assignment_counts(
            [c.result_id for c in candidates if c.distinct_rater_count == 0]
        ),
        batch_size=batch_size,
    )
    now = _now()
    if plan.mode != "in_progress":
        store.drop_batch_items(rater_id, plan.drop, now)
        if plan.add:
            store.add_batch_items(
                rater_id, batch_no + 1 if plan.new_batch else batch_no, plan.add, now
            )
    return QueueView(
        mode=plan.mode,
        result_ids=plan.offer,
        draft_dropped=dropped,
        batch_done=plan.batch_done,
        batch_total=plan.batch_total,
    )


def check_may_rate(
    store: QueueStore, rater_id: uuid.UUID, result_id: uuid.UUID, *, batch_size: int
) -> None:
    """With a draft, only the draft's item may be rated (rule 4). An open
    (queued) result must be one the reviewer's queue offers now. A result
    not yet queued keeps its old path (a first rating opens it, ARCH §14.2),
    and an archived one is refused by `submit_rating` itself."""
    view = build_queue_view(store, rater_id, batch_size=batch_size)
    if view.draft_result_id is not None and view.draft_result_id != result_id:
        raise QueueRuleError("finish and submit your saved rating first")
    if result_id in store.open_ids([result_id]) and result_id not in view.result_ids:
        raise QueueRuleError("this item is not in your review queue")


# ── drafts ──


def save_draft(
    store: QueueStore,
    rater_id: uuid.UUID,
    result_id: uuid.UUID,
    payload: dict[str, Any],
    *,
    batch_size: int,
) -> None:
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    if len(raw) > MAX_DRAFT_BYTES:
        raise QueueRuleError("draft too large")
    view = build_queue_view(store, rater_id, batch_size=batch_size)
    if view.draft_result_id is not None and view.draft_result_id != result_id:
        raise QueueRuleError("finish and submit your saved rating first")
    if result_id not in view.result_ids:  # drafts only for queue items
        raise QueueRuleError("this item is not in your review queue")
    store.put_draft(
        RatingDraft(
            rater_id=rater_id,
            result_id=result_id,
            payload_enc=get_crypto().encrypt(raw, aad=rating_draft_aad(rater_id, result_id)),
            updated_at=_now(),
        )
    )


def load_draft(store: QueueStore, rater_id: uuid.UUID) -> tuple[uuid.UUID, dict[str, Any]] | None:
    draft, _ = _check_draft(store, rater_id)
    if draft is None:
        return None
    raw = get_crypto().decrypt(draft.payload_enc, aad=rating_draft_aad(rater_id, draft.result_id))
    return draft.result_id, json.loads(raw)


def clear_draft_after_submit(store: QueueStore, rater_id: uuid.UUID, result_id: uuid.UUID) -> None:
    draft = store.get_draft(rater_id)
    if draft is not None and draft.result_id == result_id:
        store.delete_draft(rater_id)
