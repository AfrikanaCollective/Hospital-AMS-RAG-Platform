"""Reviewer queue: started-by-others first, then batches of fresh items, and
one half-finished rating (DEVIATIONS.md #294; PRD-042, ARCH §14.2)."""

from __future__ import annotations

import uuid

import pytest

from app.rubric.assignment import (
    QueueRuleError,
    build_queue_view,
    check_may_rate,
    clear_draft_after_submit,
    load_draft,
    plan_queue,
    save_draft,
)
from app.rubric.workflow import QueueCandidate
from tests.review_queue_fakes import T0, FakeStore

ME = uuid.uuid4()
OTHERS = [uuid.uuid4() for _ in range(3)]


def view(store: FakeStore, rater: uuid.UUID = ME, size: int = 12):  # noqa: ANN201
    return build_queue_view(store, rater, batch_size=size)


# ── rule 1: started by others first ──


def test_items_started_by_others_come_first_oldest_first() -> None:
    s = FakeStore(20)
    s.rate(OTHERS[0], s.results[5])
    s.rate(OTHERS[0], s.results[2])
    s.rate(OTHERS[1], s.results[2])
    v = view(s)
    assert v.mode == "in_progress"
    assert v.result_ids == [s.results[2], s.results[5]]
    assert s.items == []  # no batch handed out while started items remain


def test_items_with_three_ratings_or_rated_by_me_are_not_offered() -> None:
    s = FakeStore(3)
    for o in OTHERS:
        s.rate(o, s.results[0])  # archived
    s.rate(OTHERS[0], s.results[1])
    s.rate(ME, s.results[1])  # I already rated it
    v = view(s)
    assert v.mode == "batch" and v.result_ids == [s.results[2]]


# ── rule 2: batches ──


def test_a_batch_is_twelve_fresh_items_kept_until_finished() -> None:
    s = FakeStore(30)
    v = view(s)
    assert v.mode == "batch" and len(v.result_ids) == 12
    assert (v.batch_done, v.batch_total) == (0, 12)
    first = list(v.result_ids)
    s.rate(ME, first[0])
    s.rate(ME, first[1])
    v = view(s)
    assert v.result_ids == first[2:]  # same batch, not refilled
    assert (v.batch_done, v.batch_total) == (2, 12)
    for r in first[2:]:
        s.rate(ME, r)
    v = view(s)
    assert len(v.result_ids) == 12 and not set(v.result_ids) & set(first)  # a new batch
    assert {i.batch_no for i in s.items} == {1, 2}


def test_started_items_appear_mid_batch_and_hide_it_until_rated() -> None:
    s = FakeStore(30)
    batch = view(s).result_ids
    outside = next(r for r in s.results if r not in batch)
    s.rate(OTHERS[0], outside)  # someone else starts a fresh item
    v = view(s)
    assert v.mode == "in_progress" and v.result_ids == [outside]
    s.rate(ME, outside)
    v = view(s)
    assert v.mode == "batch" and v.result_ids == batch  # back to the same batch


def test_batches_may_overlap_but_only_with_fresh_items() -> None:
    s = FakeStore(15)
    mine = view(s, ME).result_ids
    s.rate(ME, mine[0])  # mine[0] is no longer fresh
    other = view(s, OTHERS[0])
    # OTHERS[0] sees mine[0] as started-by-others first (rule 1) ...
    assert other.mode == "in_progress" and other.result_ids == [mine[0]]
    s.rate(OTHERS[0], mine[0])
    other = view(s, OTHERS[0])
    # ... then a batch of fresh items: the 3 nobody holds first, then overlap.
    assert other.mode == "batch" and mine[0] not in other.result_ids
    unassigned = {r for r in s.results if r not in mine}
    assert unassigned <= set(other.result_ids)  # the 3 nobody holds are picked first
    assert len(set(other.result_ids) & set(mine)) == 9
    assert other.result_ids == sorted(other.result_ids, key=s.created.__getitem__)


def test_smaller_batch_when_fewer_fresh_items_then_nothing() -> None:
    s = FakeStore(5)
    v = view(s)
    assert v.mode == "batch" and v.batch_total == 5
    for r in v.result_ids:
        s.rate(ME, r)
    v = view(s)
    assert v.mode == "empty" and v.result_ids == []


# ── rule 3: archived before rated -> dropped and replaced ──


def test_archived_batch_item_is_dropped_and_replaced() -> None:
    s = FakeStore(14)
    batch = view(s).result_ids
    gone = batch[3]
    for o in OTHERS:
        s.rate(o, gone)  # archived by others
    v = view(s)
    assert gone not in v.result_ids and len(v.result_ids) == 12
    replacement = v.result_ids[-1]
    assert replacement not in batch and replacement in s.results
    assert v.batch_total == 12
    dropped = [i for i in s.items if i.result_id == gone]
    assert dropped[0].dropped_at is not None


def test_archived_batch_item_without_replacement_shrinks_the_batch() -> None:
    s = FakeStore(12)
    batch = view(s).result_ids
    for o in OTHERS:
        s.rate(o, batch[0])
    v = view(s)
    assert v.mode == "batch" and v.batch_total == 11 and batch[0] not in v.result_ids


def test_plan_is_pure_and_offers_nothing_new_while_started_items_exist() -> None:
    a, b = uuid.uuid4(), uuid.uuid4()
    plan = plan_queue(
        candidates=[
            QueueCandidate(result_id=a, created_at=T0, distinct_rater_count=1),
            QueueCandidate(result_id=b, created_at=T0, distinct_rater_count=0),
        ],
        batch=[],
        rated_by_me=set(),
        open_ids=set(),
        ever_assigned=set(),
        assignment_counts={},
        batch_size=12,
    )
    assert plan.mode == "in_progress" and plan.offer == [a] and plan.add == []


# ── rule 4: one draft ──


def test_a_draft_locks_the_queue_to_its_item() -> None:
    s = FakeStore(20)
    batch = view(s).result_ids
    save_draft(s, ME, batch[4], {"scores": {"D1": 3}}, batch_size=12)
    v = view(s)
    assert v.mode == "draft" and v.result_ids == [batch[4]] and v.draft_result_id == batch[4]
    assert load_draft(s, ME) == (batch[4], {"scores": {"D1": 3}})
    # no second draft, no rating another item
    with pytest.raises(QueueRuleError, match="saved rating first"):
        save_draft(s, ME, batch[0], {}, batch_size=12)
    with pytest.raises(QueueRuleError, match="saved rating first"):
        check_may_rate(s, ME, batch[0], batch_size=12)
    check_may_rate(s, ME, batch[4], batch_size=12)  # the draft's own item is fine
    # submitting clears it and the batch continues
    s.rate(ME, batch[4])
    clear_draft_after_submit(s, ME, batch[4])
    v = view(s)
    assert v.mode == "batch" and batch[4] not in v.result_ids and v.batch_done == 1


def test_draft_is_encrypted_and_only_for_offered_items() -> None:
    s = FakeStore(20)
    batch = view(s).result_ids
    outside = next(r for r in s.results if r not in batch)
    with pytest.raises(QueueRuleError, match="not in your review queue"):
        save_draft(s, ME, outside, {}, batch_size=12)
    save_draft(s, ME, batch[0], {"comment": "visible only to me"}, batch_size=12)
    assert b"visible only to me" not in s.drafts[ME].payload_enc
    with pytest.raises(QueueRuleError, match="too large"):
        save_draft(s, ME, batch[0], {"comment": "x" * 30_000}, batch_size=12)


def test_draft_whose_item_is_archived_by_others_is_removed_with_a_notice() -> None:
    s = FakeStore(20)
    batch = view(s).result_ids
    save_draft(s, ME, batch[0], {"scores": {}}, batch_size=12)
    for o in OTHERS:
        s.rate(o, batch[0])
    v = view(s)
    assert v.draft_dropped and v.mode == "batch" and batch[0] not in v.result_ids
    assert ME not in s.drafts and load_draft(s, ME) is None
