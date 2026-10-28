"""In-memory `QueueStore` for reviewer-queue tests (DEVIATIONS.md #294)."""

from __future__ import annotations

import uuid
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from app.db.models.eval import RatingDraft
from app.rubric.workflow import QueueCandidate

T0 = datetime(2026, 10, 1, tzinfo=UTC)


@dataclass
class _Item:
    rater_id: uuid.UUID
    batch_no: int
    result_id: uuid.UUID
    assigned_at: datetime
    dropped_at: datetime | None = None


class FakeStore:
    def __init__(self, n_results: int) -> None:
        self.results = [uuid.uuid4() for _ in range(n_results)]
        self.created = {r: T0 + timedelta(minutes=i) for i, r in enumerate(self.results)}
        self.archived: set[uuid.UUID] = set()
        self.ratings: dict[uuid.UUID, set[uuid.UUID]] = {r: set() for r in self.results}
        self.items: list[_Item] = []
        self.drafts: dict[uuid.UUID, RatingDraft] = {}
        self.preview: set[uuid.UUID] = set()  # preview accounts (DEVIATIONS.md #323)

    # helpers for tests
    def rate(self, rater: uuid.UUID, result: uuid.UUID) -> None:
        self.ratings[result].add(rater)
        if len(self.ratings[result]) >= 3:
            self.archived.add(result)

    # QueueStore
    def is_preview(self, rater_id: uuid.UUID) -> bool:
        return rater_id in self.preview

    def lock(self, rater_id: uuid.UUID) -> None:
        pass

    def candidates(self, rater_id: uuid.UUID) -> list[QueueCandidate]:
        return [
            QueueCandidate(
                result_id=r, created_at=self.created[r], distinct_rater_count=len(self.ratings[r])
            )
            for r in self.results
            if r not in self.archived and rater_id not in self.ratings[r]
        ]

    def current_batch(self, rater_id: uuid.UUID) -> tuple[int, list[uuid.UUID]]:
        mine = [i for i in self.items if i.rater_id == rater_id]
        if not mine:
            return 0, []
        n = max(i.batch_no for i in mine)
        return n, [i.result_id for i in mine if i.batch_no == n and i.dropped_at is None]

    def ever_assigned(self, rater_id: uuid.UUID) -> set[uuid.UUID]:
        return {i.result_id for i in self.items if i.rater_id == rater_id}

    def assignment_counts(self, result_ids: Collection[uuid.UUID]) -> dict[uuid.UUID, int]:
        out: dict[uuid.UUID, int] = {}
        for i in self.items:
            if i.result_id in result_ids and i.dropped_at is None:
                out[i.result_id] = out.get(i.result_id, 0) + 1
        return out

    def rated_by(self, rater_id: uuid.UUID, result_ids: Collection[uuid.UUID]) -> set[uuid.UUID]:
        return {r for r in result_ids if rater_id in self.ratings[r]}

    def open_ids(self, result_ids: Collection[uuid.UUID]) -> set[uuid.UUID]:
        return {r for r in result_ids if r in self.ratings and r not in self.archived}

    def add_batch_items(
        self, rater_id: uuid.UUID, batch_no: int, result_ids: Sequence[uuid.UUID], now: datetime
    ) -> None:
        for r in result_ids:
            assert r not in self.ever_assigned(rater_id)  # UNIQUE (rater_id, result_id)
            self.items.append(_Item(rater_id, batch_no, r, now))

    def drop_batch_items(
        self, rater_id: uuid.UUID, result_ids: Collection[uuid.UUID], now: datetime
    ) -> None:
        for i in self.items:
            if i.rater_id == rater_id and i.result_id in result_ids:
                i.dropped_at = now

    def get_draft(self, rater_id: uuid.UUID) -> RatingDraft | None:
        return self.drafts.get(rater_id)

    def put_draft(self, draft: RatingDraft) -> None:
        self.drafts[draft.rater_id] = draft

    def delete_draft(self, rater_id: uuid.UUID) -> None:
        self.drafts.pop(rater_id, None)
