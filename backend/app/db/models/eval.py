"""`eval` schema — questions, results, and the multi-rater rubric workflow
(ARCH §4.5, §14; PRD-040..PRD-048, PRD-060..PRD-073).

Required tables (per the Phase 1 brief): rubric domains, scores, rater
identity, review-queue state, IRR results, archive status. Every question/result
carries `provenance` (auto_generated | clinician_submitted). Auto-generated
questions also carry `expected_outcome` so the harness scores against
expectation, not just output quality.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPk

SCHEMA = "eval"


class EvalQuestion(UUIDPk, TimestampMixin, Base):
    __tablename__ = "eval_question"
    __table_args__ = {"schema": SCHEMA}

    text: Mapped[str] = mapped_column(Text)
    provenance: Mapped[str] = mapped_column(String(24))  # auto_generated | clinician_submitted
    expected_outcome: Mapped[str | None] = mapped_column(
        String(24)
    )  # well_supported | missing_info_expected | no_guideline_expected  (auto-generated)
    source_record_id: Mapped[uuid.UUID | None] = mapped_column()
    target_guideline_ref: Mapped[dict | None] = mapped_column(
        JSONB
    )  # None for no_guideline_expected
    gold_relevant_chunks: Mapped[list | None] = mapped_column(JSONB)
    gold_citations: Mapped[list | None] = mapped_column(JSONB)
    generator_meta: Mapped[dict | None] = mapped_column(
        JSONB
    )  # model_id, template_version, validator_report
    in_fixed_testset: Mapped[bool] = mapped_column(Boolean, default=False)


def result_answer_aad(result_id: uuid.UUID) -> bytes:
    """AAD for `Result.answer_enc` — the shared read/write contract between
    the writer (`app.eval.auto_seed._build_result`) and the reader
    (`app.api.routes.review_queue.get_queue_item`), defined once here so
    they cannot drift apart (DEVIATIONS.md #79, #113)."""
    return b"eval-result-answer:" + str(result_id).encode("utf-8")


def result_segments_aad(result_id: uuid.UUID) -> bytes:
    """AAD for `Result.answer_segments_enc` — deliberately distinct from
    `result_answer_aad` (a different label, same `result_id`) so the two
    ciphertexts, despite overlapping plaintext, are never valid for each
    other's slot (DEVIATIONS.md #120)."""
    return b"eval-result-segments:" + str(result_id).encode("utf-8")


def result_multi_stage_answer_aad(result_id: uuid.UUID) -> bytes:
    """AAD for `Result.multi_stage_answer_enc` (DEVIATIONS.md #186) — a
    third, distinct label (same pattern as `result_segments_aad` vs.
    `result_answer_aad`) so this audit-only answer's ciphertext is never
    valid in the single-stage answer's slot or vice versa."""
    return b"eval-result-multi-stage-answer:" + str(result_id).encode("utf-8")


class Result(UUIDPk, TimestampMixin, Base):
    """id == result_id referenced by ratings (ARCH §4.5)."""

    __tablename__ = "result"
    __table_args__ = {"schema": SCHEMA}

    eval_question_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey(f"{SCHEMA}.eval_question.id")
    )
    message_id: Mapped[uuid.UUID | None] = mapped_column()  # link to a live conversation turn
    provenance: Mapped[str] = mapped_column(String(24))  # inherited
    expected_outcome: Mapped[str | None] = mapped_column(String(24))  # inherited
    observed_outcome: Mapped[str | None] = mapped_column(
        String(16)
    )  # well_supported | missing_info | no_guideline | escalated
    answer_enc: Mapped[bytes | None] = mapped_column(LargeBinary)
    answer_segments_enc: Mapped[bytes | None] = mapped_column(LargeBinary)
    citations: Mapped[list] = mapped_column(JSONB, default=list)
    retrieval_snapshot: Mapped[dict] = mapped_column(JSONB, default=dict)
    grounding_report: Mapped[dict] = mapped_column(JSONB, default=dict)
    config_snapshot: Mapped[dict] = mapped_column(
        JSONB, default=dict
    )  # model ids, thresholds, corpus snapshot
    queue_state: Mapped[str] = mapped_column(
        String(16), default="not_queued"
    )  # not_queued | open | archived

    # Audit-only multi-stage (Phase 7 vocabulary-augmented, PRD-111 Arm C)
    # counterpart to the single-stage fields above (DEVIATIONS.md #186) —
    # never rated, never read by `app.rubric.workflow`; exposed read-only by
    # `GET /review-queue/{id}` for traceability/comparison. `multi_stage_query`
    # is plaintext, matching `EvalQuestion.text`'s own (unencrypted)
    # convention for generated narrative/query text; `multi_stage_fired=False`
    # means the augmentation produced no different query for this record (no
    # concept matched), in which case the *_enc/citations/reports fields below
    # are left at their column defaults rather than duplicating the
    # single-stage answer, to avoid a second real pipeline invocation (and
    # its LLM-gateway cost) for a query that would be byte-identical anyway.
    multi_stage_query: Mapped[str | None] = mapped_column(Text)
    multi_stage_fired: Mapped[bool] = mapped_column(Boolean, default=False)
    multi_stage_answer_enc: Mapped[bytes | None] = mapped_column(LargeBinary)
    multi_stage_citations: Mapped[list] = mapped_column(JSONB, default=list)
    multi_stage_retrieval_snapshot: Mapped[dict] = mapped_column(JSONB, default=dict)
    multi_stage_grounding_report: Mapped[dict] = mapped_column(JSONB, default=dict)


class RubricDomain(Base):
    """Static reference — 11 rows seeded from app.rubric.domains (ARCH §14.1)."""

    __tablename__ = "rubric_domain"
    __table_args__ = {"schema": SCHEMA}

    code: Mapped[str] = mapped_column(String(48), primary_key=True)
    ordinal: Mapped[int] = mapped_column(Integer)
    name: Mapped[str] = mapped_column(Text)
    definition: Mapped[str] = mapped_column(Text)
    anchor_1: Mapped[str] = mapped_column(Text)
    anchor_2: Mapped[str] = mapped_column(Text)
    anchor_3: Mapped[str] = mapped_column(Text)
    anchor_4: Mapped[str] = mapped_column(Text)
    anchor_5: Mapped[str] = mapped_column(Text)


class RatingRound(UUIDPk, Base):
    """One rater's pass over one result. UNIQUE (result_id, rater_id) enforces
    'distinct clinicians' — a rater completes at most one round per result
    (PRD-043)."""

    __tablename__ = "rating_round"
    __table_args__ = (
        UniqueConstraint("result_id", "rater_id", name="distinct_rater_per_result"),
        {"schema": SCHEMA},
    )

    result_id: Mapped[uuid.UUID] = mapped_column(ForeignKey(f"{SCHEMA}.result.id"))
    rater_id: Mapped[uuid.UUID] = mapped_column()
    is_original_rater: Mapped[bool] = mapped_column(Boolean, default=False)
    accept_action_id: Mapped[uuid.UUID | None] = mapped_column()  # -> hitl.hitl_decision
    submitted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class RubricRating(UUIDPk, Base):
    """One score per domain per rater per result (ARCH §4.5)."""

    __tablename__ = "rubric_rating"
    __table_args__ = (
        UniqueConstraint(
            "result_id", "rater_id", "domain_code", name="one_score_per_domain_per_rater"
        ),
        {"schema": SCHEMA},
    )

    result_id: Mapped[uuid.UUID] = mapped_column(ForeignKey(f"{SCHEMA}.result.id"))
    rater_id: Mapped[uuid.UUID] = mapped_column()
    domain_code: Mapped[str] = mapped_column(ForeignKey(f"{SCHEMA}.rubric_domain.code"))
    score: Mapped[int] = mapped_column(SmallInteger)  # 1..5
    rated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    rating_round_id: Mapped[uuid.UUID] = mapped_column(ForeignKey(f"{SCHEMA}.rating_round.id"))
    comment: Mapped[str | None] = mapped_column(Text)  # optional adjunct only


class IRRScore(UUIDPk, Base):
    """Per-result IRR (indicative). Computed at >= IRR_MIN_RATERS distinct
    raters, then the result is archived (ARCH §14.4)."""

    __tablename__ = "irr_score"
    __table_args__ = {"schema": SCHEMA}

    result_id: Mapped[uuid.UUID] = mapped_column(ForeignKey(f"{SCHEMA}.result.id"))
    domain_code: Mapped[str | None] = mapped_column(String(48))  # None = aggregate
    metric: Mapped[str] = mapped_column(String(48), default="krippendorff_alpha_ordinal")
    value: Mapped[float] = mapped_column(Float)
    n_raters: Mapped[int] = mapped_column(Integer)
    n_items: Mapped[int] = mapped_column(Integer, default=1)
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class IRRBatch(UUIDPk, TimestampMixin, Base):
    """Corpus/slice-level IRR — the statistic that matters for evidence
    (ARCH §14.4). NEVER pools auto_generated + clinician_submitted by default
    (PRD-046): the slice definition records the filter used."""

    __tablename__ = "irr_batch"
    __table_args__ = {"schema": SCHEMA}

    slice_definition: Mapped[dict] = mapped_column(
        JSONB
    )  # e.g. {provenance, expected_outcome, ...}
    metric: Mapped[str] = mapped_column(String(48), default="krippendorff_alpha_ordinal")
    per_domain: Mapped[dict] = mapped_column(JSONB, default=dict)  # {domain_code: alpha}
    secondary: Mapped[dict] = mapped_column(JSONB, default=dict)  # {gwet_ac2: {...}, icc_2k: {...}}
    n_items: Mapped[int] = mapped_column(Integer)
    n_raters: Mapped[int] = mapped_column(Integer)
    config_snapshot: Mapped[dict] = mapped_column(JSONB, default=dict)


class ResultArchive(TimestampMixin, Base):
    """Archive status + immutable rating history + IRR snapshot (ARCH §14.5)."""

    __tablename__ = "result_archive"
    __table_args__ = {"schema": SCHEMA}

    result_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey(f"{SCHEMA}.result.id"), primary_key=True
    )
    archived_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    rating_history: Mapped[dict] = mapped_column(JSONB)  # rounds, ratings, accept actions
    irr_snapshot: Mapped[dict] = mapped_column(JSONB)  # all irr_score rows at archival
    provenance: Mapped[str] = mapped_column(String(24))  # carried for separable reporting


class EvalRun(UUIDPk, TimestampMixin, Base):
    """A harness run against the fixed test set (ARCH §16; PRD-072, PRD-073)."""

    __tablename__ = "eval_run"
    __table_args__ = {"schema": SCHEMA}

    snapshot_label: Mapped[str] = mapped_column(String(64))
    corpus_snapshot_id: Mapped[uuid.UUID | None] = mapped_column()
    config_snapshot: Mapped[dict] = mapped_column(JSONB, default=dict)
    report: Mapped[dict] = mapped_column(
        JSONB, default=dict
    )  # metrics by expected_outcome + by provenance
    passed: Mapped[bool] = mapped_column(Boolean, default=False)


class ReviewBatchItem(UUIDPk, Base):
    """One fresh result assigned to a reviewer's batch (DEVIATIONS.md #294).

    A reviewer works through items other reviewers have started first; only
    when none are left are they given a batch of up to `REVIEW_BATCH_SIZE`
    fresh (unrated) results, kept until every item in it is rated by them.
    `batch_no` counts a reviewer's batches; the current batch is their
    highest. An item archived by others before this reviewer rates it is
    `dropped_at` and replaced. UNIQUE (rater_id, result_id): a result is
    assigned to a reviewer at most once, ever."""

    __tablename__ = "review_batch_item"
    __table_args__ = (
        UniqueConstraint("rater_id", "result_id", name="review_batch_item_once_per_rater"),
        {"schema": SCHEMA},
    )

    rater_id: Mapped[uuid.UUID] = mapped_column(index=True)
    batch_no: Mapped[int] = mapped_column(Integer)
    result_id: Mapped[uuid.UUID] = mapped_column(ForeignKey(f"{SCHEMA}.result.id"), index=True)
    assigned_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    dropped_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class RatingDraft(Base):
    """A reviewer's one half-finished rating (DEVIATIONS.md #294): at most one
    per reviewer (the primary key), kept until they submit it. The rubric
    state is encrypted with `rating_draft_aad`."""

    __tablename__ = "rating_draft"
    __table_args__ = {"schema": SCHEMA}

    rater_id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    result_id: Mapped[uuid.UUID] = mapped_column(ForeignKey(f"{SCHEMA}.result.id"))
    payload_enc: Mapped[bytes] = mapped_column(LargeBinary)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


def rating_draft_aad(rater_id: uuid.UUID, result_id: uuid.UUID) -> bytes:
    """AAD for `RatingDraft.payload_enc`, shared by writer and reader."""
    return f"eval.rating_draft:{rater_id}:{result_id}".encode()
