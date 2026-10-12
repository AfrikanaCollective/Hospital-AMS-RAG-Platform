"""Auto-seed the review queue from de-identified records (ARCH §14.2, §15;
DEVIATIONS.md #113)."""

from __future__ import annotations

import json
import uuid

import pytest

from app.db.models.eval import EvalQuestion, Result
from app.eval import auto_seed
from app.eval.question_gen.deterministic import build_deterministic_narrative

RECORD_ID = uuid.uuid4()
RECORD = {
    "record_id": "DEIDREC-000001",
    "mrn": "MRN-1",
    "sex": "male",
    "encounter": {
        "gestational_age_weeks": 34.0,
        "day_of_life": 3,
        "presenting_complaint": "grunting",
    },
    "problems": ["lethargy"],
}
# DEVIATIONS.md #190: narrative generation is deterministic now (was an LLM
# call) -- computed via the real function under test's own dependency,
# rather than a hand-typed literal, so this constant can't silently drift
# from what `build_deterministic_narrative` actually renders.
RECORD_TEXT = build_deterministic_narrative(RECORD, topic=auto_seed._TOPIC)


class _FakeSession:
    def __init__(self) -> None:
        self.added: list = []

    def add(self, obj: object) -> None:
        self.added.append(obj)

    def flush(self) -> None:
        for obj in self.added:
            if getattr(obj, "id", None) is None:
                obj.id = uuid.uuid4()

    def commit(self) -> None:
        # DEVIATIONS.md #115: `_generate_one_scenario` commits per scenario
        # now. This fake has no real transaction to commit — `.add`/`.flush`
        # already make everything visible to the properties below, so this
        # is a no-op that just lets the real code path run unmodified.
        pass

    @property
    def questions(self) -> list[EvalQuestion]:
        return [o for o in self.added if isinstance(o, EvalQuestion)]

    @property
    def results(self) -> list[Result]:
        return [o for o in self.added if isinstance(o, Result)]


def _well_supported_pipeline_output(*_args: object, **_kwargs: object) -> dict:
    return {
        "observed_outcome": "well_supported",
        "final_answer": {
            "segments": [
                {
                    "type": "claim",
                    "text": "Example Guideline recommends X.",
                    "citation_ids": ["c1"],
                }
            ],
            "citations": [
                {
                    "citation_id": "c1",
                    "chunk_id": "chunk-1",
                    "document_id": "doc-1",
                    "document_title": "Example Guideline",
                    "document_version_id": "v1",
                    "version_label": "2024 edition",
                    "quote": "recommends X",
                    "section_number": "1.2",
                    "page_start": 3,
                    "page_end": 3,
                    "char_start": 0,
                    "char_end": 13,
                    "quote_char_start": 0,
                    "quote_char_end": 13,
                }
            ],
        },
        "retrieval": [{"chunk_id": "chunk-1", "score": 0.9}],
        "grounding_report": {"status": "well_supported"},
        "escalation": None,
    }


def _escalated_pipeline_output(*_args: object, **_kwargs: object) -> dict:
    return {
        "observed_outcome": "escalated",
        "escalation": {"trigger_code": "grounding_failure", "message": "could not verify"},
        "retrieval": [],
        "grounding_report": {},
    }


@pytest.fixture(autouse=True)
def _fast_dedup_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.config import get_settings

    monkeypatch.setenv("QGEN_DEDUP_THRESHOLD", "0.92")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _no_prior_history(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default: no patient id has been used before, no accepted record from
    a prior run exists yet (DEVIATIONS.md #114, #188) — the common case for
    most of these tests. Tests that specifically exercise cross-run
    exclusion override one or both."""
    monkeypatch.setattr(auto_seed, "_USED_PATIENT_IDS_FN", lambda session: set())  # noqa: ARG005
    monkeypatch.setattr(auto_seed, "_EXISTING_RECORDS_FN", lambda session: [])  # noqa: ARG005


@pytest.fixture(autouse=True)
def _no_vocabulary_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default: no attested vocabulary (DEVIATIONS.md #186) — every existing
    test in this file predates the multi-stage audit fields and asserts
    nothing about them, so they must stay behaviorally unchanged regardless
    of whether `data/clinical_concepts.yaml` happens to exist and be
    attested in the environment actually running them (it does inside the
    `api`/`worker` containers, real read-only mount, but not on a bare host
    checkout — this fixture makes the difference irrelevant). Tests that
    specifically exercise the multi-stage path override this."""
    monkeypatch.setattr(
        auto_seed,
        "_LOAD_ATTESTED_VOCABULARY_FN",
        lambda path: (None, "not needed for this test"),  # noqa: ARG005
    )


def test_already_at_target_skips_without_loading_records(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auto_seed, "_COUNT_EXISTING_AUTO_SEEDED_FN", lambda session: 5)  # noqa: ARG005
    called = {"load": False}

    def _load(*_args: object, **_kwargs: object) -> list:
        called["load"] = True
        return []

    monkeypatch.setattr(auto_seed, "_LOAD_RECORDS_FN", _load)
    created = auto_seed.run_auto_seed_review_queue(
        _FakeSession(), target_count=5, composition="100,0,0"
    )
    assert created == []
    assert called["load"] is False


def test_no_deidentified_records_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auto_seed, "_COUNT_EXISTING_AUTO_SEEDED_FN", lambda session: 0)  # noqa: ARG005
    monkeypatch.setattr(auto_seed, "_LOAD_RECORDS_FN", lambda *a, **k: [])  # noqa: ARG005
    created = auto_seed.run_auto_seed_review_queue(
        _FakeSession(), target_count=3, composition="100,0,0"
    )
    assert created == []


def test_tops_up_only_the_remaining_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auto_seed, "_COUNT_EXISTING_AUTO_SEEDED_FN", lambda session: 2)  # noqa: ARG005
    monkeypatch.setattr(
        auto_seed,
        "_LOAD_RECORDS_FN",
        lambda *a, **k: [(RECORD_ID, RECORD)],  # noqa: ARG005
    )
    monkeypatch.setattr(auto_seed, "_INVOKE_PIPELINE_FN", _well_supported_pipeline_output)

    session = _FakeSession()
    created = auto_seed.run_auto_seed_review_queue(session, target_count=3, composition="100,0,0")
    assert len(created) == 1  # target 3, 2 already exist -> exactly 1 more
    assert len(session.questions) == 1
    assert len(session.results) == 1
    result = session.results[0]
    assert result.queue_state == "open"
    assert result.eval_question_id == session.questions[0].id


def test_diversity_filter_caps_generation_across_clinically_identical_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three DIFFERENT records (distinct patient ids, so the record-reuse
    cap alone can't explain the result) but with clinically identical
    content (same problems, no findings/vitals to differ on) -> only the
    first clears the diversity filter; the rest are rejected as near-
    duplicates of it, so generation stops short of the requested count
    (DEVIATIONS.md #188)."""
    monkeypatch.setattr(auto_seed, "_COUNT_EXISTING_AUTO_SEEDED_FN", lambda session: 0)  # noqa: ARG005
    monkeypatch.setattr(
        auto_seed,
        "_LOAD_RECORDS_FN",
        lambda *a, **k: [  # noqa: ARG005
            (uuid.uuid4(), RECORD),
            (uuid.uuid4(), RECORD),
            (uuid.uuid4(), RECORD),
        ],
    )
    monkeypatch.setattr(auto_seed, "_INVOKE_PIPELINE_FN", _well_supported_pipeline_output)

    session = _FakeSession()
    created = auto_seed.run_auto_seed_review_queue(session, target_count=5, composition="100,0,0")
    assert len(created) == 1  # only the first-ever narrative clears the dedup filter


def test_escalated_pipeline_output_persists_with_no_answer_or_citations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(auto_seed, "_COUNT_EXISTING_AUTO_SEEDED_FN", lambda session: 0)  # noqa: ARG005
    monkeypatch.setattr(
        auto_seed,
        "_LOAD_RECORDS_FN",
        lambda *a, **k: [(RECORD_ID, RECORD)],  # noqa: ARG005
    )
    monkeypatch.setattr(auto_seed, "_INVOKE_PIPELINE_FN", _escalated_pipeline_output)

    session = _FakeSession()
    created = auto_seed.run_auto_seed_review_queue(session, target_count=1, composition="100,0,0")
    assert len(created) == 1
    result = session.results[0]
    assert result.queue_state == "open"
    assert result.observed_outcome == "escalated"
    assert result.citations == []
    assert result.answer_enc is None
    assert result.answer_segments_enc is None
    assert session.questions[0].gold_relevant_chunks is None


def test_wellsupported_scenario_persists_its_own_citations_as_gold_relevant_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DEVIATIONS.md #122: `EvalQuestion.gold_relevant_chunks` was never
    written by anything — `app.eval.harness`'s retrieval precision/recall
    and the Phase 6 sweep (PRD-109/ARCH-040) both read it, but it was
    silently `None` for every auto-seeded row. This scenario's own real,
    grounding-verified citations are its gold set."""
    monkeypatch.setattr(auto_seed, "_COUNT_EXISTING_AUTO_SEEDED_FN", lambda session: 0)  # noqa: ARG005
    monkeypatch.setattr(
        auto_seed,
        "_LOAD_RECORDS_FN",
        lambda *a, **k: [(RECORD_ID, RECORD)],  # noqa: ARG005
    )
    monkeypatch.setattr(auto_seed, "_INVOKE_PIPELINE_FN", _well_supported_pipeline_output)

    session = _FakeSession()
    auto_seed.run_auto_seed_review_queue(session, target_count=1, composition="100,0,0")
    assert session.questions[0].gold_relevant_chunks == ["chunk-1"]


def test_transient_gateway_error_during_pipeline_invocation_skips_the_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DEVIATIONS.md #116: a timeout/503 from the real gateway during the
    graph invocation must not crash the whole run — found live, an uncaught
    one stalled generation for ~12 hours. Also asserts nothing is left
    pending in the session (no orphaned EvalQuestion with no Result)."""
    import httpx

    monkeypatch.setattr(auto_seed, "_COUNT_EXISTING_AUTO_SEEDED_FN", lambda session: 0)  # noqa: ARG005
    monkeypatch.setattr(
        auto_seed,
        "_LOAD_RECORDS_FN",
        lambda *a, **k: [(RECORD_ID, RECORD)],  # noqa: ARG005
    )

    def _flaky_invoke(*_args: object, **_kwargs: object) -> dict:
        raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr(auto_seed, "_INVOKE_PIPELINE_FN", _flaky_invoke)

    session = _FakeSession()
    created = auto_seed.run_auto_seed_review_queue(session, target_count=1, composition="100,0,0")
    assert created == []
    assert session.questions == []
    assert session.results == []


def test_run_recovers_after_a_transient_failure_on_an_earlier_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transient failure on one candidate must not poison the rest of the
    run — the next candidate still succeeds normally (DEVIATIONS.md #116)."""
    import httpx

    record_a = uuid.uuid4()
    record_b = uuid.uuid4()
    monkeypatch.setattr(auto_seed, "_COUNT_EXISTING_AUTO_SEEDED_FN", lambda session: 0)  # noqa: ARG005
    monkeypatch.setattr(
        auto_seed,
        "_LOAD_RECORDS_FN",
        lambda *a, **k: [(record_a, RECORD), (record_b, RECORD)],  # noqa: ARG005
    )

    calls = {"n": 0}

    def _fails_once_then_succeeds(*args: object, **kwargs: object) -> dict:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ReadTimeout("timed out")
        return _well_supported_pipeline_output(*args, **kwargs)

    monkeypatch.setattr(auto_seed, "_INVOKE_PIPELINE_FN", _fails_once_then_succeeds)

    session = _FakeSession()
    created = auto_seed.run_auto_seed_review_queue(session, target_count=1, composition="100,0,0")
    assert len(created) == 1
    assert calls["n"] == 2


def test_excludes_patient_ids_already_used_in_a_prior_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A record that already sourced an auto-generated scenario in some
    earlier run must never source a second one (DEVIATIONS.md #114) — the
    whole pool being excluded here means nothing is generated, same as an
    empty pool."""
    monkeypatch.setattr(auto_seed, "_COUNT_EXISTING_AUTO_SEEDED_FN", lambda session: 0)  # noqa: ARG005
    monkeypatch.setattr(
        auto_seed,
        "_LOAD_RECORDS_FN",
        lambda *a, **k: [(RECORD_ID, RECORD)],  # noqa: ARG005
    )
    monkeypatch.setattr(auto_seed, "_USED_PATIENT_IDS_FN", lambda session: {RECORD_ID})  # noqa: ARG005

    session = _FakeSession()
    created = auto_seed.run_auto_seed_review_queue(session, target_count=3, composition="100,0,0")
    assert created == []
    assert session.questions == []


def test_never_generates_two_scenarios_from_the_same_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only 2 distinct records available but 5 are requested — each record
    sources at most one scenario; the run stops short rather than reusing a
    record (DEVIATIONS.md #114). Records are clinically DIFFERENT from each
    other so the diversity filter (DEVIATIONS.md #188) can't also explain
    the result — this test isolates the record-reuse cap specifically."""
    record_a = uuid.uuid4()
    record_b = uuid.uuid4()
    # Differs clinically from RECORD (an added finding) so the diversity
    # filter (DEVIATIONS.md #188) can't also explain a rejection here --
    # this test isolates the record-reuse cap specifically.
    record_b_data = {**RECORD, "examination_findings": [{"name": "floppy", "present": True}]}
    monkeypatch.setattr(auto_seed, "_COUNT_EXISTING_AUTO_SEEDED_FN", lambda session: 0)  # noqa: ARG005
    monkeypatch.setattr(
        auto_seed,
        "_LOAD_RECORDS_FN",
        lambda *a, **k: [(record_a, RECORD), (record_b, record_b_data)],  # noqa: ARG005
    )
    monkeypatch.setattr(auto_seed, "_INVOKE_PIPELINE_FN", _well_supported_pipeline_output)

    session = _FakeSession()
    created = auto_seed.run_auto_seed_review_queue(session, target_count=5, composition="100,0,0")
    assert len(created) == 2
    assert {q.source_record_id for q in session.questions} == {record_a, record_b}


def test_rejects_near_duplicate_narrative_from_a_different_record_in_the_same_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two different records with clinically identical content — the
    second is rejected by the diversity filter, not the record-reuse
    guard: same-record reuse and near-duplicate clinical content are
    independent checks (DEVIATIONS.md #114, #188)."""
    record_a = uuid.uuid4()
    record_b = uuid.uuid4()
    monkeypatch.setattr(auto_seed, "_COUNT_EXISTING_AUTO_SEEDED_FN", lambda session: 0)  # noqa: ARG005
    monkeypatch.setattr(
        auto_seed,
        "_LOAD_RECORDS_FN",
        lambda *a, **k: [(record_a, RECORD), (record_b, RECORD)],  # noqa: ARG005
    )
    monkeypatch.setattr(auto_seed, "_INVOKE_PIPELINE_FN", _well_supported_pipeline_output)

    session = _FakeSession()
    created = auto_seed.run_auto_seed_review_queue(session, target_count=2, composition="100,0,0")
    assert len(created) == 1


def test_rejects_a_narrative_matching_a_previously_queued_records_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even though this run's own `accepted_records` starts from
    `_EXISTING_RECORDS_FN`, not empty — a prior run's queued record must
    still block a clinically-matching new one, otherwise raising
    `QGEN_AUTO_SEED_COUNT` and topping up would silently re-accept
    near-duplicates of what's already queued (DEVIATIONS.md #114, #188)."""
    monkeypatch.setattr(auto_seed, "_COUNT_EXISTING_AUTO_SEEDED_FN", lambda session: 0)  # noqa: ARG005
    monkeypatch.setattr(
        auto_seed,
        "_LOAD_RECORDS_FN",
        lambda *a, **k: [(RECORD_ID, RECORD)],  # noqa: ARG005
    )
    monkeypatch.setattr(auto_seed, "_EXISTING_RECORDS_FN", lambda session: [RECORD])  # noqa: ARG005
    monkeypatch.setattr(auto_seed, "_INVOKE_PIPELINE_FN", _well_supported_pipeline_output)

    session = _FakeSession()
    created = auto_seed.run_auto_seed_review_queue(session, target_count=1, composition="100,0,0")
    assert created == []


def test_well_supported_result_answer_decrypts_to_the_rendered_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.crypto.provider import get_crypto

    monkeypatch.setattr(auto_seed, "_COUNT_EXISTING_AUTO_SEEDED_FN", lambda session: 0)  # noqa: ARG005
    monkeypatch.setattr(
        auto_seed,
        "_LOAD_RECORDS_FN",
        lambda *a, **k: [(RECORD_ID, RECORD)],  # noqa: ARG005
    )
    monkeypatch.setattr(auto_seed, "_INVOKE_PIPELINE_FN", _well_supported_pipeline_output)

    session = _FakeSession()
    auto_seed.run_auto_seed_review_queue(session, target_count=1, composition="100,0,0")
    result = session.results[0]
    crypto = get_crypto()
    from app.db.models.eval import result_answer_aad, result_segments_aad

    plaintext = crypto.decrypt(result.answer_enc, aad=result_answer_aad(result.id))
    assert plaintext.decode("utf-8") == "Example Guideline recommends X."
    assert result.citations[0]["citation_id"] == "c1"

    # DEVIATIONS.md #120: the per-segment citation_ids, needed for the
    # review-queue UI to link a citation inline to the claim it supports,
    # are now persisted alongside the flattened answer text, not discarded.
    segments = json.loads(
        crypto.decrypt(result.answer_segments_enc, aad=result_segments_aad(result.id))
    )
    assert segments == [
        {
            "type": "claim",
            "text": "Example Guideline recommends X.",
            "citation_ids": ["c1"],
            "grounding_note": None,
        }
    ]


# ── Multi-stage audit fields (DEVIATIONS.md #186) ────────────────────────────


class _FakeAugmentedQuery:
    def __init__(self, text: str, *, fired: bool) -> None:
        self.text = text
        self.fired = fired


def _multi_stage_pipeline_output(*_args: object, **_kwargs: object) -> dict:
    """A second, distinguishable pipeline output — lets a test tell the
    single-stage and multi-stage results apart by content, not just by
    which field they landed in."""
    return {
        "observed_outcome": "well_supported",
        "final_answer": {
            "segments": [
                {
                    "type": "claim",
                    "text": "Example Guideline recommends Y for the augmented query.",
                    "citation_ids": ["c2"],
                }
            ],
            "citations": [
                {
                    "citation_id": "c2",
                    "chunk_id": "chunk-2",
                    "document_id": "doc-1",
                    "document_title": "Example Guideline",
                    "document_version_id": "v1",
                    "version_label": "2024 edition",
                    "quote": "recommends Y",
                    "section_number": "1.3",
                    "page_start": 4,
                    "page_end": 4,
                    "char_start": 0,
                    "char_end": 13,
                    "quote_char_start": 0,
                    "quote_char_end": 13,
                }
            ],
        },
        "retrieval": [{"chunk_id": "chunk-2", "score": 0.8}],
        "grounding_report": {"status": "well_supported"},
        "escalation": None,
    }


def test_multi_stage_fields_absent_when_vocabulary_not_attested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default fixture (`_no_vocabulary_by_default`) applies -- confirms the
    common case explicitly rather than only by omission."""
    monkeypatch.setattr(auto_seed, "_COUNT_EXISTING_AUTO_SEEDED_FN", lambda session: 0)  # noqa: ARG005
    monkeypatch.setattr(
        auto_seed,
        "_LOAD_RECORDS_FN",
        lambda *a, **k: [(RECORD_ID, RECORD)],  # noqa: ARG005
    )
    monkeypatch.setattr(auto_seed, "_INVOKE_PIPELINE_FN", _well_supported_pipeline_output)

    session = _FakeSession()
    auto_seed.run_auto_seed_review_queue(session, target_count=1, composition="100,0,0")
    result = session.results[0]
    assert result.multi_stage_query is None
    assert result.multi_stage_fired is False
    assert result.multi_stage_answer_enc is None
    assert result.multi_stage_citations == []


def test_multi_stage_query_recorded_but_not_run_when_augmentation_does_not_fire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Vocabulary attested, but this record matches no concept -- the query
    is recorded as identical to the single-stage one and NO second pipeline
    call is made (no wasted LLM-gateway cost for a query that wouldn't
    differ)."""
    monkeypatch.setattr(auto_seed, "_COUNT_EXISTING_AUTO_SEEDED_FN", lambda session: 0)  # noqa: ARG005
    monkeypatch.setattr(
        auto_seed,
        "_LOAD_RECORDS_FN",
        lambda *a, **k: [(RECORD_ID, RECORD)],  # noqa: ARG005
    )
    monkeypatch.setattr(
        auto_seed,
        "_LOAD_ATTESTED_VOCABULARY_FN",
        lambda path: (object(), None),  # noqa: ARG005
    )

    calls: list[str] = []

    def _invoke(question_text: str, **_kwargs: object) -> dict:
        calls.append(question_text)
        return _well_supported_pipeline_output()

    monkeypatch.setattr(auto_seed, "_INVOKE_PIPELINE_FN", _invoke)
    monkeypatch.setattr(
        auto_seed,
        "_BUILD_ARM_C_QUERY_FN",
        lambda text, vocabulary, record: _FakeAugmentedQuery(text, fired=False),  # noqa: ARG005
    )

    session = _FakeSession()
    auto_seed.run_auto_seed_review_queue(session, target_count=1, composition="100,0,0")
    result = session.results[0]
    assert len(calls) == 1  # only the single-stage invocation -- no wasted second call
    assert result.multi_stage_query == RECORD_TEXT
    assert result.multi_stage_fired is False
    assert result.multi_stage_answer_enc is None
    assert result.multi_stage_citations == []


def test_multi_stage_runs_a_second_pipeline_call_and_persists_its_own_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Augmentation fires -- a genuine second real-pipeline invocation is
    made against the augmented text, and its full answer/citations are
    persisted alongside (not instead of) the single-stage ones."""
    from app.crypto.provider import get_crypto
    from app.db.models.eval import result_multi_stage_answer_aad

    monkeypatch.setattr(auto_seed, "_COUNT_EXISTING_AUTO_SEEDED_FN", lambda session: 0)  # noqa: ARG005
    monkeypatch.setattr(
        auto_seed,
        "_LOAD_RECORDS_FN",
        lambda *a, **k: [(RECORD_ID, RECORD)],  # noqa: ARG005
    )
    monkeypatch.setattr(
        auto_seed,
        "_LOAD_ATTESTED_VOCABULARY_FN",
        lambda path: (object(), None),  # noqa: ARG005
    )

    augmented_text = RECORD_TEXT + "; fake fast breathing"
    calls: list[str] = []

    def _invoke(question_text: str, **_kwargs: object) -> dict:
        calls.append(question_text)
        if question_text == augmented_text:
            return _multi_stage_pipeline_output()
        return _well_supported_pipeline_output()

    monkeypatch.setattr(auto_seed, "_INVOKE_PIPELINE_FN", _invoke)
    monkeypatch.setattr(
        auto_seed,
        "_BUILD_ARM_C_QUERY_FN",
        lambda text, vocabulary, record: _FakeAugmentedQuery(augmented_text, fired=True),  # noqa: ARG005
    )

    session = _FakeSession()
    auto_seed.run_auto_seed_review_queue(session, target_count=1, composition="100,0,0")
    result = session.results[0]

    assert calls == [RECORD_TEXT, augmented_text]  # single-stage, then multi-stage
    assert result.multi_stage_query == augmented_text
    assert result.multi_stage_fired is True
    assert result.multi_stage_citations[0]["citation_id"] == "c2"
    assert result.multi_stage_retrieval_snapshot == {
        "items": [{"chunk_id": "chunk-2", "score": 0.8}]
    }
    assert result.multi_stage_grounding_report == {"status": "well_supported"}
    # The single-stage answer is untouched by the multi-stage run.
    assert result.citations[0]["citation_id"] == "c1"

    crypto = get_crypto()
    plaintext = crypto.decrypt(
        result.multi_stage_answer_enc, aad=result_multi_stage_answer_aad(result.id)
    )
    assert plaintext.decode("utf-8") == "Example Guideline recommends Y for the augmented query."


def test_multi_stage_transient_gateway_error_keeps_the_single_stage_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transient failure on the audit-only multi-stage side channel must
    not discard the (already valid, already paid-for) single-stage result —
    degrades to 'no multi-stage answer for this item', not a lost scenario."""
    import httpx

    monkeypatch.setattr(auto_seed, "_COUNT_EXISTING_AUTO_SEEDED_FN", lambda session: 0)  # noqa: ARG005
    monkeypatch.setattr(
        auto_seed,
        "_LOAD_RECORDS_FN",
        lambda *a, **k: [(RECORD_ID, RECORD)],  # noqa: ARG005
    )
    monkeypatch.setattr(
        auto_seed,
        "_LOAD_ATTESTED_VOCABULARY_FN",
        lambda path: (object(), None),  # noqa: ARG005
    )

    augmented_text = RECORD_TEXT + "; fake fast breathing"

    def _invoke(question_text: str, **_kwargs: object) -> dict:
        if question_text == augmented_text:
            raise httpx.ConnectTimeout("boom")
        return _well_supported_pipeline_output()

    monkeypatch.setattr(auto_seed, "_INVOKE_PIPELINE_FN", _invoke)
    monkeypatch.setattr(
        auto_seed,
        "_BUILD_ARM_C_QUERY_FN",
        lambda text, vocabulary, record: _FakeAugmentedQuery(augmented_text, fired=True),  # noqa: ARG005
    )

    session = _FakeSession()
    created = auto_seed.run_auto_seed_review_queue(session, target_count=1, composition="100,0,0")
    assert len(created) == 1  # the scenario still succeeds overall
    result = session.results[0]
    assert result.citations[0]["citation_id"] == "c1"  # single-stage answer intact
    # multi_stage_query/fired still recorded truthfully -- augmentation DID
    # produce a different query, only the run of it failed.
    assert result.multi_stage_query == augmented_text
    assert result.multi_stage_fired is True
    assert result.multi_stage_answer_enc is None
    assert result.multi_stage_citations == []
