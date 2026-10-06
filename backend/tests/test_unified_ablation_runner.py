"""The Level-1 x Level-2 x bm25_weight x K sweep, offline (PRD-112 /
ARCH-043) — exercises `sweep_questions` (the testable core), not
`run_unified_ablation` (its thin Postgres-session-touching wrapper, same
split precedent as `app.eval.retrieval_tuning.sweep.run_sweep`/
`run_full_ablation` and `app.eval.model_ablation.ablation.run_ablation`/
`run_full_ablation` — see each module's own test file's docstring).

Restructured 2026-09-23 (DEVIATIONS.md #201): Level 3 is now a single
continuous BM25/SapBERT weighted-rank-fusion sweep — MedCPT/RRF are gone,
`recall_at_k` is the new primary metric field."""

from __future__ import annotations

import uuid

import numpy as np
import pytest

from app.config import get_settings
from app.eval.ablation_config import ALL_ARMS
from app.eval.auto_seed import _TOPIC
from app.eval.model_ablation.ablation import _l2_normalize_rows, fetch_corpus
from app.eval.model_ablation.encoders import get_sapbert_encoder
from app.eval.question_gen.deterministic import build_deterministic_narrative
from app.eval.retrieval_tuning.sweep import SweepQuestion
from app.eval.unified_ablation.per_query import PerQueryResult
from app.eval.unified_ablation.runner import sweep_questions
from app.records.concepts import ConceptVocabulary
from app.retrieval.vectorstore import QdrantVectorStore
from tests.test_hybrid_retrieve import _seed_chunk

_DENSE_DIM = 384

RECORD_ID = uuid.uuid4()
RECORD = {
    "record_id": "SYNREC-TEST",
    "mrn": "SYN-TEST",
    "sex": "male",
    "encounter": {"gestational_age_weeks": 34.0, "day_of_life": 1, "care_setting": "NBU"},
    "examination_findings": [
        {"name": "grunting", "present": True},
        {"name": "convulsions", "present": False},
    ],
    "vitals": [{"resp_rate_bpm": 65.0}],
}


@pytest.fixture(autouse=True)
def _stub_backends(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EMBEDDING_BACKEND", "stub")
    monkeypatch.setenv("RERANKER_BACKEND", "stub")
    monkeypatch.setenv("MODEL_ABLATION_BACKEND", "stub")
    # Small grid: keeps the exact-row-count assertions below readable.
    # 4 (level1 x level2) x 2 bm25_weight values x 2 k values = 16 rows for
    # one question -- no more arm branching, Level 3 is one sweep now.
    monkeypatch.setenv("ABLATION_K_VALUES", "2,4")
    monkeypatch.setenv("ABLATION_BM25_WEIGHT_VALUES", "0.0,1.0")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def store() -> QdrantVectorStore:
    s = QdrantVectorStore(url=":memory:", api_key="", collection="unified_ablation_runner_test")
    s.ensure_collection(dense_dim=_DENSE_DIM)
    _seed_chunk(
        store=s,
        point_id=1,
        chunk_id="gold-chunk",
        text="Antibiotics are recommended for suspected neonatal sepsis with grunting.",
    )
    _seed_chunk(
        store=s,
        point_id=2,
        chunk_id="other-chunk",
        text="Routine newborn care does not require antibiotics.",
    )
    return s


def _sweep(
    store: QdrantVectorStore,
    *,
    questions: list[SweepQuestion],
    record_index: dict,
    vocabulary: ConceptVocabulary | None = None,
    **fusion_kwargs,
) -> list[PerQueryResult]:
    corpus = fetch_corpus(store)
    sapbert_encoder = get_sapbert_encoder()
    sapbert_matrix = _l2_normalize_rows(np.asarray(sapbert_encoder.encode(corpus.texts)))
    return list(
        sweep_questions(
            questions,
            store,
            corpus,
            sapbert_encoder=sapbert_encoder,
            sapbert_chunk_matrix=sapbert_matrix,
            record_index=record_index,
            vocabulary=vocabulary,
            experiment_id="exp-1",
            k_grid=get_settings().ablation_k_values_tuple,
            **fusion_kwargs,
        )
    )


def test_sweep_produces_exactly_the_expected_row_count_for_one_question(
    store: QdrantVectorStore,
) -> None:
    question = SweepQuestion(
        question_id="q1",
        text="ignored",
        gold_chunk_ids=frozenset({"gold-chunk"}),
        source_record_id=RECORD_ID,
    )
    rows = _sweep(store, questions=[question], record_index={RECORD_ID: RECORD})
    assert len(rows) == 16  # see the grid comment in _stub_backends above


def test_sweep_covers_all_4_arms_and_both_level1_level2_conditions(
    store: QdrantVectorStore,
) -> None:
    question = SweepQuestion(
        question_id="q1",
        text="ignored",
        gold_chunk_ids=frozenset({"gold-chunk"}),
        source_record_id=RECORD_ID,
    )
    rows = _sweep(store, questions=[question], record_index={RECORD_ID: RECORD})
    seen_arms = {(r.level1_condition, r.level2_condition, r.level3_condition) for r in rows}
    expected_arms = {(a.level1, a.level2, a.level3) for a in ALL_ARMS}
    assert seen_arms == expected_arms


def test_question_with_no_resolvable_source_record_is_skipped(store: QdrantVectorStore) -> None:
    question = SweepQuestion(
        question_id="q1",
        text="ignored",
        gold_chunk_ids=frozenset({"gold-chunk"}),
        source_record_id=uuid.uuid4(),  # not in record_index
    )
    rows = _sweep(store, questions=[question], record_index={})
    assert rows == []


def test_sweep_covers_both_configured_bm25_weight_values(store: QdrantVectorStore) -> None:
    question = SweepQuestion(
        question_id="q1",
        text="ignored",
        gold_chunk_ids=frozenset({"gold-chunk"}),
        source_record_id=RECORD_ID,
    )
    rows = _sweep(store, questions=[question], record_index={RECORD_ID: RECORD})
    assert {r.bm25_weight for r in rows} == {0.0, 1.0}


def test_present_only_and_all_assessed_produce_different_query_text(
    store: QdrantVectorStore,
) -> None:
    """The record has an assessed-absent finding (convulsions) -- Level 1
    must actually differ between the two conditions for this record,
    otherwise the whole comparison would be vacuous."""
    question = SweepQuestion(
        question_id="q1",
        text="ignored",
        gold_chunk_ids=frozenset({"gold-chunk"}),
        source_record_id=RECORD_ID,
    )
    rows = _sweep(store, questions=[question], record_index={RECORD_ID: RECORD})
    present_only_texts = {r.query_text for r in rows if r.level1_condition == "present_only"}
    all_assessed_texts = {r.query_text for r in rows if r.level1_condition == "all_assessed"}
    assert present_only_texts != all_assessed_texts
    assert not any("did NOT have" in t for t in present_only_texts)
    assert any("did NOT have convulsions" in t for t in all_assessed_texts)


def test_recall_and_reciprocal_rank_and_first_relevant_rank_are_consistent(
    store: QdrantVectorStore,
) -> None:
    question = SweepQuestion(
        question_id="q1",
        text="ignored",
        gold_chunk_ids=frozenset({"gold-chunk"}),
        source_record_id=RECORD_ID,
    )
    rows = _sweep(store, questions=[question], record_index={RECORD_ID: RECORD})
    for row in rows:
        rank = row.first_relevant_rank
        if rank is not None and rank <= row.k:
            assert row.reciprocal_rank_at_k == pytest.approx(1.0 / rank)
            assert row.recall_at_k == pytest.approx(1.0)  # single gold chunk -> 0.0 or 1.0
        else:
            assert row.reciprocal_rank_at_k == 0.0
            assert row.recall_at_k == pytest.approx(0.0)
        assert row.relevant_ids == ["gold-chunk"]
        assert len(row.retrieved_ids) <= row.k


def test_multiple_questions_each_produce_their_own_full_row_set(store: QdrantVectorStore) -> None:
    record_id_2 = uuid.uuid4()
    record_2 = {**RECORD, "record_id": "SYNREC-TEST-2"}
    questions = [
        SweepQuestion(
            question_id="q1",
            text="ignored",
            gold_chunk_ids=frozenset({"gold-chunk"}),
            source_record_id=RECORD_ID,
        ),
        SweepQuestion(
            question_id="q2",
            text="ignored",
            gold_chunk_ids=frozenset({"other-chunk"}),
            source_record_id=record_id_2,
        ),
    ]
    rows = _sweep(
        store, questions=questions, record_index={RECORD_ID: RECORD, record_id_2: record_2}
    )
    assert len(rows) == 32  # 16 rows x 2 questions
    assert {r.query_id for r in rows} == {"q1", "q2"}


def test_stored_question_topic_is_used_for_both_level1_conditions(
    store: QdrantVectorStore,
) -> None:
    """DEVIATIONS.md #208: a question seeded under an earlier topic is
    rebuilt with THAT topic (the one its gold set was produced from), not
    the current `auto_seed._TOPIC` -- and the all-assessed rebuild then
    reproduces the stored text exactly."""
    stored_text = build_deterministic_narrative(RECORD, topic="this newborn's presentation")
    question = SweepQuestion(
        question_id="q1",
        text=stored_text,
        gold_chunk_ids=frozenset({"gold-chunk"}),
        source_record_id=RECORD_ID,
    )
    rows = _sweep(store, questions=[question], record_index={RECORD_ID: RECORD})
    assert all(
        r.query_text.startswith(
            "What does the guideline recommend about this newborn's presentation based only on"
        )
        for r in rows
        if r.level2_condition == "raw"
    )
    assert {
        r.query_text
        for r in rows
        if r.level1_condition == "all_assessed" and r.level2_condition == "raw"
    } == {stored_text}


def test_unparseable_stored_text_falls_back_to_current_topic(store: QdrantVectorStore) -> None:
    question = SweepQuestion(
        question_id="q1",
        text="ignored",
        gold_chunk_ids=frozenset({"gold-chunk"}),
        source_record_id=RECORD_ID,
    )
    rows = _sweep(store, questions=[question], record_index={RECORD_ID: RECORD})
    assert all(f"recommend about {_TOPIC} based only on" in r.query_text for r in rows)


def test_rrf_fusion_method_routes_every_ranking_through_rrf(
    store: QdrantVectorStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`fusion_method="rrf"` must reach the ranking, with the configured
    damping constant (DEVIATIONS.md #227)."""
    from app.eval.unified_ablation import runner

    calls: list[int] = []
    real = runner.blend_bm25_dense_rrf

    def spy(*args, **kwargs):
        calls.append(kwargs["rrf_k"])
        return real(*args, **kwargs)

    monkeypatch.setattr(runner, "blend_bm25_dense_rrf", spy)
    question = SweepQuestion(
        question_id="q1",
        text="ignored",
        gold_chunk_ids=frozenset({"gold-chunk"}),
        source_record_id=RECORD_ID,
    )
    rows = _sweep(
        store,
        questions=[question],
        record_index={RECORD_ID: RECORD},
        fusion_method="rrf",
        rrf_damping=7,
    )
    assert rows
    # one ranking per (level1, level2, weight): 2 x 2 x 2 in this grid
    assert calls == [7] * 8


# ── four-area questions (DEVIATIONS.md #264) ──

from pathlib import Path  # noqa: E402

from app.eval.question_gen.areas import load_query_areas  # noqa: E402

_AREAS = load_query_areas(Path(__file__).resolve().parents[2] / "data" / "query_areas.yaml")


def _area_question(area: str) -> SweepQuestion:
    return SweepQuestion(
        question_id=f"q-{area}",
        text="stored text is ignored for four-area questions",
        gold_chunk_ids=frozenset({"gold-chunk"}),
        source_record_id=RECORD_ID,
        area=area,
    )


def test_four_area_question_is_rebuilt_with_its_own_area(store: QdrantVectorStore) -> None:
    rows = _sweep(
        store,
        questions=[_area_question("assessment")],
        record_index={RECORD_ID: RECORD},
        query_areas=_AREAS,
    )
    opening = next(a.opening for a in _AREAS.areas if a.name == "assessment")
    assert rows and all(r.question_area == "assessment" for r in rows)
    texts = {r.level1_condition: r.query_text for r in rows if r.level2_condition == "raw"}
    assert all(opening in t for t in texts.values())
    # assessment lists examination findings: absent findings only in all_assessed
    assert "did NOT have convulsions" in texts["all_assessed"]
    assert "convulsions" not in texts["present_only"]


def test_area_without_findings_gives_identical_level1_texts(store: QdrantVectorStore) -> None:
    rows = _sweep(
        store,
        questions=[_area_question("antibiotic_course")],
        record_index={RECORD_ID: RECORD},
        query_areas=_AREAS,
    )
    texts = {r.level1_condition: r.query_text for r in rows if r.level2_condition == "raw"}
    assert texts["all_assessed"] == texts["present_only"]


def test_four_area_question_with_unknown_area_is_skipped(store: QdrantVectorStore) -> None:
    rows = _sweep(
        store,
        questions=[_area_question("made_up_area")],
        record_index={RECORD_ID: RECORD},
        query_areas=_AREAS,
    )
    assert rows == []
