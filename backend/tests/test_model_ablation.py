"""Model-ablation harness (PRD-110 / ARCH-041), run against qdrant-client's
embedded in-memory mode with stub embedding/encoder backends — no network,
no real models, no real Postgres (CLAUDE.md §5). Mirrors
tests/test_retrieval_tuning.py's fixture pattern.

`fetch_calibration_questions` (reused from `app.eval.retrieval_tuning.sweep`)
is not covered here — see that module's own test file; this file only
exercises code new to the ablation harness."""

from __future__ import annotations

import numpy as np
import pytest

from app.config import get_settings
from app.eval.metrics import mrr, precision_recall_at_k
from app.eval.model_ablation.ablation import (
    ALL_ARMS,
    ARMS,
    MRR_K,
    MULTI_STAGE,
    REFERENCE_ARM,
    SINGLE_STAGE,
    QuestionArmRankings,
    _bm25_rank,
    _bootstrap_ci,
    _cosine_rank,
    _rrf_combine,
    fetch_corpus,
    rank_question,
    run_ablation,
)
from app.eval.model_ablation.encoders import Encoder, get_medcpt_encoders, get_sapbert_encoder
from app.eval.retrieval_tuning.sweep import SweepQuestion
from app.records.concepts import Concept, ConceptVocabulary
from app.retrieval.vectorstore import QdrantVectorStore
from app.schemas.record import PatientRecord
from tests.test_hybrid_retrieve import _seed_chunk

_DENSE_DIM = 384

_VOCABULARY = ConceptVocabulary(
    authored_by="unit-test fixture",
    authored_date="2026-09-19",
    concepts=(
        Concept(
            name="fake_tachypnoea",
            field="vitals.resp_rate_bpm",
            operator=">",
            value=60.0,
            source="unit-test fixture, not a real clinical value",
            synonyms=("fake fast breathing",),
        ),
    ),
)

_RECORD_THAT_MATCHES = PatientRecord.model_validate(
    {
        "schema_version": "1.4.0",
        "dataset_provenance": "synthetic-generator-v1",
        "record_id": "test-rec-1",
        "mrn": "SYN-TEST-1",
        "vitals": [{"resp_rate_bpm": 65.0}],
    }
)


@pytest.fixture(autouse=True)
def _stub_backends(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EMBEDDING_BACKEND", "stub")
    monkeypatch.setenv("RERANKER_BACKEND", "stub")
    monkeypatch.setenv("MODEL_ABLATION_BACKEND", "stub")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def store() -> QdrantVectorStore:
    s = QdrantVectorStore(url=":memory:", api_key="", collection="model_ablation_test")
    s.ensure_collection(dense_dim=_DENSE_DIM)
    return s


def test_rrf_combine_rewards_a_chunk_ranked_high_in_every_channel() -> None:
    ranked = _rrf_combine([["a", "b", "c"], ["b", "a", "c"]], rrf_k=60)
    assert ranked[0] in {"a", "b"}  # both channels rank one of these top-2
    assert ranked[-1] == "c"  # last in both channels


def test_rrf_combine_breaks_ties_by_chunk_id() -> None:
    ranked = _rrf_combine([["x", "y"], ["y", "x"]], rrf_k=60)
    # symmetric scores -> tie -> deterministic alphabetical order
    assert ranked == ["x", "y"]


def test_cosine_rank_orders_by_similarity_descending() -> None:
    chunk_ids = ["near", "far", "mid"]
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])
    ranked = _cosine_rank([1.0, 0.0], matrix, chunk_ids)
    assert ranked == ["near", "mid", "far"]


def test_bm25_rank_appends_zero_score_chunks_last_and_sorted(store: QdrantVectorStore) -> None:
    target_text = "Blood cultures are recommended before starting antimicrobials."
    _seed_chunk(store, 1, chunk_id="c1", text=target_text)
    _seed_chunk(store, 2, chunk_id="c2", text="zzz completely unrelated content zzz")
    _seed_chunk(store, 3, chunk_id="c3", text="yyy also completely unrelated yyy")

    ranked = _bm25_rank(store, target_text, ["c1", "c2", "c3"])
    assert ranked[0] == "c1"
    assert ranked[1:] == sorted(ranked[1:])  # zero-score tail is deterministically sorted


def test_bm25_rank_excludes_superseded_and_held_chunks(store: QdrantVectorStore) -> None:
    """An unfiltered sparse query let superseded near-copies crowd the
    `limit=len(corpus)` window and enter the ranking (DEVIATIONS.md #226)."""
    text = "Blood cultures are recommended before starting antimicrobials."
    _seed_chunk(store, 1, chunk_id="old", text=text + " blood cultures", status="superseded")
    _seed_chunk(store, 2, chunk_id="held", text=text + " blood", review_status="pending")
    _seed_chunk(store, 3, chunk_id="active", text=text)

    assert _bm25_rank(store, text, ["active"]) == ["active"]


def test_fetch_corpus_returns_every_seeded_chunk(store: QdrantVectorStore) -> None:
    _seed_chunk(store, 1, chunk_id="c1", text="first chunk")
    _seed_chunk(store, 2, chunk_id="c2", text="second chunk")

    corpus = fetch_corpus(store)
    assert set(corpus.chunk_ids) == {"c1", "c2"}
    assert "first chunk" in corpus.texts


def test_rank_question_returns_every_arm(store: QdrantVectorStore) -> None:
    target_text = "Blood cultures are recommended before starting antimicrobials."
    _seed_chunk(store, 1, chunk_id="c1", text=target_text)
    _seed_chunk(store, 2, chunk_id="c2", text="Vitamin K is given to newborns shortly after birth.")

    corpus = fetch_corpus(store)
    sapbert_encoder = get_sapbert_encoder()
    medcpt_query_encoder, medcpt_article_encoder = get_medcpt_encoders()
    sapbert_matrix = np.array(sapbert_encoder.encode(corpus.texts))
    medcpt_matrix = np.array(medcpt_article_encoder.encode(corpus.texts))

    question = SweepQuestion(question_id="q1", text=target_text, gold_chunk_ids=frozenset({"c1"}))
    result = rank_question(
        store,
        question,
        corpus,
        sapbert_encoder=sapbert_encoder,
        sapbert_chunk_matrix=sapbert_matrix,
        medcpt_query_encoder=medcpt_query_encoder,
        medcpt_chunk_matrix=medcpt_matrix,
        rrf_k=60,
    )

    assert set(result.rankings) == set(ALL_ARMS)
    for arm in ALL_ARMS:
        assert set(result.rankings[arm]) == {"c1", "c2"}
    # No vocabulary passed -- multi-stage not computed, every prior caller's
    # behavior preserved (DEVIATIONS.md #184).
    assert result.multi_stage_rankings is None
    assert result.multi_stage_fired is False


def _rank_with_vocabulary(
    store: QdrantVectorStore, *, record: PatientRecord | None
) -> QuestionArmRankings:
    target_text = "Blood cultures are recommended before starting antimicrobials."
    corpus = fetch_corpus(store)
    sapbert_encoder = get_sapbert_encoder()
    medcpt_query_encoder, medcpt_article_encoder = get_medcpt_encoders()
    sapbert_matrix = np.array(sapbert_encoder.encode(corpus.texts))
    medcpt_matrix = np.array(medcpt_article_encoder.encode(corpus.texts))

    question = SweepQuestion(question_id="q1", text=target_text, gold_chunk_ids=frozenset({"c1"}))
    return rank_question(
        store,
        question,
        corpus,
        sapbert_encoder=sapbert_encoder,
        sapbert_chunk_matrix=sapbert_matrix,
        medcpt_query_encoder=medcpt_query_encoder,
        medcpt_chunk_matrix=medcpt_matrix,
        rrf_k=60,
        vocabulary=_VOCABULARY,
        record=record,
    )


def test_rank_question_computes_multi_stage_when_augmentation_fires(
    store: QdrantVectorStore,
) -> None:
    _seed_chunk(store, 1, chunk_id="c1", text="Blood cultures are recommended.")
    _seed_chunk(store, 2, chunk_id="c2", text="Vitamin K is given to newborns.")

    result = _rank_with_vocabulary(store, record=_RECORD_THAT_MATCHES)

    assert result.multi_stage_fired is True
    assert result.multi_stage_rankings is not None
    assert set(result.multi_stage_rankings) == set(ALL_ARMS)
    for arm in ALL_ARMS:
        assert set(result.multi_stage_rankings[arm]) == {"c1", "c2"}


def test_rank_question_multi_stage_falls_back_to_single_stage_without_a_record(
    store: QdrantVectorStore,
) -> None:
    _seed_chunk(store, 1, chunk_id="c1", text="Blood cultures are recommended.")
    _seed_chunk(store, 2, chunk_id="c2", text="Vitamin K is given to newborns.")

    result = _rank_with_vocabulary(store, record=None)

    assert result.multi_stage_fired is False
    assert result.multi_stage_rankings == result.rankings


def test_encoder_stub_is_deterministic_and_independent_of_backend_instance() -> None:
    e1 = Encoder("some/model", backend="stub")
    e2 = Encoder("some/model", backend="stub")
    assert e1.encode(["hello world"]) == e2.encode(["hello world"])
    assert e1.encode(["hello world"]) != e1.encode(["goodbye world"])


def test_run_ablation_recall_and_mrr_on_a_small_known_grid() -> None:
    # sapbert_bm25 ranks gold1 first for q1; medcpt_bm25 ranks it first for q2;
    # sapbert_medcpt_bm25 and the rrf_production reference both rank the gold
    # chunk first for both, by construction.
    rankings = [
        QuestionArmRankings(
            question_id="q1",
            gold_chunk_ids=frozenset({"gold1"}),
            rankings={
                "sapbert_bm25": ["gold1", "distractor"],
                "medcpt_bm25": ["distractor", "gold1"],
                "sapbert_medcpt_bm25": ["gold1", "distractor"],
                REFERENCE_ARM: ["gold1", "distractor"],
            },
        ),
        QuestionArmRankings(
            question_id="q2",
            gold_chunk_ids=frozenset({"gold2"}),
            rankings={
                "sapbert_bm25": ["distractor", "gold2"],
                "medcpt_bm25": ["gold2", "distractor"],
                "sapbert_medcpt_bm25": ["gold2", "distractor"],
                REFERENCE_ARM: ["gold2", "distractor"],
            },
        ),
    ]

    result = run_ablation(rankings)

    recall_sapbert_bm25 = next(
        row["recall"]
        for row in result.recall_rows
        if row["k"] == 2 and row["arm"] == "sapbert_bm25"
    )
    expected = sum(
        precision_recall_at_k(r.rankings["sapbert_bm25"], set(r.gold_chunk_ids), 2)[1]
        for r in rankings
    ) / len(rankings)
    assert recall_sapbert_bm25 == pytest.approx(expected)

    recall_combined = next(
        row["recall"]
        for row in result.recall_rows
        if row["k"] == 2 and row["arm"] == "sapbert_medcpt_bm25"
    )
    assert recall_combined == pytest.approx(1.0)  # gold chunk top-2 for both questions

    mrr_medcpt_bm25 = next(row["mrr"] for row in result.mrr_rows if row["arm"] == "medcpt_bm25")
    expected_mrr = sum(
        mrr(r.rankings["medcpt_bm25"][:MRR_K], set(r.gold_chunk_ids)) for r in rankings
    ) / len(rankings)
    assert mrr_medcpt_bm25 == pytest.approx(expected_mrr)

    assert result.n_questions == 2
    assert {row["arm"] for row in result.mrr_rows} == set(ALL_ARMS)

    # Every mrr_row carries a bootstrap CI that brackets its own mean — with
    # only 2 questions this interval is wide/not statistically meaningful
    # (expected; the real report bootstraps over 67 real questions), but it
    # must still be a well-formed interval around the reported mean.
    for row in result.mrr_rows:
        assert row["ci_low"] <= row["mrr"] <= row["ci_high"]
        assert row["ci_low"] >= 0.0 and row["ci_high"] <= 1.0

    # No `multi_stage_rankings` on any fixture row -- panel B's split stays
    # off, mrr_rows carries single-stage entries only (DEVIATIONS.md #184).
    assert result.multi_stage_available is False
    assert result.multi_stage_fired_rate == pytest.approx(0.0)
    assert {row["stage"] for row in result.mrr_rows} == {SINGLE_STAGE}


def test_run_ablation_splits_mrr_rows_by_stage_when_multi_stage_is_computed() -> None:
    rankings = [
        QuestionArmRankings(
            question_id="q1",
            gold_chunk_ids=frozenset({"gold1"}),
            rankings={
                "sapbert_bm25": ["distractor", "gold1"],
                "medcpt_bm25": ["distractor", "gold1"],
                "sapbert_medcpt_bm25": ["distractor", "gold1"],
                REFERENCE_ARM: ["distractor", "gold1"],
            },
            # multi-stage augmentation fired and moved the gold chunk to rank 1
            multi_stage_rankings={
                "sapbert_bm25": ["gold1", "distractor"],
                "medcpt_bm25": ["gold1", "distractor"],
                "sapbert_medcpt_bm25": ["gold1", "distractor"],
                REFERENCE_ARM: ["gold1", "distractor"],
            },
            multi_stage_fired=True,
        ),
        QuestionArmRankings(
            question_id="q2",
            gold_chunk_ids=frozenset({"gold2"}),
            rankings={
                "sapbert_bm25": ["distractor", "gold2"],
                "medcpt_bm25": ["distractor", "gold2"],
                "sapbert_medcpt_bm25": ["distractor", "gold2"],
                REFERENCE_ARM: ["distractor", "gold2"],
            },
            # augmentation didn't fire for this question -- multi-stage ==
            # single-stage, same convention as `rank_question`'s fallback.
            multi_stage_rankings={
                "sapbert_bm25": ["distractor", "gold2"],
                "medcpt_bm25": ["distractor", "gold2"],
                "sapbert_medcpt_bm25": ["distractor", "gold2"],
                REFERENCE_ARM: ["distractor", "gold2"],
            },
            multi_stage_fired=False,
        ),
    ]

    result = run_ablation(rankings)

    assert result.multi_stage_available is True
    assert result.multi_stage_fired_rate == pytest.approx(0.5)  # fired for q1, not q2
    assert {row["stage"] for row in result.mrr_rows} == {SINGLE_STAGE, MULTI_STAGE}
    assert {row["arm"] for row in result.mrr_rows} == set(ALL_ARMS)

    single = next(
        row["mrr"]
        for row in result.mrr_rows
        if row["arm"] == "sapbert_bm25" and row["stage"] == SINGLE_STAGE
    )
    multi = next(
        row["mrr"]
        for row in result.mrr_rows
        if row["arm"] == "sapbert_bm25" and row["stage"] == MULTI_STAGE
    )
    # single-stage: gold at rank 2 for both questions -> mrr 0.5 each -> mean 0.5
    assert single == pytest.approx(0.5)
    # multi-stage: gold at rank 1 for q1 (augmentation fired), rank 2 for q2 -> mean 0.75
    assert multi == pytest.approx(0.75)

    # recall_rows (panel A) stays single-stage-only, untouched by the split.
    assert all("stage" not in row for row in result.recall_rows)


def test_bootstrap_ci_is_deterministic_with_a_fixed_seed() -> None:
    rng1 = np.random.default_rng(1234)
    rng2 = np.random.default_rng(1234)
    scores = [0.0, 0.5, 1.0, 0.0, 1.0, 0.5, 0.25]
    assert _bootstrap_ci(scores, rng=rng1) == _bootstrap_ci(scores, rng=rng2)


def test_bootstrap_ci_collapses_when_every_score_is_equal() -> None:
    rng = np.random.default_rng(1234)
    lo, hi = _bootstrap_ci([0.5, 0.5, 0.5, 0.5], rng=rng)
    assert lo == pytest.approx(0.5)
    assert hi == pytest.approx(0.5)


def test_bootstrap_ci_empty_scores_is_zero() -> None:
    rng = np.random.default_rng(1234)
    assert _bootstrap_ci([], rng=rng) == (0.0, 0.0)


def test_bootstrap_ci_brackets_the_sample_mean_on_a_realistic_n() -> None:
    rng = np.random.default_rng(1234)
    scores = [1.0, 0.5, 0.33, 0.0, 0.0, 1.0, 0.25, 0.5, 0.0, 1.0] * 7  # n=70, MRR-like
    lo, hi = _bootstrap_ci(scores, rng=rng)
    assert lo <= sum(scores) / len(scores) <= hi
    assert lo < hi  # non-degenerate with real spread in the data


def test_arms_and_reference_are_disjoint() -> None:
    # Guards against ALL_ARMS accidentally dropping or duplicating the
    # reference arm if either constant is edited later.
    assert REFERENCE_ARM not in ARMS
    assert set(ALL_ARMS) == set(ARMS) | {REFERENCE_ARM}
