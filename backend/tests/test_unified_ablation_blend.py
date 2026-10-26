"""Brute-force score blending for Level 3's BM25/SapBERT weighted-rank
fusion sweep (PRD-112). Restructured 2026-09-23 (DEVIATIONS.md #201):
MedCPT/RRF support (`combine_dense_scores`/`rrf_combine_scores`) removed."""

from __future__ import annotations

import numpy as np
import pytest

from app.config import get_settings
from app.eval.unified_ablation.blend import (
    OkapiBM25,
    blend_bm25_dense,
    blend_bm25_dense_rrf,
    bm25_raw_scores,
    cosine_raw_scores,
    fetch_okapi_bm25,
    lexical_query_vector,
)
from app.retrieval.sparse import doc_sparse_vector, query_sparse_vector
from app.retrieval.vectorstore import QdrantVectorStore
from tests.test_hybrid_retrieve import _seed_chunk

_DENSE_DIM = 384


@pytest.fixture(autouse=True)
def _stub_backends(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EMBEDDING_BACKEND", "stub")
    monkeypatch.setenv("RERANKER_BACKEND", "stub")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def store() -> QdrantVectorStore:
    s = QdrantVectorStore(url=":memory:", api_key="", collection="unified_ablation_blend_test")
    s.ensure_collection(dense_dim=_DENSE_DIM)
    return s


def test_bm25_raw_scores_favors_the_lexically_matching_chunk(store: QdrantVectorStore) -> None:
    _seed_chunk(store, 1, chunk_id="c1", text="Blood cultures are recommended before antibiotics.")
    _seed_chunk(store, 2, chunk_id="c2", text="zzz completely unrelated zzz")

    scores = bm25_raw_scores(store, "blood cultures antibiotics", ["c1", "c2"])
    assert "c1" in scores
    assert scores.get("c1", 0.0) > scores.get("c2", 0.0)


def test_cosine_raw_scores_ranks_by_similarity_and_returns_every_chunk() -> None:
    chunk_ids = ["near", "far", "mid"]
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])
    scores = cosine_raw_scores([1.0, 0.0], matrix, chunk_ids)
    assert set(scores) == {"near", "far", "mid"}
    assert scores["near"] > scores["mid"] > scores["far"]


def test_cosine_raw_scores_handles_zero_query_vector_without_crashing() -> None:
    scores = cosine_raw_scores([0.0, 0.0], np.array([[1.0, 0.0], [0.0, 1.0]]), ["a", "b"])
    assert scores["a"] == pytest.approx(0.0)
    assert scores["b"] == pytest.approx(0.0)


def test_blend_bm25_dense_weight_one_is_pure_bm25_order() -> None:
    bm25 = {"a": 10.0, "b": 1.0}
    dense = {"a": 0.0, "b": 10.0}  # dense strongly prefers b
    ranked = blend_bm25_dense(bm25, dense, bm25_weight=1.0)
    assert ranked[0] == "a"  # bm25_weight=1.0 ignores dense entirely


def test_blend_bm25_dense_weight_zero_is_pure_dense_order() -> None:
    bm25 = {"a": 10.0, "b": 1.0}
    dense = {"a": 0.0, "b": 10.0}
    ranked = blend_bm25_dense(bm25, dense, bm25_weight=0.0)
    assert ranked[0] == "b"  # bm25_weight=0.0 ignores bm25 entirely


def test_blend_bm25_dense_includes_a_chunk_present_in_only_one_channel() -> None:
    bm25 = {"a": 5.0}
    dense = {"b": 5.0}
    ranked = blend_bm25_dense(bm25, dense, bm25_weight=0.5)
    assert set(ranked) == {"a", "b"}


def test_bm25_raw_scores_ignores_superseded_and_held_chunks(store: QdrantVectorStore) -> None:
    """Superseded versions stay in the collection. Unfiltered, they filled
    the `limit=len(corpus)` BM25 window and were ranked into the top k,
    where they can never be gold (DEVIATIONS.md #226)."""
    text = "Blood cultures are recommended before antibiotics."
    _seed_chunk(store, 1, chunk_id="old", text=text + " blood cultures", status="superseded")
    _seed_chunk(store, 2, chunk_id="held", text=text + " blood", review_status="pending")
    _seed_chunk(store, 3, chunk_id="active", text=text)

    scores = bm25_raw_scores(store, "blood cultures antibiotics", ["active"])
    assert set(scores) == {"active"}


# --- weighted reciprocal-rank fusion (DEVIATIONS.md #227) ---

_BM25 = {"a": 9.0, "b": 5.0, "c": 1.0}  # d shares no term with the query
_DENSE = {"a": 0.80, "b": 0.81, "c": 0.95, "d": 0.90}


@pytest.mark.parametrize("weight", [0.0, 1.0])
def test_rrf_endpoints_rank_exactly_like_minmax(weight: float) -> None:
    assert blend_bm25_dense_rrf(_BM25, _DENSE, bm25_weight=weight, rrf_k=60) == blend_bm25_dense(
        _BM25, _DENSE, bm25_weight=weight
    )


def test_rrf_ignores_score_spread_that_lets_bm25_dominate_minmax() -> None:
    """Dense scores bunched near the top (as SapBERT cosines are) all
    min-max-normalize close to 1, so at w=0.2 min-max follows BM25. RRF sees
    only ranks: dense ranks c first, and at w=0.2 its 0.8 share keeps c on top."""
    bm25 = {"a": 100.0, "b": 50.0, "c": 1.0, "z": 0.0}
    dense = {"a": 0.97, "b": 0.975, "c": 0.98, "z": 0.10}
    assert blend_bm25_dense(bm25, dense, bm25_weight=0.2)[0] == "a"
    assert blend_bm25_dense_rrf(bm25, dense, bm25_weight=0.2, rrf_k=60)[0] == "c"


def test_rrf_chunk_without_bm25_score_gets_only_its_dense_term() -> None:
    ranked = blend_bm25_dense_rrf(_BM25, _DENSE, bm25_weight=0.5, rrf_k=60)
    assert set(ranked) == {"a", "b", "c", "d"}
    # d: dense rank 2 only -> 0.5/62; c: bm25 rank 3 + dense rank 1 -> beats d
    assert ranked.index("c") < ranked.index("d")


def test_rrf_ties_are_broken_by_chunk_id() -> None:
    ranked = blend_bm25_dense_rrf(
        {"y": 1.0, "x": 1.0}, {"y": 0.5, "x": 0.5}, bm25_weight=0.5, rrf_k=60
    )
    assert ranked == ["x", "y"]


def test_invalid_fusion_setting_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.eval.ablation_config import fusion

    monkeypatch.setenv("ABLATION_FUSION", "zscore")
    get_settings.cache_clear()
    with pytest.raises(ValueError, match="ABLATION_FUSION"):
        fusion()


# ── Okapi BM25 (DEVIATIONS.md #311) ──


def _counts(text: str) -> dict[int, float]:
    v = doc_sparse_vector(text)
    return dict(zip(v["indices"], v["values"], strict=True))


_SHORT = "Sepsis antibiotics duration."
_LONG = "sepsis " * 20 + " ".join(f"filler{i}" for i in range(200))


def test_okapi_bm25_ranks_a_short_specific_chunk_above_a_long_repetitive_one() -> None:
    """TF x IDF (the collection's sparse search) lets 20 repeats of one word
    outscore a chunk with both query terms; Okapi BM25 saturates term
    frequency and normalises by length, so the specific chunk wins."""
    okapi = OkapiBM25(
        {"short": _counts(_SHORT), "long": _counts(_LONG), "none": _counts("unrelated text")}
    )
    scores = okapi.scores("sepsis antibiotics")
    assert scores["short"] > scores["long"] > 0
    assert "none" not in scores  # no shared term: absent, as in bm25_raw_scores


def test_bm25_raw_scores_is_the_tf_idf_scorer_okapi_corrects(store: QdrantVectorStore) -> None:
    _seed_chunk(store, 1, chunk_id="short", text=_SHORT)
    _seed_chunk(store, 2, chunk_id="long", text=_LONG)
    _seed_chunk(store, 3, chunk_id="none", text="unrelated text")
    ids = ["short", "long", "none"]
    tfidf = bm25_raw_scores(store, "sepsis antibiotics", ids)
    assert tfidf["long"] > tfidf["short"]
    okapi = fetch_okapi_bm25(store, ids).scores("sepsis antibiotics")
    assert okapi["short"] > okapi["long"]


def test_fetch_okapi_bm25_reads_the_indexed_counts_of_the_given_chunks_only(
    store: QdrantVectorStore,
) -> None:
    _seed_chunk(store, 1, chunk_id="c1", text="sepsis sepsis antibiotics")
    _seed_chunk(store, 2, chunk_id="c2", text="sepsis")
    assert store.sparse_vectors()["c1"] == _counts("sepsis sepsis antibiotics")
    okapi = fetch_okapi_bm25(store, ["c1"])
    assert set(okapi.scores("sepsis")) == {"c1"}


# ── weighted area question (DEVIATIONS.md #312) ──


def test_lexical_query_vector_weights_only_the_area_question_terms() -> None:
    text = "What about sepsis antibiotics?\n\nThe patient had grunting."
    q = lexical_query_vector(text, emphasis="What about sepsis antibiotics?", weight=3.0)
    weights = dict(zip(q["indices"], q["values"], strict=True))
    by_term = {t: weights[query_sparse_vector(t)["indices"][0]] for t in ("sepsis", "grunting")}
    assert by_term == {"sepsis": 3.0, "grunting": 1.0}
    assert lexical_query_vector(text, emphasis="sepsis", weight=1.0) == query_sparse_vector(text)
    assert lexical_query_vector(text) == query_sparse_vector(text)


def test_weighting_the_area_question_moves_okapi_towards_its_terms() -> None:
    okapi = OkapiBM25(
        {"area": _counts("sepsis antibiotics duration"), "patient": _counts("grunting cyanosis")}
    )
    text = "sepsis antibiotics grunting cyanosis"
    flat = okapi.scores(text)
    weighted = okapi.scores(
        text, query_vector=lexical_query_vector(text, emphasis="sepsis antibiotics", weight=3.0)
    )
    assert weighted["area"] / weighted["patient"] == pytest.approx(
        3 * flat["area"] / flat["patient"]
    )


def test_bm25_raw_scores_uses_the_weighted_query_vector(store: QdrantVectorStore) -> None:
    _seed_chunk(store, 1, chunk_id="area", text="sepsis antibiotics duration")
    _seed_chunk(store, 2, chunk_id="patient", text="grunting cyanosis")
    text = "sepsis antibiotics grunting cyanosis"
    flat = bm25_raw_scores(store, text, ["area", "patient"])
    weighted = bm25_raw_scores(
        store,
        text,
        ["area", "patient"],
        query_vector=lexical_query_vector(text, emphasis="sepsis antibiotics", weight=3.0),
    )
    assert weighted["area"] == pytest.approx(3 * flat["area"])
    assert weighted["patient"] == pytest.approx(flat["patient"])
