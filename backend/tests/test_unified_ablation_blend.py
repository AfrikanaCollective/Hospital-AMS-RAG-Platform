"""Brute-force score blending for Level 3's BM25/SapBERT weighted-rank
fusion sweep (PRD-112). Restructured 2026-09-23 (DEVIATIONS.md #201):
MedCPT/RRF support (`combine_dense_scores`/`rrf_combine_scores`) removed."""

from __future__ import annotations

import numpy as np
import pytest

from app.config import get_settings
from app.eval.unified_ablation.blend import (
    blend_bm25_dense,
    blend_bm25_dense_rrf,
    bm25_raw_scores,
    cosine_raw_scores,
)
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
