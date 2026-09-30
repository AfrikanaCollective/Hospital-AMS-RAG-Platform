"""Brute-force score blending for Level 3's BM25/SapBERT weighted-rank
fusion sweep (UNIFIED-ABLATION-PROPOSAL.md §3.3; PRD-112).

**Restructured 2026-09-23** (operator request, DEVIATIONS.md #201): MedCPT
and RRF fusion (proposal §11, Option B) are dropped entirely — Level 3 is
now a single continuous weighted-rank sweep between BM25 and SapBERT only.
`combine_dense_scores` (the SapBERT+MedCPT combination) and
`rrf_combine_scores` (the RRF-fusion arms) are removed, not merely unused.

Reuses `retrieval_tuning.offline_fusion.weighted_rank`/`ScoredChunk`/
`min_max_normalize` **unchanged** — those are already pure,
Qdrant-independent functions once a candidate's `bm25_score`/`dense_score`
are in hand. `offline_fusion.fetch_candidate_scores` (the *other* half of
that module) is Qdrant-ANN-specific — two server-side searches against the
one production embedding — and can't be reused for SapBERT, which
`model_ablation.ablation` deliberately never writes to Qdrant (it ranks
brute-force over the whole in-memory corpus instead, `_bm25_rank`/
`_cosine_rank`). This module supplies that brute-force-corpus equivalent:
raw BM25 scores via the same Qdrant sparse-vector call `_bm25_rank` already
makes (just returning scores instead of only a sorted ranking), and raw
cosine scores mirroring `_cosine_rank`'s own query-vector normalization
exactly (same reason — that function only returns the sorted ranking, and
a weighted blend needs the underlying scores).
"""

from __future__ import annotations

import numpy as np

from app.eval.model_ablation.ablation import CORPUS_FILTER
from app.eval.retrieval_tuning.offline_fusion import ScoredChunk, min_max_normalize, weighted_rank
from app.retrieval.sparse import query_sparse_vector
from app.retrieval.vectorstore import QdrantVectorStore


def bm25_raw_scores(
    store: QdrantVectorStore, question_text: str, chunk_ids: list[str]
) -> dict[str, float]:
    """Raw (un-normalized) BM25 score per chunk that scored at all, full
    corpus depth. A chunk sharing no term with the query is simply absent
    from the result — `min_max_normalize`/`ScoredChunk` construction fills
    the gap with 0.0 the same way `fetch_candidate_scores` already does for
    its own two channels, not something this function needs to do itself."""
    sparse = query_sparse_vector(question_text)
    hits = store.single_vector_search(
        using="sparse", query=sparse, limit=len(chunk_ids), flt=CORPUS_FILTER
    )
    corpus_ids = set(chunk_ids)
    return {h["chunk_id"]: h["score"] for h in hits if h["chunk_id"] in corpus_ids}


def cosine_raw_scores(
    query_vec: list[float], chunk_matrix: np.ndarray, chunk_ids: list[str]
) -> dict[str, float]:
    """Raw cosine similarity per chunk — mirrors
    `model_ablation.ablation._cosine_rank`'s own query-vector normalization
    exactly, but returns the full `{chunk_id: score}` dict instead of only
    the sorted ranking."""
    q = np.asarray(query_vec, dtype=np.float64)
    q_norm = np.linalg.norm(q)
    if q_norm > 0:
        q = q / q_norm
    sims = chunk_matrix @ q
    return dict(zip(chunk_ids, sims.tolist(), strict=True))


def blend_bm25_dense(
    bm25_scores: dict[str, float],
    dense_scores: dict[str, float],
    *,
    bm25_weight: float,
) -> list[str]:
    """BM25 vs. SapBERT, weighted-rank-fused — both independently
    min-max-normalized first (matching `fetch_candidate_scores`'s own
    convention exactly), then handed to `weighted_rank`, reused unchanged.
    `bm25_weight` (renamed from `alpha`, DEVIATIONS.md #201 — `w_BM25` in
    the operator's own notation) is `weighted_rank`'s own `alpha` parameter
    under a clearer name; `weighted_rank` itself is untouched (shared with
    `retrieval_tuning`, which still calls it with its own `alpha=`
    keyword) — only this module's own call site renames the concept."""
    bm25_norm = min_max_normalize(bm25_scores)
    dense_norm = min_max_normalize(dense_scores)
    chunk_ids = set(bm25_norm) | set(dense_norm)
    candidates = [
        ScoredChunk(
            chunk_id=cid,
            bm25_score=bm25_norm.get(cid, 0.0),
            dense_score=dense_norm.get(cid, 0.0),
        )
        for cid in chunk_ids
    ]
    return weighted_rank(candidates, alpha=bm25_weight)


def _ranks(scores: dict[str, float]) -> dict[str, int]:
    """1-based rank per chunk, highest score first; ties broken by
    `chunk_id` so the order never depends on dict iteration order."""
    ordered = sorted(scores, key=lambda cid: (-scores[cid], cid))
    return {cid: i for i, cid in enumerate(ordered, start=1)}


def blend_bm25_dense_rrf(
    bm25_scores: dict[str, float],
    dense_scores: dict[str, float],
    *,
    bm25_weight: float,
    rrf_k: int,
) -> list[str]:
    """Weighted reciprocal-rank fusion (DEVIATIONS.md #227): each chunk scores
    `w / (rrf_k + rank_bm25) + (1 - w) / (rrf_k + rank_dense)`.

    Rank-based, so neither channel's score *distribution* matters. Under
    min-max fusion (`blend_bm25_dense`), SapBERT's cosines cluster near the
    top of the corpus and normalize to ~1.0, so BM25's wider spread decided
    the order at any weight ≥ 0.2 (#226). A chunk BM25 didn't score (no
    shared term) gets no BM25 term, the same as its 0.0 under min-max, so
    `bm25_weight` 0.0 and 1.0 rank exactly as `blend_bm25_dense` does; only
    the interior weights differ."""
    bm25_rank = _ranks(bm25_scores)
    dense_rank = _ranks(dense_scores)

    def fused(cid: str) -> float:
        score = 0.0
        if cid in bm25_rank:
            score += bm25_weight / (rrf_k + bm25_rank[cid])
        if cid in dense_rank:
            score += (1 - bm25_weight) / (rrf_k + dense_rank[cid])
        return score

    chunk_ids = set(bm25_rank) | set(dense_rank)
    return sorted(chunk_ids, key=lambda cid: (-fused(cid), cid))
