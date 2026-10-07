"""Hybrid retrieval pipeline (ARCH §7).

1. query construction (curated abbreviation expansion only — never invent
   clinical content),
2. dense + sparse search in Qdrant with access filter (CANDIDATE_K each),
3. RRF fusion (server-side, Qdrant's Query API — FUSED_K),
4. cross-encoder rerank -> TOP_K,
5. confidence assessment (RETRIEVAL_MIN_SCORE / MIN_SUPPORTING_CHUNKS),
6. conflict detection (multi-version same-topic / lexical contradiction),
7. context expansion: NOT done here — `expand_context` (parent_chunk_id ->
   surrounding text) is a Phase-3 agent *tool*, invoked on demand by the
   synthesis agent, not a step this pipeline runs unconditionally,
8. return ranked items + confidence/conflict verdict; writes a minimal
   `retrieval` audit event when a DB session is supplied (DEVIATIONS.md #50 —
   the caller decides whether a session is available; offline unit tests pass
   none and skip the write).
"""

from __future__ import annotations

import hashlib
import re
import uuid
from typing import TYPE_CHECKING

from app.agents.state import RetrievalItem
from app.audit.log import write_event
from app.config import get_settings
from app.ingestion.embed import embed_texts
from app.ingestion.review import NOT_RETRIEVABLE
from app.retrieval.confidence import assess
from app.retrieval.conflict import detect_conflicts
from app.retrieval.priority import load_priorities
from app.retrieval.rerank import rerank
from app.retrieval.sparse import query_sparse_vector
from app.retrieval.vectorstore import QdrantVectorStore, VectorStore

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

# Curated only — never invent clinical content (ARCH §7 step 1). Chosen for
# the bundled neonatal dev corpus (WHO newborn health / SBI in young infants,
# Kenya MOH newborn care protocols); extend deliberately, one confirmed term
# at a time (DEVIATIONS.md #55).
_ABBREVIATIONS: dict[str, str] = {
    "sbi": "serious bacterial infection",
    "lbw": "low birth weight",
    "kmc": "kangaroo mother care",
    "nbu": "newborn unit",
    "bp": "blood pressure",
    "hr": "heart rate",
    "rr": "respiratory rate",
    "spo2": "oxygen saturation",
    "iv": "intravenous",
    "im": "intramuscular",
}
_ABBREVIATION_PATTERNS = {
    abbr: re.compile(rf"\b{re.escape(abbr)}\b", re.IGNORECASE) for abbr in _ABBREVIATIONS
}


def _expand_abbreviations(query: str) -> str:
    """Append `(expansion)` after a recognised abbreviation; the original
    term is preserved (exact-term/BM25 matches still work) and the expansion
    adds vocabulary useful to the dense retriever. Never invents content
    beyond the curated map."""
    expanded = query
    for abbr, full in _ABBREVIATIONS.items():
        pattern = _ABBREVIATION_PATTERNS[abbr]

        def _add_expansion(m: re.Match[str], full: str = full) -> str:
            return f"{m.group(0)} ({full})"

        if pattern.search(expanded):
            expanded = pattern.sub(_add_expansion, expanded)
    return expanded


def _default_vectorstore() -> VectorStore:
    s = get_settings()
    return QdrantVectorStore(
        url=s.qdrant_url, api_key=s.qdrant_api_key, collection=s.qdrant_guideline_collection
    )


def _to_retrieval_item(candidate: dict, score: float) -> RetrievalItem:
    return RetrievalItem(
        chunk_id=candidate["chunk_id"],
        score=score,
        section_path=candidate.get("section_path"),
        section_number=candidate.get("section_number"),
        page_start=candidate["page_start"],
        page_end=candidate["page_end"],
        char_start=candidate["char_start"],
        char_end=candidate["char_end"],
        document_id=candidate["document_id"],
        document_title=candidate["document_title"],
        document_version_id=candidate["document_version_id"],
        version_label=candidate["version_label"],
        effective_date=candidate.get("effective_date"),
        version_status=candidate.get("status", "active"),
        chunk_type=candidate.get("chunk_type", "prose"),
        text=candidate["text"],
        heading=candidate.get("heading"),
        # The top-level `review_status` payload is the source of truth
        # (retrieval filters on it); the copy inside `meta` was not updated
        # by review decisions, so it is overwritten here for the verifier
        # (DEVIATIONS.md #267).
        meta={
            **(candidate.get("meta") or {}),
            "review_status": candidate.get("review_status"),
        },
    )


def _rerank_passage(candidate: dict) -> str:
    """The text the cross-encoder scores: the chunk's heading path, then its
    text -- the same leading text the dense and sparse indexes embed
    (`app.ingestion.chunking`'s `embedding_text`), so all three stages see a
    chunk the same way. Without it, a heading-dependent chunk such as a
    flowchart ("[n1] Has ONE of the following...") never says what it is
    about and is reranked below prose (DEVIATIONS.md #250)."""
    path = candidate.get("section_path")
    return f"{path}\n\n{candidate['text']}" if path else candidate["text"]


def _per_guideline_groups(
    store: VectorStore,
    dense: list[float],
    sparse: dict,
    flt: dict,
    *,
    priorities: list[tuple[int, str]],
    candidates: int,
    prefetch: int,
) -> list[list[dict]]:
    """One hybrid search per listed guideline, in priority order, plus one for
    every guideline the manifest doesn't list (last). Searching each
    guideline separately means its best chunks are considered even when other
    guidelines would fill a single global candidate list."""
    titles = [t for _, t in priorities]
    filters = [{**flt, "document_titles": [t]} for t in titles]
    filters.append({**flt, "exclude_document_titles": titles})
    return [
        store.hybrid_search(
            dense=dense, sparse=sparse, prefetch_limit=prefetch, limit=candidates, flt=f
        )
        for f in filters
    ]


def select_per_guideline(
    groups: list[list[tuple[dict, float]]],
    *,
    cap: int,
    min_score: float,
    max_parts_per_table: int | None = None,
) -> list[tuple[dict, float]]:
    """Per-guideline selection (DEVIATIONS.md #252): within each group (already
    in priority order), the best `cap` chunks per document scoring at least
    `min_score`, best first. A guideline with nothing relevant contributes
    nothing -- its slots stay empty rather than being filled with weak chunks
    or backfilled from another guideline. At most `max_parts_per_table` parts
    of one split table (same `meta.split_group_id`) are kept, so a table split
    into near-identical rows can't take every slot (DEVIATIONS.md #256)."""
    out: list[tuple[dict, float]] = []
    for group in groups:
        taken: dict[str, int] = {}
        parts: dict[str, int] = {}
        for cand, score in sorted(group, key=lambda cs: cs[1], reverse=True):
            doc = str(cand.get("document_id"))
            split = (cand.get("meta") or {}).get("split_group_id")
            if score < min_score or taken.get(doc, 0) >= cap:
                continue
            if split and max_parts_per_table is not None:
                if parts.get(split, 0) >= max_parts_per_table:
                    continue
                parts[split] = parts.get(split, 0) + 1
            taken[doc] = taken.get(doc, 0) + 1
            out.append((cand, score))
    return out


def retrieve(
    query: str,
    *,
    access_filter: dict | None = None,
    vectorstore: VectorStore | None = None,
    session: Session | None = None,
    conversation_id: uuid.UUID | None = None,
    patient_id: uuid.UUID | None = None,
    actor_id: uuid.UUID | None = None,
    actor_role: str | None = None,
    purpose: str | None = None,
    per_guideline_cap: int | None = None,
) -> tuple[list[RetrievalItem], dict]:
    """Run the full hybrid retrieval pipeline for one query.

    `vectorstore` defaults to a real `QdrantVectorStore` built from config;
    pass one explicitly for tests (e.g. `QdrantVectorStore(url=":memory:", ...)`
    ) or to reuse a warm client. `session` is optional — when given, one
    `retrieval` audit event is written (DEVIATIONS.md #50); offline unit
    tests pass none.

    `RETRIEVAL_MODE=per_guideline` (default) returns up to
    `per_guideline_cap` (default `RETRIEVAL_PER_GUIDELINE_CAP`) chunks per
    guideline, grouped in manifest `retrieval_priority` order; `fused`
    returns one global `TOP_K`. Confidence is always assessed on relevance
    (the best `TOP_K` rerank scores), never on priority order.
    """
    settings = get_settings()
    store = vectorstore or _default_vectorstore()

    expanded_query = _expand_abbreviations(query)
    dense = embed_texts([expanded_query], is_query=True)[0]
    sparse = query_sparse_vector(expanded_query)

    flt = dict(access_filter or {})
    flt.setdefault("status", "active")
    flt.setdefault("exclude_review_status", list(NOT_RETRIEVABLE))

    priorities = (
        load_priorities(settings.sample_guidelines_dir)
        if settings.retrieval_mode == "per_guideline"
        else []
    )
    mode = "per_guideline" if priorities else "fused"
    if priorities:
        groups = _per_guideline_groups(
            store,
            dense,
            sparse,
            flt,
            priorities=priorities,
            candidates=settings.retrieval_per_guideline_candidates,
            prefetch=settings.candidate_k,
        )
    else:
        groups = [
            store.hybrid_search(
                dense=dense,
                sparse=sparse,
                prefetch_limit=settings.candidate_k,
                limit=settings.fused_k,
                flt=flt,
            )
        ]

    # One rerank call over every candidate, then split back into groups.
    flat = [c for g in groups for c in g]
    flat_scores = rerank(expanded_query, [_rerank_passage(c) for c in flat])
    scored: list[list[tuple[dict, float]]] = []
    i = 0
    for g in groups:
        scored.append(list(zip(g, flat_scores[i : i + len(g)], strict=True)))
        i += len(g)
    by_relevance = sorted((cs for g in scored for cs in g), key=lambda cs: cs[1], reverse=True)

    if mode == "per_guideline":
        top = select_per_guideline(
            scored,
            cap=per_guideline_cap or settings.retrieval_per_guideline_cap,
            min_score=settings.retrieval_min_score,
            max_parts_per_table=settings.retrieval_max_parts_per_table,
        )
    else:
        top = by_relevance[: settings.top_k]

    items = [_to_retrieval_item(c, s) for c, s in top]
    confidence = assess([s for _, s in by_relevance[: settings.top_k]])
    conflicts = detect_conflicts(items)

    snapshot = {
        "query": query,
        "expanded_query": expanded_query,
        "mode": mode,
        "items": [dict(item) for item in items],
        "confidence": {
            "top_score": confidence.top_score,
            "supporting_count": confidence.supporting_count,
            "low_confidence": confidence.low_confidence,
            "essentially_empty": confidence.essentially_empty,
        },
        "conflicts": conflicts,
    }

    if session is not None:
        write_event(
            session,
            action="retrieval",
            actor_id=actor_id,
            actor_role=actor_role,
            purpose=purpose,
            conversation_id=conversation_id,
            patient_id=patient_id,
            query_hash=hashlib.sha256(query.encode("utf-8")).hexdigest(),
            retrieved=[
                {
                    "chunk_id": item["chunk_id"],
                    "score": item["score"],
                    "fusion": "rrf" if mode == "fused" else "rrf_per_guideline",
                    "rerank": settings.reranker_backend,
                }
                for item in items
            ],
        )

    return items, snapshot
