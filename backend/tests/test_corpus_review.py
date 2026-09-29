"""Corpus review gate and version-status propagation (ARCH-044, ARCH §5.1;
LAYOUT-INGESTION-PROPOSAL.md §8; DEVIATIONS.md #215). In-memory Qdrant, fake
session — no server, no DB."""

from __future__ import annotations

import uuid

import pytest

from app.ingestion.embed import embed_texts
from app.ingestion.review import (
    NOT_RETRIEVABLE,
    ChunkReviewError,
    review_chunk,
)
from app.retrieval.sparse import doc_sparse_vector, query_sparse_vector
from app.retrieval.vectorstore import QdrantVectorStore

DIM = len(embed_texts(["probe"])[0])


@pytest.fixture
def store() -> QdrantVectorStore:
    s = QdrantVectorStore(url=":memory:", api_key="", collection="review_gate_test")
    s.ensure_collection(dense_dim=DIM)
    return s


def _upsert(
    store: QdrantVectorStore, point_id: str, text: str, *, version: str, review: str | None
) -> None:
    store.upsert_chunks(
        [
            {
                "id": point_id,
                "dense": embed_texts([text])[0],
                "sparse": doc_sparse_vector(text),
                "payload": {
                    "chunk_id": point_id,
                    "text": text,
                    "document_version_id": version,
                    "status": "active",
                    "review_status": review,
                },
            }
        ]
    )


def _search(store: QdrantVectorStore, q: str, flt: dict) -> set[str]:
    hits = store.hybrid_search(
        dense=embed_texts([q], is_query=True)[0],
        sparse=query_sparse_vector(q),
        prefetch_limit=10,
        limit=10,
        flt=flt,
    )
    return {h["chunk_id"] for h in hits}


RETRIEVAL_FILTER = {"status": "active", "exclude_review_status": list(NOT_RETRIEVABLE)}


def test_held_and_rejected_chunks_are_not_retrievable(store: QdrantVectorStore) -> None:
    ids = {r: str(uuid.uuid4()) for r in ("none", "pending", "confirmed", "rejected")}
    for review, pid in ids.items():
        _upsert(
            store,
            pid,
            f"gentamicin dose table {review}",
            version="v1",
            review=None if review == "none" else review,
        )
    found = _search(store, "gentamicin dose table", RETRIEVAL_FILTER)
    assert found == {ids["none"], ids["confirmed"]}


def test_superseded_version_drops_out_of_retrieval_and_scroll(store: QdrantVectorStore) -> None:
    old, new = str(uuid.uuid4()), str(uuid.uuid4())
    _upsert(store, old, "neonatal sepsis algorithm", version="v-old", review=None)
    _upsert(store, new, "neonatal sepsis algorithm", version="v-new", review=None)
    store.set_payload_by_version("v-old", {"status": "superseded"})
    assert _search(store, "neonatal sepsis algorithm", RETRIEVAL_FILTER) == {new}
    assert {p["chunk_id"] for p in store.scroll_all(flt=RETRIEVAL_FILTER)} == {new}


class _Chunk:
    def __init__(self, meta: dict) -> None:
        self.id = uuid.uuid4()
        self.document_version_id = uuid.uuid4()
        self.meta = meta


class _Session:
    def __init__(self, chunk: _Chunk) -> None:
        self.chunk = chunk
        self.added: list = []

    def get(self, _model, _id):
        return self.chunk

    def flush(self) -> None:
        pass

    def add(self, obj) -> None:
        self.added.append(obj)


class _Store:
    def __init__(self) -> None:
        self.calls: list = []

    def set_payload(self, ids, payload) -> None:
        self.calls.append((ids, payload))


def test_confirming_a_held_chunk_updates_db_qdrant_and_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[dict] = []
    monkeypatch.setattr("app.ingestion.review.write_event", lambda session, **kw: events.append(kw))
    chunk = _Chunk({"review_status": "pending", "ocr": {"has_digits": True}})
    store = _Store()
    review_chunk(
        _Session(chunk),
        store,
        chunk.id,
        decision="confirmed",
        note="checked vs crop",
        actor_id=uuid.uuid4(),
        actor_role="admin",
    )
    assert chunk.meta["review_status"] == "confirmed"
    assert chunk.meta["review"]["reasons"] == ["ocr_numeric"]
    assert store.calls == [([str(chunk.id)], {"review_status": "confirmed"})]
    assert events and events[0]["outcome"] == "chunk_review_confirmed"


def test_review_rejects_bad_decisions_and_chunks_not_under_review() -> None:
    chunk = _Chunk({})
    with pytest.raises(ChunkReviewError):
        review_chunk(
            _Session(chunk),
            _Store(),
            chunk.id,
            decision="approve",
            note=None,
            actor_id=None,
            actor_role=None,
        )
    with pytest.raises(ChunkReviewError, match="not under review"):
        review_chunk(
            _Session(chunk),
            _Store(),
            chunk.id,
            decision="confirmed",
            note=None,
            actor_id=None,
            actor_role=None,
        )


def test_review_queue_lists_only_active_versions() -> None:
    """A re-ingest supersedes the prior version, but its held chunks keep
    `review_status = pending`; the queue must filter them out
    (DEVIATIONS.md #225)."""
    from sqlalchemy.dialects import postgresql

    from app.ingestion.review import pending_query

    sql = str(
        pending_query().compile(
            dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}
        )
    )
    assert "JOIN corpus.document_version" in sql
    assert "corpus.document_version.status = 'active'" in sql
    assert "review_status" in sql and "'pending'" in sql
