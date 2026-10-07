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
    exclude_sections,
    identical_confirmed_source,
    load_exclude_sections,
    review_chunk,
    section_excluded,
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
    ids = {r: str(uuid.uuid4()) for r in ("none", "pending", "confirmed", "rejected", "excluded")}
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
    ((ids, payload),) = store.calls
    assert ids == [str(chunk.id)] and payload["review_status"] == "confirmed"
    assert payload["meta"]["review_status"] == "confirmed"  # both copies (#267)
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


# --- excluded sections (DEVIATIONS.md #251) ---

_PREFIXES = ["", "Contents", "Rationale and impact"]


@pytest.mark.parametrize(
    ("path", "rule"),
    [
        ("Rationale and impact", "Rationale and impact"),
        ("Rationale and impact › Lumbar puncture › Why", "Rationale and impact"),
        ("Contents", "Contents"),
        (None, ""),  # no heading: title pages, logos
        ("", ""),
        ("Rationale and impacts of fluids", None),  # longer heading, same start
        ("Antibiotics for late-onset neonatal infection › Treatment duration", None),
        ("Terms used in this guideline › Early-onset", None),
    ],
)
def test_section_excluded_matches_section_and_subsections_only(
    path: str | None, rule: str | None
) -> None:
    assert section_excluded(path, _PREFIXES) == rule


def test_exclude_sections_marks_chunks_and_overrides_a_pending_hold() -> None:
    chunks = [
        {"section_path": "Rationale and impact › CRP", "meta": {"review_status": "pending"}},
        {"section_path": "Antibiotics › Choice", "meta": {}},
        {"section_path": None, "meta": {}},
    ]
    assert exclude_sections(chunks, _PREFIXES) == 2
    assert chunks[0]["meta"]["review_status"] == "excluded"
    assert chunks[0]["meta"]["exclusion"] == {
        "rule": "Rationale and impact",
        "previous_status": "pending",
    }
    assert "review_status" not in chunks[1]["meta"]
    assert chunks[2]["meta"]["review_status"] == "excluded"


def test_load_exclude_sections_validates_the_manifest_field() -> None:
    assert load_exclude_sections(None) == []
    assert load_exclude_sections({"exclude_sections": [" Contents "]}) == ["Contents"]
    with pytest.raises(ValueError, match="exclude_sections"):
        load_exclude_sections({"exclude_sections": "Contents"})


def test_an_excluded_chunk_cannot_be_reviewed_back_into_retrieval() -> None:
    chunk = _Chunk({"review_status": "excluded"})
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


# --- carrying confirmations over identical text (DEVIATIONS.md #255) ---

_OLD_TABLE = "Title line\n\nRow: 1.00\n  Agent P: 10\n\nRow: 1.25\n  Agent P: 12"


def test_identical_text_finds_its_confirmed_source() -> None:
    confirmed = [("old-1", _OLD_TABLE, "ocr")]
    part = "Title line\n\nRow: 1.25\n  Agent P: 12"  # title repeated + one row
    assert identical_confirmed_source(part, "ocr", confirmed) == "old-1"


@pytest.mark.parametrize(
    ("text", "source"),
    [
        ("Title line\n\nRow: 1.25\n  Agent P: 13", "ocr"),  # one digit differs
        ("Title line\n\nRow: 1.25\n  Agent P: 12", None),  # table source pin differs
        ("Other title\n\nRow: 1.25\n  Agent P: 12", "ocr"),  # a block not in the source
        ("Title line\n\nRow: 1.25\n  Agent P: 1", "ocr"),  # block cut short mid-number
        ("", "ocr"),
    ],
)
def test_any_difference_leaves_the_chunk_held(text: str, source: str | None) -> None:
    assert identical_confirmed_source(text, source, [("old-1", _OLD_TABLE, "ocr")]) is None


def test_review_updates_both_copies_of_the_status(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.ingestion.review.write_event", lambda session, **kw: None)
    chunk = _Chunk({"review_status": "pending", "embedding_text": "x", "ocr": {"has_digits": True}})
    store = _Store()
    review_chunk(
        _Session(chunk),
        store,
        chunk.id,
        decision="confirmed",
        note=None,
        actor_id=None,
        actor_role="admin",
    )
    ((_ids, payload),) = store.calls
    assert payload["review_status"] == "confirmed"
    assert payload["meta"]["review_status"] == "confirmed"  # #267
    assert "embedding_text" not in payload["meta"]
