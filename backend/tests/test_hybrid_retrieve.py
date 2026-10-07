"""Hybrid retrieval pipeline (ARCH §7), run against qdrant-client's embedded
in-memory mode with the offline stub embedding/reranker backends — no
network, no real models (CLAUDE.md §5)."""

from __future__ import annotations

import pytest

import app.audit.log as audit_log
from app.config import get_settings
from app.ingestion.embed import embed_texts
from app.retrieval import hybrid
from app.retrieval.hybrid import _expand_abbreviations, retrieve
from app.retrieval.sparse import doc_sparse_vector
from app.retrieval.vectorstore import QdrantVectorStore

_DENSE_DIM = 384


@pytest.fixture(autouse=True)
def _stub_backends(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EMBEDDING_BACKEND", "stub")
    monkeypatch.setenv("RERANKER_BACKEND", "stub")
    # Global top_k unless a test opts into per-guideline retrieval (#252).
    monkeypatch.setenv("RETRIEVAL_MODE", "fused")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _seed_chunk(store: QdrantVectorStore, point_id: int, *, chunk_id: str, text: str, **payload):
    dense = embed_texts([text])[0]
    sparse = doc_sparse_vector(text)
    store.upsert_chunks(
        [
            {
                "id": point_id,
                "dense": dense,
                "sparse": sparse,
                "payload": {
                    "chunk_id": chunk_id,
                    "text": text,
                    "section_path": "1 › Introduction",
                    "section_number": "1",
                    "page_start": 1,
                    "page_end": 1,
                    "char_start": 0,
                    "char_end": len(text),
                    "document_id": "doc-1",
                    "document_title": "Test Guideline",
                    "document_version_id": "v1",
                    "version_label": "2024",
                    "effective_date": "2024-01-01",
                    "status": "active",
                    "chunk_type": "prose",
                    **payload,
                },
            }
        ]
    )


@pytest.fixture
def store() -> QdrantVectorStore:
    s = QdrantVectorStore(url=":memory:", api_key="", collection="hybrid_test")
    s.ensure_collection(dense_dim=_DENSE_DIM)
    return s


def test_expand_abbreviations_appends_expansion_keeps_original() -> None:
    out = _expand_abbreviations("Suspected SBI in a newborn on the NBU")
    assert "SBI (serious bacterial infection)" in out
    assert "NBU (newborn unit)" in out


def test_expand_abbreviations_is_case_insensitive_and_whole_word() -> None:
    out = _expand_abbreviations("kmc for a preterm infant")
    assert "kmc (kangaroo mother care)" in out
    # "hr" must not match inside "there" or similar
    assert _expand_abbreviations("there is no match here") == "there is no match here"


def test_expand_abbreviations_leaves_unknown_terms_untouched() -> None:
    assert _expand_abbreviations("no abbreviations in this sentence") == (
        "no abbreviations in this sentence"
    )


def test_retrieve_ranks_the_exact_text_match_first(store: QdrantVectorStore) -> None:
    target_text = "Blood cultures are recommended before starting antimicrobials."
    _seed_chunk(store, 1, chunk_id="c1", text=target_text)
    _seed_chunk(store, 2, chunk_id="c2", text="Vitamin K is given to newborns shortly after birth.")

    items, snapshot = retrieve(target_text, vectorstore=store)

    assert items[0]["chunk_id"] == "c1"
    assert items[0]["document_id"] == "doc-1"
    assert items[0]["text"] == target_text
    assert snapshot["expanded_query"] == target_text  # no abbreviations to expand
    assert snapshot["items"][0]["chunk_id"] == "c1"


def test_retrieve_confidence_verdict_reflects_top_score(store: QdrantVectorStore) -> None:
    # MIN_SUPPORTING_CHUNKS defaults to 2 -> seed a second, similar-enough chunk
    # so confidence isn't held down purely by a thin corpus (ARCH §7 step 5).
    target_text = "Blood cultures are recommended before starting antimicrobials."
    # no heading path: the reranker then scores exactly the query text (#250)
    _seed_chunk(store, 1, chunk_id="c1", text=target_text, section_path=None)
    _seed_chunk(store, 2, chunk_id="c2", text=target_text, section_path=None)

    _items, snapshot = retrieve(target_text, vectorstore=store)
    assert snapshot["confidence"]["top_score"] == pytest.approx(1.0)
    assert snapshot["confidence"]["low_confidence"] is False


def test_retrieve_on_empty_corpus_is_low_confidence(store: QdrantVectorStore) -> None:
    items, snapshot = retrieve("anything at all", vectorstore=store)
    assert items == []
    assert snapshot["confidence"]["essentially_empty"] is True
    assert snapshot["confidence"]["low_confidence"] is True


def test_retrieve_respects_access_filter_override(store: QdrantVectorStore) -> None:
    target_text = "Blood cultures are recommended before starting antimicrobials."
    _seed_chunk(store, 1, chunk_id="c1", text=target_text, status="withdrawn")

    items, _snapshot = retrieve(target_text, vectorstore=store)  # default filter: status=active
    assert items == []

    items, _snapshot = retrieve(
        target_text, vectorstore=store, access_filter={"status": ["active", "withdrawn"]}
    )
    assert items[0]["chunk_id"] == "c1"


def test_retrieve_flags_conflicting_recommendation_chunks(store: QdrantVectorStore) -> None:
    text_a = "Prophylactic antibiotics are recommended before the procedure for all patients."
    text_b = "Prophylactic antibiotics are not recommended before the procedure for all patients."
    _seed_chunk(
        store,
        1,
        chunk_id="c1",
        text=text_a,
        chunk_type="recommendation",
        section_number="3.1",
        document_version_id="v1",
    )
    _seed_chunk(
        store,
        2,
        chunk_id="c2",
        text=text_b,
        chunk_type="recommendation",
        section_number="3.1",
        document_version_id="v2",
    )

    _items, snapshot = retrieve("prophylactic antibiotics before the procedure", vectorstore=store)
    assert len(snapshot["conflicts"]) >= 1


class _FakeSession:
    def __init__(self) -> None:
        self.added: list = []

    def add(self, obj: object) -> None:
        self.added.append(obj)

    def flush(self) -> None:
        pass


def test_retrieve_writes_a_retrieval_audit_event_when_session_given(
    store: QdrantVectorStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(audit_log, "_fetch_last_row_hash", lambda session: None)
    target_text = "Blood cultures are recommended before starting antimicrobials."
    _seed_chunk(store, 1, chunk_id="c1", text=target_text, section_path=None)

    session = _FakeSession()
    items, _snapshot = retrieve(target_text, vectorstore=store, session=session)

    assert len(session.added) == 1
    event = session.added[0]
    assert event.action == "retrieval"
    assert event.retrieved == [
        {"chunk_id": "c1", "score": pytest.approx(1.0), "fusion": "rrf", "rerank": "stub"}
    ]
    assert event.query_hash is not None


def test_retrieve_skips_audit_write_when_no_session_given(store: QdrantVectorStore) -> None:
    target_text = "Blood cultures are recommended before starting antimicrobials."
    _seed_chunk(store, 1, chunk_id="c1", text=target_text)
    # no session kwarg -> must not raise, must not attempt a DB call
    items, _snapshot = retrieve(target_text, vectorstore=store)
    assert len(items) == 1


def test_reranker_scores_the_heading_path_with_the_chunk_text(
    store: QdrantVectorStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cross-encoder sees what the indexes embed: heading path, then text
    (DEVIATIONS.md #250). A flowchart chunk's own text never names its topic."""
    _seed_chunk(
        store,
        1,
        chunk_id="flow",
        text="[n1] Has ONE of the following • made-up sign",
        section_path="Assessing for possible made-up condition",
    )
    _seed_chunk(store, 2, chunk_id="bare", text="made-up sign alone", section_path=None)
    seen: list[str] = []

    def _capture(query: str, passages: list[str]) -> list[float]:
        seen.extend(passages)
        return [0.5] * len(passages)

    monkeypatch.setattr(hybrid, "rerank", _capture)
    retrieve("made-up sign", vectorstore=store)
    assert (
        "Assessing for possible made-up condition\n\n[n1] Has ONE of the following • made-up sign"
        in seen
    )
    assert "made-up sign alone" in seen  # no heading path: the text alone


def test_retrieval_item_meta_carries_the_top_level_review_status(
    store: QdrantVectorStore,
) -> None:
    """The verifier reads `meta.review_status`; the top-level payload field
    is the source of truth (DEVIATIONS.md #267), so a confirmed chunk whose
    stale `meta` copy still says pending is reported as confirmed."""
    _seed_chunk(
        store,
        1,
        chunk_id="c1",
        text="Blood cultures are recommended before starting antimicrobials.",
        review_status="confirmed",
        meta={"review_status": "pending"},
    )
    items, _ = retrieve("blood cultures before antimicrobials", vectorstore=store)
    assert items[0]["meta"]["review_status"] == "confirmed"
