"""Per-guideline retrieval (ARCH §7, DEVIATIONS.md #252): each guideline is
searched and reranked separately, keeps its best `cap` chunks above
RETRIEVAL_MIN_SCORE, and groups are ordered by the manifest's operator-attested
`retrieval_priority`. In-memory Qdrant, stub embedding/reranker; fixture text
is deliberately non-clinical."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import get_settings
from app.ingestion.embed import embed_texts
from app.retrieval.hybrid import retrieve, select_per_guideline
from app.retrieval.priority import PriorityError, load_priorities
from app.retrieval.sparse import doc_sparse_vector
from app.retrieval.vectorstore import QdrantVectorStore

_QUERY = "made up widget calibration rule"


def _manifest(tmp_path: Path, files: dict) -> Path:
    (tmp_path / "manifest.json").write_text(json.dumps({"files": files}), encoding="utf-8")
    return tmp_path


@pytest.fixture(autouse=True)
def _stub_backends(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("EMBEDDING_BACKEND", "stub")
    monkeypatch.setenv("RERANKER_BACKEND", "stub")
    monkeypatch.setenv("RETRIEVAL_MODE", "per_guideline")
    monkeypatch.setenv("RETRIEVAL_PER_GUIDELINE_CAP", "2")
    monkeypatch.setenv(
        "SAMPLE_GUIDELINES_DIR",
        str(
            _manifest(
                tmp_path,
                {
                    "a.pdf": {"title": "Guide A", "retrieval_priority": 2},
                    "b.pdf": {"title": "Guide B", "retrieval_priority": 1},
                    "c.pdf": {"title": "Guide C"},  # no priority: searched last
                },
            )
        ),
    )
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def store() -> QdrantVectorStore:
    dim = len(embed_texts(["probe"])[0])
    s = QdrantVectorStore(url=":memory:", api_key="", collection="per_guideline_test")
    s.ensure_collection(dense_dim=dim)
    return s


def _seed(store: QdrantVectorStore, pid: int, title: str, text: str) -> None:
    store.upsert_chunks(
        [
            {
                "id": pid,
                "dense": embed_texts([text])[0],
                "sparse": doc_sparse_vector(text),
                "payload": {
                    "chunk_id": f"{title[-1]}{pid}",
                    "text": text,
                    "section_path": None,
                    "page_start": 1,
                    "page_end": 1,
                    "char_start": 0,
                    "char_end": len(text),
                    "document_id": f"doc-{title[-1]}",
                    "document_title": title,
                    "document_version_id": f"v-{title[-1]}",
                    "version_label": "1",
                    "status": "active",
                    "chunk_type": "prose",
                },
            }
        ]
    )


def _corpus(store: QdrantVectorStore) -> None:
    pid = 1
    for title in ("Guide A", "Guide B", "Guide C"):
        for text in (_QUERY, f"{_QUERY} extra", f"{_QUERY} more words", "unrelated filler"):
            _seed(store, pid, title, text)
            pid += 1


def test_groups_follow_priority_cap_applies_and_weak_chunks_are_dropped(
    store: QdrantVectorStore,
) -> None:
    _corpus(store)
    items, snapshot = retrieve(_QUERY, vectorstore=store)
    assert snapshot["mode"] == "per_guideline"
    titles = [i["document_title"] for i in items]
    # B (priority 1), then A (2), then the unlisted C; 2 per guideline (cap)
    assert titles == ["Guide B", "Guide B", "Guide A", "Guide A", "Guide C", "Guide C"]
    assert "unrelated filler" not in {i["text"] for i in items}
    scores_b = [i["score"] for i in items[:2]]
    assert scores_b == sorted(scores_b, reverse=True)  # best first within a group


def test_cap_can_be_raised_per_call(store: QdrantVectorStore) -> None:
    _corpus(store)
    items, _ = retrieve(_QUERY, vectorstore=store, per_guideline_cap=5)
    # 3 relevant chunks per guideline; the filler stays out even with room
    assert len(items) == 9


def test_a_guideline_with_nothing_relevant_leaves_its_slots_empty(
    store: QdrantVectorStore,
) -> None:
    _corpus(store)
    _seed(store, 99, "Guide D", "nothing in common at all")
    items, _ = retrieve(_QUERY, vectorstore=store)
    assert "Guide D" not in {i["document_title"] for i in items}
    assert len(items) == 6  # not backfilled from other guidelines


def test_no_priorities_falls_back_to_fused(
    store: QdrantVectorStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("SAMPLE_GUIDELINES_DIR", str(empty))
    get_settings.cache_clear()
    _corpus(store)
    items, snapshot = retrieve(_QUERY, vectorstore=store)
    assert snapshot["mode"] == "fused"
    assert len(items) == get_settings().top_k


def test_select_per_guideline_caps_per_document_and_keeps_group_order() -> None:
    def c(doc: str) -> dict:
        return {"document_id": doc}

    groups = [
        [(c("x"), 0.4), (c("x"), 0.9), (c("x"), 0.8)],
        [(c("y"), 0.2)],  # below threshold: contributes nothing
        [(c("u"), 0.7), (c("v"), 0.95), (c("u"), 0.6), (c("u"), 0.5)],  # unlisted
    ]
    picked = select_per_guideline(groups, cap=2, min_score=0.3)
    assert [(d["document_id"], s) for d, s in picked] == [
        ("x", 0.9),
        ("x", 0.8),
        ("v", 0.95),
        ("u", 0.7),
        ("u", 0.6),
    ]


def test_load_priorities_orders_and_validates(tmp_path: Path) -> None:
    assert load_priorities(tmp_path / "missing") == []
    ok = _manifest(
        tmp_path,
        {
            "n.pdf": {"title": "N", "retrieval_priority": 4},
            "k.pdf": {"title": "K", "retrieval_priority": 1},
            "x.pdf": {"title": "X"},
        },
    )
    assert load_priorities(ok) == [(1, "K"), (4, "N")]
    bad = tmp_path / "bad"
    bad.mkdir()
    _manifest(bad, {"n.pdf": {"title": "N", "retrieval_priority": 0}})
    with pytest.raises(PriorityError, match="positive integer"):
        load_priorities(bad)


def test_a_split_table_contributes_at_most_max_parts() -> None:
    def part(split: str | None) -> dict:
        return {"document_id": "k", "meta": {"split_group_id": split} if split else {}}

    rows = [(part("tbl-a"), 0.65 - i / 100) for i in range(5)]  # five near-identical rows
    other_table = [(part("tbl-b"), 0.60), (part("tbl-b"), 0.59), (part("tbl-b"), 0.58)]
    prose = [(part(None), 0.50)]  # e.g. the "reassess at two days" paragraph
    picked = select_per_guideline(
        [rows + other_table + prose], cap=5, min_score=0.3, max_parts_per_table=2
    )
    splits = [(d["meta"] or {}).get("split_group_id") for d, _ in picked]
    assert splits == ["tbl-a", "tbl-a", "tbl-b", "tbl-b", None]
    # without the limit the first table's rows take every slot
    unlimited = select_per_guideline([rows + other_table + prose], cap=5, min_score=0.3)
    assert [(d["meta"] or {}).get("split_group_id") for d, _ in unlimited] == ["tbl-a"] * 5
