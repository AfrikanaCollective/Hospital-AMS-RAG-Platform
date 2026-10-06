"""VectorStore adapter (ARCH-002). Qdrant implementation.

Guideline chunks: named vectors `dense` + `sparse`, server-side RRF fusion
(Qdrant's Query API `prefetch` + `FusionQuery(fusion=Fusion.RRF)`), payload
filtering for access scoping (status, topic_tags, allowed_doc_ids).
If PATIENT_RECORD_VECTORS_ENABLED is ever set, record vectors go in a SEPARATE
collection with mandatory patient_id payload filtering (ARCH-023).

The adapter interface keeps a swap path open (DEVIATIONS.md #1).

The `sparse` field carries `Modifier.IDF` so Qdrant applies corpus IDF at
query time over the raw term-frequency vectors `app.retrieval.sparse` builds
(DEVIATIONS.md #48).

**RRF `k` is not client-configurable** (DEVIATIONS.md #49): Qdrant's
server-side RRF fusion does not expose a `k` constant through the client API
(only `fusion=Fusion.RRF` itself). `settings.rrf_k` (default 60, matching
Qdrant's own internal constant) is therefore currently informational only —
it documents the assumption rather than tuning behaviour. If Qdrant ever
exposes a tunable `k`, wire it through here.

`url=":memory:"` runs an embedded, in-process Qdrant (no server, no network) —
used by tests; a real deployment always passes a `http(s)://` URL.
"""

from __future__ import annotations

from typing import Protocol

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm


class VectorStore(Protocol):
    def ensure_collection(self, dense_dim: int) -> None: ...
    def upsert_chunks(self, points: list[dict]) -> None: ...
    def hybrid_search(
        self,
        *,
        dense: list[float],
        sparse: dict,
        prefetch_limit: int,
        limit: int,
        flt: dict | None,
    ) -> list[dict]: ...
    def get_by_ids(self, ids: list[str]) -> list[dict]: ...
    def set_payload(self, ids: list[str], payload: dict) -> None: ...
    def set_payload_by_version(self, document_version_id: str, payload: dict) -> None: ...


def _build_filter(flt: dict) -> qm.Filter:
    must: list[qm.Condition] = []
    if "status" in flt:
        statuses = flt["status"] if isinstance(flt["status"], list) else [flt["status"]]
        must.append(qm.FieldCondition(key="status", match=qm.MatchAny(any=statuses)))
    if flt.get("topic_tags"):
        must.append(qm.FieldCondition(key="topic_tags", match=qm.MatchAny(any=flt["topic_tags"])))
    if flt.get("document_titles"):
        must.append(
            qm.FieldCondition(key="document_title", match=qm.MatchAny(any=flt["document_titles"]))
        )
    if flt.get("allowed_doc_ids"):
        must.append(
            qm.FieldCondition(key="document_id", match=qm.MatchAny(any=flt["allowed_doc_ids"]))
        )
    must_not: list[qm.Condition] = []
    if flt.get("exclude_document_titles"):
        must_not.append(
            qm.FieldCondition(
                key="document_title", match=qm.MatchAny(any=flt["exclude_document_titles"])
            )
        )
    # Chunks held for review (OCR'd numbers, low parse quality) or rejected by
    # a reviewer are never retrievable (ARCH-044, app.ingestion.review).
    if flt.get("exclude_review_status"):
        must_not.append(
            qm.FieldCondition(
                key="review_status", match=qm.MatchAny(any=list(flt["exclude_review_status"]))
            )
        )
    return qm.Filter(must=must, must_not=must_not or None)


class QdrantVectorStore:
    def __init__(self, url: str, api_key: str, collection: str) -> None:
        self.url = url
        self.collection = collection
        self._client = (
            QdrantClient(location=":memory:")
            if url == ":memory:"
            else QdrantClient(url=url, api_key=api_key or None)
        )

    def ensure_collection(self, dense_dim: int) -> None:
        """Create the collection if absent; assert dense dim otherwise
        (ARCH §6 "Dense dimension read from the model at startup and asserted
        against the Qdrant collection")."""
        if self._client.collection_exists(self.collection):
            info = self._client.get_collection(self.collection)
            existing = info.config.params.vectors["dense"].size  # type: ignore[index]
            if existing != dense_dim:
                raise RuntimeError(
                    f"Qdrant collection {self.collection!r} has dense dim {existing}, "
                    f"but the configured embedding model produces {dense_dim}. "
                    "Re-embed into a new collection (ARCH §6 'Re-embedding')."
                )
            return
        self._client.create_collection(
            collection_name=self.collection,
            vectors_config={"dense": qm.VectorParams(size=dense_dim, distance=qm.Distance.COSINE)},
            sparse_vectors_config={"sparse": qm.SparseVectorParams(modifier=qm.Modifier.IDF)},
        )

    def upsert_chunks(self, points: list[dict]) -> None:
        """Each point: {id, dense: [...], sparse: {indices, values}, payload: {...}}."""
        qpoints = [
            qm.PointStruct(
                id=p["id"],
                vector={
                    "dense": p["dense"],
                    "sparse": qm.SparseVector(
                        indices=p["sparse"]["indices"], values=p["sparse"]["values"]
                    ),
                },
                payload=p["payload"],
            )
            for p in points
        ]
        self._client.upsert(collection_name=self.collection, points=qpoints)

    def hybrid_search(
        self,
        *,
        dense: list[float],
        sparse: dict,
        prefetch_limit: int,
        limit: int,
        flt: dict | None = None,
    ) -> list[dict]:
        """`prefetch_limit` (ARCH §7 step 2, `CANDIDATE_K`) bounds each of the
        dense/sparse candidate lists before fusion; `limit` (ARCH §7 step 3,
        `FUSED_K`) bounds the fused result."""
        qfilter = _build_filter(flt) if flt else None
        result = self._client.query_points(
            collection_name=self.collection,
            prefetch=[
                qm.Prefetch(query=dense, using="dense", limit=prefetch_limit, filter=qfilter),
                qm.Prefetch(
                    query=qm.SparseVector(indices=sparse["indices"], values=sparse["values"]),
                    using="sparse",
                    limit=prefetch_limit,
                    filter=qfilter,
                ),
            ],
            query=qm.FusionQuery(fusion=qm.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return [{"id": str(pt.id), "score": pt.score, **(pt.payload or {})} for pt in result.points]

    def get_by_ids(self, ids: list[str]) -> list[dict]:
        records = self._client.retrieve(collection_name=self.collection, ids=ids, with_payload=True)
        return [{"id": str(r.id), **(r.payload or {})} for r in records]

    def single_vector_search(
        self,
        *,
        using: str,
        query: list[float] | dict[str, list],
        limit: int,
        flt: dict | None = None,
    ) -> list[dict]:
        """One named vector only (`"dense"` or `"sparse"`), no fusion.

        Used only by the offline BM25/vector weight-sweep tooling (ARCH-040,
        `app.eval.retrieval_tuning`) to get each signal's own ranked
        candidates for a client-side weighted combine. Production retrieval
        never calls this — it stays on `hybrid_search`'s server-side RRF
        fusion (ARCH-003)."""
        qfilter = _build_filter(flt) if flt else None
        q: list[float] | qm.SparseVector
        if isinstance(query, dict):
            q = qm.SparseVector(indices=query["indices"], values=query["values"])
        else:
            q = query
        result = self._client.query_points(
            collection_name=self.collection,
            query=q,
            using=using,
            limit=limit,
            query_filter=qfilter,
            with_payload=True,
        )
        return [{"id": str(pt.id), "score": pt.score, **(pt.payload or {})} for pt in result.points]

    def set_payload(self, ids: list[str], payload: dict) -> None:
        """Merge `payload` into the given points (review decisions)."""
        if ids:
            point_ids: list[qm.ExtendedPointId] = list(ids)
            self._client.set_payload(
                collection_name=self.collection, payload=payload, points=point_ids, wait=True
            )

    def set_payload_by_version(self, document_version_id: str, payload: dict) -> None:
        """Merge `payload` into every point of one document version (status
        changes on supersession / withdrawal, `app.ingestion.review`)."""
        self._client.set_payload(
            collection_name=self.collection,
            payload=payload,
            points=qm.Filter(
                must=[
                    qm.FieldCondition(
                        key="document_version_id", match=qm.MatchValue(value=document_version_id)
                    )
                ]
            ),
            wait=True,
        )

    def scroll_all(self, *, page_size: int = 256, flt: dict | None = None) -> list[dict]:
        """Every point's payload, no vector, no query — a full corpus dump.

        Used only by the model-ablation harness (ARCH-041,
        `app.eval.model_ablation`) to embed the whole guideline corpus with a
        new encoder in memory; only feasible because that corpus is small
        (314 points on the live collection, 2026-09-17). Production retrieval
        never calls this — it stays on `hybrid_search`."""
        points: list[dict] = []
        offset = None
        while True:
            batch, offset = self._client.scroll(
                collection_name=self.collection,
                limit=page_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
                scroll_filter=_build_filter(flt) if flt else None,
            )
            points.extend({"id": str(pt.id), **(pt.payload or {})} for pt in batch)
            if offset is None:
                break
        return points
