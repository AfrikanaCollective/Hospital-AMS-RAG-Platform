"""Persist `chunk_document()` output to `corpus.chunk` rows + Qdrant points
(ARCH §5.1 steps 3-4; PRD-004).

Chunks are inserted **in ordinal order** so a later chunk's `parent_ordinal`
(an index into the same `chunk_document()` output list — see
`app.ingestion.chunking`) can be resolved to the real DB id of an earlier
chunk already inserted: `chunk_document` only ever points a chunk's parent at
an earlier ordinal within the same document (ARCH §6 rule 6), so a single
forward pass is always enough.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from app.db.models.corpus import Chunk
from app.ingestion.embed import embed_texts
from app.ingestion.topics import assign_topic_tags
from app.retrieval.sparse import doc_sparse_vector

if TYPE_CHECKING:
    from datetime import date

    from sqlalchemy.orm import Session

    from app.retrieval.vectorstore import VectorStore


def persist_chunks(
    session: Session,
    vectorstore: VectorStore,
    *,
    document_version_id: uuid.UUID,
    document_id: uuid.UUID,
    document_title: str,
    version_label: str,
    effective_date: date | None,
    chunk_dicts: list[dict],
    document_topic_tags: list[str],
) -> list[Chunk]:
    """Embeds, persists (`corpus.chunk`), and upserts (Qdrant) every chunk in
    `chunk_dicts`. Returns the persisted rows, in ordinal order."""
    if not chunk_dicts:
        return []

    ordered = sorted(chunk_dicts, key=lambda c: c["ordinal"])
    embedding_texts = [c["meta"]["embedding_text"] for c in ordered]
    dense_vectors = embed_texts(embedding_texts)

    ordinal_to_id: dict[int, uuid.UUID] = {}
    rows: list[Chunk] = []
    points: list[dict] = []

    for chunk_dict, dense in zip(ordered, dense_vectors, strict=True):
        parent_ordinal = chunk_dict.get("parent_ordinal")
        parent_chunk_id = ordinal_to_id.get(parent_ordinal) if parent_ordinal is not None else None
        topic_tags = assign_topic_tags(
            chunk_dict["text"], chunk_dict.get("heading"), document_topic_tags
        )
        row = Chunk(
            document_version_id=document_version_id,
            section_path=chunk_dict["section_path"],
            section_number=chunk_dict["section_number"],
            heading=chunk_dict["heading"],
            page_start=chunk_dict["page_start"],
            page_end=chunk_dict["page_end"],
            char_start=chunk_dict["char_start"],
            char_end=chunk_dict["char_end"],
            ordinal=chunk_dict["ordinal"],
            parent_chunk_id=parent_chunk_id,
            chunk_type=chunk_dict["chunk_type"],
            text=chunk_dict["text"],
            figure_ref=chunk_dict.get("figure_ref"),
            token_count=chunk_dict.get("token_count"),
            meta={**chunk_dict.get("meta", {}), "topic_tags": topic_tags},
        )
        session.add(row)
        session.flush()
        row.vector_id = str(row.id)
        ordinal_to_id[chunk_dict["ordinal"]] = row.id
        rows.append(row)

        sparse = doc_sparse_vector(chunk_dict["meta"]["embedding_text"])
        points.append(
            {
                "id": str(row.id),
                "dense": dense,
                "sparse": sparse,
                "payload": {
                    "chunk_id": str(row.id),
                    "text": row.text,
                    "section_path": row.section_path,
                    "section_number": row.section_number,
                    "page_start": row.page_start,
                    "page_end": row.page_end,
                    "char_start": row.char_start,
                    "char_end": row.char_end,
                    "document_id": str(document_id),
                    "document_title": document_title,
                    "document_version_id": str(document_version_id),
                    "version_label": version_label,
                    "effective_date": effective_date.isoformat() if effective_date else None,
                    "status": "active",
                    # `pending`/`rejected` chunks are excluded from retrieval
                    # (ARCH-044, app.ingestion.review); absent = no review needed.
                    "review_status": row.meta.get("review_status"),
                    "chunk_type": row.chunk_type,
                    "topic_tags": topic_tags,
                    "heading": row.heading,
                    # meta minus embedding_text (large, retrieval-irrelevant, only
                    # needed at embed time above) — carries `criteria[]` for the
                    # stage-classifier / missing-info agents (ARCH §6 rule 4,
                    # DEVIATIONS.md #71).
                    "meta": {k: v for k, v in row.meta.items() if k != "embedding_text"},
                },
            }
        )

    vectorstore.upsert_chunks(points)
    return rows
