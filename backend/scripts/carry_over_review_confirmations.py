"""Carry reviewer confirmations over to re-chunked, unchanged text
(DEVIATIONS.md #255).

A re-ingest that only re-chunks a document (e.g. splitting a table into
row groups, #254) holds every OCR'd-digit chunk for review again, although an
admin already confirmed the same text in the previous version. For each held
(`pending`) chunk of an active version, this looks for a `confirmed` chunk in a
superseded version of the same document (most recent version first) whose
text has every block of the held chunk's text as a whole block, with the same
`table_source` pin. Only then is the held
chunk confirmed, through the normal `review_chunk` path (Postgres + Qdrant +
one append-only audit event per chunk), with a note naming the source chunk.
Anything not identical stays held for a human reviewer.

`--from-export PATH` takes the confirmed chunks from a JSON export instead
of superseded versions -- for a corpus rebuilt from scratch, which has no
superseded versions (DEVIATIONS.md #301). The export is a list of
`{"document": <title>, "page_start", "text", "meta"}` rows, matched to the
re-ingested document by title; the same identical-text rule applies.

Dry run by default; pass `--apply` to write.

    python -m scripts.carry_over_review_confirmations [--from-export PATH] [--apply]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sqlalchemy import select

from app.db.models.corpus import Chunk, Document, DocumentVersion
from app.db.session import session_scope
from app.ingestion.review import (
    CONFIRMED,
    PENDING,
    identical_confirmed_source,
    review_chunk,
)
from app.retrieval.hybrid import _default_vectorstore


def _source(meta: dict | None) -> str | None:
    return ((meta or {}).get("table_source") or {}).get("source")


def load_export(path: Path) -> dict[str, list[tuple[str, str, str | None]]]:
    """Exported confirmed chunks by document title, in the matcher's
    `(source label, text, table_source)` shape."""
    by_title: dict[str, list[tuple[str, str, str | None]]] = {}
    for i, row in enumerate(json.loads(path.read_text(encoding="utf-8")) or []):
        label = f"{path.name}#{i} (p.{row.get('page_start')})"
        by_title.setdefault(row["document"], []).append(
            (label, row["text"], _source(row.get("meta")))
        )
    return by_title


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--apply", action="store_true", help="write the confirmations")
    parser.add_argument(
        "--from-export", type=Path, default=None, help="confirmed chunks from a JSON export"
    )
    args = parser.parse_args(argv)
    exported = load_export(args.from_export) if args.from_export else None

    store = _default_vectorstore() if args.apply else None
    carried = kept = 0
    with session_scope() as session:
        for version in list(
            session.execute(
                select(DocumentVersion).where(DocumentVersion.status == "active")
            ).scalars()
        ):
            held = [
                c
                for c in session.execute(
                    select(Chunk)
                    .where(Chunk.document_version_id == version.id)
                    .order_by(Chunk.ordinal)
                ).scalars()
                if (c.meta or {}).get("review_status") == PENDING
            ]
            if not held:
                continue
            document = session.get(Document, version.document_id)
            if exported is not None:
                confirmed = exported.get(document.title, [])
                where = "the export"
            else:
                where = "superseded versions"
                confirmed = [
                    (str(c.id), c.text, _source(c.meta))
                    for c in session.execute(
                        select(Chunk)
                        .join(DocumentVersion, DocumentVersion.id == Chunk.document_version_id)
                        .where(
                            DocumentVersion.document_id == version.document_id,
                            DocumentVersion.status == "superseded",
                        )
                        # most recent superseded version first
                        .order_by(DocumentVersion.ingested_at.desc(), Chunk.ordinal)
                    ).scalars()
                    if (c.meta or {}).get("review_status") == CONFIRMED
                ]
            name = Path(document.source_uri or "").name
            print(f"{name}: {len(held)} held, {len(confirmed)} confirmed in {where}")
            for chunk in held:
                source = identical_confirmed_source(chunk.text, _source(chunk.meta), confirmed)
                label = f"  ordinal {chunk.ordinal:3d} {(chunk.section_path or '')[-45:]}"
                if source is None:
                    kept += 1
                    print(f"{label}: stays held (no identical confirmed text)")
                    continue
                carried += 1
                print(f"{label}: identical to confirmed chunk {source}")
                if args.apply:
                    review_chunk(
                        session,
                        store,
                        chunk.id,
                        decision=CONFIRMED,
                        note=(
                            f"carried over: text identical to confirmed chunk {source} "
                            + (
                                "from the pre-reset export (DEVIATIONS.md #255, #301)"
                                if exported is not None
                                else "of a superseded version (DEVIATIONS.md #255)"
                            )
                        ),
                        actor_id=None,
                        actor_role="system:carry_over_review",
                    )
        if not args.apply:
            session.rollback()
    verb = "confirmed" if args.apply else "would confirm"
    print(f"{verb} {carried}; {kept} stay held" + ("" if args.apply else " (dry run)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
