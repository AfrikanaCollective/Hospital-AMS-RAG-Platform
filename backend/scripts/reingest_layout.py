"""Re-ingest the guideline corpus with the layout-aware parser and remap eval
gold sets (ARCH-044, PRD-113; LAYOUT-INGESTION-PROPOSAL.md §10, operator
decision §14.3 "re-ingest + map gold ids").

For every manifest document whose active version wasn't already built by the
layout parser:

1. create a new `document_version` of the same file
   (`app.ingestion.documents.create_reparse_version`); the prior version
   becomes `superseded` — never deleted, so issued citations still resolve;
2. run the normal pipeline (`_run_process_document`) with
   `INGEST_PARSER=layout`, which also pushes `superseded` into the prior
   version's Qdrant payload so it drops out of retrieval;
3. map every old chunk to the new chunk(s) holding its content
   (`app.ingestion.lineage.map_chunks`) and record it in
   `corpus.chunk_lineage`.

Then every `EvalQuestion.gold_relevant_chunks` that references a
re-ingested chunk is remapped (`lineage.remap_gold`). The previous gold sets
are first written to `data/ingest_artifacts/gold_remap_<timestamp>.json`
(question id -> chunk ids; no patient content), so the remap can be undone.

Each document runs in its own transaction. Qdrant isn't transactional with
Postgres, so if a document fails after its points were upserted, the new
points are marked `withdrawn` and the prior version's points set back to
`active` before the error is reported.

Usage (inside the api or worker container):
    INGEST_PARSER=layout python -m scripts.reingest_layout [--dry-run]
        [--only FILENAME ...] [--force] [--no-remap-gold]
    INGEST_PARSER=layout python -m scripts.reingest_layout --remap-from \\
        data/ingest_artifacts/gold_remap_<ts>.json   # rebuild lineage + gold only
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models.corpus import Chunk, ChunkLineage, Document, DocumentVersion
from app.db.models.eval import EvalQuestion
from app.db.session import session_scope
from app.ingestion.documents import create_reparse_version
from app.ingestion.layout.docling_adapter import parser_version
from app.ingestion.lineage import METHOD, LineageLink, map_chunks, remap_gold
from app.ingestion.tasks import _default_vectorstore, _run_process_document


def _manifest(dir_: Path) -> dict:
    return json.loads((dir_ / "manifest.json").read_text(encoding="utf-8")).get("files", {})


def _active_version(session: Session, filename: str) -> tuple[Document, DocumentVersion] | None:
    docs = session.execute(select(Document)).scalars().all()
    doc = next((d for d in docs if d.source_uri and Path(d.source_uri).name == filename), None)
    if doc is None:
        return None
    version = session.execute(
        select(DocumentVersion).where(
            DocumentVersion.document_id == doc.id, DocumentVersion.status == "active"
        )
    ).scalar_one_or_none()
    return (doc, version) if version is not None else None


def _chunks(session: Session, version_id: uuid.UUID) -> list[tuple[str, str]]:
    rows = session.execute(
        select(Chunk).where(Chunk.document_version_id == version_id).order_by(Chunk.ordinal)
    ).scalars()
    return [(str(c.id), c.text) for c in rows]


def reingest_one(
    filename: str, entry: dict, *, force: bool
) -> tuple[list[LineageLink], set[str]] | None:
    store = _default_vectorstore()
    new_version_id: str | None = None
    prior_version_id: str | None = None
    try:
        with session_scope() as session:
            found = _active_version(session, filename)
            if found is None:
                print(f"[reingest] SKIP {filename!r}: not in the corpus (ingest it first)")
                return None
            _doc, prior = found
            if (prior.parser_version or "").startswith("docling") and not force:
                print(
                    f"[reingest] SKIP {filename!r}: already layout-parsed ({prior.parser_version})"
                )
                return None
            prior_version_id = str(prior.id)
            old = _chunks(session, prior.id)
            version = create_reparse_version(session, prior, parser_version=parser_version())
            new_version_id = str(version.id)
            rows = _run_process_document(
                session, new_version_id, topic_tags=entry.get("topic_tags", [])
            )
            new = [(str(r.id), r.text) for r in rows]
            links = map_chunks(old, new)
            for link in links:
                session.add(
                    ChunkLineage(
                        old_chunk_id=uuid.UUID(link.old_id),
                        new_chunk_id=uuid.UUID(link.new_id),
                        score=link.score,
                        method=METHOD,
                    )
                )
            report = version.parse_report or {}
            held = sum(1 for r in rows if (r.meta or {}).get("review_status") == "pending")
            print(
                f"[reingest] {filename!r}: {len(old)} -> {len(new)} chunk(s), "
                f"{len({lk.old_id for lk in links})}/{len(old)} old chunk(s) mapped, "
                f"{held} held for review, parser={report.get('parser')}, "
                f"quality={version.parse_quality}"
            )
            return links, {oid for oid, _ in old}
    except Exception:
        # Postgres rolled back; undo what Qdrant already holds.
        if new_version_id:
            store.set_payload_by_version(new_version_id, {"status": "withdrawn"})
        if prior_version_id:
            store.set_payload_by_version(prior_version_id, {"status": "active"})
        raise


def remap_all_gold(links: list[LineageLink], reingested: set[str], artifacts_dir: Path) -> Counter:
    stats: Counter = Counter()
    with session_scope() as session:
        questions = session.execute(select(EvalQuestion)).scalars().all()
        backup: dict[str, list[str]] = {}
        for q in questions:
            gold = q.gold_relevant_chunks
            if not isinstance(gold, list) or not gold or not (set(gold) & reingested):
                continue
            new_gold = remap_gold(gold, links, reingested)
            backup[str(q.id)] = list(gold)
            q.gold_relevant_chunks = new_gold
            stats["remapped"] += 1
            stats["now_empty"] += not new_gold
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        path = artifacts_dir / f"gold_remap_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
        path.write_text(json.dumps({"method": METHOD, "previous_gold": backup}, indent=1))
        print(f"[reingest] previous gold sets written to {path}")
    return stats


def rebuild_lineage_and_remap(backup_path: Path, artifacts_dir: Path) -> Counter:
    """Map straight from the versions that own the saved gold chunk ids to the
    active version of each document (not only to the version it directly
    superseded), then remap gold **from the saved originals** in
    `backup_path`. This lets gold go from an original baseline to the current
    parse in one step, skipping intermediate versions whose text was worse
    (DEVIATIONS.md #221). Links are recorded with the current `METHOD`."""
    previous: dict[str, list[str]] = json.loads(backup_path.read_text())["previous_gold"]
    wanted = {cid for ids in previous.values() for cid in ids}
    links: list[LineageLink] = []
    reingested: set[str] = set()
    with session_scope() as session:
        owners = {
            c.document_version_id
            for c in session.execute(
                select(Chunk).where(Chunk.id.in_([uuid.UUID(x) for x in wanted]))
            )
            .scalars()
            .all()
        }
        for old_vid in owners:
            old_v = session.get(DocumentVersion, old_vid)
            if old_v is None or old_v.status == "active":
                continue
            active = session.execute(
                select(DocumentVersion).where(
                    DocumentVersion.document_id == old_v.document_id,
                    DocumentVersion.status == "active",
                )
            ).scalar_one_or_none()
            if active is None:
                continue
            old = _chunks(session, old_v.id)
            new = _chunks(session, active.id)
            doc_links = map_chunks(old, new)
            for link in doc_links:
                session.add(
                    ChunkLineage(
                        old_chunk_id=uuid.UUID(link.old_id),
                        new_chunk_id=uuid.UUID(link.new_id),
                        score=link.score,
                        method=METHOD,
                    )
                )
            links.extend(doc_links)
            reingested |= {oid for oid, _ in old}
            print(
                f"[reingest] lineage {METHOD}: {old_v.parser_version or 'pypdf'} version "
                f"{old_v.id} -> active {active.id}: "
                f"{len({lk.old_id for lk in doc_links})}/{len(old)} old chunk(s) mapped, "
                f"{len(doc_links)} link(s)"
            )
        stats: Counter = Counter()
        for q in session.execute(select(EvalQuestion)).scalars().all():
            original = previous.get(str(q.id))
            if original is None:
                continue
            q.gold_relevant_chunks = remap_gold(original, links, reingested)
            stats["remapped"] += 1
            stats["now_empty"] += not q.gold_relevant_chunks
            stats["gold_before"] += len(original)
            stats["gold_after"] += len(q.gold_relevant_chunks)
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path(get_settings().sample_guidelines_dir))
    parser.add_argument("--only", nargs="*", default=None, help="manifest filenames to re-ingest")
    parser.add_argument(
        "--force", action="store_true", help="re-parse even if already layout-parsed"
    )
    parser.add_argument("--no-remap-gold", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--remap-from",
        type=Path,
        default=None,
        help="don't re-ingest: rebuild lineage with the current method and remap gold "
        "from this saved gold_remap_*.json (the pre-remap originals)",
    )
    args = parser.parse_args(argv)

    if args.remap_from is not None:
        stats = rebuild_lineage_and_remap(
            args.remap_from, Path(get_settings().ingest_crop_dir).parent
        )
        n = max(stats["remapped"], 1)
        print(
            f"[reingest] gold remapped from {args.remap_from.name}: "
            f"{stats['remapped']} question(s), "
            f"mean gold size {stats['gold_before'] / n:.2f} -> {stats['gold_after'] / n:.2f}, "
            f"{stats['now_empty']} now empty"
        )
        return 0

    if get_settings().ingest_parser != "layout":
        print("[reingest] set INGEST_PARSER=layout for this run", file=sys.stderr)
        return 2
    manifest = _manifest(args.dir)
    targets = [f for f in manifest if args.only is None or f in args.only]
    if args.dry_run:
        for f in targets:
            print(f"[reingest] would re-ingest {f!r}")
        return 0

    all_links: list[LineageLink] = []
    reingested: set[str] = set()
    for filename in targets:
        result = reingest_one(filename, manifest[filename], force=args.force)
        if result is not None:
            links, olds = result
            all_links.extend(links)
            reingested |= olds

    if reingested and not args.no_remap_gold:
        stats = remap_all_gold(all_links, reingested, Path(get_settings().ingest_crop_dir).parent)
        print(
            f"[reingest] gold sets remapped: {stats['remapped']} question(s); "
            f"{stats['now_empty']} now have no gold chunk (they leave the calibration pool)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
