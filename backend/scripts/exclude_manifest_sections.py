"""Apply each manifest entry's `exclude_sections` to the already-ingested
corpus (DEVIATIONS.md #251), without re-ingesting.

New ingests apply the list at chunking time (`app.ingestion.tasks`). This
script brings existing active versions into line: every chunk in an excluded
section gets `meta.review_status = "excluded"` in Postgres and the same
`review_status` payload in Qdrant (retrieval's source of truth), with one
append-only audit event per document version. With `--strip-gold`, the
excluded chunk ids are also removed from `eval_question.gold_relevant_chunks`
(the previous lists are written to a JSON backup first).

Dry run by default; pass `--apply` to write.

    python -m scripts.exclude_manifest_sections [--apply] [--strip-gold]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select

from app.audit.log import write_event
from app.config import get_settings
from app.db.models.corpus import Chunk, Document, DocumentVersion
from app.db.models.eval import EvalQuestion
from app.db.session import session_scope
from app.ingestion.page_provenance import load_manifest_entry
from app.ingestion.review import EXCLUDED, load_exclude_sections, payload_meta, section_excluded
from app.retrieval.hybrid import _default_vectorstore

_BACKUP_DIR = Path("data/ingest_artifacts")


def _exclude_version(session, store, version, prefixes: list[str], *, apply: bool) -> set[str]:
    """Exclude one version's matching chunks; returns every excluded chunk id."""
    filename = Path(session.get(Document, version.document_id).source_uri or "").name
    chunks = session.execute(select(Chunk).where(Chunk.document_version_id == version.id)).scalars()
    by_rule: Counter[str] = Counter()
    excluded: set[str] = set()
    changed: list[str] = []
    new_meta: dict[str, dict] = {}
    for chunk in chunks:
        rule = section_excluded(chunk.section_path, prefixes)
        if rule is None:
            continue
        excluded.add(str(chunk.id))
        by_rule[rule or "(no heading)"] += 1
        meta = dict(chunk.meta or {})
        if meta.get("review_status") == EXCLUDED:
            continue
        meta["exclusion"] = {"rule": rule, "previous_status": meta.get("review_status")}
        meta["review_status"] = EXCLUDED
        changed.append(str(chunk.id))
        new_meta[str(chunk.id)] = meta
        if apply:
            chunk.meta = meta
    print(f"{filename}: {len(excluded)} excluded ({len(changed)} newly)")
    for rule, n in by_rule.most_common():
        print(f"    {n:4d}  {rule}")
    if apply and changed:
        session.flush()
        for cid in changed:  # both copies of the status (DEVIATIONS.md #267)
            store.set_payload(
                [cid], {"review_status": EXCLUDED, "meta": payload_meta(new_meta[cid])}
            )
        write_event(
            session,
            action="config_change",
            outcome="chunks_excluded",
            detail={
                "document_version_id": str(version.id),
                "rules": dict(by_rule),
                "chunk_ids": changed,
                "reason": "manifest exclude_sections (DEVIATIONS.md #251)",
            },
        )
    return excluded


def _strip_gold(session, excluded_ids: set[str], *, apply: bool) -> None:
    backup: dict[str, list] = {}
    stats: Counter[str] = Counter()
    for q in session.execute(select(EvalQuestion)).scalars():
        gold = [str(c) for c in (q.gold_relevant_chunks or [])]
        kept = [c for c in gold if c not in excluded_ids]
        if kept == gold:
            continue
        backup[str(q.id)] = gold
        stats["questions_changed"] += 1
        stats["ids_removed"] += len(gold) - len(kept)
        stats["questions_now_empty"] += not kept
        if apply:
            q.gold_relevant_chunks = kept
    print("gold:", dict(stats))
    if apply and backup:
        out = _BACKUP_DIR / f"gold_before_exclusion_{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        note = "gold_relevant_chunks before DEVIATIONS.md #251"
        out.write_text(json.dumps({"note": note, "questions": backup}, indent=1))
        print("gold backup ->", out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--apply", action="store_true", help="write the changes")
    parser.add_argument("--strip-gold", action="store_true", help="also strip eval gold lists")
    args = parser.parse_args(argv)

    settings = get_settings()
    store = _default_vectorstore() if args.apply else None
    excluded_ids: set[str] = set()
    with session_scope() as session:
        versions = session.execute(
            select(DocumentVersion).where(DocumentVersion.status == "active")
        ).scalars()
        for version in list(versions):
            filename = Path(session.get(Document, version.document_id).source_uri or "").name
            prefixes = load_exclude_sections(
                load_manifest_entry(settings.sample_guidelines_dir, filename)
            )
            if prefixes:
                excluded_ids |= _exclude_version(
                    session, store, version, prefixes, apply=args.apply
                )
        if args.strip_gold:
            _strip_gold(session, excluded_ids, apply=args.apply)
        if not args.apply:
            session.rollback()
            print("dry run: nothing written (pass --apply)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
