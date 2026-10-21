"""CLI: python -m scripts.draft_area_reference_sets [--out PATH] [--core-share F]
[--export-corpus PATH] (PRD-112, ARCH-043; DEVIATIONS.md #307).

Drafts `data/area_reference_sets.yaml` for the operator to curate and attest.
Candidates per area are every retrievable chunk cited by that area's
ablation-holdout answers, most-cited first, each with its document, page,
heading, text hash, share of the area's answers that cite it, and an excerpt.
`include: true` is pre-set where the share is at least `--core-share`
(default 0.5) and `false` otherwise: a starting point, not a decision. The
attestation fields are `TODO_CONFIRM`, so nothing can score against the file
until the operator has curated and signed it.

`--export-corpus PATH` instead writes every retrievable chunk as JSON (full
text, chunk type, pages, and how many of each area's answers cite it) for the
review page (`scripts/area_reference_review.py`), and leaves the draft
untouched. Guideline text only; no patient data is read or written.

Refuses to overwrite a file whose attestation is already filled in.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models.corpus import Chunk, Document, DocumentVersion
from app.db.models.eval import EvalQuestion
from app.db.session import session_scope
from app.eval.area_reference import (
    AreaReferenceNotAttested,
    chunk_text_sha256,
    load_area_reference_sets,
)
from app.eval.auto_seed import FOUR_AREA_TEMPLATE
from app.eval.question_gen.areas import AREA_ORDER
from app.ingestion.review import NOT_RETRIEVABLE

_EXCERPT_CHARS = 240

_HEADER = """\
# Curated fixed reference sets per query area (PRD-112, ARCH-043;
# DEVIATIONS.md #307). Drafted by scripts.draft_area_reference_sets from the
# chunks the ablation-holdout answers cited; the operator curates and attests.
#
# Each area lists the guideline passages that answer that area's question.
# Every question in an area shares the area's wording template, so one set
# per area replaces the per-question gold sets (the system's own citations)
# when scoring a unified-ablation run (scripts.score_area_reference).
#
# To curate: set `include` to true for each passage that answers the area's
# question and false for one that doesn't. `include: true` was pre-set where
# at least {core_pct}% of the area's answers cited the passage; that is a
# starting point only. Review, add passages no answer cited, and attest with
# the local review page: python3 backend/scripts/area_reference_review.py
# (DEVIATIONS.md #308). Passages are matched by `text_sha256`; the other
# fields are for reading only.
#
# Code may use this file only while every attestation field is filled in
# (no TODO_CONFIRM) and every area has at least one included passage.

"""


def _entry(chunk: Chunk, title: str) -> dict:
    excerpt = " ".join(chunk.text.split())
    return {
        "document": title,
        "page": chunk.page_start,
        "heading": chunk.heading or chunk.section_path or "",
        "text_sha256": chunk_text_sha256(chunk.text),
        "excerpt": excerpt[:_EXCERPT_CHARS] + ("..." if len(excerpt) > _EXCERPT_CHARS else ""),
    }


def _retrievable_chunks(session: Session) -> dict[str, tuple[Chunk, str]]:
    rows = session.execute(
        select(Chunk, Document.title)
        .join(DocumentVersion, Chunk.document_version_id == DocumentVersion.id)
        .join(Document, DocumentVersion.document_id == Document.id)
        .where(DocumentVersion.status == "active")
    ).all()
    return {
        str(chunk.id): (chunk, title)
        for chunk, title in rows
        if (chunk.meta or {}).get("review_status") not in NOT_RETRIEVABLE
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Draft per-area reference sets (DEVIATIONS #307)")
    parser.add_argument("--out", default="data/area_reference_sets.yaml")
    parser.add_argument("--core-share", type=float, default=0.5)
    parser.add_argument("--export-corpus", default=None)
    args = parser.parse_args(argv)

    out = Path(args.out)
    if out.exists() and not args.export_corpus:
        try:
            load_area_reference_sets(out)
        except AreaReferenceNotAttested:
            pass  # still a draft: safe to regenerate
        else:
            print(f"[area-reference] {out} is attested; refusing to overwrite", file=sys.stderr)
            return 1

    with session_scope() as session:
        chunks = _retrievable_chunks(session)
        questions = session.execute(
            select(EvalQuestion).where(
                EvalQuestion.generator_meta["template_version"].astext == FOUR_AREA_TEMPLATE,
                EvalQuestion.generator_meta["purpose"].astext == "ablation_holdout",
            )
        ).scalars()
        cited: dict[str, Counter[str]] = defaultdict(Counter)
        answered: Counter[str] = Counter()
        for q in questions:
            area = (q.generator_meta or {}).get("area")
            if area not in AREA_ORDER or not q.gold_relevant_chunks:
                continue
            answered[area] += 1
            cited[area].update(str(c) for c in set(q.gold_relevant_chunks))

        if args.export_corpus:
            export = sorted(
                (
                    {
                        **_entry(chunk, title),
                        "page_end": chunk.page_end,
                        "chunk_type": chunk.chunk_type,
                        "text": chunk.text,
                        "cited_by": {
                            area: [cited[area][chunk_id], answered[area]] for area in AREA_ORDER
                        },
                    }
                    for chunk_id, (chunk, title) in chunks.items()
                ),
                key=lambda e: (e["document"], e["page"], e["heading"]),
            )
            Path(args.export_corpus).write_text(json.dumps(export, ensure_ascii=False, indent=1))
            print(
                f"[area-reference] wrote {len(export)} retrievable chunk(s) to {args.export_corpus}"
            )
            return 0

        areas: dict[str, dict] = {}
        for area in AREA_ORDER:
            n = answered[area]
            passages = []
            for chunk_id, count in cited[area].most_common():
                if chunk_id not in chunks:
                    continue  # no longer retrievable
                chunk, title = chunks[chunk_id]
                share = count / n
                passages.append(
                    {
                        "include": share >= args.core_share,
                        **_entry(chunk, title),
                        "cited_by": f"{count} of {n} answers ({share:.0%})",
                    }
                )
            areas[area] = {"passages": passages}
            print(f"[area-reference] {area}: {len(passages)} candidate(s) from {n} answers")

        body = {
            "attested_by": "TODO_CONFIRM",
            "attested_role": "TODO_CONFIRM",
            "attested_date": "TODO_CONFIRM",
            "areas": areas,
        }
        out.write_text(
            _HEADER.format(core_pct=round(args.core_share * 100))
            + yaml.safe_dump(body, sort_keys=False, allow_unicode=True, width=100)
        )
        print(f"[area-reference] wrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
