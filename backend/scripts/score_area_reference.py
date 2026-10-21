"""CLI: python -m scripts.score_area_reference RUN_DIR [--path PATH]
(PRD-112, ARCH-043; DEVIATIONS.md #307).

Rescores an existing unified-ablation run (`results/ablation/<run_id>/`)
against the operator-attested per-area reference sets in
`data/area_reference_sets.yaml`, without re-running retrieval: each row's
`retrieved_ids` is scored against its area's set. Writes
`area_reference_scores.csv` (one row per arm x bm25_weight x K x metric,
with a 95% bootstrap CI and the same metric against the run's own gold set
for comparison) and `area_reference_summary.json` into RUN_DIR.

Fails closed if the file isn't attested, or if any included passage is not
in the active corpus (the run must be scored against the corpus it ran on).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path

from sqlalchemy import select

from app.db.models.corpus import Chunk, DocumentVersion
from app.db.session import session_scope
from app.eval.area_reference import (
    chunk_text_sha256,
    load_area_reference_sets,
    resolve_reference_sets,
    score_rows,
)

_HEADLINE_K = 12


def _rows(path: Path) -> Iterator[dict]:
    with path.open() as fh:
        for line in fh:
            yield json.loads(line)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score a run against area reference sets")
    parser.add_argument("run_dir")
    parser.add_argument("--path", default="data/area_reference_sets.yaml")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    sets = load_area_reference_sets(args.path)
    with session_scope() as session:
        by_hash: dict[str, list[str]] = defaultdict(list)
        for chunk_id, text in session.execute(
            select(Chunk.id, Chunk.text)
            .join(DocumentVersion, Chunk.document_version_id == DocumentVersion.id)
            .where(DocumentVersion.status == "active")
        ):
            by_hash[chunk_text_sha256(text)].append(str(chunk_id))
    reference = resolve_reference_sets(sets, by_hash)

    scores = score_rows(_rows(run_dir / "per_query_results.jsonl"), reference)
    csv_path = run_dir / "area_reference_scores.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(scores[0])))
        writer.writeheader()
        writer.writerows(asdict(s) for s in scores)
    summary = {
        "reference_sets_sha256": sets.sha256,
        "attested_by": f"{sets.attested_by} ({sets.attested_role}), {sets.attested_date}",
        "passages_per_area": {a: len(h) for a, h in sets.passages.items()},
        "chunks_per_area": {a: len(ids) for a, ids in reference.items()},
        "k12": [asdict(s) for s in scores if s.k == _HEADLINE_K],
    }
    (run_dir / "area_reference_summary.json").write_text(json.dumps(summary, indent=2))
    for s in scores:
        if s.k == _HEADLINE_K and s.bm25_weight in (0.0, 1.0):
            print(
                f"[area-reference] {s.level1}/{s.level2}/w={s.bm25_weight} {s.metric}@12 "
                f"{s.mean:.3f} [{s.ci_low:.3f}, {s.ci_high:.3f}] "
                f"(own gold {s.current_gold_mean:.3f})"
            )
    print(f"[area-reference] wrote {csv_path} and area_reference_summary.json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
