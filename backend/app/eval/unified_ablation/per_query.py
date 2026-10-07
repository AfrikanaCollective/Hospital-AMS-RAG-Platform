"""Per-query result persistence (UNIFIED-ABLATION-PROPOSAL.md §3.5, §4
point 2; PRD-112, requirement IX).

File-based, not a Postgres table (operator decision, proposal §4 point 2):

    results/
    └── ablation/
        └── <run_id>/
            ├── configuration.json       # reproducibility snapshot, §3.8
            └── per_query_results.jsonl  # one line per (query, level1, level2, k, bm25_weight)

**Restructured 2026-09-23** (operator request, DEVIATIONS.md #201):
`alpha` renamed to `bm25_weight` (`w_BM25` in the operator's own
notation); `recall_at_k` added as the new primary metric, alongside the
pre-existing `reciprocal_rank_at_k` (MRR, now secondary).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

_DEFAULT_RESULTS_ROOT = Path("results/ablation")


@dataclass(frozen=True)
class PerQueryResult:
    """One row per (query, level1, level2, k, bm25_weight) — exactly the
    fields requirement IX lists, plus `recall_at_k` (DEVIATIONS.md #201).
    The same query is evaluated under every Level-1 x Level-2 combination
    swept across the full `bm25_weight`/k grid (`app.eval.ablation_config`),
    so `query_id` repeats across many rows by design — that's what makes
    the paired comparisons in §3.7 possible."""

    query_id: str
    patient_id_or_case_id: str
    experiment_id: str

    level1_condition: str
    level2_condition: str
    level3_condition: str

    k: int
    bm25_weight: float

    query_text: str
    concept_enriched_query: str

    retrieved_ids: list[str]
    relevant_ids: list[str]

    first_relevant_rank: int | None
    recall_at_k: float
    reciprocal_rank_at_k: float  # secondary (MRR) metric, kept for continuity

    # Four-area questions (DEVIATIONS.md #264): the query area; None for the
    # single-topic pool and for rows written before this field existed.
    question_area: str | None = None

    # Secondary metric for the figures (DEVIATIONS.md #269): 1.0 if any gold
    # chunk is in the top k. None for rows written before this field existed;
    # the figures derive it from `first_relevant_rank` there.
    hit_at_k: float | None = None

    def to_json_dict(self) -> dict:
        return asdict(self)


def run_dir_for(run_id: str, *, results_root: Path | None = None) -> Path:
    root = results_root if results_root is not None else _DEFAULT_RESULTS_ROOT
    d = root / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def file_sha256(path: Path) -> str | None:
    """SHA-256 of `path`'s raw bytes, or `None` if it doesn't exist.
    Recorded in `configuration.json` for the concept vocabulary so a run's
    Level-2 results can be tied to the exact file version they used —
    attestation status alone can't tell two edited versions apart
    (DEVIATIONS.md #210/#211)."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        return None


def write_configuration(run_dir: Path, config: dict) -> Path:
    path = run_dir / "configuration.json"
    path.write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    return path


def write_per_query_results(run_dir: Path, rows: Iterable[PerQueryResult]) -> Path:
    """Streams rows to disk one line at a time rather than building the
    whole list in memory first — a real run's row count is large (4 arms x
    up to 11 bm25_weight values x `len(K_VALUES)` k's per query, x up to
    thousands of queries)."""
    path = run_dir / "per_query_results.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.to_json_dict()) + "\n")
    return path


def read_per_query_results(path: Path) -> list[PerQueryResult]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if not stripped:
                continue
            rows.append(PerQueryResult(**json.loads(stripped)))
    return rows
