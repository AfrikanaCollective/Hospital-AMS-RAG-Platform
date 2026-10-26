"""Curated fixed reference sets per query area (PRD-112, ARCH-043;
DEVIATIONS.md #307).

Each area's four-area questions share one wording template, so the passages
that answer them are, by design, the same for every record. The default gold
set (`EvalQuestion.gold_relevant_chunks`, the chunks the system's own answer
cited) is circular and carries citation noise. `data/area_reference_sets.yaml`
replaces it with one operator-attested passage set per area, and
`score_rows` rescores an existing unified-ablation run against it, without
re-running retrieval.

Passages are identified by the sha256 of their chunk text, not by chunk id,
so the file survives a re-ingestion that reproduces the same text (chunk ids
change on every ingest). `resolve_reference_sets` maps each hash to the
current chunk id(s) and fails if any included passage no longer exists.

What this measures: whether retrieval, given the patient-specific query,
surfaces the passages that answer the area's question. It does not measure
patient-specific retrieval: every question in an area has the same answer.

The loader fails closed (`AreaReferenceNotAttested`) on a missing file, any
missing or `TODO_CONFIRM` attestation field, an area set other than
`AREA_ORDER`, or an area with no included passage.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from app.eval.bootstrap import DEFAULT_BOOTSTRAP_SEED, bootstrap_ci
from app.eval.question_gen.areas import AREA_ORDER

_PLACEHOLDER = "TODO_CONFIRM"


class AreaReferenceNotAttested(ValueError):
    pass


class AreaReferenceUnresolved(ValueError):
    pass


@dataclass(frozen=True)
class AreaReferenceSets:
    attested_by: str
    attested_role: str
    attested_date: str
    # area -> sha256 of every included passage's chunk text
    passages: dict[str, frozenset[str]]
    sha256: str


def chunk_text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _text(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value.strip() or _PLACEHOLDER in value:
        raise AreaReferenceNotAttested(f"{what} is missing or still a placeholder")
    return value.strip()


def load_area_reference_sets(path: str | Path) -> AreaReferenceSets:
    """Load and validate the attested reference sets; fails closed."""
    p = Path(path)
    if not p.exists():
        raise AreaReferenceNotAttested(f"{p} not found")
    raw_bytes = p.read_bytes()
    data = yaml.safe_load(raw_bytes) or {}
    attested_by = _text(data.get("attested_by"), "attested_by")
    attested_role = _text(data.get("attested_role"), "attested_role")
    attested_date = _text(str(data.get("attested_date") or ""), "attested_date")
    raw_areas = data.get("areas") or {}
    if set(raw_areas) != set(AREA_ORDER):
        raise AreaReferenceNotAttested(f"areas must be exactly {list(AREA_ORDER)}")
    passages: dict[str, frozenset[str]] = {}
    for name in AREA_ORDER:
        entries = (raw_areas[name] or {}).get("passages") or []
        included = set()
        for i, entry in enumerate(entries):
            include = entry.get("include")
            if not isinstance(include, bool):
                raise AreaReferenceNotAttested(
                    f"area {name!r} passage {i}: include must be true or false"
                )
            if include:
                included.add(_text(entry.get("text_sha256"), f"area {name!r} passage {i} hash"))
        if not included:
            raise AreaReferenceNotAttested(f"area {name!r} has no included passage")
        passages[name] = frozenset(included)
    return AreaReferenceSets(
        attested_by=attested_by,
        attested_role=attested_role,
        attested_date=attested_date,
        passages=passages,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


def resolve_reference_sets(
    sets: AreaReferenceSets, chunk_ids_by_text_sha256: Mapping[str, Iterable[str]]
) -> dict[str, frozenset[str]]:
    """area -> current chunk ids. `chunk_ids_by_text_sha256` covers the
    active corpus. Fails if any included passage has no current chunk."""
    resolved: dict[str, frozenset[str]] = {}
    missing: list[str] = []
    for area, hashes in sets.passages.items():
        ids: set[str] = set()
        for h in hashes:
            found = {str(c) for c in chunk_ids_by_text_sha256.get(h, ())}
            if not found:
                missing.append(f"{area}:{h[:12]}")
            ids |= found
        resolved[area] = frozenset(ids)
    if missing:
        raise AreaReferenceUnresolved(
            f"{len(missing)} included passage(s) not in the active corpus: {missing}"
        )
    return resolved


@dataclass(frozen=True)
class AreaReferenceScore:
    level1: str
    level2: str
    level3: str  # lexical scorer (DEVIATIONS.md #311)
    bm25_weight: float
    k: int
    metric: str  # recall | mrr | hit
    n: int
    mean: float
    ci_low: float
    ci_high: float
    current_gold_mean: float  # the same metric against the run's own gold set


def _metrics(retrieved: list[str], relevant: frozenset[str]) -> tuple[float, float, float]:
    ranks = [i for i, cid in enumerate(retrieved, start=1) if cid in relevant]
    recall = len({retrieved[r - 1] for r in ranks}) / len(relevant)
    return recall, (1.0 / ranks[0] if ranks else 0.0), (1.0 if ranks else 0.0)


def score_rows(
    rows: Iterable[Mapping[str, Any]],
    reference: Mapping[str, frozenset[str]],
    *,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> list[AreaReferenceScore]:
    """Mean Recall/MRR/Hit@k per arm x weight x k against the area reference
    sets, with a percentile-bootstrap 95% CI, from `per_query_results.jsonl`
    rows. Rows whose area has no reference set are skipped."""
    ours: dict[tuple, list[tuple[float, float, float]]] = defaultdict(list)
    theirs: dict[tuple, list[tuple[float, float, float]]] = defaultdict(list)
    for row in rows:
        relevant = reference.get(row.get("question_area") or "")
        if not relevant:
            continue
        key = (
            row["level1_condition"],
            row["level2_condition"],
            row.get("level3_condition", "bm25_sapbert"),
            row["bm25_weight"],
            row["k"],
        )
        ours[key].append(_metrics(list(row["retrieved_ids"]), relevant))
        theirs[key].append(
            (row["recall_at_k"], row["reciprocal_rank_at_k"], row.get("hit_at_k", 0.0))
        )
    rng = np.random.default_rng(seed)
    out: list[AreaReferenceScore] = []
    for key in sorted(ours):
        for idx, metric in enumerate(("recall", "mrr", "hit")):
            values = [v[idx] for v in ours[key]]
            lo, hi = bootstrap_ci(values, rng=rng)
            out.append(
                AreaReferenceScore(
                    level1=key[0],
                    level2=key[1],
                    level3=key[2],
                    bm25_weight=key[3],
                    k=key[4],
                    metric=metric,
                    n=len(values),
                    mean=float(np.mean(values)),
                    ci_low=lo,
                    ci_high=hi,
                    current_gold_mean=float(np.mean([v[idx] for v in theirs[key]])),
                )
            )
    return out
