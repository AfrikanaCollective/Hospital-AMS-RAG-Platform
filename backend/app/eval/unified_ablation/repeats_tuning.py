"""Choosing `ABLATION_AREA_QUESTION_REPEATS` (N) without tuning on the
evaluation set (PRD-112 / ARCH-043; DEVIATIONS.md #312, #313).

The four-area pool is split by source record, by hash, into a `tune` half
and a `heldout` half. Each N in the grid is scored on both halves with the
four patient arms (`sweep_questions`, one K). N is chosen on `tune` only;
`heldout` numbers are reported for the chosen N against the current default
and N = 1, so the choice can be checked on questions it never saw.

**Selection score (my call, #313):** a question's own-gold MRR@K averaged
over the four patient arms and every BM25 weight in the grid, then over the
`tune` questions. Averaging over weights keeps N from being picked for one
weight that the Level-3 sweep then reports as best. Ties go to the smaller N.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from app.eval.bootstrap import DEFAULT_BOOTSTRAP_SEED, paired_bootstrap_test
from app.eval.retrieval_tuning.sweep import SweepQuestion
from app.eval.unified_ablation.per_query import PerQueryResult

SPLITS = ("tune", "heldout")


def split_of(question: SweepQuestion) -> str:
    """`tune` or `heldout`, from a hash of the source record id: stable
    across runs and the same for every question of one record."""
    key = str(question.source_record_id or question.question_id)
    return SPLITS[int(hashlib.sha256(key.encode()).hexdigest(), 16) % 2]


@dataclass(frozen=True)
class CellScore:
    """Mean over one split's questions for one (N, arm, weight)."""

    split: str
    repeats: int
    level1: str
    level2: str
    bm25_weight: float
    mrr: float
    recall: float
    n_questions: int


@dataclass(frozen=True)
class RepeatsComparison:
    """Held-out paired difference in the selection score, `repeats` minus
    `baseline_repeats`."""

    repeats: int
    baseline_repeats: int
    mean_delta: float
    ci_low: float
    ci_high: float
    p_value: float
    n_questions: int


@dataclass(frozen=True)
class RepeatsTuning:
    chosen: int
    cells: tuple[CellScore, ...]
    # split -> N -> mean selection score over that split's questions
    selection_scores: dict[str, dict[int, float]]
    heldout_comparisons: tuple[RepeatsComparison, ...]


def _question_scores(rows: Iterable[PerQueryResult]) -> dict[str, float]:
    by_question: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_question[r.query_id].append(r.reciprocal_rank_at_k)
    return {q: float(np.mean(v)) for q, v in by_question.items()}


def _cells(split: str, repeats: int, rows: list[PerQueryResult]) -> list[CellScore]:
    groups: dict[tuple[str, str, float], list[PerQueryResult]] = defaultdict(list)
    for r in rows:
        groups[(r.level1_condition, r.level2_condition, r.bm25_weight)].append(r)
    return [
        CellScore(
            split=split,
            repeats=repeats,
            level1=l1,
            level2=l2,
            bm25_weight=w,
            mrr=float(np.mean([r.reciprocal_rank_at_k for r in g])),
            recall=float(np.mean([r.recall_at_k for r in g])),
            n_questions=len({r.query_id for r in g}),
        )
        for (l1, l2, w), g in sorted(groups.items())
    ]


def tune_repeats(
    questions: list[SweepQuestion],
    sweep: Callable[[list[SweepQuestion], int], Iterable[PerQueryResult]],
    *,
    repeats_grid: tuple[int, ...],
    baselines: tuple[int, ...] = (1, 3),
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> RepeatsTuning:
    """`sweep(questions, n)` runs the four patient arms at one K with the
    area question repeated `n` times (`sweep_questions`). Choose N on the
    `tune` split; compare it on `heldout` with each of `baselines` that is
    in the grid and is not the chosen N."""
    if not repeats_grid or min(repeats_grid) < 1:
        raise ValueError(f"repeats_grid must be non-empty and >= 1, got {repeats_grid}")
    by_split: dict[str, list[SweepQuestion]] = {s: [] for s in SPLITS}
    for q in questions:
        by_split[split_of(q)].append(q)

    cells: list[CellScore] = []
    per_question: dict[str, dict[int, dict[str, float]]] = {s: {} for s in SPLITS}
    for n in repeats_grid:
        for split in SPLITS:
            rows = list(sweep(by_split[split], n))
            cells.extend(_cells(split, n, rows))
            per_question[split][n] = _question_scores(rows)

    selection = {
        split: {n: float(np.mean(list(s.values()))) if s else 0.0 for n, s in by_n.items()}
        for split, by_n in per_question.items()
    }
    chosen = max(sorted(repeats_grid), key=lambda n: (selection["tune"][n], -n))

    rng = np.random.default_rng(seed)
    held = per_question["heldout"]
    comparisons: list[RepeatsComparison] = []
    for base in baselines:
        if base == chosen or base not in held:
            continue
        ids = sorted(set(held[chosen]) & set(held[base]))
        a = [held[chosen][q] for q in ids]
        b = [held[base][q] for q in ids]
        lo, hi, p = paired_bootstrap_test(a, b, rng=rng)
        comparisons.append(
            RepeatsComparison(
                repeats=chosen,
                baseline_repeats=base,
                mean_delta=float(np.mean(a) - np.mean(b)) if ids else 0.0,
                ci_low=lo,
                ci_high=hi,
                p_value=p,
                n_questions=len(ids),
            )
        )
    return RepeatsTuning(
        chosen=chosen,
        cells=tuple(cells),
        selection_scores=selection,
        heldout_comparisons=tuple(comparisons),
    )
