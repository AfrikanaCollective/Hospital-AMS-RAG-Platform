"""Pure aggregation over `PerQueryResult` rows into the statistical
comparisons requirement XII asks for (UNIFIED-ABLATION-PROPOSAL.md §3.7,
§4 points 4/5, §12): Level 1 (present-only vs. all-assessed), Level 2
(enriched vs. raw, within each Level 1), Level 3 (recall@k as a function
of `bm25_weight`, the continuous BM25/SapBERT weighted-rank-fusion sweep).
No I/O — offline-testable on a plain list of `PerQueryResult`, mirroring
every other pure-compute layer already in this codebase (CLAUDE.md §5).

**Restructured 2026-09-23** (operator request, DEVIATIONS.md #201): the
primary metric is now recall@k (`PerQueryResult.recall_at_k`), not MRR@k
— `per_query_scores`/every `summarize_*` function takes a `metric`
selector (`"recall"` default, `"mrr"` for the still-available secondary
view, reading `reciprocal_rank_at_k`). Level 3's old "each arm vs. `bm25`"
comparison (`summarize_level3`) and the RRF-vs-alpha-blend mechanism
comparison (`summarize_level3_mechanism`, proposal §11) are both removed
— MedCPT and RRF fusion are dropped entirely, and Level 3 is no longer a
set of named arms to compare, it's one continuous curve over
`bm25_weight`. `summarize_level3_curve` replaces both.

**Extended same day** (operator request, DEVIATIONS.md #202): every
`DeltaScore` now carries a bootstrap `p_value` alongside its CI
(`app.eval.bootstrap.paired_bootstrap_test`, one resampling pass, not a
second independent one). Two new functions:
`summarize_level3_by_weight_and_k` — recall@k for every (`bm25_weight`,
`k`) combination, pooled across Level 1 x Level 2 (the full grid, not
collapsed to one headline `k` or sliced by level); and
`summarize_best_weight_vs_bm25` — an explicit, EXPLICITLY POST-HOC test of
whether BM25 weighting helps at all, comparing whichever `bm25_weight`
empirically maximizes recall@k against `bm25_weight=1.0` (pure BM25). The
weight is selected AFTER seeing the data (the max of 11 empirical
means) — see that function's own docstring for why its CI/p-value must be
read differently from every other comparison in this module.

**Extended 2026-10-07** (operator request, DEVIATIONS.md #272):
`summarize_popularity_baseline` — a query-blind leave-one-out "popularity"
ranking (chunks ordered by how many OTHER queries' gold sets contain them),
scored on Recall@k/MRR@k/Hit@k and compared, paired, against every
Level 1 x Level 2 x bm25_weight arm. Both question pools concentrate gold
on a handful of chunks (#271), so an arm only shows real query-driven
ranking if it beats this.

Pooling convention (proposal §4 point 4, extended consistently to every
comparison here — a judgment call, flagged): a comparison names the
condition(s) it varies; every OTHER free dimension is marginalized by
averaging a query's metric value across it first, so each query
contributes exactly one scalar score per side of the comparison. `k` is
always fixed to one caller-supplied value (typically
`app.eval.ablation_config.mrr_k()`), never pooled across k.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Literal

import numpy as np

from app.eval.ablation_config import Level1Condition, Level2Condition
from app.eval.bootstrap import DEFAULT_BOOTSTRAP_SEED, bootstrap_ci, paired_bootstrap_test
from app.eval.metrics import hit_at_k, mrr, precision_recall_at_k
from app.eval.unified_ablation.per_query import PerQueryResult

Metric = Literal["recall", "mrr", "hit"]


def _metric_value(row: PerQueryResult, metric: Metric) -> float:
    if metric == "recall":
        return row.recall_at_k
    if metric == "mrr":
        return row.reciprocal_rank_at_k
    if row.hit_at_k is not None:
        return row.hit_at_k
    # Rows written before `hit_at_k` existed (DEVIATIONS.md #269).
    rank = row.first_relevant_rank
    return 1.0 if rank is not None and rank <= row.k else 0.0


@dataclass(frozen=True)
class ArmScore:
    label: str
    n: int
    mean: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class DeltaScore:
    """`mean_delta` = mean(scores_a) - mean(scores_b), over the queries
    common to both (a true paired comparison — a query missing from either
    side contributes to neither). `p_value` (DEVIATIONS.md #202) is a
    two-sided bootstrap p-value on the null hypothesis mean_delta == 0,
    from the SAME resampling pass as `ci_low`/`ci_high`."""

    label_a: str
    label_b: str
    n: int
    mean_delta: float
    ci_low: float
    ci_high: float
    p_value: float


def per_query_scores(
    rows: list[PerQueryResult],
    *,
    k: int,
    metric: Metric = "recall",
    level1: Level1Condition | None = None,
    level2: Level2Condition | None = None,
    bm25_weight: float | None = None,
) -> dict[str, float]:
    """One score per `query_id`: the mean of `metric` over every row at
    this `k` matching the given filters — `None` leaves that dimension
    free (pooled/marginalized), a value pins it."""
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.k != k:
            continue
        if level1 is not None and row.level1_condition != level1:
            continue
        if level2 is not None and row.level2_condition != level2:
            continue
        if bm25_weight is not None and row.bm25_weight != bm25_weight:
            continue
        buckets[row.query_id].append(_metric_value(row, metric))
    return {qid: float(np.mean(scores)) for qid, scores in buckets.items()}


def _arm_score(label: str, scores: dict[str, float], *, rng: np.random.Generator) -> ArmScore:
    values = list(scores.values())
    lo, hi = bootstrap_ci(values, rng=rng)
    mean = float(np.mean(values)) if values else 0.0
    return ArmScore(label=label, n=len(values), mean=mean, ci_low=lo, ci_high=hi)


def _delta_score(
    label_a: str,
    label_b: str,
    scores_a: dict[str, float],
    scores_b: dict[str, float],
    *,
    rng: np.random.Generator,
) -> DeltaScore:
    shared = sorted(set(scores_a) & set(scores_b))
    a = [scores_a[q] for q in shared]
    b = [scores_b[q] for q in shared]
    lo, hi, p_value = paired_bootstrap_test(a, b, rng=rng)
    mean_delta = float(np.mean(a) - np.mean(b)) if shared else 0.0
    return DeltaScore(
        label_a=label_a,
        label_b=label_b,
        n=len(shared),
        mean_delta=mean_delta,
        ci_low=lo,
        ci_high=hi,
        p_value=p_value,
    )


@dataclass(frozen=True)
class Level1Summary:
    present_only: ArmScore
    all_assessed: ArmScore
    delta: DeltaScore


def summarize_level1(
    rows: list[PerQueryResult],
    *,
    k: int,
    metric: Metric = "recall",
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> Level1Summary:
    """Present-only vs. all-assessed, pooled over Level 2 x bm25_weight
    (proposal §3.7 item 1, §4 point 4)."""
    rng = np.random.default_rng(seed)
    present = per_query_scores(rows, k=k, metric=metric, level1="present_only")
    assessed = per_query_scores(rows, k=k, metric=metric, level1="all_assessed")
    return Level1Summary(
        present_only=_arm_score("present_only", present, rng=rng),
        all_assessed=_arm_score("all_assessed", assessed, rng=rng),
        delta=_delta_score("present_only", "all_assessed", present, assessed, rng=rng),
    )


@dataclass(frozen=True)
class Level2Summary:
    level1: Level1Condition
    raw: ArmScore
    enriched: ArmScore
    delta: DeltaScore


def summarize_level2(
    rows: list[PerQueryResult],
    *,
    k: int,
    metric: Metric = "recall",
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[Level1Condition, Level2Summary]:
    """Enriched vs. raw, WITHIN each Level 1 condition (proposal §3.7 item
    2), pooled over bm25_weight."""
    rng = np.random.default_rng(seed)
    out: dict[Level1Condition, Level2Summary] = {}
    for level1 in ("present_only", "all_assessed"):
        raw = per_query_scores(rows, k=k, metric=metric, level1=level1, level2="raw")
        enriched = per_query_scores(rows, k=k, metric=metric, level1=level1, level2="enriched")
        out[level1] = Level2Summary(
            level1=level1,
            raw=_arm_score(f"{level1}/raw", raw, rng=rng),
            enriched=_arm_score(f"{level1}/enriched", enriched, rng=rng),
            delta=_delta_score(f"{level1}/enriched", f"{level1}/raw", enriched, raw, rng=rng),
        )
    return out


@dataclass(frozen=True)
class Level3CurvePoint:
    bm25_weight: float
    score: ArmScore


@dataclass(frozen=True)
class Level3Curve:
    """One (Level 1, Level 2) slice's recall@k-vs-bm25_weight curve, plus
    the paired delta between the sweep's two endpoints — `bm25_weight=0.0`
    (pure SapBERT) vs. `bm25_weight=1.0` (pure BM25) — as the headline
    comparison, matching the paired-delta convention every other level in
    this module already uses."""

    level1: Level1Condition
    level2: Level2Condition
    points: list[Level3CurvePoint]  # ordered by bm25_weight, ascending
    endpoints_delta: DeltaScore  # bm25_weight=1.0 (BM25) - bm25_weight=0.0 (SapBERT)


def summarize_level3_curve(
    rows: list[PerQueryResult],
    *,
    k: int,
    weight_values: tuple[float, ...],
    metric: Metric = "recall",
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> list[Level3Curve]:
    """recall@k (or MRR@k) as a function of `bm25_weight`, one curve per
    Level 1 x Level 2 slice (4 total) — the weighted-rank-fusion sweep
    itself, not a fixed set of named arms to compare (DEVIATIONS.md #201).
    `weight_values` is caller-supplied (`app.eval.ablation_config
    .bm25_weight_values()`), never hardcoded."""
    rng = np.random.default_rng(seed)
    out: list[Level3Curve] = []
    for level1 in ("present_only", "all_assessed"):
        for level2 in ("raw", "enriched"):
            points = [
                Level3CurvePoint(
                    bm25_weight=w,
                    score=_arm_score(
                        f"{level1}/{level2}/w={w}",
                        per_query_scores(
                            rows, k=k, metric=metric, level1=level1, level2=level2, bm25_weight=w
                        ),
                        rng=rng,
                    ),
                )
                for w in weight_values
            ]
            sapbert_only = per_query_scores(
                rows, k=k, metric=metric, level1=level1, level2=level2, bm25_weight=0.0
            )
            bm25_only = per_query_scores(
                rows, k=k, metric=metric, level1=level1, level2=level2, bm25_weight=1.0
            )
            out.append(
                Level3Curve(
                    level1=level1,
                    level2=level2,
                    points=points,
                    endpoints_delta=_delta_score(
                        f"{level1}/{level2}/bm25_weight=1.0",
                        f"{level1}/{level2}/bm25_weight=0.0",
                        bm25_only,
                        sapbert_only,
                        rng=rng,
                    ),
                )
            )
    return out


@dataclass(frozen=True)
class WeightKPoint:
    bm25_weight: float
    k: int
    score: ArmScore


def summarize_level3_by_weight_and_k(
    rows: list[PerQueryResult],
    *,
    k_values: tuple[int, ...],
    weight_values: tuple[float, ...],
    metric: Metric = "recall",
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> list[WeightKPoint]:
    """recall@k (or MRR@k) for EVERY (`bm25_weight`, `k`) combination
    (DEVIATIONS.md #202, operator request 2026-09-23: "Recall @ k for k
    range between 2 and 20... for each of the following [weights]" — 6 since
    DEVIATIONS.md #207, 11 when written) —
    the full grid, not collapsed to one headline `k`. Pooled across Level 1
    x Level 2 (same pooling convention as every other comparison in this
    module, proposal §4 point 4, extended here): Level 3 doesn't vary L1/L2,
    so a query's score at a given (weight, k) is averaged across whichever
    L1/L2 rows it has, same as `summarize_level1`/`summarize_level2` pool
    over bm25_weight for their own comparisons. `k_values`/`weight_values`
    are caller-supplied (`app.eval.ablation_config.k_values()`/
    `.bm25_weight_values()`), never hardcoded."""
    rng = np.random.default_rng(seed)
    return [
        WeightKPoint(
            bm25_weight=w,
            k=k,
            score=_arm_score(
                f"w={w}/k={k}",
                per_query_scores(rows, k=k, metric=metric, bm25_weight=w),
                rng=rng,
            ),
        )
        for w in weight_values
        for k in k_values
    ]


@dataclass(frozen=True)
class BestWeightVsBM25:
    """Whether BM25 weighting helps at all: the `bm25_weight` with the
    highest EMPIRICAL recall@k (selected AFTER seeing the data — the max
    of `len(weight_values)` sample means) vs. `bm25_weight=1.0` (pure
    BM25), at one caller-specified `k`.

    **Read `delta`'s CI/p-value with the selection in mind.** This is a
    post-hoc comparison, not a pre-registered one: picking whichever of 11
    weights happened to score highest on this sample, then testing THAT
    weight against BM25, is a textbook multiple-comparisons / "winner's
    curse" setup — the nominal 95% CI and p-value describe the sampling
    distribution of *that specific pre-chosen* comparison, not of
    "the best of 11 empirical picks," so the true false-positive rate for
    "the best weight really beats BM25" is higher than the CI/p-value
    alone would suggest. `selected_weight` is reported explicitly so the
    selection is never hidden in the number."""

    k: int
    selected_weight: float
    selected_weight_score: ArmScore
    bm25_score: ArmScore  # bm25_weight = 1.0
    delta: DeltaScore  # selected_weight - bm25_weight=1.0


def summarize_best_weight_vs_bm25(
    rows: list[PerQueryResult],
    *,
    k: int,
    weight_values: tuple[float, ...],
    metric: Metric = "recall",
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> BestWeightVsBM25:
    """`weight_values` should include `1.0` (pure BM25) — if it doesn't,
    `bm25_weight=1.0` is still evaluated separately as the comparison
    baseline (DEVIATIONS.md #202)."""
    rng = np.random.default_rng(seed)
    per_weight = {
        w: per_query_scores(rows, k=k, metric=metric, bm25_weight=w) for w in weight_values
    }

    def _mean(scores: dict[str, float]) -> float:
        return float(np.mean(list(scores.values()))) if scores else float("-inf")

    selected_weight = max(weight_values, key=lambda w: _mean(per_weight[w]))
    selected_scores = per_weight[selected_weight]
    bm25_scores = per_weight.get(1.0) or per_query_scores(rows, k=k, metric=metric, bm25_weight=1.0)

    return BestWeightVsBM25(
        k=k,
        selected_weight=selected_weight,
        selected_weight_score=_arm_score(f"best(w={selected_weight})", selected_scores, rng=rng),
        bm25_score=_arm_score("w=1.0(bm25)", bm25_scores, rng=rng),
        delta=_delta_score(
            f"best(w={selected_weight})", "w=1.0(bm25)", selected_scores, bm25_scores, rng=rng
        ),
    )


_POPULARITY_METRICS: tuple[Metric, ...] = ("recall", "mrr", "hit")


def popularity_baseline_scores(
    rows: list[PerQueryResult], *, k: int
) -> dict[Metric, dict[str, float]]:
    """Per-query Recall@k / MRR@k / Hit@k of the leave-one-out popularity
    ranking (DEVIATIONS.md #272): for query q, every chunk is ranked by how
    many OTHER queries' gold sets contain it (ties by chunk id), ignoring
    q's text entirely. Leaving q out keeps q's own gold from inflating its
    score. Gold sets are read from the rows themselves (`relevant_ids` is
    the same on every row of a query)."""
    gold: dict[str, set[str]] = {}
    for row in rows:
        gold.setdefault(row.query_id, set(row.relevant_ids))
    counts = Counter(c for g in gold.values() for c in g)
    order = sorted(counts, key=lambda c: (-counts[c], c))
    out: dict[Metric, dict[str, float]] = {m: {} for m in _POPULARITY_METRICS}
    for qid, g in gold.items():
        # Subtracting q's own contribution only lowers its gold chunks'
        # counts, so re-sort just enough of the global order to fill top-k.
        loo = {c: counts[c] - (1 if c in g else 0) for c in order[: k + len(g)]}
        ranking = sorted((c for c, n in loo.items() if n > 0), key=lambda c: (-loo[c], c))
        # A chunk below the first k + |g| in the global order can't climb
        # into the top k, so the cut-off above loses nothing.
        top_k = ranking[:k]
        out["recall"][qid] = precision_recall_at_k(top_k, g, k)[1]
        out["mrr"][qid] = mrr(top_k, g)
        out["hit"][qid] = hit_at_k(top_k, g, k)
    return out


@dataclass(frozen=True)
class PopularityBaseline:
    """The leave-one-out popularity ranking's scores at one `k`, plus a
    paired delta (arm - baseline) for every Level 1 x Level 2 x
    bm25_weight arm, per metric. A positive delta means the arm ranks gold
    higher than a ranker that never reads the query."""

    k: int
    method: str
    baseline: dict[str, ArmScore]  # metric -> score
    arm_vs_baseline: dict[str, list[DeltaScore]]  # metric -> one per arm


def summarize_popularity_baseline(
    rows: list[PerQueryResult],
    *,
    k: int,
    weight_values: tuple[float, ...],
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> PopularityBaseline:
    rng = np.random.default_rng(seed)
    base = popularity_baseline_scores([r for r in rows if r.k == k], k=k)
    baseline = {m: _arm_score(f"popularity/{m}", base[m], rng=rng) for m in _POPULARITY_METRICS}
    deltas: dict[str, list[DeltaScore]] = {}
    for m in _POPULARITY_METRICS:
        deltas[m] = []
        for level1 in ("present_only", "all_assessed"):
            for level2 in ("raw", "enriched"):
                for w in weight_values:
                    arm = per_query_scores(
                        rows, k=k, metric=m, level1=level1, level2=level2, bm25_weight=w
                    )
                    deltas[m].append(
                        _delta_score(
                            f"{level1}/{level2}/w={w}", "popularity", arm, base[m], rng=rng
                        )
                    )
    return PopularityBaseline(
        k=k,
        method="leave-one-out gold-frequency ranking (query-blind; ties by chunk id)",
        baseline=baseline,
        arm_vs_baseline=deltas,
    )
