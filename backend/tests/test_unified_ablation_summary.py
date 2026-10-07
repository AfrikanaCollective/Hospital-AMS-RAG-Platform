"""Pure aggregation of `PerQueryResult` rows into the Level 1/2/3 paired
comparisons requirement XII asks for (PRD-112; UNIFIED-ABLATION-PROPOSAL.md
§3.7). Restructured 2026-09-23 (DEVIATIONS.md #201): recall@k is the new
primary metric (`metric="recall"` default); Level 3 is a continuous
bm25_weight sweep (`summarize_level3_curve`), not a set of named arms."""

from __future__ import annotations

import pytest

from app.eval.unified_ablation.per_query import PerQueryResult
from app.eval.unified_ablation.summary import (
    per_query_scores,
    popularity_baseline_scores,
    popularity_report_rows,
    popularity_scores_from_gold,
    summarize_best_weight_vs_bm25,
    summarize_level1,
    summarize_level2,
    summarize_level3_by_weight_and_k,
    summarize_level3_curve,
    summarize_popularity_baseline,
    summarize_popularity_baseline_by_k,
)

_K = 4
_WEIGHTS = (0.0, 0.5, 1.0)


def _row(
    *,
    query_id: str,
    level1: str,
    level2: str,
    bm25_weight: float,
    recall: float,
    mrr: float | None = None,
) -> PerQueryResult:
    return PerQueryResult(
        query_id=query_id,
        patient_id_or_case_id="rec-1",
        experiment_id="exp-1",
        level1_condition=level1,
        level2_condition=level2,
        level3_condition="bm25_sapbert",
        k=_K,
        bm25_weight=bm25_weight,
        query_text="ignored",
        concept_enriched_query="ignored",
        retrieved_ids=["c1"],
        relevant_ids=["c1"],
        first_relevant_rank=1,
        recall_at_k=recall,
        reciprocal_rank_at_k=mrr if mrr is not None else recall,
    )


def _full_grid_rows(
    query_id: str, *, present_only_recall: float, all_assessed_recall: float
) -> list[PerQueryResult]:
    """One query's full Level1 x Level2 x bm25_weight row set — `present_only_recall`
    used for every present_only row, `all_assessed_recall` for every
    all_assessed row, so pooling never changes the expected mean (every
    pooled-over row carries the same value)."""
    rows = []
    for level1, recall in (
        ("present_only", present_only_recall),
        ("all_assessed", all_assessed_recall),
    ):
        for level2 in ("raw", "enriched"):
            for weight in _WEIGHTS:
                rows.append(
                    _row(
                        query_id=query_id,
                        level1=level1,
                        level2=level2,
                        bm25_weight=weight,
                        recall=recall,
                    )
                )
    return rows


def test_per_query_scores_pools_across_every_unpinned_dimension() -> None:
    rows = [
        _row(query_id="q1", level1="present_only", level2="raw", bm25_weight=1.0, recall=1.0),
        _row(query_id="q1", level1="present_only", level2="enriched", bm25_weight=1.0, recall=0.5),
        _row(query_id="q1", level1="all_assessed", level2="raw", bm25_weight=1.0, recall=0.0),
    ]
    scores = per_query_scores(rows, k=_K, level1="present_only")
    assert scores == {"q1": (1.0 + 0.5) / 2}


def test_per_query_scores_filters_by_k() -> None:
    rows = [
        _row(query_id="q1", level1="present_only", level2="raw", bm25_weight=1.0, recall=1.0),
    ]
    assert per_query_scores(rows, k=999, level1="present_only") == {}


def test_per_query_scores_metric_selector_reads_the_right_field() -> None:
    rows = [
        _row(
            query_id="q1", level1="present_only", level2="raw", bm25_weight=1.0, recall=1.0, mrr=0.3
        ),
    ]
    assert per_query_scores(rows, k=_K, metric="recall", level1="present_only") == {"q1": 1.0}
    assert per_query_scores(rows, k=_K, metric="mrr", level1="present_only") == {"q1": 0.3}


def test_summarize_level1_mean_and_delta_match_the_constant_per_query_values() -> None:
    rows = _full_grid_rows(
        "q1", present_only_recall=0.8, all_assessed_recall=0.2
    ) + _full_grid_rows("q2", present_only_recall=0.6, all_assessed_recall=0.4)
    summary = summarize_level1(rows, k=_K)
    assert summary.present_only.mean == pytest.approx(0.7)  # (0.8 + 0.6) / 2
    assert summary.all_assessed.mean == pytest.approx(0.3)  # (0.2 + 0.4) / 2
    assert summary.present_only.n == 2
    assert summary.delta.mean_delta == pytest.approx(0.7 - 0.3)
    assert summary.delta.n == 2
    assert summary.delta.ci_low <= summary.delta.mean_delta <= summary.delta.ci_high
    assert 0.0 <= summary.delta.p_value <= 1.0


def test_summarize_level1_zero_delta_when_conditions_are_identical() -> None:
    rows = _full_grid_rows("q1", present_only_recall=0.5, all_assessed_recall=0.5)
    summary = summarize_level1(rows, k=_K)
    assert summary.delta.mean_delta == 0.0
    assert summary.delta.ci_low == 0.0
    assert summary.delta.ci_high == 0.0
    assert summary.delta.p_value == pytest.approx(1.0)  # no difference -> p=1


def test_summarize_level2_produces_one_entry_per_level1_condition() -> None:
    rows = _full_grid_rows("q1", present_only_recall=0.9, all_assessed_recall=0.1)
    summary = summarize_level2(rows, k=_K)
    assert set(summary) == {"present_only", "all_assessed"}
    assert summary["present_only"].raw.mean == pytest.approx(0.9)
    assert summary["present_only"].enriched.mean == pytest.approx(0.9)
    assert summary["present_only"].delta.mean_delta == pytest.approx(0.0)


def test_summarize_level3_curve_produces_one_curve_per_slice_with_all_weight_points() -> None:
    rows = _full_grid_rows("q1", present_only_recall=0.9, all_assessed_recall=0.1)
    curves = summarize_level3_curve(rows, k=_K, weight_values=_WEIGHTS)
    assert len(curves) == 4  # 2 level1 x 2 level2
    seen = {(c.level1, c.level2) for c in curves}
    assert len(seen) == 4
    for curve in curves:
        assert [p.bm25_weight for p in curve.points] == list(_WEIGHTS)
        # constant-per-query rows -> every weight point has the same mean
        assert all(p.score.mean == pytest.approx(curve.points[0].score.mean) for p in curve.points)
    # constant-per-query rows -> endpoints are equal -> zero delta
    assert all(c.endpoints_delta.mean_delta == pytest.approx(0.0) for c in curves)


def test_summarize_level3_curve_reflects_a_real_weight_dependent_difference() -> None:
    rows = [
        _row(query_id="q1", level1="present_only", level2="raw", bm25_weight=0.0, recall=0.2),
        _row(query_id="q1", level1="present_only", level2="raw", bm25_weight=1.0, recall=0.9),
    ]
    curves = summarize_level3_curve(rows, k=_K, weight_values=(0.0, 1.0))
    matching = next(c for c in curves if c.level1 == "present_only" and c.level2 == "raw")
    by_weight = {p.bm25_weight: p.score.mean for p in matching.points}
    assert by_weight[0.0] == pytest.approx(0.2)
    assert by_weight[1.0] == pytest.approx(0.9)
    assert matching.endpoints_delta.mean_delta == pytest.approx(0.7)  # w=1.0 - w=0.0


def test_summarize_level3_curve_metric_selector_uses_mrr_when_requested() -> None:
    rows = [
        _row(
            query_id="q1",
            level1="present_only",
            level2="raw",
            bm25_weight=1.0,
            recall=1.0,
            mrr=0.25,
        ),
    ]
    curves = summarize_level3_curve(rows, k=_K, weight_values=(1.0,), metric="mrr")
    matching = next(c for c in curves if c.level1 == "present_only" and c.level2 == "raw")
    assert matching.points[0].score.mean == pytest.approx(0.25)


# ── summarize_level3_by_weight_and_k ──────────────────────────────────────

_K_VALUES = (2, 4)


def _row_for_k(*, query_id: str, k: int, bm25_weight: float, recall: float) -> PerQueryResult:
    return PerQueryResult(
        query_id=query_id,
        patient_id_or_case_id="rec-1",
        experiment_id="exp-1",
        level1_condition="present_only",
        level2_condition="raw",
        level3_condition="bm25_sapbert",
        k=k,
        bm25_weight=bm25_weight,
        query_text="ignored",
        concept_enriched_query="ignored",
        retrieved_ids=["c1"],
        relevant_ids=["c1"],
        first_relevant_rank=1,
        recall_at_k=recall,
        reciprocal_rank_at_k=recall,
    )


def test_summarize_level3_by_weight_and_k_covers_the_full_grid() -> None:
    rows = [
        _row_for_k(query_id="q1", k=k, bm25_weight=w, recall=0.5)
        for w in _WEIGHTS
        for k in _K_VALUES
    ]
    points = summarize_level3_by_weight_and_k(rows, k_values=_K_VALUES, weight_values=_WEIGHTS)
    assert len(points) == len(_WEIGHTS) * len(_K_VALUES)
    seen = {(p.bm25_weight, p.k) for p in points}
    assert seen == {(w, k) for w in _WEIGHTS for k in _K_VALUES}
    assert all(p.score.mean == pytest.approx(0.5) for p in points)


def test_summarize_level3_by_weight_and_k_pools_across_level1_and_level2() -> None:
    """Level 3 doesn't vary L1/L2 -- a query's score at one (weight, k)
    pools across whichever L1/L2 rows it has (same convention as Level 1/2
    pooling over bm25_weight)."""
    rows = [
        PerQueryResult(
            query_id="q1",
            patient_id_or_case_id="rec-1",
            experiment_id="exp-1",
            level1_condition="present_only",
            level2_condition="raw",
            level3_condition="bm25_sapbert",
            k=2,
            bm25_weight=1.0,
            query_text="ignored",
            concept_enriched_query="ignored",
            retrieved_ids=["c1"],
            relevant_ids=["c1"],
            first_relevant_rank=1,
            recall_at_k=1.0,
            reciprocal_rank_at_k=1.0,
        ),
        PerQueryResult(
            query_id="q1",
            patient_id_or_case_id="rec-1",
            experiment_id="exp-1",
            level1_condition="all_assessed",
            level2_condition="enriched",
            level3_condition="bm25_sapbert",
            k=2,
            bm25_weight=1.0,
            query_text="ignored",
            concept_enriched_query="ignored",
            retrieved_ids=["c1"],
            relevant_ids=["c1"],
            first_relevant_rank=1,
            recall_at_k=0.0,
            reciprocal_rank_at_k=0.0,
        ),
    ]
    points = summarize_level3_by_weight_and_k(rows, k_values=(2,), weight_values=(1.0,))
    assert len(points) == 1
    assert points[0].score.mean == pytest.approx((1.0 + 0.0) / 2)


# ── summarize_best_weight_vs_bm25 ─────────────────────────────────────────


def test_summarize_best_weight_vs_bm25_selects_the_highest_scoring_weight() -> None:
    rows = [
        _row_for_k(query_id="q1", k=4, bm25_weight=0.0, recall=0.2),
        _row_for_k(query_id="q1", k=4, bm25_weight=0.5, recall=0.9),  # the best
        _row_for_k(query_id="q1", k=4, bm25_weight=1.0, recall=0.3),
    ]
    result = summarize_best_weight_vs_bm25(rows, k=4, weight_values=(0.0, 0.5, 1.0))
    assert result.selected_weight == pytest.approx(0.5)
    assert result.selected_weight_score.mean == pytest.approx(0.9)
    assert result.bm25_score.mean == pytest.approx(0.3)
    assert result.delta.mean_delta == pytest.approx(0.6)  # 0.9 - 0.3


def test_summarize_best_weight_vs_bm25_zero_delta_when_bm25_is_already_best() -> None:
    rows = [
        _row_for_k(query_id="q1", k=4, bm25_weight=0.0, recall=0.2),
        _row_for_k(query_id="q1", k=4, bm25_weight=1.0, recall=0.9),  # bm25 itself is the best
    ]
    result = summarize_best_weight_vs_bm25(rows, k=4, weight_values=(0.0, 1.0))
    assert result.selected_weight == pytest.approx(1.0)
    assert result.delta.mean_delta == pytest.approx(0.0)


def _gold_row(query_id: str, gold: list[str], *, weight: float = 1.0, recall: float = 0.0):  # noqa: ANN202
    row = _row(
        query_id=query_id, level1="all_assessed", level2="raw", bm25_weight=weight, recall=recall
    )
    return PerQueryResult(**{**row.to_json_dict(), "relevant_ids": gold})


def test_popularity_baseline_is_leave_one_out() -> None:
    """DEVIATIONS.md #272: q1's own gold never counts toward its ranking. "a"
    is in all three gold sets, "b" only in q1's, so for q1 the ranking is
    a (2 others), c (1 other); b drops out (0 others)."""
    rows = [
        _gold_row("q1", ["a", "b"]),
        _gold_row("q2", ["a", "c"]),
        _gold_row("q3", ["a", "c"]),
    ]
    scores = popularity_baseline_scores(rows, k=1)
    assert scores["hit"] == {"q1": 1.0, "q2": 1.0, "q3": 1.0}  # "a" first for all
    assert scores["recall"]["q1"] == pytest.approx(0.5)
    assert scores["mrr"]["q2"] == pytest.approx(1.0)


def test_popularity_baseline_misses_a_query_whose_gold_nobody_else_cites() -> None:
    rows = [_gold_row("q1", ["x"]), _gold_row("q2", ["a"]), _gold_row("q3", ["a"])]
    scores = popularity_baseline_scores(rows, k=1)
    assert scores["hit"]["q1"] == 0.0
    assert scores["mrr"]["q1"] == 0.0


def test_popularity_cutoff_matches_a_full_resort() -> None:
    """The top-(k + |gold|) shortcut gives the same scores as re-sorting
    every chunk."""
    import random  # noqa: PLC0415

    from app.eval.metrics import mrr  # noqa: PLC0415

    rnd = random.Random(7)
    chunks = [f"c{i:02d}" for i in range(30)]
    gold = {f"q{i}": rnd.sample(chunks, rnd.randint(1, 6)) for i in range(40)}
    rows = [_gold_row(q, g) for q, g in gold.items()]
    for k in (1, 3, 8):
        got = popularity_baseline_scores(rows, k=k)["mrr"]
        for q, g in gold.items():
            others = [c for q2, g2 in gold.items() if q2 != q for c in g2]
            counts = {c: others.count(c) for c in set(others)}
            full = sorted(counts, key=lambda c: (-counts[c], c))
            assert got[q] == pytest.approx(mrr(full[:k], set(g)))


def test_summarize_popularity_baseline_compares_every_arm() -> None:
    rows = []
    for q, gold in (("q1", ["a"]), ("q2", ["a"]), ("q3", ["a"])):
        for level1 in ("present_only", "all_assessed"):
            for level2 in ("raw", "enriched"):
                for w in _WEIGHTS:
                    row = _row(query_id=q, level1=level1, level2=level2, bm25_weight=w, recall=0.25)
                    rows.append(PerQueryResult(**{**row.to_json_dict(), "relevant_ids": gold}))
    summary = summarize_popularity_baseline(rows, k=_K, weight_values=_WEIGHTS)
    assert summary.baseline["recall"].mean == pytest.approx(1.0)  # "a" ranked first
    assert set(summary.arm_vs_baseline) == {"recall", "mrr", "hit"}
    assert len(summary.arm_vs_baseline["recall"]) == 4 * len(_WEIGHTS)
    assert all(
        d.mean_delta == pytest.approx(-0.75) and d.label_b == "popularity"
        for d in summary.arm_vs_baseline["recall"]
    )


def _all_arm_rows(gold_by_query: dict[str, list[str]], ks: tuple[int, ...]) -> list:  # noqa: ANN401
    rows = []
    for q, gold in gold_by_query.items():
        for k in ks:
            for level1 in ("present_only", "all_assessed"):
                for level2 in ("raw", "enriched"):
                    for w in _WEIGHTS:
                        row = _row(
                            query_id=q, level1=level1, level2=level2, bm25_weight=w, recall=0.25
                        )
                        rows.append(
                            PerQueryResult(**{**row.to_json_dict(), "k": k, "relevant_ids": gold})
                        )
    return rows


def test_popularity_baseline_by_k_reports_each_k_separately() -> None:
    """DEVIATIONS.md #278: one PopularityBaseline per K, each scored at its own K."""
    rows = _all_arm_rows({"q1": ["a"], "q2": ["a"], "q3": ["a"]}, ks=(2, 4))
    out = summarize_popularity_baseline_by_k(rows, k_values=(2, 4), weight_values=_WEIGHTS)
    assert [pb.k for pb in out] == [2, 4]
    assert all(len(pb.arm_vs_baseline["mrr"]) == 4 * len(_WEIGHTS) for pb in out)


def test_popularity_report_rows_flattens_every_k_metric_and_arm() -> None:
    rows = _all_arm_rows({"q1": ["a"], "q2": ["a"], "q3": ["a"]}, ks=(2, 4))
    records = popularity_report_rows(
        summarize_popularity_baseline_by_k(rows, k_values=(2, 4), weight_values=_WEIGHTS)
    )
    assert len(records) == 2 * 3 * 4 * len(_WEIGHTS)
    first = next(r for r in records if r["metric"] == "recall")
    assert first["level1"] == "present_only" and first["level2"] == "raw"
    assert first["bm25_weight"] == pytest.approx(_WEIGHTS[0])
    assert first["baseline_mean"] == pytest.approx(1.0)
    assert first["arm_mean"] == pytest.approx(0.25)
    assert first["delta"] == pytest.approx(-0.75)
    assert first["vs_baseline"] in {"below", "not significant"}


def test_popularity_scores_from_gold_matches_the_row_based_version() -> None:
    rows = [_gold_row("q1", ["a", "b"]), _gold_row("q2", ["a", "c"]), _gold_row("q3", ["a", "c"])]
    gold = {"q1": {"a", "b"}, "q2": {"a", "c"}, "q3": {"a", "c"}}
    assert popularity_scores_from_gold(gold, k=2) == popularity_baseline_scores(rows, k=2)
