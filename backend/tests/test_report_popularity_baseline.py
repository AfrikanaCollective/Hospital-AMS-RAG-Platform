"""Every arm reported against the question-blind baseline at every K
(PRD-112 / ARCH-043; DEVIATIONS.md #278). Synthetic rows, no DB/Qdrant."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.eval.unified_ablation.per_query import PerQueryResult
from scripts.report_popularity_baseline import REPORT_NAME, write_popularity_report

_WEIGHTS = (0.0, 1.0)


def _rows() -> list[PerQueryResult]:
    rows = []
    for q, gold in (("q1", ["a"]), ("q2", ["a"]), ("q3", ["b"])):
        for k in (2, 4):
            for level1 in ("present_only", "all_assessed"):
                for level2 in ("raw", "enriched"):
                    for w in _WEIGHTS:
                        rows.append(
                            PerQueryResult(
                                query_id=q,
                                patient_id_or_case_id="p",
                                experiment_id="e",
                                level1_condition=level1,
                                level2_condition=level2,
                                level3_condition="bm25_sapbert",
                                k=k,
                                bm25_weight=w,
                                query_text="",
                                concept_enriched_query="",
                                retrieved_ids=["a", "b"],
                                relevant_ids=gold,
                                recall_at_k=0.5,
                                reciprocal_rank_at_k=0.5,
                                first_relevant_rank=2,
                            )
                        )
    return rows


def test_writes_csv_and_adds_by_k_key_without_touching_others(tmp_path: Path) -> None:
    (tmp_path / "statistical_summary.json").write_text(json.dumps({"k": 12, "level1": "kept"}))
    out = write_popularity_report(tmp_path, _rows(), k_values=(2, 4), weights=_WEIGHTS)

    assert out == tmp_path / REPORT_NAME
    with out.open() as fh:
        records = list(csv.DictReader(fh))
    assert len(records) == 2 * 3 * 4 * len(_WEIGHTS)  # K x metric x arm
    assert {r["k"] for r in records} == {"2", "4"}
    assert {r["vs_baseline"] for r in records} <= {"above", "below", "not significant"}

    summary = json.loads((tmp_path / "statistical_summary.json").read_text())
    assert summary["k"] == 12 and summary["level1"] == "kept"
    assert [b["k"] for b in summary["popularity_baseline_by_k"]] == [2, 4]
