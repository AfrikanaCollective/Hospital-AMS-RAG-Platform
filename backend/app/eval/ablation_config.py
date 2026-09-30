"""Explicit configuration representation for the unified hierarchical
ablation (PRD-112 / ARCH-043; UNIFIED-ABLATION-PROPOSAL.md §3.4, §3.6, §12).

**Restructured 2026-09-23 (operator request, DEVIATIONS.md #201)**: Level 3
is now a single continuous weighted-rank-fusion sweep between BM25 and
SapBERT only — `w_BM25 ∈ {0.0, 0.2, ..., 1.0}` (6 points since 2026-09-25,
DEVIATIONS.md #207; 11 points at 0.1 before that) — not a set of
distinct named arms. MedCPT and RRF fusion (proposal §11, Option B) are
both dropped entirely, not merely excluded from the default sweep; that
code has been removed, not deprecated in place. `AblationArm.level3` is
kept as a single-member `Literal` (`"bm25_sapbert"`) rather than deleted
outright, so `AblationArm`/`ALL_ARMS`/the per-query `level3_condition`
field all stay structurally the same shape as before — a leaf
configuration is still `(level1, level2, level3)`, level3 just no longer
varies.

`ALL_ARMS` is 4 leaf configurations (2 Level-1 x 2 Level-2 x 1 Level-3),
each swept across all 6 `bm25_weight` values and the full `k` grid — 24
(level1, level2, bm25_weight) combinations total, `len(k_values())` rows
each.

K and `bm25_weight` are read from `app.config.Settings` (never hardcoded)
— shared by this module and `app.eval.unified_ablation` only; the three
pre-existing ablation modules (`model_ablation`, `retrieval_tuning`,
`orchestration_ablation`) keep their own independent K/alpha/MRR_K
constants for now (proposal §8: not silently migrated in this phase).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

from app.config import get_settings

Level1Condition = Literal["present_only", "all_assessed"]
Level2Condition = Literal["enriched", "raw"]
Level3Condition = Literal["bm25_sapbert"]

LEVEL1_CONDITIONS: tuple[Level1Condition, ...] = get_args(Level1Condition)
LEVEL2_CONDITIONS: tuple[Level2Condition, ...] = get_args(Level2Condition)
LEVEL3_CONDITIONS: tuple[Level3Condition, ...] = get_args(Level3Condition)


@dataclass(frozen=True)
class AblationArm:
    """One leaf configuration. `experiment_id` per `AblationArm` is
    deliberately NOT part of this dataclass -- an arm is a fixed point in
    the hierarchy's own design space, independent of which run it was
    evaluated in; `experiment_id` belongs to the run (§3.8), not the arm."""

    level1: Level1Condition
    level2: Level2Condition
    level3: Level3Condition

    @property
    def label(self) -> str:
        """A stable, human-readable identifier for reports/logs -- e.g.
        `"L1-present_only__L2-enriched__L3-bm25_sapbert"`."""
        return f"L1-{self.level1}__L2-{self.level2}__L3-{self.level3}"


ALL_ARMS: tuple[AblationArm, ...] = tuple(
    AblationArm(level1=l1, level2=l2, level3=l3)
    for l1 in LEVEL1_CONDITIONS
    for l2 in LEVEL2_CONDITIONS
    for l3 in LEVEL3_CONDITIONS
)  # 4 = 2 x 2 x 1, generated -- never hand-enumerated (proposal §3.4)


def k_values() -> tuple[int, ...]:
    return get_settings().ablation_k_values_tuple


def bm25_weight_values() -> tuple[float, ...]:
    return get_settings().ablation_bm25_weight_values_tuple


FUSION_METHODS = ("minmax", "rrf")


def fusion() -> str:
    """`minmax` or `rrf` (DEVIATIONS.md #227); anything else is a config
    error, not a silent fallback."""
    method = get_settings().ablation_fusion
    if method not in FUSION_METHODS:
        raise ValueError(f"ABLATION_FUSION must be one of {FUSION_METHODS}, got {method!r}")
    return method


def rrf_k() -> int:
    return get_settings().ablation_rrf_k


def mrr_k() -> int:
    """The single headline `k` used for Level 1/2's point-with-CI summary
    panels (both the primary recall@k and the secondary MRR@k) -- Level
    3's own primary metric is reported as a curve across the FULL `k_values()`
    range instead (proposal, "K should be allowed to range... not hard
    coded"), not collapsed to one point. Name kept as `mrr_k`/`ABLATION_MRR_K`
    (not renamed to something metric-neutral) -- a deliberate minimal-diff
    choice; it now doubles as the recall@k headline k too."""
    return get_settings().ablation_mrr_k
