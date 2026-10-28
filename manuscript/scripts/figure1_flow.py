"""Figure 1: flow of records through the study (TRIPOD-LLM 16a).

Counts come from the aggregate outputs where they exist
(`manuscript/derived/table1.json` for set sizes and triage splits, the
ablation run's `configuration.json` for questions per area). The generation
steps (844 first run, 456 untried, 320 near-duplicates, 136 top-up) are
recorded only in the run logs and DEVIATIONS.md #303-#305, and the
source-extract counts (44,626 / 44,389 / 7,994) were supplied by the PI
(2026-10-27), so they are constants here, checked against the totals.
Writes PNG (300 dpi) and PDF.

    taskset -c 0-7,12-31 python3 -I manuscript/scripts/figure1_flow.py \
      manuscript/derived/table1.json \
      backend/results/ablation/20261026T075410Z-e78b8cf6/configuration.json \
      manuscript/figures/figure1_flow
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

# Source extract (PI, 2026-10-27).
SOURCE = 44_626  # admissions with >=1 pSBI sign, 8 hospitals, 2021-2024
COMPLETE = 44_389  # after excluding >30% missing across 14 features
UNIQUE = 7_994  # one admission per unique combination of the 14 features
PER_CATEGORY = 300

# Generation steps (DEVIATIONS.md #303-#305; run logs 20261015, 20261016).
FIRST_RUN = 844
UNTRIED = 456
NEAR_DUPLICATES = 320
TOP_UP = 136

INK = "#1f1f1f"
EDGE = "#8a8a8a"
FILL = "#ffffff"
FILL_KEY = "#efefef"
FONT = 7.2

CATEGORIES = ["Critical", "Substantial", "Moderate", "Low", "Minimal"]
AREAS = [
    ("assessment", "Assessment"),
    ("investigations", "Investigations"),
    ("severity_classification", "Severity classification"),
    ("antibiotic_course", "Antibiotic course"),
]

# Column centres and widths (axes fraction).
LX, LW = 0.18, 0.33  # clinician evaluation
RX, RW = 0.555, 0.33  # retrieval ablation
EX, EW = 0.865, 0.25  # exclusions


def _n(n: int) -> str:
    return f"{n:,}"


def _box(ax, x, y, w, h, text, *, key=False):  # noqa: ANN001, PLR0913
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=0.01",
            linewidth=0.8,
            edgecolor=EDGE,
            facecolor=FILL_KEY if key else FILL,
        )
    )
    ax.text(
        x, y, text, ha="center", va="center", fontsize=FONT, color=INK, linespacing=1.4
    )


def _arrow(ax, x0, y0, x1, y1):  # noqa: ANN001
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops={
            "arrowstyle": "-|>",
            "color": EDGE,
            "lw": 0.8,
            "shrinkA": 0,
            "shrinkB": 0,
        },
    )


def main(argv: list[str]) -> int:  # noqa: PLR0915
    sets = json.loads(Path(argv[1]).read_text(encoding="utf-8"))["sets"]
    config = json.loads(Path(argv[2]).read_text(encoding="utf-8"))
    out = Path(argv[3])

    n_all = sets["all_ingested"]["n_records"]
    n_review = sets["review_queue"]["n_records"]
    n_holdout = sets["holdout"]["n_records"]
    n_analysed = sets["holdout_analysed"]["n_records"]
    n_no_gold = sets["holdout_excluded_no_gold"]["n_records"]
    per_area = config["questions_per_area"]
    n_eligible = n_all - n_review

    # The logged generation steps must reconcile with the outputs.
    assert FIRST_RUN + UNTRIED == n_eligible
    assert NEAR_DUPLICATES + TOP_UP == UNTRIED
    assert FIRST_RUN + TOP_UP == n_holdout
    assert n_holdout - n_no_gold == n_analysed == sum(per_area.values())
    assert config["distinct_queries_evaluated"] == n_analysed
    assert len(CATEGORIES) * PER_CATEGORY == n_all <= UNIQUE <= COMPLETE <= SOURCE

    fig, ax = plt.subplots(figsize=(7.4, 9.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.31)
    ax.axis("off")

    # Source extract.
    top_x = (LX + RX) / 2
    for y, text in (
        (
            1.27,
            f"Newborn-unit admissions with ≥1 pSBI sign, n = {_n(SOURCE)}\n"
            "8 CIN hospitals, January 2021 to December 2024",
        ),
        (1.15, f"≤30% of 14 clinical features missing, n = {_n(COMPLETE)}"),
        (1.05, f"One per unique combination of the 14 features, n = {_n(UNIQUE)}"),
    ):
        _box(ax, top_x, y, 0.62, 0.07 if y > 1.2 else 0.05, text)
    _arrow(ax, top_x, 1.235, top_x, 1.175)
    _arrow(ax, top_x, 1.125, top_x, 1.075)
    _arrow(ax, top_x, 1.025, top_x, 0.98)
    for y, text in (
        (1.205, f"Excluded: >30% missing\nn = {_n(SOURCE - COMPLETE)}"),
        (1.0025, f"Not sampled\nn = {_n(UNIQUE - n_all)}"),
    ):
        _arrow(ax, top_x, y, EX - EW / 2, y)
        _box(ax, EX, y, EW, 0.05, text)

    _box(
        ax,
        top_x,
        0.935,
        0.62,
        0.09,
        f"Study records, n = {_n(n_all)}\n"
        f"{PER_CATEGORY} sampled at random per triage category (seed 20261013):\n"
        + ", ".join(CATEGORIES),
        key=True,
    )
    _arrow(ax, top_x, 0.89, top_x, 0.86)
    ax.plot([LX, RX], [0.86, 0.86], color=EDGE, lw=0.8)
    _arrow(ax, LX, 0.86, LX, 0.82)
    _arrow(ax, RX, 0.86, RX, 0.82)

    # Clinician evaluation (primary).
    _box(
        ax,
        LX,
        0.775,
        LW,
        0.09,
        f"Clinician evaluation sample\nn = {_n(n_review)} records\n40 per triage category",
        key=True,
    )
    _arrow(ax, LX, 0.73, LX, 0.68)
    _box(
        ax,
        LX,
        0.625,
        LW,
        0.11,
        f"Review items, n = {_n(4 * n_review)}\n"
        f"{_n(n_review)} records × 4 query areas:\n"
        "assessment, investigations,\nseverity classification,\nantibiotic course",
    )
    _arrow(ax, LX, 0.57, LX, 0.52)
    _box(
        ax,
        LX,
        0.465,
        LW,
        0.11,
        "Each item to be rated by\nthree independent clinicians\n[PLACEHOLDER: items rated,\nraters, completed ratings]",
    )

    # Retrieval ablation (secondary).
    _box(
        ax,
        RX,
        0.775,
        RW,
        0.09,
        f"Eligible for the retrieval ablation\nn = {_n(n_eligible)} records\none question per record",
        key=True,
    )
    _arrow(ax, RX, 0.73, RX, 0.68)
    _box(
        ax,
        RX,
        0.645,
        RW,
        0.07,
        f"First generation run\n{_n(FIRST_RUN)} records used; {_n(UNTRIED)} not yet tried",
    )
    _arrow(ax, RX, 0.61, RX, 0.56)
    _box(
        ax,
        RX,
        0.525,
        RW,
        0.07,
        f"Top-up run: {_n(UNTRIED)} records tried once\n{_n(TOP_UP)} added",
    )
    _arrow(ax, RX, 0.49, RX, 0.425)
    holdout_triage = sets["holdout"]["triage_category"]
    _box(
        ax,
        RX,
        0.37,
        RW,
        0.11,
        f"Holdout questions, n = {_n(n_holdout)}\n"
        + ",\n".join(
            ", ".join(f"{c} {holdout_triage[c]['n']}" for c in group)
            for group in (CATEGORIES[:2], CATEGORIES[2:4], CATEGORIES[4:])
        ),
    )
    _arrow(ax, RX, 0.315, RX, 0.20)
    _box(
        ax,
        RX,
        0.13,
        RW,
        0.14,
        f"Analysed, n = {_n(n_analysed)}\n"
        + "\n".join(f"{lab} {per_area[k]}" for k, lab in AREAS),
        key=True,
    )

    # Exclusions branch right from the ablation path.
    for y_line, y_box, text in (
        (
            0.525,
            0.525,
            f"Excluded: near-duplicate\nof a selected record\nn = {_n(NEAR_DUPLICATES)}",
        ),
        (
            0.255,
            0.255,
            f"Excluded: no gold set\n(answer cited no passage;\nall antibiotic course)\nn = {_n(n_no_gold)}",
        ),
    ):
        x_from = RX if y_line < 0.3 else RX + RW / 2
        _arrow(ax, x_from, y_line, EX - EW / 2, y_line)
        _box(ax, EX, y_box, EW, 0.09 if y_line > 0.3 else 0.10, text)

    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(
            out.with_suffix(suffix), dpi=300, bbox_inches="tight", facecolor="white"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
