"""Triage-agreement review question (ARCH-046, PRD-114; DEVIATIONS.md #299).

Every review-queue item in the `assessment` query area ends with one extra
question: the record's own `triage_category`, the operator-attested
mortality for that category (`data/triage_mortality.yaml`), and a 1-5
agreement score (1 = strongly disagree, 5 = strongly agree). No other area
shows, asks for or accepts it.

The score is stored as a `rubric_rating` row under its own domain code,
`triage_agreement`, so per-domain IRR (ARCH §14.4) and archival cover it with
no new table. It is not one of the 11 rubric domains (`RUBRIC_DOMAINS`): it
rates the record's triage label, not the answer.

The category is never used for guideline retrieval: no query area lists it
as a record fact, and the single-topic question generator does not render
it. The mortality figure is a local statistic, not guideline content; it
reaches reviewers only through this module and never enters an answer.

The narrative is fixed template text, not model output. The loader fails
closed (`TriageMortalityNotAttested`) on a missing file, any missing or
`TODO_CONFIRM` field, or a category set other than `TRIAGE_CATEGORIES`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from app.rubric.domains import RubricDomain

TRIAGE_REVIEW_AREA = "assessment"
TRIAGE_AGREEMENT_DOMAIN_CODE = "triage_agreement"
# Highest to lowest risk; must match the dataset's values exactly.
TRIAGE_CATEGORIES = ("Critical", "Substantial", "Moderate", "Low", "Minimal")

# Ordinal 12, after the 11 rubric domains. Anchors 2-4 are the standard
# Likert midpoints; the operator gave only the two ends (DEVIATIONS.md #299).
TRIAGE_AGREEMENT_DOMAIN = RubricDomain(
    TRIAGE_AGREEMENT_DOMAIN_CODE,
    12,
    "Agreement with triage category",
    "How strongly do you agree with this newborn's triage category, given the "
    "clinical features shown?",
    ("Strongly disagree", "Disagree", "Neither agree nor disagree", "Agree", "Strongly agree"),
    required=True,
)

_NARRATIVE = (
    "This newborn was assigned triage category {category} based on the information "
    "available at the time of admission. In {citation}, newborns in this category had a "
    "mortality rate of {mortality}. How strongly do you agree with this triage category, "
    "given the clinical features shown?"
)
_PLACEHOLDER = "TODO_CONFIRM"


class TriageMortalityNotAttested(ValueError):
    pass


@dataclass(frozen=True)
class TriageMortality:
    attested_by: str
    attested_role: str
    attested_date: str
    citation: str
    url: str
    mortality: dict[str, str]
    sha256: str


def _text(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value.strip() or _PLACEHOLDER in value:
        raise TriageMortalityNotAttested(f"{what} is missing or still a placeholder")
    return value.strip()


def load_triage_mortality(path: str | Path) -> TriageMortality:
    """Load and validate the attested mortality table; fails closed."""
    p = Path(path)
    if not p.exists():
        raise TriageMortalityNotAttested(f"{p} not found")
    raw_bytes = p.read_bytes()
    data = yaml.safe_load(raw_bytes) or {}
    source = data.get("source") or {}
    categories = data.get("categories") or {}
    if set(categories) != set(TRIAGE_CATEGORIES):
        raise TriageMortalityNotAttested(f"categories must be exactly {list(TRIAGE_CATEGORIES)}")
    return TriageMortality(
        attested_by=_text(data.get("attested_by"), "attested_by"),
        attested_role=_text(data.get("attested_role"), "attested_role"),
        attested_date=_text(str(data.get("attested_date") or ""), "attested_date"),
        citation=_text(source.get("citation"), "source.citation"),
        url=_text(source.get("url"), "source.url"),
        mortality={
            name: _text((categories[name] or {}).get("mortality"), f"{name} mortality")
            for name in TRIAGE_CATEGORIES
        },
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


def record_triage_category(record: dict) -> str | None:
    """The record's triage category if it is one of `TRIAGE_CATEGORIES`."""
    value = (record.get("encounter") or {}).get("triage_category")
    return value if value in TRIAGE_CATEGORIES else None


def triage_review_applies(area: str | None, category: str | None) -> bool:
    """Whether an item asks the triage question: assessment area, known category."""
    return area == TRIAGE_REVIEW_AREA and category in TRIAGE_CATEGORIES


def triage_narrative(category: str, table: TriageMortality) -> str:
    return _NARRATIVE.format(
        category=category, citation=table.citation, mortality=table.mortality[category]
    )


def triage_review_payload(category: str, table: TriageMortality) -> dict[str, Any]:
    """What the review UI shows for the triage question."""
    return {
        "category": category,
        "mortality": table.mortality[category],
        "narrative": triage_narrative(category, table),
        "source_citation": table.citation,
        "source_url": table.url,
        "domain": TRIAGE_AGREEMENT_DOMAIN.as_reference(),
    }
