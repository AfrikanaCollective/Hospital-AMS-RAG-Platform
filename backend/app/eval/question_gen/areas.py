"""Four-area guideline questions (DEVIATIONS.md #258; operator-attested
`data/query_areas.yaml`).

The single fixed topic (`app.eval.auto_seed._TOPIC`) pulled every question
toward the same broad sections. Each record now yields one question per
area -- assessment, investigations, severity / risk classification,
antibiotic course -- each with:

- the area's attested `opening` in the existing question template ("What
  does the guideline recommend about <opening> based only on the content
  provided below:"), so `deterministic.extract_topic` still recovers it;
- only the record facts the area lists (`record_facts`), rendered from the
  record's own field values: nothing is invented, assessed-absent findings
  are stated only when actually assessed (ARCH-039 tri-state), and
  medications/interventions never appear (DEVIATIONS.md #155);
- the area's `per_guideline_cap` for per-guideline retrieval (#252).

Every opening asks what the guidelines say; none asks the system to place
this baby in a category or choose its treatment (CLAUDE.md §3 rules 2, 4).
The age band and weight renderings map recorded values onto the dose
tables' own labels for retrieval only; they never select a dose.

The file is attestation-gated like `data/clinical_concepts.yaml`: the loader
fails closed (`QueryAreasNotAttested`) on any missing or placeholder
attestation field, unknown record fact, or malformed spec.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from app.eval.question_gen.deterministic import (
    _QUESTION_PREFIX,
    _QUESTION_SUFFIX,
    _examination_findings_lines,
    _maternal_risk_factors_line,
    _vitals_line,
)
from app.records.concepts import Concept, render_value

AREA_ORDER = ("assessment", "investigations", "severity_classification", "antibiotic_course")
RECORD_FACTS = frozenset(
    {
        "age_days",
        "age_days_with_band",
        "gestational_age",
        "care_setting",
        "presenting_complaint",
        "weight_kg",
        "maternal_risk_factors",
        "vitals",
        "examination_findings",
    }
)
_PLACEHOLDER = "TODO_CONFIRM"


class QueryAreasNotAttested(ValueError):
    pass


@dataclass(frozen=True)
class AgeBand:
    from_day: int
    to_day: int
    text: str


@dataclass(frozen=True)
class WeightSpec:
    prefer_field: str
    fallback_field: str
    scale: float
    renderings: tuple[tuple[float | None, int | None], ...]


@dataclass(frozen=True)
class QueryArea:
    name: str
    opening: str
    record_facts: tuple[str, ...]
    per_guideline_cap: int
    age_bands: tuple[AgeBand, ...] = ()
    weight: WeightSpec | None = None


@dataclass(frozen=True)
class QueryAreas:
    authored_by: str
    authored_role: str
    authored_date: str
    areas: tuple[QueryArea, ...]
    sha256: str


def _text(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value.strip() or _PLACEHOLDER in value:
        raise QueryAreasNotAttested(f"{what} is missing or still a placeholder")
    return value.strip()


def _positive_int(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise QueryAreasNotAttested(f"{what} must be a positive integer")
    return value


def _age_bands(raw: Any, area: str) -> tuple[AgeBand, ...]:
    if not isinstance(raw, list) or not raw:
        raise QueryAreasNotAttested(f"area {area!r}: age_days_with_band needs age_bands")
    bands = []
    for b in raw:
        lo = b.get("from_day") if isinstance(b, dict) else None
        hi = b.get("to_day") if isinstance(b, dict) else None
        if isinstance(lo, bool) or isinstance(hi, bool) or not isinstance(lo, int):
            raise QueryAreasNotAttested(f"area {area!r}: malformed age band {b!r}")
        if not isinstance(hi, int) or lo < 0 or hi < lo:
            raise QueryAreasNotAttested(f"area {area!r}: malformed age band {b!r}")
        bands.append(AgeBand(lo, hi, _text(b.get("text"), f"area {area!r} age band text")))
    return tuple(bands)


def _weight(raw: Any, area: str) -> WeightSpec:
    if not isinstance(raw, dict):
        raise QueryAreasNotAttested(f"area {area!r}: weight_kg needs a weight spec")
    scale = raw.get("scale")
    if isinstance(scale, bool) or not isinstance(scale, int | float) or scale <= 0:
        raise QueryAreasNotAttested(f"area {area!r}: weight scale must be positive")
    renderings: list[tuple[float | None, int | None]] = []
    for r in raw.get("renderings") or []:
        step, decimals = (r or {}).get("round_to"), (r or {}).get("decimals")
        if step is not None and (isinstance(step, bool) or not isinstance(step, int | float)):
            raise QueryAreasNotAttested(f"area {area!r}: bad weight rendering {r!r}")
        if step is not None and step <= 0:
            raise QueryAreasNotAttested(f"area {area!r}: bad weight rendering {r!r}")
        if decimals is not None and (
            isinstance(decimals, bool) or not isinstance(decimals, int) or decimals < 0
        ):
            raise QueryAreasNotAttested(f"area {area!r}: bad weight rendering {r!r}")
        renderings.append((float(step) if step is not None else None, decimals))
    if not renderings:
        raise QueryAreasNotAttested(f"area {area!r}: weight needs at least one rendering")
    return WeightSpec(
        prefer_field=_text(raw.get("prefer_field"), f"area {area!r} weight prefer_field"),
        fallback_field=_text(raw.get("fallback_field"), f"area {area!r} weight fallback_field"),
        scale=float(scale),
        renderings=tuple(renderings),
    )


def load_query_areas(path: str | Path) -> QueryAreas:
    """Load and validate the attested query areas; fails closed."""
    p = Path(path)
    if not p.exists():
        raise QueryAreasNotAttested(f"{p} not found")
    raw_bytes = p.read_bytes()
    data = yaml.safe_load(raw_bytes) or {}
    authored_by = _text(data.get("authored_by"), "authored_by")
    authored_role = _text(data.get("authored_role"), "authored_role")
    authored_date = _text(str(data.get("authored_date") or ""), "authored_date")
    raw_areas = data.get("areas") or {}
    if set(raw_areas) != set(AREA_ORDER):
        raise QueryAreasNotAttested(f"areas must be exactly {list(AREA_ORDER)}")
    areas = []
    for name in AREA_ORDER:
        spec = raw_areas[name] or {}
        _text(spec.get("attested"), f"area {name!r} attested")
        facts = tuple(spec.get("record_facts") or ())
        unknown = [f for f in facts if f not in RECORD_FACTS]
        if not facts or unknown:
            raise QueryAreasNotAttested(f"area {name!r}: unknown or empty record_facts {unknown}")
        areas.append(
            QueryArea(
                name=name,
                opening=" ".join(_text(spec.get("opening"), f"area {name!r} opening").split()),
                record_facts=facts,
                per_guideline_cap=_positive_int(
                    spec.get("per_guideline_cap"), f"area {name!r} per_guideline_cap"
                ),
                age_bands=_age_bands(spec.get("age_bands"), name)
                if "age_days_with_band" in facts
                else (),
                weight=_weight(spec.get("weight"), name) if "weight_kg" in facts else None,
            )
        )
    return QueryAreas(
        authored_by=authored_by,
        authored_role=authored_role,
        authored_date=authored_date,
        areas=tuple(areas),
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


def _field(record: dict, dotted: str) -> Any:
    section, _, key = dotted.partition(".")
    if section == "vitals":
        first = (record.get("vitals") or [{}])[0] or {}
        return first.get(key)
    return (record.get(section) or {}).get(key)


def _weight_line(record: dict, spec: WeightSpec) -> str | None:
    current = _field(record, spec.prefer_field)
    birth = _field(record, spec.fallback_field)
    grams, label = (current, "current weight") if current is not None else (birth, "birth weight")
    if isinstance(grams, bool) or not isinstance(grams, int | float):
        return None
    concept = Concept(
        name="weight",
        field=spec.prefer_field,
        operator="value",
        source="data/query_areas.yaml",
        scale=spec.scale,
        unit=None,
        renderings=spec.renderings,
    )
    note = label
    if label == "current weight" and isinstance(birth, int | float) and not isinstance(birth, bool):
        note += f"; birth weight {birth * spec.scale:g} kg"
    return f"Weight (kg): {', '.join(render_value(concept, grams))} ({note})"


def _patient_sentence(record: dict, area: QueryArea) -> str | None:
    encounter = record.get("encounter") or {}
    parts: list[str] = []
    dol = encounter.get("day_of_life")
    if "age_days_with_band" in area.record_facts and dol is not None:
        band = next((b.text for b in area.age_bands if b.from_day <= dol <= b.to_day), None)
        parts.append(f"is aged {dol} days" + (f" ({band})" if band else ""))
    elif "age_days" in area.record_facts and dol is not None:
        parts.append(f"is {dol} days old")
    ga = encounter.get("gestational_age_weeks")
    if "gestational_age" in area.record_facts and ga is not None:
        parts.append(f"born at {ga} weeks gestation")
    setting = encounter.get("care_setting")
    if "care_setting" in area.record_facts and setting:
        parts.append(f"currently in {setting}")
    complaint = encounter.get("presenting_complaint")
    if "presenting_complaint" in area.record_facts and complaint:
        parts.append(f"presenting with {complaint}")
    return "The newborn patient " + ", ".join(parts) + "." if parts else None


def build_area_question(record: dict, area: QueryArea, *, include_absent: bool = True) -> str:
    """One area's question for one record: the opening in the standard
    template, then only the facts the area lists. `include_absent=False`
    drops assessed-absent findings (the ablation's present-only variant)."""
    lines = [f"{_QUESTION_PREFIX}{area.opening}{_QUESTION_SUFFIX}", ""]
    if sentence := _patient_sentence(record, area):
        lines.append(sentence)
    weight = (
        _weight_line(record, area.weight)
        if "weight_kg" in area.record_facts and area.weight is not None
        else None
    )
    if weight:
        lines.append(weight)
    maternal = (
        _maternal_risk_factors_line(record, include_absent=include_absent)
        if "maternal_risk_factors" in area.record_facts
        else None
    )
    if maternal:
        lines.append(maternal)
    if "vitals" in area.record_facts and (vitals := _vitals_line(record)):
        lines.append(vitals)
    if "examination_findings" in area.record_facts:
        lines.extend(_examination_findings_lines(record, include_absent=include_absent))
    return "\n".join(lines)
