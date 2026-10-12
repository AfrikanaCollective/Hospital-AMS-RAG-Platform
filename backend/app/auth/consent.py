"""Reviewer informed-consent text (DEVIATIONS.md #282).

Loaded from `data/reviewer_consent.yaml` (`REVIEWER_CONSENT_PATH`). Fails
closed, like `data/query_areas.yaml`: the text is refused while any
attestation field is empty or `TODO_CONFIRM`, or any shown text still holds
a `[CONFIRM` marker. Registration is impossible until then. The sha256 of
the file is recorded with every consent so a later edit is distinguishable.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

_PLACEHOLDER = "TODO_CONFIRM"
_UNRESOLVED_MARKER = "[CONFIRM"
_ATTESTATION_FIELDS = ("authored_by", "authored_role", "authored_date", "ethics_approved_version")


class ConsentNotAttested(ValueError):
    """The consent file is missing, malformed, unattested, or unfinished."""


@dataclass(frozen=True)
class ConsentSection:
    heading: str
    body: str


@dataclass(frozen=True)
class ConsentText:
    version: str
    sha256: str
    title: str
    sections: tuple[ConsentSection, ...]
    agreement: str
    ethics_approved_version: str


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConsentNotAttested(f"{name} is missing")
    return value.strip()


def load_consent(path: str | Path) -> ConsentText:
    p = Path(path)
    try:
        raw = p.read_bytes()
        data = yaml.safe_load(raw)
    except (OSError, yaml.YAMLError) as exc:
        raise ConsentNotAttested(f"cannot read {p}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConsentNotAttested(f"{p} is not a mapping")
    for name in _ATTESTATION_FIELDS:
        value = data.get(name)
        if not isinstance(value, str) or not value.strip() or value.strip() == _PLACEHOLDER:
            raise ConsentNotAttested(f"{name} is not filled in by the operator")
    sections_raw = data.get("sections")
    if not isinstance(sections_raw, list) or not sections_raw:
        raise ConsentNotAttested("sections is missing")
    sections = tuple(
        ConsentSection(
            heading=_text(s.get("heading") if isinstance(s, dict) else None, "section heading"),
            body=_text(s.get("body") if isinstance(s, dict) else None, "section body"),
        )
        for s in sections_raw
    )
    consent = ConsentText(
        version=_text(data.get("version"), "version"),
        sha256=hashlib.sha256(raw).hexdigest(),
        title=_text(data.get("title"), "title"),
        sections=sections,
        agreement=_text(data.get("agreement"), "agreement"),
        ethics_approved_version=data["ethics_approved_version"].strip(),
    )
    shown = [consent.title, consent.agreement, *(f"{s.heading} {s.body}" for s in sections)]
    if any(_UNRESOLVED_MARKER in t for t in shown):
        raise ConsentNotAttested("the text still contains [CONFIRM ...] items")
    return consent
