"""Deterministic wording / scope filter (ARCH §8.3 step 4, ARCH-037; SCOPE-1.2, PRD-088).

Second line of defense behind the prompt templates. Blocks:
- second-person imperatives / "you should" / "recommend that you",
- dosing or therapy specifics not present in a cited quote,
- text that reads as a next-step plan for a specific patient.
A trip => safety_filter escalation. Safe to extend the marker lists.

The marker lists here are usable now and are covered by tests. Phase 3 adds
the dosing/therapy-beyond-source check and wires this into the verifier
(`app.grounding.verifier.verify`).
"""

from __future__ import annotations

import re

DIRECTIVE_PATTERNS: tuple[str, ...] = (
    r"\byou should\b",
    r"\byou must\b",
    r"\bwe recommend (that )?you\b",
    r"\bi recommend\b",
    r"\bstart (the patient on|them on)\b",
    r"\badminister\b .*\bto (this|the) patient\b",
    r"\bthe next step (for (this|the) patient )?is\b",
    r"\bmy advice\b",
)

_COMPILED = [re.compile(p, re.IGNORECASE) for p in DIRECTIVE_PATTERNS]

# A dose/frequency-shaped numeric token: "5mg", "2.5 ml/kg", "10 units", "8 hourly".
_DOSING_TOKEN_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s?(?:mg|mcg|g|ml|l|units?|iu|mmol|mcg/kg|mg/kg|ml/kg)\b"
    r"(?:\s?(?:/\s?kg|/\s?day|/\s?dose))?",
    re.IGNORECASE,
)


def has_directive_phrasing(text: str) -> bool:
    return any(p.search(text) for p in _COMPILED)


def _dosing_beyond_source(text: str, cited_quotes: list[str]) -> bool:
    """A dosing/frequency figure in `text` that does not appear verbatim in
    any cited quote is a fabricated specific — even if it happens to be
    clinically plausible, it isn't in the retrieved source (ARCH §8.3 step 4).
    """
    tokens = {_squash(m.group(0)) for m in _DOSING_TOKEN_RE.finditer(text)}
    if not tokens:
        return False
    quote_text = _squash(" ".join(cited_quotes))
    return any(token not in quote_text for token in tokens)


def _squash(text: str) -> str:
    """Lower-case with ALL whitespace removed -- including the non-breaking
    and thin spaces guideline PDFs put between a number and its unit -- so
    "5 mg/kg" in an answer matches "5\u00a0mg/kg" in the source. The digits
    and unit must still match exactly (DEVIATIONS.md #267)."""
    return re.sub(r"\s+", "", text).lower()


def scan_segment(text: str, *, cited_quotes: list[str]) -> list[str]:
    """Return a list of violation reason codes for one segment. Empty => clean."""
    reasons: list[str] = []
    if has_directive_phrasing(text):
        reasons.append("directive_phrasing")
    if _dosing_beyond_source(text, cited_quotes):
        reasons.append("dosing_beyond_source")
    return reasons
