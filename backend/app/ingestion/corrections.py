"""Attested text corrections — errata and OCR overrides (ARCH-044;
LAYOUT-INGESTION-PROPOSAL.md §5.11, sub-phase 9a).

An evident error in a source guideline (e.g. Kenya MoH p. 47 prints the
temperature criterion with its inequalities transposed) can be corrected at
ingestion **only** by an operator manifest entry naming a clinician
attester, a rationale and evidence:

    "text_corrections": [{"id", "page", "original", "corrected", "kind",
                          "rationale", "evidence", "attested_by", "attested_on"}]

Rules, all deterministic:

- **Scope limit, checked when the manifest is loaded**: the numeric and
  symbol tokens of `corrected` must equal those of `original` as a multiset
  (no value added, removed or changed), and every word added or removed must
  come from a small relational/function-word allowlist, so no drug, sign or
  population can be introduced. Fixing the *relationship* between existing
  values passes; a new clinical value needs a corrigendum or a new source
  version. This is what keeps an erratum from becoming a local guidance
  change (SCOPE-2.4 — see DEVIATIONS.md #214).
- **Exactly-once match** on the named source page, whitespace-tolerant (the
  same rule as quote integrity, `find_verbatim_quote`). Zero or several
  matches stop ingestion of the document (`CorrectionError`): never a guess.
- **Provenance**: the replaced range is recorded as an `attested` span with
  the correction id, and every citation that touches it must display the
  correction (enforced in `app.grounding.verifier` / `app.citations.model`).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from app.citations.model import find_verbatim_quote

REQUIRED_FIELDS = (
    "id",
    "page",
    "original",
    "corrected",
    "rationale",
    "evidence",
    "attested_by",
    "attested_on",
)
KINDS = ("erratum", "ocr_override")

# Words a correction may add or remove. Relational and function words only:
# nothing clinical (no drug, sign, dose, population).
RELATIONAL_ALLOWLIST = frozenset(
    {
        "less",
        "more",
        "greater",
        "fewer",
        "than",
        "equal",
        "to",
        "or",
        "and",
        "not",
        "at",
        "least",
        "most",
        "above",
        "below",
        "under",
        "over",
        "is",
        "of",
        "the",
        "a",
        "an",
    }
)

_TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)*|[^\W\d_]+|[^\w\s]")


class CorrectionError(ValueError):
    """A manifest correction is malformed, out of scope, or doesn't match the
    document exactly once. Ingestion of that document stops (fails closed)."""


@dataclass(frozen=True)
class Correction:
    id: str
    page: int
    original: str
    corrected: str
    kind: str
    rationale: str
    evidence: str
    attested_by: str
    attested_on: str

    def public(self) -> dict:
        """What a citation carries and the UI shows."""
        return {
            "id": self.id,
            "original": self.original,
            "corrected": self.corrected,
            "kind": self.kind,
            "rationale": self.rationale,
            "attested_by": self.attested_by,
            "attested_on": self.attested_on,
        }


def _tokens(text: str) -> tuple[Counter[str], Counter[str], Counter[str]]:
    numbers: Counter[str] = Counter()
    symbols: Counter[str] = Counter()
    words: Counter[str] = Counter()
    for tok in _TOKEN_RE.findall(text):
        if tok[0].isdigit():
            numbers[tok] += 1
        elif tok[0].isalpha():
            words[tok.lower()] += 1
        else:
            symbols[tok] += 1
    return numbers, symbols, words


def check_scope(original: str, corrected: str) -> list[str]:
    """Return the scope-limit violations (empty = allowed)."""
    o_num, o_sym, o_words = _tokens(original)
    c_num, c_sym, c_words = _tokens(corrected)
    problems = []
    if o_num != c_num:
        problems.append(
            f"numeric tokens differ: {sorted(o_num.elements())} -> {sorted(c_num.elements())}"
        )
    if o_sym != c_sym:
        problems.append(
            f"symbol/unit tokens differ: {sorted(o_sym.elements())} -> {sorted(c_sym.elements())}"
        )
    changed = (c_words - o_words) + (o_words - c_words)
    outside = sorted(w for w in changed if w not in RELATIONAL_ALLOWLIST)
    if outside:
        problems.append(f"words added/removed outside the relational allowlist: {outside}")
    return problems


def load_corrections(entries: list[dict] | None) -> list[Correction]:
    """Validate manifest `text_corrections`. Raises `CorrectionError` on any
    malformed or out-of-scope entry — the whole document is then not ingested."""
    out: list[Correction] = []
    seen: set[str] = set()
    for raw in entries or []:
        missing = [f for f in REQUIRED_FIELDS if not str(raw.get(f, "")).strip()]
        if missing:
            raise CorrectionError(f"text_correction {raw.get('id')!r} missing {missing}")
        kind = raw.get("kind", "erratum")
        if kind not in KINDS:
            raise CorrectionError(f"text_correction {raw['id']!r}: kind must be one of {KINDS}")
        if raw["id"] in seen:
            raise CorrectionError(f"duplicate text_correction id {raw['id']!r}")
        seen.add(raw["id"])
        problems = check_scope(raw["original"], raw["corrected"])
        if problems:
            raise CorrectionError(
                f"text_correction {raw['id']!r} is out of scope: {'; '.join(problems)}"
            )
        out.append(
            Correction(
                id=raw["id"],
                page=int(raw["page"]),
                original=raw["original"],
                corrected=raw["corrected"],
                kind=kind,
                rationale=raw["rationale"],
                evidence=raw["evidence"],
                attested_by=raw["attested_by"],
                attested_on=raw["attested_on"],
            )
        )
    return out


@dataclass
class AppliedCorrection:
    correction: Correction
    unit_index: int  # which text unit on the page it was applied to
    start: int  # span of the corrected text within that unit (after replacement)
    end: int
    original_end: int  # where the replaced original text ended (before replacement)

    @property
    def delta(self) -> int:
        """Length change; later offsets in the same unit shift by this much."""
        return self.end - self.original_end


def apply_to_units(
    units: list[str],
    correction: Correction,
    *,
    protected: list[list[tuple[int, int]]] | None = None,
) -> tuple[list[str], AppliedCorrection]:
    """Apply one correction to the text units of its page (block texts, in
    reading order). Exactly one whitespace-tolerant match across all units is
    required; a match overlapping a `protected` (structure) span is refused.
    Returns the new unit list and where the corrected text now sits."""
    matches = []
    for i, unit in enumerate(units):
        pos = 0
        while True:
            span = find_verbatim_quote(correction.original, unit[pos:])
            if span is None:
                break
            matches.append((i, pos + span[0], pos + span[1]))
            pos += span[1]
    if len(matches) != 1:
        raise CorrectionError(
            f"text_correction {correction.id!r}: original text found {len(matches)} time(s) on "
            f"page {correction.page}; exactly one match is required"
        )
    i, start, end = matches[0]
    if protected:
        for s, e in protected[i]:
            if start < e and s < end:
                raise CorrectionError(
                    f"text_correction {correction.id!r} overlaps generated structure text; "
                    "corrections apply to source text only"
                )
    new_unit = units[i][:start] + correction.corrected + units[i][end:]
    new_units = [*units[:i], new_unit, *units[i + 1 :]]
    return new_units, AppliedCorrection(
        correction, i, start, start + len(correction.corrected), original_end=end
    )
