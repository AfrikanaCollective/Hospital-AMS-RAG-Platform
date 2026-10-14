"""Narrative generation (ARCH §15.1 steps 2-6; PRD-060, PRD-061, PRD-064).

- extract a field subset actually present in the (possibly sparsened) record
  (step 3),
- generate a scope-1-framed narrative via `LLMGateway` (`MODEL_ID`), low
  temperature, fixed template (step 4),
- validate grounding of the narrative (`app.eval.question_gen.validate`,
  step 5) and reject-and-retry up to `QGEN_MAX_RETRIES`,
- return a dict shaped for an `eval_question` row (step 6 — persistence
  itself is a follow-on, not done here).

Picking the source record (step 2: matching a guideline's applicability,
sparsening for `missing_info_expected`, choosing a corpus-gap scenario for
`no_guideline_expected`) and the diversity filter / gold re-check (steps 7-8,
which need a populated corpus to check against) are **not** implemented here
— see DEVIATIONS.md #67. This module implements steps 3-6 for an
already-chosen `source_record`.

**Scope-1 framing is enforced twice** (DEVIATIONS.md #66): once in the
prompt template itself, and independently in `_is_scope1_framed` after
generation — a model can drift from instructions, and this question set
becomes the actual query input to the RAG system, so a scope violation here
would mean the eval harness itself manufactures out-of-scope queries. Never
"what should be done" / "what treatment" / "next step for this patient" —
only "what does the guideline recommend for a patient/infant presenting with
X" (ARCH §15.1 step 4's own mandated wording).
"""

from __future__ import annotations

import re
from typing import Any, Protocol

from app.config import get_settings
from app.eval.question_gen.validate import ValidatorReport, validate_narrative
from app.grounding.wording import has_directive_phrasing
from app.schemas.enums import ExpectedOutcome, Provenance

QUESTION_TEMPLATE_VERSION = "v1"

_SYSTEM_PROMPT = """You turn a structured, de-identified newborn/infant clinical \
record into exactly ONE natural-language guideline-lookup question.

Hard rules:
- The question must ask what a clinical guideline recommends or says for a \
patient/infant presenting with the findings given below — nothing else. The \
ONLY acceptable shape is: "What does the guideline recommend for a(n) \
<newborn/infant/patient> presenting with <findings>?" (light rephrasing of \
this exact shape is fine; asking for anything else is not).
- NEVER ask what should be done, what treatment to give, what the next step \
is, what medication or dose to use, or anything else that asks the reader to \
decide or act for this specific patient. This is a guideline LOOKUP question, \
not a request for a clinical decision.
- Mention ONLY clinical findings, history, vitals, and demographics that are \
explicitly listed below. Do not invent, assume, or infer any finding, \
timeline, or history not listed. Do not state that a finding is absent \
unless it is explicitly listed as absent/negative.
- Use only information that would be available at admission.
- Output exactly one sentence, ending in a question mark. No preamble, no \
explanation, just the question."""

_USER_TEMPLATE = """Patient presentation — the ONLY facts you may use:
{field_lines}

Write the one guideline-lookup question."""

# DEVIATIONS.md #115: found live, against a real gateway model
# (qwen3.5:9b) — the system prompt above already says "do not invent,
# assume, or infer any finding... not listed," but the model still reached
# a clinical impression not present in the record (e.g. medications
# "gentamicin"/"penicillin" -> the word "sepsis", a diagnosis label no
# field of that record actually stated) essentially every time; the retry
# loop then sent the exact same prompt again, so the same bad instinct just
# repeated across all `QGEN_MAX_RETRIES` attempts. This names the specific
# words `validate_narrative` rejected on the previous attempt, the same
# "name the exact wrong behavior" fix already used for the synthesis
# agent's own retry suffix (DEVIATIONS.md #111), rather than a generic
# restatement of the same rule the model already ignored once.
_RETRY_SUFFIX_TEMPLATE = """

STRICT: your previous question used word(s) not present in the patient
presentation above, and not just a framing word: {unmapped}. You may not
infer a diagnosis, condition, or clinical impression from what IS listed.
Rewrite the question using only the exact findings, vitals, and
demographics listed above, in approximately their own wording. It must
still ask what the GUIDELINE recommends (the word "guideline" must appear)
— do not drop that shape while fixing the wording above."""


def _retry_suffix(unmapped: list[str]) -> str:
    return _RETRY_SUFFIX_TEMPLATE.format(unmapped=", ".join(unmapped))


# Any of these in the generated text means the model drifted from the
# mandated shape (a directive/decision-seeking question) -> reject + retry.
_SCOPE1_VIOLATION_PATTERNS: tuple[str, ...] = (
    r"\bshould\b",
    r"\bwould you\b",
    r"\bnext step\b",
    r"\bwhat (treatment|medication|antibiotic|dose|drug)\b",
    r"\bhow (should|would|do) (you|i|we)\b",
    r"\bis it appropriate to\b",
)
_COMPILED_VIOLATIONS = [re.compile(p, re.IGNORECASE) for p in _SCOPE1_VIOLATION_PATTERNS]

_SECTION_FIELDS: tuple[tuple[str, str], ...] = (
    ("gestational_age_weeks", "gestational age (weeks)"),
    ("birth_weight_g", "birth weight (g)"),
    ("day_of_life", "day of life"),
    ("care_setting", "care setting"),
    ("presenting_complaint", "presenting complaint"),
    # triage_category deliberately absent: it is never used for guideline
    # retrieval, only for the reviewers' triage question (ARCH-046, #299).
)
_VITAL_FIELDS: tuple[tuple[str, str], ...] = (
    ("heart_rate_bpm", "heart rate (bpm)"),
    ("resp_rate_bpm", "respiratory rate (bpm)"),
    ("temperature_c", "temperature (C)"),
    ("spo2_percent", "SpO2 (%)"),
    ("weight_g", "weight (g)"),
    ("capillary_refill_seconds", "capillary refill (s)"),
)


class _ChatGateway(Protocol):
    def chat(self, *, system: str, messages: list[dict], **params: object) -> Any: ...


class QuestionGenerationFailed(RuntimeError):
    """Raised when no valid, scope-1-framed narrative was produced within
    `QGEN_MAX_RETRIES` attempts."""


def _humanize(name: str) -> str:
    """`difficulty_feeding` -> `difficulty feeding`. A snake_case field/EAV
    member name is an internal identifier, not prose (DEVIATIONS.md #155) —
    without this, the model reproduces the underscore verbatim into the
    generated question (the system prompt tells it to use the listed
    findings "in approximately their own wording," and it does, literally).
    Safe against the no-fabrication validator: `app.retrieval.sparse.analyze`
    already tokenizes on non-alphanumeric boundaries, so "difficulty_feeding"
    and "difficulty feeding" produce the same two tokens either way."""
    return name.replace("_", " ")


def _field_subset_lines(record: dict) -> str:
    """ARCH §15.1 step 3 — every structured field actually present in
    `record`, grouped for the prompt. Absent (`None`/empty) fields are
    omitted, never stated as negative (that would itself be a fabrication —
    PRD-061).

    **Medications and interventions are never included** (DEVIATIONS.md
    #155, operator instruction): seeing a specific drug/intervention name
    let a real gateway model infer an unstated diagnosis almost every time
    (DEVIATIONS.md #115's "gentamicin -> sepsis" finding) — excluding them
    from the prompt entirely is a stronger fix than the reject-and-retry
    mitigation already in place, which stays as a backstop.
    `app.eval.question_gen.validate._EXCLUDED_VALUE_FIELDS` additionally
    excludes both from the validator's allowed vocabulary, so even a model
    that mentions one anyway (from its own general knowledge, not this
    prompt) gets rejected and retried, not silently accepted."""
    lines: list[str] = []
    if record.get("sex"):
        lines.append(f"- sex: {record['sex']}")

    encounter = record.get("encounter") or {}
    for key, label in _SECTION_FIELDS:
        value = encounter.get(key)
        if value not in (None, ""):
            lines.append(f"- {label}: {value}")

    for key, label in (("problems", "problems"), ("allergies", "known allergies")):
        values = record.get(key) or []
        if values:
            lines.append(f"- {label}: {', '.join(str(v) for v in values)}")

    findings = [
        _humanize(f["name"]) for f in (record.get("examination_findings") or []) if f.get("present")
    ]
    if findings:
        lines.append(f"- examination findings present: {', '.join(findings)}")

    vitals_list = record.get("vitals") or []
    if vitals_list:
        v = vitals_list[0]
        parts = [f"{label}={v[key]}" for key, label in _VITAL_FIELDS if v.get(key) is not None]
        if parts:
            lines.append(f"- admission vitals: {', '.join(parts)}")

    labs = record.get("labs") or []
    lab_parts = [
        f"{lab['analyte']}={lab.get('value')}{lab.get('unit') or ''}"
        for lab in labs
        if lab.get("analyte")
    ]
    if lab_parts:
        lines.append(f"- labs: {', '.join(lab_parts)}")

    return "\n".join(lines) if lines else "- (no structured clinical fields present in this record)"


def _is_scope1_framed(text: str) -> bool:
    if not text.strip().endswith("?"):
        return False
    if "guideline" not in text.lower():
        return False
    if has_directive_phrasing(text):
        return False
    return not any(p.search(text) for p in _COMPILED_VIOLATIONS)


def generate_question(
    source_record: dict,
    expected: ExpectedOutcome,
    *,
    target_guideline_topic: str | None = None,
    gateway: _ChatGateway | None = None,
    max_retries: int | None = None,
) -> dict:
    """Generate one auto-generated `eval_question` dict for `source_record`.

    `target_guideline_topic` (e.g. a matched guideline's title/topic phrase)
    is optional context: it's added to the validator's allowed vocabulary
    (mentioning the topic being asked about isn't a fabricated patient fact)
    and recorded on `target_guideline_ref`, but is never itself something the
    prompt is told to name — the narrative describes the *patient*, not the
    guideline.
    """
    settings = get_settings()
    retries = settings.qgen_max_retries if max_retries is None else max_retries
    gw: _ChatGateway
    if gateway is not None:
        gw = gateway
    else:
        # Lazy: avoids constructing a real LLMGateway (which validates MODEL_ID
        # is configured) for callers that always inject a gateway/fake.
        from app.llm.gateway import LLMGateway  # noqa: PLC0415

        # LLMGateway.chat has one more optional named parameter (contains_phi)
        # than _ChatGateway's **params-only signature declares; a strict
        # superset at runtime, but mypy's structural Protocol match can't see
        # that through **kwargs.
        gw = LLMGateway()  # type: ignore[assignment]

    field_lines = _field_subset_lines(source_record)
    allowed_topic_terms = [target_guideline_topic] if target_guideline_topic else None

    last_report: ValidatorReport | None = None
    last_text: str | None = None
    for attempt in range(retries + 1):
        user_content = _USER_TEMPLATE.format(field_lines=field_lines)
        if last_report is not None and last_report.unmapped_entities:
            user_content += _retry_suffix(last_report.unmapped_entities)
        result = gw.chat(
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
            temperature=0.2,
        )
        question_text = result.text.strip()
        last_text = question_text
        report = validate_narrative(
            question_text, source_record, allowed_topic_terms=allowed_topic_terms
        )
        last_report = report

        if report.ok and _is_scope1_framed(question_text):
            target_guideline_ref = (
                {"topic": target_guideline_topic}
                if target_guideline_topic and expected != ExpectedOutcome.NO_GUIDELINE_EXPECTED
                else None
            )
            return {
                "text": question_text,
                "provenance": Provenance.AUTO_GENERATED,
                "expected_outcome": expected,
                "source_record_id": source_record.get("record_id"),
                "target_guideline_ref": target_guideline_ref,
                "generator_meta": {
                    "model_id": result.model_id,
                    "template_version": QUESTION_TEMPLATE_VERSION,
                    "attempt": attempt,
                    "validator_report": {
                        "ok": report.ok,
                        "mapped_entities": report.mapped_entities,
                        "unmapped_entities": report.unmapped_entities,
                        "notes": report.notes,
                    },
                },
            }

    raise QuestionGenerationFailed(
        f"no valid scope-1-framed narrative for record "
        f"{source_record.get('record_id')!r} after {retries + 1} attempt(s); "
        f"last text={last_text!r}, last validator report={last_report}"
    )
