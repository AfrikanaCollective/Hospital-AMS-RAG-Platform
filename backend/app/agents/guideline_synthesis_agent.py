"""Guideline-synthesis agent (SCOPE-1; ARCH §10.2, §8.2, §9.1).

Does: turn (question + retrieved chunks [+ stage label] [+ missing-info]) into
SEGMENTED output — claim segments each with citation_ids and a verbatim quote;
framing segments non-directive. Emits `no_guideline` if the retrieved set can't
support an answer.
Does NOT: see raw PHI (only an orchestrator-built feature summary); use
knowledge outside the retrieved chunks; produce directive text ("you should…").
Access: ONLY the chunk texts passed in + the question. No store access.
Tools: get_chunk (restricted to this turn's set), get_citation_metadata.

Prompt template: app/agents/prompts/guideline_synthesis.md — enforces the
"Guideline X recommends…" framing (SCOPE-1.2). The §8.3 wording check is a
second line of defense (app.agents.citation_verifier_agent, next in the graph).

Retrieval-confidence gating happens HERE, before any model call (ARCH §7.5):
- essentially empty retrieval -> "no guideline found" directly (terminal, not
  held — RELEASE_POLICY[NO_GUIDELINE] == "terminal_no_guideline"), no model
  call, no claims, nothing for the citation-verifier to check.
- conflicting sources -> escalate (conflicting_sources), no model call.
- low confidence (but not essentially empty / no conflict) -> escalate
  (low_confidence), no model call.
Only a genuinely supported retrieval reaches the model.

Malformed / structurally invalid model output (ARCH §8.2: free-form prose
without segment structure — including a real, observed failure mode where a
gateway model wrote flattened prose with inline `[c1]`-style citation
markers instead of the required JSON list, DEVIATIONS.md #111) is rejected
and regenerated up to `_MAX_ATTEMPTS - 1` times; exhausting every attempt
escalates (grounding_failure) rather than passing bad output downstream.

Context budget (DEVIATIONS.md #260): the rendered prompt must leave the model
room to answer. `fit_retrieval_to_budget` drops the lowest-scoring chunks
(keeping at least one per guideline while possible) until the estimated prompt
fits `LLM_CONTEXT_TOKENS - LLM_OUTPUT_RESERVE_TOKENS`; the dropped chunks also
leave `state["retrieval"]`, so the verifier only accepts citations to chunks
the model actually saw. A reply the gateway reports as cut off at its length
limit (`finish_reason == "length"`) is retried with a smaller prompt budget
rather than the same prompt (which would be cut off at the same point), and
escalation then says the output was truncated.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from app.agents.state import GraphState
from app.config import get_settings
from app.grounding.segments import SegmentParseError, is_structurally_valid, parse_segments
from app.llm.gateway import LLMGateway
from app.schemas.enums import EscalationTrigger, ObservedOutcome

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
# 3, not 2 (DEVIATIONS.md #111): a real gateway model has been observed
# ignoring the JSON output format entirely on a retry too — reverting to
# free-form prose with inline `[c1]`-style citation markers instead of the
# required JSON list — so the single retry this used to allow wasn't always
# enough. One more attempt costs latency only on the already-failing path.
_MAX_ATTEMPTS = 3

NO_GUIDELINE_TEXT = (
    "No guideline in the retrieved corpus addresses this question. No recommendation "
    "is available from this system for this question."
)

# Indirection point for tests: Callable[[str], str] taking the rendered prompt
# and returning the model's raw text. None => construct a real LLMGateway
# (requires a non-placeholder MODEL_ID; not usable offline).
_CHAT_FN: Callable[[str], str] | None = None


def _load_template() -> str:
    return (_PROMPTS_DIR / "guideline_synthesis.md").read_text()


def _default_chat_fn(state: GraphState) -> Callable[[str], str]:
    """Real (non-test) chat path. Closes over `state` only to record which
    model actually answered (`state["model_id"]`) — ARCH-035's "answer" audit
    event needs it, and the `_CHAT_FN` test seam's `str -> str` contract can't
    carry it back some other way without every test double having to fake a
    richer return type it has no reason to care about."""

    def _fn(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        result = LLMGateway().chat(system="", messages=messages, contains_phi=True)
        state["model_id"] = result.model_id
        ctx = state.setdefault("synthesis_context", {})
        ctx["last_finish_reason"] = getattr(result, "finish_reason", None)
        return result.text

    return _fn


def _resolve_chat_fn(state: GraphState) -> Callable[[str], str]:
    return _CHAT_FN or _default_chat_fn(state)


def _build_sources(retrieval: Sequence[Mapping[str, Any]]) -> str:
    lines = []
    for i, item in enumerate(retrieval):
        lines.append(
            f"[c{i + 1}] ({item.get('document_title')} {item.get('version_label')}, "
            f"section {item.get('section_number') or item.get('section_path') or '?'}, "
            f"p.{item.get('page_start')}-{item.get('page_end')}):\n{item.get('text')}"
        )
    return "\n\n".join(lines)


def _render_prompt(state: GraphState) -> str:
    template = _load_template()
    stage = state.get("stage_classification")
    missing = state.get("missing_info")
    scope2_lines = []
    if stage:
        scope2_lines.append(f"Stage classification: {json.dumps(stage)}")
    if missing:
        scope2_lines.append(f"Missing information: {json.dumps(missing)}")
    return (
        template.replace("{{question}}", state.get("query", ""))
        .replace("{{feature_summary}}", json.dumps(state.get("patient_features") or {}))
        .replace("{{scope2_context}}", "\n".join(scope2_lines))
        .replace("{{hospital_constraint}}", state.get("hospital_constraint") or "")
        .replace("{{sources}}", _build_sources(state.get("retrieval") or []))
    )


def estimate_tokens(text: str) -> int:
    """Conservative token estimate for the gateway model (no local tokenizer
    for it): characters / LLM_CHARS_PER_TOKEN, rounded up."""
    return math.ceil(len(text) / max(get_settings().llm_chars_per_token, 1.0))


def fit_retrieval_to_budget(
    items: list[Any], render: Callable[[list[Any]], str], budget_tokens: int
) -> tuple[list[Any], list[str]]:
    """Drop chunks until `render(kept)` fits `budget_tokens` (estimated).
    Lowest rerank score goes first, but a guideline keeps its last chunk
    until every guideline is down to one; order is preserved (priority
    groups stay in place). Returns (kept items, dropped chunk ids)."""
    kept = list(items)
    dropped: list[str] = []
    while kept and estimate_tokens(render(kept)) > budget_tokens:
        per_doc: dict[str, int] = {}
        for item in kept:
            doc = str(item.get("document_id"))
            per_doc[doc] = per_doc.get(doc, 0) + 1
        removable = [i for i in kept if per_doc[str(i.get("document_id"))] > 1] or kept
        victim = min(removable, key=lambda i: float(i.get("score") or 0.0))
        kept.remove(victim)
        dropped.append(str(victim.get("chunk_id")))
    return kept, dropped


def _no_guideline(state: GraphState) -> GraphState:
    state["candidate_segments"] = [{"type": "framing", "text": NO_GUIDELINE_TEXT}]
    state["candidate_citations"] = []
    state["observed_outcome"] = ObservedOutcome.NO_GUIDELINE
    return state


def _escalate(state: GraphState, trigger: EscalationTrigger, message: str) -> GraphState:
    state["escalation"] = {"trigger_code": trigger, "message": message}
    return state


def run(state: GraphState) -> GraphState:
    if state.get("escalation"):
        # An earlier node (patient_record / stage_classifier / missing_info)
        # already decided to escalate (e.g. stage_classification_uncertain,
        # missing_critical_info). The graph still routes here unconditionally
        # (app.agents.graph), but there is nothing to synthesize — pass
        # through untouched so the routing after citation_verifier sends this
        # straight to the escalation node.
        return state

    confidence = state.get("retrieval_confidence") or {}
    if confidence.get("essentially_empty"):
        return _no_guideline(state)
    if confidence.get("conflicts"):
        return _escalate(
            state,
            EscalationTrigger.CONFLICTING_SOURCES,
            "The retrieved sources conflict on this question and need clinician review.",
        )
    if confidence.get("low_confidence"):
        return _escalate(
            state,
            EscalationTrigger.LOW_CONFIDENCE,
            "Retrieval confidence was too low to synthesize a grounded answer.",
        )

    return _synthesize(state)


def _synthesize(state: GraphState) -> GraphState:
    """Prompt the model within the context budget and parse its segments;
    escalate (grounding_failure) after `_MAX_ATTEMPTS` failed attempts."""
    chat_fn = _resolve_chat_fn(state)
    settings = get_settings()
    full_retrieval = list(state.get("retrieval") or [])
    reserve = settings.llm_output_reserve_tokens
    ctx = state.setdefault("synthesis_context", {})
    ctx.update(context_tokens=settings.llm_context_tokens, truncated_attempts=0)
    last_error: Exception | None = None
    # DEVIATIONS.md #111: the original one-line "STRICT" suffix wasn't
    # forceful enough on its own — a real gateway model was observed
    # reverting to prose with inline `[c1]` markers even on the retry.
    # Spelled out explicitly what NOT to do (prose, inline citation
    # brackets, markdown, commentary), not just what TO do, since a model
    # that already ignored the positive instruction once needs the failure
    # mode named directly, not just restated more tersely.
    retry_suffix = (
        "\n\nSTRICT: your previous reply was not a valid JSON list of segments. "
        "Do not write prose. Do not use inline citation markers like [c1] in "
        "running text. Do not use markdown or a code fence. Reply with ONLY a "
        "raw JSON array, starting with [ and ending with ], e.g.: "
        '[{"type": "framing", "text": "..."}, {"type": "claim", "text": "...", '
        '"citation_ids": ["c1"], "quote": "<verbatim quote from c1>"}]. No text '
        "before or after the array."
    )

    def _render(items: list[Any]) -> str:
        state["retrieval"] = items
        return _render_prompt(state)

    malformed_before = False
    for _attempt in range(_MAX_ATTEMPTS):
        # the STRICT suffix answers a malformed reply, not a truncated one
        suffix = retry_suffix if malformed_before else ""
        budget = settings.llm_context_tokens - reserve - estimate_tokens(suffix)
        kept, dropped = fit_retrieval_to_budget(full_retrieval, _render, budget)
        if not kept:
            return _escalate(
                state,
                EscalationTrigger.GROUNDING_FAILURE,
                "The retrieved sources do not fit the model's context window "
                f"(LLM_CONTEXT_TOKENS={settings.llm_context_tokens}).",
            )
        state["retrieval"] = kept
        prompt = _render_prompt(state)
        ctx.update(
            prompt_budget_tokens=budget,
            prompt_tokens_estimated=estimate_tokens(prompt),
            dropped_chunk_ids=dropped,
        )
        ctx.pop("last_finish_reason", None)
        raw = chat_fn(prompt + suffix)
        if ctx.get("last_finish_reason") == "length":
            # Cut off at the model's length limit: the same prompt would be
            # cut off at the same point, so shrink the prompt and retry.
            ctx["truncated_attempts"] += 1
            reserve *= 2
            last_error = SegmentParseError(
                "model output was truncated at the context limit (finish_reason=length)"
            )
            continue
        try:
            segments = parse_segments(raw)
        except SegmentParseError as exc:
            last_error = exc
            malformed_before = True
            continue
        if not is_structurally_valid(segments):
            last_error = SegmentParseError("structurally invalid segment list")
            malformed_before = True
            continue
        state["candidate_segments"] = segments
        return state

    return _escalate(
        state,
        EscalationTrigger.GROUNDING_FAILURE,
        f"Synthesis output could not be parsed into valid answer segments: {last_error}",
    )
