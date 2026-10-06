"""Guideline-synthesis agent (SCOPE-1; ARCH §8.2, §9.1)."""

from __future__ import annotations

import json

import app.agents.guideline_synthesis_agent as gsa
from app.schemas.enums import EscalationTrigger, ObservedOutcome, SegmentType

CHUNK = {
    "chunk_id": "ch1",
    "text": "Record respiratory rate at presentation for every neonate.",
    "document_title": "Newborn Care Guideline",
    "version_label": "2021",
    "section_number": "3.2",
    "page_start": 5,
    "page_end": 5,
}


def _valid_response() -> str:
    return json.dumps(
        [
            {"type": "framing", "text": "Per the retrieved guideline:"},
            {
                "type": "claim",
                "text": "Guideline X recommends recording respiratory rate at presentation.",
                "citation_ids": ["c1"],
                "quote": "Record respiratory rate at presentation",
            },
        ]
    )


def test_default_chat_fn_records_model_id_on_state(monkeypatch) -> None:  # noqa: ANN001
    """The real (non-`_CHAT_FN`-overridden) path must record which model
    answered, for the "answer" audit event (ARCH-035, DEVIATIONS.md #91)."""

    class _FakeResult:
        text = _valid_response()
        model_id = "stub-model-v7"

    class _FakeGateway:
        def chat(self, **kwargs):  # noqa: ANN003, ARG002
            return _FakeResult()

    monkeypatch.setattr(gsa, "_CHAT_FN", None)
    monkeypatch.setattr(gsa, "LLMGateway", _FakeGateway)
    state = {
        "query": "what does the guideline say?",
        "retrieval": [CHUNK],
        "retrieval_confidence": {
            "essentially_empty": False,
            "low_confidence": False,
            "conflicts": [],
        },
    }
    out = gsa.run(state)  # type: ignore[arg-type]
    assert out["model_id"] == "stub-model-v7"


def test_essentially_empty_retrieval_returns_no_guideline_without_model_call(monkeypatch) -> None:  # noqa: ANN001
    calls = []
    monkeypatch.setattr(gsa, "_CHAT_FN", lambda prompt: calls.append(prompt) or "[]")  # noqa: ARG005
    state = {
        "query": "what does the guideline say about scurvy?",
        "retrieval": [],
        "retrieval_confidence": {
            "essentially_empty": True,
            "low_confidence": True,
            "conflicts": [],
        },
    }
    out = gsa.run(state)  # type: ignore[arg-type]
    assert out["observed_outcome"] == ObservedOutcome.NO_GUIDELINE
    assert out["candidate_segments"][0]["text"] == gsa.NO_GUIDELINE_TEXT
    assert calls == []  # no model call was made


def test_conflicting_sources_escalates_without_model_call(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(
        gsa,
        "_CHAT_FN",
        lambda prompt: (_ for _ in ()).throw(AssertionError("should not be called")),
    )
    state = {
        "query": "x",
        "retrieval": [CHUNK],
        "retrieval_confidence": {
            "essentially_empty": False,
            "low_confidence": False,
            "conflicts": [{"a": 1}],
        },
    }
    out = gsa.run(state)  # type: ignore[arg-type]
    assert out["escalation"]["trigger_code"] == EscalationTrigger.CONFLICTING_SOURCES


def test_low_confidence_escalates_without_model_call(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(
        gsa,
        "_CHAT_FN",
        lambda prompt: (_ for _ in ()).throw(AssertionError("should not be called")),
    )
    state = {
        "query": "x",
        "retrieval": [CHUNK],
        "retrieval_confidence": {
            "essentially_empty": False,
            "low_confidence": True,
            "conflicts": [],
        },
    }
    out = gsa.run(state)  # type: ignore[arg-type]
    assert out["escalation"]["trigger_code"] == EscalationTrigger.LOW_CONFIDENCE


def test_good_retrieval_calls_model_and_parses_segments(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(gsa, "_CHAT_FN", lambda prompt: _valid_response())
    state = {
        "query": "what does the guideline recommend for a neonate presenting with fever?",
        "retrieval": [CHUNK],
        "retrieval_confidence": {
            "essentially_empty": False,
            "low_confidence": False,
            "conflicts": [],
        },
    }
    out = gsa.run(state)  # type: ignore[arg-type]
    assert "escalation" not in out
    assert out["candidate_segments"][1]["type"] == SegmentType.CLAIM


def test_malformed_output_retries_then_escalates(monkeypatch) -> None:  # noqa: ANN001
    calls = {"n": 0}

    def _chat(prompt: str) -> str:  # noqa: ARG001
        calls["n"] += 1
        return "not json"

    monkeypatch.setattr(gsa, "_CHAT_FN", _chat)
    state = {
        "query": "x",
        "retrieval": [CHUNK],
        "retrieval_confidence": {
            "essentially_empty": False,
            "low_confidence": False,
            "conflicts": [],
        },
    }
    out = gsa.run(state)  # type: ignore[arg-type]
    assert calls["n"] == 3  # two retries (DEVIATIONS.md #111: _MAX_ATTEMPTS raised 2 -> 3)
    assert out["escalation"]["trigger_code"] == EscalationTrigger.GROUNDING_FAILURE


def test_second_attempt_succeeding_recovers(monkeypatch) -> None:  # noqa: ANN001
    calls = {"n": 0}

    def _chat(prompt: str) -> str:  # noqa: ARG001
        calls["n"] += 1
        return "not json" if calls["n"] == 1 else _valid_response()

    monkeypatch.setattr(gsa, "_CHAT_FN", _chat)
    state = {
        "query": "x",
        "retrieval": [CHUNK],
        "retrieval_confidence": {
            "essentially_empty": False,
            "low_confidence": False,
            "conflicts": [],
        },
    }
    out = gsa.run(state)  # type: ignore[arg-type]
    assert "escalation" not in out
    assert calls["n"] == 2


def test_third_attempt_succeeding_recovers(monkeypatch) -> None:  # noqa: ANN001
    """DEVIATIONS.md #111: _MAX_ATTEMPTS raised 2 -> 3 after a real model was
    observed ignoring the JSON format on the first retry too — the second
    retry must still get a chance."""
    calls = {"n": 0}

    def _chat(prompt: str) -> str:  # noqa: ARG001
        calls["n"] += 1
        return "not json" if calls["n"] < 3 else _valid_response()

    monkeypatch.setattr(gsa, "_CHAT_FN", _chat)
    state = {
        "query": "x",
        "retrieval": [CHUNK],
        "retrieval_confidence": {
            "essentially_empty": False,
            "low_confidence": False,
            "conflicts": [],
        },
    }
    out = gsa.run(state)  # type: ignore[arg-type]
    assert "escalation" not in out
    assert calls["n"] == 3


def test_prose_with_inline_citation_markers_is_rejected_not_parsed(monkeypatch) -> None:  # noqa: ANN001
    """DEVIATIONS.md #111: reproduces the exact real-world failure shape —
    the model ignoring the JSON schema entirely and writing flattened prose
    with inline `[c1]`-style citation markers instead. This must never be
    treated as valid output (there is no verbatim quote to machine-verify
    against a source, only a paraphrase with a bracket) — it should fail to
    parse on every attempt and escalate, same as any other malformed output."""
    prose = (
        "The guideline addresses the management of small and sick newborns "
        "presenting with specific danger signs such as apnoea, fever, and low "
        "birth weight. Per [source], small and sick newborns are adequately "
        "monitored, appropriately reassessed and receive supportive care "
        "according to MoH guidelines.[c1]"
    )
    monkeypatch.setattr(gsa, "_CHAT_FN", lambda prompt: prose)  # noqa: ARG005
    state = {
        "query": "x",
        "retrieval": [CHUNK],
        "retrieval_confidence": {
            "essentially_empty": False,
            "low_confidence": False,
            "conflicts": [],
        },
    }
    out = gsa.run(state)  # type: ignore[arg-type]
    assert out["escalation"]["trigger_code"] == EscalationTrigger.GROUNDING_FAILURE


# ── context budget and truncated output (DEVIATIONS.md #260) ──


def _chunk(cid: str, doc: str, score: float, words: int = 200) -> dict:
    return {
        **CHUNK,
        "chunk_id": cid,
        "document_id": doc,
        "score": score,
        "text": f"{cid} " + "made-up guideline text " * words,
    }


def _state(retrieval: list[dict]) -> dict:
    return {
        "query": "what does the guideline say?",
        "retrieval": retrieval,
        "retrieval_confidence": {
            "essentially_empty": False,
            "low_confidence": False,
            "conflicts": [],
        },
    }


def _settings(monkeypatch, context: int, reserve: int) -> None:  # noqa: ANN001
    from app.config import get_settings

    monkeypatch.setenv("LLM_CONTEXT_TOKENS", str(context))
    monkeypatch.setenv("LLM_OUTPUT_RESERVE_TOKENS", str(reserve))
    monkeypatch.setenv("LLM_CHARS_PER_TOKEN", "3.2")
    get_settings.cache_clear()
    monkeypatch.setattr(gsa, "get_settings", get_settings)


def test_fit_drops_lowest_scores_but_keeps_one_per_guideline() -> None:
    items = [
        _chunk("k1", "kenya", 0.9),
        _chunk("k2", "kenya", 0.3),
        _chunk("m1", "moh", 0.5),
        _chunk("w1", "who", 0.4),
        _chunk("w2", "who", 0.2),
    ]

    def render(kept: list[dict]) -> str:
        return "".join(i["text"] for i in kept)

    one = len(render(items[:1]))
    kept, dropped = gsa.fit_retrieval_to_budget(items, render, budget_tokens=int(3.3 * one / 3.2))
    assert dropped == ["w2", "k2"]  # lowest first, never a guideline's last chunk
    assert [i["chunk_id"] for i in kept] == ["k1", "m1", "w1"]  # order preserved


def test_prompt_is_fitted_and_dropped_chunks_leave_retrieval(monkeypatch) -> None:  # noqa: ANN001
    _settings(monkeypatch, context=6000, reserve=1000)
    prompts: list[str] = []
    monkeypatch.setattr(gsa, "_CHAT_FN", lambda p: prompts.append(p) or _valid_response())
    retrieval = [_chunk(f"c{i}", f"doc{i % 2}", 1 - i / 10) for i in range(8)]
    out = gsa.run(_state(retrieval))  # type: ignore[arg-type]
    assert "escalation" not in out
    assert gsa.estimate_tokens(prompts[0]) <= 5000
    assert out["synthesis_context"]["dropped_chunk_ids"]  # 8 x ~1.4k tokens can't all fit
    kept = {i["chunk_id"] for i in out["retrieval"]}
    assert kept and kept.isdisjoint(out["synthesis_context"]["dropped_chunk_ids"])
    assert {i["document_id"] for i in out["retrieval"]} == {"doc0", "doc1"}


def test_truncated_reply_retries_with_a_smaller_prompt(monkeypatch) -> None:  # noqa: ANN001
    _settings(monkeypatch, context=6000, reserve=1000)
    prompts: list[str] = []
    replies = [('[{"type": "claim", "text": "Guideline', "length"), (_valid_response(), "stop")]

    class _Result:
        def __init__(self, text: str, reason: str) -> None:
            self.text, self.finish_reason, self.model_id = text, reason, "m"

    class _Gateway:
        def chat(self, **kw):  # noqa: ANN003
            prompts.append(kw["messages"][0]["content"])
            return _Result(*replies[len(prompts) - 1])

    monkeypatch.setattr(gsa, "_CHAT_FN", None)
    monkeypatch.setattr(gsa, "LLMGateway", _Gateway)
    retrieval = [_chunk(f"c{i}", f"doc{i % 3}", 1 - i / 10, words=120) for i in range(9)]
    out = gsa.run(_state(retrieval))  # type: ignore[arg-type]
    assert "escalation" not in out
    assert len(prompts) == 2 and len(prompts[1]) < len(prompts[0])
    assert "STRICT" not in prompts[1]  # truncation is not a malformed reply
    assert out["synthesis_context"]["truncated_attempts"] == 1


def test_always_truncated_escalates_saying_so(monkeypatch) -> None:  # noqa: ANN001
    _settings(monkeypatch, context=6000, reserve=500)

    class _Result:
        text, finish_reason, model_id = '[{"type": "claim", "text": "Guid', "length", "m"

    class _Gateway:
        def chat(self, **kw):  # noqa: ANN003, ARG002
            return _Result()

    monkeypatch.setattr(gsa, "_CHAT_FN", None)
    monkeypatch.setattr(gsa, "LLMGateway", _Gateway)
    out = gsa.run(_state([_chunk("c1", "d", 0.9, words=50)]))  # type: ignore[arg-type]
    assert out["escalation"]["trigger_code"] == EscalationTrigger.GROUNDING_FAILURE
    assert "truncated" in out["escalation"]["message"]


def test_prompt_requires_separate_positions_when_guidelines_differ() -> None:
    """Rule 8 (DEVIATIONS.md #262): differences between guidelines are reported
    side by side, never resolved, merged or dropped (PRD-014; report-and-cite)."""
    template = gsa._load_template()
    rule = template[template.index("8. **Several guidelines") : template.index("# Output format")]
    for phrase in (
        "own `claim` segment(s)",
        "neutral `framing` segment",
        "Never choose between them",
        "never merge them",
        "never leave out a guideline's position",
    ):
        assert phrase in rule
    assert "should be followed" in rule  # only inside the prohibition
