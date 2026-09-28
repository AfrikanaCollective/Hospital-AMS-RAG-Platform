"""Vision-LLM table transcription (ARCH-044 D12, sub-phase 9c;
LAYOUT-INGESTION-PROPOSAL.md §18). Offline: the gateway call is a fake or an
httpx MockTransport; tables are synthetic (tests/layout_fixtures.py)."""

from __future__ import annotations

import json
from dataclasses import dataclass

import httpx
import pytest

from app.config import Settings
from app.ingestion.chunking import chunk_document
from app.ingestion.layout.assemble import AssemblyOptions, assemble
from app.ingestion.layout.vlm_tables import (
    TableTranscriber,
    TranscriptionError,
    merge,
    parse_transcription,
    to_table_data,
)
from app.llm.gateway import LLMGateway, LLMGatewayError
from tests.layout_fixtures import dose_table, synthetic_document

MODEL = "vision-test-model"


def _transcription(**overrides) -> dict:
    t = {
        "title": ["Synthetic doses for group one"],
        "columns": 3,
        "header_rows": [
            [
                {"text": "Weight (kg)", "col": 0, "col_span": 1, "row_span": 2},
                {"text": "Agent P (10 u/kg)", "col": 1, "col_span": 1, "row_span": 1},
                {"text": "Agent Q (2 u/kg)", "col": 2, "col_span": 1, "row_span": 1},
            ],
            [
                {"text": "12 hrly", "col": 1, "col_span": 1, "row_span": 1},
                {"text": "24 hrly", "col": 2, "col_span": 1, "row_span": 1},
            ],
        ],
        "body_rows": [
            [
                {"text": v, "col": i, "col_span": 1, "row_span": 1}
                for i, v in enumerate(["1.0", "10", "2"])
            ],
            [
                {"text": v, "col": i, "col_span": 1, "row_span": 1}
                for i, v in enumerate(["2.0", "20", "4"])
            ],
        ],
        "notes": [],
        "illegible": [],
    }
    t.update(overrides)
    return t


@dataclass
class _Answer:
    response: object
    model: str = MODEL
    done_reason: str | None = "stop"


class _FakeGateway:
    def __init__(self, answer: _Answer) -> None:
        self.answer = answer
        self.calls = 0

    def __call__(self, png: bytes, prompt: str) -> _Answer:
        assert png and "JSON" in prompt
        self.calls += 1
        return self.answer


def _transcriber(answer: _Answer, tmp_path) -> tuple[TableTranscriber, _FakeGateway]:
    fake = _FakeGateway(answer)
    return TableTranscriber(fake, expected_model=MODEL, cache_dir=tmp_path), fake


# ── parsing (§18.2 response shapes, §18.5 schema) ──


@pytest.mark.parametrize(
    "response",
    [
        _transcription(),  # gateway already parsed the JSON
        json.dumps(_transcription()),  # pretty-printed JSON string
        "```json\n" + json.dumps(_transcription()) + "\n```",  # fenced
    ],
)
def test_all_three_gateway_response_shapes_parse(response) -> None:
    assert parse_transcription(response).columns == 3


@pytest.mark.parametrize(
    "response",
    [
        "This image shows a dosing table with…",  # prose is never salvaged
        {"response": _transcription()},  # gateway-unwrap key must not appear
        _transcription(commentary="looks like a dosing table"),  # extra key
        [_transcription()],  # a list, not an object
    ],
)
def test_unusable_responses_are_rejected(response) -> None:
    with pytest.raises(TranscriptionError):
        parse_transcription(response)


# ── merge (§18.6) ──


def test_agreeing_table_merges_cleanly_and_vlm_headers_are_used() -> None:
    ocr = dose_table()
    for c in ocr.cells:  # an OCR header misread, like Kenya's "Weight ()"
        if c.text == "Weight (kg)":
            c.text = "Weight ()"
    result = merge(to_table_data(parse_transcription(_transcription())), ocr)
    assert result.agreement == {
        "cells": 6,
        "agreed": 6,
        "vlm_only": 0,
        "ocr_only": 0,
        "numeric_disagreements": 0,
    }
    assert any(c.text == "Weight (kg)" for c in result.table.cells)


def test_numeric_disagreement_is_flagged_and_vlm_value_used() -> None:
    t = _transcription()
    t["body_rows"][1][1]["text"] = "25"
    result = merge(to_table_data(parse_transcription(t)), dose_table())
    assert result.agreement["numeric_disagreements"] == 1
    assert result.cell_diff == [{"row": 1, "col": 1, "ocr": "20", "vlm": "25", "numeric": True}]


def test_illegible_vlm_cell_falls_back_to_ocr() -> None:
    t = _transcription()
    t["body_rows"][0][2]["text"] = "[illegible]"
    result = merge(to_table_data(parse_transcription(t)), dose_table())
    assert result.agreement["ocr_only"] == 1
    body = [c.text for c in result.table.cells if c.row == 3]
    assert body == ["1.0", "10", "2"]


def test_row_count_mismatch_is_rejected() -> None:
    t = _transcription()
    t["body_rows"] = t["body_rows"][:1]
    with pytest.raises(TranscriptionError, match="body rows"):
        merge(to_table_data(parse_transcription(t)), dose_table())


# ── transcriber: model check, truncation, cache (§18.3-§18.4) ──


def test_wrong_model_is_rejected_and_ocr_kept(tmp_path) -> None:
    tr, _ = _transcriber(_Answer(_transcription(), model="some-other-model"), tmp_path)
    assert tr.transcribe(b"png", dose_table()) is None
    assert "expected" in tr.report["rejected"][0]["reason"]


def test_truncated_answer_is_rejected(tmp_path) -> None:
    tr, _ = _transcriber(_Answer(_transcription(), done_reason="length"), tmp_path)
    assert tr.transcribe(b"png", dose_table()) is None


def test_gateway_failure_keeps_ocr(tmp_path) -> None:
    def boom(png: bytes, prompt: str):
        raise LLMGatewayError("vision endpoint returned 503")

    tr = TableTranscriber(boom, expected_model=MODEL, cache_dir=tmp_path)
    assert tr.transcribe(b"png", dose_table()) is None
    assert "503" in tr.report["rejected"][0]["reason"]


def test_answer_is_cached_by_crop_and_reused(tmp_path) -> None:
    tr, fake = _transcriber(_Answer(_transcription()), tmp_path)
    first = tr.transcribe(b"same-crop", dose_table())
    second = tr.transcribe(b"same-crop", dose_table())
    assert first is not None and second is not None
    assert fake.calls == 1 and tr.report["cache_hits"] == 1
    assert first[1]["vlm"]["model"] == MODEL


# ── assembly + review hold (§18.7) ──


def test_transcribed_table_is_always_held_even_when_it_agrees(tmp_path) -> None:
    tr, _ = _transcriber(_Answer(_transcription()), tmp_path)
    seen = []

    def hook(el):
        seen.append(el.origin)
        return tr.transcribe(b"crop", el.table)

    parsed = assemble(synthetic_document(), AssemblyOptions(table_transcriber=hook))
    [table] = [
        c
        for c in chunk_document(parsed, format_profile="clinical_protocol")
        if c["chunk_type"] == "table"
    ]
    assert seen == ["ocr"]  # only the OCR table is sent
    assert table["meta"]["text_origins"] == ["vlm_transcription"]
    assert table["meta"]["review_status"] == "pending"
    assert "vlm_transcription" in table["meta"]["review_reasons"]
    assert table["meta"]["vlm"]["agreement"]["numeric_disagreements"] == 0
    assert "Weight (kg)" in table["meta"]["ocr_alternative"]
    assert "commentary" not in table["text"]


def test_text_layer_table_is_never_sent(tmp_path) -> None:
    doc = synthetic_document()
    table_el = next(el for el in doc.pages[1].elements if el.kind == "table")
    table_el.origin = "text_layer"
    calls = []
    assemble(doc, AssemblyOptions(table_transcriber=lambda el: calls.append(el) or None))
    assert calls == []


# ── gateway client (§18.2) ──


def _gateway(handler) -> LLMGateway:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        model_id="chat-test-model",
        llm_gateway_url="http://llm-gateway:8080",
        llm_gateway_api_key="k-test",
        ingest_vlm_max_retries=1,
    )
    gw = LLMGateway(settings)
    gw._client = httpx.Client(
        base_url=settings.llm_gateway_url, transport=httpx.MockTransport(handler)
    )
    return gw


def test_generate_with_image_sends_multipart_with_bearer_and_parses(monkeypatch) -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = request.content
        return httpx.Response(
            200,
            json={
                "response": "{}",
                "model": MODEL,
                "timestamp": "t",
                "metrics": {"done_reason": "stop", "latency_ms": 5},
            },
        )

    result = _gateway(handler).generate_with_image(b"\x89PNG-bytes", "transcribe")
    assert seen["path"] == "/generate-with-image"
    assert seen["auth"] == "Bearer k-test"
    assert b'name="image"' in seen["body"] and b'name="prompt"' in seen["body"]
    assert result.model == MODEL and result.done_reason == "stop"


def test_429_is_retried_and_413_is_not(monkeypatch) -> None:
    monkeypatch.setattr("app.llm.gateway.time.sleep", lambda s: None)
    codes = iter([429, 200])

    def flaky(request: httpx.Request) -> httpx.Response:
        code = next(codes)
        if code == 429:
            return httpx.Response(429, headers={"Retry-After": "1"})
        return httpx.Response(200, json={"response": "{}", "model": MODEL, "metrics": {}})

    assert _gateway(flaky).generate_with_image(b"png", "p").model == MODEL

    calls = []

    def too_big(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(413)

    with pytest.raises(LLMGatewayError, match="413"):
        _gateway(too_big).generate_with_image(b"png", "p")
    assert len(calls) == 1
