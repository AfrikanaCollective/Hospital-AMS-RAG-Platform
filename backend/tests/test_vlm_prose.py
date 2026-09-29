"""Vision-LLM transcription of OCR prose (ARCH-044 D12 extended to prose;
DEVIATIONS.md #222). Offline: fake gateway, synthetic layout fixture."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from app.ingestion.chunking import chunk_document
from app.ingestion.layout.assemble import AssemblyOptions, assemble
from app.ingestion.layout.vlm_prose import ProseTranscriber, compare, parse_prose
from app.ingestion.layout.vlm_tables import TranscriptionError
from tests.layout_fixtures import synthetic_document

MODEL = "vision-test-model"
OCR_TEXT = "Dose already adjusted for weight in days"


@dataclass
class _Answer:
    response: object
    model: str = MODEL
    done_reason: str | None = "stop"


def _transcriber(response: object, tmp_path, *, model: str = MODEL) -> ProseTranscriber:
    return ProseTranscriber(
        lambda png, prompt: _Answer(response, model=model),
        expected_model=MODEL,
        cache_dir=tmp_path,
    )


@pytest.mark.parametrize(
    "response",
    [
        {"text": "• Dose already adjusted for weight in DAYS"},
        json.dumps({"text": "Dose already adjusted for weight in DAYS"}),
        '```json\n{"text":"Dose already adjusted for weight in DAYS"}\n```',
    ],
)
def test_prose_response_shapes_parse_and_leading_bullet_is_dropped(response) -> None:
    assert parse_prose(response) == "Dose already adjusted for weight in DAYS"


@pytest.mark.parametrize(
    "response",
    ["The image shows a note about dosing.", {"text": "x", "summary": "y"}, {"text": "  "}],
)
def test_unusable_prose_responses_are_rejected(response) -> None:
    with pytest.raises(TranscriptionError):
        parse_prose(response)


def test_numbers_are_compared_and_route_i1_is_not_a_number() -> None:
    check = compare(
        "Amoxycillin 50mg/kg/dose 12hrly i.e. 100mg/kg/day",
        "Amoxycillin sOmg/kg/dose 12hrly i.e. 1Omg/kg/day",
    )
    assert check["numeric_disagreements"] > 0
    assert "50" in check["vlm_only_numbers"]
    assert compare("give I.V over 2-3 mins", "give 1.V over 2-3 mins")["numeric_disagreements"] == 0


def test_low_similarity_transcription_is_rejected_and_ocr_kept(tmp_path) -> None:
    tr = _transcriber({"text": "Completely different passage about feeding schedules"}, tmp_path)
    assert tr.transcribe_prose(b"crop", OCR_TEXT) is None
    assert "similarity" in tr.report["rejected"][0]["reason"]


def test_transcription_that_absorbed_a_neighbouring_bullet_is_rejected(tmp_path) -> None:
    """Seen live (crop bleed): the model appended the next bullet."""
    tr = _transcriber(
        {"text": "Kanamycin or Spectinomycin 25mg/kg (Max 75 mg) IM, or Ceftriaxone 50mg/kg"},
        tmp_path,
    )
    assert (
        tr.transcribe_prose(b"crop", "Kanamycin or Spectinomycin 25mg/kg (Max 75 mg) IM, or")
        is None
    )
    assert "length" in tr.report["rejected"][0]["reason"]


def test_ocr_misreads_are_within_tolerance(tmp_path) -> None:
    ocr = "Amoxycillin dispersible table – sOmg/kg/dose 12hrly i.e. 1Omg/kg/day divided in 2 doses"
    tr = _transcriber(
        {
            "text": "Amoxycillin dispersible table – 50mg/kg/dose 12hrly i.e. "
            "100mg/kg/day divided in 2 doses"
        },
        tmp_path,
    )
    result = tr.transcribe_prose(b"crop", ocr)
    assert result is not None and result[1]["vlm"]["agreement"]["numeric_disagreements"] > 0


def test_wrong_model_is_rejected(tmp_path) -> None:
    tr = _transcriber({"text": OCR_TEXT}, tmp_path, model="other-model")
    assert tr.transcribe_prose(b"crop", OCR_TEXT) is None


def test_ocr_list_item_is_transcribed_and_always_held(tmp_path) -> None:
    tr = _transcriber({"text": "Dose already adjusted for weight in DAYS"}, tmp_path)
    sent = []

    def hook(el):
        sent.append((el.kind, el.origin))
        return tr.transcribe_prose(b"crop-" + el.text.encode(), el.text)

    parsed = assemble(synthetic_document(), AssemblyOptions(prose_transcriber=hook))
    chunks = chunk_document(parsed, format_profile="clinical_protocol")
    [held] = [c for c in chunks if "weight in DAYS" in c["text"]]
    assert sent == [("list_item", "ocr")]  # text-layer prose and headings are never sent
    assert held["text"].startswith("• Dose already adjusted")
    assert held["meta"]["review_status"] == "pending"
    assert "vlm_transcription" in held["meta"]["review_reasons"]
    [p] = held["meta"]["prose_vlm"]
    assert p["ocr_alternative"] == OCR_TEXT and p["agreement"]["numeric_disagreements"] == 0
