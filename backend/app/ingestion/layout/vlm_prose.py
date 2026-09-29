"""Vision-LLM transcription of OCR prose (ARCH-044 D12 extended to prose,
operator decision 2026-09-28; DEVIATIONS.md #222).

Prose with no text layer — paragraphs, list items, captions and footnotes
that were OCR'd (Kenya MoH p. 48 notes, the MoH guideline's scanned
passages) — is re-transcribed from its own crop by the gateway's vision
model, exactly like OCR tables (`vlm_tables`), and held to the same rules:

- strict single-key schema (`{"text": "..."}`), model-identity and
  truncation checks, prose around the JSON rejected;
- **cross-checked against the OCR reading**: below `MIN_SIMILARITY`
  character similarity, or outside the `MIN_LENGTH_RATIO`-`MAX_LENGTH_RATIO`
  length band, the transcription is rejected (neighbouring text absorbed, a
  wrong crop or an invented passage) and the OCR text is kept; every number that differs
  between the two readings is a numeric disagreement, listed for the
  reviewer;
- the element's origin becomes `vlm_transcription`, so the chunk is always
  held until an admin confirms it against the crop the model saw;
- a text-layer element is never sent; headings stay on OCR.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from app.ingestion.layout.vlm_tables import (
    _FENCE_RE,
    _ROUTE_I_RE,
    TableTranscriber,
    TranscriptionError,
    VisionCall,
)

PROMPT_VERSION = "prose_transcribe_v1"
_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / f"{PROMPT_VERSION}.txt"
# A faithful transcription differs from OCR only by misread characters. An
# earlier floor of 0.6 accepted transcriptions that had absorbed a
# neighbouring bullet (crop bleed); 0.8 plus a length band rejects those.
MIN_SIMILARITY = 0.8
MIN_LENGTH_RATIO = 0.8
MAX_LENGTH_RATIO = 1.25
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
_LEADING_BULLET_RE = re.compile(r"^[•▪◦●○■□‣⁃∙·✓✔\-–*]\s*")


class _Prose(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str


def parse_prose(response: object) -> str:
    obj: Any = response
    if isinstance(obj, str):
        text = _FENCE_RE.sub("", obj.strip())
        try:
            obj, end = json.JSONDecoder(strict=False).raw_decode(text)
        except json.JSONDecodeError as exc:
            raise TranscriptionError("response is not JSON (prose is never salvaged)") from exc
        if text[end:].strip(" \n\r\t}]"):
            raise TranscriptionError("text after the JSON object (prose is never salvaged)")
    if not isinstance(obj, dict):
        raise TranscriptionError(f"response is a {type(obj).__name__}, not a JSON object")
    try:
        value = _Prose.model_validate(obj).text
    except ValidationError as exc:
        raise TranscriptionError(f"schema violation: {exc.error_count()} error(s)") from exc
    value = _LEADING_BULLET_RE.sub("", " ".join(value.split()), count=1)
    if not value:
        raise TranscriptionError("empty transcription")
    return value


def _numbers(s: str) -> Counter[str]:
    return Counter(_NUMBER_RE.findall(_ROUTE_I_RE.sub("I", s)))


def compare(vlm: str, ocr: str) -> dict:
    """Similarity and number-by-number comparison with the OCR reading."""
    ocr_norm = _LEADING_BULLET_RE.sub("", " ".join(ocr.split()), count=1)
    similarity = difflib.SequenceMatcher(None, ocr_norm.lower(), vlm.lower()).ratio()
    v_nums, o_nums = _numbers(vlm), _numbers(ocr_norm)
    return {
        "similarity": round(similarity, 3),
        "numeric_disagreements": sum(((v_nums - o_nums) + (o_nums - v_nums)).values()),
        "vlm_only_numbers": sorted((v_nums - o_nums).elements()),
        "ocr_only_numbers": sorted((o_nums - v_nums).elements()),
    }


class ProseTranscriber(TableTranscriber):
    """Same cache, model and truncation checks as the table transcriber."""

    def __init__(
        self,
        call: VisionCall,
        *,
        expected_model: str,
        cache_dir: str | Path,
        refresh: bool = False,
    ) -> None:
        super().__init__(call, expected_model=expected_model, cache_dir=cache_dir, refresh=refresh)
        self._prompt_version = PROMPT_VERSION
        self._prompt = _PROMPT_PATH.read_text(encoding="utf-8")

    def transcribe_prose(self, png: bytes, ocr_text: str) -> tuple[str, dict] | None:
        self.report["attempted"] += 1
        crop_sha = hashlib.sha256(png).hexdigest()
        try:
            answer = self._checked_answer(png)
            text = parse_prose(answer.get("response"))
            check = compare(text, ocr_text)
            if check["similarity"] < MIN_SIMILARITY:
                raise TranscriptionError(
                    f"similarity {check['similarity']} to the OCR reading is below {MIN_SIMILARITY}"
                )
            ratio = len(text) / max(len(ocr_text.strip()), 1)
            if not MIN_LENGTH_RATIO <= ratio <= MAX_LENGTH_RATIO:
                raise TranscriptionError(
                    f"length {ratio:.2f}x the OCR reading (extra or missing text; "
                    f"allowed {MIN_LENGTH_RATIO}-{MAX_LENGTH_RATIO}x)"
                )
        except TranscriptionError as exc:
            self.report["rejected"].append({"crop_sha256": crop_sha, "reason": str(exc)})
            return None
        except Exception as exc:  # gateway down, timeout, …: keep OCR
            self.report["rejected"].append(
                {"crop_sha256": crop_sha, "reason": f"{type(exc).__name__}: {exc}"}
            )
            return None
        self.report["accepted"] += 1
        meta = {
            "model": answer["model"],
            "prompt_version": PROMPT_VERSION,
            "done_reason": answer.get("done_reason"),
            "crop_sha256": crop_sha,
            "agreement": check,
        }
        return text, {"vlm": meta, "ocr_alternative": ocr_text}
