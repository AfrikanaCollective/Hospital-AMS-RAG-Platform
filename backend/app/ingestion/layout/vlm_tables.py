"""Vision-LLM table transcription (ARCH-044 decision D12, sub-phase 9c;
LAYOUT-INGESTION-PROPOSAL.md §18).

For each table whose cells came from OCR (`INGEST_VLM_TABLES=ocr_only`), the
table's page crop goes to the gateway's image endpoint with a strict
transcription prompt. The model's answer is **source text transcribed by a
model** (origin `vlm_transcription`), and it is handled that way:

1. **Parse** tolerantly (the gateway may return a JSON string, a dict or a
   list; code fences are stripped), then **validate** against a strict
   schema. Prose around the JSON, a truncated answer (`done_reason=length`),
   an inconsistent grid, a body-row count different from TableFormer's, or a
   model other than `VISION_MODEL_ID` rejects the transcription outright.
   Rejected means the OCR table is used unchanged.
2. **Cross-check** every body cell against the OCR cell at the same
   position. Agreeing cells keep the shared value; disagreeing cells take the
   transcription's value, with both kept in `cell_diff`. A cell where the VLM
   is empty or illegible keeps the OCR value. A cell whose **digits** differ
   is a numeric disagreement; the cell is flagged for the reviewer.
3. The merged grid is serialized by the unchanged deterministic
   `render_table`. The chunk is **always held for admin review** (reason
   `vlm_transcription`), agreement or not. A person confirms it against the
   crop before it can be retrieved or cited. Nothing the model says *about*
   the table (notes, commentary) is ever inserted.

Answers are cached by crop hash + prompt version, so a re-ingest reuses the
same transcription (the endpoint exposes no temperature or seed).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.ingestion.layout.model import TableCell, TableData

PROMPT_VERSION = "table_transcribe_v1"
_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / f"{PROMPT_VERSION}.txt"
ILLEGIBLE = "[illegible]"
MAX_COLUMN_DIFFERENCE = 1  # TableFormer mis-splits titles; more than this is a real mismatch

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)
_DIGITS_RE = re.compile(r"\d")


def load_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


class TranscriptionError(ValueError):
    """The model's answer can't be used; the OCR table is kept."""


class _Cell(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str
    col: int = Field(ge=0)
    col_span: int = Field(default=1, ge=1)
    row_span: int = Field(default=1, ge=1)


class TableTranscription(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: list[str] = Field(default_factory=list)
    columns: int = Field(ge=1)
    header_rows: list[list[_Cell]] = Field(default_factory=list)
    body_rows: list[list[_Cell]]
    notes: list[str] = Field(default_factory=list)
    illegible: list[str] = Field(default_factory=list)


def parse_transcription(response: object) -> TableTranscription:
    """Accept the gateway's `response` in any of its three forms and return a
    validated transcription, or raise `TranscriptionError`."""
    obj: Any = response
    if isinstance(obj, str):
        text = _FENCE_RE.sub("", obj.strip())
        try:
            obj = json.loads(text, strict=False)
        except json.JSONDecodeError as exc:
            raise TranscriptionError("response is not JSON (prose is never salvaged)") from exc
    if not isinstance(obj, dict):
        raise TranscriptionError(f"response is a {type(obj).__name__}, not a JSON object")
    if "response" in obj:
        raise TranscriptionError("unexpected top-level 'response' key")
    try:
        return TableTranscription.model_validate(obj)
    except ValidationError as exc:
        raise TranscriptionError(f"schema violation: {exc.error_count()} error(s)") from exc


def to_table_data(t: TableTranscription) -> TableData:
    """Transcription -> `TableData`. Title lines become a first header row
    spanning every column, which `render_table` turns back into the title line."""
    cells: list[TableCell] = []
    row = 0
    for line in t.title:
        cells.append(TableCell(row, 0, 1, t.columns, line, is_header=True))
        row += 1
    for kind_rows, is_header in ((t.header_rows, True), (t.body_rows, False)):
        for r in kind_rows:
            for c in r:
                cells.append(
                    TableCell(row, c.col, c.row_span, c.col_span, c.text, is_header=is_header)
                )
            row += 1
    return TableData(num_rows=row, num_cols=t.columns, cells=cells)


def _check_grid(t: TableTranscription) -> None:
    """Spans in range, no overlapping cells (a row-spanning cell occupies the
    rows below it too)."""
    occupied: set[tuple[int, int]] = set()
    rows = [*t.header_rows, *t.body_rows]
    for r, row in enumerate(rows):
        for c in row:
            if c.col + c.col_span > t.columns:
                raise TranscriptionError(
                    f"cell at row {r} col {c.col} spans past {t.columns} columns"
                )
            for rr in range(r, min(r + c.row_span, len(rows))):
                for cc in range(c.col, c.col + c.col_span):
                    if (rr, cc) in occupied:
                        raise TranscriptionError(f"overlapping cells at row {rr} col {cc}")
                    occupied.add((rr, cc))


def _grid(table: TableData) -> list[list[str]]:
    grid = [[""] * table.num_cols for _ in range(table.num_rows)]
    for cell in table.cells:
        for r in range(cell.row, min(cell.row + cell.row_span, table.num_rows)):
            for c in range(cell.col, min(cell.col + cell.col_span, table.num_cols)):
                grid[r][c] = " ".join(cell.text.split())
    return grid


def _body_start(table: TableData) -> int:
    header_rows = {c.row + k for c in table.cells if c.is_header for k in range(c.row_span)}
    n = 0
    while n in header_rows:
        n += 1
    return n


def _digits(s: str) -> str:
    return "".join(_DIGITS_RE.findall(s))


@dataclass
class MergeResult:
    table: TableData
    cell_diff: list[dict] = field(default_factory=list)
    agreement: dict = field(default_factory=dict)


def merge(vlm: TableData, ocr: TableData) -> MergeResult:
    """Cell-level cross-check (proposal §18.6), aligned by body row and column."""
    vg, og = _grid(vlm), _grid(ocr)
    vb, ob = _body_start(vlm), _body_start(ocr)
    v_body, o_body = vg[vb:], og[ob:]
    if len(v_body) != len(o_body):
        raise TranscriptionError(
            f"body rows differ from the table structure: VLM {len(v_body)} vs OCR {len(o_body)}"
        )
    if abs(vlm.num_cols - ocr.num_cols) > MAX_COLUMN_DIFFERENCE:
        raise TranscriptionError(f"columns differ: VLM {vlm.num_cols} vs OCR {ocr.num_cols}")
    aligned = vlm.num_cols == ocr.num_cols

    stats = {"cells": 0, "agreed": 0, "vlm_only": 0, "ocr_only": 0, "numeric_disagreements": 0}
    diff: list[dict] = []
    merged_body = [list(r) for r in v_body]
    if aligned:
        for r, (vrow, orow) in enumerate(zip(v_body, o_body, strict=True)):
            for c, (v, o) in enumerate(zip(vrow, orow, strict=True)):
                stats["cells"] += 1
                if v == o:
                    stats["agreed"] += 1
                    continue
                if v in ("", ILLEGIBLE) and o:
                    merged_body[r][c] = o
                    stats["ocr_only"] += 1
                    continue
                numeric = _digits(v) != _digits(o)
                stats["vlm_only"] += 1
                stats["numeric_disagreements"] += numeric
                diff.append({"row": r, "col": c, "ocr": o, "vlm": v, "numeric": numeric})
    else:
        # Column counts differ by one (a mis-split title): cells can't be
        # aligned, so nothing is compared; the reviewer checks the whole table.
        stats["unaligned"] = True

    cells = [c for c in vlm.cells if c.row < vb]
    for r, row in enumerate(merged_body):
        for c, text in enumerate(row):
            cells.append(TableCell(vb + r, c, 1, 1, text))
    return MergeResult(TableData(vlm.num_rows, vlm.num_cols, cells), diff, stats)


# (png bytes, prompt) -> object with .response, .model, .done_reason
VisionCall = Callable[[bytes, str], Any]


class TableTranscriber:
    """Transcribes one OCR table from its crop, with cache and all checks.
    `transcribe` returns `(merged_table, meta)` or `None` (keep OCR) and
    records why in `self.report`."""

    def __init__(
        self,
        call: VisionCall,
        *,
        expected_model: str,
        cache_dir: str | Path,
        refresh: bool = False,
    ) -> None:
        self._call = call
        self._expected_model = expected_model
        self._cache_dir = Path(cache_dir)
        self._refresh = refresh
        self._prompt = load_prompt()
        self.report: dict = {"attempted": 0, "accepted": 0, "rejected": [], "cache_hits": 0}

    def _cached_answer(self, png: bytes) -> dict:
        key = f"{hashlib.sha256(png).hexdigest()}_{PROMPT_VERSION}"
        path = self._cache_dir / f"{key}.json"
        if path.exists() and not self._refresh:
            self.report["cache_hits"] += 1
            return json.loads(path.read_text(encoding="utf-8"))
        result = self._call(png, self._prompt)
        answer = {
            "response": result.response,
            "model": result.model,
            "done_reason": result.done_reason,
            "prompt_version": PROMPT_VERSION,
            "at": datetime.now(UTC).isoformat(),
        }
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(answer, ensure_ascii=False), encoding="utf-8")
        return answer

    def transcribe(self, png: bytes, ocr_table: TableData) -> tuple[TableData, dict] | None:
        self.report["attempted"] += 1
        crop_sha = hashlib.sha256(png).hexdigest()
        try:
            answer = self._cached_answer(png)
            if answer.get("model") != self._expected_model:
                raise TranscriptionError(
                    f"answered by {answer.get('model')!r}, expected {self._expected_model!r}"
                )
            if answer.get("done_reason") == "length":
                raise TranscriptionError("truncated (done_reason=length)")
            transcription = parse_transcription(answer.get("response"))
            _check_grid(transcription)
            result = merge(to_table_data(transcription), ocr_table)
        except TranscriptionError as exc:
            self.report["rejected"].append({"crop_sha256": crop_sha, "reason": str(exc)})
            return None
        except Exception as exc:  # gateway down, timeout, …: keep OCR, report it
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
            "agreement": result.agreement,
        }
        return result.table, {"vlm": meta, "cell_diff": result.cell_diff}
