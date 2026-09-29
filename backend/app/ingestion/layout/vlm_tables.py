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

# v1 (per-cell objects with col/row spans) was replaced after its first live
# run: the verbose output ran past the gateway's output cap on an 8x12 table,
# and the model's row spans contradicted its own next row (DEVIATIONS.md #219).
PROMPT_VERSION = "table_transcribe_v2"
_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / f"{PROMPT_VERSION}.txt"
ILLEGIBLE = "[illegible]"
MAX_COLUMN_DIFFERENCE = 1  # TableFormer mis-splits titles; more than this is a real mismatch

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)
_DIGITS_RE = re.compile(r"\d")


def load_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


class TranscriptionError(ValueError):
    """The model's answer can't be used; the OCR table is kept."""


class TableTranscription(BaseModel):
    """Every row is exactly `columns` strings: a column-spanning cell repeats
    its text, a row-spanning cell gives its text in its first row and ""
    below. There are no span fields, so nothing can overlap."""

    model_config = ConfigDict(extra="forbid")
    title: list[str] = Field(default_factory=list)
    columns: int = Field(ge=1)
    header_rows: list[list[str]] = Field(default_factory=list)
    body_rows: list[list[str]] = Field(min_length=1)


def parse_transcription(response: object) -> TableTranscription:
    """Accept the gateway's `response` in any of its three forms and return a
    validated transcription, or raise `TranscriptionError`."""
    obj: Any = response
    if isinstance(obj, str):
        text = _FENCE_RE.sub("", obj.strip())
        try:
            obj, end = json.JSONDecoder(strict=False).raw_decode(text)
        except json.JSONDecodeError as exc:
            raise TranscriptionError("response is not JSON (prose is never salvaged)") from exc
        # A complete object followed only by stray closing brackets (seen
        # live: "…]]}}") is a syntax slip, not extra content. Anything else
        # after the object — prose included — is rejected.
        if text[end:].strip(" \n\r\t}]"):
            raise TranscriptionError("text after the JSON object (prose is never salvaged)")
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
    if t.title:  # one title row, as printed lines joined (a table has one title)
        title = " ".join(" ".join(line.split()) for line in t.title)
        cells.append(TableCell(row, 0, 1, t.columns, title, is_header=True))
        row += 1
    for kind_rows, is_header in ((t.header_rows, True), (t.body_rows, False)):
        for r in kind_rows:
            for col, text in enumerate(r):
                cells.append(TableCell(row, col, 1, 1, " ".join(text.split()), is_header=is_header))
            row += 1
    return TableData(num_rows=row, num_cols=t.columns, cells=cells)


def _check_grid(t: TableTranscription) -> None:
    """Every row has exactly `columns` cells."""
    for kind, rows in (("header", t.header_rows), ("body", t.body_rows)):
        for r, row in enumerate(rows):
            if len(row) != t.columns:
                raise TranscriptionError(
                    f"{kind} row {r} has {len(row)} cells, expected {t.columns}"
                )


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


# OCR reads the letter I as the digit 1 in route abbreviations ("1.V",
# "1.M"). Such a swap is still listed for the reviewer but is not a numeric
# disagreement.
_ROUTE_I_RE = re.compile(r"\b1(?=\.\s?[VM]\b)")


def _digits(s: str) -> str:
    return "".join(_DIGITS_RE.findall(_ROUTE_I_RE.sub("I", s)))


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

    if aligned:
        _compare_headers(vlm, ocr, diff, stats)

    cells = [c for c in vlm.cells if c.row < vb]
    for r, row in enumerate(merged_body):
        for c, text in enumerate(row):
            cells.append(TableCell(vb + r, c, 1, 1, text))
    return MergeResult(TableData(vlm.num_rows, vlm.num_cols, cells), diff, stats)


def _compare_headers(vlm: TableData, ocr: TableData, diff: list[dict], stats: dict) -> None:
    """Compare each column's full header path (title excluded). Header rows
    can't be aligned cell by cell — the two readings split them differently —
    but a column's path must say the same thing. Seen live: the model shifted
    the route and frequency rows one column left, putting "8 hrly" under a
    drug printed "6 hrly". Differing digits count as a numeric disagreement."""
    from app.ingestion.layout.tables import render_table  # noqa: PLC0415

    v_paths = render_table(vlm, max_tokens=10**9).header_paths
    o_paths = render_table(ocr, max_tokens=10**9).header_paths
    for c, (v, o) in enumerate(zip(v_paths, o_paths, strict=True)):
        if _norm_header(v) == _norm_header(o):
            continue
        numeric = _digits(v) != _digits(o)
        stats["header_disagreements"] = stats.get("header_disagreements", 0) + 1
        stats["numeric_disagreements"] += numeric
        diff.append({"row": "header", "col": c, "ocr": o, "vlm": v, "numeric": numeric})


def _norm_header(s: str) -> str:
    """Ignore joins and spacing; I/1 and O/0 confusions still count as differences."""
    return re.sub(r"[\s·]+", "", s).lower()


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
        self._prompt_version = PROMPT_VERSION
        self._prompt = load_prompt()
        self.report: dict = {"attempted": 0, "accepted": 0, "rejected": [], "cache_hits": 0}

    def _cached_answer(self, png: bytes) -> dict:
        key = f"{hashlib.sha256(png).hexdigest()}_{self._prompt_version}"
        path = self._cache_dir / f"{key}.json"
        if path.exists() and not self._refresh:
            self.report["cache_hits"] += 1
            return json.loads(path.read_text(encoding="utf-8"))
        result = self._call(png, self._prompt)
        answer = {
            "response": result.response,
            "model": result.model,
            "done_reason": result.done_reason,
            "prompt_version": self._prompt_version,
            "at": datetime.now(UTC).isoformat(),
        }
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(answer, ensure_ascii=False), encoding="utf-8")
        return answer

    def _checked_answer(self, png: bytes) -> dict:
        """The (cached) answer, after the model-identity and truncation checks."""
        answer = self._cached_answer(png)
        if answer.get("model") != self._expected_model:
            raise TranscriptionError(
                f"answered by {answer.get('model')!r}, expected {self._expected_model!r}"
            )
        if answer.get("done_reason") == "length":
            raise TranscriptionError("truncated (done_reason=length)")
        return answer

    def transcribe(self, png: bytes, ocr_table: TableData) -> tuple[TableData, dict] | None:
        self.report["attempted"] += 1
        crop_sha = hashlib.sha256(png).hexdigest()
        try:
            answer = self._checked_answer(png)
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
