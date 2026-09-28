"""Layout-aware parse entry point with the `pypdf` fallback (ARCH-044, ARCH §5.1
step 2; LAYOUT-INGESTION-PROPOSAL.md §4).

`parse_with_layout(path, manifest_entry)` returns a `ParsedDocument` exactly
like `app.ingestion.pdf_parse.parse_document`, so the rest of the pipeline
(page provenance, chunking, persistence) is unchanged.

- Manifest options (ARCH-038 extension): `boilerplate_patterns`,
  `text_corrections`, `flowchart_attestations`, and the existing
  `source_pages`.
- **Corrections fail closed**: a malformed, out-of-scope or non-matching
  correction raises `CorrectionError` and the document is not ingested. It
  never falls back to `pypdf`, because that would silently ingest the
  uncorrected text the operator asked to fix.
- **Parser failure falls back**: any other exception from Docling/pdfplumber
  logs a warning and uses `pypdf`, with `parse_quality` capped at 0.5 so the
  document is held for admin review (below `INGEST_MIN_PARSE_QUALITY`).
- Figure/table crops are written content-addressed to `INGEST_CROP_DIR` for
  the reviewer UI.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from pathlib import Path

from app.config import get_settings
from app.ingestion.corrections import CorrectionError, load_corrections
from app.ingestion.layout.assemble import AssemblyOptions, assemble
from app.ingestion.layout.model import Element, TableData
from app.ingestion.layout.vlm_tables import TableTranscriber, VisionCall
from app.ingestion.pdf_parse import ParsedDocument, parse_document

logger = logging.getLogger(__name__)

FALLBACK_QUALITY_CAP = 0.5


def save_crops(crops: dict[str, bytes], crop_dir: str | Path) -> int:
    d = Path(crop_dir)
    d.mkdir(parents=True, exist_ok=True)
    written = 0
    for sha, png in crops.items():
        path = d / f"{sha}.png"
        if not path.exists():
            path.write_bytes(png)
            written += 1
    return written


def assembly_options(manifest_entry: dict | None) -> AssemblyOptions:
    s = get_settings()
    entry = manifest_entry or {}
    return AssemblyOptions(
        margin_zone=s.ingest_margin_zone,
        repeat_ratio=s.ingest_boilerplate_repeat_ratio,
        boilerplate_min_pages=s.ingest_boilerplate_min_pages,
        boilerplate_patterns=list(entry.get("boilerplate_patterns") or []),
        table_max_tokens=s.ingest_table_max_tokens,
        images_scale=s.ingest_images_scale,
        low_density_chars=s.ingest_ocr_force_page_below_chars,
        source_pages=list(entry["source_pages"]) if entry.get("source_pages") else None,
        corrections=load_corrections(entry.get("text_corrections")),
        flowchart_attestations=list(entry.get("flowchart_attestations") or []),
    )


def parse_with_layout(path: str, manifest_entry: dict | None = None) -> ParsedDocument:
    """Layout-parse one guideline file. Non-PDF sources go straight to
    `parse_document` (markdown needs no layout analysis)."""
    if Path(path).suffix.lower() != ".pdf":
        return parse_document(path)

    s = get_settings()
    opts = assembly_options(manifest_entry)  # CorrectionError propagates: fail closed
    try:
        from app.ingestion.layout.docling_adapter import parse_layout  # noqa: PLC0415

        layout = parse_layout(
            path,
            ocr_engine=s.ingest_ocr_engine,
            images_scale=s.ingest_images_scale,
            lang=((manifest_entry or {}).get("language") or "en").split("-")[0],
        )
    except CorrectionError:
        raise
    except Exception as exc:
        logger.warning(
            "layout parser failed on %s (%s: %s); falling back to pypdf with parse_quality "
            "capped at %.1f so the document is held for admin review",
            path,
            type(exc).__name__,
            exc,
            FALLBACK_QUALITY_CAP,
            exc_info=True,
        )
        parsed = parse_document(path)
        parsed.parse_quality = min(parsed.parse_quality, FALLBACK_QUALITY_CAP)
        parsed.parser_version = "pypdf-fallback"
        parsed.parse_report = {"parser": "pypdf-fallback", "error": f"{type(exc).__name__}: {exc}"}
        return parsed

    transcriber = build_table_transcriber(path, layout.crops) if s.vision_enabled else None
    if transcriber is not None:
        opts.table_transcriber = transcriber[0]
    parsed = assemble(layout, opts)
    if transcriber is not None:
        parsed.parse_report["vlm_tables"] = transcriber[1].report
    parsed.parse_report["crops_written"] = save_crops(layout.crops, s.ingest_crop_dir)
    return parsed


def build_table_transcriber(
    path: str, crops: dict[str, bytes], *, call: VisionCall | None = None
) -> tuple[Callable[[Element], tuple[TableData, dict] | None], TableTranscriber]:
    """The vision-LLM table transcription hook (D12, proposal §18.4): renders
    each OCR table's region at `INGEST_VLM_CROP_SCALE` and transcribes it.
    The rendered crop is stored with the other crops so the reviewer sees
    exactly the image the model saw."""
    from app.ingestion.layout.docling_adapter import render_region  # noqa: PLC0415

    s = get_settings()
    if call is None:
        from app.llm.gateway import LLMGateway  # noqa: PLC0415

        call = LLMGateway().generate_with_image
    transcriber = TableTranscriber(
        call,
        expected_model=s.vision_model_id,
        cache_dir=s.ingest_vlm_cache_dir,
        refresh=s.ingest_vlm_refresh,
    )

    def hook(el: Element) -> tuple[TableData, dict] | None:
        assert el.table is not None
        png = render_region(
            path,
            el.page_no,
            el.bbox,
            scale=s.ingest_vlm_crop_scale,
            pad=s.ingest_vlm_crop_pad_pt,
            max_bytes=int(s.ingest_vlm_max_image_mb * 1024 * 1024),
        )
        crops[hashlib.sha256(png).hexdigest()] = png
        return transcriber.transcribe(png, el.table)

    return hook, transcriber
