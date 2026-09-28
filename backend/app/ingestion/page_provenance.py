"""Page-provenance remapping for a guideline document ingested as a partial
page-range extract of a larger source publication (ARCH-038 extension,
DEVIATIONS.md #166).

ARCH-038 already requires ingest metadata to be operator-attested via the
manifest, never inferred from the PDF. This adds one more attested field,
`source_pages`: the true page numbers (in the original publication) that
this file's own physical pages 1..N correspond to, in order. Without it, a
document that is really "pages 42-45 of a 100-page guideline" would have its
chunks' `page_start`/`page_end` computed relative to the *extract's own*
page count (1..4) — silently wrong citations pointing a clinician at the
wrong page of the real source document.

Fails closed on any manifest/file mismatch (page-count mismatch, malformed
entries) rather than ingesting with silently wrong page citations, matching
`app.ingestion.records`' attestation-gate pattern.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.ingestion.pdf_parse import ParsedDocument


class PageProvenanceError(ValueError):
    """A `source_pages` manifest entry doesn't match the actual document —
    fails closed rather than ingesting with silently wrong page citations."""


def load_manifest_entry(sample_guidelines_dir: str | Path, filename: str) -> dict | None:
    """The whole manifest entry for `filename` (ARCH-038), or `None`. The layout
    parser reads its per-document options from it (`text_corrections`,
    `boilerplate_patterns`, `flowchart_attestations`, `source_pages`)."""
    manifest_path = Path(sample_guidelines_dir) / "manifest.json"
    if not manifest_path.exists():
        return None
    files = json.loads(manifest_path.read_text(encoding="utf-8")).get("files", {})
    return files.get(filename)


def load_source_pages(sample_guidelines_dir: str | Path, filename: str) -> list[int] | None:
    """`source_pages` for `filename` from `manifest.json`, or `None` if there
    is no manifest, no entry for this file, or the entry doesn't declare
    `source_pages` — the common case of a document ingested in full, which
    needs no remapping."""
    manifest_path = Path(sample_guidelines_dir) / "manifest.json"
    if not manifest_path.exists():
        return None
    files = json.loads(manifest_path.read_text(encoding="utf-8")).get("files", {})
    entry = files.get(filename)
    if not entry:
        return None
    source_pages = entry.get("source_pages")
    if not source_pages:
        return None
    return list(source_pages)


def apply_source_pages(doc: ParsedDocument, source_pages: list[int]) -> None:
    """Remap `doc` in place: local page `i` (1-indexed) -> `source_pages[i-1]`.
    Every downstream page number (`Chunk.page_start`/`page_end`, computed via
    `doc.page_for_offset` at chunk-emit time) reflects the remap automatically
    once `doc.page_starts` is rewritten here — this must run before
    `chunk_document(doc, ...)`, not after.

    Fails closed: raises `PageProvenanceError` rather than ingesting a
    manifest that doesn't actually match the file it describes."""
    if len(source_pages) != doc.page_count:
        raise PageProvenanceError(
            f"manifest source_pages has {len(source_pages)} entries but the "
            f"document has {doc.page_count} page(s) — fix the manifest or the file"
        )
    if not all(isinstance(p, int) and p > 0 for p in source_pages):
        raise PageProvenanceError("source_pages must be positive integers")
    if source_pages != sorted(source_pages):
        raise PageProvenanceError(
            "source_pages must be in ascending (or equal-run) order, matching "
            "the extracted document's own physical page order"
        )

    doc.page_starts = [
        (source_pages[i], char_start) for i, (_, char_start) in enumerate(doc.page_starts)
    ]
    for s in doc.sections:
        s["page_start"] = doc.page_for_offset(s["char_start"])
        s["page_end"] = doc.page_for_offset(max(s["char_end"] - 1, s["char_start"]))
