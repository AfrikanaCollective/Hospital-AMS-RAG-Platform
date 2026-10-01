"""Citation object (ARCH-014; PRD-011, PRD-016).

Minimum per constraint #4: document ID + version + section/page + chunk offset.
The `quote` + its offsets are what the grounding check (ARCH-015) and the UI
highlight use, and what makes a citation re-verifiable against stored chunk text.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field

from app.schemas.enums import DocumentVersionStatus


class Citation(BaseModel):
    citation_id: str = Field(description="stable id within one answer, e.g. 'c1'")

    # document identity + version (minimum)
    document_id: str
    document_title: str
    document_version_id: str
    version_label: str
    effective_date: date | None = None
    version_status: DocumentVersionStatus = DocumentVersionStatus.ACTIVE

    # locus (minimum: section/page)
    chunk_id: str
    section_number: str | None = None
    section_path: str | None = None
    page_start: int
    page_end: int

    # chunk offset (minimum: chunk offset)
    char_start: int = Field(ge=0, description="offset within the normalized document text")
    char_end: int = Field(ge=0)

    # verbatim supporting span — substring of the cited chunk's stored text
    quote: str
    quote_char_start: int = Field(ge=0)
    quote_char_end: int = Field(ge=0)

    # Operator-attested corrections the quote overlaps (ARCH-044,
    # LAYOUT-INGESTION-PROPOSAL.md §5.11): the source prints `original`, the
    # stored (and quoted) text reads `corrected`. Always displayed with the
    # citation; set by `app.citations.model.build_citation`, never by a model.
    corrections: list[dict] = Field(default_factory=list)


def correction_notice(correction: dict) -> str:
    """The fixed wording shown wherever a corrected quote appears."""
    role = correction.get("attester_role")
    by = f"{correction.get('attested_by')} ({role})" if role else correction.get("attested_by")
    return (
        f"Corrected at ingestion by {by} on "
        f'{correction.get("attested_on")}. The source prints: "{correction.get("original")}"'
    )


def correction_notices(citations: list[Citation] | list[dict]) -> list[str]:
    seen: list[str] = []
    for c in citations:
        corrections = c.get("corrections") or [] if isinstance(c, dict) else c.corrections
        for corr in corrections:
            notice = correction_notice(corr)
            if notice not in seen:
                seen.append(notice)
    return seen
