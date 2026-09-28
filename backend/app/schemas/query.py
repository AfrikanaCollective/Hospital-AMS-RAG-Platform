"""Query request/response models (PRD-011..PRD-016, PRD-087, PRD-105; ARCH §8.2).

The answer is an ordered list of segments. A claim segment carries >= 1 citation
and a verbatim quote; a framing segment is non-claim connective text. Every
response carries a non-removable disclaimer (ARCH-037) — the API will refuse to
emit a response payload without it.

`QueryJobAccepted`/`QueryJobStatus` back the async pipeline (`POST /query/async`,
`GET /query/jobs/{job_id}`, DEVIATIONS.md #94) — the same eventual `QueryResponse`
shape, just reached via a Celery job handle instead of inline.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from app.schemas.citation import Citation, correction_notices
from app.schemas.enums import (
    EscalationTrigger,
    ObservedOutcome,
    ScopeLabel,
    SegmentType,
)

DISCLAIMER_TEXT = (
    "Reported guideline content for clinician reference only. This is not medical "
    "advice and does not constitute an independent clinical recommendation. "
    "Clinical judgement remains with the treating clinician."
)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    conversation_id: str | None = None
    patient_id: str | None = Field(
        default=None,
        description="At most one patient per conversation (PRD-NG-011). "
        "Requires purpose-of-use and authorization.",
    )
    hospital_constraint: str | None = Field(
        default=None,
        description="Optional local constraint (SCOPE-2.5). Only surfaces an "
        "alternative already present in retrieved guideline text; otherwise escalates.",
    )


class AnswerSegment(BaseModel):
    type: SegmentType
    text: str
    citation_ids: list[str] = Field(default_factory=list)  # required (>=1) iff type == CLAIM
    grounding_note: str | None = None  # e.g. "weakly supported", "removed for lack of support"


class EscalationInfo(BaseModel):
    escalation_id: str
    trigger_code: EscalationTrigger
    message: str  # user-facing explanation; never a recommendation


class QueryResponse(BaseModel):
    conversation_id: str
    message_id: str
    observed_outcome: ObservedOutcome
    scope_label: ScopeLabel

    # Present when an answer was released.
    segments: list[AnswerSegment] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)

    # Present when escalated / withheld. Mutually informative with observed_outcome.
    escalation: EscalationInfo | None = None

    # Non-removable (ARCH-037). Always set.
    disclaimer: str = DISCLAIMER_TEXT

    # One notice per attested source correction any citation quotes (ARCH-044,
    # §5.11). Derived from `citations` by the validator below, so it can't be
    # omitted or edited by whoever builds the response.
    correction_notices: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _derive_correction_notices(self) -> QueryResponse:
        self.correction_notices = correction_notices(self.citations)
        return self


class QueryJobAccepted(BaseModel):
    job_id: str
    conversation_id: str
    status: str = "pending"


class QueryJobStatus(BaseModel):
    job_id: str
    status: str  # pending | done | failed
    result: QueryResponse | None = None
    error: str | None = None
