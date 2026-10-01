// Shared types — kept aligned with the backend's actual Pydantic schemas
// (ARCH-010). Hand-maintained; a generated (openapi-typescript) client is the
// natural follow-up once the API surface stabilizes post-Phase-5.

export type ObservedOutcome = "well_supported" | "missing_info" | "no_guideline" | "escalated";

export type SegmentType = "claim" | "framing";

export interface Citation {
  citation_id: string;
  document_id: string;
  document_title: string;
  version_label: string;
  version_status: "active" | "superseded" | "withdrawn";
  chunk_id: string;
  section_number?: string | null;
  section_path?: string | null;
  page_start: number;
  page_end: number;
  char_start: number;
  char_end: number;
  quote: string;
  // Operator-attested source corrections the quote overlaps (ARCH-044,
  // LAYOUT-INGESTION-PROPOSAL.md §5.11). Always shown with the citation.
  corrections?: CitationCorrection[];
}

export interface CitationCorrection {
  id: string;
  original: string;
  corrected: string;
  kind: string;
  rationale: string;
  attested_by: string;
  attester_role?: string | null;
  attested_on: string;
}

export interface AnswerSegment {
  type: SegmentType;
  text: string;
  citation_ids: string[];
  grounding_note?: string | null;
}

export interface EscalationInfo {
  escalation_id: string;
  trigger_code: string;
  message: string;
}

export interface QueryResponse {
  conversation_id: string;
  message_id: string;
  observed_outcome: ObservedOutcome;
  scope_label: string;
  segments: AnswerSegment[];
  citations: Citation[];
  escalation?: EscalationInfo | null;
  disclaimer: string; // always present (ARCH-037)
  // One notice per attested source correction quoted (ARCH-044 §5.11).
  correction_notices?: string[];
}

// A guideline chunk held for admin review (ARCH-044): OCR'd numbers or a
// low-parse-quality document. Confirmed chunks become retrievable.
export interface HeldChunk {
  id: string;
  document_version_id: string;
  document_title: string | null;
  version_label: string | null;
  section_path?: string | null;
  page_start: number;
  page_end: number;
  chunk_type: string;
  text: string;
  figure_ref?: { page?: number; bbox?: number[]; image_sha256?: string | null } | null;
  meta: Record<string, unknown>;
  review_reasons: string[];
}

// Async /query job handle (DEVIATIONS.md #94).
export interface QueryJobAccepted {
  job_id: string;
  conversation_id: string;
  status: "pending";
}

export interface QueryJobStatus {
  job_id: string;
  status: "pending" | "done" | "failed";
  result?: QueryResponse | null;
  error?: string | null;
}

// Rubric (rank mode) — 11 domains, 5-point Likert (PRD-040).
export interface RubricDomain {
  code: string;
  ordinal: number;
  name: string;
  definition: string;
  anchor_1: string;
  anchor_2: string;
  anchor_3: string;
  anchor_4: string;
  anchor_5: string;
  required: boolean;
}

// Accept axis (PRD-032; DEVIATIONS.md #84 added out_of_scope as a 4th action).
export type AcceptAction = "full_accept" | "partial_accept" | "reject" | "out_of_scope";

// Local (client-only) form state for the accept-axis fieldset
// (components/AcceptAxisControls.tsx) — camelCase, translated to the
// snake_case API field names at the two call sites that submit it.
export interface AcceptAxisValue {
  action: AcceptAction;
  editedAnswer: string;
  reasonCode: string;
}

// Rank mode + accept axis, submitted together (ARCH §13.2 "Both axes
// together", DEVIATIONS.md #99).
export interface RatingSubmission {
  scores: Record<string, number>;
  comment: string;
  accept: AcceptAxisValue;
}

// GET /review-queue item (rank mode).
export interface QueueResult {
  result_id: string;
  provenance: string;
  expected_outcome: string | null;
  observed_outcome: string | null;
}

// GET /review-queue/{id} detail.
export interface QueueResultDetail {
  result_id: string;
  provenance: string;
  expected_outcome: string | null;
  // The auto-generated (or clinician-submitted) question this result
  // answers; null for a result with no linked `eval_question` row
  // (DEVIATIONS.md #120).
  question: string | null;
  answer: string | null;
  // Per-segment citation_ids for inline rendering (DEVIATIONS.md #120);
  // null for a result written before this existed, or an escalated result
  // with no released answer — callers fall back to `answer`/`citations`.
  segments: AnswerSegment[] | null;
  citations: Citation[];
  grounding_report: Record<string, unknown>;
}

export interface RatingRoundSubmitResponse {
  rating_round_id: string;
  accept_action_id: string;
}

// GET /hitl/escalations item (DEVIATIONS.md #97).
export interface EscalationSummary {
  id: string;
  conversation_id: string | null;
  trigger_code: string;
  state: "open" | "in_review" | "resolved";
  created_at: string;
}

// GET /hitl/escalations/{id} detail.
export interface EscalationDetail {
  id: string;
  conversation_id: string | null;
  trigger_code: string;
  trigger_detail: Record<string, unknown>;
  candidate_answer: string | null;
  state: "open" | "in_review" | "resolved";
  resolution: string | null;
  resolved_at: string | null;
}

export interface HitlDecisionResponse {
  decision_id: string;
  action: AcceptAction;
}

// Auth (ARCH-011).
export interface DevLoginResponse {
  access_token: string;
  token_type: "bearer";
  user_id: string;
  roles: string[];
}
