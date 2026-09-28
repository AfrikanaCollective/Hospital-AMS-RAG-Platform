import type { QueryResponse } from "../types";
import AnswerSegments from "./AnswerSegments";

// Renders a released answer (segments + citations) OR an escalation notice.
// Never renders an ungrounded answer (backend guarantees this — PRD-NFR-2).
export default function AnswerView({ resp }: { resp: QueryResponse }) {
  if (resp.escalation) {
    return (
      <div className="rounded-md border border-danger p-3">
        <strong className="text-ink">Sent for clinician review</strong>
        <p className="my-1.5 text-ink">{resp.escalation.message}</p>
        <small className="text-ink-muted">
          trigger: {resp.escalation.trigger_code} · outcome: {resp.observed_outcome}
        </small>
      </div>
    );
  }

  if (resp.observed_outcome === "no_guideline") {
    return (
      <div className="rounded-md border border-border p-3 text-ink">
        No matching guideline was retrieved for this question. The system does not
        answer from general knowledge.
      </div>
    );
  }

  return (
    <>
      {(resp.correction_notices ?? []).length > 0 && (
        <ul className="mb-2 rounded-md border border-danger/60 p-2 text-[13px] text-ink">
          {(resp.correction_notices ?? []).map((n) => (
            <li key={n}>{n}</li>
          ))}
        </ul>
      )}
      <AnswerSegments segments={resp.segments} citations={resp.citations} />
    </>
  );
}
