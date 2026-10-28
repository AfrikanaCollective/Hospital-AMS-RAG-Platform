import { useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import ReviewQueue from "../components/ReviewQueue";
import Button from "../components/ui/Button";
import type { ReviewQueueView } from "../types";
import { api, ApiError } from "../api/client";

// Reviewer rank-mode queue (ARCH §13, §14.2). Each case is rated on its own
// page (`RatePage`, /review/:resultId) — the panel used to open at the foot
// of this list, far from the row tapped, on a phone (DEVIATIONS.md #294).
//
// What is listed is the server's rule (`app.rubric.assignment`): items other
// reviewers have started come first, then the reviewer's own batch of fresh
// items, kept until all are rated; while a half-finished rating exists only
// its item is offered and it must be submitted before any other.
//
// Standalone escalation resolution lives on its own page
// (`EscalationsPage.tsx`, DEVIATIONS.md #120).
export default function ReviewPage() {
  const navigate = useNavigate();
  const notice = (useLocation().state as { notice?: string } | null)?.notice ?? null;
  const [view, setView] = useState<ReviewQueueView | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .getReviewQueue()
      .then(setView)
      .catch((e) => setError(e instanceof ApiError ? e.message : String(e)));
  }, []);

  const open = (resultId: string) => navigate(`/review/${resultId}`);

  return (
    <section>
      <h2 className="text-base font-semibold text-ink">Review queue</h2>
      <p className="text-[13px] text-ink-muted">
        For each case: the 11-domain rubric and an accept-axis decision, submitted together. A case
        stays in the queue until 3 reviewers have rated it.
      </p>

      {notice && <p className="mt-2 text-[13px] text-accent-strong">{notice}</p>}
      {view?.draft_dropped && (
        <p role="status" className="mt-2 text-[13px] text-ink">
          Your half-finished rating was removed: three other reviewers completed that case first, so
          it can no longer be rated.
        </p>
      )}
      {error && <p className="mt-2 text-[13px] text-danger">{error}</p>}
      {!view && !error && <p className="mt-3 text-[13px] text-ink-muted">Loading…</p>}

      {view && <QueueHeading view={view} onContinue={open} />}
      {view && view.mode !== "empty" && (
        <div className="mt-3">
          <ReviewQueue items={view.items} selectedId={null} onSelect={open} />
        </div>
      )}
    </section>
  );
}

function QueueHeading({
  view,
  onContinue,
}: {
  view: ReviewQueueView;
  onContinue: (resultId: string) => void;
}) {
  if (view.mode === "draft" && view.draft_result_id) {
    const id = view.draft_result_id;
    return (
      <div className="mt-3 rounded-lg border border-accent-strong bg-accent/10 p-3">
        <p className="text-[14px] text-ink">
          You have a half-finished rating. Finish and submit it before choosing another case.
        </p>
        <Button
          type="button"
          variant="primary"
          onClick={() => onContinue(id)}
          className="mt-2 w-full sm:w-auto"
        >
          Continue rating
        </Button>
      </div>
    );
  }
  if (view.mode === "preview")
    return (
      <div className="mt-3">
        <h3 className="text-[15px] font-semibold text-ink">Preview: every open case</h3>
        <p className="text-[13px] text-ink-muted">
          {view.items.length} cases. This account can open and step through any of them, but is
          never assigned one and cannot save or submit a rating.
        </p>
      </div>
    );
  if (view.mode === "in_progress")
    return (
      <div className="mt-3">
        <h3 className="text-[15px] font-semibold text-ink">Started by other reviewers</h3>
        <p className="text-[13px] text-ink-muted">
          Please rate these first. Your own batch continues after them.
        </p>
      </div>
    );
  if (view.mode === "batch" && view.batch)
    return (
      <div className="mt-3">
        <h3 className="text-[15px] font-semibold text-ink">Your batch</h3>
        <p className="text-[13px] text-ink-muted">
          {view.batch.done} of {view.batch.total} rated
        </p>
      </div>
    );
  return <p className="mt-3 text-[14px] text-ink">Nothing to review right now.</p>;
}
