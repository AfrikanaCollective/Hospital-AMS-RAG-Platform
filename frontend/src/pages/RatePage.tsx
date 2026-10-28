import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import RubricForm from "../components/RubricForm";
import AnswerSegments from "../components/AnswerSegments";
import CitationList from "../components/CitationList";
import { AreaBadge } from "../components/ReviewQueue";
import type {
  QueueResultDetail,
  RatingDraftState,
  RatingSubmission,
  ReviewQueueView,
} from "../types";
import { api, ApiError } from "../api/client";

const SAVE_DELAY_MS = 800;

type SaveState = "idle" | "saving" | "saved" | "error";

// One case, full screen (DEVIATIONS.md #294): the generated query and answer,
// then the rubric wizard. Rank mode = the 11-domain rubric AND an
// accept-axis decision in one submission (ARCH §13.2, DEVIATIONS.md #99).
//
// The rubric is saved as the reviewer's one half-finished rating once any
// score or comment is entered, and restored when they come back. It cannot
// be discarded, only submitted; "Submit & next" opens the next case in the
// queue.
export default function RatePage() {
  const { resultId = "" } = useParams();
  // Remount per case, so "Submit & next" starts the next one fresh.
  return <RateCase key={resultId} resultId={resultId} />;
}

function RateCase({ resultId }: { resultId: string }) {
  const navigate = useNavigate();
  const [view, setView] = useState<ReviewQueueView | null>(null);
  const [detail, setDetail] = useState<QueueResultDetail | null>(null);
  const [initial, setInitial] = useState<Partial<RatingDraftState> | undefined>(undefined);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [saveState, setSaveState] = useState<SaveState>("idle");
  const timer = useRef<number | undefined>(undefined);
  // A change not yet sent; flushed if the reviewer leaves before the timer.
  const pending = useRef<RatingDraftState | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const v = await api.getReviewQueue();
        if (cancelled) return;
        // A half-finished rating elsewhere must be finished first.
        if (v.draft_result_id && v.draft_result_id !== resultId) {
          navigate(`/review/${v.draft_result_id}`, { replace: true });
          return;
        }
        setView(v);
        const [d, draft] = await Promise.all([
          api.getReviewQueueItem(resultId),
          v.draft_result_id === resultId ? api.getRatingDraft() : Promise.resolve(null),
        ]);
        if (cancelled) return;
        setDetail(d);
        if (draft) {
          setInitial(draft.payload);
          setSaveState("saved");
        }
        setReady(true);
      } catch (e) {
        if (!cancelled) setError(e instanceof ApiError ? e.message : String(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [resultId, navigate]);

  useEffect(
    () => () => {
      window.clearTimeout(timer.current);
      if (pending.current) void api.saveRatingDraft(resultId, pending.current).catch(() => {});
    },
    [resultId],
  );

  const preview = view?.mode === "preview";

  const onChange = useCallback(
    (state: RatingDraftState) => {
      // A preview account saves nothing (DEVIATIONS.md #323).
      if (preview) return;
      // A draft starts with the first score or comment, not on opening.
      if (!Object.keys(state.scores).length && !state.comment.trim()) return;
      window.clearTimeout(timer.current);
      pending.current = state;
      timer.current = window.setTimeout(() => {
        pending.current = null;
        setSaveState("saving");
        api.saveRatingDraft(resultId, state).then(
          () => setSaveState("saved"),
          () => setSaveState("error"),
        );
      }, SAVE_DELAY_MS);
    },
    [resultId, preview],
  );

  const submit = async (submission: RatingSubmission) => {
    window.clearTimeout(timer.current);
    pending.current = null;
    setBusy(true);
    setError(null);
    try {
      await api.submitRating(resultId, submission);
      const next = await api.getReviewQueue();
      const nextId = next.items[0]?.result_id;
      if (nextId) navigate(`/review/${nextId}`);
      else
        navigate("/review", {
          state: { notice: "Rating submitted. Nothing to review right now." },
        });
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const inQueue = view?.items.some((i) => i.result_id === resultId) ?? false;
  const position = view?.items.findIndex((i) => i.result_id === resultId) ?? -1;
  // A preview account steps through cases without rating them.
  const prevId = preview && position > 0 ? view.items[position - 1].result_id : null;
  const nextId =
    preview && position >= 0 && position < view.items.length - 1
      ? view.items[position + 1].result_id
      : null;

  return (
    <section>
      <div className="flex items-center justify-between gap-2">
        <Link
          className="text-[14px] text-accent-strong underline-offset-2 hover:underline"
          to="/review"
        >
          ← Queue
        </Link>
        {saveState !== "idle" && (
          <span role="status" className="text-[12px] text-ink-muted">
            {saveState === "saving"
              ? "Saving…"
              : saveState === "saved"
                ? "Progress saved"
                : "Not saved — check your connection"}
          </span>
        )}
      </div>

      {error && <p className="mt-3 text-[13px] text-danger">{error}</p>}
      {!ready && !error && <p className="mt-3 text-[13px] text-ink-muted">Loading…</p>}

      {ready && view && !inQueue && (
        <p className="mt-3 text-[14px] text-ink">
          This case is not in your review queue right now.{" "}
          <Link className="text-accent-strong underline" to="/review">
            Back to the queue
          </Link>
        </p>
      )}

      {ready && view && detail && inQueue && (
        <div className="mt-3 space-y-4">
          <div className="flex flex-wrap items-center gap-2">
            <AreaBadge area={detail.query_area} />
            <span className="text-[12px] text-ink-muted">
              {preview
                ? `Preview: case ${position + 1} of ${view.items.length}, nothing is saved`
                : view.mode === "batch" && view.batch
                  ? `Your batch: ${view.batch.done + 1} of ${view.batch.total}`
                  : view.mode === "in_progress"
                    ? `Started by others: ${position + 1} of ${view.items.length}`
                    : "Half-finished rating"}
            </span>
            {preview && (
              <span className="ml-auto flex gap-3 text-[14px]">
                {prevId && (
                  <Link className="text-accent-strong underline" to={`/review/${prevId}`}>
                    ← Previous case
                  </Link>
                )}
                {nextId && (
                  <Link className="text-accent-strong underline" to={`/review/${nextId}`}>
                    Next case →
                  </Link>
                )}
              </span>
            )}
          </div>
          {detail.question && (
            <p className="text-ink-muted italic">
              <strong className="text-ink not-italic">Generated query:</strong> {detail.question}
            </p>
          )}
          <div>
            <p className="mb-2 text-ink">
              <strong>Generated AI LLM response:</strong>
            </p>
            {detail.segments ? (
              <AnswerSegments segments={detail.segments} citations={detail.citations} />
            ) : detail.answer ? (
              <>
                <p className="whitespace-pre-wrap text-ink">{detail.answer}</p>
                <CitationList citations={detail.citations} />
              </>
            ) : (
              <p className="text-ink-muted">No answer text recorded for this result.</p>
            )}
          </div>
          <RubricForm
            key={resultId}
            candidateAnswer={detail.answer}
            triageReview={detail.triage_review}
            busy={busy}
            initial={initial}
            onChange={onChange}
            onSubmit={submit}
            submitLabel="Submit & next"
            readOnly={preview}
          />
        </div>
      )}
    </section>
  );
}
