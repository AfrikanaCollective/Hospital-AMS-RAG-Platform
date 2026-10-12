import { useEffect, useState } from "react";
import ReviewQueue from "../components/ReviewQueue";
import { queryAreaLabel } from "../queryArea";
import RubricForm from "../components/RubricForm";
import AnswerSegments from "../components/AnswerSegments";
import CitationList from "../components/CitationList";
import type { QueueResult, QueueResultDetail } from "../types";
import { api, ApiError } from "../api/client";

// Reviewer rank-mode workflow (ARCH §13): for each case a ranker reviews (an
// open `eval.result`), they complete the 11-domain rubric AND pick an
// accept-axis option, TOGETHER, in one submission (ARCH §13.2 "Both axes
// together", DEVIATIONS.md #99) — `RubricForm` embeds `AcceptAxisControls`
// for exactly this reason. Standalone escalation resolution (a different
// workflow, on a different backend entity, `hitl.escalation`) lives on its
// own page (`EscalationsPage.tsx`) — moved out of here so it no longer
// appears stacked under a rank-mode task with a confusingly identical
// "Accept axis" heading (DEVIATIONS.md #120).
export default function ReviewPage() {
  return (
    <section>
      <h2 className="text-base font-semibold text-ink">Review queue</h2>

      <RankModeSection />
    </section>
  );
}

function RankModeSection() {
  const [items, setItems] = useState<QueueResult[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<QueueResultDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);

  const refreshQueue = () => {
    api
      .getReviewQueue()
      .then(setItems)
      .catch((e) => setError(e instanceof ApiError ? e.message : String(e)));
  };

  useEffect(refreshQueue, []);

  useEffect(() => {
    if (!selectedId) return;
    api
      .getReviewQueueItem(selectedId)
      .then((d) => {
        setDetail(d);
        setError(null);
      })
      .catch((e) => setError(e instanceof ApiError ? e.message : String(e)));
  }, [selectedId]);

  const select = (resultId: string | null) => {
    setSelectedId(resultId);
    if (!resultId) setDetail(null);
  };

  return (
    <div>
      <h3 className="text-[15px] font-semibold text-ink">Rank mode</h3>
      <p className="text-[13px] text-ink-muted">
        For each case: the 11-domain rubric AND an accept-axis decision, submitted
        together. Results stay in this queue until 3 distinct reviewers have rated
        them, then are archived with a per-domain IRR score.
      </p>
      <ReviewQueue items={items} selectedId={selectedId} onSelect={select} />

      {error && <p className="text-[13px] text-danger">{error}</p>}
      {notice && <p className="text-[13px] text-accent-strong">{notice}</p>}

      {detail && (
        <div className="mt-4 space-y-4 rounded-md border border-border p-3">
          <h4 className="text-sm font-semibold text-ink">
            Result {detail.result_id.slice(0, 8)} · {queryAreaLabel(detail.query_area)}
          </h4>
          {detail.question && (
            <p className="italic text-ink-muted">
              <strong className="not-italic text-ink">Generated query:</strong> {detail.question}
            </p>
          )}
          <div>
            {/* Same tag shape/classes as "Generated query:" above (plain
                <strong>, no size override) so the two labels render at
                identical font size/face — operator request. */}
            <p className="mb-2 text-ink">
              <strong>Generated AI LLM response:</strong>
            </p>
            {detail.segments ? (
              <AnswerSegments segments={detail.segments} citations={detail.citations} />
            ) : detail.answer ? (
              // A result written before segments were persisted (DEVIATIONS.md
              // #120) — no per-segment citation_ids to link inline, only the
              // flat answer text plus the same citation list, unlinked.
              <>
                <p className="whitespace-pre-wrap text-ink">{detail.answer}</p>
                <CitationList citations={detail.citations} />
              </>
            ) : (
              <p className="text-ink-muted">No answer text recorded for this result.</p>
            )}
          </div>

          <RubricForm
            key={detail.result_id}
            candidateAnswer={detail.answer}
            busy={busy}
            onSubmit={async (submission) => {
              setBusy(true);
              setError(null);
              setNotice(null);
              try {
                await api.submitRating(detail.result_id, submission);
                setNotice(`Rating submitted (accept axis: ${submission.accept.action}).`);
                select(null);
                refreshQueue();
              } catch (e) {
                setError(e instanceof ApiError ? e.message : String(e));
              } finally {
                setBusy(false);
              }
            }}
          />
        </div>
      )}
    </div>
  );
}
