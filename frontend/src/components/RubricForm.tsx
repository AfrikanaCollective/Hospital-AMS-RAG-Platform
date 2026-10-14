import { useEffect, useRef, useState } from "react";
import * as RadioGroup from "@radix-ui/react-radio-group";
import * as Collapsible from "@radix-ui/react-collapsible";
import type {
  AcceptAxisValue,
  RatingDraftState,
  RatingSubmission,
  RubricDomain,
  TriageReview,
} from "../types";
import { api } from "../api/client";
import AcceptAxisControls from "./AcceptAxisControls";
import Button from "./ui/Button";
import { defaultAcceptAxisValue, isAcceptAxisValueValid } from "../acceptAxis";

const ANCHOR_KEYS = ["anchor_1", "anchor_2", "anchor_3", "anchor_4", "anchor_5"] as const;

// Rank mode (PRD-031, PRD-040), combined with the accept axis in ONE
// submission (ARCH §13.2 "Both axes together"): for each case a ranker
// reviews, they do exactly two tasks — complete the 11-domain rubric AND
// pick an accept-axis option — together, not as two independent actions,
// and nothing more: no reason/justification text is asked for, and
// submitting never creates an escalation (DEVIATIONS #100, correcting #99's
// carried-over reason-code requirement). The parent mounts this with
// `key={resultId}` so React gives it fresh local state for each newly
// selected result, rather than resetting state via an effect.
//
// Rendered as a one-domain-per-screen wizard, not a single scrolling
// 11-row table (DEVIATIONS #133): a stacked table of native radios failed
// on three counts — sub-44px tap targets, no progress sense, and losing the
// answer/citation context while scrolling past domain after domain. Each
// step here is a domain's definition (collapsible), its 1-5 segmented
// control, and a link back to the candidate answer; a sticky bottom bar
// carries Back/Next + an "N/11 rated" count. The accept axis is its own
// final, larger-format step (AcceptAxisControls), not folded into the
// per-domain flow.
//
// `initial` restores a saved half-finished rating and `onChange` reports
// every change so the page can save it (DEVIATIONS.md #294).
//
// `triageReview` (assessment-area items only, ARCH-046 / DEVIATIONS.md #299)
// adds one more required step after the accept axis: the triage-agreement
// question. Its score rides in `scores` (so the draft keeps it) and is split
// out as `triageAgreement` on submit.
export default function RubricForm({
  candidateAnswer,
  triageReview,
  onSubmit,
  busy,
  initial,
  onChange,
  submitLabel = "Submit rating",
}: {
  candidateAnswer?: string | null;
  triageReview?: TriageReview | null;
  onSubmit: (submission: RatingSubmission) => void;
  busy?: boolean;
  initial?: Partial<RatingDraftState>;
  onChange?: (state: RatingDraftState) => void;
  submitLabel?: string;
}) {
  const [domains, setDomains] = useState<RubricDomain[]>([]);
  const [scores, setScores] = useState<Record<string, number>>(initial?.scores ?? {});
  const [comment, setComment] = useState(initial?.comment ?? "");
  const [accept, setAccept] = useState<AcceptAxisValue>(
    initial?.accept ?? defaultAcceptAxisValue(candidateAnswer),
  );
  const [step, setStep] = useState(initial?.step ?? 0);

  // Latest callback in a ref, so a new function identity each render does
  // not itself count as a change.
  const onChangeRef = useRef(onChange);
  useEffect(() => {
    onChangeRef.current = onChange;
  });
  useEffect(() => {
    onChangeRef.current?.({ scores, comment, accept, step });
  }, [scores, comment, accept, step]);
  const [defOpen, setDefOpen] = useState(false);
  const [answerOpen, setAnswerOpen] = useState(false);

  useEffect(() => {
    api
      .getRubricDomains()
      .then(setDomains)
      .catch(() => setDomains([]));
  }, []);

  const triageDomain = triageReview?.domain ?? null;
  // + the accept-axis step, + the triage step when this item asks it
  const totalSteps = domains.length + 1 + (triageDomain ? 1 : 0);
  const onAcceptStep = step === domains.length;
  const onTriageStep = triageDomain !== null && step > domains.length;
  const onLastStep = step >= totalSteps - 1;
  const domain = step < domains.length ? domains[step] : null;
  const ratedCount = domains.filter((d) => scores[d.code]).length;

  const scoresComplete = domains.length > 0 && domains.every((d) => scores[d.code]);
  const triageComplete = !triageDomain || Boolean(scores[triageDomain.code]);
  const complete = scoresComplete && triageComplete && isAcceptAxisValueValid(accept, false);

  const canAdvance = domain ? Boolean(scores[domain.code]) : true;

  const progressPct = totalSteps > 0 ? ((step + 1) / totalSteps) * 100 : 0;
  const formRef = useRef<HTMLFormElement>(null);

  return (
    <form
      ref={formRef}
      onSubmit={(e) => {
        e.preventDefault();
        if (!triageDomain) {
          onSubmit({ scores, comment, accept });
          return;
        }
        const { [triageDomain.code]: triageAgreement, ...rubricScores } = scores;
        onSubmit({ scores: rubricScores, comment, accept, triageAgreement });
      }}
      className="grid gap-4 rounded-xl border border-border bg-surface p-4 pb-20 sm:pb-4"
    >
      {/* Progress (DEVIATIONS #133) — numeral + label, never a bare color bar. */}
      <div>
        <div className="mb-2 flex items-center justify-between text-[13px] text-ink-muted">
          <span>
            {onTriageStep || (onAcceptStep && !triageDomain)
              ? "Final step"
              : onAcceptStep
                ? "Next to last step"
                : `Domain ${step + 1} of ${domains.length}`}
          </span>
          <span>
            {onTriageStep ? "Triage category" : onAcceptStep ? "Accept axis" : domain?.name}
          </span>
        </div>
        <div className="h-1 overflow-hidden rounded-full bg-surface-alt">
          <div
            className="h-full bg-accent-strong transition-[width]"
            style={{ width: `${progressPct}%` }}
          />
        </div>
      </div>

      {domain && (
        <DomainStep
          domain={domain}
          value={scores[domain.code]}
          onChange={(v) => setScores((s) => ({ ...s, [domain.code]: v }))}
          candidateAnswer={candidateAnswer}
          defOpen={defOpen}
          setDefOpen={setDefOpen}
          answerOpen={answerOpen}
          setAnswerOpen={setAnswerOpen}
        />
      )}

      {onAcceptStep && (
        <div className="grid gap-4">
          <h4 className="text-[15px] font-semibold text-ink">
            Accept axis{" "}
            <span className="font-normal text-ink-muted">(required for every case)</span>
          </h4>
          <AcceptAxisControls value={accept} onChange={setAccept} requireReason={false} />
          <label className="grid gap-1 text-[13px] text-ink-muted">
            Comment (optional adjunct — not a substitute for scores)
            <textarea
              value={comment}
              onChange={(e) => setComment(e.target.value)}
              rows={2}
              className="w-full rounded-lg border border-border bg-surface p-2 text-[15px] text-ink"
            />
          </label>
        </div>
      )}

      {onTriageStep && triageReview && triageDomain && (
        <div className="grid gap-3">
          <h4 className="text-[15px] font-semibold text-ink">
            Triage category{" "}
            <span className="font-normal text-ink-muted">(required for this case)</span>
          </h4>
          <p className="text-[15px] leading-relaxed text-ink">{triageReview.narrative}</p>
          <p className="text-[12px] text-ink-muted">
            Mortality figure: operator-supplied local statistic, not guideline content.{" "}
            <a
              className="text-accent-strong underline"
              href={triageReview.source_url}
              target="_blank"
              rel="noreferrer"
            >
              Source
            </a>
          </p>
          <DomainStep
            domain={triageDomain}
            value={scores[triageDomain.code]}
            onChange={(v) => setScores((s) => ({ ...s, [triageDomain.code]: v }))}
            defOpen={defOpen}
            setDefOpen={setDefOpen}
            answerOpen={false}
            setAnswerOpen={() => {}}
          />
        </div>
      )}

      {/* Sticky thumb-reach nav (DEVIATIONS #133). */}
      <div className="fixed inset-x-0 bottom-0 z-10 flex items-center gap-2 border-t border-border bg-surface p-3 sm:static sm:border-0 sm:bg-transparent sm:p-0">
        <Button
          type="button"
          onClick={() => setStep((s) => Math.max(0, s - 1))}
          disabled={step === 0}
          className="flex-1"
        >
          ← Back
        </Button>
        <span className="shrink-0 text-[12px] text-ink-muted">
          {ratedCount}/{domains.length} rated
        </span>
        {onLastStep ? (
          // type="button" + requestSubmit(), NOT type="submit" — this slot
          // renders in the same tree position as the Next button below, so
          // React patches the existing <button> node's attributes in place
          // rather than remounting it. A type="submit" button here would
          // mean the *previous* click (Next, on the last domain) could flip
          // this node's type from "button" to "submit" mid-dispatch — the
          // browser resolves a click's native default action against the
          // element's type *after* handlers run, so that click's own
          // default action would submit the form immediately, with
          // whatever accept-axis value was still at its default, before the
          // rater ever saw this step. Found via the preview harness
          // (DEVIATIONS #133): every "finish the last domain" Next click
          // was silently auto-submitting a `full_accept` rating.
          <Button
            type="button"
            variant="primary"
            onClick={() => formRef.current?.requestSubmit()}
            disabled={!complete || busy}
            className="flex-1"
          >
            {busy ? "Submitting…" : submitLabel}
          </Button>
        ) : (
          <Button
            type="button"
            variant="primary"
            onClick={() => setStep((s) => Math.min(totalSteps - 1, s + 1))}
            disabled={!canAdvance}
            className="flex-1"
          >
            Next →
          </Button>
        )}
      </div>
    </form>
  );
}

function DomainStep({
  domain: d,
  value,
  onChange,
  candidateAnswer,
  defOpen,
  setDefOpen,
  answerOpen,
  setAnswerOpen,
}: {
  domain: RubricDomain;
  value: number | undefined;
  onChange: (v: number) => void;
  candidateAnswer?: string | null;
  defOpen: boolean;
  setDefOpen: (v: boolean) => void;
  answerOpen: boolean;
  setAnswerOpen: (v: boolean) => void;
}) {
  return (
    <div className="grid gap-3">
      <Collapsible.Root open={defOpen} onOpenChange={setDefOpen}>
        <Collapsible.Trigger asChild>
          <button
            type="button"
            className="flex w-full items-center justify-between rounded-xl border border-border bg-surface-alt p-3.5 text-left"
          >
            <span className="text-[15px] font-medium text-ink">
              {d.ordinal}. {d.name}
              {d.required ? " *" : ""}
            </span>
            <span aria-hidden className="text-ink-muted">
              {defOpen ? "▲" : "▼"}
            </span>
          </button>
        </Collapsible.Trigger>
        <Collapsible.Content className="rounded-b-xl border border-t-0 border-border bg-surface-alt p-3.5 pt-0">
          <p className="mb-2 text-[13px] leading-relaxed text-ink-muted">{d.definition}</p>
          <dl className="grid gap-1 text-[13px] text-ink-muted">
            {ANCHOR_KEYS.map((k, i) => (
              <div key={k} className="flex gap-2">
                <dt className="shrink-0 font-semibold text-ink">{i + 1}.</dt>
                <dd>{d[k]}</dd>
              </div>
            ))}
          </dl>
        </Collapsible.Content>
      </Collapsible.Root>

      {candidateAnswer && (
        <Collapsible.Root open={answerOpen} onOpenChange={setAnswerOpen}>
          <Collapsible.Trigger asChild>
            <button
              type="button"
              className="flex w-full items-center justify-between rounded-xl border border-border bg-surface p-2.5 text-left text-[13px] text-ink-muted"
            >
              <span>📄 View answer and citations</span>
              <span aria-hidden>{answerOpen ? "▲" : "›"}</span>
            </button>
          </Collapsible.Trigger>
          <Collapsible.Content className="max-h-64 overflow-y-auto rounded-b-xl border border-t-0 border-border bg-surface p-3 text-[15px] leading-relaxed whitespace-pre-wrap text-ink">
            {candidateAnswer}
          </Collapsible.Content>
        </Collapsible.Root>
      )}

      <RadioGroup.Root
        // Always a defined string ("" = unrated) — never `undefined`. Radix's
        // RadioGroup falls back to uncontrolled mode the moment `value` is
        // `undefined` even once, and then *keeps* the previously-checked
        // item's value internally; the next domain would silently render
        // with the prior domain's rating pre-checked (with no matching
        // `scores` entry) and a click on that already-"checked" item fires
        // no `onValueChange`, since Radix sees no value change — silently
        // blocking the rater from ever recording that domain (found via the
        // preview harness in DEVIATIONS #133, not by inspection).
        value={value !== undefined ? value.toString() : ""}
        onValueChange={(v) => onChange(Number(v))}
        aria-label={`${d.name} rating, 1 to 5`}
        className="grid grid-cols-5 gap-1.5"
      >
        {[1, 2, 3, 4, 5].map((v) => {
          const selected = value === v;
          return (
            <RadioGroup.Item
              key={v}
              value={v.toString()}
              aria-label={`${v} — ${d[ANCHOR_KEYS[v - 1]]}`}
              className={`h-[52px] rounded-lg border text-[15px] font-medium transition-colors
                ${
                  selected
                    ? "border-accent-strong bg-accent-strong text-on-accent"
                    : "border-border bg-surface text-ink hover:bg-surface-alt"
                }`}
            >
              {v}
            </RadioGroup.Item>
          );
        })}
      </RadioGroup.Root>
      <div className="flex justify-between text-[12px] text-ink-muted">
        <span>1 · {d.anchor_1}</span>
        <span className="text-right">5 · {d.anchor_5}</span>
      </div>
    </div>
  );
}
