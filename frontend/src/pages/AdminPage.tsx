import { useCallback, useEffect, useState } from "react";
import { api, ApiError } from "../api/client";
import type { RegistrationOptions, ReviewerProgress, ReviewerRow } from "../types";
import Button from "../components/ui/Button";

// Admin queues (DEVIATIONS.md #282): (a) registrations waiting for approval,
// (b) approved reviewers with the cases they have completed, marked once
// they reach REVIEWER_COMPLETION_THRESHOLD.

type Tab = "pending" | "progress";

// Must match the server's RejectRequest (DEVIATIONS.md #285).
const REASON_MIN = 5;
const REASON_MAX = 500;

function errorText(e: unknown): string {
  return e instanceof ApiError ? e.message : String(e);
}

function labelFor(options: RegistrationOptions | null, key: keyof RegistrationOptions) {
  return (value: string | null) => {
    const list = options?.[key];
    if (!value || !Array.isArray(list)) return value ?? "—";
    return list.find((o) => o.value === value)?.label ?? value;
  };
}

function PendingRegistrations({ options }: { options: RegistrationOptions | null }) {
  const [rows, setRows] = useState<ReviewerRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);
  // The card whose rejection reason is being written, and the draft.
  const [rejecting, setRejecting] = useState<string | null>(null);
  const [reason, setReason] = useState("");
  const sex = labelFor(options, "sex");
  const cadre = labelFor(options, "cadre");
  const expertise = labelFor(options, "expertise_level");

  const load = useCallback(() => {
    api.listReviewers("pending").then(setRows, (e) => setError(errorText(e)));
  }, []);
  useEffect(load, [load]);

  const decide = async (userId: string, decision: "approve" | "reject") => {
    setBusyId(userId);
    setError(null);
    try {
      if (decision === "approve") await api.approveReviewer(userId);
      else await api.rejectReviewer(userId, reason.trim());
      setRejecting(null);
      setReason("");
      load();
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusyId(null);
    }
  };

  if (error && !rows) return <p className="text-[13px] text-danger">{error}</p>;
  if (!rows) return <p className="text-[13px] text-ink-muted">Loading…</p>;
  if (!rows.length)
    return <p className="text-[14px] text-ink-muted">No registrations are waiting for approval.</p>;

  return (
    <div className="grid gap-3">
      {error && <p className="text-[13px] text-danger">{error}</p>}
      {rows.map((r) => (
        <article key={r.user_id} className="rounded-lg border border-border bg-surface p-4">
          <header className="flex flex-wrap items-baseline justify-between gap-2">
            <h3 className="text-[15px] font-semibold text-ink">
              {r.email}
              {r.submission_count > 1 && (
                <span className="ml-2 rounded-md border border-border bg-surface-alt px-1.5 py-0.5 text-[12px] font-normal text-ink">
                  Resubmission #{r.submission_count}
                  {r.last_rejected_at
                    ? `, rejected ${new Date(r.last_rejected_at).toLocaleDateString()}`
                    : ""}
                </span>
              )}
            </h3>
            <span className="text-[12px] text-ink-muted">
              {r.auth_method === "google" ? "Google" : "Email + password"} · submitted{" "}
              {r.submitted_at ? new Date(r.submitted_at).toLocaleString() : "—"}
            </span>
          </header>
          <dl className="mt-2 grid grid-cols-1 gap-x-6 gap-y-1 text-[14px] sm:grid-cols-2">
            {(
              [
                ["Sex", sex(r.sex)],
                [
                  "Clinical cadre",
                  r.cadre_other ? `${cadre(r.cadre)}: ${r.cadre_other}` : cadre(r.cadre),
                ],
                ["Expertise level", expertise(r.expertise_level)],
                ["Years post-internship", r.years_post_internship ?? "—"],
                ["Organisation", r.organisation ?? "—"],
                ["Role / title", r.role_title ?? "—"],
                ["Phone (Safaricom/Airtel)", r.safaricom_phone ?? "—"],
                ["Consent version", r.consent_version],
              ] as const
            ).map(([k, v]) => (
              <div key={k} className="flex gap-2">
                <dt className="text-ink-muted">{k}:</dt>
                <dd className="text-ink">{v}</dd>
              </div>
            ))}
          </dl>
          {r.resubmissions_left === 0 && (
            <p className="mt-2 text-[13px] font-medium text-danger">
              Final submission: if rejected, this reviewer cannot resubmit.
            </p>
          )}
          {r.submission_count > 1 && r.rejection_reason && (
            <p className="mt-2 text-[13px] text-ink-muted">
              Previous rejection reason: <span className="text-ink">{r.rejection_reason}</span>
            </p>
          )}
          {rejecting === r.user_id ? (
            <form
              className="mt-3 grid gap-2"
              onSubmit={(e) => {
                e.preventDefault();
                void decide(r.user_id, "reject");
              }}
            >
              <label className="grid gap-1 text-[13px] text-ink-muted">
                Reason for rejection (shown to the reviewer)
                <textarea
                  value={reason}
                  onChange={(e) => setReason(e.target.value)}
                  minLength={REASON_MIN}
                  maxLength={REASON_MAX}
                  rows={3}
                  className="w-full rounded-lg border border-border bg-surface p-2 text-[14px] text-ink"
                  required
                  autoFocus
                />
                <span>
                  {reason.trim().length} / {REASON_MAX} characters (at least {REASON_MIN})
                </span>
              </label>
              <div className="flex gap-2">
                <Button
                  type="submit"
                  variant="danger"
                  disabled={busyId === r.user_id || reason.trim().length < REASON_MIN}
                >
                  Reject with this reason
                </Button>
                <Button
                  type="button"
                  onClick={() => {
                    setRejecting(null);
                    setReason("");
                  }}
                >
                  Cancel
                </Button>
              </div>
            </form>
          ) : (
            <div className="mt-3 flex gap-2">
              <Button
                type="button"
                variant="primary"
                disabled={busyId === r.user_id}
                onClick={() => void decide(r.user_id, "approve")}
              >
                Approve
              </Button>
              <Button
                type="button"
                variant="danger"
                disabled={busyId === r.user_id}
                onClick={() => {
                  setRejecting(r.user_id);
                  setReason("");
                }}
              >
                Reject…
              </Button>
            </div>
          )}
        </article>
      ))}
    </div>
  );
}

function SetPassword({ userId }: { userId: string }) {
  const [open, setOpen] = useState(false);
  const [value, setValue] = useState("");
  const [note, setNote] = useState<string | null>(null);
  if (!open)
    return (
      <button
        type="button"
        className="text-[12px] text-accent-strong underline-offset-2 hover:underline"
        onClick={() => setOpen(true)}
      >
        Set new password
      </button>
    );
  return (
    <form
      className="flex flex-wrap items-center gap-2"
      onSubmit={(e) => {
        e.preventDefault();
        api.setReviewerPassword(userId, value).then(
          () => {
            setNote("Password updated. Share it with the reviewer securely.");
            setValue("");
          },
          (err) => setNote(errorText(err)),
        );
      }}
    >
      <input
        type="text"
        value={value}
        minLength={10}
        onChange={(e) => setValue(e.target.value)}
        className="rounded-lg border border-border bg-surface p-1 text-[13px] text-ink"
        placeholder="at least 10 characters"
        required
      />
      <Button type="submit">Save</Button>
      {note && <span className="text-[12px] text-ink-muted">{note}</span>}
    </form>
  );
}

function ReviewerProgressList({ options }: { options: RegistrationOptions | null }) {
  const [data, setData] = useState<ReviewerProgress | null>(null);
  const [passwordUsers, setPasswordUsers] = useState<Set<string>>(new Set());
  const [error, setError] = useState<string | null>(null);
  const cadre = labelFor(options, "cadre");

  useEffect(() => {
    api.getReviewerProgress().then(setData, (e) => setError(errorText(e)));
    api
      .listReviewers("approved")
      .then((rows) =>
        setPasswordUsers(
          new Set(rows.filter((r) => r.auth_method === "password").map((r) => r.user_id)),
        ),
      )
      .catch(() => undefined);
  }, []);

  if (error) return <p className="text-[13px] text-danger">{error}</p>;
  if (!data) return <p className="text-[13px] text-ink-muted">Loading…</p>;
  if (!data.reviewers.length)
    return <p className="text-[14px] text-ink-muted">No approved reviewers yet.</p>;

  const reached = data.reviewers.filter((r) => r.threshold_met).length;
  return (
    <div className="grid gap-3">
      <p className="text-[14px] text-ink">
        Target: <strong>{data.threshold}</strong> completed cases. {reached} of{" "}
        {data.reviewers.length} reviewer{data.reviewers.length === 1 ? "" : "s"} reached it.
      </p>
      <ul className="grid gap-2">
        {data.reviewers.map((r) => {
          const pct = Math.min(100, Math.round((r.completed_cases / data.threshold) * 100));
          return (
            <li
              key={r.user_id}
              className={`rounded-lg border p-3 ${
                r.threshold_met ? "border-accent-strong bg-surface-alt" : "border-border bg-surface"
              }`}
            >
              <div className="flex flex-wrap items-baseline justify-between gap-2">
                <span className="text-[15px] font-semibold text-ink">{r.email}</span>
                <span className="text-[13px] text-ink-muted">
                  {cadre(r.cadre)}
                  {r.organisation ? ` · ${r.organisation}` : ""}
                </span>
              </div>
              <div className="mt-2 flex items-center gap-3">
                <div
                  className="h-2 flex-1 overflow-hidden rounded-full bg-surface-alt"
                  role="progressbar"
                  aria-valuemin={0}
                  aria-valuemax={data.threshold}
                  aria-valuenow={r.completed_cases}
                  aria-label={`${r.email}: ${r.completed_cases} of ${data.threshold} cases`}
                >
                  <div className="h-full bg-accent-strong" style={{ width: `${pct}%` }} />
                </div>
                <span className="w-[140px] text-right text-[14px] text-ink">
                  {r.completed_cases} / {data.threshold} cases
                </span>
              </div>
              {r.threshold_met && (
                <p className="mt-2 text-[14px] font-semibold text-accent-strong">
                  ★ Target reached: {r.completed_cases} cases completed
                </p>
              )}
              {passwordUsers.has(r.user_id) && (
                <div className="mt-2">
                  <SetPassword userId={r.user_id} />
                </div>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}

export default function AdminPage() {
  const [tab, setTab] = useState<Tab>("pending");
  const [options, setOptions] = useState<RegistrationOptions | null>(null);
  useEffect(() => {
    api.getRegistrationOptions().then(setOptions, () => undefined);
  }, []);

  return (
    <section className="mt-6 grid gap-4">
      <h2 className="text-lg font-semibold text-ink">Reviewer administration</h2>
      <div className="flex gap-2 border-b border-border" role="tablist">
        {(
          [
            ["pending", "Registrations awaiting approval"],
            ["progress", "Reviewers and completed cases"],
          ] as const
        ).map(([key, label]) => (
          <button
            key={key}
            type="button"
            role="tab"
            aria-selected={tab === key}
            onClick={() => setTab(key)}
            className={`-mb-px border-b-2 px-3 py-2 text-[14px] ${
              tab === key
                ? "border-accent-strong font-semibold text-ink"
                : "border-transparent text-ink-muted"
            }`}
          >
            {label}
          </button>
        ))}
      </div>
      {tab === "pending" ? (
        <PendingRegistrations options={options} />
      ) : (
        <ReviewerProgressList options={options} />
      )}
    </section>
  );
}
