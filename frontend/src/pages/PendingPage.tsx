import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, ApiError } from "../api/client";
import { homePath, useAuth } from "../useAuth";
import Button from "../components/ui/Button";

// Shown after registration until an administrator decides (DEVIATIONS.md
// #282). "Check again" fetches a fresh token, so an approved reviewer gets
// the reviewer role without signing out. A rejected reviewer can update and
// resubmit (#284).
export default function PendingPage() {
  const { auth, refresh, logout } = useAuth();
  const navigate = useNavigate();
  const [message, setMessage] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const rejected = auth?.registrationStatus === "rejected";
  const [reason, setReason] = useState<string | null>(null);
  const [left, setLeft] = useState<number | null>(null);
  const canResubmit = rejected && left !== 0;

  // The admin's reason (DEVIATIONS.md #285) and resubmissions left (#286).
  useEffect(() => {
    if (!rejected) return;
    api.getMe().then(
      (me) => {
        setReason(me.rejection_reason);
        setLeft(me.resubmissions_left);
      },
      () => undefined,
    );
  }, [rejected]);

  const check = async () => {
    setBusy(true);
    setMessage(null);
    try {
      const next = await refresh();
      if (next.registrationStatus === "approved") navigate(homePath(next), { replace: true });
      else if (next.registrationStatus === "pending")
        setMessage("Your registration is still waiting for approval.");
    } catch (e) {
      setMessage(e instanceof ApiError ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="mx-auto mt-10 grid max-w-[480px] gap-3">
      <h2 className="text-lg font-semibold text-ink">
        {rejected ? "Registration not approved" : "Registration received"}
      </h2>
      <p className="text-[14px] text-ink">
        {rejected
          ? canResubmit
            ? "An administrator did not approve this registration. You can correct your details and submit them again, or contact the study team."
            : "An administrator did not approve this registration, and no further resubmissions are possible. Please contact the study team."
          : "Thank you. An administrator will review your details. You can start reviewing cases once your registration is approved."}
      </p>
      {rejected && reason && (
        <div className="rounded-lg border border-border bg-surface-alt p-3">
          <p className="text-[12px] font-semibold uppercase tracking-wide text-ink-muted">
            Reason given by the administrator
          </p>
          <p className="mt-1 whitespace-pre-line text-[14px] text-ink">{reason}</p>
        </div>
      )}
      {canResubmit && left !== null && (
        <p className="text-[13px] text-ink-muted">
          You can resubmit {left} more time{left === 1 ? "" : "s"}.
        </p>
      )}
      {message && <p className="text-[13px] text-ink-muted">{message}</p>}
      <div className="flex gap-3">
        {rejected ? (
          canResubmit && (
            <Button type="button" variant="primary" onClick={() => navigate("/register")}>
              Update and resubmit
            </Button>
          )
        ) : (
          <Button type="button" variant="primary" disabled={busy} onClick={() => void check()}>
            Check again
          </Button>
        )}
        <Button
          type="button"
          onClick={() => {
            logout();
            navigate("/login", { replace: true });
          }}
        >
          Sign out
        </Button>
      </div>
    </section>
  );
}
