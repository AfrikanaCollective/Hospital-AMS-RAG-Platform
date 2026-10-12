import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, ApiError } from "../api/client";
import { useAuth } from "../useAuth";
import type { ConsentText } from "../types";
import Button from "../components/ui/Button";

// Informed consent before first registration (DEVIATIONS.md #282). Agreeing
// records the text's version and hash; declining signs the user out and
// returns them to the sign-in page.
export default function ConsentPage() {
  const { setRegistrationStatus, logout } = useAuth();
  const navigate = useNavigate();
  const [consent, setConsent] = useState<ConsentText | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api.getConsent().then(setConsent, (e) =>
      setError(e instanceof ApiError ? e.message : String(e)),
    );
  }, []);

  const decide = async (agree: boolean) => {
    if (!consent) return;
    setBusy(true);
    setError(null);
    try {
      const me = await api.decideConsent(agree, consent);
      if (agree) {
        setRegistrationStatus(me.registration_status);
        navigate("/register", { replace: true });
      } else {
        logout();
        navigate("/login", {
          replace: true,
          state: {
            notice:
              "You chose not to take part, so no account details were recorded. " +
              "You can sign in again if you change your mind.",
          },
        });
      }
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  if (error && !consent) {
    return (
      <section className="mx-auto mt-10 max-w-[640px] grid gap-3">
        <h2 className="text-base font-semibold text-ink">Registration is not open yet</h2>
        <p className="text-[13px] text-ink-muted">{error}</p>
        <Button
          type="button"
          onClick={() => {
            logout();
            navigate("/login", { replace: true });
          }}
        >
          Back to sign-in
        </Button>
      </section>
    );
  }
  if (!consent) return <p className="mt-10 text-[13px] text-ink-muted">Loading…</p>;

  return (
    <section className="mx-auto mt-6 grid max-w-[720px] gap-4">
      <header>
        <p className="text-[12px] uppercase tracking-wide text-ink-muted">
          Participant information and consent
        </p>
        <h2 className="text-lg font-semibold text-ink">{consent.title}</h2>
      </header>
      <div className="grid max-h-[60vh] gap-3 overflow-y-auto rounded-lg border border-border bg-surface p-4">
        {consent.sections.map((s) => (
          <div key={s.heading}>
            <h3 className="text-[14px] font-semibold text-ink">{s.heading}</h3>
            <p className="whitespace-pre-line text-[14px] leading-relaxed text-ink">{s.body}</p>
          </div>
        ))}
      </div>
      <p className="rounded-lg border border-border bg-surface-alt p-3 text-[14px] text-ink">
        {consent.agreement}
      </p>
      {error && (
        <p role="alert" className="text-[13px] text-danger">
          {error}
        </p>
      )}
      <div className="flex flex-wrap gap-3">
        <Button type="button" variant="primary" disabled={busy} onClick={() => void decide(true)}>
          I agree
        </Button>
        <Button type="button" disabled={busy} onClick={() => void decide(false)}>
          I do not agree
        </Button>
      </div>
      <p className="text-[12px] text-ink-muted">Consent text version {consent.version}</p>
    </section>
  );
}
