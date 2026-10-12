import { useEffect, useRef, useState, type FormEvent } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { homePath, useAuth } from "../useAuth";
import { api, ApiError } from "../api/client";
import type { StoredAuth } from "../auth";
import type { AuthConfig } from "../types";
import Button from "../components/ui/Button";

// Sign-in (DEVIATIONS.md #282): Google, email + password (for reviewers
// without Google), and the single .env admin. Dev-login (seeded users, no
// password) appears only when the backend runs AUTH_PROVIDER=devjwt.

interface GoogleId {
  initialize: (opts: { client_id: string; callback: (r: { credential: string }) => void }) => void;
  renderButton: (el: HTMLElement, opts: Record<string, unknown>) => void;
}
declare global {
  interface Window {
    google?: { accounts: { id: GoogleId } };
  }
}

const GSI_SRC = "https://accounts.google.com/gsi/client";

function loadGoogleScript(): Promise<void> {
  if (window.google?.accounts?.id) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const existing = document.querySelector<HTMLScriptElement>(`script[src="${GSI_SRC}"]`);
    const script = existing ?? document.createElement("script");
    script.addEventListener("load", () => resolve());
    script.addEventListener("error", () => reject(new Error("could not load Google sign-in")));
    if (!existing) {
      script.src = GSI_SRC;
      script.async = true;
      document.head.appendChild(script);
    }
  });
}

const inputClass = "w-full rounded-lg border border-border bg-surface p-2 text-[15px] text-ink";
const labelClass = "grid gap-1 text-[13px] text-ink-muted";

function errorText(e: unknown): string {
  return e instanceof ApiError ? e.message : String(e);
}

export default function LoginPage() {
  const { login, loginGoogle, loginPassword, registerPassword, loginAdmin } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const notice = (location.state as { notice?: string } | null)?.notice ?? null;

  const [config, setConfig] = useState<AuthConfig | null>(null);
  const [mode, setMode] = useState<"signin" | "create">("signin");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [showAdmin, setShowAdmin] = useState(false);
  const [adminUser, setAdminUser] = useState("");
  const [adminPassword, setAdminPassword] = useState("");
  const [devEmail, setDevEmail] = useState("reviewer1@example.dev");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const googleButton = useRef<HTMLDivElement>(null);

  const done = (next: StoredAuth) => navigate(homePath(next), { replace: true });

  const run = async (action: () => Promise<StoredAuth>) => {
    setError(null);
    setBusy(true);
    try {
      done(await action());
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => {
    api.getAuthConfig().then(setConfig, (e) => setError(errorText(e)));
  }, []);

  useEffect(() => {
    const clientId = config?.google_client_id;
    if (!clientId || !googleButton.current) return;
    let cancelled = false;
    loadGoogleScript()
      .then(() => {
        if (cancelled || !googleButton.current || !window.google) return;
        window.google.accounts.id.initialize({
          client_id: clientId,
          callback: (r) => void run(() => loginGoogle(r.credential)),
        });
        window.google.accounts.id.renderButton(googleButton.current, {
          theme: "outline",
          size: "large",
          text: "continue_with",
          width: 320,
        });
      })
      .catch((e) => setError(errorText(e)));
    return () => {
      cancelled = true;
    };
    // `run`/`loginGoogle` are stable enough for a one-time button setup.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [config?.google_client_id]);

  const submitPassword = (e: FormEvent) => {
    e.preventDefault();
    if (mode === "create" && password !== confirm) {
      setError("the two passwords do not match");
      return;
    }
    void run(() =>
      mode === "create" ? registerPassword(email, password) : loginPassword(email, password),
    );
  };

  return (
    <section className="mx-auto mt-10 grid max-w-[380px] gap-5">
      <div>
        <h2 className="text-base font-semibold text-ink">Sign in</h2>
        <p className="text-[13px] text-ink-muted">
          Clinician reviewers: continue with Google, or use an email address and password if you
          do not have a Google account. First-time reviewers are asked for consent and a few
          details, then an administrator approves the account.
        </p>
      </div>

      {notice && (
        <p className="rounded-lg border border-border bg-surface-alt p-3 text-[13px] text-ink">
          {notice}
        </p>
      )}
      {error && (
        <p role="alert" className="text-[13px] text-danger">
          {error}
        </p>
      )}

      {config?.google_client_id ? (
        <div ref={googleButton} className="min-h-[44px]" />
      ) : (
        config && (
          <p className="text-[13px] text-ink-muted">Google sign-in is not configured.</p>
        )
      )}

      <div className="flex items-center gap-3 text-[12px] text-ink-muted">
        <span className="h-px flex-1 bg-border" />
        or with email and password
        <span className="h-px flex-1 bg-border" />
      </div>

      <form onSubmit={submitPassword} className="grid gap-2">
        <div className="flex gap-2 text-[13px]" role="tablist">
          {(["signin", "create"] as const).map((m) => (
            <button
              key={m}
              type="button"
              role="tab"
              aria-selected={mode === m}
              onClick={() => {
                setMode(m);
                setError(null);
              }}
              className={`rounded-md px-2 py-1 ${
                mode === m ? "bg-surface-alt font-semibold text-ink" : "text-ink-muted"
              }`}
            >
              {m === "signin" ? "Sign in" : "Create account"}
            </button>
          ))}
        </div>
        <label className={labelClass}>
          Email
          <input
            type="email"
            autoComplete="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className={inputClass}
            required
          />
        </label>
        <label className={labelClass}>
          Password{mode === "create" && " (at least 10 characters)"}
          <input
            type="password"
            autoComplete={mode === "create" ? "new-password" : "current-password"}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className={inputClass}
            minLength={mode === "create" ? 10 : 1}
            required
          />
        </label>
        {mode === "create" && (
          <label className={labelClass}>
            Confirm password
            <input
              type="password"
              autoComplete="new-password"
              value={confirm}
              onChange={(e) => setConfirm(e.target.value)}
              className={inputClass}
              required
            />
          </label>
        )}
        <Button type="submit" variant="primary" disabled={busy}>
          {busy ? "Please wait…" : mode === "create" ? "Create account" : "Sign in"}
        </Button>
        {mode === "signin" && (
          <p className="text-[12px] text-ink-muted">
            Forgot your password? Ask the study administrator to set a new one.
          </p>
        )}
      </form>

      {config?.admin_sign_in && (
        <div className="border-t border-border pt-3">
          <button
            type="button"
            className="text-[13px] text-accent-strong underline-offset-2 hover:underline"
            onClick={() => setShowAdmin((v) => !v)}
            aria-expanded={showAdmin}
          >
            Administrator sign-in
          </button>
          {showAdmin && (
            <form
              className="mt-2 grid gap-2"
              onSubmit={(e) => {
                e.preventDefault();
                void run(() => loginAdmin(adminUser, adminPassword));
              }}
            >
              <label className={labelClass}>
                Username
                <input
                  autoComplete="username"
                  value={adminUser}
                  onChange={(e) => setAdminUser(e.target.value)}
                  className={inputClass}
                  required
                />
              </label>
              <label className={labelClass}>
                Password
                <input
                  type="password"
                  autoComplete="current-password"
                  value={adminPassword}
                  onChange={(e) => setAdminPassword(e.target.value)}
                  className={inputClass}
                  required
                />
              </label>
              <Button type="submit" disabled={busy}>
                Sign in as administrator
              </Button>
            </form>
          )}
        </div>
      )}

      {config?.dev_login && (
        <details className="border-t border-border pt-3 text-[13px] text-ink-muted">
          <summary className="cursor-pointer">Developer sign-in (seeded users)</summary>
          <form
            className="mt-2 grid gap-2"
            onSubmit={(e) => {
              e.preventDefault();
              setError(null);
              setBusy(true);
              login(devEmail)
                .then(() => navigate("/"))
                .catch((err) => setError(errorText(err)))
                .finally(() => setBusy(false));
            }}
          >
            <input
              type="email"
              value={devEmail}
              onChange={(e) => setDevEmail(e.target.value)}
              className={inputClass}
              required
            />
            <Button type="submit" disabled={busy}>
              Dev sign-in
            </Button>
          </form>
        </details>
      )}
    </section>
  );
}
