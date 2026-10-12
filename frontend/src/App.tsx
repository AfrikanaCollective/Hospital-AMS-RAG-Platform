import type { ReactNode } from "react";
import { Link, Navigate, Route, Routes } from "react-router-dom";
import QueryPage from "./pages/QueryPage";
import ReviewPage from "./pages/ReviewPage";
import EscalationsPage from "./pages/EscalationsPage";
import CorpusReviewPage from "./pages/CorpusReviewPage";
import LoginPage from "./pages/LoginPage";
import ConsentPage from "./pages/ConsentPage";
import RegisterPage from "./pages/RegisterPage";
import PendingPage from "./pages/PendingPage";
import AdminPage from "./pages/AdminPage";
import DisclaimerBanner from "./components/DisclaimerBanner";
import Button from "./components/ui/Button";
import { AuthProvider } from "./AuthContext";
import { homePath, useAuth } from "./useAuth";
import { hasRole } from "./auth";
import type { RegistrationStatus } from "./types";

// Signed in AND approved; otherwise sent to the step they are on (consent,
// registration, waiting for approval: DEVIATIONS.md #282). `role` also
// requires that role, else the user's own home page.
function RequireAuth({ children, role }: { children: ReactNode; role?: string }) {
  const { auth } = useAuth();
  if (!auth) return <Navigate to="/login" replace />;
  if (auth.registrationStatus !== "approved") return <Navigate to={homePath(auth)} replace />;
  if (role && !hasRole(auth, role)) return <Navigate to={homePath(auth)} replace />;
  return <>{children}</>;
}

// Registration steps: only reachable while the user is on that step.
function RequireStep({ step, children }: { step: RegistrationStatus[]; children: ReactNode }) {
  const { auth } = useAuth();
  if (!auth) return <Navigate to="/login" replace />;
  if (!step.includes(auth.registrationStatus)) return <Navigate to={homePath(auth)} replace />;
  return <>{children}</>;
}

function Shell() {
  const { auth, logout } = useAuth();
  const approved = auth?.registrationStatus === "approved";
  const canQuery = hasRole(auth, "clinician");
  const canReview = hasRole(auth, "reviewer") || hasRole(auth, "admin");
  const isAdmin = hasRole(auth, "admin");

  return (
    <div className="mx-auto max-w-[960px] p-4 font-sans text-ink">
      <header className="flex items-baseline gap-4">
        <h1 className="text-xl font-semibold">Hospital RAG Platform</h1>
        {auth && approved && (
          <nav className="flex gap-3">
            {canQuery && (
              <Link className="text-accent-strong underline-offset-2 hover:underline" to="/">
                Query
              </Link>
            )}
            {canReview && (
              <Link className="text-accent-strong underline-offset-2 hover:underline" to="/review">
                Review queue
              </Link>
            )}
            {canReview && (
              <Link
                className="text-accent-strong underline-offset-2 hover:underline"
                to="/escalations"
              >
                Escalations
              </Link>
            )}
            {isAdmin && (
              <Link
                className="text-accent-strong underline-offset-2 hover:underline"
                to="/corpus-review"
              >
                Corpus review
              </Link>
            )}
            {isAdmin && (
              <Link className="text-accent-strong underline-offset-2 hover:underline" to="/admin">
                Reviewers
              </Link>
            )}
          </nav>
        )}
        <span className="ml-auto text-[13px] text-ink-muted">
          {auth ? (
            <>
              {auth.roles.length ? auth.roles.join(", ") : "registering"}{" "}
              <Button type="button" onClick={logout} className="ml-2">
                Sign out
              </Button>
            </>
          ) : (
            <Link className="text-accent-strong underline-offset-2 hover:underline" to="/login">
              Sign in
            </Link>
          )}
        </span>
      </header>

      {/* Non-removable disclaimer, always mounted on answer-bearing views (ARCH-037). */}
      {auth && approved && <DisclaimerBanner />}

      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route
          path="/consent"
          element={
            <RequireStep step={["needs_consent"]}>
              <ConsentPage />
            </RequireStep>
          }
        />
        <Route
          path="/register"
          element={
            <RequireStep step={["needs_registration", "rejected"]}>
              <RegisterPage />
            </RequireStep>
          }
        />
        <Route
          path="/pending"
          element={
            <RequireStep step={["pending", "rejected"]}>
              <PendingPage />
            </RequireStep>
          }
        />
        <Route
          path="/admin"
          element={
            <RequireAuth role="admin">
              <AdminPage />
            </RequireAuth>
          }
        />
        <Route
          path="/"
          element={
            <RequireAuth role="clinician">
              <QueryPage />
            </RequireAuth>
          }
        />
        <Route
          path="/review"
          element={
            <RequireAuth>
              <ReviewPage />
            </RequireAuth>
          }
        />
        <Route
          path="/corpus-review"
          element={
            <RequireAuth>
              <CorpusReviewPage />
            </RequireAuth>
          }
        />
        <Route
          path="/escalations"
          element={
            <RequireAuth>
              <EscalationsPage />
            </RequireAuth>
          }
        />
      </Routes>

      <footer className="mt-8 text-xs text-ink-muted">
        This system reports and cites guideline content; it does not generate
        independent clinical recommendations.
      </footer>
    </div>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <Shell />
    </AuthProvider>
  );
}
