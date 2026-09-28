import type { ReactNode } from "react";
import { Link, Navigate, Route, Routes } from "react-router-dom";
import QueryPage from "./pages/QueryPage";
import ReviewPage from "./pages/ReviewPage";
import EscalationsPage from "./pages/EscalationsPage";
import CorpusReviewPage from "./pages/CorpusReviewPage";
import LoginPage from "./pages/LoginPage";
import DisclaimerBanner from "./components/DisclaimerBanner";
import Button from "./components/ui/Button";
import { AuthProvider } from "./AuthContext";
import { useAuth } from "./useAuth";
import { hasRole } from "./auth";

function RequireAuth({ children }: { children: ReactNode }) {
  const { auth } = useAuth();
  if (!auth) return <Navigate to="/login" replace />;
  return <>{children}</>;
}

function Shell() {
  const { auth, logout } = useAuth();
  const canReview = hasRole(auth, "reviewer") || hasRole(auth, "admin");
  const isAdmin = hasRole(auth, "admin");

  return (
    <div className="mx-auto max-w-[960px] p-4 font-sans text-ink">
      <header className="flex items-baseline gap-4">
        <h1 className="text-xl font-semibold">Hospital RAG Platform</h1>
        {auth && (
          <nav className="flex gap-3">
            <Link className="text-accent-strong underline-offset-2 hover:underline" to="/">
              Query
            </Link>
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
          </nav>
        )}
        <span className="ml-auto text-[13px] text-ink-muted">
          {auth ? (
            <>
              {auth.roles.join(", ")}{" "}
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
      {auth && <DisclaimerBanner />}

      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route
          path="/"
          element={
            <RequireAuth>
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
