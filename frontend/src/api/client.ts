// API client (ARCH-010). Attaches `Authorization: Bearer` from localStorage
// (see ../auth.ts) to every call; a 401 clears the stored token and sends the
// user back to /login (a full reload — simplest reliable way to reset all
// React state for an MVP tool, not worth a pub-sub layer for).
import type {
  AuthConfig,
  ConsentText,
  DevLoginResponse,
  MeResponse,
  RegistrationForm,
  RegistrationOptions,
  ReviewerProgress,
  ReviewerRow,
  TokenResponse,
  EscalationDetail,
  EscalationSummary,
  HeldChunk,
  HitlDecisionResponse,
  QueryJobAccepted,
  QueryJobStatus,
  QueryResponse,
  QueueResult,
  QueueResultDetail,
  RatingRoundSubmitResponse,
  RatingSubmission,
  RubricDomain,
} from "../types";
import { clearStoredAuth, getStoredAuth } from "../auth";

const BASE = "/api";

// Every patient-scoped/answer-producing call in this app is for direct
// clinical use of the reported guideline content (ARCH-034 purpose-of-use).
// No other purpose value is exercised anywhere in this system yet.
const PURPOSE = "clinical_care";

class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const auth = getStoredAuth();
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(auth ? { Authorization: `Bearer ${auth.token}` } : {}),
    ...((init?.headers as Record<string, string>) ?? {}),
  };
  const res = await fetch(`${BASE}${path}`, { ...init, headers });
  // A 401 with no stored token is a refused sign-in (wrong password, invalid
  // Google token), not an expired session: show its message instead.
  if (res.status === 401 && auth) {
    clearStoredAuth();
    if (window.location.pathname !== "/login") window.location.href = "/login";
    throw new ApiError(401, "session expired");
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = (await res.json()) as {
        detail?: string | { loc?: (string | number)[]; msg?: string }[];
      };
      if (typeof body.detail === "string") detail = body.detail;
      // FastAPI validation errors (422): one line per field.
      else if (Array.isArray(body.detail))
        detail = body.detail
          .map((e) => `${(e.loc ?? []).slice(-1)[0] ?? "field"}: ${e.msg ?? "invalid"}`)
          .join("; ");
    } catch {
      /* body wasn't JSON; keep statusText */
    }
    throw new ApiError(res.status, detail);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

export const api = {
  // ── reviewer sign-in, consent, registration, admin (DEVIATIONS.md #282) ──
  getAuthConfig: () => req<AuthConfig>("/auth/config"),
  googleLogin: (idToken: string) =>
    req<TokenResponse>("/auth/google", {
      method: "POST",
      body: JSON.stringify({ id_token: idToken }),
    }),
  passwordLogin: (email: string, password: string) =>
    req<TokenResponse>("/auth/password/login", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    }),
  passwordRegister: (email: string, password: string) =>
    req<TokenResponse>("/auth/password/register", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    }),
  adminLogin: (username: string, password: string) =>
    req<TokenResponse>("/auth/admin/login", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    }),
  refreshToken: () => req<TokenResponse>("/auth/refresh", { method: "POST" }),
  getMe: () => req<MeResponse>("/auth/me"),
  getConsent: () => req<ConsentText>("/auth/consent"),
  decideConsent: (agree: boolean, consent: ConsentText) =>
    req<MeResponse>("/auth/consent", {
      method: "POST",
      body: JSON.stringify({ agree, version: consent.version, sha256: consent.sha256 }),
    }),
  getRegistrationOptions: () => req<RegistrationOptions>("/auth/registration/options"),
  getOwnRegistration: () => req<RegistrationForm>("/auth/registration"),
  submitRegistration: (form: RegistrationForm) =>
    req<MeResponse>("/auth/registration", { method: "POST", body: JSON.stringify(form) }),
  listReviewers: (status: "pending" | "approved" | "rejected" | "all") =>
    req<ReviewerRow[]>(`/admin/reviewers?status=${status}`),
  approveReviewer: (userId: string) =>
    req<{ user_id: string; status: string }>(`/admin/reviewers/${userId}/approve`, {
      method: "POST",
    }),
  // The reason is shown to the reviewer (DEVIATIONS.md #285).
  rejectReviewer: (userId: string, reason: string) =>
    req<{ user_id: string; status: string }>(`/admin/reviewers/${userId}/reject`, {
      method: "POST",
      body: JSON.stringify({ reason }),
    }),
  getReviewerProgress: () => req<ReviewerProgress>("/admin/reviewers/progress"),
  setReviewerPassword: (userId: string, newPassword: string) =>
    req<void>(`/admin/reviewers/${userId}/password`, {
      method: "POST",
      body: JSON.stringify({ new_password: newPassword }),
    }),

  devLogin: (email: string) =>
    req<DevLoginResponse>("/auth/dev-login", { method: "POST", body: JSON.stringify({ email }) }),

  submitQuery: (body: {
    question: string;
    conversation_id?: string;
    patient_id?: string;
    hospital_constraint?: string;
  }) =>
    req<QueryResponse>("/query", {
      method: "POST",
      body: JSON.stringify(body),
      headers: { "X-Purpose-Of-Use": PURPOSE },
    }),

  submitQueryAsync: (body: {
    question: string;
    conversation_id?: string;
    patient_id?: string;
    hospital_constraint?: string;
  }) =>
    req<QueryJobAccepted>("/query/async", {
      method: "POST",
      body: JSON.stringify(body),
      headers: { "X-Purpose-Of-Use": PURPOSE },
    }),

  getQueryJob: (jobId: string) => req<QueryJobStatus>(`/query/jobs/${jobId}`),

  getRubricDomains: () => req<RubricDomain[]>("/rubric/domains"),

  getReviewQueue: () => req<QueueResult[]>("/review-queue"),

  getReviewQueueItem: (resultId: string) =>
    req<QueueResultDetail>(`/review-queue/${resultId}`),

  // Both HITL axes, one submission (ARCH §13.2 "Both axes together"): every
  // rank-mode rating includes an accept-axis decision. Exactly two tasks —
  // no reason code is sent (that's only required by the separate
  // `submitHitlDecision` escalation-resolution call below), matching
  // `RatingRoundRequest`'s schema (DEVIATIONS #100). `accept_edited_answer`
  // is always optional, sent only if the ranker chose to write one
  // (DEVIATIONS #101) — never required for `partial_accept`.
  submitRating: (resultId: string, submission: RatingSubmission) =>
    req<RatingRoundSubmitResponse>(`/rubric/results/${resultId}/ratings`, {
      method: "POST",
      body: JSON.stringify({
        scores: Object.entries(submission.scores).map(([domain_code, score]) => ({
          domain_code,
          score,
        })),
        comment: submission.comment || undefined,
        accept_action: submission.accept.action,
        accept_edited_answer:
          submission.accept.action === "partial_accept"
            ? submission.accept.editedAnswer || undefined
            : undefined,
      }),
    }),

  listEscalations: (state?: string) =>
    req<EscalationSummary[]>(`/hitl/escalations${state ? `?state=${state}` : ""}`),

  getEscalation: (escalationId: string) =>
    req<EscalationDetail>(`/hitl/escalations/${escalationId}`),

  submitHitlDecision: (
    escalationId: string,
    body: { action: string; edited_answer?: string; reason_code?: string },
  ) =>
    req<HitlDecisionResponse>(`/hitl/escalations/${escalationId}/decision`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  // Corpus review gate (ARCH-044): admin confirms or rejects held chunks.
  getCorpusReviewQueue: () => req<HeldChunk[]>("/corpus/review-queue"),

  reviewChunk: (chunkId: string, decision: "confirmed" | "rejected", note?: string) =>
    req<HeldChunk>(`/corpus/chunks/${chunkId}/review`, {
      method: "POST",
      body: JSON.stringify({ decision, note: note || undefined }),
    }),

  // Crops need the bearer token, so they're fetched as a blob and shown via
  // an object URL rather than a plain <img src>.
  getCropObjectUrl: async (sha256: string): Promise<string> => {
    const auth = getStoredAuth();
    const res = await fetch(`${BASE}/corpus/crops/${sha256}.png`, {
      headers: auth ? { Authorization: `Bearer ${auth.token}` } : {},
    });
    if (!res.ok) throw new ApiError(res.status, res.statusText);
    return URL.createObjectURL(await res.blob());
  },
};

export { ApiError };
