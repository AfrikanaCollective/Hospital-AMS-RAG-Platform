// Session token auth (ARCH-011): dev-login, or the Google / password / admin
// sign-ins of DEVIATIONS.md #282. Token lives in localStorage; `Authorization:
// Bearer` is attached to every API call by api/client.ts. `parseRoles` decodes the
// JWT payload client-side for UI purposes ONLY (which nav links to show) —
// it does not verify the signature, so it must never be treated as an
// authorization decision; the backend (`app.api.deps.require_role`) is the
// only real enforcement point.

import type { RegistrationStatus } from "./types";

const STORAGE_KEY = "hrag_token";

export interface StoredAuth {
  token: string;
  userId: string;
  roles: string[];
  // Where the user is in consent -> registration -> approval (DEVIATIONS.md
  // #282). Dev-login users are always "approved".
  registrationStatus: RegistrationStatus;
}

export function getStoredAuth(): StoredAuth | null {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw) as StoredAuth;
    // Tokens stored before registration existed belong to dev-login users.
    return { ...parsed, registrationStatus: parsed.registrationStatus ?? "approved" };
  } catch {
    return null;
  }
}

export function setStoredAuth(auth: StoredAuth): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(auth));
}

export function clearStoredAuth(): void {
  localStorage.removeItem(STORAGE_KEY);
}

export function hasRole(auth: StoredAuth | null, role: string): boolean {
  return !!auth && auth.roles.includes(role);
}
