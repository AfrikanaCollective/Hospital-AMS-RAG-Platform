import { createContext, useContext } from "react";
import type { StoredAuth } from "./auth";
import type { RegistrationStatus } from "./types";

export interface AuthContextValue {
  auth: StoredAuth | null;
  /** Dev-login: seeded users only, no password (AUTH_PROVIDER=devjwt). */
  login: (email: string) => Promise<void>;
  loginGoogle: (idToken: string) => Promise<StoredAuth>;
  loginPassword: (email: string, password: string) => Promise<StoredAuth>;
  registerPassword: (email: string, password: string) => Promise<StoredAuth>;
  loginAdmin: (username: string, password: string) => Promise<StoredAuth>;
  /** New token with current roles, e.g. once an admin has approved the user. */
  refresh: () => Promise<StoredAuth>;
  setRegistrationStatus: (status: RegistrationStatus) => void;
  logout: () => void;
}

export const AuthContext = createContext<AuthContextValue | null>(null);

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}

/** Where a signed-in user belongs next (DEVIATIONS.md #282). */
export function homePath(auth: StoredAuth): string {
  switch (auth.registrationStatus) {
    case "needs_consent":
      return "/consent";
    case "needs_registration":
      return "/register";
    case "pending":
    case "rejected":
      return "/pending";
    default:
      if (auth.roles.includes("admin")) return "/admin";
      if (auth.roles.includes("clinician")) return "/";
      return "/review";
  }
}
