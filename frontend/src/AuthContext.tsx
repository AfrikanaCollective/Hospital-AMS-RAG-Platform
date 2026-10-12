import { useMemo, useState, type ReactNode } from "react";
import { clearStoredAuth, getStoredAuth, setStoredAuth, type StoredAuth } from "./auth";
import { api } from "./api/client";
import type { TokenResponse } from "./types";
import { AuthContext, type AuthContextValue } from "./useAuth";

function fromToken(resp: TokenResponse): StoredAuth {
  return {
    token: resp.access_token,
    userId: resp.user_id,
    roles: resp.roles,
    registrationStatus: resp.registration_status,
  };
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [auth, setAuth] = useState<StoredAuth | null>(getStoredAuth());

  const value = useMemo<AuthContextValue>(() => {
    const keep = (next: StoredAuth): StoredAuth => {
      setStoredAuth(next);
      setAuth(next);
      return next;
    };
    return {
      auth,
      login: async (email: string) => {
        const resp = await api.devLogin(email);
        keep({
          token: resp.access_token,
          userId: resp.user_id,
          roles: resp.roles,
          registrationStatus: "approved",
        });
      },
      loginGoogle: async (idToken) => keep(fromToken(await api.googleLogin(idToken))),
      loginPassword: async (email, password) =>
        keep(fromToken(await api.passwordLogin(email, password))),
      registerPassword: async (email, password) =>
        keep(fromToken(await api.passwordRegister(email, password))),
      loginAdmin: async (username, password) =>
        keep(fromToken(await api.adminLogin(username, password))),
      refresh: async () => keep(fromToken(await api.refreshToken())),
      setRegistrationStatus: (status) => {
        if (auth) keep({ ...auth, registrationStatus: status });
      },
      logout: () => {
        clearStoredAuth();
        setAuth(null);
      },
    };
  }, [auth]);

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}
