from __future__ import annotations

from typing import Any

from supabase import Client, create_client


class AuthService:
    def __init__(
        self,
        supabase_url: str,
        supabase_key: str,
        access_token: str | None = None,
        refresh_token: str | None = None,
    ) -> None:
        self.client: Client = create_client(supabase_url, supabase_key)
        if access_token and refresh_token:
            try:
                self.client.auth.set_session(access_token, refresh_token)
            except Exception:
                # If token restoration fails, caller will show login UI.
                pass

    def sign_up(self, email: str, password: str) -> Any:
        return self.client.auth.sign_up({"email": email, "password": password})

    def sign_in(self, email: str, password: str) -> Any:
        return self.client.auth.sign_in_with_password({"email": email, "password": password})

    def sign_out(self) -> None:
        self.client.auth.sign_out()

    def get_user(self) -> Any:
        response = self.client.auth.get_user()
        return getattr(response, "user", None)

    @staticmethod
    def extract_tokens(auth_response: Any) -> tuple[str | None, str | None]:
        session = getattr(auth_response, "session", None)
        if not session:
            return None, None
        return getattr(session, "access_token", None), getattr(session, "refresh_token", None)
