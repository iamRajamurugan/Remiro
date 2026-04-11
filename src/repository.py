from __future__ import annotations

from typing import Any

from supabase import Client


class ChatRepository:
    def __init__(self, client: Client) -> None:
        self.client = client

    def list_sessions(self) -> list[dict[str, Any]]:
        response = (
            self.client.table("chat_sessions")
            .select("id,title,created_at,updated_at")
            .order("updated_at", desc=True)
            .execute()
        )
        return response.data or []

    def create_session(self, title: str = "New chat") -> dict[str, Any]:
        payload = {"title": title}
        response = self.client.table("chat_sessions").insert(payload).execute()
        if not response.data:
            raise RuntimeError("Failed to create chat session.")
        return response.data[0]

    def rename_session(self, session_id: str, title: str) -> None:
        payload = {"title": title}
        self.client.table("chat_sessions").update(payload).eq("id", session_id).execute()

    def get_messages(self, session_id: str) -> list[dict[str, Any]]:
        response = (
            self.client.table("chat_messages")
            .select("id,role,content,created_at")
            .eq("session_id", session_id)
            .order("created_at", desc=False)
            .execute()
        )
        return response.data or []

    def add_message(self, session_id: str, role: str, content: str) -> dict[str, Any]:
        payload = {"session_id": session_id, "role": role, "content": content}
        response = self.client.table("chat_messages").insert(payload).execute()
        if not response.data:
            raise RuntimeError("Failed to save message.")
        return response.data[0]
