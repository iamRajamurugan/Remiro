from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

import jwt
from fastapi import FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from src.config import get_settings
from src.llm_service import CareerGuideLLM


def utc_now() -> datetime:
    return datetime.now(UTC)


class SessionCreateRequest(BaseModel):
    title: str = Field(default="New chat", min_length=1, max_length=200)


class SessionUpdateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class MessageCreateRequest(BaseModel):
    content: str = Field(min_length=1)


class ChatMongoRepository:
    def __init__(self, db: Database) -> None:
        self.sessions: Collection = db["chat_sessions"]
        self.messages: Collection = db["chat_messages"]
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        self.sessions.create_index([("user_id", ASCENDING), ("updated_at", DESCENDING)])
        self.messages.create_index([("session_id", ASCENDING), ("created_at", ASCENDING)])

    @staticmethod
    def _session_public(session: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": str(session["_id"]),
            "title": session.get("title", "New chat"),
            "created_at": session.get("created_at"),
            "updated_at": session.get("updated_at"),
        }

    @staticmethod
    def _message_public(message: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": str(message["_id"]),
            "role": message.get("role", "assistant"),
            "content": message.get("content", ""),
            "created_at": message.get("created_at"),
        }

    def list_sessions(self, user_id: str) -> list[dict[str, Any]]:
        docs = list(self.sessions.find({"user_id": user_id}).sort("updated_at", DESCENDING))
        return [self._session_public(doc) for doc in docs]

    def create_session(self, user_id: str, title: str = "New chat") -> dict[str, Any]:
        now = utc_now()
        doc = {
            "user_id": user_id,
            "title": title,
            "created_at": now,
            "updated_at": now,
        }
        inserted = self.sessions.insert_one(doc)
        stored = self.sessions.find_one({"_id": inserted.inserted_id})
        if not stored:
            raise RuntimeError("Failed to create session.")
        return self._session_public(stored)

    def ensure_session_owned(self, user_id: str, session_id: str) -> dict[str, Any]:
        from bson import ObjectId

        if not ObjectId.is_valid(session_id):
            raise HTTPException(status_code=404, detail="Session not found")

        doc = self.sessions.find_one({"_id": ObjectId(session_id), "user_id": user_id})
        if not doc:
            raise HTTPException(status_code=404, detail="Session not found")
        return doc

    def rename_session(self, user_id: str, session_id: str, title: str) -> None:
        from bson import ObjectId

        self.ensure_session_owned(user_id, session_id)
        self.sessions.update_one(
            {"_id": ObjectId(session_id), "user_id": user_id},
            {"$set": {"title": title, "updated_at": utc_now()}},
        )

    def delete_session(self, user_id: str, session_id: str) -> None:
        from bson import ObjectId

        self.ensure_session_owned(user_id, session_id)
        self.messages.delete_many({"session_id": session_id, "user_id": user_id})
        self.sessions.delete_one({"_id": ObjectId(session_id), "user_id": user_id})

    def list_messages(self, user_id: str, session_id: str) -> list[dict[str, Any]]:
        from bson import ObjectId

        self.ensure_session_owned(user_id, session_id)
        docs = list(self.messages.find({"session_id": session_id}).sort("created_at", ASCENDING))
        return [self._message_public(doc) for doc in docs]

    def add_message(self, user_id: str, session_id: str, role: str, content: str) -> dict[str, Any]:
        from bson import ObjectId

        self.ensure_session_owned(user_id, session_id)
        now = utc_now()
        doc = {
            "session_id": session_id,
            "user_id": user_id,
            "role": role,
            "content": content,
            "created_at": now,
        }
        inserted = self.messages.insert_one(doc)
        self.sessions.update_one(
            {"_id": ObjectId(session_id), "user_id": user_id},
            {"$set": {"updated_at": now}},
        )
        stored = self.messages.find_one({"_id": inserted.inserted_id})
        if not stored:
            raise RuntimeError("Failed to add message.")
        return self._message_public(stored)


def get_required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def parse_user_id_from_token(auth_header: str | None, x_remiro_token: str | None) -> str:
    token = ""
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
    elif x_remiro_token:
        token = x_remiro_token.strip()

    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing token")

    jwt_secret = get_required_env("JWT_SECRET")
    try:
        payload = jwt.decode(token, jwt_secret, algorithms=["HS256"])
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from exc

    user_id = str(payload.get("id") or "").strip()
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")
    return user_id


mongo_uri = get_required_env("MONGODB_URI")
mongo_db_name = os.getenv("MONGODB_DB_NAME", "remiro_ai_chat").strip() or "remiro_ai_chat"
mongo_client = MongoClient(mongo_uri)
mongo_db = mongo_client[mongo_db_name]
repo = ChatMongoRepository(mongo_db)

settings = get_settings()
llm = CareerGuideLLM(
    settings.google_api_key,
    settings.gemini_model,
    settings.serper_api_key,
)

app = FastAPI(title="Remiro AI Chat API", version="1.0.0")

allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in allowed_origins if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "service": "ai-chat-backend"}


@app.get("/api/chat/sessions")
def list_sessions(
    authorization: str | None = Header(default=None),
    x_remiro_token: str | None = Header(default=None),
) -> dict[str, Any]:
    user_id = parse_user_id_from_token(authorization, x_remiro_token)
    return {"sessions": repo.list_sessions(user_id)}


@app.post("/api/chat/sessions", status_code=201)
def create_session(
    payload: SessionCreateRequest,
    authorization: str | None = Header(default=None),
    x_remiro_token: str | None = Header(default=None),
) -> dict[str, Any]:
    user_id = parse_user_id_from_token(authorization, x_remiro_token)
    session = repo.create_session(user_id, payload.title.strip() or "New chat")
    return {"session": session}


@app.patch("/api/chat/sessions/{session_id}")
def rename_session(
    session_id: str,
    payload: SessionUpdateRequest,
    authorization: str | None = Header(default=None),
    x_remiro_token: str | None = Header(default=None),
) -> dict[str, Any]:
    user_id = parse_user_id_from_token(authorization, x_remiro_token)
    repo.rename_session(user_id, session_id, payload.title.strip())
    return {"ok": True}


@app.delete("/api/chat/sessions/{session_id}")
def delete_session(
    session_id: str,
    authorization: str | None = Header(default=None),
    x_remiro_token: str | None = Header(default=None),
) -> dict[str, Any]:
    user_id = parse_user_id_from_token(authorization, x_remiro_token)
    repo.delete_session(user_id, session_id)
    return {"ok": True}


@app.get("/api/chat/sessions/{session_id}/messages")
def list_messages(
    session_id: str,
    authorization: str | None = Header(default=None),
    x_remiro_token: str | None = Header(default=None),
) -> dict[str, Any]:
    user_id = parse_user_id_from_token(authorization, x_remiro_token)
    return {"messages": repo.list_messages(user_id, session_id)}


@app.post("/api/chat/sessions/{session_id}/messages")
def create_message(
    session_id: str,
    payload: MessageCreateRequest,
    authorization: str | None = Header(default=None),
    x_remiro_token: str | None = Header(default=None),
) -> dict[str, Any]:
    user_id = parse_user_id_from_token(authorization, x_remiro_token)
    user_message = repo.add_message(user_id, session_id, "user", payload.content.strip())
    history = repo.list_messages(user_id, session_id)
    reply = llm.generate_reply(history)
    assistant_message = repo.add_message(user_id, session_id, "assistant", reply)
    return {"user_message": user_message, "assistant_message": assistant_message}
