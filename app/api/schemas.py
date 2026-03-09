from typing import Any

from pydantic import BaseModel


# ── Ask ───────────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    query: str


class SourceItem(BaseModel):
    text: str
    contact_name: str
    mine: bool
    date: str
    score: float = 0.0


class AskResponse(BaseModel):
    answer: str
    intent: str
    contact: str | None = None
    sources: list[SourceItem] = []


# ── Contacts ──────────────────────────────────────────────────────────────────

class ContactResponse(BaseModel):
    contact_id: str
    name: str
    ai_mode: str
    relationship: str | None = None
    updated: str
    profile: dict[str, Any] | None = None


class ContactAIModeRequest(BaseModel):
    mode: str  # "auto" | "never" | "ask"


class MessageItem(BaseModel):
    text: str
    mine: bool
    date: str
    contact_name: str


class ContactMessagesResponse(BaseModel):
    contact_id: str
    name: str
    messages: list[MessageItem]
    total: int


# ── Summary ───────────────────────────────────────────────────────────────────

class SummaryResponse(BaseModel):
    summary: str
    dialogs_count: int
    hours: int
    generated_at: str


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    telegram: str        # "connected" | "disconnected"
    qdrant: str          # "ok" | "error"
    uptime_seconds: int
    active_contacts: int
    indexed_messages: int


# ── WebSocket ─────────────────────────────────────────────────────────────────

class WsEvent(BaseModel):
    type: str  # "new_message" | "draft" | "approved" | "skipped" | "error"
    payload: dict[str, Any] = {}
