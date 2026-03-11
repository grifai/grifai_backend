from typing import Any, Optional, Literal
from datetime import datetime

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


# ── Consent ────────────────────────────────────────────────────────────────

class UserConsentBase(BaseModel):
    consent_type: Literal["memory", "calls", "voice"]


class UserConsentCreate(UserConsentBase):
    pass


class UserConsentRevoke(UserConsentBase):
    pass


class UserConsentStatus(UserConsentBase):
    granted_at: Optional[datetime]
    revoked_at: Optional[datetime]


class UserConsentInDB(UserConsentStatus):
    id: int
    user_id: int

    class Config:
        orm_mode = True
