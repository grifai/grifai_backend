from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, EmailStr

# ── Auth ──────────────────────────────────────────────────────────────────────


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    name: Optional[str] = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: int
    email: str
    name: Optional[str]
    is_active: bool

    class Config:
        from_attributes = True


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
    telegram: str  # "connected" | "disconnected"
    qdrant: str  # "ok" | "error"
    uptime_seconds: int
    active_contacts: int
    indexed_messages: int


# ── WebSocket ─────────────────────────────────────────────────────────────────


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
