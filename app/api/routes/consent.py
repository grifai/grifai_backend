from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.dependencies import get_db
from app.api.schemas import UserConsentCreate, UserConsentInDB, UserConsentRevoke
from app.config import settings
from app.services.consent import (
    get_all_consents,
    grant_user_consent,
    revoke_user_consent,
)
from app.services.user_data import delete_all_user_data

router = APIRouter(tags=["consent"])


# ── GET /user/{user_id}/consent ────────────────────────────────────────────────
@router.get("/user/{user_id}/consent", response_model=List[UserConsentInDB])
def get_consents(user_id: int, db: Session = Depends(get_db)):
    """Список всех согласий пользователя."""
    return get_all_consents(db, user_id)


# ── POST /user/{user_id}/consent ───────────────────────────────────────────────
@router.post("/user/{user_id}/consent", response_model=UserConsentInDB, status_code=201)
def grant_consent(user_id: int, data: UserConsentCreate, db: Session = Depends(get_db)):
    """Выдать согласие на хранение данных определённого типа."""
    return grant_user_consent(db, user_id, data.consent_type)


# ── DELETE /user/{user_id}/consent ─────────────────────────────────────────────
@router.delete("/user/{user_id}/consent", response_model=UserConsentInDB)
def revoke_consent(user_id: int, data: UserConsentRevoke, db: Session = Depends(get_db)):
    """Отозвать согласие — данные этого типа будут помечены к удалению."""
    consent = revoke_user_consent(db, user_id, data.consent_type)
    if not consent:
        raise HTTPException(status_code=404, detail="Consent not found or already revoked")
    return consent


# ── DELETE /user/{user_id}/data ────────────────────────────────────────────────
@router.delete("/user/{user_id}/data")
def delete_user_data(user_id: int, db: Session = Depends(get_db)):
    """
    GDPR: удалить все данные пользователя из Postgres и Qdrant.
    Логирует действие в audit_log.
    """
    result = delete_all_user_data(db, user_id, qdrant_url=settings.qdrant_url)
    return {"status": "deleted", "detail": result}
