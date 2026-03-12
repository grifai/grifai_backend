import json
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from app.db.audit import log_action
from app.db.consent import UserConsent


def get_user_consent(db: Session, user_id: int, consent_type: str) -> Optional[UserConsent]:
    return (
        db.query(UserConsent)
        .filter_by(user_id=user_id, consent_type=consent_type)
        .order_by(UserConsent.id.desc())
        .first()
    )


def get_all_consents(db: Session, user_id: int) -> list[UserConsent]:
    return db.query(UserConsent).filter_by(user_id=user_id).all()


def grant_user_consent(db: Session, user_id: int, consent_type: str) -> UserConsent:
    consent = UserConsent(
        user_id=user_id,
        consent_type=consent_type,
        granted_at=datetime.utcnow(),
        revoked_at=None,
    )
    db.add(consent)
    db.commit()
    db.refresh(consent)
    log_action(db, user_id, "grant_consent", json.dumps({"consent_type": consent_type}))
    return consent


def revoke_user_consent(db: Session, user_id: int, consent_type: str) -> Optional[UserConsent]:
    consent = get_user_consent(db, user_id, consent_type)
    if consent and consent.revoked_at is None:
        consent.revoked_at = datetime.utcnow()
        db.commit()
        db.refresh(consent)
        log_action(db, user_id, "revoke_consent", json.dumps({"consent_type": consent_type}))
    return consent
