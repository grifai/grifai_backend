from sqlalchemy import Column, Integer, String, DateTime, Text
from sqlalchemy.orm import Session
from datetime import datetime

from app.db.models import Base


class AuditLog(Base):
    """Каждое privacy-действие пользователя (grant/revoke consent, delete data)."""

    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False, index=True)
    action = Column(String, nullable=False)  # "grant_consent" | "revoke_consent" | "delete_data"
    detail = Column(Text, nullable=True)     # JSON-строка с деталями (consent_type и т.д.)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


def log_action(db: Session, user_id: int, action: str, detail: str = "") -> None:
    """Записывает audit-запись в БД и коммитит."""
    entry = AuditLog(user_id=user_id, action=action, detail=detail)
    db.add(entry)
    db.commit()
