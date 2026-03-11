import json
import logging

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from sqlalchemy.orm import Session

from app.db.audit import log_action
from app.db.consent import UserConsent
from app.db.models import Agent, Event, Message, Note, Task

logger = logging.getLogger(__name__)

QDRANT_COLLECTION = "user_memory"


def delete_all_user_data(db: Session, user_id: int, qdrant_url: str) -> dict:
    """
    GDPR-like удаление всех данных пользователя:
    1. Удаляет из Qdrant все векторы с user_id
    2. Удаляет из Postgres: messages, events, tasks, notes, agents, consents
    3. Логирует действие в audit_log
    4. НЕ удаляет саму запись User (можно выбрать anonymize вместо delete)
    """
    deleted = {}

    # ── Qdrant ────────────────────────────────────────────────────────────────
    try:
        qdrant = QdrantClient(url=qdrant_url)
        qdrant.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="user_id",
                        match=MatchValue(value=user_id),
                    )
                ]
            ),
        )
        deleted["qdrant"] = "ok"
        logger.info(f"[privacy] Deleted Qdrant vectors for user_id={user_id}")
    except Exception as e:
        deleted["qdrant"] = f"error: {e}"
        logger.error(f"[privacy] Qdrant delete failed for user_id={user_id}: {e}")

    # ── Postgres: каскадное удаление ──────────────────────────────────────────
    # messages → events → agents; tasks; notes; consents
    agents = db.query(Agent).filter_by(user_id=user_id).all()
    for agent in agents:
        events = db.query(Event).filter_by(agent_id=agent.id).all()
        for event in events:
            msg_count = db.query(Message).filter_by(event_id=event.id).delete()
            deleted["messages"] = deleted.get("messages", 0) + msg_count
        ev_count = db.query(Event).filter_by(agent_id=agent.id).delete()
        deleted["events"] = deleted.get("events", 0) + ev_count
    ag_count = db.query(Agent).filter_by(user_id=user_id).delete()
    deleted["agents"] = ag_count

    deleted["tasks"] = db.query(Task).filter_by(user_id=user_id).delete()
    deleted["notes"] = db.query(Note).filter_by(user_id=user_id).delete()
    deleted["consents"] = db.query(UserConsent).filter_by(user_id=user_id).delete()

    db.commit()
    logger.info(f"[privacy] Deleted PG data for user_id={user_id}: {deleted}")

    # ── Audit log ─────────────────────────────────────────────────────────────
    log_action(db, user_id, "delete_data", json.dumps(deleted))

    return deleted
