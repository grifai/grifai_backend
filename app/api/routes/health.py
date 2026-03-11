from datetime import datetime, timezone

from fastapi import APIRouter, Request

from app.api.schemas import HealthResponse
from app.memory.contacts import JarvisMemory
from app.memory.rag import VectorMemory

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    started_at = getattr(request.app.state, "started_at", datetime.now(timezone.utc))
    uptime = int((datetime.now(timezone.utc) - started_at).total_seconds())

    tg_status = (
        "connected"
        if getattr(request.app.state, "telegram_connected", False)
        else "disconnected"
    )

    vm: VectorMemory | None = getattr(request.app.state, "vector_memory", None)
    try:
        qdrant_status = "ok" if vm is not None else "error"
        indexed = vm.index_size() if vm else 0
    except Exception:
        qdrant_status = "error"
        indexed = 0

    memory: JarvisMemory | None = getattr(request.app.state, "memory", None)
    active_contacts = len(memory.data.get("contacts", {})) if memory else 0

    return HealthResponse(
        status="ok",
        telegram=tg_status,
        qdrant=qdrant_status,
        uptime_seconds=uptime,
        active_contacts=active_contacts,
        indexed_messages=indexed,
    )
