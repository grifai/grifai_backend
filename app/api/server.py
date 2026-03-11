from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import ask, contacts, health, summary, consent
from app.config import settings


# ── WebSocket connection manager ───────────────────────────────────────────────

class ConnectionManager:
    """Broadcasts JSON events to all connected WebSocket clients."""

    def __init__(self):
        self._active: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._active.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self._active.discard(ws) if hasattr(self._active, "discard") else None
        if ws in self._active:
            self._active.remove(ws)

    async def broadcast(self, data: dict) -> None:
        dead = []
        for ws in list(self._active):
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    @property
    def connection_count(self) -> int:
        return len(self._active)


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ────────────────────────────────────────────────────────────────
    app.state.started_at = datetime.now(timezone.utc)
    app.state.ws_manager = ConnectionManager()

    # Reuse VectorMemory already initialized by main.py's rag.init(),
    # or initialize standalone if running the API without the full bot.
    from app.memory import rag
    from app.llm import get_llm, get_embeddings

    try:
        app.state.vector_memory = rag._get()
    except RuntimeError:
        # Running API standalone (e.g. during tests / direct uvicorn)
        rag.init(settings.openai_key)
        app.state.vector_memory = rag._get()

    app.state.llm = get_llm(api_key=settings.openai_key, model=settings.model)
    app.state.embeddings = get_embeddings(
        api_key=settings.openai_key, model=settings.embedding_model
    )

    # telegram_connected / memory are set by main.py before server starts.
    # Provide safe defaults so health endpoint works even without the bot.
    if not hasattr(app.state, "telegram_connected"):
        app.state.telegram_connected = False
    if not hasattr(app.state, "memory"):
        from app.memory.contacts import JarvisMemory
        app.state.memory = JarvisMemory(settings.memory_file)

    # TTS — always available (OpenAI key is required, ElevenLabs is optional)
    if not hasattr(app.state, "tts"):
        from app.services.tts import TTSService
        app.state.tts = TTSService(
            elevenlabs_key=settings.elevenlabs_key,
            openai_key=settings.openai_key,
        )

    yield
    # ── Shutdown (nothing to clean up yet) ────────────────────────────────────


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Jarvis API",
    version="0.2.0",
    description="REST + WebSocket API for the Jarvis Telegram assistant",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ask.router,      prefix="/api/v1", tags=["search"])
app.include_router(contacts.router, prefix="/api/v1", tags=["contacts"])
app.include_router(summary.router,  prefix="/api/v1", tags=["digest"])
app.include_router(health.router,   prefix="/api/v1", tags=["system"])
app.include_router(consent.router,  prefix="/api/v1", tags=["consent"])


# ── WebSocket ──────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    manager: ConnectionManager = websocket.app.state.ws_manager
    await manager.connect(websocket)
    try:
        while True:
            # Accept approve/reject commands from the client
            data = await websocket.receive_json()
            # Future: route data["type"] == "approve" | "reject" to bot
            await websocket.send_json({"type": "ack", "payload": data})
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# ── Backward-compat factory used by old main.py ────────────────────────────────

def create_app() -> FastAPI:
    """Kept for backward compatibility. Returns the module-level app."""
    return app
