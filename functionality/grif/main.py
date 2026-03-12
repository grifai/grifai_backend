"""
GRIF FastAPI application.
Stage 1: Health check + DB connectivity.
Full API endpoints are added in Stage 6.
"""

from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from grif.config import get_settings
from grif.database import engine
from grif.models.db import Base

log = structlog.get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    log.info("grif_startup", version=settings.app_version, env=settings.app_env)
    # In production, migrations are run via `alembic upgrade head`.
    # In development, we create tables automatically for convenience.
    if settings.app_env == "development":
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        log.info("db_tables_synced")
    yield
    await engine.dispose()
    log.info("grif_shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="GRIF",
        description="Self-creating AI agents platform",
        version=settings.app_version,
        lifespan=lifespan,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        log.error("unhandled_exception", path=str(request.url), error=str(exc))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"},
        )

    # ── Health endpoints ──────────────────────────────────────────────────────

    @app.get("/health", tags=["system"])
    async def health() -> dict[str, Any]:
        return {"status": "ok", "version": settings.app_version}

    @app.get("/health/db", tags=["system"])
    async def health_db() -> dict[str, Any]:
        try:
            async with engine.connect() as conn:
                await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
            return {"status": "ok", "db": "connected"}
        except Exception as e:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"status": "error", "db": str(e)},
            )

    # ── API routers ───────────────────────────────────────────────────────────
    from grif.api import agents, tasks, ws  # noqa: PLC0415

    app.include_router(tasks.router, prefix="/tasks", tags=["tasks"])
    app.include_router(agents.router, prefix="/agents", tags=["agents"])
    app.include_router(ws.router, prefix="/ws", tags=["websocket"])

    return app


app = create_app()
