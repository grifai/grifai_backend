"""FastAPI dependency injection helpers for GRIF API."""
from typing import AsyncGenerator

from fastapi import Header
from sqlalchemy.ext.asyncio import AsyncSession

from grif.database import AsyncSessionFactory
from grif.llm.gateway import LLMGateway


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield a fresh async DB session scoped to the request."""
    async with AsyncSessionFactory() as session:
        async with session.begin():
            yield session


def get_gateway() -> LLMGateway:
    """Return a configured LLMGateway instance."""
    return LLMGateway()


def get_user_id(x_user_id: str = Header(default="anonymous")) -> str:
    """Extract user_id from X-User-Id header. Defaults to 'anonymous'."""
    return x_user_id
