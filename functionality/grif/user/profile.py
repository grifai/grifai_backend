"""
UserProfileManager — CRUD for UserProfileDB.

Stores user preferences, contacts, channels, biorhythms, and Style Guide.
No LLM calls — pure data management.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from grif.models.db import UserProfileDB

log = structlog.get_logger(__name__)


class ProfileNotFoundError(Exception):
    pass


class UserProfileManager:
    """
    Manages user profiles: get, create, update, and merge preferences.

    Usage:
        mgr = UserProfileManager(session)
        profile = await mgr.get_or_create(user_id="u1")
        await mgr.update_preferences(user_id="u1", preferences={"language": "ru"})
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, user_id: str) -> UserProfileDB | None:
        result = await self._session.execute(
            select(UserProfileDB).where(UserProfileDB.user_id == user_id)
        )
        return result.scalar_one_or_none()

    async def get_or_create(self, user_id: str) -> UserProfileDB:
        profile = await self.get(user_id)
        if profile is None:
            profile = UserProfileDB(
                user_id=user_id,
                preferences={},
                contacts={},
                channels={},
                biorhythms={},
            )
            self._session.add(profile)
            await self._session.flush()
            log.info("user_profile_created", user_id=user_id)
        return profile

    async def update_preferences(
        self,
        user_id: str,
        preferences: dict[str, Any],
        merge: bool = True,
    ) -> UserProfileDB:
        """Update user preferences. If merge=True, deep-merges into existing."""
        profile = await self.get_or_create(user_id)
        if merge:
            merged = dict(profile.preferences or {})
            merged.update(preferences)
            profile.preferences = merged
        else:
            profile.preferences = preferences
        await self._session.flush()
        log.debug("user_preferences_updated", user_id=user_id)
        return profile

    async def update_contacts(
        self,
        user_id: str,
        contacts: dict[str, Any],
    ) -> UserProfileDB:
        """Update contact info (email, telegram_chat_id, phone, etc.)."""
        profile = await self.get_or_create(user_id)
        merged = dict(profile.contacts or {})
        merged.update(contacts)
        profile.contacts = merged
        await self._session.flush()
        return profile

    async def update_channels(
        self,
        user_id: str,
        channels: dict[str, Any],
    ) -> UserProfileDB:
        """Update notification channel config (which channels are active, tokens, etc.)."""
        profile = await self.get_or_create(user_id)
        merged = dict(profile.channels or {})
        merged.update(channels)
        profile.channels = merged
        await self._session.flush()
        return profile

    async def update_biorhythms(
        self,
        user_id: str,
        biorhythms: dict[str, Any],
    ) -> UserProfileDB:
        """
        Update user biorhythm data:
        wake_hour, sleep_hour, active_days, preferred_notification_time, etc.
        """
        profile = await self.get_or_create(user_id)
        merged = dict(profile.biorhythms or {})
        merged.update(biorhythms)
        profile.biorhythms = merged
        await self._session.flush()
        return profile

    async def set_style_guide(
        self,
        user_id: str,
        style_guide: str,
    ) -> UserProfileDB:
        """Store the generated Style Guide text (200-300 tokens)."""
        profile = await self.get_or_create(user_id)
        profile.style_guide = style_guide
        profile.style_guide_updated_at = datetime.now(tz=timezone.utc)
        await self._session.flush()
        log.info("style_guide_updated", user_id=user_id, length=len(style_guide))
        return profile

    async def get_style_guide(self, user_id: str) -> str | None:
        profile = await self.get(user_id)
        return profile.style_guide if profile else None

    async def get_preferred_channels(self, user_id: str) -> list[str]:
        """Return list of active notification channel names for the user."""
        profile = await self.get(user_id)
        if not profile or not profile.channels:
            return []
        return [
            ch for ch, cfg in profile.channels.items()
            if isinstance(cfg, dict) and cfg.get("enabled", False)
        ]

    async def delete(self, user_id: str) -> bool:
        """Delete user profile. Returns True if found and deleted."""
        profile = await self.get(user_id)
        if profile is None:
            return False
        await self._session.delete(profile)
        await self._session.flush()
        log.info("user_profile_deleted", user_id=user_id)
        return True
