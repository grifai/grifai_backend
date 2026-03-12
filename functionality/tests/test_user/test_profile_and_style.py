"""Tests for user/profile.py and user/style_cloner.py."""
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from grif.user.profile import ProfileNotFoundError, UserProfileManager
from grif.user.style_cloner import StyleCloner


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_profile(user_id: str = "u1") -> MagicMock:
    profile = MagicMock()
    profile.id = uuid.uuid4()
    profile.user_id = user_id
    profile.preferences = {}
    profile.contacts = {}
    profile.channels = {}
    profile.biorhythms = {}
    profile.style_guide = None
    profile.style_guide_updated_at = None
    return profile


def _make_session(profile: MagicMock | None = None) -> AsyncMock:
    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.delete = AsyncMock()
    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = profile
    session.execute = AsyncMock(return_value=result_mock)
    return session


# ═══════════════════════════════════════════════════════════════════════════════
# UserProfileManager
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_get_returns_existing_profile() -> None:
    profile = _make_profile("u1")
    session = _make_session(profile)
    mgr = UserProfileManager(session)
    result = await mgr.get("u1")
    assert result is profile


@pytest.mark.asyncio
async def test_get_returns_none_when_not_found() -> None:
    session = _make_session(None)
    mgr = UserProfileManager(session)
    result = await mgr.get("u1")
    assert result is None


@pytest.mark.asyncio
async def test_get_or_create_returns_existing() -> None:
    profile = _make_profile("u1")
    session = _make_session(profile)
    mgr = UserProfileManager(session)
    result = await mgr.get_or_create("u1")
    assert result is profile
    session.add.assert_not_called()


@pytest.mark.asyncio
async def test_get_or_create_creates_new_profile() -> None:
    session = _make_session(None)
    mgr = UserProfileManager(session)
    result = await mgr.get_or_create("u1")
    session.add.assert_called_once()
    session.flush.assert_called()


@pytest.mark.asyncio
async def test_update_preferences_merges() -> None:
    profile = _make_profile("u1")
    profile.preferences = {"language": "ru", "theme": "dark"}
    session = _make_session(profile)
    mgr = UserProfileManager(session)
    await mgr.update_preferences("u1", {"language": "en", "notifications": True})
    # Merged: theme preserved, language updated
    assert profile.preferences["language"] == "en"
    assert profile.preferences["theme"] == "dark"
    assert profile.preferences["notifications"] is True


@pytest.mark.asyncio
async def test_update_preferences_replace_mode() -> None:
    profile = _make_profile("u1")
    profile.preferences = {"language": "ru", "old_key": "old_val"}
    session = _make_session(profile)
    mgr = UserProfileManager(session)
    await mgr.update_preferences("u1", {"language": "en"}, merge=False)
    assert profile.preferences == {"language": "en"}
    assert "old_key" not in profile.preferences


@pytest.mark.asyncio
async def test_update_contacts() -> None:
    profile = _make_profile("u1")
    profile.contacts = {}
    session = _make_session(profile)
    mgr = UserProfileManager(session)
    await mgr.update_contacts("u1", {"telegram_chat_id": "12345", "email": "user@example.com"})
    assert profile.contacts["telegram_chat_id"] == "12345"


@pytest.mark.asyncio
async def test_update_channels_enables_channel() -> None:
    profile = _make_profile("u1")
    profile.channels = {}
    session = _make_session(profile)
    mgr = UserProfileManager(session)
    await mgr.update_channels("u1", {"telegram": {"enabled": True}})
    assert profile.channels["telegram"]["enabled"] is True


@pytest.mark.asyncio
async def test_set_style_guide() -> None:
    profile = _make_profile("u1")
    session = _make_session(profile)
    mgr = UserProfileManager(session)
    guide = "Casual, direct. Russian language. Brief responses."
    await mgr.set_style_guide("u1", guide)
    assert profile.style_guide == guide
    assert profile.style_guide_updated_at is not None


@pytest.mark.asyncio
async def test_get_style_guide_returns_guide() -> None:
    profile = _make_profile("u1")
    profile.style_guide = "My style guide"
    session = _make_session(profile)
    mgr = UserProfileManager(session)
    result = await mgr.get_style_guide("u1")
    assert result == "My style guide"


@pytest.mark.asyncio
async def test_get_style_guide_returns_none_when_not_found() -> None:
    session = _make_session(None)
    mgr = UserProfileManager(session)
    result = await mgr.get_style_guide("u1")
    assert result is None


@pytest.mark.asyncio
async def test_get_preferred_channels_enabled() -> None:
    profile = _make_profile("u1")
    profile.channels = {
        "telegram": {"enabled": True},
        "email": {"enabled": False},
    }
    session = _make_session(profile)
    mgr = UserProfileManager(session)
    channels = await mgr.get_preferred_channels("u1")
    assert "telegram" in channels
    assert "email" not in channels


@pytest.mark.asyncio
async def test_get_preferred_channels_empty_when_no_profile() -> None:
    session = _make_session(None)
    mgr = UserProfileManager(session)
    channels = await mgr.get_preferred_channels("u1")
    assert channels == []


@pytest.mark.asyncio
async def test_delete_returns_true_when_found() -> None:
    profile = _make_profile("u1")
    session = _make_session(profile)
    mgr = UserProfileManager(session)
    result = await mgr.delete("u1")
    assert result is True
    session.delete.assert_called_once_with(profile)


@pytest.mark.asyncio
async def test_delete_returns_false_when_not_found() -> None:
    session = _make_session(None)
    mgr = UserProfileManager(session)
    result = await mgr.delete("u1")
    assert result is False


# ═══════════════════════════════════════════════════════════════════════════════
# StyleCloner
# ═══════════════════════════════════════════════════════════════════════════════

def _make_style_cloner(guide_text: str = "Casual style, Russian.") -> tuple[StyleCloner, AsyncMock, AsyncMock]:
    gateway = AsyncMock()
    llm_response = MagicMock()
    llm_response.content = guide_text
    gateway.complete = AsyncMock(return_value=llm_response)

    profile = _make_profile("u1")
    session = _make_session(profile)
    profile_mgr = UserProfileManager(session)

    cloner = StyleCloner(gateway=gateway, profile_manager=profile_mgr)
    return cloner, gateway, session


@pytest.mark.asyncio
async def test_style_cloner_calls_llm() -> None:
    cloner, gateway, _ = _make_style_cloner()
    await cloner.clone("u1", samples=["Hello! How are you?", "Thanks a lot!"])
    gateway.complete.assert_called_once()


@pytest.mark.asyncio
async def test_style_cloner_returns_guide() -> None:
    cloner, _, _ = _make_style_cloner("Casual and direct. English.")
    guide = await cloner.clone("u1", samples=["Hello!", "Great work!"])
    assert "Casual" in guide


@pytest.mark.asyncio
async def test_style_cloner_no_samples_uses_default() -> None:
    cloner, gateway, _ = _make_style_cloner()
    guide = await cloner.clone("u1", samples=[])
    # No LLM call when no samples
    gateway.complete.assert_not_called()
    assert len(guide) > 0


@pytest.mark.asyncio
async def test_style_cloner_fallback_on_llm_error() -> None:
    cloner, gateway, _ = _make_style_cloner()
    gateway.complete = AsyncMock(side_effect=RuntimeError("LLM down"))
    guide = await cloner.clone("u1", samples=["Some text"])
    # Falls back to default guide
    assert len(guide) > 10


@pytest.mark.asyncio
async def test_style_cloner_limits_samples() -> None:
    cloner, gateway, _ = _make_style_cloner()
    samples = [f"Sample {i}" for i in range(20)]
    await cloner.clone("u1", samples=samples, max_samples=5)
    call_args = gateway.complete.call_args
    # The user message should contain at most 5 samples
    user_msg = call_args[1]["messages"][1]["content"]
    # Count separator occurrences (max 4 separators for 5 samples)
    assert user_msg.count("---") <= 4


@pytest.mark.asyncio
async def test_style_cloner_saves_guide_to_profile() -> None:
    profile = _make_profile("u1")
    session = _make_session(profile)
    gateway = AsyncMock()
    llm_response = MagicMock()
    llm_response.content = "Direct and professional style."
    gateway.complete = AsyncMock(return_value=llm_response)

    profile_mgr = UserProfileManager(session)
    cloner = StyleCloner(gateway=gateway, profile_manager=profile_mgr)
    await cloner.clone("u1", samples=["Hello there!"])

    # Profile style_guide was set
    assert profile.style_guide == "Direct and professional style."
