"""Tests for notifications/notifier.py — mocked HTTP."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from grif.notifications.notifier import (
    NotificationChannel,
    NotificationPayload,
    Notifier,
    TelegramSender,
    WebhookSender,
)


# ─── NotificationPayload ──────────────────────────────────────────────────────

def test_payload_defaults() -> None:
    p = NotificationPayload(
        user_id="u1",
        subject="Test",
        body="Hello",
        channel=NotificationChannel.TELEGRAM,
    )
    assert p.metadata == {}
    assert p.channel == NotificationChannel.TELEGRAM


# ─── Notifier: no sender registered ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_notifier_returns_false_when_no_sender() -> None:
    notifier = Notifier()
    payload = NotificationPayload(
        user_id="u1", subject="Test", body="Body",
        channel=NotificationChannel.TELEGRAM,
    )
    result = await notifier.notify(payload)
    assert result is False


# ─── Notifier: registered sender ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_notifier_calls_sender() -> None:
    sender = AsyncMock()
    sender.send = AsyncMock(return_value=True)

    notifier = Notifier()
    notifier.register(NotificationChannel.TELEGRAM, sender)

    payload = NotificationPayload(
        user_id="u1", subject="Test", body="Body",
        channel=NotificationChannel.TELEGRAM,
    )
    result = await notifier.notify(payload)
    assert result is True
    sender.send.assert_called_once_with(payload)


@pytest.mark.asyncio
async def test_notifier_returns_sender_result_on_failure() -> None:
    sender = AsyncMock()
    sender.send = AsyncMock(return_value=False)

    notifier = Notifier()
    notifier.register(NotificationChannel.EMAIL, sender)

    payload = NotificationPayload(
        user_id="u1", subject="Test", body="Body",
        channel=NotificationChannel.EMAIL,
    )
    result = await notifier.notify(payload)
    assert result is False


# ─── notify_all ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_notify_all_sends_to_multiple_channels() -> None:
    tg_sender = AsyncMock()
    tg_sender.send = AsyncMock(return_value=True)
    email_sender = AsyncMock()
    email_sender.send = AsyncMock(return_value=True)

    notifier = Notifier()
    notifier.register(NotificationChannel.TELEGRAM, tg_sender)
    notifier.register(NotificationChannel.EMAIL, email_sender)

    results = await notifier.notify_all(
        user_id="u1",
        subject="Done",
        body="Task completed",
        channels=[NotificationChannel.TELEGRAM, NotificationChannel.EMAIL],
    )
    assert results[NotificationChannel.TELEGRAM] is True
    assert results[NotificationChannel.EMAIL] is True


@pytest.mark.asyncio
async def test_notify_all_partial_failure() -> None:
    tg_sender = AsyncMock()
    tg_sender.send = AsyncMock(return_value=True)

    notifier = Notifier()
    notifier.register(NotificationChannel.TELEGRAM, tg_sender)
    # Email not registered

    results = await notifier.notify_all(
        user_id="u1",
        subject="Done",
        body="Result",
        channels=[NotificationChannel.TELEGRAM, NotificationChannel.EMAIL],
    )
    assert results[NotificationChannel.TELEGRAM] is True
    assert results[NotificationChannel.EMAIL] is False


# ─── TelegramSender ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_telegram_sender_returns_false_without_token() -> None:
    sender = TelegramSender(bot_token="")
    payload = NotificationPayload(
        user_id="u1", subject="Test", body="Body",
        channel=NotificationChannel.TELEGRAM,
        metadata={"telegram_chat_id": "123456"},
    )
    result = await sender.send(payload)
    assert result is False


@pytest.mark.asyncio
async def test_telegram_sender_returns_false_without_chat_id() -> None:
    sender = TelegramSender(bot_token="fake-token")
    payload = NotificationPayload(
        user_id="u1", subject="Test", body="Body",
        channel=NotificationChannel.TELEGRAM,
        metadata={},  # No chat_id
    )
    result = await sender.send(payload)
    assert result is False


@pytest.mark.asyncio
async def test_telegram_sender_sends_message() -> None:
    sender = TelegramSender(bot_token="fake-token")
    payload = NotificationPayload(
        user_id="u1", subject="Test", body="Hello World",
        channel=NotificationChannel.TELEGRAM,
        metadata={"telegram_chat_id": "123456"},
    )

    mock_response = MagicMock()
    mock_response.status_code = 200

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        result = await sender.send(payload)
        assert result is True


# ─── WebhookSender ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_webhook_sender_returns_false_without_url() -> None:
    sender = WebhookSender()
    payload = NotificationPayload(
        user_id="u1", subject="Test", body="Body",
        channel=NotificationChannel.WEBHOOK,
        metadata={},
    )
    result = await sender.send(payload)
    assert result is False


@pytest.mark.asyncio
async def test_webhook_sender_sends_post() -> None:
    sender = WebhookSender()
    payload = NotificationPayload(
        user_id="u1", subject="Test", body="Body",
        channel=NotificationChannel.WEBHOOK,
        metadata={"webhook_url": "https://example.com/hook"},
    )
    mock_response = MagicMock()
    mock_response.status_code = 200

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        result = await sender.send(payload)
        assert result is True


# ─── from_settings ────────────────────────────────────────────────────────────

def test_from_settings_creates_notifier() -> None:
    notifier = Notifier.from_settings()
    assert isinstance(notifier, Notifier)
