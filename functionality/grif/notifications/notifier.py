"""
Notifier — sends agent results and status updates to users via configured channels.

Channels: Telegram, Email (extensible).
All sends are non-blocking: failures are logged, not raised.

No LLM calls — pure dispatch logic.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Protocol

import structlog

log = structlog.get_logger(__name__)


class NotificationChannel(StrEnum):
    TELEGRAM = "telegram"
    EMAIL = "email"
    WEBHOOK = "webhook"


class NotificationPayload:
    """Message to send to the user."""

    def __init__(
        self,
        user_id: str,
        subject: str,
        body: str,
        channel: NotificationChannel,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.user_id = user_id
        self.subject = subject
        self.body = body
        self.channel = channel
        self.metadata = metadata or {}


class ChannelSender(Protocol):
    """Interface every channel adapter must implement."""

    async def send(self, payload: NotificationPayload) -> bool:
        """Return True on success, False on failure."""
        ...


# ── Channel adapters ──────────────────────────────────────────────────────────

class TelegramSender:
    """Sends a message to a Telegram chat via HTTP."""

    def __init__(self, bot_token: str) -> None:
        self._token = bot_token

    async def send(self, payload: NotificationPayload) -> bool:
        if not self._token:
            log.warning("telegram_sender_no_token", user_id=payload.user_id)
            return False
        chat_id = payload.metadata.get("telegram_chat_id")
        if not chat_id:
            log.warning("telegram_sender_no_chat_id", user_id=payload.user_id)
            return False
        try:
            import httpx
            url = f"https://api.telegram.org/bot{self._token}/sendMessage"
            text = f"*{payload.subject}*\n\n{payload.body}"
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"})
                success = resp.status_code == 200
                if not success:
                    log.warning("telegram_send_failed", status=resp.status_code, user=payload.user_id)
                return success
        except Exception as exc:
            log.warning("telegram_send_exception", error=str(exc), user=payload.user_id)
            return False


class EmailSender:
    """Sends an email via SMTP."""

    def __init__(self, smtp_host: str, smtp_port: int, username: str, password: str) -> None:
        self._host = smtp_host
        self._port = smtp_port
        self._username = username
        self._password = password

    async def send(self, payload: NotificationPayload) -> bool:
        to_addr = payload.metadata.get("email")
        if not to_addr:
            log.warning("email_sender_no_address", user=payload.user_id)
            return False
        try:
            import asyncio
            import smtplib
            from email.mime.text import MIMEText

            def _send_sync() -> bool:
                msg = MIMEText(payload.body, "plain", "utf-8")
                msg["Subject"] = payload.subject
                msg["From"] = self._username
                msg["To"] = to_addr
                with smtplib.SMTP(self._host, self._port, timeout=10) as server:
                    server.starttls()
                    server.login(self._username, self._password)
                    server.sendmail(self._username, [to_addr], msg.as_string())
                return True

            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, _send_sync)
        except Exception as exc:
            log.warning("email_send_exception", error=str(exc), user=payload.user_id)
            return False


class WebhookSender:
    """Posts a JSON notification to a webhook URL."""

    async def send(self, payload: NotificationPayload) -> bool:
        url = payload.metadata.get("webhook_url")
        if not url:
            log.warning("webhook_sender_no_url", user=payload.user_id)
            return False
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(url, json={
                    "user_id": payload.user_id,
                    "subject": payload.subject,
                    "body": payload.body,
                    "metadata": payload.metadata,
                })
                return resp.status_code < 400
        except Exception as exc:
            log.warning("webhook_send_exception", error=str(exc), user=payload.user_id)
            return False


# ── Notifier ──────────────────────────────────────────────────────────────────

class Notifier:
    """
    Routes notifications to the correct channel adapter.

    Usage:
        notifier = Notifier()
        notifier.register(NotificationChannel.TELEGRAM, TelegramSender(token))
        await notifier.notify(payload)
    """

    def __init__(self) -> None:
        self._senders: dict[NotificationChannel, ChannelSender] = {}

    def register(self, channel: NotificationChannel, sender: ChannelSender) -> None:
        self._senders[channel] = sender

    async def notify(self, payload: NotificationPayload) -> bool:
        """Send notification. Returns True on success, False on failure."""
        sender = self._senders.get(payload.channel)
        if sender is None:
            log.warning(
                "notifier_no_sender",
                channel=payload.channel,
                user=payload.user_id,
            )
            return False
        return await sender.send(payload)

    async def notify_all(
        self,
        user_id: str,
        subject: str,
        body: str,
        channels: list[NotificationChannel],
        metadata: dict[str, Any] | None = None,
    ) -> dict[NotificationChannel, bool]:
        """Send to multiple channels. Returns per-channel success map."""
        results: dict[NotificationChannel, bool] = {}
        for ch in channels:
            payload = NotificationPayload(
                user_id=user_id,
                subject=subject,
                body=body,
                channel=ch,
                metadata=metadata or {},
            )
            results[ch] = await self.notify(payload)
        return results

    @classmethod
    def from_settings(cls) -> "Notifier":
        """Build Notifier pre-configured from app settings."""
        from grif.config import get_settings
        s = get_settings()
        n = cls()
        if s.telegram_bot_token:
            n.register(NotificationChannel.TELEGRAM, TelegramSender(s.telegram_bot_token))
        return n
