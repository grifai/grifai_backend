"""
Email Client Tool — SMTP (send) + IMAP (read).
send_email: WRITE_PUBLIC
read_email: READ
"""

import asyncio
import email as emaillib
import imaplib
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

import structlog

from grif.models.enums import ToolCategory
from grif.tools.base import BaseTool, ToolResult

log = structlog.get_logger(__name__)


class EmailClientTool(BaseTool):
    name = "email_client"
    description = (
        "Send and read emails via SMTP/IMAP. "
        "Use send_email to send, read_email to check inbox."
    )
    category = ToolCategory.WRITE_PUBLIC

    def __init__(
        self,
        smtp_host: str = "",
        smtp_port: int = 587,
        imap_host: str = "",
        username: str = "",
        password: str = "",
        use_tls: bool = True,
    ) -> None:
        self._smtp_host = smtp_host
        self._smtp_port = smtp_port
        self._imap_host = imap_host
        self._username = username
        self._password = password
        self._use_tls = use_tls

    async def execute(
        self,
        action: str,
        to: str | None = None,
        subject: str | None = None,
        body: str | None = None,
        limit: int = 10,
        **kwargs: Any,
    ) -> ToolResult:
        try:
            if action == "send_email":
                return await self._send(to=to, subject=subject, body=body)
            elif action == "read_email":
                return await self._read(limit=limit)
            else:
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    error=f"Unknown action: {action}. Use send_email | read_email",
                )
        except Exception as exc:
            log.error("email_tool_failed", action=action, error=str(exc))
            return ToolResult(tool_name=self.name, success=False, error=str(exc))

    async def _send(
        self, to: str | None, subject: str | None, body: str | None
    ) -> ToolResult:
        if not all([to, subject, body]):
            return ToolResult(
                tool_name=self.name,
                success=False,
                error="send_email requires to, subject, and body",
            )

        loop = asyncio.get_event_loop()

        def _sync_send() -> None:
            msg = MIMEMultipart("alternative")
            msg["From"] = self._username
            msg["To"] = to
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "html"))

            with smtplib.SMTP(self._smtp_host, self._smtp_port) as server:
                if self._use_tls:
                    server.starttls()
                server.login(self._username, self._password)
                server.send_message(msg)

        await loop.run_in_executor(None, _sync_send)
        log.info("email_sent", to=to, subject=subject)
        return ToolResult(
            tool_name=self.name,
            success=True,
            output=f"Email sent to {to}: {subject}",
        )

    async def _read(self, limit: int) -> ToolResult:
        loop = asyncio.get_event_loop()

        def _sync_read() -> list[dict]:
            with imaplib.IMAP4_SSL(self._imap_host) as imap:
                imap.login(self._username, self._password)
                imap.select("INBOX")
                _, msg_ids = imap.search(None, "UNSEEN")
                ids = msg_ids[0].split()[-limit:]
                messages = []
                for uid in reversed(ids):
                    _, data = imap.fetch(uid, "(RFC822)")
                    raw = data[0][1]  # type: ignore
                    msg = emaillib.message_from_bytes(raw)
                    body_text = ""
                    if msg.is_multipart():
                        for part in msg.walk():
                            if part.get_content_type() == "text/plain":
                                body_text = part.get_payload(decode=True).decode()
                                break
                    else:
                        body_text = msg.get_payload(decode=True).decode()
                    messages.append({
                        "from": msg.get("From"),
                        "subject": msg.get("Subject"),
                        "date": msg.get("Date"),
                        "body": body_text[:500],
                    })
                return messages

        messages = await loop.run_in_executor(None, _sync_read)
        return ToolResult(
            tool_name=self.name,
            success=True,
            output=messages,
            metadata={"count": len(messages)},
        )

    def _get_parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["send_email", "read_email"],
                },
                "to": {"type": "string", "description": "Recipient email (for send_email)."},
                "subject": {"type": "string", "description": "Email subject."},
                "body": {"type": "string", "description": "Email body (HTML supported)."},
                "limit": {
                    "type": "integer",
                    "description": "Max emails to read (for read_email).",
                    "default": 10,
                },
            },
            "required": ["action"],
        }
