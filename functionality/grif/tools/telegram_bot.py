"""
Telegram Bot Tool — wraps Bot API.
post_message: WRITE_PUBLIC (requires approval unless trust-escalated)
read_messages: READ
get_analytics: READ
"""

from typing import Any

import httpx
import structlog

from grif.models.enums import ToolCategory
from grif.tools.base import BaseTool, ToolResult

log = structlog.get_logger(__name__)

_API_BASE = "https://api.telegram.org/bot{token}"


class TelegramBotTool(BaseTool):
    """
    Unified Telegram tool that supports:
    - post_message (WRITE_PUBLIC)
    - read_messages (READ)
    - get_analytics (READ — views/reactions from channel)
    """

    name = "telegram_bot"
    description = (
        "Interact with Telegram Bot API: post messages to channels/chats, "
        "read incoming messages, get channel analytics (views, reactions)."
    )
    category = ToolCategory.WRITE_PUBLIC  # Most restrictive; registry checks per-action

    def __init__(self, bot_token: str) -> None:
        self._token = bot_token
        self._base = _API_BASE.format(token=bot_token)

    async def execute(
        self,
        action: str,
        chat_id: str | int | None = None,
        text: str | None = None,
        parse_mode: str = "HTML",
        **kwargs: Any,
    ) -> ToolResult:
        """
        action: "post_message" | "read_messages" | "get_analytics"
        """
        try:
            if action == "post_message":
                return await self._post_message(chat_id=chat_id, text=text, parse_mode=parse_mode)
            elif action == "read_messages":
                return await self._read_messages(chat_id=chat_id, **kwargs)
            elif action == "get_analytics":
                return await self._get_analytics(chat_id=chat_id, **kwargs)
            else:
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    error=f"Unknown action: {action}. Use post_message | read_messages | get_analytics",
                )
        except Exception as exc:
            log.error("telegram_tool_failed", action=action, error=str(exc))
            return ToolResult(tool_name=self.name, success=False, error=str(exc))

    async def _post_message(
        self,
        chat_id: str | int | None,
        text: str | None,
        parse_mode: str,
    ) -> ToolResult:
        if not chat_id or not text:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error="post_message requires chat_id and text",
            )
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self._base}/sendMessage",
                json={"chat_id": chat_id, "text": text, "parse_mode": parse_mode},
                timeout=15,
            )
            data = resp.json()
            if data.get("ok"):
                msg_id = data["result"]["message_id"]
                log.info("telegram_message_sent", chat_id=chat_id, msg_id=msg_id)
                return ToolResult(
                    tool_name=self.name,
                    success=True,
                    output=f"Message sent. message_id={msg_id}",
                    metadata={"message_id": msg_id, "chat_id": chat_id},
                )
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=data.get("description", "Unknown Telegram error"),
            )

    async def _read_messages(
        self,
        chat_id: str | int | None,
        limit: int = 10,
        **kwargs: Any,
    ) -> ToolResult:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self._base}/getUpdates",
                params={"limit": limit},
                timeout=15,
            )
            data = resp.json()
            if data.get("ok"):
                updates = data.get("result", [])
                messages = [
                    {
                        "from": u.get("message", {}).get("from", {}).get("username"),
                        "text": u.get("message", {}).get("text", ""),
                        "date": u.get("message", {}).get("date"),
                    }
                    for u in updates
                    if u.get("message")
                ]
                return ToolResult(
                    tool_name=self.name,
                    success=True,
                    output=messages,
                    metadata={"count": len(messages)},
                )
            return ToolResult(
                tool_name=self.name, success=False, error=data.get("description", "Failed")
            )

    async def _get_analytics(
        self,
        chat_id: str | int | None,
        **kwargs: Any,
    ) -> ToolResult:
        """
        Returns channel member count as a proxy for analytics.
        Real views/reactions require Telegram Channel API (Bot must be admin).
        """
        if not chat_id:
            return ToolResult(tool_name=self.name, success=False, error="chat_id required")
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self._base}/getChatMemberCount",
                params={"chat_id": chat_id},
                timeout=15,
            )
            data = resp.json()
            if data.get("ok"):
                count = data["result"]
                return ToolResult(
                    tool_name=self.name,
                    success=True,
                    output=f"Member count: {count}",
                    metadata={"member_count": count},
                )
            return ToolResult(
                tool_name=self.name, success=False, error=data.get("description", "Failed")
            )

    def _get_parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["post_message", "read_messages", "get_analytics"],
                    "description": "The Telegram action to perform.",
                },
                "chat_id": {
                    "type": "string",
                    "description": "Target chat/channel ID or @username.",
                },
                "text": {
                    "type": "string",
                    "description": "Message text (for post_message).",
                },
                "parse_mode": {
                    "type": "string",
                    "enum": ["HTML", "Markdown", "MarkdownV2"],
                    "default": "HTML",
                },
            },
            "required": ["action"],
        }
