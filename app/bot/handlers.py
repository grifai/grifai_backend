"""Re-exports from client.py for backward compatibility."""

from app.bot.client import ask_approval, fetch_messages, format_dialog

__all__ = ["ask_approval", "format_dialog", "fetch_messages"]
