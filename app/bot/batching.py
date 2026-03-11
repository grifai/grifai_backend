"""Message batching logic — lives in JarvisBot._buffer_message / _process_batch in client.py."""

# Future extraction point: pull batching state machine out of JarvisBot
# into a standalone BufferManager class when the bot grows more complex.
from app.bot.client import JarvisBot

__all__ = ["JarvisBot"]
