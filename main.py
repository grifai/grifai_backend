import asyncio
from telethon import TelegramClient

import config
import ai
import rag
from memory import JarvisMemory
from telegram_bot import JarvisBot


async def run():
    ai.init_openai(config.OPENAI_KEY)
    rag.init(config.OPENAI_KEY)
    memory = JarvisMemory(config.MEMORY_FILE)

    client = TelegramClient(config.SESSION_FILE, config.API_ID, config.API_HASH)
    await client.start()
    me = await client.get_me()

    rag_size = rag.index_size()
    rag_status = f"{rag_size} сообщений" if rag_size else "нет (запусти python index.py)"

    print("=" * 55)
    print(f"  JARVIS — online")
    print(f"  Account: {me.first_name} (@{me.username})")
    print(f"  RAG индекс: {rag_status}")
    print("=" * 55)

    bot = JarvisBot(
        client=client,
        memory=memory,
        model=config.MODEL,
        batch_wait_sec=config.BATCH_WAIT_SEC,
        scan_messages=config.SCAN_MESSAGES,
        context_window=config.CONTEXT_WINDOW,
        scan_contacts=config.SCAN_CONTACTS,
    )

    await bot.prescan()

    bot.register_handlers()

    print("\n" + "=" * 55)
    print("  Listening for messages. Ctrl+C to stop.")
    memory.print_stats()
    print("=" * 55 + "\n")

    await client.run_until_disconnected()


if __name__ == "__main__":
    asyncio.run(run())
