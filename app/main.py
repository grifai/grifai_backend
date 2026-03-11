import asyncio

import uvicorn
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI
from telethon import TelegramClient

from app.api.server import app
from app.bot.client import JarvisBot
from app.config import settings
from app.llm import get_embeddings, get_llm
from app.llm import openai_provider as ai
from app.memory import rag
from app.memory.contacts import JarvisMemory
from app.services.consumers import (
    approval_ui_consumer,
    ghost_writer_consumer,
    learner_consumer,
    profiler_consumer,
)
from app.services.digest import DigestService
from app.services.event_bus import EventBus
from app.services.ghost_writer import GhostWriter
from app.services.tts import TTSService
from app.utils.logging import setup_logging

app = FastAPI()


@app.get("/")
def root():
    return {"status": "ok"}


async def run():
    setup_logging()

    ai.init_openai(settings.openai_key)
    rag.init(settings.openai_key)
    memory = JarvisMemory(settings.memory_file)

    client = TelegramClient(
        settings.session_file, settings.tg_api_id, settings.tg_api_hash
    )
    await client.start()
    me = await client.get_me()

    rag_size = rag.index_size()
    rag_status = (
        f"{rag_size} сообщений"
        if rag_size
        else "нет (запусти: python scripts/index.py)"
    )

    # ── Services ───────────────────────────────────────────────────────────────
    tts = TTSService(
        elevenlabs_key=settings.elevenlabs_key,
        openai_key=settings.openai_key,
    )
    digest_service = DigestService(
        llm=get_llm(api_key=settings.openai_key, model=settings.model),
        vector_memory=rag._get(),
        tts=tts,
    )

    bus = EventBus(settings.redis_url)
    await bus.connect()

    ghost_writer = GhostWriter(
        llm=get_llm(api_key=settings.openai_key, model=settings.model),
        vector_memory=rag._get(),
        contact_store=memory,
    )

    bot = JarvisBot(
        client=client,
        memory=memory,
        model=settings.model,
        batch_wait_sec=settings.batch_wait_sec,
        scan_messages=settings.scan_messages,
        context_window=settings.context_window,
        scan_contacts=settings.scan_contacts,
        ghost_writer=ghost_writer,
        event_bus=bus,
    )

    await bot.prescan()
    bot.register_handlers()

    print("=" * 55)
    print(f"  JARVIS — online")
    print(f"  Account: {me.first_name} (@{me.username})")
    print(f"  RAG индекс: {rag_status}")
    print(f"  API: http://{settings.api_host}:{settings.api_port}/docs")
    print("=" * 55)
    print("\n  Listening for messages. Ctrl+C to stop.")
    memory.print_stats()
    print("=" * 55 + "\n")

    # Share state with FastAPI before server starts
    app.state.memory = memory
    app.state.telegram_connected = True
    app.state.telegram_account = {
        "name": me.first_name,
        "username": me.username,
        "id": me.id,
    }
    app.state.tts = tts

    # ── Scheduled digests ──────────────────────────────────────────────────────
    scheduler = AsyncIOScheduler()

    async def morning_digest():
        print("\n[scheduler] Running morning digest...")
        result = await digest_service.generate_and_speak(hours=12, client=client)
        print(
            f"[scheduler] Morning digest: {result.dialogs_count} dialogs, audio={result.audio_path}"
        )

    async def evening_digest():
        print("\n[scheduler] Running evening digest...")
        result = await digest_service.generate_and_speak(hours=24, client=client)
        print(
            f"[scheduler] Evening digest: {result.dialogs_count} dialogs, audio={result.audio_path}"
        )

    scheduler.add_job(morning_digest, "cron", hour=7, minute=0)
    scheduler.add_job(evening_digest, "cron", hour=22, minute=0)
    scheduler.start()

    # ── Server config ──────────────────────────────────────────────────────────
    server_config = uvicorn.Config(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_level="warning",
        loop="none",
    )
    server = uvicorn.Server(server_config)

    try:
        await asyncio.gather(
            client.run_until_disconnected(),
            server.serve(),
            ghost_writer_consumer(
                bus,
                ghost_writer,
                client,
                memory,
                context_window=settings.context_window,
                scan_messages=settings.scan_messages,
            ),
            approval_ui_consumer(bus, ghost_writer, client, memory),
            learner_consumer(bus, ghost_writer),
            profiler_consumer(
                bus, client, memory, scan_messages=settings.scan_messages
            ),
        )
    finally:
        scheduler.shutdown(wait=False)
        await bus.close()


if __name__ == "__main__":
    asyncio.run(run())
