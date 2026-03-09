#!/usr/bin/env python3
"""
Индексатор переписок в RAG.

Запуск (из корня проекта):
    python scripts/index.py          — проиндексировать первые SCAN_CONTACTS личных диалогов
    python scripts/index.py --force  — переиндексировать даже если уже есть индекс
    python scripts/index.py --update — добавить только новые сообщения (быстро, каждый день)
"""

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from telethon import TelegramClient
from app.config import settings
from app.memory import rag
from app.bot.client import fetch_messages


async def run(force: bool = False):
    rag.init(settings.openai_key)
    if not force:
        size = rag.index_size()
        if size > 0:
            print(f"RAG index уже существует ({size} сообщений).")
            print("Используй --force чтобы переиндексировать.")
            return
    client = TelegramClient(settings.session_file, settings.tg_api_id, settings.tg_api_hash)
    await client.start()
    me = await client.get_me()
    print(f"Подключён как {me.first_name} (@{me.username})")
    print(f"Сканирую до {settings.scan_contacts} личных диалогов...")

    docs: list[dict] = []
    scanned = 0

    async for dialog in client.iter_dialogs(limit=500):
        if not dialog.is_user:
            continue
        if scanned >= settings.scan_contacts:
            break

        name = dialog.name or str(dialog.id)
        msgs, my_c, their_c = await fetch_messages(client, dialog.id, settings.scan_messages)

        added = 0
        for m in msgs:
            if not m["text"].strip():
                continue
            docs.append({
                "text": m["text"],
                "contact_name": name,
                "mine": m["mine"],
                "date": m["date"],
            })
            added += 1

        print(f"  {name}: {added} сообщений")
        scanned += 1
        await asyncio.sleep(0.2)

    await client.disconnect()

    if not docs:
        print("Нет сообщений для индексации.")
        return

    print(f"\nВсего собрано: {len(docs)} сообщений из {scanned} диалогов")
    rag.build_index(docs)
    print("Готово! Теперь доступны:\n  python -m app.services.search <запрос>\n  python -m app.main")


async def run_update():
    """Fetch only messages newer than the current index and append them."""
    rag.init(settings.openai_key)
    max_date = rag.get_max_date()
    if not max_date:
        print("Индекс пуст. Запусти: python scripts/index.py")
        return

    cutoff = datetime.fromisoformat(max_date).replace(tzinfo=timezone.utc)
    print(f"Обновление индекса: добавляю сообщения новее {max_date[:10]}...")

    client = TelegramClient(settings.session_file, settings.tg_api_id, settings.tg_api_hash)
    await client.start()
    me = await client.get_me()
    print(f"Подключён как {me.first_name} (@{me.username})")

    new_docs: list[dict] = []
    scanned = 0

    async for dialog in client.iter_dialogs(limit=500):
        if not dialog.is_user:
            continue
        if scanned >= settings.scan_contacts:
            break

        name = dialog.name or str(dialog.id)
        added = 0

        async for msg in client.iter_messages(dialog.id, limit=settings.scan_messages):
            if not msg.text or not msg.text.strip():
                continue
            msg_dt = msg.date.replace(tzinfo=timezone.utc) if msg.date.tzinfo is None else msg.date
            if msg_dt <= cutoff:
                break
            new_docs.append({
                "text": msg.text,
                "contact_name": name,
                "mine": msg.sender_id == me.id,
                "date": msg.date.isoformat(),
            })
            added += 1

        if added:
            print(f"  {name}: +{added} новых сообщений")
        scanned += 1
        await asyncio.sleep(0.2)

    await client.disconnect()

    if not new_docs:
        print("Нет новых сообщений.")
        return

    print(f"\nВсего новых: {len(new_docs)} сообщений из {scanned} диалогов")
    rag.append_to_index(new_docs)
    print("Готово!")


if __name__ == "__main__":
    if "--update" in sys.argv:
        asyncio.run(run_update())
    else:
        force = "--force" in sys.argv
        asyncio.run(run(force=force))
