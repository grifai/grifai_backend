#!/usr/bin/env python3
"""
Дневное саммари переписок.

Использование:
    python summary.py           — за последние 24 часа
    python summary.py 48        — за последние 48 часов
    python summary.py 72        — за 3 дня
"""

import asyncio
import sys
from datetime import datetime, timedelta, timezone

from telethon import TelegramClient

import ai
import config

PROMPT = """Ты — личный ассистент. Пользователь дал тебе свои Telegram переписки за последние {hours} часов.

Составь СТРУКТУРИРОВАННЫЙ отчёт на русском:

## 📬 Требуют ответа
Контакты, которым ещё не ответил или разговор завис. Укажи суть.

## 🔑 Ключевые события
Договорённости, решения, важные новости из диалогов.

## 💬 Активность
С кем общался, что обсуждал (кратко по контактам).

## ⚡ На завтра
Что нужно не забыть сделать или ответить.

Пиши кратко. Если раздел пустой — пропусти его."""


async def run(hours: int = 24):
    ai.init_openai(config.OPENAI_KEY)

    client = TelegramClient(config.SESSION_FILE, config.API_ID, config.API_HASH)
    await client.start()
    me = await client.get_me()
    my_id = me.id

    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    print(f"Сбор сообщений за последние {hours}ч...")

    active_dialogs: list[str] = []
    scanned = 0

    async for dialog in client.iter_dialogs(limit=500):
        if not dialog.is_user:
            continue
        if scanned >= config.SCAN_CONTACTS:
            break

        name = dialog.name or str(dialog.id)
        recent: list[tuple[str, str]] = []

        async for msg in client.iter_messages(dialog.id, limit=80):
            if not msg.text:
                continue
            if msg.date < cutoff:
                break
            who = "Я" if msg.sender_id == my_id else name
            recent.append((who, msg.text[:300]))

        if recent:
            recent.reverse()
            block = "\n".join(f"[{who}]: {text}" for who, text in recent)
            active_dialogs.append(f"=== {name} ===\n{block}")

        scanned += 1

    await client.disconnect()

    if not active_dialogs:
        print(f"За последние {hours} часов нет активных переписок.")
        return

    print(f"Активно: {len(active_dialogs)} чатов. Генерирую саммари...")

    full_text = "\n\n".join(active_dialogs)
    if len(full_text) > 14000:
        full_text = full_text[:14000] + "\n...(обрезано)"

    resp = ai._oai().chat.completions.create(
        model=config.MODEL,
        max_tokens=1200,
        messages=[
            {"role": "system", "content": PROMPT.format(hours=hours)},
            {"role": "user", "content": full_text},
        ],
    )
    summary = resp.choices[0].message.content.strip()

    now = datetime.now().strftime("%d.%m.%Y %H:%M")
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  JARVIS SUMMARY — {now}  (последние {hours}ч)")
    print(f"  Диалогов с активностью: {len(active_dialogs)}")
    print(f"{sep}\n")
    print(summary)
    print(f"\n{sep}\n")


if __name__ == "__main__":
    hours = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 24
    asyncio.run(run(hours))
