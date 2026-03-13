import asyncio
from telethon import TelegramClient

import config
import ai
import rag
from memory import JarvisMemory
from telegram_bot import JarvisBot
from sources.whatsapp import WhatsAppSource
from sources.vk import VKSource


async def on_whatsapp_message(platform, chat_id, sender_name, sender_id, text):
    """Обрабатывает входящее WhatsApp-сообщение и возвращает ответ или None."""
    print(f"\n{'=' * 55}")
    print(f"[WhatsApp] From {sender_name}: {text}")

    rag_results = rag.search(text, k=5, min_score=0.4)
    rag_context = rag.format_rag_context(rag_results)

    contact_id = f"wa_{sender_id}"

    draft = ai.generate_reply(
        sender=sender_name,
        contact_id=contact_id,
        incoming_batch=[text],
        chat_context="",
        memory=_memory,
        model=config.MODEL,
        rag_context=rag_context,
    )

    if draft == "[SKIP]":
        print("Jarvis: no reply needed")
        return None

    print(f"Jarvis draft: {draft!r}")
    print("-" * 55)
    print("  1 Send   2 Edit   3 Skip")
    ch = input("-> ").strip()

    if ch == "1":
        _memory.add_example(sender_name, text, draft, "approved", draft)
        return draft
    elif ch == "2":
        final = input("Your text: ").strip()
        _memory.add_example(sender_name, text, draft, "revised", final)
        return final
    else:
        _memory.add_example(sender_name, text, draft, "skipped")
        return None


_memory: JarvisMemory = None


async def run():
    global _memory

    ai.init_openai(config.OPENAI_KEY)
    rag.init(config.OPENAI_KEY)
    _memory = JarvisMemory(config.MEMORY_FILE)

    # ── Telegram ──────────────────────────────────────────────────────────────
    client = TelegramClient(config.SESSION_FILE, config.API_ID, config.API_HASH)
    await client.start()
    me = await client.get_me()

    rag_size = rag.index_size()
    rag_status = f"{rag_size} сообщений" if rag_size else "нет (запусти python index.py)"

    print("=" * 55)
    print(f"  JARVIS — online")
    print(f"  Telegram: {me.first_name} (@{me.username})")
    print(f"  RAG индекс: {rag_status}")
    print("=" * 55)

    tg_bot = JarvisBot(
        client=client,
        memory=_memory,
        model=config.MODEL,
        batch_wait_sec=config.BATCH_WAIT_SEC,
        scan_messages=config.SCAN_MESSAGES,
        context_window=config.CONTEXT_WINDOW,
        scan_contacts=config.SCAN_CONTACTS,
    )

    await tg_bot.prescan()
    tg_bot.register_handlers()

    # ── WhatsApp ──────────────────────────────────────────────────────────────
    wa = WhatsAppSource(
        on_message=on_whatsapp_message,
        memory=_memory,
        model=config.MODEL,
        scan_contacts=config.SCAN_CONTACTS,
        scan_messages=config.SCAN_MESSAGES,
    )
    await wa.start()
    await wa.prescan()

    print("\n" + "=" * 55)
    print("  Listening for messages. Ctrl+C to stop.")
    _memory.print_stats()
    print("=" * 55 + "\n")

    # ── VK ────────────────────────────────────────────────────────────────────
    if config.VK_TOKEN:
        print("\n" + "!" * 55)
        print("  ВНИМАНИЕ: VK — ТЕСТОВАЯ ФУНКЦИЯ")
        print("!" * 55)
        print("  Использование автоматических ответов от имени")
        print("  пользователя нарушает правила VK и может привести к:")
        print("    - временной или постоянной блокировке аккаунта")
        print("    - ограничению доступа к сообщениям")
        print("    - другим последствиям со стороны платформы")
        print()
        print("  Нажимая Enter/«продолжить», вы подтверждаете, что")
        print("  осознаёте риски и берёте ответственность на себя.")
        print("!" * 55)
        confirm = input("\n  Продолжить? (да / нет): ").strip().lower()
        if confirm not in ("да", "д", "yes", "y"):
            print("VK: отменено.")
        else:
            vk = VKSource(
                on_message=on_whatsapp_message,
                memory=_memory,
                model=config.MODEL,
                token=config.VK_TOKEN,
                scan_contacts=config.SCAN_CONTACTS,
                scan_messages=config.SCAN_MESSAGES,
            )
            await vk.start()
            await vk.prescan()
    else:
        print("VK: skipped (VK_TOKEN not set)")

    await client.run_until_disconnected()


if __name__ == "__main__":
    asyncio.run(run())
