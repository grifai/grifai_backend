"""
Max messenger runner — запускает только Max интеграцию (без Telegram).

Требования:
  - MAX_BOT_TOKEN в .env (получить у MasterBot в приложении Max)
  - OPENAI_KEY в .env
  - Установить библиотеку: pip install maxapi

Запуск: python run_max.py
"""
import asyncio

import ai
import config
import rag
from memory import JarvisMemory

_memory: JarvisMemory = None


async def _ask(prompt: str) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: input(prompt).strip())


async def on_max_message(platform, chat_id, sender_name, sender_id, text):
    print(f"\n{'=' * 55}")
    print(f"[Max] From {sender_name}: {text}")

    rag_results = rag.search(text, k=5, min_score=0.4)
    rag_context = rag.format_rag_context(rag_results)

    contact_id = f"max_{sender_id}"

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
    ch = await _ask("-> ")

    if ch == "1":
        _memory.add_example(sender_name, text, draft, "approved", draft)
        return draft
    elif ch == "2":
        final = await _ask("Your text: ")
        _memory.add_example(sender_name, text, draft, "revised", final)
        return final
    else:
        _memory.add_example(sender_name, text, draft, "skipped")
        return None


async def run():
    global _memory

    if not config.MAX_BOT_TOKEN:
        print("MAX_BOT_TOKEN не задан в .env — выход.")
        print("Получи токен у MasterBot в приложении Max.")
        return

    try:
        from sources.max import MaxSource
    except ImportError as e:
        print(f"Не удалось импортировать MaxSource: {e}")
        print("Установи библиотеку: pip install maxapi")
        return

    ai.init_openai(config.OPENAI_KEY)
    rag.init(config.OPENAI_KEY)
    _memory = JarvisMemory(config.MEMORY_FILE)

    max_source = MaxSource(
        token=config.MAX_BOT_TOKEN,
        on_message=on_max_message,
        memory=_memory,
        model=config.MODEL,
    )

    await max_source.start()
    await max_source.prescan()

    print("\n" + "=" * 55)
    print("  Max — слушаю сообщения. Ctrl+C для остановки.")
    _memory.print_stats()
    print("=" * 55 + "\n")

    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(run())
