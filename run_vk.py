"""
VK runner — запускает только VK интеграцию (без Telegram).

Требования:
  - VK_TOKEN в .env (пользовательский токен с правами messages)
  - OPENAI_KEY в .env

ВНИМАНИЕ: автоответы от имени пользователя нарушают правила VK.
Возможна блокировка аккаунта. Используйте на свой риск.

Запуск: python run_vk.py
"""
import asyncio
import signal

import ai
import config
import rag
from compose import compose_flow
from memory import JarvisMemory
from sources.vk import VKSource

_memory: JarvisMemory = None
_vk: VKSource = None


async def _ask(prompt: str) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: input(prompt).strip())


async def _generate(sender_name, sender_id, text, refinement=""):
    """Генерирует черновик в executor чтобы не блокировать event loop."""
    loop = asyncio.get_event_loop()
    rag_results = rag.search(text, k=5, min_score=0.4)
    rag_context = rag.format_rag_context(rag_results)
    contact_id = f"vk_{sender_id}"
    return await loop.run_in_executor(
        None,
        lambda: ai.generate_reply(
            sender=sender_name,
            contact_id=contact_id,
            incoming_batch=[text],
            chat_context="",
            memory=_memory,
            model=config.MODEL,
            rag_context=rag_context,
            refinement=refinement,
        ),
    )


async def _approval_loop(sender_name, sender_id, text, draft) -> str | None:
    """Approval UI: показывает черновик и спрашивает что делать."""
    while True:
        print(f"\nJarvis: {draft!r}")
        print("  1 Отправить   2 Редактировать   3 Уточнить   4 Перегенерировать   5 Пропустить")
        ch = await _ask("-> ")

        if ch == "1":
            _memory.add_example(sender_name, text, draft, "approved", draft)
            return draft
        elif ch == "2":
            final = await _ask("  Твой текст: ")
            _memory.add_example(sender_name, text, draft, "revised", final)
            return final
        elif ch == "3":
            instruction = await _ask("  Как переписать? (напр. 'короче', 'неформально'): ")
            if instruction:
                print("  Переписываю...")
                draft = await _generate(sender_name, sender_id, text, refinement=instruction)
        elif ch == "4":
            print("  Перегенерирую...")
            draft = await _generate(sender_name, sender_id, text)
        else:
            _memory.add_example(sender_name, text, draft, "skipped")
            return None


async def on_vk_message(platform, chat_id, sender_name, sender_id, text):
    print(f"\n{'=' * 55}")
    print(f"[VK] От {sender_name}: {text[:200]}")
    print("  Генерирую черновик...")
    draft = await _generate(sender_name, sender_id, text)

    if draft == "[SKIP]":
        print("  Grif: ответ не нужен")
        return None

    return await _approval_loop(sender_name, sender_id, text, draft)


async def _review_unread():
    """Интерактивный просмотр непрочитанных — сначала список, потом детали."""
    print("\nЗагружаю непрочитанные...")
    chats = await _vk.get_unread_chats()

    if not chats:
        print("  Нет непрочитанных диалогов.")
        return

    while True:
        print(f"\n{'=' * 55}")
        print(f"  Непрочитанные ({len(chats)}):")
        print(f"{'=' * 55}")
        for i, chat in enumerate(chats, 1):
            last_in = next(
                (m["text"] for m in reversed(chat["messages"]) if not m["mine"]), ""
            )
            print(f"  {i:>3}.  {chat['name']} [{chat['unread']} непрочит.]")
            if last_in:
                print(f"         >> {last_in[:70]}")
        print(f"{'=' * 55}")
        print("  Введи номер чата или q — выход")

        raw = await _ask("-> ")
        if raw.lower() in ("q", ""):
            return
        if not raw.isdigit() or not (1 <= int(raw) <= len(chats)):
            print("  Неверный номер.")
            continue

        chat = chats[int(raw) - 1]

        # Показываем историю переписки
        print(f"\n{'=' * 55}")
        print(f"  [VK] {chat['name']} — {chat['unread']} непрочит.")
        print(f"{'=' * 55}")
        for m in chat["messages"]:
            who = "Я" if m["mine"] else chat["name"]
            print(f"  [{who}]: {m['text'][:200]}")
        print(f"{'=' * 55}")
        print("  1 Ответить (AI)   2 Написать вручную   3 Другой чат   q Выход")

        ch = await _ask("-> ")
        if ch == "1":
            incoming_text = "\n".join(
                m["text"] for m in chat["messages"] if not m["mine"]
            )
            print("  Генерирую черновик...")
            draft = await _generate(chat["name"], str(chat["peer_id"]), incoming_text)
            if draft == "[SKIP]":
                print("  Grif: ответ не нужен")
            else:
                reply = await _approval_loop(
                    chat["name"], str(chat["peer_id"]), incoming_text, draft
                )
                if reply:
                    await asyncio.sleep(1)
                    await _vk.send_message(chat["peer_id"], reply)
                    print("  Отправлено ✓")
                    chats.remove(chat)
        elif ch == "2":
            text = await _ask("  Твой текст: ")
            if text:
                await _vk.send_message(chat["peer_id"], text)
                print("  Отправлено ✓")
                chats.remove(chat)
        elif ch == "3":
            continue
        else:
            return


async def _listen_mode():
    stop = asyncio.Event()
    loop = asyncio.get_event_loop()
    loop.add_signal_handler(signal.SIGINT, stop.set)
    print("\nСлушаю новые сообщения... Ctrl+C для возврата в меню\n")
    await stop.wait()
    loop.remove_signal_handler(signal.SIGINT)
    print("\nВозврат в меню...")


async def run():
    global _memory, _vk

    if not config.VK_TOKEN:
        print("VK_TOKEN не задан в .env — выход.")
        return

    print("\n" + "!" * 55)
    print("  ВНИМАНИЕ: VK — ТЕСТОВАЯ ФУНКЦИЯ")
    print("!" * 55)
    print("  Автоответы от имени пользователя нарушают правила VK.")
    print("  Возможна блокировка аккаунта. Используйте на свой риск.")
    print("!" * 55)
    confirm = (await _ask("\n  Продолжить? (да / нет): ")).lower()
    if confirm not in ("да", "д", "yes", "y"):
        print("Отменено.")
        return

    ai.init_openai(config.OPENAI_KEY)
    rag.init(config.OPENAI_KEY)
    _memory = JarvisMemory(config.MEMORY_FILE)

    _vk = VKSource(
        on_message=on_vk_message,
        memory=_memory,
        model=config.MODEL,
        token=config.VK_TOKEN,
        scan_contacts=config.SCAN_CONTACTS,
        scan_messages=config.SCAN_MESSAGES,
    )

    await _vk.start()

    print("=" * 55)
    print("  VK — online")
    print("=" * 55)

    while True:
        print("\n" + "=" * 55)
        print("  Что делаем?")
        print("  1  Слушать новые сообщения")
        print("  2  Просмотреть непрочитанные")
        print("  3  Написать сообщение")
        print("  4  Пересканировать контакты")
        print("  5  Статистика памяти")
        print("  6  Инструкции для AI")
        print("  q  Выход")
        print("=" * 55)

        ch = await _ask("-> ")

        if ch == "1":
            await _listen_mode()
        elif ch == "2":
            await _review_unread()
        elif ch == "3":
            async def vk_send(contact_id, name, text):
                try:
                    peer_id = int(contact_id.removeprefix("vk_"))
                    await _vk.send_message(peer_id, text)
                    return True
                except Exception as e:
                    print(f"  Ошибка: {e}")
                    return False
            vk_contacts = await _vk.get_contacts()
            await compose_flow(
                _memory, config.MODEL, vk_send,
                platform="vk",
                contacts_override=vk_contacts,
            )
        elif ch == "4":
            await _vk.prescan()
        elif ch == "5":
            _memory.print_stats()
        elif ch == "6":
            current = _memory.get_personal_prompt()
            print(f"\n  Текущий промпт: {current!r}" if current else "\n  Промпт не задан.")
            print("  Введи новый промпт (Enter — оставить, 'clear' — удалить):")
            new_prompt = await _ask("  -> ")
            if new_prompt.lower() == "clear":
                _memory.set_personal_prompt("")
                print("  Промпт удалён.")
            elif new_prompt:
                _memory.set_personal_prompt(new_prompt)
                print("  Промпт сохранён ✓")
        elif ch in ("q", "Q"):
            print("Выход.")
            break
        else:
            print("  Введи 1-6 или q")


if __name__ == "__main__":
    asyncio.run(run())
