"""
WhatsApp runner — запускает только WhatsApp интеграцию (без Telegram).

Требования:
  - Запущенный Baileys-мост: node whatsapp_bridge/index.js
  - В .env заданы OPENAI_KEY (или нужные ключи)

Запуск: python run_whatsapp.py
"""
import asyncio
import signal

import ai
import config
import rag
from compose import compose_flow
from memory import JarvisMemory
from sources.whatsapp import WhatsAppSource

_memory: JarvisMemory = None
_wa: WhatsAppSource = None


async def _ask(prompt: str) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: input(prompt).strip())


async def on_whatsapp_message(platform, chat_id, sender_name, sender_id, text):
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

    print(f"\n{'=' * 55}")
    print(f"[WhatsApp] From {sender_name}: {text[:200]}")
    print("-" * 55)

    while True:
        print(f"Jarvis: {draft!r}")
        print("  1 Send   2 Edit   3 Refine   4 Regen   5 Skip")
        ch = await _ask("-> ")

        if ch == "1":
            _memory.add_example(sender_name, text, draft, "approved", draft)
            return draft
        elif ch == "2":
            final = await _ask("Your text: ")
            _memory.add_example(sender_name, text, draft, "revised", final)
            return final
        elif ch == "3":
            instruction = await _ask("  Как переписать? (напр. 'короче', 'неформально'): ")
            if instruction:
                print("Переписываю...")
                draft = ai.generate_reply(
                    sender=sender_name,
                    contact_id=f"wa_{sender_id}",
                    incoming_batch=[text],
                    chat_context="",
                    memory=_memory,
                    model=config.MODEL,
                    rag_context=rag.format_rag_context(rag.search(text, k=5, min_score=0.4)),
                    refinement=instruction,
                )
        elif ch == "4":
            print("Перегенерирую...")
            draft = ai.generate_reply(
                sender=sender_name,
                contact_id=f"wa_{sender_id}",
                incoming_batch=[text],
                chat_context="",
                memory=_memory,
                model=config.MODEL,
                rag_context=rag.format_rag_context(rag.search(text, k=5, min_score=0.4)),
            )
        else:
            _memory.add_example(sender_name, text, draft, "skipped")
            return None


async def _process_message(data: dict):
    """Обрабатывает одно сообщение из очереди с approval UI."""
    chat_id = data["chat_id"]
    sender_name = data.get("sender_name", chat_id)
    sender_id = data.get("sender_id", chat_id)
    text = data["text"]

    reply = await on_whatsapp_message(
        platform="whatsapp",
        chat_id=chat_id,
        sender_name=sender_name,
        sender_id=sender_id,
        text=text,
    )
    if reply:
        await _wa.send(chat_id, reply)


async def _listen_mode():
    stop = asyncio.Event()
    loop = asyncio.get_event_loop()
    loop.add_signal_handler(signal.SIGINT, stop.set)
    print("\nСлушаю новые сообщения... Ctrl+C для возврата в меню\n")

    while not stop.is_set():
        try:
            data = await asyncio.wait_for(_wa.queue.get(), timeout=0.5)
            await _process_message(data)
        except asyncio.TimeoutError:
            continue

    loop.remove_signal_handler(signal.SIGINT)
    print("\nВозврат в меню...")


async def run():
    global _memory, _wa

    ai.init_openai(config.OPENAI_KEY)
    rag.init(config.OPENAI_KEY)
    _memory = JarvisMemory(config.MEMORY_FILE)

    _wa = WhatsAppSource(
        on_message=on_whatsapp_message,
        memory=_memory,
        model=config.MODEL,
        scan_contacts=config.SCAN_CONTACTS,
        scan_messages=config.SCAN_MESSAGES,
    )

    bridge_up = await _wa.is_bridge_up()
    if not bridge_up:
        print("WhatsApp bridge не запущен.")
        print("Запусти сначала: node whatsapp_bridge/index.js")
        return

    # Ждём подключения WhatsApp (до 15 секунд)
    print("Проверяю подключение WhatsApp", end="", flush=True)
    for _ in range(15):
        if await _wa.is_connected():
            print(" ✓")
            break
        print(".", end="", flush=True)
        await asyncio.sleep(1)
    else:
        print("\nWhatsApp не подключён — отсканируй QR-код в терминале моста и перезапусти.")
        return

    await _wa.start()

    print("=" * 55)
    print("  WhatsApp — online")
    print("=" * 55)

    while True:
        print("\n" + "=" * 55)
        print("  Что делаем?")
        print("  1  Слушать новые сообщения")
        print("  2  Непрочитанные сообщения")
        print("  3  Все последние чаты")
        print("  4  Написать сообщение")
        print("  5  Пересканировать контакты")
        print("  6  Статистика памяти")
        print("  7  Инструкции для AI")
        print("  q  Выход")
        print("=" * 55)

        ch = await _ask("-> ")

        if ch == "1":
            await _listen_mode()
        elif ch == "2":
            result = await _wa.review_unread(on_approval=on_whatsapp_message, show_all=False)
            if result == "no_unread":
                print("  Непрочитанных нет. Показываю все последние чаты...")
                await _wa.review_unread(on_approval=on_whatsapp_message, show_all=True)
        elif ch == "3":
            await _wa.review_unread(on_approval=on_whatsapp_message, show_all=True)
        elif ch == "4":
            async def wa_send(contact_id, name, text):
                try:
                    chat_id = contact_id.removeprefix("wa_")
                    await _wa.send(chat_id, text)
                    return True
                except Exception as e:
                    print(f"  Ошибка: {e}")
                    return False
            wa_contacts = await _wa.get_contacts()
            await compose_flow(
                _memory, config.MODEL, wa_send,
                platform="whatsapp",
                contacts_override=wa_contacts,
            )
        elif ch == "5":
            await _wa.prescan()
        elif ch == "6":
            _memory.print_stats()
        elif ch == "7":
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
            print("  Введи 1-7 или q")


if __name__ == "__main__":
    asyncio.run(run())
