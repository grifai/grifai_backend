"""
Compose flow — позволяет написать сообщение контакту с заданным намерением.
Используется в Telegram, WhatsApp, VK.
"""
import asyncio

import ai
from memory import JarvisMemory


async def _ask(prompt: str) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: input(prompt).strip())


def _search_contact(query: str, contacts: list[tuple[str, str, dict]]) -> list[tuple[int, str, str]]:
    """
    Ищет контакт по имени (нечёткий поиск).
    contacts: [(contact_id, name, profile), ...]
    Возвращает [(index, contact_id, name), ...]
    """
    q = query.lower()
    results = []
    for i, (cid, name, _) in enumerate(contacts):
        if q in name.lower():
            results.append((i, cid, name))
    return results


async def compose_flow(
    memory: JarvisMemory,
    model: str,
    send_fn,  # async (contact_id, name, text) -> bool
    platform: str = "telegram",
    prefix: str = "",  # фильтр по префиксу contact_id (напр. "wa_" для WhatsApp)
    contacts_override: list[tuple[str, str, dict]] | None = None,  # [(id, name, profile)]
):
    """
    Интерактивный flow для отправки сообщения:
    1. Показывает список контактов
    2. Пользователь выбирает по номеру или имени
    3. Вводит намерение (что хочет сказать)
    4. AI генерирует сообщение в стиле пользователя
    5. Approval UI (отправить / редактировать / перегенерировать / отмена)
    """
    if contacts_override is not None:
        contacts = contacts_override
    else:
        raw_contacts = memory.data.get("contacts", {})
        contacts = [
            (cid, data.get("name", cid), data.get("profile", {}))
            for cid, data in raw_contacts.items()
            if not prefix or cid.startswith(prefix)
        ]
        contacts.sort(key=lambda x: x[1].lower())

    if not contacts:
        print(f"Контактов не найдено{' для ' + platform if prefix else ''}.")
        print("Сначала запусти сканирование контактов.")
        return

    # ── Показываем список ──────────────────────────────────────────────────────
    print(f"\n{'=' * 55}")
    print(f"  Контакты ({len(contacts)}):")
    print(f"{'=' * 55}")
    for i, (_, name, profile) in enumerate(contacts, 1):
        rel = ""
        if isinstance(profile, dict) and profile.get("relationship"):
            rel = f"  [{profile['relationship'][:30]}]"
        print(f"  {i:>3}.  {name}{rel}")
    print(f"{'=' * 55}")
    print("  Введи номер или часть имени (q — отмена)")

    # ── Выбор контакта ─────────────────────────────────────────────────────────
    while True:
        raw = await _ask("-> ")
        if raw.lower() in ("q", ""):
            print("Отмена.")
            return

        contact_id, name = None, None

        # Попытка выбрать по номеру
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(contacts):
                contact_id, name = contacts[idx][0], contacts[idx][1]
            else:
                print(f"  Нет контакта с номером {raw}. Попробуй ещё.")
                continue
        else:
            # Нечёткий поиск по имени
            matches = _search_contact(raw, contacts)
            if not matches:
                print(f"  Контакт '{raw}' не найден. Попробуй иначе.")
                continue
            if len(matches) == 1:
                contact_id, name = matches[0][1], matches[0][2]
            else:
                print(f"  Найдено несколько совпадений:")
                for i, (idx, cid, n) in enumerate(matches[:8], 1):
                    print(f"    {i}. {n}")
                pick = await _ask("  Выбери номер: ")
                if pick.isdigit() and 1 <= int(pick) <= len(matches):
                    contact_id, name = matches[int(pick)-1][1], matches[int(pick)-1][2]
                else:
                    print("  Отмена.")
                    return

        # ── Подтверждение контакта ─────────────────────────────────────────────
        confirm = await _ask(f"\n  Написать {name}? (Enter — да, n — нет): ")
        if confirm.lower() == "n":
            continue

        # ── Ввод намерения ─────────────────────────────────────────────────────
        print(f"\n  Что хочешь сказать {name}?")
        print("  Примеры: 'скажи что скучаю', 'напомни купить продукты', 'спроси как дела'")
        intent = await _ask("  -> ")
        if not intent:
            print("  Отмена.")
            return

        # ── Генерация ──────────────────────────────────────────────────────────
        print(f"\nГенерирую сообщение для {name}...")
        draft = ai.compose_message(
            sender=name,
            contact_id=contact_id,
            intent=intent,
            memory=memory,
            model=model,
        )

        # ── Approval loop ──────────────────────────────────────────────────────
        while True:
            print(f"\n{'=' * 55}")
            print(f"  Кому: {name}")
            print(f"  Текст: {draft!r}")
            print(f"{'=' * 55}")
            print("  1 Отправить   2 Редактировать   3 Уточнить   4 Перегенерировать   q Отмена")
            ch = await _ask("-> ")

            if ch == "1":
                ok = await send_fn(contact_id, name, draft)
                if ok:
                    memory.add_example(name, f"[compose] {intent}", draft, "approved", draft)
                    print(f"  Отправлено ✓")
                else:
                    print(f"  Ошибка отправки.")
                return

            elif ch == "2":
                draft = await _ask("  Твой текст: ")
                if not draft:
                    print("  Отмена.")
                    return
                ok = await send_fn(contact_id, name, draft)
                if ok:
                    memory.add_example(name, f"[compose] {intent}", draft, "revised", draft)
                    print(f"  Отправлено ✓")
                return

            elif ch == "3":
                instruction = await _ask("  Как переписать? (напр. 'короче', 'потеплее'): ")
                if instruction:
                    print("Переписываю...")
                    draft = ai.compose_message(
                        sender=name,
                        contact_id=contact_id,
                        intent=intent,
                        memory=memory,
                        model=model,
                        refinement=instruction,
                    )

            elif ch == "4":
                print("Перегенерирую...")
                draft = ai.compose_message(
                    sender=name,
                    contact_id=contact_id,
                    intent=intent,
                    memory=memory,
                    model=model,
                )
                continue

            else:
                print("  Отмена.")
                return
