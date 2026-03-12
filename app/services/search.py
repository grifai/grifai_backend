#!/usr/bin/env python3
"""
Поиск по переписке на естественном языке.

Просто пиши что хочешь узнать:

    python -m app.services.search 'Сколько раз я написал привет Лизе?'
    python -m app.services.search 'О чём мы говорили с Лизой вчера?'
    python -m app.services.search 'С кем я обсуждал сломанную дверь?'
    python -m app.services.search 'Кто писал мне про деньги?'
"""

import json
import sys
from datetime import datetime, timedelta

from openai import OpenAI

from app.config import settings
from app.llm.prompts import SEARCH_ANSWER_PROMPT
from app.memory import rag

# ── GPT-парсер намерения ───────────────────────────────────────────────────────

_PARSE_SYSTEM = """\
Ты — парсер запросов к истории переписок. Сегодня: {today} (вчера: {yesterday}).

Разбери запрос пользователя и верни JSON (только JSON, без markdown):
{{
  "intent": "count" | "analyze" | "search",
  "contact": "имя контакта или null",
  "search_term": "точное слово/фраза для подсчёта (только для count, иначе null)",
  "question": "вопрос для анализа (для analyze/search)",
  "only_mine": true или false,
  "date_from": "YYYY-MM-DD или null",
  "date_to": "YYYY-MM-DD или null",
  "plan": "одна строка — что именно ты сделаешь"
}}

Правила:
- intent=count: "сколько раз", "как часто" → точный текстовый поиск без LLM
- intent=analyze: открытый вопрос о разговоре с КОНКРЕТНЫМ контактом → GPT читает весь диалог
- intent=search: найди похожие сообщения → семантический поиск
- ВАЖНО: если в запросе есть "с кем", "кто", "кому" БЕЗ конкретного имени — ВСЕГДА intent=search, contact=null
- ВАЖНО: если контакт НЕ указан — ВСЕГДА intent=search (не analyze!), иначе придётся читать 75000 сообщений
- only_mine=true: если спрашивают что Я говорил/писал/обещал
- only_mine=false: если спрашивают что писали оба или только собеседник, или "с кем обсуждал"
- "вчера" → date_from=date_to={yesterday}
- "сегодня" → date_from=date_to={today}
- "на этой неделе" → date_from={week_start}, date_to={today}
- "за последние 7 дней" → date_from={week_ago}, date_to={today}
- Если дат нет — оставь null
- contact: извлеки имя если упомянуто явно, иначе null
- search_term: только для count — конкретное слово которое нужно посчитать
"""


def parse_query(query: str) -> dict:
    """Использует GPT чтобы понять что хочет пользователь."""
    client = OpenAI(api_key=settings.openai_key)

    today = datetime.now()
    yesterday = today - timedelta(days=1)
    week_start = today - timedelta(days=today.weekday())
    week_ago = today - timedelta(days=7)

    system = _PARSE_SYSTEM.format(
        today=today.strftime("%Y-%m-%d"),
        yesterday=yesterday.strftime("%Y-%m-%d"),
        week_start=week_start.strftime("%Y-%m-%d"),
        week_ago=week_ago.strftime("%Y-%m-%d"),
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=300,
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": query},
        ],
    )
    raw = resp.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
    return json.loads(raw)


# ── Answer generation ──────────────────────────────────────────────────────────


def format_search_results_for_llm(results: list[dict]) -> str:
    lines = []
    for r in results:
        score = r.get("score", 0)
        if score > 0 and score < 0.43:
            continue
        contact = r.get("contact_name", "?")
        date = r.get("date", "")[:10]
        text = r.get("text", "")[:300]
        direction = f"Я → {contact}" if r.get("mine") else f"{contact} → Мне"
        score_str = f" (релевантность: {score:.0%})" if score > 0 else ""
        lines.append(f"[{date}] {direction}{score_str}: {text}")
    return "\n".join(lines)


def answer_search(query: str, results: list[dict]) -> str:
    context = format_search_results_for_llm(results)
    if not context:
        context = format_search_results_for_llm([{**r, "score": 1.0} for r in results])
    client = OpenAI(api_key=settings.openai_key)
    user_msg = f"Результаты поиска по переписке:\n\n{context}\n\nВопрос пользователя: {query}"
    resp = client.chat.completions.create(
        model=settings.model,
        max_tokens=400,
        temperature=0.3,
        messages=[
            {"role": "system", "content": SEARCH_ANSWER_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    return resp.choices[0].message.content.strip()


# ── Executors ─────────────────────────────────────────────────────────────────


def run_count(params: dict):
    term = params.get("search_term") or ""
    if not term:
        print("Не удалось извлечь слово для подсчёта. Попробуй: 'Сколько раз я написал «привет» Лизе?'")
        return

    contact = params.get("contact") or ""
    only_mine = params.get("only_mine", False)
    date_from = params.get("date_from")
    date_to = params.get("date_to")

    if contact:
        names = rag.list_matching_contacts(contact)
        if not names:
            print(f"Контакт '{contact}' не найден в индексе.")
            return
        print(f"Контакт: {', '.join(names)}")

    date_str = f"  за {date_from} — {date_to}" if date_from else ""
    mine_str = " (только мои)" if only_mine else ""
    print(f"Считаю точные вхождения: «{term}»{mine_str}{date_str}\n")

    total, matches = rag.count_and_find(term, contact, only_mine, date_from, date_to)

    if total == 0:
        all_c = rag.get_contact_messages(contact)
        if all_c:
            dates = [m["date"][:10] for m in all_c]
            print(f"«{term}» не найдено.")
            print(f'Индекс для {contact or "всех"}: {min(dates)} — {max(dates)}, {len(all_c)} сообщений.')
            print(f"Нужная переписка может быть старше. Переиндексируй: python scripts/index.py --force")
        else:
            print(f"«{term}» не найдено и нет сообщений для контакта «{contact}» в индексе.")
        return

    by_who: dict[str, int] = {}
    by_date: dict[str, int] = {}
    for m in matches:
        who = "Я" if m["mine"] else m["contact_name"]
        by_who[who] = by_who.get(who, 0) + 1
        d = m["date"][:10]
        by_date[d] = by_date.get(d, 0) + 1

    print(f"Всего совпадений: {total}")
    for who, cnt in sorted(by_who.items(), key=lambda x: -x[1]):
        print(f"  {who}: {cnt} раз")

    if len(by_date) > 1:
        print(f"\nПо датам:")
        for d, cnt in sorted(by_date.items()):
            print(f"  {d}: {cnt}")

    print(f"\nПримеры ({min(10, total)} из {total}):")
    print("─" * 65)
    for m in matches[:10]:
        who = "Я" if m["mine"] else m["contact_name"]
        date = m["date"][:10]
        text = m["text"]
        idx = text.lower().find(term.lower())
        start = max(0, idx - 35)
        snippet = ("…" if start > 0 else "") + text[start : idx + len(term) + 60].strip()
        print(f"[{date}] {who}: {snippet}")
    if total > 10:
        print(f"  … и ещё {total - 10}")
    print()


def run_analyze(params: dict):
    contact = params.get("contact") or ""
    only_mine = False  # always both sides for dialog analysis
    date_from = params.get("date_from")
    date_to = params.get("date_to")
    question = params.get("question") or params.get("query", "")

    if contact:
        names = rag.list_matching_contacts(contact)
        if not names:
            print(f"Контакт '{contact}' не найден.")
            return
        print(f"Контакт: {', '.join(names)}")

    msgs = rag.get_contact_messages(contact, only_mine, date_from, date_to)

    if not msgs:
        period = f" за {date_from}" + (f"–{date_to}" if date_to != date_from else "") if date_from else ""
        print(f"Нет сообщений{period} {'с ' + contact if contact else ''}.")
        return

    dates = [m["date"][:10] for m in msgs]
    mine_c = sum(1 for m in msgs if m["mine"])
    print(f"Анализирую {len(msgs)} сообщений [{min(dates)} — {max(dates)}]  моих: {mine_c}, их: {len(msgs)-mine_c}")

    answer = rag.answer(question, msgs)

    print("\n" + "═" * 65)
    print(answer)
    print("═" * 65)

    if date_from:
        print(f"\nСообщения за {date_from}:")
        for m in msgs:
            who = "Я" if m["mine"] else m["contact_name"]
            print(f"  [{m['date'][11:16]}] {who}: {m['text'][:100]}")
    else:
        print("\nПоследние 5 сообщений:")
        for m in msgs[-5:]:
            who = "Я" if m["mine"] else m["contact_name"]
            print(f"  [{m['date'][:10]} {m['date'][11:16]}] {who}: {m['text'][:100]}")
    print()


def run_search(params: dict):
    question = params.get("question") or params.get("query", "")
    contact = params.get("contact") or ""
    only_mine = params.get("only_mine", False)
    date_from = params.get("date_from")
    date_to = params.get("date_to")

    if contact:
        names = rag.list_matching_contacts(contact)
        if not names:
            print(f"Контакт '{contact}' не найден.")
            return
        print(f"Контакт: {', '.join(names)}")

    k = 12 if contact else 20

    results = rag.search(
        question,
        k=k,
        only_mine=only_mine,
        contact_filter=contact,
        min_score=0.3,
        date_from=date_from,
        date_to=date_to,
    )

    if len(results) < 3:
        keywords = params.get("search_term") or ""
        if not keywords:
            words = [w for w in question.split() if len(w) > 4]
            keywords = " ".join(words[:3])
        if keywords:
            _, text_matches = rag.count_and_find(keywords, contact, only_mine, date_from, date_to)
            existing = {(r["text"], r["date"]) for r in results}
            for m in text_matches[:10]:
                if (m["text"], m["date"]) not in existing:
                    results.append({**m, "score": 0.0})
                    existing.add((m["text"], m["date"]))

    if not results:
        print("Ничего не найдено.")
        return

    if not contact:
        by_contact: dict[str, list[dict]] = {}
        for r in results:
            by_contact.setdefault(r["contact_name"], []).append(r)
        contacts_found = sorted(by_contact.keys())
        print(f"Найдено в {len(by_contact)} диалог(ах): {', '.join(contacts_found)}\n")
    else:
        print(f"Найдено {len(results)} совпадений.\n")

    answer = answer_search(question, results)
    print("═" * 65)
    print(answer)
    print("═" * 65 + "\n")

    print(f"Источники ({len(results)} сообщений):\n")
    for i, r in enumerate(results, 1):
        who = "Я" if r["mine"] else r["contact_name"]
        to = f"→ {r['contact_name']}" if r["mine"] else "→ мне"
        score = r.get("score", 0)
        bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
        print(f"#{i}  [{r['date'][:10]}]  {who} {to}  [{bar}] {score:.2f}")
        print(f"    {r['text'][:200]}")
        print()


# ── Routing ───────────────────────────────────────────────────────────────────


def route(params: dict) -> str:
    intent = params.get("intent", "search")
    contact = params.get("contact") or ""
    if intent == "count":
        return "count"
    if intent == "analyze" and contact:
        return "analyze"
    return "search"


# ── CLI entry point ───────────────────────────────────────────────────────────


def main():
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return

    query = " ".join(args)

    rag.init(settings.openai_key)
    if rag.index_size() == 0:
        print("RAG индекс пуст. Запусти: python scripts/index.py")
        return

    print(f'🔍 "{query}"\n')
    print("Анализирую запрос...", end=" ", flush=True)

    try:
        params = parse_query(query)
    except Exception as e:
        print(f"Ошибка парсинга: {e}")
        return

    params["query"] = query
    intent = route(params)
    print(f"→ {params.get('plan', intent)}\n")

    if intent == "count":
        run_count(params)
    elif intent == "analyze":
        run_analyze(params)
    else:
        run_search(params)


if __name__ == "__main__":
    main()
