#!/usr/bin/env python3
"""
Jarvis — dashboard статистики.

Использование:
  python stats.py            — всё
  python stats.py contacts   — только контакты
  python stats.py stats      — только статистика по действиям
  python stats.py recent     — последние 20 взаимодействий
  python stats.py search Маша — найти контакт
"""

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

MEMORY_FILE = Path("jarvis_memory.json")
W = 95


def load() -> dict:
    return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))


def _age(iso: str) -> str:
    h = (datetime.now() - datetime.fromisoformat(iso)).total_seconds() / 3600
    if h < 1:
        return f"{int(h*60)}m"
    if h < 48:
        return f"{h:.0f}h"
    return f"{h/24:.1f}d"


def show_contacts(data: dict, filter_name: str = ""):
    contacts = data.get("contacts", {})
    now = datetime.now()

    # count approved/revised per sender name
    sent_by: dict[str, int] = defaultdict(int)
    for e in data.get("examples", []):
        if e.get("action") in ("approved", "revised"):
            sent_by[e["sender"]] += 1

    print(f"\n{'='*W}")
    print(f"  КОНТАКТЫ  ({len(contacts)} профилей)")
    print(f"{'='*W}")
    print(f"  {'Имя':<26} {'AI':<4}  {'Отношения':<32} {'Возраст':>7}  {'Отправлено':>10}")
    print(f"  {'-'*(W-4)}")

    shown = 0
    for cid, c in contacts.items():
        name = c["name"]
        if filter_name and filter_name.lower() not in name.lower():
            continue

        p = c.get("profile", {})
        rel = ""
        if isinstance(p, dict) and not p.get("parse_error"):
            rel = (p.get("relationship") or "")[:30]

        age = _age(c["updated"])
        expired = " !" if (datetime.now() - datetime.fromisoformat(c["updated"])).total_seconds() > 172800 else ""
        mode = c.get("ai_mode", "auto")
        ai = "ON " if mode == "auto" else "OFF"
        sent = sent_by.get(name, 0)

        print(f"  {name:<26} {ai:<4}  {rel:<32} {age+expired:>7}  {sent:>10}")
        shown += 1

    if shown == 0:
        print(f"  (ничего не найдено для '{filter_name}')")
    print(f"\n  ! = профиль протух (>48ч), пересканируется при следующем запуске")


def show_stats(data: dict):
    stats = data.get("stats", {})
    examples = data.get("examples", [])
    contacts = data.get("contacts", {})

    ai_off = sum(1 for c in contacts.values() if c.get("ai_mode") == "never")

    print(f"\n{'='*W}")
    print(f"  ОБЩАЯ СТАТИСТИКА")
    print(f"{'='*W}")
    print(f"  Профилей контактов : {len(contacts)}  (AI выключен для {ai_off})")
    print(f"  Мой стиль-профиль  : {'построен' if data.get('my_profile') else 'не построен'}")
    print(f"  Примеров сохранено : {len(examples)}")
    total = sum(stats.get(k, 0) for k in ("approved", "revised", "skipped"))
    if total:
        apr = stats.get("approved", 0)
        rev = stats.get("revised", 0)
        skp = stats.get("skipped", 0)
        print(f"  Одобрено           : {apr}  ({apr/total*100:.0f}%)")
        print(f"  Отредактировано    : {rev}  ({rev/total*100:.0f}%)")
        print(f"  Пропущено          : {skp}  ({skp/total*100:.0f}%)")

    # per-sender table
    sender_stats: dict[str, dict] = defaultdict(lambda: {"approved": 0, "revised": 0, "skipped": 0})
    for e in examples:
        action = e.get("action", "skipped")
        sender_stats[e["sender"]][action] = sender_stats[e["sender"]].get(action, 0) + 1

    if sender_stats:
        print(f"\n  {'Контакт':<28} {'Одобрено':>9} {'Ред.':>6} {'Пропуск':>8}")
        print(f"  {'-'*53}")
        for name, s in sorted(sender_stats.items(),
                               key=lambda x: -(x[1].get("approved", 0) + x[1].get("revised", 0))):
            print(f"  {name:<28} {s.get('approved',0):>9} {s.get('revised',0):>6} {s.get('skipped',0):>8}")


def show_recent(data: dict, n: int = 20):
    examples = data.get("examples", [])
    recent = list(reversed(examples[-n:]))

    print(f"\n{'='*W}")
    print(f"  ПОСЛЕДНИЕ {n} ВЗАИМОДЕЙСТВИЙ")
    print(f"{'='*W}")

    if not recent:
        print("  (нет данных)")
        return

    for e in recent:
        ts = e["ts"][:16].replace("T", " ")
        action = {"approved": "ОТПРАВЛЕНО", "revised": "ОТРЕДАКТИРОВАНО",
                  "skipped": "ПРОПУЩЕНО"}.get(e.get("action", ""), e.get("action", "").upper())
        print(f"  [{ts}] {e['sender']:<22} {action}")
        print(f"    >> {e['incoming'][:80]}")
        reply = e.get("final_reply") or (e.get("llm_draft", "") if e.get("action") != "skipped" else "")
        if reply:
            print(f"    <- {reply[:80]}")
        print()


def show_profile(data: dict, name: str):
    for c in data.get("contacts", {}).values():
        if name.lower() in c["name"].lower():
            print(f"\n  Профиль: {c['name']}")
            print(f"  AI режим: {'ON' if c.get('ai_mode','auto') == 'auto' else 'OFF'}")
            print(f"  Обновлён: {_age(c['updated'])} назад")
            p = c.get("profile", {})
            if isinstance(p, dict):
                print(json.dumps(p, ensure_ascii=False, indent=2))
            return
    print(f"  Контакт '{name}' не найден")


if __name__ == "__main__":
    if not MEMORY_FILE.exists():
        print("jarvis_memory.json не найден. Запустите main.py сначала.")
        sys.exit(1)

    data = load()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    arg = sys.argv[2] if len(sys.argv) > 2 else ""

    if cmd == "contacts":
        show_contacts(data)
    elif cmd == "stats":
        show_stats(data)
    elif cmd == "recent":
        show_recent(data, int(arg) if arg.isdigit() else 20)
    elif cmd == "search":
        show_contacts(data, filter_name=arg)
    elif cmd == "profile":
        show_profile(data, arg)
    else:  # all
        show_contacts(data)
        show_stats(data)
        show_recent(data, 10)
        print()
