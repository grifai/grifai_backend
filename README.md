# JARVIS — AI-ассистент для мессенджеров

Автоматически генерирует ответы на сообщения в Telegram, WhatsApp, VK и Max, используя OpenAI/Claude + RAG по истории переписки.

---

## Быстрый старт

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
pip install maxapi          # только если нужен Max
```

### 2. Настройка `.env`

```bash
cp .env.example .env
```

Заполни нужные поля (минимум для Telegram):

```env
TG_API_ID=...
TG_API_HASH=...
OPENAI_KEY=sk-...
```

### 3. Индексация сообщений (RAG)

```bash
python index.py
```

Запускается один раз перед первым стартом. Индексирует историю переписки для смыслового поиска.

---

## Запуск

Каждая платформа запускается **отдельно** в своём терминале.

### Telegram

```bash
python main.py
```

### WhatsApp

Сначала запусти мост (в отдельном терминале):

```bash
node whatsapp_bridge/index.js
```

Затем бот:

```bash
python run_whatsapp.py
```

### VK

Требует `VK_TOKEN` в `.env` (пользовательский токен с правами `messages`).

> **Внимание:** автоответы от имени пользователя нарушают правила VK. Используй на свой риск.

```bash
python run_vk.py
```

### Max

Требует `MAX_BOT_TOKEN` в `.env`. Токен выдаёт **MasterBot** внутри приложения Max (аналог @BotFather).

```bash
python run_max.py
```

---

## Переменные окружения

| Переменная | Где взять | Обязательна |
|---|---|---|
| `TG_API_ID` / `TG_API_HASH` | [my.telegram.org](https://my.telegram.org) | Для Telegram |
| `OPENAI_KEY` | [platform.openai.com](https://platform.openai.com) | Да |
| `ANTHROPIC_KEY` | [console.anthropic.com](https://console.anthropic.com) | Нет |
| `VK_TOKEN` | [vk.com/dev](https://vk.com/dev) → токен пользователя | Для VK |
| `MAX_BOT_TOKEN` | MasterBot в приложении Max | Для Max |
| `MODEL` | — | Нет (по умолчанию `claude-haiku-4-5-20251001`) |

---

## Как работает одобрение ответов

При входящем сообщении JARVIS генерирует черновик и спрашивает:

```
Jarvis draft: "Привет, всё хорошо!"
-------------------------------------------------------
  1 Send   2 Edit   3 Skip
->
```

- `1` — отправить как есть
- `2` — ввести свой текст
- `3` — пропустить

---

## Структура проекта

```
main.py              ← запуск Telegram
run_whatsapp.py      ← запуск WhatsApp
run_vk.py            ← запуск VK
run_max.py           ← запуск Max

sources/
  whatsapp.py        ← WhatsApp интеграция
  vk.py              ← VK интеграция
  max.py             ← Max интеграция

app/                 ← FastAPI сервер (REST API)
scripts/             ← утилиты (индексация, статистика)
whatsapp_bridge/     ← Node.js мост для WhatsApp (Baileys)
```

---

## FastAPI сервер (опционально)

```bash
bash scripts/dev.sh
```

Запускает docker-compose (FastAPI + Postgres) на `http://localhost:8080`.
