"""
RAG — векторный поиск по истории переписок.

Использование:
    rag.init(api_key)              # один раз при старте
    rag.search("деньги проект")    # -> list[dict]
    rag.index_size()               # -> int
"""

import pickle
import numpy as np
from pathlib import Path
from openai import OpenAI

RAG_FILE = Path("jarvis_rag.pkl")
EMBED_MODEL = "text-embedding-3-small"

_client: OpenAI | None = None


def init(api_key: str):
    global _client
    _client = OpenAI(api_key=api_key)


def _embed(texts: list[str]) -> np.ndarray:
    if _client is None:
        raise RuntimeError("Call rag.init(api_key) first")
    resp = _client.embeddings.create(model=EMBED_MODEL, input=texts)
    return np.array([e.embedding for e in resp.data], dtype=np.float32)


def build_index(docs: list[dict]) -> None:
    """
    docs: list of {text, contact_name, mine: bool, date: str}
    Эмбеддит батчами по 100 и сохраняет в RAG_FILE.
    """
    texts = [d["text"][:500] for d in docs]
    all_vecs = []
    batch = 100
    print(f"  Embedding {len(texts)} messages...", flush=True)
    for i in range(0, len(texts), batch):
        chunk = texts[i : i + batch]
        all_vecs.append(_embed(chunk))
        print(f"  {min(i + batch, len(texts))}/{len(texts)}", end="\r", flush=True)
    print()
    matrix = np.vstack(all_vecs).astype(np.float32)
    # L2-normalize for fast cosine via dot product
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix /= norms + 1e-8
    with open(RAG_FILE, "wb") as f:
        pickle.dump({"docs": docs, "vectors": matrix}, f)
    print(f"RAG index saved: {len(docs)} messages -> {RAG_FILE}")


def _load() -> tuple[list[dict], np.ndarray] | tuple[None, None]:
    if not RAG_FILE.exists():
        return None, None
    with open(RAG_FILE, "rb") as f:
        data = pickle.load(f)
    return data["docs"], data["vectors"]


def _contact_matches(contact_name: str, f: str) -> bool:
    """Умный матч: 'Лиза' не должна матчить 'Елизавета'."""
    cn = contact_name.lower().strip()
    f = f.lower().strip()
    if not f:
        return True
    if cn == f:
        return True
    # слово целиком
    if f in cn.split():
        return True
    # начинается с фильтра (имя + фамилия)
    if cn.startswith(f + " ") or cn.startswith(f):
        return True
    return False


def _date_ok(date_str: str, date_from: str | None, date_to: str | None) -> bool:
    d = date_str[:10]
    if date_from and d < date_from:
        return False
    if date_to and d > date_to:
        return False
    return True


def get_contact_messages(
    contact_filter: str = "",
    only_mine: bool = False,
    date_from: str | None = None,
    date_to: str | None = None,
    max_messages: int = 2000,
) -> list[dict]:
    """Возвращает ВСЕ сообщения с фильтрами по контакту, дате, отправителю."""
    docs, _ = _load()
    if docs is None:
        return []
    matched = []
    for d in docs:
        if contact_filter and not _contact_matches(d["contact_name"], contact_filter):
            continue
        if only_mine and not d.get("mine"):
            continue
        if not _date_ok(d["date"], date_from, date_to):
            continue
        matched.append(d)
    matched.sort(key=lambda x: x["date"])
    return matched[:max_messages]


def count_and_find(
    substring: str,
    contact_filter: str = "",
    only_mine: bool = False,
    date_from: str | None = None,
    date_to: str | None = None,
) -> tuple[int, list[dict]]:
    """Точный текстовый поиск без LLM — для 'сколько раз', 'найди все'."""
    docs, _ = _load()
    if docs is None:
        return 0, []
    s = substring.lower()
    results = []
    for d in docs:
        if contact_filter and not _contact_matches(d["contact_name"], contact_filter):
            continue
        if only_mine and not d.get("mine"):
            continue
        if not _date_ok(d["date"], date_from, date_to):
            continue
        if s in d["text"].lower():
            results.append(d)
    results.sort(key=lambda x: x["date"])
    return len(results), results


def list_matching_contacts(contact_filter: str) -> list[str]:
    """Список уникальных имён контактов, которые матчат фильтр."""
    docs, _ = _load()
    if docs is None:
        return []
    seen: set[str] = set()
    result = []
    for d in docs:
        cn = d["contact_name"]
        if _contact_matches(cn, contact_filter) and cn not in seen:
            seen.add(cn)
            result.append(cn)
    return result


def search(
    query: str,
    k: int = 6,
    only_mine: bool = False,
    contact_filter: str = "",
    min_score: float = 0.35,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[dict]:
    """Возвращает top-k наиболее релевантных сообщений."""
    if _client is None or not RAG_FILE.exists():
        return []

    docs, vectors = _load()
    if docs is None:
        return []

    # Optional filters
    mask = list(range(len(docs)))
    if only_mine:
        mask = [i for i in mask if docs[i].get("mine")]
    if contact_filter:
        mask = [i for i in mask if _contact_matches(docs[i]["contact_name"], contact_filter)]
    if date_from or date_to:
        mask = [i for i in mask if _date_ok(docs[i]["date"], date_from, date_to)]

    if not mask:
        return []

    filtered_docs = [docs[i] for i in mask]
    filtered_vecs = vectors[mask]

    q_vec = _embed([query])[0].astype(np.float32)
    q_vec /= np.linalg.norm(q_vec) + 1e-8

    sims = filtered_vecs @ q_vec
    top_idx = np.argsort(sims)[-k:][::-1]

    results = []
    for i in top_idx:
        score = float(sims[i])
        if score >= min_score:
            results.append({"score": score, **filtered_docs[i]})
    return results


def index_size() -> int:
    docs, _ = _load()
    return len(docs) if docs else 0


def get_max_date() -> str | None:
    """Returns the latest message date in the index (ISO string), or None."""
    docs, _ = _load()
    if not docs:
        return None
    dates = [d["date"] for d in docs if d.get("date")]
    return max(dates) if dates else None


def append_to_index(new_docs: list[dict]) -> None:
    """Embed only new_docs and merge into existing index without full re-embedding."""
    if not new_docs:
        print("Нет новых сообщений для добавления.")
        return

    docs, vectors = _load()
    texts = [d["text"][:500] for d in new_docs]
    all_vecs = []
    batch = 100
    print(f"  Embedding {len(texts)} new messages...", flush=True)
    for i in range(0, len(texts), batch):
        chunk = texts[i: i + batch]
        all_vecs.append(_embed(chunk))
        print(f"  {min(i + batch, len(texts))}/{len(texts)}", end="\r", flush=True)
    print()
    new_matrix = np.vstack(all_vecs).astype(np.float32)
    norms = np.linalg.norm(new_matrix, axis=1, keepdims=True)
    new_matrix /= norms + 1e-8

    if docs is None:
        merged_docs = new_docs
        merged_vecs = new_matrix
    else:
        merged_docs = docs + new_docs
        merged_vecs = np.vstack([vectors, new_matrix])

    with open(RAG_FILE, "wb") as f:
        pickle.dump({"docs": merged_docs, "vectors": merged_vecs}, f)
    old_count = len(docs) if docs else 0
    print(f"RAG index updated: {old_count} → {len(merged_docs)} messages")


def format_rag_context(results: list[dict], max_chars: int = 800) -> str:
    """Форматирует результаты поиска в строку для системного промпта."""
    if not results:
        return ""
    lines = []
    for r in results:
        who = "Я" if r["mine"] else r["contact_name"]
        date = r["date"][:10]
        lines.append(f"[{date}] {who}: {r['text'][:200]}")
    block = "\n".join(lines)
    return block[:max_chars]


def answer(query: str, messages: list[dict], model: str = "gpt-4o-mini") -> str:
    """
    Отвечает на вопрос пользователя на основе переданных сообщений.
    messages — либо результаты search(), либо get_contact_messages() (полный диалог).
    """
    if _client is None or not messages:
        return ""

    context_lines = []
    for m in messages:
        who = "Я" if m["mine"] else m["contact_name"]
        date = m["date"][:10]
        context_lines.append(f"[{date}] {who}: {m['text'][:300]}")

    context = "\n".join(context_lines)
    if len(context) > 20000:
        context = context[:20000] + "\n...(обрезано)"

    system = (
        "Ты — личный ассистент. Тебе дана история переписок пользователя с конкретным человеком.\n"
        "Отвечай на вопросы ТОЛЬКО на основе этих сообщений.\n"
        "Будь конкретным: считай упоминания встреч, дат, событий если тебя об этом спрашивают.\n"
        "Если данных не хватает — скажи что именно не хватает.\n"
        "Отвечай на русском языке."
    )
    user_msg = f"История переписки:\n{context}\n\nВопрос: {query}"

    resp = _client.chat.completions.create(
        model=model,
        max_tokens=600,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
    )
    return resp.choices[0].message.content.strip()
