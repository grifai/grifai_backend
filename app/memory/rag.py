"""
RAG — векторный поиск по истории переписок (Qdrant backend).

Использование:
    rag.init(api_key)              # один раз при старте
    rag.search("деньги проект")    # -> list[dict]
    rag.index_size()               # -> int
"""

import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, MatchAny, Range,
    PayloadSchemaType,
)

from app.llm.embeddings import EmbeddingProvider
from app.llm.prompts import SEARCH_ANSWER_PROMPT, ANALYZE_ANSWER_PROMPT

# Legacy path — kept for scripts/migrate_rag.py
RAG_FILE = Path("data/jarvis_rag.pkl")

VECTOR_DIM = 1536  # text-embedding-3-small
COLLECTION = "jarvis_messages"


# ── Contact matching helper ────────────────────────────────────────────────────

def _contact_matches(contact_name: str, f: str) -> bool:
    """Fuzzy match: 'Лиза' не матчит 'Елизавета'."""
    cn = contact_name.lower().strip()
    f = f.lower().strip()
    if not f:
        return True
    if cn == f:
        return True
    if f in cn.split():
        return True
    if cn.startswith(f + " ") or cn.startswith(f):
        return True
    return False


# ── VectorMemory class ────────────────────────────────────────────────────────

class VectorMemory:
    def __init__(self, qdrant_url: str, collection_name: str = COLLECTION):
        self.client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name
        self._embeddings: EmbeddingProvider | None = None
        self._llm = None
        self._contact_names_cache: list[str] | None = None

    def set_providers(self, embeddings: EmbeddingProvider, llm) -> None:
        self._embeddings = embeddings
        self._llm = llm

    def init_collection(self) -> None:
        """Create collection + payload indices if they don't exist."""
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name in existing:
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        for field, schema in [
            ("contact_name", PayloadSchemaType.KEYWORD),
            ("mine",         PayloadSchemaType.BOOL),
            ("date",         PayloadSchemaType.KEYWORD),
        ]:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field,
                field_schema=schema,
            )
        print(f"Qdrant коллекция '{self.collection_name}' создана.")

    # ── Contact name resolution ────────────────────────────────────────────────

    def _fetch_all_contact_names(self) -> list[str]:
        names: set[str] = set()
        offset = None
        while True:
            batch, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                with_payload=["contact_name"],
                with_vectors=False,
                limit=1000,
                offset=offset,
            )
            for point in batch:
                if point.payload:
                    names.add(point.payload.get("contact_name", ""))
            if next_offset is None:
                break
            offset = next_offset
        return list(names)

    def _resolve_contacts(self, contact_filter: str) -> list[str]:
        """Return exact contact names that fuzzy-match contact_filter."""
        if self._contact_names_cache is None:
            self._contact_names_cache = self._fetch_all_contact_names()
        return [cn for cn in self._contact_names_cache if _contact_matches(cn, contact_filter)]

    def _invalidate_cache(self) -> None:
        self._contact_names_cache = None

    # ── Filter builder ─────────────────────────────────────────────────────────

    def _make_filter(
        self,
        contact_filter: str = "",
        only_mine: bool = False,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> "Filter | None | str":
        """Build a Qdrant Filter. Returns 'NO_MATCH' if contact resolves to nothing."""
        conditions = []

        if contact_filter:
            exact_names = self._resolve_contacts(contact_filter)
            if not exact_names:
                return "NO_MATCH"
            conditions.append(
                FieldCondition(key="contact_name", match=MatchAny(any=exact_names))
            )

        if only_mine:
            conditions.append(FieldCondition(key="mine", match=MatchValue(value=True)))

        if date_from or date_to:
            range_kwargs: dict = {}
            if date_from:
                range_kwargs["gte"] = date_from
            if date_to:
                range_kwargs["lte"] = date_to
            conditions.append(FieldCondition(key="date", range=Range(**range_kwargs)))

        return Filter(must=conditions) if conditions else None

    # ── Scroll helper ──────────────────────────────────────────────────────────

    def _scroll_all(
        self,
        contact_filter: str = "",
        only_mine: bool = False,
        date_from: str | None = None,
        date_to: str | None = None,
        payload_fields: list[str] | None = None,
    ) -> list[dict]:
        f = self._make_filter(contact_filter, only_mine, date_from, date_to)
        if f == "NO_MATCH":
            return []

        results = []
        offset = None
        while True:
            batch, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                query_filter=f,
                with_payload=payload_fields if payload_fields else True,
                with_vectors=False,
                limit=1000,
                offset=offset,
            )
            for point in batch:
                if point.payload:
                    results.append(point.payload)
            if next_offset is None:
                break
            offset = next_offset
        return results

    # ── Write operations ───────────────────────────────────────────────────────

    def _upsert_batch(self, docs: list[dict], vectors) -> None:
        batch_size = 100
        for i in range(0, len(docs), batch_size):
            chunk_docs = docs[i : i + batch_size]
            chunk_vecs = vectors[i : i + batch_size]
            points = [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=chunk_vecs[j].tolist(),
                    payload={
                        "text": chunk_docs[j]["text"],
                        "contact_name": chunk_docs[j]["contact_name"],
                        "mine": bool(chunk_docs[j]["mine"]),
                        "date": chunk_docs[j]["date"],
                    },
                )
                for j in range(len(chunk_docs))
            ]
            self.client.upsert(collection_name=self.collection_name, points=points)
            print(f"  {min(i + batch_size, len(docs))}/{len(docs)} загружено", end="\r", flush=True)
        print()

    def build_index(self, docs: list[dict]) -> None:
        """Embed all docs and replace the collection."""
        if self._embeddings is None:
            raise RuntimeError("Call set_providers() first")
        self.client.delete_collection(self.collection_name)
        self.init_collection()
        self._invalidate_cache()

        texts = [d["text"][:500] for d in docs]
        print(f"  Embedding {len(texts)} messages...", flush=True)
        matrix = self._embeddings.embed_batch(texts)

        self._upsert_batch(docs, matrix)
        print(f"RAG index: {len(docs)} сообщений -> Qdrant '{self.collection_name}'")

    def append_to_index(self, new_docs: list[dict]) -> None:
        """Embed and append without removing existing data."""
        if not new_docs:
            print("Нет новых сообщений для добавления.")
            return
        if self._embeddings is None:
            raise RuntimeError("Call set_providers() first")

        old_count = self.index_size()
        texts = [d["text"][:500] for d in new_docs]
        print(f"  Embedding {len(texts)} new messages...", flush=True)
        matrix = self._embeddings.embed_batch(texts)
        self._invalidate_cache()
        self._upsert_batch(new_docs, matrix)
        print(f"RAG index updated: {old_count} → {old_count + len(new_docs)} messages")

    # ── Read operations ────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        k: int = 10,
        only_mine: bool = False,
        contact_filter: str = "",
        min_score: float = 0.3,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[dict]:
        if self._embeddings is None:
            return []
        q_vec = self._embeddings.embed_query(query).tolist()
        f = self._make_filter(contact_filter, only_mine, date_from, date_to)
        if f == "NO_MATCH":
            return []
        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=q_vec,
            query_filter=f,
            limit=k,
            score_threshold=min_score,
            with_payload=True,
        )
        return [{"score": hit.score, **hit.payload} for hit in hits]

    def count_and_find(
        self,
        substring: str,
        contact_filter: str = "",
        only_mine: bool = False,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> tuple[int, list[dict]]:
        """Qdrant scroll + Python substring filter (Qdrant has no full-text search)."""
        all_docs = self._scroll_all(contact_filter, only_mine, date_from, date_to)
        s = substring.lower()
        matches = [d for d in all_docs if s in d.get("text", "").lower()]
        matches.sort(key=lambda x: x.get("date", ""))
        return len(matches), matches

    def get_contact_messages(
        self,
        contact_filter: str = "",
        only_mine: bool = False,
        date_from: str | None = None,
        date_to: str | None = None,
        max_messages: int = 2000,
    ) -> list[dict]:
        all_docs = self._scroll_all(contact_filter, only_mine, date_from, date_to)
        all_docs.sort(key=lambda x: x.get("date", ""))
        return all_docs[:max_messages]

    def list_matching_contacts(self, contact_filter: str) -> list[str]:
        return self._resolve_contacts(contact_filter)

    def index_size(self) -> int:
        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count or 0
        except Exception:
            return 0

    def get_max_date(self) -> str | None:
        """Scroll only the date field and return the maximum."""
        dates = []
        offset = None
        while True:
            batch, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                with_payload=["date"],
                with_vectors=False,
                limit=1000,
                offset=offset,
            )
            for point in batch:
                if point.payload and point.payload.get("date"):
                    dates.append(point.payload["date"])
            if next_offset is None:
                break
            offset = next_offset
        return max(dates) if dates else None

    def answer(self, query: str, messages: list[dict]) -> str:
        if self._llm is None or not messages:
            return ""
        for m in messages:
            if "score" not in m:
                m["score"] = 1.0
        context = format_search_results_for_llm(messages, min_score=0.0)
        if len(context) > 20000:
            context = context[:20000] + "\n...(обрезано)"
        contacts = {m.get("contact_name", "") for m in messages}
        if len(contacts) == 1:
            system = ANALYZE_ANSWER_PROMPT.format(contact_name=contacts.pop())
        else:
            system = SEARCH_ANSWER_PROMPT
        return self._llm.generate(
            system_prompt=system,
            user_message=f"Результаты из переписки:\n\n{context}\n\nВопрос: {query}",
            temperature=0.3,
            max_tokens=600,
        )


# ── Formatting helpers ─────────────────────────────────────────────────────────

def format_rag_context(results: list[dict], max_chars: int = 800) -> str:
    if not results:
        return ""
    lines = []
    for r in results:
        who = "Я" if r.get("mine") else r.get("contact_name", "?")
        date = r.get("date", "")[:10]
        lines.append(f"[{date}] {who}: {r.get('text', '')[:200]}")
    return "\n".join(lines)[:max_chars]


def format_search_results_for_llm(results: list[dict], min_score: float = 0.43) -> str:
    lines = []
    for r in results:
        score = r.get("score", 0)
        if score < min_score:
            continue
        contact = r.get("contact_name", "?")
        date = r.get("date", "")[:10]
        text = r.get("text", "")[:300]
        direction = f"Я → {contact}" if r.get("mine") else f"{contact} → Мне"
        lines.append(f"[{date}] {direction} (релевантность {score:.0%}): {text}")
    return "\n".join(lines)


# ── Module-level backward-compatible API ───────────────────────────────────────

_vm: VectorMemory | None = None


def init(api_key: str = "", qdrant_url: str = "") -> None:
    global _vm
    from app.llm import get_llm, get_embeddings
    from app.config import settings

    url = qdrant_url or settings.qdrant_url
    _vm = VectorMemory(qdrant_url=url)
    _vm.init_collection()

    if api_key:
        emb = get_embeddings(api_key=api_key)
        llm = get_llm(api_key=api_key)
    else:
        emb = get_embeddings()
        llm = get_llm()
    _vm.set_providers(emb, llm)


def _get() -> VectorMemory:
    if _vm is None:
        raise RuntimeError("Call rag.init() first")
    return _vm


def build_index(docs: list[dict]) -> None:
    _get().build_index(docs)


def append_to_index(new_docs: list[dict]) -> None:
    _get().append_to_index(new_docs)


def search(
    query: str,
    k: int = 6,
    only_mine: bool = False,
    contact_filter: str = "",
    min_score: float = 0.35,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[dict]:
    if _vm is None:
        return []
    return _vm.search(
        query, k=k, only_mine=only_mine, contact_filter=contact_filter,
        min_score=min_score, date_from=date_from, date_to=date_to,
    )


def count_and_find(
    substring: str,
    contact_filter: str = "",
    only_mine: bool = False,
    date_from: str | None = None,
    date_to: str | None = None,
) -> tuple[int, list[dict]]:
    return _get().count_and_find(substring, contact_filter, only_mine, date_from, date_to)


def get_contact_messages(
    contact_filter: str = "",
    only_mine: bool = False,
    date_from: str | None = None,
    date_to: str | None = None,
    max_messages: int = 2000,
) -> list[dict]:
    return _get().get_contact_messages(contact_filter, only_mine, date_from, date_to, max_messages)


def list_matching_contacts(contact_filter: str) -> list[str]:
    return _get().list_matching_contacts(contact_filter)


def index_size() -> int:
    if _vm is None:
        return 0
    return _vm.index_size()


def get_max_date() -> str | None:
    return _get().get_max_date()


def answer(query: str, messages: list[dict]) -> str:
    return _get().answer(query, messages)
