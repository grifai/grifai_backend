#!/usr/bin/env python3
"""
Миграция: jarvis_rag.pkl -> Qdrant.

Использует уже готовые векторы из pickle — не перегенерирует эмбеддинги.

Запуск (из корня проекта):
    docker-compose up -d qdrant
    python scripts/migrate_rag.py
    python scripts/migrate_rag.py --dry-run  — только показать статистику
"""

import pickle
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    PayloadSchemaType,
)

from app.config import settings
from app.memory.rag import COLLECTION, VECTOR_DIM

PICKLE_FILE = Path("data/jarvis_rag.pkl")
BATCH_SIZE = 100


def migrate(dry_run: bool = False) -> None:
    if not PICKLE_FILE.exists():
        print(f"Файл {PICKLE_FILE} не найден. Нечего мигрировать.")
        sys.exit(1)

    print(f"Читаю {PICKLE_FILE}...")
    with open(PICKLE_FILE, "rb") as f:
        data = pickle.load(f)

    docs = data["docs"]
    vectors = data["vectors"]

    print(f"Найдено: {len(docs)} сообщений, вектора shape={vectors.shape}")

    if dry_run:
        from collections import Counter
        contacts = Counter(d["contact_name"] for d in docs)
        print(f"\nКонтакты ({len(contacts)}):")
        for name, cnt in contacts.most_common(10):
            print(f"  {name}: {cnt}")
        dates = [d["date"][:10] for d in docs]
        print(f"\nДиапазон дат: {min(dates)} — {max(dates)}")
        print("\n--dry-run: в Qdrant ничего не записано.")
        return

    client = QdrantClient(url=settings.qdrant_url)

    # Recreate collection
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION in existing:
        print(f"Коллекция '{COLLECTION}' уже существует. Удаляю...")
        client.delete_collection(COLLECTION)

    print(f"Создаю коллекцию '{COLLECTION}' (dim={VECTOR_DIM}, cosine)...")
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )
    for field, schema in [
        ("contact_name", PayloadSchemaType.KEYWORD),
        ("mine",         PayloadSchemaType.BOOL),
        ("date",         PayloadSchemaType.KEYWORD),
    ]:
        client.create_payload_index(
            collection_name=COLLECTION,
            field_name=field,
            field_schema=schema,
        )

    # Upload vectors in batches
    print(f"Загружаю {len(docs)} точек батчами по {BATCH_SIZE}...")
    uploaded = 0
    for i in range(0, len(docs), BATCH_SIZE):
        chunk_docs = docs[i : i + BATCH_SIZE]
        chunk_vecs = vectors[i : i + BATCH_SIZE]
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
        client.upsert(collection_name=COLLECTION, points=points)
        uploaded += len(points)
        print(f"  {uploaded}/{len(docs)}", end="\r", flush=True)
    print()

    # Verify
    info = client.get_collection(COLLECTION)
    count_in_qdrant = info.points_count or 0
    print(f"\nПроверка: pickle={len(docs)}, Qdrant={count_in_qdrant}", end=" ")
    if count_in_qdrant == len(docs):
        print("✓ совпадает")
    else:
        print(f"НЕСООТВЕТСТВИЕ! Разница: {abs(count_in_qdrant - len(docs))}")

    print(f"\nМиграция завершена. Qdrant: {count_in_qdrant} сообщений.")
    print("Теперь можно запустить: python -m app.services.search 'тест'")


if __name__ == "__main__":
    migrate(dry_run="--dry-run" in sys.argv)
