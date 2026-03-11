from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

QDRANT_HOST = "qdrant"
QDRANT_PORT = 6333
COLLECTION = "user_memory"
VECTOR_SIZE = 384  # для all-MiniLM-L6-v2

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def create_collection():
    if COLLECTION not in [c.name for c in client.get_collections().collections]:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print(f"Qdrant collection '{COLLECTION}' created.")
    else:
        print(f"Qdrant collection '{COLLECTION}' already exists.")


if __name__ == "__main__":
    create_collection()
