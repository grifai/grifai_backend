import numpy as np
from openai import OpenAI


class EmbeddingProvider:

    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def embed(self, text: str) -> list[float]:
        """Эмбеддинг одного текста."""
        resp = self.client.embeddings.create(model=self.model, input=[text[:500]])
        return resp.data[0].embedding

    def embed_batch(self, texts: list[str], batch_size: int = 100) -> np.ndarray:
        """
        Эмбеддинг батча. Возвращает L2-нормализованную матрицу.
        Заменяет rag._embed() + нормализацию из build_index().
        """
        truncated = [t[:500] for t in texts]
        all_vecs = []
        for i in range(0, len(truncated), batch_size):
            chunk = truncated[i : i + batch_size]
            resp = self.client.embeddings.create(model=self.model, input=chunk)
            vecs = [e.embedding for e in resp.data]
            all_vecs.extend(vecs)
            print(
                f"  {min(i + batch_size, len(truncated))}/{len(truncated)}",
                end="\r",
                flush=True,
            )
        print()
        matrix = np.array(all_vecs, dtype=np.float32)
        return self.normalize(matrix)

    def embed_query(self, query: str) -> np.ndarray:
        """Эмбеддинг поискового запроса — нормализованный вектор."""
        vec = np.array(self.embed(query), dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    @staticmethod
    def normalize(matrix: np.ndarray) -> np.ndarray:
        """L2-нормализация для cosine similarity через dot product."""
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / (norms + 1e-8)
