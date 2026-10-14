from __future__ import annotations

from sentence_transformers import SentenceTransformer

from .config import Settings


class Embedder:
    def __init__(self, settings: Settings):
        self.prefix = settings.query_prefix
        self.model = SentenceTransformer(settings.embed_model)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(
            texts, batch_size=32, normalize_embeddings=True, show_progress_bar=len(texts) > 64
        ).tolist()

    def embed_query(self, query: str) -> list[float]:
        return self.model.encode(self.prefix + query, normalize_embeddings=True).tolist()