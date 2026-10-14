from __future__ import annotations

import chromadb

from .chunking import Chunk
from .config import Settings

_BATCH = 500


class VectorStore:
    def __init__(self, settings: Settings):
        client = chromadb.PersistentClient(str(settings.db_path))
        self.col = client.get_or_create_collection(
            settings.collection, metadata={"hnsw:space": "cosine"}
        )

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        for i in range(0, len(chunks), _BATCH):
            c, e = chunks[i:i + _BATCH], embeddings[i:i + _BATCH]
            self.col.upsert(
                ids=[x.id for x in c],
                documents=[x.text for x in c],
                metadatas=[x.metadata for x in c],
                embeddings=e,
            )

    def dense_search(self, query_emb: list[float], n: int) -> list[str]:
        total = self.col.count()
        if total == 0:
            return []
        return self.col.query(query_embeddings=[query_emb], n_results=min(n, total))["ids"][0]

    def all(self) -> dict:
        return self.col.get()   # fine for small/medium corpora; page it for large ones

    def count(self) -> int:
        return self.col.count()