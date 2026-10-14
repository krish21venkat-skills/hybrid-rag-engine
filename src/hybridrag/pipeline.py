from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from .chunking import Chunker
from .config import Settings, load_settings
from .embeddings import Embedder
from .generation import Generator, make_generator
from .loaders import load_pages
from .logging_utils import timed
from .retrieval import Hit, Retriever
from .store import VectorStore

log = logging.getLogger(__name__)
NO_ANSWER = "I couldn't find relevant information in the documents."


@dataclass
class Answer:
    question: str
    text: str
    hits: list[Hit]
    latency_ms: float


class RAGPipeline:
    def __init__(self, settings: Settings | None = None, generator: Generator | None = None):
        self.s = settings or load_settings()
        self.chunker = Chunker(self.s)
        self.embedder = Embedder(self.s)
        self.store = VectorStore(self.s)
        self.retriever = Retriever(self.s, self.embedder, self.store)
        self.generator = generator or make_generator(self.s)

    def ingest(self, folder: str) -> int:
        unique, pages = {}, 0                          # dedupe by content hash
        for page in load_pages(folder):
            pages += 1
            for c in self.chunker.split(page):
                unique.setdefault(c.id, c)
        chunks = list(unique.values())
        if not chunks:
            log.warning("No chunks produced from %s", folder)
            return 0
        with timed(log, f"embed {len(chunks)} chunks"):
            embs = self.embedder.embed_documents([c.text for c in chunks])
        self.store.upsert(chunks, embs)
        self.retriever.refresh()
        log.info("Ingested pages=%d unique_chunks=%d total_in_store=%d",
                 pages, len(chunks), self.store.count())
        return len(chunks)

    def retrieve(self, query: str, k: int | None = None) -> list[Hit]:
        return self.retriever.search(query, k)

    def ask(self, question: str) -> Answer:
        t0 = time.perf_counter()
        hits = self.retrieve(question)
        text = self.generator.generate(question, hits) if hits else NO_ANSWER
        return Answer(question, text, hits, (time.perf_counter() - t0) * 1000)