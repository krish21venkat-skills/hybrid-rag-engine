from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from .config import Settings
from .embeddings import Embedder
from .logging_utils import timed
from .store import VectorStore

log = logging.getLogger(__name__)
_TOKEN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())   # strips punctuation so "error-code," still matches


@dataclass
class Hit:
    id: str
    text: str
    metadata: dict
    score: float


def rrf(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion: merges rankings without needing comparable scores."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)
    return fused


class Retriever:
    def __init__(self, settings: Settings, embedder: Embedder, store: VectorStore):
        self.s, self.embedder, self.store = settings, embedder, store
        self.reranker = CrossEncoder(settings.rerank_model)
        self._bm25: BM25Okapi | None = None
        self.refresh()

    def refresh(self) -> None:
        """(Re)build the BM25 index from the store. Call after ingest."""
        data = self.store.all()
        self._ids = data["ids"]
        self._lookup = {i: (d, m) for i, d, m in zip(data["ids"], data["documents"], data["metadatas"])}
        self._bm25 = BM25Okapi([tokenize(d) for d in data["documents"]]) if self._ids else None
        log.info("Retriever index built: %d chunks", len(self._ids))

    def search(self, query: str, k: int | None = None) -> list[Hit]:
        k = k or self.s.top_k
        if self._bm25 is None:
            return []

        with timed(log, "dense"):
            dense_ids = self.store.dense_search(self.embedder.embed_query(query), self.s.candidates)

        with timed(log, "bm25"):
            scores = self._bm25.get_scores(tokenize(query))
            top = sorted(range(len(scores)), key=lambda i: -scores[i])[: self.s.candidates]
            bm25_ids = [self._ids[i] for i in top if scores[i] > 0]

        fused = rrf([dense_ids, bm25_ids], self.s.rrf_k)
        cands = sorted(fused, key=fused.get, reverse=True)[: self.s.candidates]

        with timed(log, "rerank"):
            ce = self.reranker.predict([(query, self._lookup[c][0]) for c in cands])
        ranked = sorted(zip(ce, cands), key=lambda x: -x[0])[:k]
        hits = [Hit(c, *self._lookup[c], float(sc)) for sc, c in ranked if sc >= self.s.min_rerank_score]

        log.info("query=%r dense=%d bm25=%d fused=%d kept=%d top_score=%.3f",
                 query, len(dense_ids), len(bm25_ids), len(cands), len(hits),
                 hits[0].score if hits else 0.0)
        return hits