from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from .pipeline import RAGPipeline

log = logging.getLogger(__name__)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).lower()


def _rank(hits, row) -> int | None:
    """1-based rank of the first chunk from the right source (and containing the answer, if given)."""
    needle = row.get("answer_contains")
    for i, h in enumerate(hits, 1):
        if h.metadata.get("source") != row["source"]:
            continue
        if not needle or _norm(needle) in _norm(h.text):
            return i
    return None


def evaluate(pipeline: RAGPipeline, path: str, k: int = 5) -> dict:
    """JSONL: {"question", "source", "answer_contains"?}. source=null marks an out-of-scope question."""
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    pos = [r for r in rows if r.get("source")]
    neg = [r for r in rows if not r.get("source")]

    found, rr = 0, 0.0
    for r in pos:
        hits = pipeline.retrieve(r["question"], k)
        rank = _rank(hits, r)
        if rank:
            found += 1
            rr += 1.0 / rank
        else:
            log.warning("MISS %r | top=%s", r["question"],
                        [(h.metadata.get("source"), round(h.score, 2)) for h in hits[:3]])

    abstained = 0
    for r in neg:
        hits = pipeline.retrieve(r["question"], k)
        if not hits:
            abstained += 1
        else:
            log.warning("NOT ABSTAINED %r | top=%s %.2f", r["question"],
                        hits[0].metadata.get("source"), hits[0].score)

    return {
        "answerable": len(pos), f"hit@{k}": round(found / max(len(pos), 1), 3),
        "mrr": round(rr / max(len(pos), 1), 3),
        "out_of_scope": len(neg), "abstain_rate": round(abstained / max(len(neg), 1), 3),
    }