"""Where does the right chunk rank for valid questions? Retrieval only, no LLM."""
import json
import sys
from collections import Counter
from dataclasses import replace

from hybridrag.config import load_settings
from hybridrag.pipeline import RAGPipeline

s = load_settings()
rag = RAGPipeline(replace(s, min_rerank_score=-1.0))
ranks, late = Counter(), []
for line in open(sys.argv[1], encoding="utf-8-sig"):
    if not line.strip():
        continue
    r = json.loads(line)
    if not r.get("source"):
        continue
    hits = rag.retrieve(r["question"], 5)
    pos = next((i for i, h in enumerate(hits, 1)
                if r["source"].lower() in str(h.metadata.get("source", "")).lower()
                and r["answer_contains"].lower() in h.text.lower()), None)
    ranks[pos] += 1
    if pos is None or pos > 3:
        late.append((pos, r["question"]))
print("rank -> count:", dict(sorted(ranks.items(), key=lambda x: (x[0] is None, x[0]))))
for pos, q in late:
    print(f"  rank {pos} | {q}")