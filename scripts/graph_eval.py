"""End-to-end eval through the graph. Resumable (graph_eval.json); delete it after changing the pipeline.
python scripts\\graph_eval.py eval\\all.jsonl --every 3   # all out-of-scope + hard valid + every 3rd easy valid
python scripts\\graph_eval.py eval\\unseen2.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path

from hybridrag.graph import GraphRAG
from hybridrag.logging_utils import setup_logging

#CACHE = Path("graph_eval.json")
p = argparse.ArgumentParser()
p.add_argument("file")
p.add_argument("--every", type=int, default=1)
args = p.parse_args()

import os
CACHE = Path(f"graph_eval_{os.getenv('RAG_LLM_MODEL', 'default').replace(':', '_')}_{Path(args.file).stem}.json")

rows = [json.loads(l) for l in Path(args.file).read_text(encoding="utf-8").splitlines() if l.strip()]
top1 = {r["question"]: float(r["top1"]) for r in csv.DictReader(open("sweep.csv", encoding="utf-8"))}
if args.every > 1:
    kept, n = [], 0
    for r in rows:
        if r.get("source") and top1.get(r["question"], 0) >= 0.3:      # easy valid: sample
            n += 1
            if n % args.every:
                continue
        kept.append(r)
    rows = kept


def covers(expected: str, text: str) -> bool:
    et = re.findall(r"\w+", expected.lower())
    tt = set(re.findall(r"\w+", text.lower()))
    return not et or sum(t in tt for t in et) / len(et) >= 0.6


setup_logging("WARNING")
g = GraphRAG()
cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
st = {"valid_answered": 0, "valid_abstained": 0, "valid_rescued_by_retry": 0, "valid_check_numbers": 0,
      "oos_abstained": 0, "oos_leaked": 0, "oos_leaked_after_retry": 0, "blocked_by_quote_check": 0,
      "valid_blocked_by_fit": 0, "oos_caught_by_fit": 0}
t0 = time.time()

for n, r in enumerate(rows, 1):
    q, valid = r["question"], bool(r.get("source"))
    if q not in cache:
        out = g.ask(q)
        cache[q] = {k: out[k] for k in ("decision", "answer", "quote", "top1", "tries", "said",
                                        "quote_ok", "fit_blocked")}
        CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    o = cache[q]
    answered = o["decision"] == "answer"
    if o.get("fit_blocked"):
        st["valid_blocked_by_fit" if valid else "oos_caught_by_fit"] += 1
        print(f"FIT BLOCKED ({'valid' if valid else 'trap'}) | {q}")
    if not answered and o["said"] and not o["quote_ok"]:
        st["blocked_by_quote_check"] += 1
    if valid:
        if answered:
            st["valid_answered"] += 1
            st["valid_rescued_by_retry"] += o["tries"] > 0
            if not covers(r.get("answer_contains", ""), o["answer"] + " " + o["quote"]):
                st["valid_check_numbers"] += 1
                print(f"CHECK answer | {q}\n      answer: {o['answer'][:120]} | expected: {r.get('answer_contains')}")
        else:
            st["valid_abstained"] += 1
            print(f"BAD VALID abstained (tries={o['tries']} top1={o['top1']:.3f}) | {q}")
    else:
        if answered:
            st["oos_leaked"] += 1
            st["oos_leaked_after_retry"] += o["tries"] > 0
            print(f"BAD LEAK (tries={o['tries']}) | {q}\n      answer: {o['answer'][:120]}")
        else:
            st["oos_abstained"] += 1
    print(f"[{n}/{len(rows)}] {time.time() - t0:.0f}s", flush=True)

print(st)