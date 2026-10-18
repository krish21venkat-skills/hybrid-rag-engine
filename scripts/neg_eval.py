"""Quick check of the yes/no-negation rule through the real graph. Resumable.
python scripts\\neg_test.py              # current graph.py (new rule)
python scripts\\neg_test.py --baseline   # same graph with the new rule removed
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import hybridrag.graph as G
from hybridrag.logging_utils import setup_logging

BASE = "--baseline" in sys.argv
if BASE:
    G.PROMPT = "\n".join(l for l in G.PROMPT.splitlines() if not l.startswith("- If the question is a yes/no"))

CASES = [
    ("neg", "Can unused sick leave be cashed out?"),
    ("neg", "Can I carry forward my sick leave to next year?"),
    ("pos", "How many vacation days do staff get each year?"),
    ("pos", "What is the hotel limit for contractors?"),
    ("trap", "What type of battery does the Orion meter use?"),
    ("trap", "How many days of marriage leave are allowed?"),
    ("trap", "How often are firmware updates pushed to the meters?"),
    ("trap", "What is the delay between payment gateway retry attempts?"),
    ("trap", "What time does the Tuesday release finish?"),
    ("trap", "Can contractors claim overtime pay?"),
    ("trap", "Do contractors get health insurance?"),
    ("trap", "Is remote work allowed for interns?"),
]

CACHE = Path("neg_base.json" if BASE else "neg_new.json")
setup_logging("WARNING")
cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
g = G.GraphRAG()
answered = {"neg": 0, "pos": 0, "trap": 0}
total = {"neg": 0, "pos": 0, "trap": 0}

for n, (kind, q) in enumerate(CASES, 1):
    if q not in cache:
        o = g.ask(q)
        cache[q] = {"decision": o["decision"], "answer": o["answer"], "fit_blocked": o["fit_blocked"]}
        CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    o = cache[q]
    ans = o["decision"] == "answer"
    total[kind] += 1
    answered[kind] += ans
    print(f"[{n}/{len(CASES)}] {kind:4} {'ANSWERED' if ans else 'abstain '}"
          f"{' (fit-blocked)' if o['fit_blocked'] else ''} | {q}", flush=True)
    if ans:
        print(f"      -> {o['answer'][:90]}")

print(f"mode={'baseline' if BASE else 'new rule'} | negation answered {answered['neg']}/{total['neg']}"
      f" | positives answered {answered['pos']}/{total['pos']} | traps answered (leaks) {answered['trap']}/{total['trap']}")