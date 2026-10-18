"""Judge test over sweep.csv questions above the abstain cut-off. Retrieved chunks are cached
in contexts.json, so prompt changes re-run only the LLM calls.
python scripts\\judge_test.py            # 'rule' prompt
python scripts\\judge_test.py strict     # original strict prompt, to compare
"""
from __future__ import annotations

import csv
import json
import re
import sys
from dataclasses import replace
from pathlib import Path

import ollama

from hybridrag.config import load_settings

MODE = sys.argv[1] if len(sys.argv) > 1 else "rule"
LOW = 0.01
CACHE = Path("contexts.json")

PROMPTS = {
    "strict": """You check whether a context states the answer to a question.
Reply with JSON only: {{"answer_found": true or false, "quote": "<sentence copied from the context, or empty>"}}
Rules:
- answer_found is true only if the context explicitly states the specific fact asked for.
- If the context only mentions the same topic but not the specific fact, answer_found is false.
- The quote must be copied word for word from the context.

Context:
{context}

Question: {question}
JSON:""",
    "rule": """You decide whether a context answers a question.
Reply with JSON only: {{"answer_found": true or false, "quote": "<one sentence copied from the context, or empty>"}}
Rules:
- answer_found is true if the context contains a rule, number or statement that directly answers the question, even when the question words it differently. Example: a question about a "huge delay" can be answered by a rule about "delays longer than 2 hours".
- answer_found is false if the context only covers the same topic, or names the subject, but does not give the specific thing asked. Example: a context that says a feature exists does not answer a question about its price.
- The quote must be the single sentence that supports your answer, copied word for word from the context.

Context:
{context}

Question: {question}
JSON:""",
}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).lower().strip()


rows = [r for r in csv.DictReader(open("sweep.csv", encoding="utf-8")) if float(r["top1"]) >= LOW]
settings = load_settings()

cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
missing = [r["question"] for r in rows if r["question"] not in cache]
if missing:
    from hybridrag.pipeline import RAGPipeline
    rag = RAGPipeline(replace(settings, min_rerank_score=-1.0))      # gate off: top 5 chunks
    for q in missing:
        cache[q] = [h.text for h in rag.retrieve(q, 5)]
    CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

stats = {"valid_yes": 0, "valid_no": 0, "oos_yes": 0, "oos_no": 0}
for r in rows:
    q, valid = r["question"], r["answerable"] == "True"
    ctx = "\n\n".join(cache[q])
    resp = ollama.chat(
        model=settings.llm_model, format="json",
        options={"temperature": 0, "num_ctx": 4096},
        messages=[{"role": "user", "content": PROMPTS[MODE].format(context=ctx, question=q)}],
    )
    raw = resp["message"]["content"]
    try:
        out = json.loads(raw)
    except json.JSONDecodeError:
        out = {}
    quote = str(out.get("quote", ""))
    said = bool(out.get("answer_found"))
    verified = said and len(quote) > 10 and norm(quote) in norm(ctx)
    stats[("valid_" if valid else "oos_") + ("yes" if verified else "no")] += 1
    if verified != valid:
        why = "false accept" if verified else ("quote not verbatim" if said else "judge said no")
        print(f"BAD {float(r['top1']):>7.4f} {'VALID' if valid else 'oos  '} ({why}) | {q}")
        print(f"      raw: {raw[:160]}")

print(f"mode={MODE}", stats)