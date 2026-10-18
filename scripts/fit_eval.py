"""Test answer-fit checks on (question, quote, answer) pairs.
python scripts\\fit_test.py --prompt lenient
python scripts\\fit_test.py --prompt extract
python scripts\\fit_test.py --prompt extract --model qwen2.5:7b
Options: --prompt strict|lenient|extract (default strict), --model <ollama model>, --quick (final.jsonl only)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import ollama

from hybridrag.config import load_settings

PAIRS = Path("pairs.json")
SETS = ["eval/final.jsonl"] if "--quick" in sys.argv else ["eval/final.jsonl", "eval/fresh.jsonl"]
TIME = "second|minute|hour|day|week|month|year"
UNITS = ["days", "weeks", "months", "years", "hours", "minutes", "characters",
         "connections", "requests", "orders", "times", "attempts", "items"]
FREQ = re.compile(r"\b(every|each|daily|weekly|monthly|yearly|annual\w*|twice|once|per|hourly|times|"
                  r"a (?:year|month|week|day))\b", re.I)

FIT = """Does the sentence directly state the specific thing the question asks for, or is it only about something related?
Reply with JSON only: {{"direct": true or false}}

Question: What is the delivery fee?
Sentence: Delivery is available on weekdays.
JSON: {{"direct": false}}

Question: What is the return window?
Sentence: Items can be returned within 14 days of purchase.
JSON: {{"direct": true}}

Question: {question}
Sentence: {quote}
JSON:"""

LENIENT = """Question: {question}
Sentence: {quote}
Does this sentence give the information the question asks for? Reply with JSON only: {{"direct": true or false}}"""

EXTRACT = """Answer the question using only the sentence, in 8 words or fewer. If the sentence does not give the specific thing the question asks for, reply NONE.

Question: {question}
Sentence: {quote}
Answer:"""

ARGS = sys.argv[1:]
MODEL = ARGS[ARGS.index("--model") + 1] if "--model" in ARGS else load_settings().llm_model
PROMPT = ARGS[ARGS.index("--prompt") + 1] if "--prompt" in ARGS else "strict"
KEY = f"fit_{PROMPT}_{MODEL}"


def rows(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def unit_ok(question: str, text: str) -> tuple[bool, str]:
    """No-LLM check: does the answer use the unit or kind of fact the question asks for?"""
    q, t = question.lower(), text.lower()
    if re.search(r"\bhow many\b", q):
        found = [(q.find(u), u) for u in UNITS if re.search(rf"\b{u}\b", q)]
        if found:
            u = min(found)[1]
            if not re.search(rf"\b{u[:-1]}s?\b", t):
                return False, f"expects '{u}'"
    if re.search(r"\bhow (long|soon|quickly|early)\b", q) and not re.search(rf"\b({TIME})s?\b", t):
        return False, "expects a time unit"
    if re.search(r"\bhow often\b", q) and not FREQ.search(t):
        return False, "expects a frequency"
    return True, ""


def fit_ok(question: str, quote: str, model: str, prompt: str) -> bool:
    """Narrow LLM check: sees only the question and the one quoted sentence."""
    if prompt == "extract":
        r = ollama.chat(model=model, options={"temperature": 0, "num_predict": 24},
                        messages=[{"role": "user", "content": EXTRACT.format(question=question, quote=quote)}])
        return not r["message"]["content"].strip().upper().startswith("NONE")
    tmpl = FIT if prompt == "strict" else LENIENT
    r = ollama.chat(model=model, format="json", options={"temperature": 0, "num_predict": 20},
                    messages=[{"role": "user", "content": tmpl.format(question=question, quote=quote)}])
    try:
        return bool(json.loads(r["message"]["content"]).get("direct"))
    except json.JSONDecodeError:
        return True        # unreadable output: do not block


def save(pairs: dict) -> None:
    PAIRS.write_text(json.dumps(pairs, ensure_ascii=False), encoding="utf-8")


def collect() -> dict:
    pairs = json.loads(PAIRS.read_text(encoding="utf-8")) if PAIRS.exists() else {}
    gp = Path("graph_eval.json")                      # free: answers saved by the last graph_eval run
    if gp.exists():
        labels = {r["question"]: bool(r.get("source")) for r in rows("eval/all.jsonl")}
        for q, o in json.loads(gp.read_text(encoding="utf-8")).items():
            if q in labels and q not in pairs:
                pairs[q] = {"valid": labels[q], "answered": o["decision"] == "answer",
                            "answer": o["answer"], "quote": o["quote"]}
    todo = [r for f in SETS for r in rows(f) if r["question"] not in pairs]
    if todo:
        from hybridrag.graph import GraphRAG
        g = GraphRAG()
        for n, r in enumerate(todo, 1):
            o = g.ask(r["question"])
            pairs[r["question"]] = {"valid": bool(r.get("source")), "answered": o["decision"] == "answer",
                                    "answer": o["answer"], "quote": o["quote"]}
            save(pairs)
            print(f"collect {n}/{len(todo)}", flush=True)
    return pairs


if __name__ == "__main__":
    print(f"prompt={PROMPT} model={MODEL}")
    pairs = collect()
    answered = {q: p for q, p in pairs.items() if p["answered"]}
    res = []
    for n, (q, p) in enumerate(answered.items(), 1):
        u, why = unit_ok(q, p["answer"] + " " + p["quote"])
        if KEY not in p:
            p[KEY] = fit_ok(q, p["quote"], MODEL, PROMPT)
            save(pairs)
            print(f"fit {n}/{len(answered)}", flush=True)
        res.append((q, p["valid"], u, p[KEY], why))

    def count(valid: bool, blocks) -> int:
        return sum(1 for r in res if r[1] == valid and blocks(r))

    leaks, valid = count(False, lambda r: True), count(True, lambda r: True)
    print(f"\nanswered pairs: {valid} valid, {leaks} leaks")
    for name, blocks in (("unit check", lambda r: not r[2]), ("fit check", lambda r: not r[3]),
                         ("both", lambda r: not r[2] or not r[3])):
        print(f"{name:10} leaks caught {count(False, blocks)}/{leaks} | valid refused {count(True, blocks)}/{valid}")
    for q, v, u, f, why in res:
        if not v and u and f:
            print("STILL LEAKS  :", q)
        if v and not (u and f):
            print("VALID REFUSED:", f"unit ({why})" if not u else "fit", "|", q)