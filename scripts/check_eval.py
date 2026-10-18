"""Check an eval file against the corpus (no models needed).
python scripts\\check_eval.py eval\\frozen.jsonl            # label, duplicate and type checks
python scripts\\check_eval.py eval\\frozen.jsonl --freeze   # record a hash; later runs report if the file changed
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

DOCS = Path("data/docs")
TYPES = {"trap", "offdomain", "messy"}
STOP = set("a an the is are was were be do does did can could i we you my our your of to in on at for from by "
           "with and or if it its this that what when where which who how many much long often there their any "
           "about into than then so as not no get got us".split())


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).lower().strip()


def words(s: str) -> set[str]:
    return {w.rstrip("s") for w in re.findall(r"\w+", s.lower()) if w not in STOP and len(w) > 2}


def load(p) -> list[dict]:
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8-sig").splitlines() if l.strip()]


path = sys.argv[1]
rows = load(path)
docs = {f.name: f.read_text(encoding="utf-8", errors="ignore")
        for f in DOCS.rglob("*") if f.suffix.lower() in {".txt", ".md"}}
sentences = [(n, s) for n, t in docs.items() for s in re.split(r"(?<=[.!?])\s+|\n+", t) if len(s) > 20]

errors = warns = 0
types: dict = {}
per_src: dict = {}
for r in rows:
    q = r["question"]
    if r.get("source"):
        per_src[r["source"]] = per_src.get(r["source"], 0) + 1
        text = docs.get(r["source"])
        if text is None:
            print("LABEL ERROR (no such file):", q)
            errors += 1
        elif not r.get("answer_contains") or norm(r["answer_contains"]) not in norm(text):
            print("LABEL ERROR (answer_contains is not in the document):", q)
            errors += 1
    else:
        t = r.get("type")
        types[t] = types.get(t, 0) + 1
        if t not in TYPES:
            print("WARN missing or unknown type:", q)
            warns += 1
        qw = words(q)
        if qw:
            name, sent = max(sentences, key=lambda ns: len(qw & words(ns[1])))
            ratio = len(qw & words(sent)) / len(qw)
            if ratio >= 0.7:
                print(f"REVIEW possibly answerable ({ratio:.0%} overlap with {name}): {q}")
                warns += 1

seen = {}
for f in Path("eval").glob("*.jsonl"):
    if f.resolve() != Path(path).resolve():
        for r in load(f):
            seen[norm(r["question"])] = f.name
for r in rows:
    if norm(r["question"]) in seen:
        print(f"DUPLICATE of {seen[norm(r['question'])]}: {r['question']}")
        errors += 1

print(f"\n{len(rows)} rows | answerable {sum(1 for r in rows if r.get('source'))} per source {per_src}"
      f" | out-of-scope by type {types}")
h = hashlib.sha256(Path(path).read_bytes()).hexdigest()
hf = Path("reports") / (Path(path).stem + ".sha256")
if "--freeze" in sys.argv:
    hf.parent.mkdir(exist_ok=True)
    hf.write_text(h)
    print("frozen:", hf)
elif hf.exists():
    print("FROZEN FILE", "OK" if hf.read_text().strip() == h else "CHANGED SINCE FREEZE")
print(f"errors {errors}, warnings {warns}")
sys.exit(1 if errors else 0)