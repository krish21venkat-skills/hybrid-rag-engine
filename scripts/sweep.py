"""Sweep gating signals over the eval set.

python scripts\\sweep.py eval\\all.jsonl    # retrieve once (slow), save sweep.csv, report
python scripts\\sweep.py --cached           # re-report from sweep.csv (instant)
"""
from __future__ import annotations

import csv
import itertools
import json
import sys
import zlib
from dataclasses import replace
from pathlib import Path

from hybridrag.config import load_settings
from hybridrag.evaluate import _rank
from hybridrag.pipeline import RAGPipeline
from hybridrag.retrieval import tokenize

CSV = Path("sweep.csv")
STOP = set(
    "a an the is are was were be do does did can could i we you my our your of to in on at for "
    "from by with and or if it its this that what when where which who how many much long often "
    "there their any about into than then so as not no get got us".split()
)


def terms(text: str) -> set[str]:
    return {t.rstrip("s") for t in tokenize(text) if t not in STOP and len(t) > 2}


def collect(files: list[str]) -> list[dict]:
    rag = RAGPipeline(replace(load_settings(), min_rerank_score=-1.0))   # threshold off

    df: dict[str, int] = {}                       # corpus vocabulary
    for text, _ in rag.retriever._lookup.values():
        for t in terms(text):
            df[t] = df.get(t, 0) + 1

    rows = []
    for f in files:
        for line in Path(f).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            q = r["question"]
            hits = rag.retrieve(q, 5)
            s = [h.score for h in hits] + [0.0, 0.0]
            qt = terms(q)
            cov = len(qt & terms(" ".join(h.text for h in hits[:3]))) / len(qt) if qt else 0.0
            rows.append({
                "question": q,
                "answerable": bool(r.get("source")),
                "found": bool(r.get("source")) and _rank(hits, r) is not None,
                "top1": round(s[0], 4),
                "gap": round(s[0] - s[1], 4),
                "cov": round(cov, 3),
                "oov": sum(1 for t in qt if t not in df),     # question terms found nowhere in the corpus
                "split": "hold" if zlib.crc32(q.encode()) % 10 < 3 else "tune",
            })
    with CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def load_csv() -> list[dict]:
    with CSV.open(encoding="utf-8") as fh:
        return [{**r, "answerable": r["answerable"] == "True", "found": r["found"] == "True",
                 "top1": float(r["top1"]), "gap": float(r["gap"]),
                 "cov": float(r["cov"]), "oov": int(r["oov"])}
                for r in csv.DictReader(fh)]


def rates(rows, t, c, o=99):
    """(recall on answerable, false-accept rate on out-of-scope) for: top1>=t and cov>=c and oov<=o."""
    pos = [r for r in rows if r["answerable"]]
    neg = [r for r in rows if not r["answerable"]]
    ok = lambda r: r["top1"] >= t and r["cov"] >= c and r["oov"] <= o
    return (sum(r["found"] and ok(r) for r in pos) / max(len(pos), 1),
            sum(ok(r) for r in neg) / max(len(neg), 1))


def report(rows):
    tune = [r for r in rows if r["split"] == "tune"]
    hold = [r for r in rows if r["split"] == "hold"]
    grid = list(itertools.product([0, .01, .02, .05, .1, .2, .3, .5],
                                  [0, .4, .5, .6, .7, .8, 1.0], [99, 2, 1, 0]))

    def objective(g):
        rec, fa = rates(tune, *g)
        return rec - fa

    def best(cands):
        return max(cands, key=objective)

    rules = {
        "current (score>=0.05)": (0.05, 0, 99),
        "best score only": best([g for g in grid if g[1] == 0 and g[2] == 99]),
        "+ coverage": best([g for g in grid if g[2] == 99]),
        "+ coverage + oov": best(grid),
    }
    print(f"{len(tune)} tune / {len(hold)} hold rows\n")
    print(f"{'rule':22} {'t':>5} {'cov':>4} {'oov':>3} | tune rec/fa | hold rec/fa")
    for name, (t, c, o) in rules.items():
        a, b = rates(tune, t, c, o), rates(hold, t, c, o)
        print(f"{name:22} {t:>5} {c:>4} {o:>3} | {a[0]:.2f} / {a[1]:.2f} | {b[0]:.2f} / {b[1]:.2f}")

    t, c, o = rules["+ coverage + oov"]
    print()
    for r in rows:
        ok = r["top1"] >= t and r["cov"] >= c and r["oov"] <= o
        if r["answerable"] and not (r["found"] and ok):
            print("REJECTED VALID :", r["top1"], r["cov"], r["oov"], r["question"])
        if not r["answerable"] and ok:
            print("ACCEPTED OOS   :", r["top1"], r["cov"], r["oov"], r["question"])


if __name__ == "__main__":
    args = sys.argv[1:]
    report(load_csv() if "--cached" in args else collect(args or ["eval/all.jsonl"]))