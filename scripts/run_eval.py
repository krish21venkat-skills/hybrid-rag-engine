"""Production-style eval run: fingerprinted cache, per-type metrics, latency, saved report, pass/fail gate.

python scripts\\run_eval.py eval\\dev.jsonl                  # run and compare with reports\\baseline_dev.json
python scripts\\run_eval.py eval\\dev.jsonl --set-baseline   # accept this run as the new baseline
python scripts\\run_eval.py eval\\dev.jsonl --limit 8        # quick smoke run on an evenly spaced subset

Rows: {"question","source","answer_contains"}  or  {"question","source":null,"type":"trap|offdomain|messy"}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path

import hybridrag.graph as G
from hybridrag.logging_utils import setup_logging

REPORTS = Path("reports")


def fingerprint(settings) -> str:
    """Changes when settings or graph.py (prompts, regexes, routing) change, so stale cached answers are never reused."""
    blob = json.dumps(asdict(settings), default=str, sort_keys=True) + Path(G.__file__).read_text(encoding="utf-8")
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def covers(expected: str, text: str) -> bool:
    et = re.findall(r"\w+", expected.lower())
    tt = set(re.findall(r"\w+", text.lower()))
    return not et or sum(t in tt for t in et) / len(et) >= 0.6


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--set-baseline", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.file).read_text(encoding="utf-8-sig").splitlines() if l.strip()]
    if args.limit and args.limit < len(rows):
        rows = rows[:: max(1, len(rows) // args.limit)][: args.limit]

    setup_logging("WARNING")
    g = G.GraphRAG()
    fp, stem = fingerprint(g.s), Path(args.file).stem
    cache_path = REPORTS / ".cache" / f"{stem}_{fp}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}

    out = []
    for n, r in enumerate(rows, 1):
        q = r["question"]
        if q not in cache:
            t0 = time.perf_counter()
            try:
                o = g.ask(q)
                cache[q] = {"decision": o["decision"], "answer": o["answer"], "quote": o["quote"],
                            "top1": round(o["top1"], 4), "fit_blocked": o["fit_blocked"], "error": ""}
            except Exception as e:                      # one bad question must not kill the run
                cache[q] = {"decision": "error", "answer": "", "quote": "", "top1": 0.0,
                            "fit_blocked": False, "error": repr(e)[:200]}
            cache[q]["seconds"] = round(time.perf_counter() - t0, 1)
            cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        out.append((r, cache[q]))
        print(f"[{n}/{len(rows)}]", end=" ", flush=True)
    print()

    valid = [(r, o) for r, o in out if r.get("source")]
    oos = [(r, o) for r, o in out if not r.get("source")]
    is_correct = lambda r, o: o["decision"] == "answer" and covers(r.get("answer_contains", ""), o["answer"] + " " + o["quote"])
    correct = sum(is_correct(r, o) for r, o in valid)
    answered = sum(o["decision"] == "answer" for _, o in valid)
    leaks = sum(o["decision"] == "answer" for _, o in oos)
    errors = sum(o["decision"] == "error" for _, o in out)
    by_type: dict[str, list[int]] = {}
    for r, o in oos:
        t = by_type.setdefault(r.get("type", "oos"), [0, 0])
        t[0] += 1
        t[1] += o["decision"] == "answer"
    ans_secs = [o["seconds"] for _, o in out if o["decision"] == "answer"]
    abs_secs = [o["seconds"] for _, o in out if o["decision"] == "abstain"]

    for r, o in valid:
        if o["decision"] != "answer":
            print(f"MISSED VALID ({o['decision']}, top1={o['top1']}) | {r['question']}")
        elif not is_correct(r, o):
            print(f"WRONG ANSWER | {r['question']}\n      got: {o['answer'][:100]} | expected: {r.get('answer_contains')}")
    for r, o in oos:
        if o["decision"] == "answer":
            print(f"LEAK [{r.get('type', 'oos')}] | {r['question']}\n      got: {o['answer'][:100]}")
    for r, o in out:
        if o["decision"] == "error":
            print(f"ERROR | {r['question']} | {o['error']}")

    summary = {"n_valid": len(valid), "valid_answered": answered, "valid_correct": correct,
               "n_oos": len(oos), "leaks": leaks, "errors": errors,
               "leaks_by_type": {k: f"{v[1]}/{v[0]}" for k, v in by_type.items()},
               "answer_p50_s": pct(ans_secs, .5), "answer_p95_s": pct(ans_secs, .95),
               "abstain_p50_s": pct(abs_secs, .5)}
    print(f"\nfingerprint={fp} file={args.file} questions={len(rows)}")
    print(f"valid: answered {answered}/{len(valid)}, correct {correct}/{len(valid)}")
    print(f"out-of-scope: leaks {leaks}/{len(oos)}  " + "  ".join(f"{k}: {v[1]}/{v[0]}" for k, v in by_type.items()))
    print(f"errors: {errors} | latency answered p50 {summary['answer_p50_s']}s p95 {summary['answer_p95_s']}s,"
          f" abstained p50 {summary['abstain_p50_s']}s")

    REPORTS.mkdir(exist_ok=True)
    report = REPORTS / f"{stem}_{fp}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    report.write_text(json.dumps({"file": args.file, "fingerprint": fp, "settings": asdict(g.s),
                                  "summary": summary,
                                  "rows": [{"question": r["question"], **o} for r, o in out]},
                                 ensure_ascii=False, default=str, indent=1), encoding="utf-8")

    base_path = REPORTS / f"baseline_{stem}.json"
    problems = []
    if errors:
        problems.append(f"{errors} errors")
    if base_path.exists() and not args.limit:
        b = json.loads(base_path.read_text(encoding="utf-8"))["summary"]
        if leaks > b["leaks"]:
            problems.append(f"leaks {leaks} > baseline {b['leaks']}")
        if correct < b["valid_correct"]:
            problems.append(f"valid correct {correct} < baseline {b['valid_correct']}")
        print(f"baseline: correct {b['valid_correct']}/{b['n_valid']}, leaks {b['leaks']}/{b['n_oos']}")
    elif not base_path.exists():
        print("no baseline yet (use --set-baseline to accept this run)")
    print("GATE:", "FAIL - " + "; ".join(problems) if problems else "PASS")

    if args.set_baseline and not args.limit:
        base_path.write_text(report.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"baseline saved: {base_path}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())