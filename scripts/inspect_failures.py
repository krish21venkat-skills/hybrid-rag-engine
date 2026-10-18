"""Copy frozen-set failures into eval\\dev2.jsonl and diagnose missed valid questions that passed the gate."""
import json
from pathlib import Path

frozen = {}
for l in Path("eval/frozen.jsonl").read_text(encoding="utf-8-sig").splitlines():
    if l.strip():
        r = json.loads(l)
        frozen[r["question"]] = r

report = json.loads(Path("reports/baseline_frozen.json").read_text(encoding="utf-8"))
bad, passed_gate = [], []
for row in report["rows"]:
    r = frozen[row["question"]]
    valid = bool(r.get("source"))
    if valid and row["decision"] != "answer":
        bad.append(r)
        if row["top1"] >= 0.01:
            passed_gate.append(r)
    elif not valid and row["decision"] == "answer":
        bad.append(r)

Path("eval/dev2.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in bad) + "\n", encoding="utf-8")
print(f"wrote eval/dev2.jsonl with {len(bad)} failing rows")

if passed_gate:
    from hybridrag.graph import GraphRAG
    g = GraphRAG()
    for r in passed_gate:
        o = g.ask(r["question"])
        print(f"\n{r['question']}\n  top1 {o['top1']:.3f} | said {o['said']} | quote_ok {o['quote_ok']} | fit_blocked {o['fit_blocked']}")
        print("  quote :", o["quote"][:110])
        print("  answer:", o["answer"][:80])