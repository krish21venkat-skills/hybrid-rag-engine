"""Zone table from sweep.csv (no retrieval, runs instantly).
python scripts\\zones.py              # low=0.015 high=0.25
python scripts\\zones.py 0.01 0.3
"""
import csv
import sys

low = float(sys.argv[1]) if len(sys.argv) > 1 else 0.015
high = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25

rows = list(csv.DictReader(open("sweep.csv", encoding="utf-8")))
zones = {"abstain": [], "borderline": [], "answer": []}
for r in rows:
    t = float(r["top1"])
    zones["abstain" if t < low else "answer" if t >= high else "borderline"].append(r)

print(f"low={low} high={high}\n")
for z, rs in zones.items():
    v = sum(r["answerable"] == "True" for r in rs)
    print(f"{z:10} valid={v:2}  out_of_scope={len(rs) - v:2}")

print("\nBLOCKED VALID (abstain zone):")
for r in zones["abstain"]:
    if r["answerable"] == "True":
        print(f"  {r['top1']:>7} {r['question']}")
print("\nLEAKS (answer zone, out-of-scope):")
for r in zones["answer"]:
    if r["answerable"] == "False":
        print(f"  {r['top1']:>7} {r['question']}")
print("\nBORDERLINE (needs escalation):")
for r in sorted(zones["borderline"], key=lambda r: float(r["top1"])):
    print(f"  {r['top1']:>7} {'VALID' if r['answerable'] == 'True' else 'oos  '} {r['question']}")