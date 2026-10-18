from __future__ import annotations

import argparse
import json
import time

from .logging_utils import setup_logging
from .pipeline import RAGPipeline

def chat(rag: RAGPipeline) -> None:
    print("Ask a question. ':s <question>' = retrieval only, ':q' = quit")
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line in {":q", ":quit"}:
            break
        if line.startswith(":s "):
            hits = rag.retrieve(line[3:])
        else:
            ans = rag.ask(line)
            hits = ans.hits
            print(f"\n{ans.text}\n({ans.latency_ms:.0f} ms)")
        if not hits:
            print("  no chunks passed the threshold")
        for i, h in enumerate(hits, 1):
            print(f"  [{i}] {h.score:.3f} | {h.metadata['source']} | {h.text[:90]!r}")

def main() -> None:
    p = argparse.ArgumentParser(prog="hybridrag")
    p.add_argument("--log", default="INFO")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest").add_argument("folder")
    sub.add_parser("search").add_argument("query", nargs="+")
    sub.add_parser("ask").add_argument("question", nargs="+")
    sub.add_parser("chat")
    sub.add_parser("gask").add_argument("question", nargs="+")
    ev = sub.add_parser("eval")
    ev.add_argument("file")
    ev.add_argument("-k", type=int, default=5)
    args = p.parse_args()

    setup_logging(args.log)
    rag = RAGPipeline()

    if args.cmd == "ingest":
        print(f"Ingested {rag.ingest(args.folder)} chunks.")
    elif args.cmd == "search":
        for h in rag.retrieve(" ".join(args.query)):
            print(f"\n--- {h.score:.3f} | {h.metadata['source']} p.{h.metadata['page'] or '-'} ---\n{h.text}")
    elif args.cmd == "ask":
        a = rag.ask(" ".join(args.question))
        print(a.text, f"\n\n({len(a.hits)} sources, {a.latency_ms:.0f} ms)")
    elif args.cmd == "chat":
        chat(rag)
    elif args.cmd == "gask":
        from .graph import GraphRAG
        t0 = time.perf_counter()
        st = GraphRAG().ask(" ".join(args.question))
        print(st.get("partial") or st["answer"], f"\n\n({(time.perf_counter() - t0) * 1000:.0f} ms)")
    else:
        from .evaluate import evaluate
        print(json.dumps(evaluate(rag, args.file, args.k), indent=2))