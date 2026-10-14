from __future__ import annotations

import argparse
import json

from .logging_utils import setup_logging
from .pipeline import RAGPipeline


def main() -> None:
    p = argparse.ArgumentParser(prog="hybridrag")
    p.add_argument("--log", default="INFO")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest").add_argument("folder")
    sub.add_parser("search").add_argument("query", nargs="+")
    sub.add_parser("ask").add_argument("question", nargs="+")
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
    else:
        from .evaluate import evaluate
        print(json.dumps(evaluate(rag, args.file, args.k), indent=2))