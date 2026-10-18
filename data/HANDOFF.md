# Hybrid RAG: hand-off notes

## Status (14 Oct 2026)
- Repo: hybrid-rag-engine (GitHub), tags v0.1-baseline and v0.2
- Eval (24 answerable + 3 out-of-scope): hit@5 1.000, MRR 0.958, abstain 1.0
- Stack: Python, ChromaDB, bge-small embeddings, BM25, bge-reranker-base, Ollama llama3.1:latest
- Next planned: LangGraph on branch feature/langgraph

## How it works
Ingest: load -> clean -> chunk (heading + file name prefix) -> dedupe (hash) -> embed -> ChromaDB
Query: dense top 20 + BM25 top 20 -> RRF fusion -> cross-encoder rerank -> keep top 5 with score >= min_rerank_score -> LLM answers from chunks with [n] citations. If nothing passes, it returns "couldn't find" and never calls the LLM.

Why each part exists:
- Dense search finds paraphrases, BM25 finds exact terms and codes, RRF merges them without score scaling
- The reranker is slow but precise, so it only scores about 20 candidates
- The threshold is what makes the system refuse safely instead of guessing

## Tuning rules
| Change | Re-ingest? |
|---|---|
| min_rerank_score, top_k, candidates, llm_model | No |
| chunk_size, chunk_overlap, embed_model, chunk text format | Yes (delete storage/ first) |

Loop: change one thing, wipe storage if needed, run `hybridrag eval`, record the numbers, keep it only if they improve.

Diagnosing a miss: run `search` with `RAG_MIN_RERANK_SCORE=0` and check the scores.
- Right chunk, low score -> threshold or chunk too diluted
- Wrong file on top -> embedding or chunk context problem
- Right chunk at rank 4-5 -> ranking problem

## Known weaknesses
- Eval set is small and self-written, and only 3 out-of-scope questions, so abstain rate is weak evidence
- Rerank on CPU is slow (about 6 s per query)
- Characters, not tokens, define chunk size
- BM25 and lookup rebuilt in memory from the whole store (fine for small corpora only)
- No conversation memory, no query rewriting, and multi-fact questions across documents answer poorly

## What to improve (priority order)
1. Add 10 or more out-of-scope questions and 10 or more answerable ones, from live testing failures
2. Run `hybridrag calibrate` and set the threshold from data
3. Try chunk 500/75 vs 1000/150 and bge-base vs bge-small, one at a time
4. Ablation: dense only, BM25 only, no reranker, to show what each part adds (good README material)
5. Try nomic-embed-text (already installed in Ollama) as an alternative embedder
6. Parent-child chunking (small chunks to match, larger text to the LLM)
7. Metadata filters (restrict by file or document type)
8. Query rewriting and retry via LangGraph, the fallback node, conversation memory
9. Bedrock/Azure generator behind the existing Generator interface, FastAPI endpoint for UiPath

## Skill practice (write yourself, don't paste)
1. clean(), tokenize(), rrf(), evaluate() from memory
2. Your own chunker without LangChain
3. A NumPy cosine-similarity search before relying on Chroma

## Commands
pytest -q | hybridrag ingest data\docs | search | ask | chat | eval eval\questions.jsonl | calibrate eval\questions.jsonl
Windows: set $env:PYTHONUTF8 = "1" first.
Git: commit with the GitHub noreply email, and never commit .venv/, storage/ or real documents.