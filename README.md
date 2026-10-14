# hybrid-rag-engine

A hybrid Retrieval-Augmented Generation (RAG) pipeline in pure Python. It runs fully locally with no API keys, and includes an evaluation harness so every change is measured.

## Features

- Structure-aware chunking (Markdown headings and file names kept as context)
- Hybrid retrieval: dense vectors (ChromaDB) + BM25 keyword search
- Reciprocal Rank Fusion (RRF) to merge both rankings
- Cross-encoder reranking and a score threshold, so weak matches are dropped
- Grounded answers with source citations, using a local LLM via Ollama
- Evaluation: hit@k, MRR and abstain rate on out-of-scope questions
- Per-stage timing logs for review

## Pipeline

```
Documents -> clean -> chunk -> dedupe -> embed -> ChromaDB
Question  -> dense + BM25 -> RRF fusion -> rerank -> score filter -> LLM answer
```

## Quick start

Requires Python 3.10+ and [Ollama](https://ollama.com).

```bash
git clone https://github.com/krish21venkat-skills/hybrid-rag-engine.git
cd hybrid-rag-engine
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
ollama pull llama3.1
```

Add documents (PDF, MD, TXT) to `data/docs/`, then run:

```bash
hybridrag ingest data/docs
hybridrag search "your question"
hybridrag ask "your question"
hybridrag eval eval/questions.jsonl
```

## Configuration

Edit `config/settings.yaml`. Environment variables such as `RAG_TOP_K=7` override it.

| Setting | Default |
|---|---|
| `embed_model` | `BAAI/bge-small-en-v1.5` |
| `rerank_model` | `BAAI/bge-reranker-base` |
| `llm_model` | `llama3.1:latest` |
| `chunk_size` / `chunk_overlap` | 1000 / 150 |
| `top_k` | 5 |
| `min_rerank_score` | 0.05 |

## Evaluation

Test cases live in `eval/questions.jsonl`:

```json
{"question": "What is the hotel limit for contractors?", "source": "ContractorPolicy.txt", "answer_contains": "3,500"}
{"question": "What is the CEO's salary?", "source": null}
```

A question counts as a hit only if the right file is retrieved and the chunk contains the expected answer text. Questions with `"source": null` are out of scope, so the correct behaviour is returning no results.

**Current results** (24 answerable and 3 out-of-scope questions):

| hit@5 | MRR | Abstain rate |
|---|---|---|
| 0.958 | 0.917 | 1.0 |

## Project structure

```
hybrid-rag-engine/
├── pyproject.toml
├── config/settings.yaml
├── data/docs/              # sample documents (fictional)
├── eval/questions.jsonl
├── src/hybridrag/
│   ├── config.py  loaders.py  chunking.py  embeddings.py
│   ├── store.py   retrieval.py  generation.py  pipeline.py
│   ├── evaluate.py  logging_utils.py  cli.py
└── tests/
```

## Roadmap

- [ ] LangGraph agentic flow (query rewriting and retry)
- [ ] Bedrock and Azure OpenAI generators
- [ ] FastAPI endpoint for HTTP and UiPath integration

## Tests

```bash
pytest -q
```

## License

MIT