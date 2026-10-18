# hybrid-rag-engine

> Status: work in progress, still learning. Feedback welcome.

A local, CPU-only hybrid RAG pipeline in Python whose main job is **refusing questions the documents cannot answer**.
It is a hands-on practice project built to production-style standards (eval gates, a frozen test set, retries, tests), not a live service.

The corpus is 7 small fictional documents (leave rules, company policy, contractor policy, IT FAQ, product specs, tech notes, incident runbook), 25 chunks in total.

## How it works

```
question
   |
   v
[retrieve]  dense (embeddings) + BM25  ->  RRF fusion  ->  cross-encoder rerank  ->  top 5
   |
   v
[gate]      best rerank score < abstain_below (0.001)?  -> abstain (no LLM call)
   |
   v
[judge]     LLM returns JSON: answer_found, verbatim quote, answer
            quote must be ~90% made of words from the context (no-LLM check)
            "not mentioned" style answers are refused
   |
   v
[fit]       "answer in 8 words or NONE" extraction on the quote
            + no-LLM unit check (how many -> a count unit, how often -> a frequency, ...)
   |
   v
answer   or   abstain  (with the closest passage shown when the judge refused)
```

Flow is a LangGraph state graph (`src/hybridrag/graph.py`). A query-rewrite node exists but is switched off (it rescued 1 valid question and caused 1 leak).

Why not just a score threshold? In the sweep, valid questions had top-1 scores as low as 0.0018 while out-of-scope ones reached 0.99, so no threshold separates them. The gate is only a near-zero floor; the real decision is made by the judge and the fit check.

## Stack

| Part | Choice |
|---|---|
| Language / env | Python, venv, VS Code, Windows PowerShell |
| Embeddings | `BAAI/bge-small-en-v1.5` |
| Reranker | `BAAI/bge-reranker-base` |
| Vector store | Chroma |
| Lexical search | BM25 |
| Fusion | Reciprocal Rank Fusion (dense 20 + BM25 10 -> fused 20 -> rerank -> keep 5) |
| LLM | `llama3.1:latest` via Ollama, temperature 0 |
| Orchestration | LangGraph |
| Config | `config/settings.yaml`, overridable with `RAG_<KEY>` environment variables |
| Tests | pytest (23 tests, mocked retriever and LLM, no models needed) |

Frozen settings: chunk 1000 / overlap 150, top_k 5, `abstain_below` 0.001, `max_retries` 0.

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
ollama pull llama3.1:latest
```

Run once without `HF_HUB_OFFLINE` so the Hugging Face models download. After that, before every run:

```powershell
$env:PYTHONUTF8="1"; $env:HF_HUB_OFFLINE="1"; $env:TQDM_DISABLE="1"
```

## Usage

```powershell
hybridrag ingest data\docs
hybridrag search "annual leave carry forward"
hybridrag ask "How many days of annual leave do employees get?"      # classic pipeline
hybridrag --log WARNING gask "Can unused sick leave be cashed out?"   # graph pipeline (judge + fit)
hybridrag chat
```

On a CPU, an answered question takes about 30 to 40 s (the judge call dominates); a refusal at the gate is much faster.

## Evaluation

```powershell
python scripts\check_eval.py eval\X.jsonl [--freeze]     # labels, duplicates, trap overlap (no models)
python scripts\run_eval.py eval\X.jsonl [--limit N] [--set-baseline]
```

`run_eval.py` caches results by a fingerprint of settings plus `graph.py`, reports leaks per type, latency, and prints a GATE line: leaks must not rise and correct must not drop against the saved baseline. After re-ingest or chunking changes: `Remove-Item -Recurse reports\.cache`.

| Set | Valid / out-of-scope | Role |
|---|---|---|
| `dev.jsonl` | 37 / 41 | tuned against |
| `dev2.jsonl` | 6 / 5 | past failures, for testing fixes |
| `frozen.jsonl` | 48 / 50 | hand-written, hash-frozen, no longer unseen |
| `frozen2.jsonl` | 20 / 20 | fresh, hash-frozen, run once |

Row format: `{"question", "source", "answer_contains"}` for valid rows, `{"question", "source": null, "type": "trap|messy|offdomain"}` for out-of-scope rows.

### Results

| Set | Correct (valid) | Leaks (out-of-scope) |
|---|---|---|
| dev | 33/37 | 2/41 |
| frozen (rerun, no tuning) | 44/48 | 5/50 (trap 5/30, messy 0/8, off-domain 0/12) |
| frozen2 (fresh) | 18/20 | 0/20 (trap 0/10, messy 0/4, off-domain 0/6) |

How to read these: dev numbers are tuned and say little about generalisation. frozen2 is the cleanest number, but 20 per side is small, so 0 leaks only means the true rate is probably low, not zero. frozen2 was written for the final test and its traps are easier than the old frozen set's.

## Known limitations

- **Near-domain leaks:** 5 of 30 trap questions on `frozen.jsonl` return an adjacent fact with confidence (for example the token validity when asked for the refresh endpoint timeout).
- **Table rows and fragments:** a quote without its subject (a markdown table row such as the SM-300 warranty, or "This means your device certificate has expired") can be refused by the fit check.
- **Negation questions** ("can sick leave be cashed out?") are refused by the judge.
- **English only:** Hinglish and heavy slang questions score near zero in retrieval.
- **Latency:** CPU only, so tens of seconds per answered question.

## Layout

```
config/settings.yaml
data/docs/                 7 fictional documents
eval/                      JSONL question sets
reports/                   baselines, frozen hashes (.cache is ignored)
scripts/                   run_eval.py check_eval.py inspect_failures.py (+ older experiment scripts)
src/hybridrag/             config chunking embeddings store retrieval generation pipeline
                           graph llm evaluate logging_utils cli
tests/                     test_chunking test_config test_graph test_llm
```

## Robustness and tracing

- `llm.py` wraps every Ollama call: 180 s timeout, 2 retries with backoff, and a clear error state instead of a traceback. Model-missing errors fail fast.
- `ask` rejects empty questions and questions over 500 characters.
- Every query gets a trace id, and each stage logs one greppable line: `trace=a1b2c3d4 stage=judge ms=22400 said=False quote_ok=False`.

## Tests

```powershell
pytest -q
```
