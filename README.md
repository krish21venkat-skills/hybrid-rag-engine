# hybrid-rag-engine

> Status: work in progress, still learning. Feedback welcome.

A local, CPU-only hybrid RAG pipeline in Python whose main job is **knowing when not to answer**.

It is a hands-on practice project built to production-style standards (eval gates, frozen test sets, retries, tests), not a live service. The corpus is 7 small fictional documents (leave rules, company policy, contractor policy, IT FAQ, product specs, tech notes, incident runbook), 25 chunks in total.

## Highlights

- **Hybrid retrieval:** dense search + BM25, rank fusion (RRF), cross-encoder rerank.
- **Answers have to earn their place:** the LLM must give a word-for-word quote, code verifies the quote, and a second check confirms the quote answers the actual question.
- **Useful refusals:** when it can't answer, it shows the closest passage and says it may be about something different.
- **Evaluation discipline:** hash-frozen test sets that are never tuned on, and a gate that blocks any change that adds wrong answers.
- **Engineering basics:** timeouts and retries around the LLM, input validation, per-query trace IDs, 24 unit tests that run in seconds with no models.

## Why not just a score threshold?

The first version refused when the top rerank score fell below a threshold. A sweep showed that cannot work: valid questions scored as low as 0.0018 while out-of-scope questions scored up to 0.99. The right chunk was in the top 5 for every valid question, so retrieval was fine. The refusal decision had to move out of the retriever and into checks on the answer itself.

## Architecture

```
question
  |
  v
[0. input check]   empty -> asked to enter a question; over 500 chars -> rejected
  |                repeat of an earlier question -> cached answer (no LLM)
  v
[1. retrieve]      dense (20) + BM25 (10) -> RRF fusion (20) -> bge reranker -> top 5
  |                no LLM
  v
[2. gate]          top score < 0.001 ? --yes--> abstain (plain refusal, no LLM call)
  |no              (a near-zero floor only; it was 0.05 in the first version)
  v
[3. judge]         LLM (llama3.1, temperature 0) returns JSON:
  |                  answer_found, a verbatim quote, a short answer
  |                code checks, no LLM:
  |                  - quote must be ~90% made of words from the context
  |                  - "not mentioned"-style answers are refused
  |                judge says no --> abstain + closest passage shown
  v
[4. fit check]     LLM extraction ("answer in 8 words or NONE") on the quote
  |                + no-LLM unit check (how many -> a count unit, how often -> a frequency)
  |                fails --> abstain (plain refusal)
  v
[5. answer]        short answer returned

around everything: LLM wrapper (180 s timeout, 2 retries, clear error if Ollama is down),
                   trace id per question with one log line per stage,
                   keep_alive + optional RAG_NUM_THREAD for faster CPU inference
```

The flow is a LangGraph state graph in `src/hybridrag/graph.py`. A query-rewrite node exists but is switched off.

### What each layer catches

| Layer | Catches | Uses LLM |
|---|---|---|
| Gate (0.001) | Questions with essentially no retrieval signal | No |
| Quote check | Judge inventing an answer that isn't in the text | No |
| `NOT_FOUND` guard | "The documents do not mention..." dressed up as an answer | No |
| Fit check + unit check | Adjacent facts: right topic, wrong specific thing | Small call plus code |
| Closest passage | Turns a flat refusal into something useful | No |

## How it evolved

| Original (classic) | Final (graph) |
|---|---|
| Score gate at 0.05 decides refusal | Gate is only a 0.001 floor; judge and fit check decide |
| LLM free-text answer, trusted to say "I don't know" | JSON with a verbatim quote, verified in code |
| Flat "not found" | Closest passage shown when the judge refuses |
| No retries or input checks | Timeout, retries, validation, error state |
| Plain logs | Trace ID and one line per stage |
| Ad hoc checks | 24 mocked unit tests and a gated eval harness |

## Experiments

| Change | Outcome |
|---|---|
| Lower `abstain_below` from 0.01 to 0.001 | Kept. Rescued 3 valid questions that scored 0.002 to 0.005. dev 32/37 to 33/37, leaks stayed 2/41. Cost: refusals reach the judge more often, so they are slower. |
| Fit check sees the text before the quote | Rejected. Rescued 1 question on the failure set but raised dev leaks from 2 to 5. |
| Query rewrite | Left off. Rescued 1 valid question, caused 1 leak, added latency. |
| Negation rule in the judge prompt | No effect. |
| Partial answers (closest passage) | Kept. No change to any decision. |
| Send only the top 3 chunks to the judge | Not tried. Check ranks first with `scripts/rank_check.py` (retrieval only, no LLM). |

## Evaluation

```powershell
python scripts\check_eval.py eval\X.jsonl [--freeze]       # labels, duplicates, trap overlap (no models)
python scripts\run_eval.py eval\X.jsonl [--limit N] [--set-baseline]
```

`run_eval.py` caches results by a fingerprint of settings plus `graph.py`, reports leaks by type and latency, and prints a GATE line: leaks must not rise and correct must not drop against the saved baseline. After re-ingest or chunking changes, run `Remove-Item -Recurse reports\.cache`.

| Set | Valid / out-of-scope | Role |
|---|---|---|
| `dev.jsonl` | 37 / 41 | Tuned against |
| `dev2.jsonl` | 6 / 5 | Past failures, for testing fixes |
| `frozen.jsonl` | 48 / 50 | Hand-written, hash-frozen, no longer unseen |
| `frozen2.jsonl` | 20 / 20 | Fresh, hash-frozen, run once |

### Results

| Set | Correct (valid) | Leaks (out-of-scope) |
|---|---|---|
| dev | 33/37 | 2/41 |
| frozen (rerun, no tuning) | 44/48 | 5/50 (trap 5/30, messy 0/8, off-domain 0/12) |
| frozen2 (fresh) | 18/20 | 0/20 (trap 0/10, messy 0/4, off-domain 0/6) |

How to read these: dev numbers are tuned and say little about generalisation. frozen2 is the cleanest number, but 20 per side is small, so zero leaks only means the true rate is probably low, not zero. frozen2 was written for the final test and its traps are easier than the old frozen set's. The same fictional documents are used throughout.

## Quick start (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
ollama pull llama3.1:latest
```

Run once without `HF_HUB_OFFLINE` so the Hugging Face models download. After that, before every session:

```powershell
$env:PYTHONUTF8="1"; $env:HF_HUB_OFFLINE="1"; $env:TQDM_DISABLE="1"
```

Optional: set `$env:RAG_NUM_THREAD` to your physical core count for faster CPU inference. Keep it constant within a session, because changing it makes Ollama reload the model.

## Usage

```powershell
hybridrag ingest data\docs
hybridrag search "annual leave carry forward"
hybridrag ask "How many days of annual leave do employees get?"       # classic pipeline
hybridrag --log WARNING gask "Can unused sick leave be cashed out?"    # graph pipeline (judge + fit check)
hybridrag chat
pytest -q
```

Speed on a CPU: a warm question takes roughly 18 to 35 s, and the judge call is most of that. Questions that fail the gate skip the LLM entirely.

## Stack

| Part | Choice |
|---|---|
| Embeddings | `BAAI/bge-small-en-v1.5` |
| Reranker | `BAAI/bge-reranker-base` |
| Vector store | Chroma |
| Lexical search | BM25 |
| Fusion | Reciprocal Rank Fusion (dense 20 + BM25 10 -> fused 20 -> rerank -> keep 5) |
| LLM | `llama3.1:latest` via Ollama, temperature 0 |
| Orchestration | LangGraph |
| Config | `config/settings.yaml`, overridable with `RAG_<KEY>` environment variables |
| Tests | pytest |

Frozen settings: chunk 1000 / overlap 150, top_k 5, `abstain_below` 0.001, `max_retries` 0.

## Project layout

```
config/settings.yaml
data/docs/                 7 fictional documents
eval/                      dev, dev2, frozen, frozen2 question sets
reports/                   baselines and frozen-set hashes
scripts/                   run_eval.py, check_eval.py, inspect_failures.py, partial_check.py, rank_check.py
src/hybridrag/             config, chunking, embeddings, store, retrieval, generation, pipeline,
                           graph, llm, evaluate, logging_utils, cli
tests/                     test_chunking, test_config, test_graph, test_llm
HANDOFF.md                 detailed notes: decisions, failures, next ideas
```

## Known limitations and next ideas

- **Near-miss questions:** 5 of 30 trap questions on `frozen.jsonl` return an adjacent fact with confidence (for example token validity when asked for the refresh endpoint timeout).
- **Table rows and fragments:** a quote without its subject (a markdown table row, or "This means your device certificate has expired") can be wrongly refused by the fit check.
- **Negation questions** such as "can sick leave be cashed out?" are refused by the judge.
- **English only:** Hinglish and heavy slang score near zero in retrieval. Multilingual models would be the next step.
- **Latency:** CPU only, so tens of seconds per answered question.
- **Ideas:** structure-aware chunking that keeps table row labels, a targeted fix for adjacent-fact leaks, the top-3 chunk experiment, and a third fresh frozen set before making any new claims.
