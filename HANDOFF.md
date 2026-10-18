# hybrid-rag-engine: handoff (18 Oct 2026)

Status: work in progress, not actively worked on right now.

## Purpose
Hands-on practice project, built to production-style standards. Local hybrid RAG whose main job is refusing questions the documents cannot answer. Not a live production flow.

## Environment
Windows PowerShell, VS Code, `.venv`, CPU only, 8 physical cores. Ollama models: `llama3.1:latest`, `nomic-embed-text`. Before every session:
`$env:PYTHONUTF8="1"; $env:HF_HUB_OFFLINE="1"; $env:TQDM_DISABLE="1"; $env:RAG_NUM_THREAD="8"`
Keep `RAG_NUM_THREAD` constant within a session (changing it forces an Ollama reload, about 45 s). Measured warm refusal: about 18 s with 8 threads, about 24 s with default threads (one run each).

## Original solution (classic pipeline)
```
question -> retrieve (dense + BM25 -> RRF -> rerank, top 5)
         -> score gate: top score < 0.05 -> refuse
         -> LLM generate ("answer only from context, else say you don't know")
         -> answer
```
Problem: refusal depended on a score threshold and on the LLM's willingness to say "I don't know". Valid questions scored as low as 0.0018, out-of-scope up to 0.99, so no threshold separated them, and a high-scoring neighbouring fact simply got answered. The sweep showed the right chunk was in the top 5 for 44/44 valid questions, so retrieval was fine.

## Final solution (graph pipeline)
```
question
  |
  v
[0. input check]   empty -> "Please enter a question"; over 500 chars -> rejected
  |                repeat of an earlier question -> cached answer (no LLM)
  v
[1. retrieve]      dense (20) + BM25 (10) -> RRF fusion (20) -> bge reranker -> top 5
  |                no LLM, same as the original
  v
[2. gate]          top score < 0.001 ? --yes--> abstain (plain refusal, no LLM call)
  |no              (was 0.05; now only a near-zero floor)
  v
[3. judge]         LLM (llama3.1, temp 0) returns JSON: answer_found, verbatim quote, short answer
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

Around everything:
  llm.py wrapper: 180 s timeout, 2 retries with backoff (2 s, 4 s), fail fast on 400/404, clear error if Ollama is down
  trace id per question, one log line per stage (start, retrieve, judge, fit, abstain, done)
  keep_alive 30 min and optional RAG_NUM_THREAD
```
Orchestrated as a LangGraph state graph in `src/hybridrag/graph.py`. Rewrite node exists but is off (`max_retries` 0).

### What each layer is for
| Layer | Catches | LLM? |
|---|---|---|
| Gate (0.001) | Questions with essentially no retrieval signal | No |
| Quote check | Judge inventing an answer that isn't in the text | No |
| `NOT_FOUND` guard | "The documents do not mention..." dressed up as an answer | No |
| Fit check + unit check | Adjacent facts (right topic, wrong specific thing) | Small call plus code |
| Closest passage | Turns a flat refusal into something useful | No |

## Stack and frozen config
Embeddings `BAAI/bge-small-en-v1.5`, reranker `BAAI/bge-reranker-base`, Chroma, BM25, RRF, `llama3.1:latest` via Ollama, LangGraph, pytest. Config in `config/settings.yaml`, overridable with `RAG_<KEY>` env vars. Chunk 1000/150, top_k 5, `abstain_below` 0.001, `max_retries` 0, `min_rerank_score` 0.05 (classic pipeline only). Corpus: 7 fictional docs (LeaveRules, CompanyPolicy, ContractorPolicy, TechNotes, ITHelpdeskFAQ, ProductSpecs, IncidentRunbook), 25 chunks.

## Changes in order
1. Sweep: retrieval fine, score gate is the wrong tool.
2. LangGraph pipeline with judge, quote check and fit check.
3. Eval harness: `check_eval.py` (labels, duplicates, trap overlap, hash freeze), `run_eval.py` (fingerprinted cache, per-type leaks, latency, GATE against a baseline), `inspect_failures.py`.
4. Experiment A rejected (fit-check lead-in: dev leaks 2 to 5). Experiment B kept (`abstain_below` 0.001: dev 32/37 to 33/37, leaks 2/41; rescued "12-day leave", "95% of requests", "meter data").
5. Partial answers (`closest_sentence()`, no LLM, at least 2 shared content words) and `hybridrag gask` for the graph pipeline.
6. Robustness: `llm_chat` wrapper, input validation, error state.
7. Tracing: `trace_id`, `_traced()` node wrapper, `stage()` in `logging_utils.py`.
8. Tests: 24 with a fake retriever and patched `llm_chat` (helpers, gate, answer path, NOT_FOUND guard, unsupported quote, fit block, partial answer, bad JSON, invalid input, LLM down, trace ids, cache; plus chunking, config, llm retry tests).
9. Speed: `keep_alive`, `num_thread`, in-memory answer cache (helps `chat`, not one-shot `gask` processes).
10. Fresh frozen set `frozen2.jsonl`, README, handoff, LinkedIn post.

## Rejected or dropped
Query rewrite (off), negation prompt rule (no effect; remove from `PROMPT` if still present, it was not in the final file), fit-check lead-in, more LLM downloads, feedback-log review.

## Results
| Set | Correct | Leaks |
|---|---|---|
| dev (tuned) | 33/37 | 2/41 |
| frozen (rerun, no tuning, no longer unseen) | 44/48 (was 42/48) | 5/50: trap 5/30, messy 0/8, offdomain 0/12 |
| frozen2 (fresh, run once) | 18/20 | 0/20: trap 0/10, messy 0/4, offdomain 0/6 |
Latency (CPU): answered p50 about 31 to 39 s, abstained p50 about 22 s in eval runs. Runs are repeatable at temperature 0.
frozen2 misses: "Tuesday" label too narrow for "Tuesdays and Wednesdays" (scorer issue, row kept) and the SM-300 warranty table row (fragment problem).
Honest reading: frozen.jsonl is no longer unseen; frozen2 is the clean number; 20 per side is small; near-domain leaks were not halved.

## Known failures
- Five adjacent-fact trap leaks on frozen: refresh timeout gets token validity, SEV1 phone gets the outage extension, the 10:00 AM call, ratings decider gets the scale, 90-day exit criteria.
- Fit check blocks fragment quotes (E-4021, "vpn wont connect says 707", SM-300 warranty table row).
- Negation limit ("cashed out", "carry forward sick leave", "paid sick day as contractor").
- "Join on the 20th" gets the "2 days per month" fact instead of the accrual rule.
- Hinglish and slang (English-only models).

## Eval sets
| File | Valid / out-of-scope | Role |
|---|---|---|
| dev.jsonl | 37 / 41 | tuned against |
| dev2.jsonl | 6 / 5 | past failures for testing fixes |
| frozen.jsonl | 48 / 50 | hash-frozen, no longer unseen |
| frozen2.jsonl | 20 / 20 | fresh, hash-frozen, run once |
Row format: `{"question","source","answer_contains"}` or `{"question","source":null,"type":"trap|messy|offdomain"}`. Files may start with a UTF-8 BOM: read with `utf-8-sig`.

## Files
`src/hybridrag/`: graph.py (graph, judge, fit, helpers, cache), llm.py (Ollama wrapper), logging_utils.py (`stage()`), cli.py (`gask`). `scripts/`: run_eval.py, check_eval.py, inspect_failures.py, partial_check.py, rank_check.py (retrieval rank analysis, no LLM). `tests/`: test_graph.py, test_llm.py, test_chunking.py, test_config.py. `reports/`: baselines and frozen hashes. After re-ingest or chunking changes: `Remove-Item -Recurse reports\.cache`.

## Rules
Never tune on frozen sets (a failing frozen question moves to dev2 and gets replaced). Change one thing at a time. Keep a change only if the GATE passes. Report off-domain and near-domain leaks separately. Don't quote dev numbers as generalisation. Eval fingerprint includes `graph.py`, so any edit there makes the next eval a full run.

## Next ideas
1. Structure-aware chunking that keeps table row labels (fragment problem).
2. Targeted fix for adjacent-fact leaks (goal: halve 5/30 near-domain leaks).
3. Optional speed idea: send only the top 3 chunks to the judge. First run `python scripts\rank_check.py eval\X.jsonl` on dev, frozen and frozen2 (retrieval only) and confirm valid questions rank 1 to 3; then gate on dev2 and dev.
4. Multilingual models for Hinglish.
5. A third fresh frozen set (about 20 per side) before any new claims.
6. Optional: remove the old per-node `log.info` lines now that stage tracing exists.

## Preferences
Claude free tier is limited: short answers, step-by-step with code and run commands, minimal logging.
