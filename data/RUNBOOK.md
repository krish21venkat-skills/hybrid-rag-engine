# hybrid-rag-engine: Runbook and Project Notes

## Part 1: Runbook

### 1. Daily workflow

```powershell
cd D:\Hands_On_Projects\Python\hybrid-rag-engine
.venv\Scripts\activate
$env:PYTHONUTF8 = "1"
ollama list                      # confirm llama3.1:latest is installed
```

### 2. Command reference

| Task | Command |
|---|---|
| Install project (once) | `pip install -e ".[dev]"` |
| Run unit tests | `pytest -q` |
| Build the knowledge base | `hybridrag ingest data\docs` |
| Inspect retrieval only | `hybridrag search "question"` |
| Full grounded answer | `hybridrag ask "question"` |
| Measure quality | `hybridrag eval eval\questions.jsonl` |
| See only failures | `hybridrag eval eval\questions.jsonl 2>&1 \| Select-String "MISS"` |
| More detailed logs | `hybridrag --log DEBUG ask "question"` |
| Wipe and rebuild the DB | `Remove-Item -Recurse -Force storage` then ingest again |

### 3. The tuning loop (how to improve quality)

1. Run `hybridrag eval` and record hit@5, MRR and abstain rate.
2. Change **one** thing (chunk size, model, threshold, candidates).
3. If the change affects chunking or embeddings, wipe `storage\` and re-ingest.
4. Run eval again and compare. Keep the change only if the numbers improve.
5. Record the run in the results table in the README.

| Changed | Re-ingest needed? |
|---|---|
| `chunk_size`, `chunk_overlap`, `embed_model`, chunk text format | Yes (wipe `storage\` first) |
| `min_rerank_score`, `top_k`, `candidates`, `rerank_model`, `llm_model` | No |
| Added or edited documents | Yes (ingest again) |

### 4. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: hybridrag` | Package not installed | `pip install -e ".[dev]"` inside the venv |
| `model 'llama3.1:8b' not found` | Model tag mismatch | Set `llm_model: llama3.1:latest` |
| Garbled characters or decode errors | Windows encoding | `$env:PYTHONUTF8 = "1"` |
| First `ingest` is slow | Models downloading once | Wait; later runs use the local cache |
| Rerank takes several seconds | Cross-encoder on CPU | Lower `candidates` to 10, or accept it for now |
| Chroma dimension error after changing the embedding model | Old vectors have another size | Delete `storage\` and re-ingest |
| `kept=0` for questions that should work | `min_rerank_score` too high | Lower it and compare scores in