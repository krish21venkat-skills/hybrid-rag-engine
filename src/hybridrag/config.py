from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Settings:
    embed_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "BAAI/bge-reranker-base"
    llm_provider: str = "ollama"
    llm_model: str = "llama3.1:latest"
    query_prefix: str = "Represent this sentence for searching relevant passages: "  # bge query instruction

    chunk_size: int = 1000          # characters
    chunk_overlap: int = 150
    min_chunk_len: int = 80         # drop header-only / noise chunks

    candidates: int = 20            # per retriever, before fusion
    top_k: int = 5                  # final chunks sent to the LLM
    min_rerank_score: float = 0.05  # tune on your eval set
    rrf_k: int = 60

    db_path: Path = Path("storage/chroma")
    collection: str = "docs"


def load_settings(path: str | Path = "config/settings.yaml") -> Settings:
    """defaults < YAML file < RAG_<NAME> environment variables."""
    values: dict = {}
    p = Path(path)
    if p.exists():
        values = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    names = {f.name for f in fields(Settings)}
    unknown = set(values) - names
    if unknown:
        raise ValueError(f"Unknown settings in {p}: {sorted(unknown)}")

    for name in names:
        env = os.getenv(f"RAG_{name.upper()}")
        if env is not None:
            values[name] = env

    defaults = Settings()
    for name, v in values.items():
        values[name] = type(getattr(defaults, name))(v)   # cast to int / float / Path / str
    return Settings(**values)