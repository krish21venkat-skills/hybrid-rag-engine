from __future__ import annotations

from typing import Protocol

import ollama

from .config import Settings
from .retrieval import Hit

SYSTEM = (
    "You answer strictly from the provided context. Cite sources as [1], [2]. "
    "If the context does not contain the answer, say you don't know. Do not use outside knowledge."
)


def build_context(hits: list[Hit]) -> str:
    return "\n\n".join(
        f"[{i}] ({h.metadata.get('source')}, p.{h.metadata.get('page') or '-'})\n{h.text}"
        for i, h in enumerate(hits, 1)
    )


class Generator(Protocol):
    def generate(self, question: str, hits: list[Hit]) -> str: ...


class OllamaGenerator:
    def __init__(self, settings: Settings):
        self.model = settings.llm_model

    def generate(self, question: str, hits: list[Hit]) -> str:
        resp = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Context:\n{build_context(hits)}\n\nQuestion: {question}"},
            ],
            options={"temperature": 0},
        )
        return resp["message"]["content"]


def make_generator(settings: Settings) -> Generator:
    if settings.llm_provider == "ollama":
        return OllamaGenerator(settings)
    # Later: add BedrockGenerator / AzureOpenAIGenerator and register them here
    raise ValueError(f"Unknown llm_provider: {settings.llm_provider!r}")