"""Split multi-part questions into sub-questions (partial answers, step 1)."""
from __future__ import annotations

import json
import logging
import re
from typing import Callable

logger = logging.getLogger(__name__)

MAX_SUBQUESTIONS = 4

# Cheap signals that a question may have several parts.
_MULTI_PART = re.compile(r"\?.*\?|\band\b|\balso\b|;|\bas well as\b", re.I | re.S)

_PROMPT = (
    "Split the question into independent sub-questions. "
    "Return ONLY a JSON array of strings. "
    "If it is a single question, return an array with just that question.\n\n"
    "Question: {question}"
)


def has_multi_part_signal(question: str) -> bool:
    """Return True if the question looks like it has more than one part."""
    return bool(_MULTI_PART.search(question))


def _parse_list(raw: str) -> list[str]:
    """Extract a JSON array of non-empty strings from an LLM reply."""
    match = re.search(r"\[.*\]", raw, re.S)
    if not match:
        return []
    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return [s.strip() for s in items if isinstance(s, str) and s.strip()]


def split_question(question: str, llm: Callable[[str], str]) -> list[str]:
    """Return sub-questions, or [question] if no split is needed or possible.

    `llm` is any function that takes a prompt and returns text, so the
    existing llama3.1 call can be passed in and tests can use a fake.
    """
    if not has_multi_part_signal(question):
        return [question]

    parts = _parse_list(llm(_PROMPT.format(question=question)))
    if len(parts) < 2:
        logger.info("split: fallback to single question")
        return [question]

    parts = parts[:MAX_SUBQUESTIONS]
    logger.info("split: %d sub-questions", len(parts))
    return parts