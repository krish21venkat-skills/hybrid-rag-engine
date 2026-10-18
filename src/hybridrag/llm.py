from __future__ import annotations

import logging
import os
import time

import ollama

log = logging.getLogger(__name__)

TIMEOUT_S = 180      # judge p95 is about 65 s on CPU
RETRIES = 2
_client = ollama.Client(timeout=TIMEOUT_S)


class LLMUnavailable(RuntimeError):
    """Ollama did not answer after all retries."""


def llm_chat(retries: int = RETRIES, **kw):
    kw.setdefault("keep_alive", "30m")              # keep the model loaded between questions
    opts = dict(kw.get("options") or {})
    if os.getenv("RAG_NUM_THREAD"):                 # optional CPU thread override
        opts["num_thread"] = int(os.environ["RAG_NUM_THREAD"])
    kw["options"] = opts
    last: Exception | None = None
    for i in range(retries + 1):
        try:
            return _client.chat(**kw)
        except ollama.ResponseError as e:           # e.g. model missing: retrying won't help
            if e.status_code in (400, 404):
                raise LLMUnavailable(str(e)) from e
            last = e
        except Exception as e:                      # timeout, connection refused
            last = e
        log.warning("ollama call failed (%d/%d): %s", i + 1, retries + 1, last)
        if i < retries:
            time.sleep(2 * (i + 1))
    raise LLMUnavailable(str(last)) from last