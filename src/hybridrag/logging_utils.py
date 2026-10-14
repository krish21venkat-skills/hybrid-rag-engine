from __future__ import annotations

import logging
import time
from contextlib import contextmanager


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "chromadb", "sentence_transformers", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@contextmanager
def timed(logger: logging.Logger, label: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        logger.info("%s took %.0f ms", label, (time.perf_counter() - start) * 1000)