from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

log = logging.getLogger(__name__)
SUPPORTED = {".pdf", ".md", ".txt"}


@dataclass
class Page:
    text: str
    source: str
    page: int | None = None


def load_pages(folder: str | Path) -> Iterator[Page]:
    for f in sorted(Path(folder).rglob("*")):
        if f.suffix.lower() not in SUPPORTED:
            continue
        try:
            if f.suffix.lower() == ".pdf":
                for i, p in enumerate(PdfReader(str(f)).pages, 1):
                    yield Page(p.extract_text() or "", f.name, i)
            else:
                yield Page(f.read_text(encoding="utf-8", errors="ignore"), f.name)
        except Exception:
            log.exception("Failed to load %s, skipping", f)