from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from .config import Settings
from .loaders import Page


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict


def clean(text: str) -> str:
    text = re.sub(r"-\n(\w)", r"\1", text)   # re-join hyphenated line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class Chunker:
    def __init__(self, settings: Settings):
        self.s = settings
        self._md = MarkdownHeaderTextSplitter(
            [("#", "h1"), ("##", "h2"), ("###", "h3")], strip_headers=False
        )
        self._text = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],  # paragraph first, then sentence
        )

    def split(self, page: Page) -> list[Chunk]:
        out: list[Chunk] = []
        for section in self._md.split_text(clean(page.text)):
            heading = " > ".join(section.metadata.values())
            for piece in self._text.split_text(section.page_content):
                piece = piece.strip()
                if len(piece) < self.s.min_chunk_len:
                    continue
                # Heading prefix gives the embedding section context
                #body = f"[{heading}]\n{piece}" if heading and not piece.startswith("#") else piece
                label = " > ".join(filter(None, [Path(page.source).stem, heading]))
                body = f"[{label}]\n{piece}" if not piece.startswith("#") else piece    
                cid = hashlib.sha1(f"{page.source}|{body}".encode()).hexdigest()
                out.append(Chunk(cid, body, {
                    "source": page.source,
                    "page": page.page or 0,      # 0 = not paged (Chroma rejects None)
                    "section": heading,
                }))
        return out