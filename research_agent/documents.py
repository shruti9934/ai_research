"""PDF loading and chunking, keeping the page number of every chunk for citations."""
from __future__ import annotations

import io
from dataclasses import asdict, dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from research_agent.config import settings


@dataclass
class Chunk:
    text: str
    source: str  # file name
    page: int    # 1-based page number

    def to_dict(self) -> dict:
        return asdict(self)


def load_pdf(data: bytes, filename: str) -> list[tuple[int, str]]:
    """Return (page_number, text) for each page that has extractable text."""
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((i, text))
    if not pages:
        raise ValueError(f"No extractable text in {filename} (is it a scanned PDF?)")
    return pages


def chunk_pages(pages: list[tuple[int, str]], filename: str) -> list[Chunk]:
    """Split each page into overlapping chunks so every chunk maps to exactly one page."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for page_no, text in pages:
        for piece in splitter.split_text(text):
            chunks.append(Chunk(text=piece, source=filename, page=page_no))
    return chunks


def pdf_to_chunks(data: bytes, filename: str) -> list[Chunk]:
    return chunk_pages(load_pdf(data, filename), filename)
