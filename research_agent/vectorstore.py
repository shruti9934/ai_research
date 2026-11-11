"""A small FAISS vector store, one index per chat session, persisted to disk.

Vectors are L2-normalised and stored in an ``IndexFlatIP`` so the inner
product equals cosine similarity.
"""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path

import faiss
import numpy as np
from langchain_core.embeddings import Embeddings

from research_agent.documents import Chunk

EMBED_BATCH = 64


def _normalise(vectors: list[list[float]]) -> np.ndarray:
    arr = np.asarray(vectors, dtype="float32")
    faiss.normalize_L2(arr)
    return arr


class FaissStore:
    def __init__(self, embeddings: Embeddings, path: Path | None = None):
        self.embeddings = embeddings
        self.path = path
        self.index: faiss.IndexFlatIP | None = None
        self.chunks: list[Chunk] = []
        self._lock = threading.Lock()
        if path and (path / "index.faiss").exists():
            self._load()

    # ---------------------------------------------------------------- write
    def add(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        vectors = []
        for i in range(0, len(chunks), EMBED_BATCH):
            vectors += self.embeddings.embed_documents([c.text for c in chunks[i:i + EMBED_BATCH]])
        arr = _normalise(vectors)
        with self._lock:
            if self.index is None:
                self.index = faiss.IndexFlatIP(arr.shape[1])
            self.index.add(arr)
            self.chunks.extend(chunks)
            self._save()
        return len(chunks)

    # ---------------------------------------------------------------- read
    def search(self, query: str, k: int = 5) -> list[dict]:
        if self.index is None or self.index.ntotal == 0:
            return []
        q = _normalise([self.embeddings.embed_query(query)])
        scores, ids = self.index.search(q, min(k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append({**chunk.to_dict(), "score": float(score)})
        return results

    @property
    def is_empty(self) -> bool:
        return self.index is None or self.index.ntotal == 0

    def documents(self) -> list[dict]:
        summary: dict[str, dict] = {}
        for c in self.chunks:
            d = summary.setdefault(c.source, {"source": c.source, "chunks": 0, "pages": set()})
            d["chunks"] += 1
            d["pages"].add(c.page)
        return [{"source": d["source"], "chunks": d["chunks"], "pages": len(d["pages"])} for d in summary.values()]

    # ---------------------------------------------------------------- persistence
    def _save(self) -> None:
        if not self.path:
            return
        self.path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.path / "index.faiss"))
        (self.path / "chunks.json").write_text(json.dumps([c.to_dict() for c in self.chunks]))

    def _load(self) -> None:
        self.index = faiss.read_index(str(self.path / "index.faiss"))
        self.chunks = [Chunk(**c) for c in json.loads((self.path / "chunks.json").read_text())]


class StoreRegistry:
    """Hands out one FaissStore per session id (cached in memory, persisted on disk)."""

    def __init__(self, embeddings: Embeddings, root: Path | None):
        self.embeddings = embeddings
        self.root = root
        self._stores: dict[str, FaissStore] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str) -> FaissStore:
        safe = re.sub(r"[^A-Za-z0-9_-]", "_", session_id)[:64] or "default"
        with self._lock:
            if safe not in self._stores:
                path = self.root / safe if self.root else None
                self._stores[safe] = FaissStore(self.embeddings, path)
            return self._stores[safe]

    def drop(self, session_id: str) -> None:
        import shutil
        safe = re.sub(r"[^A-Za-z0-9_-]", "_", session_id)[:64] or "default"
        with self._lock:
            self._stores.pop(safe, None)
            if self.root and (self.root / safe).exists():
                shutil.rmtree(self.root / safe)
