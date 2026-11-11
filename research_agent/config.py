"""Settings, read from environment variables (or a .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    # "gemini" (free tier available) or "openai"
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "gemini").lower())
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3.6-flash"))
    gemini_embedding_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-5-mini"))
    openai_embedding_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))

    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    top_k: int = int(os.getenv("TOP_K", "5"))
    web_results: int = int(os.getenv("WEB_RESULTS", "4"))
    max_reflection_retries: int = int(os.getenv("MAX_REFLECTION_RETRIES", "1"))
    history_turns: int = int(os.getenv("HISTORY_TURNS", "6"))

    index_dir: Path = field(default_factory=lambda: Path(os.getenv("INDEX_DIR", ROOT / "data" / "indexes")))
    api_url: str = field(default_factory=lambda: os.getenv("API_URL", "http://localhost:8000"))


settings = Settings()
