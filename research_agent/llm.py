"""Factory functions for the chat model and embedding model.

Switch providers with the LLM_PROVIDER env var; the rest of the code only
sees LangChain's common BaseChatModel / Embeddings interfaces.
"""
from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from research_agent.config import settings


def get_chat_model() -> BaseChatModel:
    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=settings.openai_model)
    if settings.llm_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=settings.gemini_model, temperature=0.2)
    raise ValueError(f"Unknown LLM_PROVIDER {settings.llm_provider!r} (use 'gemini' or 'openai')")


def get_embeddings() -> Embeddings:
    if settings.llm_provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=settings.openai_embedding_model)
    if settings.llm_provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model=settings.gemini_embedding_model)
    raise ValueError(f"Unknown LLM_PROVIDER {settings.llm_provider!r}")


def text_of(content) -> str:
    """Message content can be a plain string or a list of content blocks (Gemini) - flatten to text."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)
