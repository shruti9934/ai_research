"""FastAPI backend for the research assistant.

Run with:  uvicorn api:app --reload
Docs at:   http://localhost:8000/docs
"""
from __future__ import annotations

import json
from typing import AsyncIterator

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from research_agent.config import settings
from research_agent.documents import pdf_to_chunks
from research_agent.graph import build_graph, turn_input
from research_agent.service import aiter_events, config, result_payload
from research_agent.tools import ddg_search
from research_agent.vectorstore import StoreRegistry

MAX_UPLOAD_MB = 25


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)
    question: str = Field(min_length=1, max_length=4000)


def create_app(llm=None, embeddings=None, search_fn=None, index_dir=settings.index_dir) -> FastAPI:
    """App factory - tests inject fake models; normal runs build real ones from env settings."""
    if llm is None or embeddings is None:
        from research_agent.llm import get_chat_model, get_embeddings
        llm = llm or get_chat_model()
        embeddings = embeddings or get_embeddings()

    stores = StoreRegistry(embeddings, index_dir)
    graph = build_graph(llm, stores, search_fn=search_fn or ddg_search)
    app = FastAPI(title="AI Research Assistant", version="1.0.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "provider": settings.llm_provider}

    @app.post("/sessions/{session_id}/documents")
    async def upload(session_id: str, files: list[UploadFile] = File(...)):
        store = stores.get(session_id)
        added = []
        for f in files:
            if not (f.filename or "").lower().endswith(".pdf"):
                raise HTTPException(400, f"{f.filename}: only PDF files are supported")
            data = await f.read()
            if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
                raise HTTPException(413, f"{f.filename} is larger than {MAX_UPLOAD_MB} MB")
            try:
                chunks = pdf_to_chunks(data, f.filename)
            except ValueError as exc:
                raise HTTPException(422, str(exc))
            n = await run_in_threadpool(store.add, chunks)
            added.append({"file": f.filename, "chunks": n})
        return {"added": added, "documents": store.documents()}

    @app.get("/sessions/{session_id}/documents")
    def list_documents(session_id: str):
        return {"documents": stores.get(session_id).documents()}

    @app.delete("/sessions/{session_id}")
    def delete_session(session_id: str):
        stores.drop(session_id)
        graph.checkpointer.delete_thread(session_id)
        return {"deleted": session_id}

    @app.post("/chat")
    async def chat(req: ChatRequest):
        await graph.ainvoke(turn_input(req.session_id, req.question), config(req.session_id))
        return result_payload(graph, req.session_id)

    @app.post("/chat/stream")
    async def chat_stream(req: ChatRequest):
        """Streams newline-delimited JSON events (see research_agent/service.py for the event types)."""
        async def events() -> AsyncIterator[str]:
            async for event in aiter_events(graph, req.session_id, req.question):
                yield json.dumps(event) + "\n"

        return StreamingResponse(events(), media_type="application/x-ndjson")

    app.state.graph = graph
    app.state.stores = stores
    return app


def __getattr__(name):
    # Build the real app lazily so `import api` in tests doesn't need API keys.
    if name == "app":
        globals()["app"] = create_app()
        return globals()["app"]
    raise AttributeError(name)
