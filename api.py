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
from langchain_core.messages import AIMessage, AIMessageChunk
from pydantic import BaseModel, Field

from research_agent.config import settings
from research_agent.documents import pdf_to_chunks
from research_agent.graph import build_graph, turn_input
from research_agent.llm import text_of
from research_agent.tools import ddg_search
from research_agent.vectorstore import StoreRegistry

MAX_UPLOAD_MB = 25
STEP_LABELS = {
    "planner": "Planning how to answer",
    "retriever": "Searching your documents",
    "web_search": "Searching the web",
    "writer": "Writing the answer",
    "grader": "Checking the answer against sources",
    "finalize": "Done",
}


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

    def cfg(session_id: str) -> dict:
        return {"configurable": {"thread_id": session_id}}

    def result_payload(session_id: str) -> dict:
        values = graph.get_state(cfg(session_id)).values
        return {"answer": values.get("answer", ""), "sources": values.get("sources", []),
                "route": values.get("route", "")}

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
        await graph.ainvoke(turn_input(req.session_id, req.question), cfg(req.session_id))
        return result_payload(req.session_id)

    @app.post("/chat/stream")
    async def chat_stream(req: ChatRequest):
        """Streams newline-delimited JSON events:
        {"type": "step", "node", "label"} - an agent finished / started
        {"type": "token", "text"}         - answer text as it is generated
        {"type": "retry", "reason"}       - reviewer rejected the draft, a new one follows
        {"type": "final", "answer", "sources", "route"}
        """
        route = {"value": ""}

        def step(node: str, label: str | None = None) -> dict:
            return {"type": "step", "node": node, "label": label or STEP_LABELS[node]}

        def search_step() -> dict:
            return step("web_search") if route["value"] == "web" else step("retriever")

        def progress_events(node: str, update: dict) -> list[dict]:
            """Turn a node's state update into UI events (what happened + what runs next)."""
            if node == "planner":
                route["value"] = update.get("route", "")
                nxt = {"documents": search_step(), "both": step("retriever"), "web": step("web_search")}
                return [step("route", f"Route: {route['value']}"), nxt.get(route["value"], step("writer"))]
            if node == "retriever":
                found = step("retriever", f"Found {len(update.get('doc_results', []))} relevant passages")
                return [found, step("web_search") if route["value"] == "both" else step("writer")]
            if node == "web_search":
                return [step("web_search", f"Found {len(update.get('web_results', []))} web results"),
                        step("writer")]
            if node == "writer":
                return [step("grader")]
            if node == "grader" and update.get("retry"):
                return [{"type": "retry", "reason": update.get("feedback", "")}, search_step()]
            return []

        async def events() -> AsyncIterator[str]:
            yield json.dumps({"type": "step", "node": "planner", "label": STEP_LABELS["planner"]}) + "\n"
            try:
                async for mode, chunk in graph.astream(turn_input(req.session_id, req.question),
                                                       cfg(req.session_id), stream_mode=["updates", "messages"]):
                    if mode == "messages":
                        msg, meta = chunk
                        if meta.get("langgraph_node") == "writer" and isinstance(msg, (AIMessageChunk, AIMessage)):
                            text = text_of(msg.content)
                            if text:
                                yield json.dumps({"type": "token", "text": text}) + "\n"
                    else:  # updates: {node_name: state_update}
                        for node, update in chunk.items():
                            for event in progress_events(node, update or {}):
                                yield json.dumps(event) + "\n"
                yield json.dumps({"type": "final", **result_payload(req.session_id)}) + "\n"
            except Exception as exc:  # surface model/API errors to the UI instead of a broken stream
                yield json.dumps({"type": "error", "message": f"{type(exc).__name__}: {exc}"}) + "\n"

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
