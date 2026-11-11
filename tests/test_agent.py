"""End-to-end tests of the RAG pipeline, LangGraph workflow and FastAPI endpoints.

They use a scripted fake LLM + deterministic fake embeddings, so no API key is needed.
Run with:  pytest -q
"""
from __future__ import annotations

import io
import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage, ToolMessage
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from api import create_app
from research_agent.documents import pdf_to_chunks
from research_agent.tools import calculator
from research_agent.vectorstore import FaissStore
from tests.fakes import ScriptedChatModel


def make_pdf(pages: list[str]) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    for text in pages:
        y = 800
        for line in text.split("\n"):
            c.drawString(50, y, line)
            y -= 16
        c.showPage()
    c.save()
    return buf.getvalue()


PDF = make_pdf([
    "Acme Corp Annual Report 2025\nRevenue grew to 1250 crore rupees in 2025.",
    "Risks\nThe main risk is rising cloud costs.\nRevenue in 2024 was 980 crore rupees.",
])


def fake_search(query: str, n: int) -> list[dict]:
    return [{"title": f"Result for {query}", "url": "https://example.com/a", "snippet": "Web fact about Acme."}]


@pytest.fixture
def embeddings():
    return DeterministicFakeEmbedding(size=64)


# --------------------------------------------------------------------------- units
def test_chunks_keep_page_numbers():
    chunks = pdf_to_chunks(PDF, "report.pdf")
    assert {c.page for c in chunks} == {1, 2}
    assert all(c.source == "report.pdf" for c in chunks)


def test_faiss_store_search_and_persist(tmp_path, embeddings):
    store = FaissStore(embeddings, tmp_path / "s1")
    store.add(pdf_to_chunks(PDF, "report.pdf"))
    exact = store.chunks[0].text
    hits = store.search(exact, k=2)
    assert hits[0]["text"] == exact and hits[0]["score"] == pytest.approx(1.0, abs=1e-4)
    reloaded = FaissStore(embeddings, tmp_path / "s1")  # loads from disk
    assert len(reloaded.chunks) == len(store.chunks)
    assert reloaded.documents() == [{"source": "report.pdf", "chunks": len(store.chunks), "pages": 2}]


def test_calculator_is_safe():
    assert calculator.invoke({"expression": "(1250 - 980) / 980 * 100"}) == "27.55102"
    assert calculator.invoke({"expression": "__import__('os').system('ls')"}).startswith("Error")


# --------------------------------------------------------------------------- full workflow via the API
def test_documents_route_with_tool_call_reflection_and_memory(tmp_path, embeddings):
    def writer(messages):
        # first call: ask the calculator; after the tool result: answer with a citation
        if not any(isinstance(m, ToolMessage) for m in messages):
            return AIMessage("", tool_calls=[{"name": "calculator", "args": {"expression": "(1250-980)/980*100"},
                                              "id": "c1"}])
        result = [m for m in messages if isinstance(m, ToolMessage)][-1].content
        return AIMessage(f"Revenue grew {result}% [1].")

    llm = ScriptedChatModel(plan={"route": "documents", "doc_query": "Acme revenue growth"},
                            grades=[{"grounded": False, "reason": "missing 2024 figure",
                                     "better_query": "revenue 2024"}],
                            writer=writer, calls={})
    client = TestClient(create_app(llm=llm, embeddings=embeddings, search_fn=fake_search, index_dir=tmp_path))

    r = client.post("/sessions/abc/documents", files=[("files", ("report.pdf", PDF, "application/pdf"))])
    assert r.status_code == 200 and r.json()["added"][0]["chunks"] >= 2

    r = client.post("/chat", json={"session_id": "abc", "question": "How much did revenue grow?"})
    body = r.json()
    assert r.status_code == 200
    assert body["route"] == "documents"
    assert body["answer"] == "Revenue grew 27.55102% [1]."
    assert len(body["sources"]) == 1 and body["sources"][0]["type"] == "document"
    assert len(llm.calls["Grade"]) == 1     # reviewer ran once and asked for a retry
    assert len(llm.calls["writer"]) == 4    # 2 drafts x (tool call + final answer)

    # conversation memory: the second turn's planner sees the first answer
    client.post("/chat", json={"session_id": "abc", "question": "And what about risks?"})
    planner_prompt = llm.calls["Plan"][-1][-1].content
    assert "Revenue grew 27.55102%" in planner_prompt


def test_web_route_when_no_documents_and_streaming(tmp_path, embeddings):
    llm = ScriptedChatModel(plan={"route": "documents", "doc_query": "acme"},  # no docs -> forced to web
                            writer=lambda m: AIMessage("Acme is a company [1]."), calls={})
    client = TestClient(create_app(llm=llm, embeddings=embeddings, search_fn=fake_search, index_dir=tmp_path))

    with client.stream("POST", "/chat/stream", json={"session_id": "s2", "question": "What is Acme?"}) as r:
        events = [json.loads(line) for line in r.iter_lines() if line]
    types = [e["type"] for e in events]
    assert "token" in types and types[-1] == "final"
    final = events[-1]
    assert final["route"] == "web"
    assert final["sources"][0]["url"] == "https://example.com/a"
    assert "".join(e["text"] for e in events if e["type"] == "token") == "Acme is a company [1]."


def test_rejects_non_pdf(tmp_path, embeddings):
    client = TestClient(create_app(llm=ScriptedChatModel(calls={}), embeddings=embeddings, index_dir=tmp_path))
    r = client.post("/sessions/x/documents", files=[("files", ("notes.txt", b"hi", "text/plain"))])
    assert r.status_code == 400
