"""Turns a LangGraph run into a stream of simple UI events.

Shared by the FastAPI backend (api.py) and the Streamlit app when it runs the
agent in-process (ui.py without API_URL, e.g. on Streamlit Community Cloud).

Event types:
    {"type": "step", "node", "label"}             progress of the agents
    {"type": "token", "text"}                     answer text as it is generated
    {"type": "retry", "reason"}                   reviewer rejected the draft; a new one follows
    {"type": "final", "answer", "sources", "route"}
    {"type": "error", "message"}
"""
from __future__ import annotations

from typing import AsyncIterator, Iterator

from langchain_core.messages import AIMessage, AIMessageChunk

from research_agent.graph import turn_input
from research_agent.llm import text_of

STEP_LABELS = {
    "planner": "Planning how to answer",
    "retriever": "Searching your documents",
    "web_search": "Searching the web",
    "writer": "Writing the answer",
    "grader": "Checking the answer against sources",
}
STREAM_MODES = ["updates", "messages"]


def config(session_id: str) -> dict:
    return {"configurable": {"thread_id": session_id}}


def result_payload(graph, session_id: str) -> dict:
    values = graph.get_state(config(session_id)).values
    return {"answer": values.get("answer", ""), "sources": values.get("sources", []),
            "route": values.get("route", "")}


class EventTranslator:
    """Converts raw (mode, chunk) items from graph.stream into UI events."""

    def __init__(self):
        self.route = ""

    @staticmethod
    def step(node: str, label: str | None = None) -> dict:
        return {"type": "step", "node": node, "label": label or STEP_LABELS[node]}

    def _search_step(self) -> dict:
        return self.step("web_search") if self.route == "web" else self.step("retriever")

    def _progress(self, node: str, update: dict) -> list[dict]:
        if node == "planner":
            self.route = update.get("route", "")
            nxt = {"documents": self.step("retriever"), "both": self.step("retriever"),
                   "web": self.step("web_search")}.get(self.route, self.step("writer"))
            return [self.step("route", f"Route: {self.route}"), nxt]
        if node == "retriever":
            found = self.step("retriever", f"Found {len(update.get('doc_results', []))} relevant passages")
            return [found, self.step("web_search") if self.route == "both" else self.step("writer")]
        if node == "web_search":
            return [self.step("web_search", f"Found {len(update.get('web_results', []))} web results"),
                    self.step("writer")]
        if node == "writer":
            return [self.step("grader")]
        if node == "grader" and update.get("retry"):
            return [{"type": "retry", "reason": update.get("feedback", "")}, self._search_step()]
        return []

    def translate(self, mode: str, chunk) -> list[dict]:
        if mode == "messages":
            msg, meta = chunk
            if meta.get("langgraph_node") == "writer" and isinstance(msg, (AIMessageChunk, AIMessage)):
                text = text_of(msg.content)
                if text:
                    return [{"type": "token", "text": text}]
            return []
        events = []
        for node, update in chunk.items():  # updates: {node_name: state_update}
            events += self._progress(node, update or {})
        return events


def iter_events(graph, session_id: str, question: str) -> Iterator[dict]:
    """Synchronous event stream (used by the Streamlit app in local mode)."""
    yield EventTranslator.step("planner")
    translator = EventTranslator()
    try:
        for mode, chunk in graph.stream(turn_input(session_id, question), config(session_id),
                                        stream_mode=STREAM_MODES):
            yield from translator.translate(mode, chunk)
        yield {"type": "final", **result_payload(graph, session_id)}
    except Exception as exc:  # surface model/API errors to the UI instead of crashing
        yield {"type": "error", "message": f"{type(exc).__name__}: {exc}"}


async def aiter_events(graph, session_id: str, question: str) -> AsyncIterator[dict]:
    """Async event stream (used by the FastAPI backend)."""
    yield EventTranslator.step("planner")
    translator = EventTranslator()
    try:
        async for mode, chunk in graph.astream(turn_input(session_id, question), config(session_id),
                                               stream_mode=STREAM_MODES):
            for event in translator.translate(mode, chunk):
                yield event
        yield {"type": "final", **result_payload(graph, session_id)}
    except Exception as exc:
        yield {"type": "error", "message": f"{type(exc).__name__}: {exc}"}
