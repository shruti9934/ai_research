"""The LangGraph multi-agent workflow.

                    ┌────────────┐
       question ──▶ │  planner   │  decides route + rewrites the query
                    └─────┬──────┘
         documents/both   │   web          direct
        ┌─────────────────┼────────────┬───────────────┐
        ▼                 │            ▼               │
  ┌───────────┐  both     │     ┌────────────┐         │
  │ retriever │ ─────────────▶  │ web_search │         │
  └─────┬─────┘                 └─────┬──────┘         │
        └───────────────┬─────────────┘                │
                        ▼                              ▼
                  ┌──────────┐   tool calls    ┌────────────┐
                  │  writer  │ ◀─────────────▶ │ calculator │
                  └────┬─────┘                 └────────────┘
                       ▼
                  ┌──────────┐  not grounded → better query → retriever / web_search
                  │  grader  │ ─────────────────────────────────────────────────────
                  └────┬─────┘
                       ▼ grounded
                  ┌──────────┐
                  │ finalize │ → saves answer to conversation memory
                  └──────────┘
"""
from __future__ import annotations

import re
from typing import Annotated, Callable, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from research_agent.config import settings
from research_agent.llm import text_of
from research_agent.prompts import GRADER_SYSTEM, PLANNER_SYSTEM, WRITER_NO_SOURCES, WRITER_SYSTEM
from research_agent.tools import calculator, ddg_search
from research_agent.vectorstore import StoreRegistry

MAX_TOOL_ROUNDS = 3


# --------------------------------------------------------------------------- state & schemas
class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]  # conversation memory (checkpointed)
    session_id: str
    question: str
    route: str
    doc_query: str
    web_query: str
    doc_results: list[dict]
    web_results: list[dict]
    answer: str
    sources: list[dict]
    attempts: int
    retry: bool
    feedback: str


class Plan(BaseModel):
    """Routing decision made by the planner."""
    route: Literal["documents", "web", "both", "direct"] = Field(description="Where to look for the answer")
    doc_query: str = Field(default="", description="Stand-alone query for searching the uploaded documents")
    web_query: str = Field(default="", description="Stand-alone query for a web search")


class Grade(BaseModel):
    """Reviewer verdict on a draft answer."""
    grounded: bool = Field(description="True if the answer is fully supported by the sources and answers the question")
    reason: str = Field(default="", description="One sentence explaining the verdict")
    better_query: str = Field(default="", description="A better search query if grounded is false")


# --------------------------------------------------------------------------- helpers
def _history(state: AgentState) -> list[AnyMessage]:
    """Previous turns (excluding the current question), trimmed to the last N turns."""
    msgs = [m for m in state.get("messages", [])[:-1] if isinstance(m, (HumanMessage, AIMessage))]
    return msgs[-2 * settings.history_turns:]


def _history_text(state: AgentState) -> str:
    lines = [f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {text_of(m.content)[:500]}"
             for m in _history(state)]
    return "\n".join(lines) or "(no previous messages)"


def _numbered_sources(state: AgentState) -> list[dict]:
    sources, n = [], 1
    for r in state.get("doc_results", []):
        sources.append({"id": n, "type": "document", "source": r["source"], "page": r["page"], "text": r["text"]})
        n += 1
    for r in state.get("web_results", []):
        sources.append({"id": n, "type": "web", "title": r["title"], "url": r["url"], "text": r["snippet"]})
        n += 1
    return sources


def _format_sources(sources: list[dict]) -> str:
    blocks = []
    for s in sources:
        label = f"{s['source']}, page {s['page']}" if s["type"] == "document" else f"{s['title']} ({s['url']})"
        blocks.append(f"[{s['id']}] {label}\n{s['text']}")
    return "\n\n".join(blocks)


def _merge(old: list[dict], new: list[dict], key: Callable[[dict], object]) -> list[dict]:
    seen = {key(r) for r in old}
    return old + [r for r in new if key(r) not in seen]


# --------------------------------------------------------------------------- graph factory
def build_graph(llm: BaseChatModel, stores: StoreRegistry,
                search_fn: Callable[[str, int], list[dict]] | None = None, checkpointer=None):
    search_fn = search_fn or ddg_search
    planner_llm = llm.with_structured_output(Plan)
    grader_llm = llm.with_structured_output(Grade)
    writer_llm = llm.bind_tools([calculator])
    tools_by_name = {calculator.name: calculator}

    # ---- Planner agent ------------------------------------------------------
    def planner(state: AgentState) -> dict:
        store = stores.get(state["session_id"])
        docs = ", ".join(d["source"] for d in store.documents()) or "none uploaded"
        plan: Plan = planner_llm.invoke([
            SystemMessage(PLANNER_SYSTEM.format(documents=docs)),
            HumanMessage(f"Conversation so far:\n{_history_text(state)}\n\nLatest question: {state['question']}"),
        ])
        route = plan.route
        if store.is_empty and route in ("documents", "both"):
            route = "web"  # nothing to retrieve from
        return {
            "route": route,
            "doc_query": plan.doc_query or state["question"],
            "web_query": plan.web_query or plan.doc_query or state["question"],
        }

    # ---- Retriever agent (RAG over FAISS) ------------------------------------
    def retriever(state: AgentState) -> dict:
        hits = stores.get(state["session_id"]).search(state["doc_query"], k=settings.top_k)
        merged = _merge(state.get("doc_results", []), hits, key=lambda r: (r["source"], r["page"], r["text"]))
        return {"doc_results": merged}

    # ---- Web-search agent ----------------------------------------------------
    def web_search(state: AgentState) -> dict:
        hits = search_fn(state["web_query"], settings.web_results)
        merged = _merge(state.get("web_results", []), hits, key=lambda r: r["url"])
        return {"web_results": merged}

    # ---- Writer agent (with calculator tool calling) -------------------------
    def writer(state: AgentState) -> dict:
        sources = _numbered_sources(state)
        if sources:
            feedback = (f"\nA reviewer rejected the previous draft: {state['feedback']}\n"
                        if state.get("feedback") else "")
            system = WRITER_SYSTEM.format(context=_format_sources(sources), feedback=feedback)
        else:
            system = WRITER_NO_SOURCES
        messages: list[AnyMessage] = [SystemMessage(system), *_history(state), HumanMessage(state["question"])]

        response = writer_llm.invoke(messages)
        for _ in range(MAX_TOOL_ROUNDS):
            if not getattr(response, "tool_calls", None):
                break
            messages.append(response)
            for call in response.tool_calls:
                result = tools_by_name[call["name"]].invoke(call["args"]) if call["name"] in tools_by_name \
                    else f"Unknown tool {call['name']}"
                messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
            response = writer_llm.invoke(messages)
        return {"answer": text_of(response.content).strip(), "sources": sources, "retry": False}

    # ---- Grader / reflection agent ------------------------------------------
    def grader(state: AgentState) -> dict:
        if state["route"] == "direct" or state.get("attempts", 0) >= settings.max_reflection_retries:
            return {"retry": False}
        grade: Grade = grader_llm.invoke([
            SystemMessage(GRADER_SYSTEM),
            HumanMessage(f"Question: {state['question']}\n\nSources:\n{_format_sources(state.get('sources', []))}"
                         f"\n\nDraft answer:\n{state['answer']}"),
        ])
        if grade.grounded:
            return {"retry": False}
        query = grade.better_query or state["question"]
        return {"retry": True, "attempts": state.get("attempts", 0) + 1, "feedback": grade.reason,
                "doc_query": query, "web_query": query}

    # ---- Finalize: keep cited sources, save answer to memory -----------------
    def finalize(state: AgentState) -> dict:
        cited = {int(n) for n in re.findall(r"\[(\d+)\]", state.get("answer", ""))}
        sources = [s for s in state.get("sources", []) if s["id"] in cited]
        return {"sources": sources, "messages": [AIMessage(state["answer"])]}

    # ---- routing --------------------------------------------------------------
    def after_planner(state: AgentState) -> str:
        return {"documents": "retriever", "both": "retriever", "web": "web_search"}.get(state["route"], "writer")

    def after_retriever(state: AgentState) -> str:
        return "web_search" if state["route"] == "both" else "writer"

    def after_grader(state: AgentState) -> str:
        if not state.get("retry"):
            return "finalize"
        return "web_search" if state["route"] == "web" else "retriever"

    g = StateGraph(AgentState)
    for name, fn in [("planner", planner), ("retriever", retriever), ("web_search", web_search),
                     ("writer", writer), ("grader", grader), ("finalize", finalize)]:
        g.add_node(name, fn)
    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", after_planner, ["retriever", "web_search", "writer"])
    g.add_conditional_edges("retriever", after_retriever, ["web_search", "writer"])
    g.add_edge("web_search", "writer")
    g.add_edge("writer", "grader")
    g.add_conditional_edges("grader", after_grader, ["retriever", "web_search", "finalize"])
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer or MemorySaver())


def turn_input(session_id: str, question: str) -> dict:
    """State update for a new user turn: add the question, reset per-turn fields."""
    return {
        "messages": [HumanMessage(question)], "session_id": session_id, "question": question,
        "doc_results": [], "web_results": [], "answer": "", "sources": [], "attempts": 0,
        "retry": False, "feedback": "", "route": "", "doc_query": "", "web_query": "",
    }
