"""Streamlit chat UI.

Two ways to run it:
* Local mode (default): `streamlit run ui.py` runs the LangGraph agent inside the Streamlit app.
  This is also how it runs on Streamlit Community Cloud.
* API mode: set API_URL (e.g. http://localhost:8000) and start `uvicorn api:app` first; the UI then
  talks to the FastAPI backend over HTTP (used by docker-compose).
"""
from __future__ import annotations

import json
import os
import uuid

import streamlit as st

st.set_page_config(page_title="AI Research Assistant", page_icon="🔎", layout="wide")

# On Streamlit Cloud the API key lives in st.secrets - expose it as an env var before the agent reads settings.
SECRET_NAMES: list[str] = []
try:
    for _key, _value in st.secrets.items():
        # accept top-level keys and keys inside a [section]
        items = _value.items() if hasattr(_value, "items") else [(_key, _value)]
        for _name, _val in items:
            if isinstance(_val, str) and _val.strip():
                SECRET_NAMES.append(_name)
                os.environ.setdefault(_name.upper(), _val.strip())
except Exception:  # no secrets file when running locally
    pass
if not os.getenv("GOOGLE_API_KEY") and os.getenv("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

API_URL = os.getenv("API_URL", "").strip()


# --------------------------------------------------------------------------- backends
class RemoteBackend:
    """Talks to the FastAPI server."""

    def __init__(self, url: str):
        import requests
        self.url, self.http = url.rstrip("/"), requests

    def upload(self, sid, files):
        r = self.http.post(f"{self.url}/sessions/{sid}/documents",
                           files=[("files", (f.name, f.getvalue(), "application/pdf")) for f in files], timeout=300)
        if not r.ok:
            raise RuntimeError(r.json().get("detail", r.text))
        return r.json()["added"]

    def documents(self, sid):
        return self.http.get(f"{self.url}/sessions/{sid}/documents", timeout=10).json()["documents"]

    def reset(self, sid):
        self.http.delete(f"{self.url}/sessions/{sid}", timeout=10)

    def stream(self, sid, question):
        with self.http.post(f"{self.url}/chat/stream", json={"session_id": sid, "question": question},
                            stream=True, timeout=300) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if line:
                    yield json.loads(line)


class LocalBackend:
    """Runs the agent in this process."""

    def __init__(self):
        from research_agent.config import settings
        from research_agent.graph import build_graph
        from research_agent.llm import get_chat_model, get_embeddings
        from research_agent.vectorstore import StoreRegistry

        self.stores = StoreRegistry(get_embeddings(), settings.index_dir)
        self.graph = build_graph(get_chat_model(), self.stores)

    def upload(self, sid, files):
        from research_agent.documents import pdf_to_chunks
        added = []
        for f in files:
            try:
                chunks = pdf_to_chunks(f.getvalue(), f.name)
            except ValueError as exc:
                raise RuntimeError(str(exc))
            added.append({"file": f.name, "chunks": self.stores.get(sid).add(chunks)})
        return added

    def documents(self, sid):
        return self.stores.get(sid).documents()

    def reset(self, sid):
        self.stores.drop(sid)
        self.graph.checkpointer.delete_thread(sid)

    def stream(self, sid, question):
        from research_agent.service import iter_events
        yield from iter_events(self.graph, sid, question)


@st.cache_resource(show_spinner="Starting the agent...")
def get_backend():
    return RemoteBackend(API_URL) if API_URL else LocalBackend()


try:
    backend = get_backend()
except Exception as exc:
    st.error(f"Could not start the agent: {exc}")
    st.info("Add your API key: locally in a `.env` file (see `.env.example`), "
            "or on Streamlit Cloud under **App settings → Secrets** as `GOOGLE_API_KEY = \"...\"`.")
    st.caption("Secret names found: " + (", ".join(f"`{n}`" for n in SECRET_NAMES) or "none"))
    st.stop()

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:16]
    st.session_state.history = []
sid = st.session_state.session_id


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            if s["type"] == "document":
                st.markdown(f"**[{s['id']}] 📄 {s['source']} — page {s['page']}**")
            else:
                st.markdown(f"**[{s['id']}] 🌐 [{s['title']}]({s['url']})**")
            st.caption(s["text"][:400] + ("…" if len(s["text"]) > 400 else ""))


# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("📚 Your documents")
    files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if files and st.button("Index documents", type="primary", width="stretch"):
        with st.spinner("Chunking and embedding..."):
            try:
                for a in backend.upload(sid, files):
                    st.success(f"{a['file']}: {a['chunks']} chunks indexed")
            except Exception as exc:
                st.error(str(exc))

    try:
        docs = backend.documents(sid)
    except Exception:
        docs = []
        st.error(f"Can't reach the API at {API_URL}. Is `uvicorn api:app` running?")
    for d in docs:
        st.write(f"• **{d['source']}** · {d['pages']} pages · {d['chunks']} chunks")
    if not docs:
        st.caption("No documents yet — questions will be answered from the web.")

    st.divider()
    if st.button("🗑️ New chat", width="stretch"):
        backend.reset(sid)
        st.session_state.clear()
        st.rerun()

# --------------------------------------------------------------------------- chat
st.title("🔎 AI Research Assistant")
st.caption("Multi-agent RAG · LangGraph planner → retriever / web search → writer → reviewer")

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        render_sources(turn.get("sources", []))

if question := st.chat_input("Ask about your documents or anything on the web..."):
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        status = st.status("Thinking...", expanded=False)
        placeholder = st.empty()
        text, final = "", None
        try:
            for event in backend.stream(sid, question):
                if event["type"] == "step":
                    status.update(label=event["label"])
                    status.write(event["label"])
                elif event["type"] == "token":
                    text += event["text"]
                    placeholder.markdown(text + "▌")
                elif event["type"] == "retry":
                    status.write(f"↺ Reviewer asked for a better answer: {event['reason']}")
                    text = ""
                    placeholder.markdown("_Improving the answer..._")
                elif event["type"] == "final":
                    final = event
                elif event["type"] == "error":
                    st.error(event["message"])
        except Exception as exc:
            st.error(f"Request failed: {exc}")

        status.update(label="Done", state="complete")
        if final:
            placeholder.markdown(final["answer"])
            render_sources(final["sources"])
            st.session_state.history.append(
                {"role": "assistant", "content": final["answer"], "sources": final["sources"]})
