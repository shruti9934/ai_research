"""Streamlit chat UI. Talks to the FastAPI backend.

Run with:  streamlit run ui.py   (start the API first: uvicorn api:app)
"""
from __future__ import annotations

import json
import os
import uuid

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="AI Research Assistant", page_icon="🔎", layout="wide")

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
            resp = requests.post(f"{API_URL}/sessions/{sid}/documents",
                                 files=[("files", (f.name, f.getvalue(), "application/pdf")) for f in files],
                                 timeout=300)
        if resp.ok:
            for a in resp.json()["added"]:
                st.success(f"{a['file']}: {a['chunks']} chunks indexed")
        else:
            st.error(resp.json().get("detail", resp.text))

    try:
        docs = requests.get(f"{API_URL}/sessions/{sid}/documents", timeout=10).json()["documents"]
    except requests.RequestException:
        docs = []
        st.error(f"Can't reach the API at {API_URL}. Is `uvicorn api:app` running?")
    for d in docs:
        st.write(f"• **{d['source']}** · {d['pages']} pages · {d['chunks']} chunks")
    if not docs:
        st.caption("No documents yet — questions will be answered from the web.")

    st.divider()
    if st.button("🗑️ New chat", width="stretch"):
        requests.delete(f"{API_URL}/sessions/{sid}", timeout=10)
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
            with requests.post(f"{API_URL}/chat/stream", json={"session_id": sid, "question": question},
                               stream=True, timeout=300) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    event = json.loads(line)
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
        except requests.RequestException as exc:
            st.error(f"Request failed: {exc}")

        status.update(label="Done", state="complete")
        if final:
            placeholder.markdown(final["answer"])
            render_sources(final["sources"])
            st.session_state.history.append(
                {"role": "assistant", "content": final["answer"], "sources": final["sources"]})
