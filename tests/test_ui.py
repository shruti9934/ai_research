"""Runs the Streamlit UI in local mode with the fake LLM (no API key needed)."""
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage
from streamlit.testing.v1 import AppTest

import research_agent.llm as llm_module
from tests.fakes import ScriptedChatModel


def test_ui_local_mode_answers_with_sources(monkeypatch, tmp_path):
    monkeypatch.delenv("API_URL", raising=False)
    monkeypatch.setenv("INDEX_DIR", str(tmp_path))
    fake = ScriptedChatModel(plan={"route": "web", "web_query": "acme"},
                             writer=lambda m: AIMessage("Acme builds rockets [1]."), calls={})
    monkeypatch.setattr(llm_module, "get_chat_model", lambda: fake)
    monkeypatch.setattr(llm_module, "get_embeddings", lambda: DeterministicFakeEmbedding(size=16))
    monkeypatch.setattr("research_agent.graph.ddg_search", lambda q, n: [
        {"title": "Acme", "url": "https://example.com/acme", "snippet": "Acme builds rockets."}])

    at = AppTest.from_file("../ui.py", default_timeout=60)
    at.run()
    assert not at.exception
    at.chat_input[0].set_value("What does Acme do?").run()
    assert not at.exception
    history = at.session_state["history"]
    assert history[-1]["content"] == "Acme builds rockets [1]."
    assert history[-1]["sources"][0]["url"] == "https://example.com/acme"
