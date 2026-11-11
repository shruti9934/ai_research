# 🔎 AI Research Assistant — Multi-Agent RAG System

An agentic research assistant that answers questions from **your uploaded PDFs and the web**, with
**page-level citations**. Built with **LangGraph** (multi-agent workflow), **LangChain**, **FAISS**,
**FastAPI** and **Streamlit**.

## Features

- **Multi-agent LangGraph workflow**: Planner → Retriever / Web-Search → Writer → Reviewer, with shared state and conditional routing
- **RAG pipeline**: PDF parsing → chunking (with page numbers) → embeddings → FAISS cosine-similarity search
- **Citations**: every claim is cited as `[n]` and linked to the file + page (or web URL) it came from
- **Self-reflection**: a Reviewer agent checks the draft against the sources; if it isn't grounded, it writes a better query, re-retrieves and the Writer tries again
- **Tool calling**: the Writer calls a safe calculator tool for any arithmetic instead of guessing
- **Conversation memory**: follow-up questions ("what about 2024?") work, via LangGraph's checkpointer (one thread per chat session)
- **Streaming**: FastAPI streams agent steps and answer tokens as NDJSON; the Streamlit UI renders them live
- **Multi-document upload**, one FAISS index per session, persisted to disk
- **Swappable LLM provider**: Google Gemini (free tier) or OpenAI via one env var
- **Tested without API keys**: a scripted fake LLM drives the full graph + API in `pytest`
- **Dockerised**: `docker compose up` runs the API and UI

## Architecture

```
                 ┌────────────┐
   question ───▶ │  Planner   │  picks a route + rewrites the query (structured output / function calling)
                 └─────┬──────┘
       documents/both  │  web                     direct
      ┌────────────────┼──────────────┬──────────────────┐
      ▼                │              ▼                  │
┌───────────┐   both   │       ┌────────────┐            │
│ Retriever │ ──────────────▶  │ Web Search │            │
│  (FAISS)  │                  │(DuckDuckGo)│            │
└─────┬─────┘                  └─────┬──────┘            │
      └────────────────┬─────────────┘                   │
                       ▼                                 ▼
                 ┌──────────┐  tool calls  ┌────────────┐
                 │  Writer  │ ◀──────────▶ │ Calculator │
                 └────┬─────┘              └────────────┘
                      ▼
                 ┌──────────┐  not grounded → better query → back to Retriever / Web Search
                 │ Reviewer │
                 └────┬─────┘
                      ▼ grounded
                 ┌──────────┐
                 │ Finalize │  keeps only cited sources, saves the turn to memory
                 └──────────┘
```

| Component | File |
|---|---|
| LangGraph state, agents, routing | `research_agent/graph.py` |
| Agent prompts | `research_agent/prompts.py` |
| PDF loading + chunking | `research_agent/documents.py` |
| FAISS vector store (per session, persisted) | `research_agent/vectorstore.py` |
| Tools (web search, calculator) | `research_agent/tools.py` |
| LLM / embeddings provider switch | `research_agent/llm.py` |
| REST + streaming API | `api.py` |
| Chat UI | `ui.py` |
| Tests (fake LLM, no keys needed) | `tests/` |

## Quick start

```bash
git clone https://github.com/shruti9934/ai-research-assistant.git
cd ai-research-assistant
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env        # then paste your API key into .env
```

Get a free Gemini key at <https://aistudio.google.com/apikey> (or use an OpenAI key and set `LLM_PROVIDER=openai`).

Run the API and the UI in two terminals:

```bash
uvicorn api:app --reload          # http://localhost:8000/docs
streamlit run ui.py               # http://localhost:8501
```

Upload one or more PDFs in the sidebar, click **Index documents**, and ask questions. With no documents
uploaded, questions are answered from the web.

### With Docker

```bash
cp .env.example .env   # add your key
docker compose up --build
```

UI → <http://localhost:8501>, API docs → <http://localhost:8000/docs>

## API

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/sessions/{id}/documents` | Upload PDFs (multipart, field `files`) |
| `GET` | `/sessions/{id}/documents` | List indexed documents |
| `POST` | `/chat` | `{"session_id", "question"}` → `{"answer", "sources", "route"}` |
| `POST` | `/chat/stream` | Same input; streams NDJSON events: `step`, `token`, `retry`, `final` |
| `DELETE` | `/sessions/{id}` | Delete a session's index and memory |

## Tests

```bash
pytest -q
```

The tests use a scripted fake chat model and deterministic fake embeddings, so they run offline. They cover
PDF chunking with page numbers, FAISS search and persistence, the calculator's safety, the documents route with a tool
call + reviewer retry + conversation memory, the web fallback, and token streaming.

## How it works (talking points)

- **Why LangGraph instead of a single chain?** The flow has branches (documents vs web vs both) and a loop
  (reviewer → re-retrieve → rewrite). LangGraph models that as a state machine: each agent is a node that reads
  and updates a shared typed state, and conditional edges decide what runs next.
- **Chunking**: `RecursiveCharacterTextSplitter`, 1000 characters with 150 overlap, split *per page* so every
  chunk keeps its page number. That's what makes page-level citations possible.
- **Retrieval**: vectors are L2-normalised and stored in a FAISS `IndexFlatIP`, so the inner product equals cosine
  similarity. Top-k = 5. Each chat session gets its own index, saved to disk.
- **Reducing hallucinations**: the Writer may only use the numbered sources and must cite them. The Reviewer grades
  groundedness with structured output; if it fails, it proposes a better query and the graph loops once more
  (capped by `MAX_REFLECTION_RETRIES`).
- **Tool calling**: the Writer is bound to a `calculator` tool. If the model returns a tool call, the node executes it
  with an AST-based safe evaluator (no `eval`), appends a `ToolMessage` and calls the model again.
- **Memory**: LangGraph's `MemorySaver` checkpointer stores the message history per `thread_id` (= session id).
  The Planner uses the history to rewrite follow-ups into stand-alone queries.
- **Streaming**: `graph.astream(stream_mode=["updates", "messages"])` gives node updates (shown as progress steps)
  and LLM tokens from the Writer node (shown as the answer typing out).

## Possible improvements

- Hybrid search (BM25 + vectors) and a cross-encoder re-ranker
- OCR for scanned PDFs
- Persistent checkpointer (SQLite / Postgres) so memory survives restarts
- Evaluation set with RAGAS-style faithfulness / answer-relevance metrics
