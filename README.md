# RAG Chat App

A full-stack Retrieval-Augmented Generation chat application. Upload PDFs, ask questions about them, and get grounded answers with cited sources — powered by a fairly advanced retrieval/reasoning pipeline rather than a plain "embed and search" setup.

**Stack:** FastAPI + LangChain + LangGraph (backend) · Next.js 16 + React 19 (frontend) · ChromaDB (vectors) · PostgreSQL (chat history) · Groq (`llama-3.1-8b-instant`) as the LLM · `all-MiniLM-L6-v2` for embeddings.

## What makes the pipeline notable

- **Hybrid retrieval** — combines vector similarity (Chroma) with BM25 keyword search, blended via a tunable `alpha` weight.
- **Query rewriting** — vague queries are expanded into multiple phrasings via the LLM before retrieval, then results are merged (multi-query retrieval).
- **Multi-hop reasoning** — complex questions are auto-detected and decomposed into sub-questions, retrieved sequentially, with entities extracted along the way.
- **Reflection / self-check** — generated answers are verified against retrieved context; low-confidence or unsupported answers trigger a regeneration attempt to reduce hallucination.
- **Streaming responses** — `/query/stream` returns Server-Sent Events with metadata (confidence, hallucination risk, sources) followed by the answer.
- **Built-in evaluation** — every query is logged with latency, cost, retrieval precision/recall, and hallucination-rate metrics, exposed via `/metrics`.
- **Multi-turn chat** — sessions and message history are persisted in PostgreSQL and fed back into the LLM as conversation context.

## Project structure

```
backend/
  app/
    main.py          FastAPI app & routes
    vector_store.py  Ingestion, hybrid/multi-hop retrieval, generation, reflection
    evaluation.py     Metrics tracking & logging
    database.py       SQLAlchemy models (chat sessions/messages)
  test_*.py, verify_*.py   Ad-hoc test/verification scripts
frontend/
  app/
    page.tsx                 Main chat UI
    components/               ChatMessage, SourcesPanel, ChunksPreview
    hooks/useStreamingQuery.ts  SSE streaming hook
```

## API overview

| Endpoint | Description |
|---|---|
| `POST /upload` | Upload a PDF, chunk it, and add it to the vector store |
| `POST /query` | Ask a question, get an answer + sources (non-streaming) |
| `POST /query/stream` | Same, but streamed via SSE |
| `GET /documents` | List all ingested documents |
| `POST /delete-document` | Remove a document by source name |
| `POST /clear-db` | Wipe the vector database |
| `POST /session` / `GET /session/{id}/history` | Chat session management |
| `GET /metrics`, `GET /metrics/summary`, `GET /logs/export` | Evaluation metrics |

## Setup

### Backend
```bash
cd backend
pip install -r requirements.txt
```
Create a `.env` file in `backend/`:
```
GROQ_API_KEY=your_groq_api_key
DATABASE_URL=postgresql://user:password@localhost:5432/rag_db
```
Run the API:
```bash
uvicorn app.main:app --reload
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```
Runs on `http://localhost:3000` and expects the backend at `http://localhost:8000` (CORS is pre-configured for local dev).

## Notes

- Requires a running PostgreSQL instance for chat history (Chroma vector data is stored locally on disk under `backend/chroma_db`).
- Only PDF upload is currently supported for ingestion.
- The `backend/test_*.py` and `verify_*.py` scripts are standalone manual test/debug scripts, not a pytest suite.