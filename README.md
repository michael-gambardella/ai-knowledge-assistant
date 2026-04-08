# AI Knowledge Assistant

A production-ready RAG (Retrieval-Augmented Generation) system that lets you upload documents and ask questions about them in natural language. Answers are grounded in your documents and include source citations.

Built to demonstrate applied AI engineering skills: LLM integration, vector search, embedding pipelines, evaluation, and production system design.

---

## What It Does

1. **Upload** a PDF or text document
2. **Ask** a question in natural language
3. **Get** a precise answer sourced directly from your documents, with citations showing exactly which chunks were used

---

## How It Works — The RAG Pipeline

Most people think AI apps are just "call ChatGPT." Production systems are more deliberate than that. This app uses a pattern called **RAG (Retrieval-Augmented Generation)**, which solves a core problem: LLMs have a fixed knowledge cutoff and can't answer questions about *your* documents. RAG bridges that gap.

### The Three Stages

```
┌─────────────────────────────────────────────────────────────┐
│  STAGE 1 — INDEXING  (runs when you upload a document)      │
│                                                             │
│  Document → Parse → Chunk → Embed → Store in Vector DB      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  STAGE 2 — RETRIEVAL  (runs when you ask a question)        │
│                                                             │
│  Question → Embed → Vector Search → Top-K relevant chunks   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  STAGE 3 — GENERATION  (final answer)                       │
│                                                             │
│  Question + Retrieved Chunks → LLM → Answer with Sources    │
└─────────────────────────────────────────────────────────────┘
```

### Stage 1: Indexing

**Parse** — Extract raw text from uploaded files (PDF, TXT, MD).

**Chunk** — Split text into overlapping segments (~500 tokens each, 50-token overlap). Chunking with overlap ensures that sentences at chunk boundaries aren't lost — a detail that directly impacts retrieval quality.

**Embed** — Convert each chunk into a high-dimensional numerical vector using a sentence embedding model (`all-MiniLM-L6-v2`). Semantically similar text produces similar vectors — this is what enables meaning-based search rather than keyword matching.

**Store** — Save the vectors and their metadata (source filename, chunk index, raw text) in ChromaDB, a local vector database.

### Stage 2: Retrieval

When you ask a question, the same embedding model converts your question into a vector. ChromaDB performs a **cosine similarity search** across all stored chunk vectors, returning the top-K most semantically similar chunks — regardless of exact keyword overlap. This is how "what causes inflation?" can match a chunk that says "rising prices are driven by..." without sharing any words.

### Stage 3: Generation

The retrieved chunks become the **context** in a carefully engineered prompt sent to Claude. The LLM is instructed to answer using *only* the provided context and to cite sources. This prevents hallucination — the model cannot invent facts because it's constrained to what was retrieved.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Frontend (React)                     │
│   Upload UI  │  Chat Interface  │  Source Citations      │
└─────────────────────┬───────────────────────────────────┘
                      │ HTTP / REST
┌─────────────────────▼───────────────────────────────────┐
│                  FastAPI Backend                         │
│                                                         │
│  POST /documents/upload  →  Indexing Pipeline           │
│  GET  /documents/        →  List indexed documents      │
│  POST /query/            →  RAG query pipeline          │
│  GET  /health            →  Health check                │
└──────────┬──────────────────────────┬───────────────────┘
           │                          │
┌──────────▼──────────┐   ┌──────────▼──────────┐
│   ChromaDB          │   │   Anthropic API      │
│   (Vector Store)    │   │   claude-haiku-4-5   │
│   • chunk vectors   │   │   • answer generation│
│   • metadata        │   │   • source grounding │
└─────────────────────┘   └─────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Backend | Python + FastAPI | Industry standard for AI APIs — async, typed, auto-generates API docs |
| Vector DB | ChromaDB | Local, persistent, no cloud account required — swap to Pinecone for scale |
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) | Runs locally, no API cost, 384-dimensional dense vectors |
| LLM | Anthropic Claude Haiku | Fast, cheap, highly capable for grounded Q&A |
| Chunking | LangChain text splitters | Battle-tested recursive character chunking with configurable overlap |
| Frontend | React + Vite + TypeScript | Fast dev setup, leverages existing full-stack skills |

---

## Project Structure

```
ai-knowledge-assistant/
├── backend/
│   ├── app/
│   │   ├── main.py                    # FastAPI app, lifespan, middleware
│   │   ├── config.py                  # Pydantic Settings — all config from .env
│   │   ├── routers/
│   │   │   ├── health.py              # GET /health — uptime + env check
│   │   │   ├── documents.py           # POST /documents/upload, GET, DELETE
│   │   │   └── query.py               # POST /query — core RAG endpoint
│   │   ├── services/
│   │   │   ├── document_processor.py  # Parse + chunk documents
│   │   │   ├── embeddings.py          # Embedding model wrapper (singleton)
│   │   │   ├── vector_store.py        # ChromaDB interface
│   │   │   ├── rag_pipeline.py        # RAG orchestration: retrieve → generate
│   │   │   └── llm_client.py          # Anthropic API wrapper with retry logic
│   │   ├── models/
│   │   │   └── schemas.py             # Pydantic request/response contracts
│   │   └── evaluation/
│   │       ├── metrics.py             # Relevance scoring, latency tracking
│   │       └── logger.py              # Structured logging + tracing
│   ├── tests/
│   │   ├── test_rag_pipeline.py
│   │   ├── test_retrieval.py
│   │   └── test_evaluation.py
│   ├── requirements.txt
│   ├── .env.example                   # Copy to .env and fill in secrets
│   └── .env                           # Git-ignored — never committed
├── frontend/
│   └── src/
│       ├── components/
│       │   ├── DocumentUpload.tsx
│       │   ├── ChatInterface.tsx
│       │   └── SourceCitations.tsx
│       └── App.tsx
├── .gitignore
└── README.md
```

---

## Build Progress

This project is being built step by step, with each step committed separately to show the development process.

| Step | Focus | Status |
|---|---|---|
| 1 | Backend foundation — FastAPI, config, health check | ✅ Complete |
| 2 | Document pipeline — upload, parse, chunk | ✅ Complete |
| 3 | Embeddings + Vector DB — embed chunks, store and retrieve | ✅ Complete |
| 4 | RAG query pipeline — retrieval + LLM generation | ✅ Complete |
| 5 | Evaluation layer — latency, relevance scoring, logging | 🔄 Next |
| 6 | Frontend — upload UI, chat interface, source citations | ⬜ Planned |
| 7 | Polish — streaming responses, error handling, Docker | ⬜ Planned |

---

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- An [Anthropic API key](https://console.anthropic.com/)

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# Start the server
uvicorn app.main:app --reload
```

API is now running at `http://localhost:8000`
Interactive docs at `http://localhost:8000/docs`

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Frontend is now running at `http://localhost:5173`

---

## Key Concepts for AI Engineers

**Why chunking overlap matters** — If you split a 1,000-word document into 500-word chunks with no overlap, a sentence at the boundary of chunk 1 and chunk 2 is split in half, making both chunks harder to retrieve. A 50-word overlap ensures boundary content appears in full in at least one chunk.

**Why cosine similarity** — Vector search doesn't compare exact values; it measures the *angle* between two vectors. Vectors pointing in the same direction (cosine similarity → 1.0) represent semantically similar content, regardless of length or magnitude.

**Why constrained generation** — Telling the LLM "answer only from the provided context" is the difference between a hallucination-prone chatbot and a reliable document Q&A system. The retrieved chunks act as grounding evidence.

**Why latency is tracked per stage** — Retrieval and generation have very different latency profiles. Measuring them separately lets you identify bottlenecks: slow retrieval means the vector index needs optimization; slow generation means model or prompt changes.

---

## Performance Targets

| Metric | Target |
|---|---|
| Retrieval latency | < 200ms |
| End-to-end response | < 3s |
| Chunk relevance score | > 0.7 cosine similarity |
