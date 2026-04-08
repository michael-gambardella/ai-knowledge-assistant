"""
RAG pipeline: orchestrates the full retrieve → generate flow.

This is the core of the system. Given a natural language question, it:
  1. Embeds the question into a vector using the same model used at index time
  2. Searches ChromaDB for the top-K most semantically similar chunks
  3. Builds a structured prompt containing those chunks as grounding context
  4. Calls Claude, instructing it to answer ONLY from the provided context
  5. Returns the answer with source citations and per-stage latency metrics

WHY PER-STAGE LATENCY:
  Retrieval and generation have fundamentally different latency profiles.
  Retrieval (vector search) is typically < 50ms. Generation (LLM call) is
  typically 500ms–2000ms. Tracking them separately tells you exactly where
  time is going: slow retrieval → optimize the vector index or reduce top_k;
  slow generation → try a smaller model or shorten the prompt.

WHY "ANSWER ONLY FROM CONTEXT":
  Without this constraint the LLM will blend retrieved content with its
  pre-training knowledge, producing answers that sound confident but may
  not be grounded in your documents. The instruction forces the model to
  treat the retrieved chunks as its only evidence — equivalent to an open-book
  exam where you can only cite the provided materials.

WHY NO_ANSWER HANDLING:
  If the top-K chunks don't contain relevant information, we want the model
  to say so explicitly rather than hallucinate. The system prompt instructs
  it to respond with a specific phrase when it can't answer from context.
"""

import logging
import time

from app.models.schemas import QueryResponse, SourceChunk
from app.services.embeddings import EmbeddingService
from app.services.llm_client import LLMClient
from app.services.vector_store import VectorStore, SearchResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a precise document assistant. Your job is to answer questions using ONLY the document excerpts provided in the user message.

Rules:
- Base your answer exclusively on the provided context. Do not use outside knowledge.
- If the context does not contain enough information to answer the question, respond with exactly: "I don't have enough information in the provided documents to answer this question."
- Be concise and direct. Cite which source (by filename) supports each claim when relevant.
- Do not mention that you are using "context" or "excerpts" — just answer naturally."""


def _build_user_message(question: str, chunks: list[SearchResult]) -> str:
    """
    Format retrieved chunks into a structured prompt for Claude.

    Each chunk is labelled with its source filename and relevance score
    so Claude can attribute answers to specific documents.
    """
    context_blocks = []
    for i, chunk in enumerate(chunks, 1):
        context_blocks.append(
            f"[Source {i}: {chunk.filename} | relevance: {chunk.relevance_score:.2f}]\n{chunk.content}"
        )

    context = "\n\n---\n\n".join(context_blocks)
    return f"Context from documents:\n\n{context}\n\n---\n\nQuestion: {question}"


class RAGPipeline:
    """
    Orchestrates the retrieve → generate pipeline.

    Holds references to all three services (embedder, vector store, LLM).
    Each call to `query()` is stateless — safe for concurrent requests.
    """

    def __init__(self, embedding_service: EmbeddingService, vector_store: VectorStore, llm_client: LLMClient):
        self._embedder = embedding_service
        self._vector_store = vector_store
        self._llm = llm_client

    def query(self, question: str, top_k: int) -> QueryResponse:
        """
        Run the full RAG pipeline for a question.

        Args:
            question: the user's natural language question
            top_k:    number of chunks to retrieve from the vector store

        Returns:
            QueryResponse with answer, source citations, and latency metrics
        """
        total_start = time.perf_counter()

        # --- Stage 1: Retrieval ---
        retrieval_start = time.perf_counter()
        query_embedding = self._embedder.embed_one(question)
        chunks = self._vector_store.search(query_embedding, top_k=top_k)
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        logger.info(
            "Retrieval complete — top_k=%d chunks_found=%d retrieval_ms=%.1f",
            top_k,
            len(chunks),
            retrieval_ms,
        )

        # --- Stage 2: Generation ---
        generation_start = time.perf_counter()

        if not chunks:
            # No documents indexed — skip the LLM call entirely
            answer = "I don't have enough information in the provided documents to answer this question."
            logger.warning("No chunks retrieved — vector store may be empty")
        else:
            user_message = _build_user_message(question, chunks)
            answer = self._llm.complete(SYSTEM_PROMPT, user_message)

        generation_ms = (time.perf_counter() - generation_start) * 1000
        total_ms = (time.perf_counter() - total_start) * 1000

        logger.info(
            "Generation complete — generation_ms=%.1f total_ms=%.1f",
            generation_ms,
            total_ms,
        )

        sources = [
            SourceChunk(
                document_id=c.document_id,
                filename=c.filename,
                content=c.content,
                relevance_score=c.relevance_score,
                chunk_index=c.chunk_index,
            )
            for c in chunks
        ]

        return QueryResponse(
            answer=answer,
            sources=sources,
            retrieval_ms=round(retrieval_ms, 1),
            generation_ms=round(generation_ms, 1),
            total_ms=round(total_ms, 1),
        )
