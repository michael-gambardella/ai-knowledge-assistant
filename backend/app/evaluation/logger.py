"""
Structured query tracing for the RAG pipeline.

WHY STRUCTURED LOGGING:
  Standard Python logging emits human-readable lines like:
    "2026-04-08 | INFO | Retrieval complete — chunks_found=5 retrieval_ms=24.1"

  That's fine for reading in a terminal. But in production you want to *query*
  your logs: "show me all queries where mean_relevance < 0.5" or "what's the
  p95 generation latency this week?" Human-readable strings don't support that.

  Structured logging emits a single JSON object per event. Log aggregators
  (Datadog, CloudWatch, Grafana Loki) index every field, making the queries above
  trivial. One JSON line per query also means a complete trace is atomic —
  you don't have to join 4 separate log lines to reconstruct what happened.

WHAT GETS LOGGED:
  - The question (truncated for privacy/size)
  - Per-stage latency: retrieval_ms, generation_ms, total_ms
  - Retrieval quality: chunk_count, mean/max relevance, chunks above threshold
  - Whether the answer was grounded or a "no information" response

  Notably absent: the full answer text. Log pipelines aren't the right place
  for long text — they bloat storage and make grepping noisy. The answer is
  already returned to the caller; the trace captures the *quality signals*.
"""

import json
import logging

from app.evaluation.metrics import RetrievalMetrics
from app.models.schemas import QueryResponse

# Separate logger name so operators can route query traces to their own sink
# (e.g., a dedicated file or log stream) independently of the main app logs.
trace_logger = logging.getLogger("rag.query_trace")

_MAX_QUESTION_LOG_LENGTH = 200  # truncate long questions in logs


def log_query_trace(
    question: str,
    response: QueryResponse,
    metrics: RetrievalMetrics,
) -> None:
    """
    Emit a single structured JSON trace for a completed RAG query.

    This is called once per query, after the pipeline returns. The trace
    captures enough signal to answer: "was this query handled well?"

    Args:
        question: the user's original question
        response: the QueryResponse returned by the pipeline
        metrics:  retrieval quality metrics computed from response.sources
    """
    # Truncate the question so log lines don't balloon on long inputs
    question_preview = question[:_MAX_QUESTION_LOG_LENGTH]
    if len(question) > _MAX_QUESTION_LOG_LENGTH:
        question_preview += "..."

    no_answer = "don't have enough information" in response.answer.lower()

    trace = {
        "event": "rag_query",
        "question_preview": question_preview,
        "retrieval_ms": response.retrieval_ms,
        "generation_ms": response.generation_ms,
        "total_ms": response.total_ms,
        "chunk_count": metrics.chunk_count,
        "mean_relevance": metrics.mean_relevance,
        "max_relevance": metrics.max_relevance,
        "min_relevance": metrics.min_relevance,
        "chunks_above_threshold": metrics.chunks_above_threshold,
        "no_answer": no_answer,
    }

    trace_logger.info(json.dumps(trace))
