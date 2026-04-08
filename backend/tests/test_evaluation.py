"""
Tests for the evaluation layer: metrics computation and query tracing.

These are pure unit tests — no external services, no API calls, no model loading.
"""

import json
import logging
import pytest

from app.evaluation.metrics import compute_retrieval_metrics, RetrievalMetrics, RELEVANCE_THRESHOLD
from app.evaluation.logger import log_query_trace
from app.models.schemas import SourceChunk, QueryResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_source(score: float, filename: str = "doc.pdf", chunk_index: int = 0) -> SourceChunk:
    return SourceChunk(
        document_id="doc-1",
        filename=filename,
        content="Some content.",
        relevance_score=score,
        chunk_index=chunk_index,
    )


def make_response(
    sources: list[SourceChunk],
    answer: str = "The answer.",
    retrieval_ms: float = 24.0,
    generation_ms: float = 1200.0,
    total_ms: float = 1224.0,
    mean_relevance_score: float = 0.75,
) -> QueryResponse:
    return QueryResponse(
        answer=answer,
        sources=sources,
        retrieval_ms=retrieval_ms,
        generation_ms=generation_ms,
        total_ms=total_ms,
        mean_relevance_score=mean_relevance_score,
    )


# ---------------------------------------------------------------------------
# Tests: compute_retrieval_metrics
# ---------------------------------------------------------------------------

class TestComputeRetrievalMetrics:
    def test_returns_retrieval_metrics_object(self):
        sources = [make_source(0.8)]
        result = compute_retrieval_metrics(sources)
        assert isinstance(result, RetrievalMetrics)

    def test_empty_sources_returns_zero_metrics(self):
        result = compute_retrieval_metrics([])
        assert result.chunk_count == 0
        assert result.mean_relevance == 0.0
        assert result.max_relevance == 0.0
        assert result.min_relevance == 0.0
        assert result.chunks_above_threshold == 0

    def test_chunk_count_matches_sources(self):
        sources = [make_source(0.8), make_source(0.6), make_source(0.9)]
        result = compute_retrieval_metrics(sources)
        assert result.chunk_count == 3

    def test_mean_relevance_is_average(self):
        sources = [make_source(0.8), make_source(0.6)]
        result = compute_retrieval_metrics(sources)
        assert result.mean_relevance == pytest.approx(0.7, abs=0.001)

    def test_max_relevance_is_highest_score(self):
        sources = [make_source(0.5), make_source(0.9), make_source(0.7)]
        result = compute_retrieval_metrics(sources)
        assert result.max_relevance == pytest.approx(0.9, abs=0.001)

    def test_min_relevance_is_lowest_score(self):
        sources = [make_source(0.5), make_source(0.9), make_source(0.7)]
        result = compute_retrieval_metrics(sources)
        assert result.min_relevance == pytest.approx(0.5, abs=0.001)

    def test_chunks_above_threshold_counts_correctly(self):
        # RELEVANCE_THRESHOLD is 0.7
        sources = [
            make_source(0.9),   # above
            make_source(0.7),   # exactly at threshold — counts
            make_source(0.65),  # below
            make_source(0.4),   # below
        ]
        result = compute_retrieval_metrics(sources)
        assert result.chunks_above_threshold == 2

    def test_all_chunks_below_threshold(self):
        sources = [make_source(0.3), make_source(0.5), make_source(0.6)]
        result = compute_retrieval_metrics(sources)
        assert result.chunks_above_threshold == 0

    def test_all_chunks_above_threshold(self):
        sources = [make_source(0.8), make_source(0.75), make_source(0.95)]
        result = compute_retrieval_metrics(sources)
        assert result.chunks_above_threshold == 3

    def test_single_chunk_metrics(self):
        sources = [make_source(0.82)]
        result = compute_retrieval_metrics(sources)
        assert result.chunk_count == 1
        assert result.mean_relevance == pytest.approx(0.82, abs=0.001)
        assert result.max_relevance == result.min_relevance


# ---------------------------------------------------------------------------
# Tests: log_query_trace
# ---------------------------------------------------------------------------

class TestLogQueryTrace:
    def _capture_trace(self, question: str, response: QueryResponse, metrics) -> dict:
        """Run log_query_trace and capture the emitted JSON."""
        records = []

        class _Capture(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        handler = _Capture()
        trace_log = logging.getLogger("rag.query_trace")
        trace_log.addHandler(handler)
        trace_log.setLevel(logging.INFO)

        try:
            log_query_trace(question, response, metrics)
        finally:
            trace_log.removeHandler(handler)

        assert len(records) == 1, f"Expected 1 log record, got {len(records)}"
        return json.loads(records[0])

    def _make_metrics(self, **kwargs):
        from app.evaluation.metrics import RetrievalMetrics
        defaults = dict(chunk_count=3, mean_relevance=0.75, max_relevance=0.9,
                        min_relevance=0.6, chunks_above_threshold=2)
        defaults.update(kwargs)
        return RetrievalMetrics(**defaults)

    def test_emits_one_json_log_line(self):
        sources = [make_source(0.8)]
        response = make_response(sources)
        metrics = self._make_metrics()
        trace = self._capture_trace("What is RAG?", response, metrics)
        assert isinstance(trace, dict)

    def test_trace_has_required_fields(self):
        sources = [make_source(0.8)]
        response = make_response(sources)
        metrics = self._make_metrics()
        trace = self._capture_trace("Test question?", response, metrics)

        required = {"event", "question_preview", "retrieval_ms", "generation_ms",
                    "total_ms", "chunk_count", "mean_relevance", "max_relevance",
                    "min_relevance", "chunks_above_threshold", "no_answer"}
        assert required.issubset(trace.keys())

    def test_event_field_is_rag_query(self):
        sources = [make_source(0.8)]
        response = make_response(sources)
        metrics = self._make_metrics()
        trace = self._capture_trace("Question?", response, metrics)
        assert trace["event"] == "rag_query"

    def test_latency_fields_match_response(self):
        sources = [make_source(0.8)]
        response = make_response(sources, retrieval_ms=30.0, generation_ms=1500.0, total_ms=1530.0)
        metrics = self._make_metrics()
        trace = self._capture_trace("Question?", response, metrics)
        assert trace["retrieval_ms"] == 30.0
        assert trace["generation_ms"] == 1500.0
        assert trace["total_ms"] == 1530.0

    def test_no_answer_is_false_for_normal_response(self):
        sources = [make_source(0.8)]
        response = make_response(sources, answer="The answer is 42.")
        metrics = self._make_metrics()
        trace = self._capture_trace("Question?", response, metrics)
        assert trace["no_answer"] is False

    def test_no_answer_is_true_when_not_enough_info(self):
        sources = []
        response = make_response(
            sources,
            answer="I don't have enough information in the provided documents to answer this question.",
        )
        metrics = self._make_metrics(chunk_count=0, mean_relevance=0.0, max_relevance=0.0,
                                     min_relevance=0.0, chunks_above_threshold=0)
        trace = self._capture_trace("Unanswerable question?", response, metrics)
        assert trace["no_answer"] is True

    def test_long_question_is_truncated(self):
        long_question = "A" * 300
        sources = [make_source(0.8)]
        response = make_response(sources)
        metrics = self._make_metrics()
        trace = self._capture_trace(long_question, response, metrics)
        assert len(trace["question_preview"]) <= 203  # 200 chars + "..."
        assert trace["question_preview"].endswith("...")

    def test_short_question_is_not_truncated(self):
        question = "What is inflation?"
        sources = [make_source(0.8)]
        response = make_response(sources)
        metrics = self._make_metrics()
        trace = self._capture_trace(question, response, metrics)
        assert trace["question_preview"] == question

    def test_metrics_fields_match_computed_metrics(self):
        sources = [make_source(0.8)]
        response = make_response(sources)
        metrics = self._make_metrics(
            chunk_count=4, mean_relevance=0.77, max_relevance=0.91,
            min_relevance=0.55, chunks_above_threshold=3,
        )
        trace = self._capture_trace("Question?", response, metrics)
        assert trace["chunk_count"] == 4
        assert trace["mean_relevance"] == 0.77
        assert trace["max_relevance"] == 0.91
        assert trace["min_relevance"] == 0.55
        assert trace["chunks_above_threshold"] == 3
