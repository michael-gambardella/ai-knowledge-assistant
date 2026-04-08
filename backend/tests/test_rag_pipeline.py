"""
Tests for the RAG pipeline.

The LLM client is mocked in all tests — these tests verify that the pipeline
orchestrates retrieval and generation correctly, not that the LLM produces
good answers (that would require a live API key and is an evaluation concern).

The vector store and embedder are also mocked so these tests are pure unit
tests: fast, deterministic, and require no external services or models.
"""

import pytest
from unittest.mock import MagicMock

from app.services.rag_pipeline import RAGPipeline, _build_user_message
from app.services.vector_store import SearchResult
from app.models.schemas import QueryResponse


def make_search_result(doc_id: str = "doc-1", filename: str = "test.pdf",
                       content: str = "Some relevant text.", score: float = 0.85,
                       chunk_index: int = 0) -> SearchResult:
    return SearchResult(
        chunk_id=f"{doc_id}-chunk-{chunk_index}",
        document_id=doc_id,
        filename=filename,
        content=content,
        chunk_index=chunk_index,
        relevance_score=score,
    )


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    embedder.embed_one.return_value = [0.1] * 384
    return embedder


@pytest.fixture
def mock_vector_store():
    store = MagicMock()
    store.search.return_value = [
        make_search_result(content="Inflation is caused by increased money supply.", score=0.92),
        make_search_result(content="Central banks raise interest rates to fight inflation.", score=0.78),
    ]
    return store


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.complete.return_value = "Inflation is caused by increased money supply."
    return llm


@pytest.fixture
def pipeline(mock_embedder, mock_vector_store, mock_llm):
    return RAGPipeline(
        embedding_service=mock_embedder,
        vector_store=mock_vector_store,
        llm_client=mock_llm,
    )


class TestRAGPipelineQuery:
    def test_returns_query_response(self, pipeline):
        result = pipeline.query("What causes inflation?", top_k=5)
        assert isinstance(result, QueryResponse)

    def test_answer_is_populated(self, pipeline, mock_llm):
        mock_llm.complete.return_value = "Inflation is driven by money supply."
        result = pipeline.query("What causes inflation?", top_k=5)
        assert result.answer == "Inflation is driven by money supply."

    def test_sources_match_retrieved_chunks(self, pipeline, mock_vector_store):
        mock_vector_store.search.return_value = [
            make_search_result(doc_id="doc-A", filename="finance.pdf", content="Text A", score=0.9),
            make_search_result(doc_id="doc-B", filename="economics.pdf", content="Text B", score=0.7),
        ]
        result = pipeline.query("Test question?", top_k=2)
        assert len(result.sources) == 2
        assert result.sources[0].filename == "finance.pdf"
        assert result.sources[1].filename == "economics.pdf"

    def test_source_chunk_fields_are_correct(self, pipeline, mock_vector_store):
        mock_vector_store.search.return_value = [
            make_search_result(doc_id="doc-1", filename="report.pdf", content="Important fact.", score=0.88, chunk_index=3),
        ]
        result = pipeline.query("What is the key finding?", top_k=1)
        source = result.sources[0]
        assert source.document_id == "doc-1"
        assert source.filename == "report.pdf"
        assert source.content == "Important fact."
        assert source.relevance_score == 0.88
        assert source.chunk_index == 3

    def test_latency_fields_are_non_negative(self, pipeline):
        result = pipeline.query("Any question?", top_k=5)
        assert result.retrieval_ms >= 0
        assert result.generation_ms >= 0
        assert result.total_ms >= 0

    def test_total_ms_is_sum_of_stages(self, pipeline):
        result = pipeline.query("Any question?", top_k=5)
        # total_ms should be >= sum of stages (it also includes overhead)
        assert result.total_ms >= result.retrieval_ms + result.generation_ms - 1  # -1ms tolerance for float rounding

    def test_embedder_called_with_question(self, pipeline, mock_embedder):
        pipeline.query("What is RAG?", top_k=3)
        mock_embedder.embed_one.assert_called_once_with("What is RAG?")

    def test_vector_store_called_with_correct_top_k(self, pipeline, mock_vector_store):
        pipeline.query("Test?", top_k=7)
        mock_vector_store.search.assert_called_once()
        _, kwargs = mock_vector_store.search.call_args
        assert kwargs["top_k"] == 7

    def test_llm_called_with_system_and_user_message(self, pipeline, mock_llm):
        pipeline.query("What is the capital of France?", top_k=3)
        mock_llm.complete.assert_called_once()
        args = mock_llm.complete.call_args[0]
        system_prompt, user_message = args
        assert "context" in system_prompt.lower() or "document" in system_prompt.lower()
        assert "What is the capital of France?" in user_message


class TestEmptyVectorStore:
    def test_empty_store_skips_llm_call(self, mock_embedder, mock_llm):
        empty_store = MagicMock()
        empty_store.search.return_value = []
        pipeline = RAGPipeline(mock_embedder, empty_store, mock_llm)

        result = pipeline.query("Any question?", top_k=5)

        mock_llm.complete.assert_not_called()
        assert "don't have enough information" in result.answer.lower()

    def test_empty_store_returns_empty_sources(self, mock_embedder, mock_llm):
        empty_store = MagicMock()
        empty_store.search.return_value = []
        pipeline = RAGPipeline(mock_embedder, empty_store, mock_llm)

        result = pipeline.query("Any question?", top_k=5)
        assert result.sources == []


class TestBuildUserMessage:
    def test_question_appears_in_message(self):
        chunks = [make_search_result(content="Some context.")]
        msg = _build_user_message("What is inflation?", chunks)
        assert "What is inflation?" in msg

    def test_chunk_content_appears_in_message(self):
        chunks = [make_search_result(content="Prices rise due to demand.")]
        msg = _build_user_message("Why do prices rise?", chunks)
        assert "Prices rise due to demand." in msg

    def test_filename_appears_in_message(self):
        chunks = [make_search_result(filename="economics_101.pdf", content="Some text.")]
        msg = _build_user_message("Question?", chunks)
        assert "economics_101.pdf" in msg

    def test_multiple_chunks_all_included(self):
        chunks = [
            make_search_result(content="First chunk content."),
            make_search_result(content="Second chunk content."),
            make_search_result(content="Third chunk content."),
        ]
        msg = _build_user_message("Question?", chunks)
        assert "First chunk content." in msg
        assert "Second chunk content." in msg
        assert "Third chunk content." in msg
