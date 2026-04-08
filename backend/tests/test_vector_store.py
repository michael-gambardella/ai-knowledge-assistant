"""
Tests for the vector store (ChromaDB interface).

Uses a temporary ChromaDB directory so tests are fully isolated from the
development database and clean up after themselves.
"""

import pytest
from unittest.mock import patch
from app.services.vector_store import VectorStore, SearchResult
from app.services.document_processor import DocumentChunk


def make_chunk(document_id: str, filename: str, index: int, content: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"{document_id}-chunk-{index}",
        document_id=document_id,
        filename=filename,
        content=content,
        chunk_index=index,
        total_chunks=3,
    )


def make_embedding(seed: float, dims: int = 384) -> list[float]:
    """Deterministic fake embedding for testing."""
    import math
    raw = [math.sin(seed + i * 0.1) for i in range(dims)]
    magnitude = sum(x ** 2 for x in raw) ** 0.5
    return [x / magnitude for x in raw]


@pytest.fixture
def store(tmp_path):
    """A VectorStore backed by a temp directory — isolated per test."""
    with patch("app.services.vector_store.settings") as mock_settings:
        mock_settings.chroma_persist_dir = str(tmp_path / "chroma")
        mock_settings.chroma_collection_name = "test_collection"
        yield VectorStore()


class TestAddAndCount:
    def test_empty_store_has_zero_chunks(self, store):
        assert store.count() == 0

    def test_add_chunks_increases_count(self, store):
        chunks = [make_chunk("doc-1", "file.txt", i, f"Content {i}") for i in range(3)]
        embeddings = [make_embedding(i) for i in range(3)]
        store.add_chunks(chunks, embeddings)
        assert store.count() == 3

    def test_add_empty_list_is_safe(self, store):
        store.add_chunks([], [])
        assert store.count() == 0

    def test_multiple_documents_stored_separately(self, store):
        for doc_num in range(3):
            chunks = [make_chunk(f"doc-{doc_num}", f"file{doc_num}.txt", 0, f"Document {doc_num} content")]
            store.add_chunks(chunks, [make_embedding(doc_num * 10)])
        assert store.count() == 3


class TestSearch:
    def test_search_returns_results(self, store):
        chunks = [make_chunk("doc-1", "file.txt", i, f"Sentence number {i}") for i in range(5)]
        embeddings = [make_embedding(i * 2.0) for i in range(5)]
        store.add_chunks(chunks, embeddings)

        query_embedding = make_embedding(0.0)  # most similar to chunk 0
        results = store.search(query_embedding, top_k=3)
        assert len(results) == 3

    def test_search_returns_search_result_objects(self, store):
        chunks = [make_chunk("doc-1", "file.txt", 0, "Sample text")]
        store.add_chunks(chunks, [make_embedding(1.0)])

        results = store.search(make_embedding(1.0), top_k=1)
        assert isinstance(results[0], SearchResult)

    def test_search_result_has_all_fields(self, store):
        chunk = make_chunk("doc-abc", "myfile.txt", 2, "Important content here")
        store.add_chunks([chunk], [make_embedding(5.0)])

        results = store.search(make_embedding(5.0), top_k=1)
        r = results[0]
        assert r.document_id == "doc-abc"
        assert r.filename == "myfile.txt"
        assert r.chunk_index == 2
        assert r.content == "Important content here"
        assert 0.0 <= r.relevance_score <= 1.0

    def test_search_on_empty_store_returns_empty(self, store):
        results = store.search(make_embedding(1.0), top_k=5)
        assert results == []

    def test_top_k_limits_results(self, store):
        chunks = [make_chunk("doc-1", "file.txt", i, f"Content {i}") for i in range(10)]
        embeddings = [make_embedding(float(i)) for i in range(10)]
        store.add_chunks(chunks, embeddings)

        results = store.search(make_embedding(0.0), top_k=3)
        assert len(results) == 3

    def test_relevance_score_is_between_zero_and_one(self, store):
        chunks = [make_chunk("doc-1", "file.txt", i, f"Text {i}") for i in range(5)]
        embeddings = [make_embedding(float(i)) for i in range(5)]
        store.add_chunks(chunks, embeddings)

        results = store.search(make_embedding(0.0), top_k=5)
        for r in results:
            assert 0.0 <= r.relevance_score <= 1.0, f"Score out of range: {r.relevance_score}"


class TestDelete:
    def test_delete_removes_chunks_for_document(self, store):
        chunks = [make_chunk("doc-to-delete", "delete_me.txt", i, f"Content {i}") for i in range(3)]
        embeddings = [make_embedding(float(i)) for i in range(3)]
        store.add_chunks(chunks, embeddings)
        assert store.count() == 3

        deleted = store.delete_document("doc-to-delete")
        assert deleted == 3
        assert store.count() == 0

    def test_delete_only_removes_target_document(self, store):
        for doc_id in ["doc-keep", "doc-delete"]:
            chunks = [make_chunk(doc_id, f"{doc_id}.txt", i, f"Content {i}") for i in range(2)]
            embeddings = [make_embedding(float(i)) for i in range(2)]
            store.add_chunks(chunks, embeddings)

        store.delete_document("doc-delete")
        assert store.count() == 2  # doc-keep's chunks remain

    def test_delete_nonexistent_document_returns_zero(self, store):
        deleted = store.delete_document("does-not-exist")
        assert deleted == 0
