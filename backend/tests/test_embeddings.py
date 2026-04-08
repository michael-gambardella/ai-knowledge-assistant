"""
Tests for the embedding service.

These tests load the actual model (not a mock) so they run slower than unit
tests but verify that the model produces correctly shaped, meaningful output.
The model is downloaded once and cached locally — subsequent runs are fast.
"""

import pytest
from app.services.embeddings import EmbeddingService


@pytest.fixture(scope="module")
def embedder():
    """Load the model once for the entire test module — not per test."""
    return EmbeddingService()


class TestEmbeddingShape:
    def test_single_text_returns_one_vector(self, embedder):
        result = embedder.embed(["Hello world"])
        assert len(result) == 1

    def test_batch_returns_one_vector_per_text(self, embedder):
        texts = ["First sentence.", "Second sentence.", "Third sentence."]
        result = embedder.embed(texts)
        assert len(result) == 3

    def test_vector_has_correct_dimensions(self, embedder):
        result = embedder.embed(["Test"])
        # all-MiniLM-L6-v2 produces 384-dim vectors
        assert len(result[0]) == 384

    def test_embed_one_returns_flat_list(self, embedder):
        result = embedder.embed_one("Single text")
        assert isinstance(result, list)
        assert isinstance(result[0], float)

    def test_empty_input_returns_empty_list(self, embedder):
        result = embedder.embed([])
        assert result == []

    def test_vectors_are_plain_python_lists(self, embedder):
        """Vectors must be plain Python lists, not numpy arrays, for JSON serialization."""
        result = embedder.embed(["Test"])
        assert isinstance(result[0], list), "Expected list, not numpy array"


class TestEmbeddingSemantics:
    """
    These tests verify the model captures meaning, not just syntax.
    They demonstrate WHY embeddings are useful for semantic search.
    """

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = sum(x ** 2 for x in a) ** 0.5
        mag_b = sum(x ** 2 for x in b) ** 0.5
        return dot / (mag_a * mag_b)

    def test_similar_sentences_have_high_similarity(self, embedder):
        v1 = embedder.embed_one("The cat sat on the mat.")
        v2 = embedder.embed_one("A cat was sitting on a mat.")
        similarity = self._cosine_similarity(v1, v2)
        assert similarity > 0.85, f"Similar sentences should have high similarity, got {similarity:.3f}"

    def test_unrelated_sentences_have_lower_similarity(self, embedder):
        v1 = embedder.embed_one("The cat sat on the mat.")
        v2 = embedder.embed_one("The stock market crashed today.")
        similarity = self._cosine_similarity(v1, v2)
        assert similarity < 0.75, f"Unrelated sentences should have lower similarity, got {similarity:.3f}"

    def test_semantic_match_beats_keyword_mismatch(self, embedder):
        """
        This is the core value proposition of embeddings over keyword search.
        A question and its answer should be more similar than two sentences
        that share no words but are semantically related.
        """
        query = embedder.embed_one("What causes inflation?")
        answer = embedder.embed_one("Rising prices are driven by increased money supply and demand.")
        unrelated = embedder.embed_one("The football game ended in a draw.")

        sim_answer = self._cosine_similarity(query, answer)
        sim_unrelated = self._cosine_similarity(query, unrelated)

        assert sim_answer > sim_unrelated, (
            f"Semantic answer (sim={sim_answer:.3f}) should score higher "
            f"than unrelated text (sim={sim_unrelated:.3f})"
        )
