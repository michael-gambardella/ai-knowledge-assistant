"""
Embedding service: converts text into dense numerical vectors.

This is where Natural Language Processing happens at a fundamental level.
An embedding model maps text of any length to a fixed-size vector (e.g. 384
numbers for all-MiniLM-L6-v2). Texts that are semantically similar end up
with vectors that point in similar directions — enabling meaning-based search.

WHY THIS MODEL:
  all-MiniLM-L6-v2 is a sentence transformer trained specifically to produce
  high-quality sentence/paragraph embeddings for semantic similarity tasks.
  It's fast (runs on CPU), small (~80 MB), and produces 384-dimensional vectors
  that perform well on retrieval benchmarks. In production you might swap this
  for a larger model or a hosted API (e.g. OpenAI text-embedding-3-small), but
  the interface stays identical — that's the point of this wrapper.

WHAT IS COSINE SIMILARITY:
  We represent a sentence as a point in 384-dimensional space. Cosine similarity
  measures the angle between two points (vectors). An angle of 0° → similarity 1.0
  (identical meaning). An angle of 90° → similarity 0.0 (unrelated). An angle of
  180° → similarity -1.0 (opposite meaning). This is more useful than Euclidean
  distance because it's length-invariant — a long document and a short document
  on the same topic get similar scores.
"""

import logging
import time

from sentence_transformers import SentenceTransformer

from app.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Thin wrapper around SentenceTransformer.

    Loaded once at startup (in main.py lifespan). Calling embed() is then a fast
    in-process operation — no network call, no startup cost per request.
    """

    def __init__(self):
        logger.info("Loading embedding model '%s' — this may take a moment on first run...", settings.embedding_model)
        t0 = time.perf_counter()

        # SentenceTransformer downloads the model from HuggingFace Hub on first
        # use and caches it locally. Subsequent starts load from the local cache.
        self._model = SentenceTransformer(settings.embedding_model)

        elapsed = time.perf_counter() - t0
        self.dimensions = self._model.get_sentence_embedding_dimension()
        logger.info(
            "Embedding model loaded in %.2fs — model=%s dimensions=%d",
            elapsed,
            settings.embedding_model,
            self.dimensions,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts.

        Batching is important for performance: embedding 100 chunks one at a time
        is much slower than embedding them all in one call because the model can
        process them in parallel on the hardware level.

        Args:
            texts: list of strings to embed

        Returns:
            list of float vectors, one per input text
        """
        if not texts:
            return []

        # show_progress_bar=False keeps logs clean during uploads
        vectors = self._model.encode(texts, show_progress_bar=False)

        # .encode() returns a numpy array; convert to plain Python lists so
        # they're JSON-serialisable and ChromaDB-compatible
        return [v.tolist() for v in vectors]

    def embed_one(self, text: str) -> list[float]:
        """
        Convenience method for embedding a single string (e.g. a query).

        At query time we only need one embedding — this avoids the caller
        having to wrap/unwrap a single-item list.
        """
        return self.embed([text])[0]
