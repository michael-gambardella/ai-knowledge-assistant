"""
Vector store: persists chunk embeddings and retrieves the most relevant ones.

ChromaDB is an embedded vector database — it runs in the same process as the
FastAPI app, persists data to disk, and requires no separate server. This makes
it ideal for development and small-to-medium production workloads. The interface
here is designed so that swapping ChromaDB for Pinecone or Weaviate later only
requires changing this file.

HOW VECTOR SEARCH WORKS:
  1. Each chunk's text was converted to a 384-dimension vector at upload time.
  2. At query time, the question is converted to a vector using the same model.
  3. ChromaDB computes the cosine distance between the query vector and every
     stored chunk vector. Lower distance = more similar meaning.
  4. It returns the top-K closest chunks along with their distance scores.
  5. We convert distance → similarity: similarity = 1 - distance
     (ChromaDB returns cosine *distance*, not cosine *similarity*.)

WHY PRE-COMPUTED EMBEDDINGS:
  ChromaDB supports pluggable embedding functions (it can call the model itself).
  We deliberately pre-compute embeddings in the EmbeddingService and pass raw
  vectors to ChromaDB. This separates concerns: embedding logic lives in one
  place, storage logic lives here. It also makes the embedding step independently
  testable and avoids ChromaDB needing a reference to the model.
"""

import logging
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings
from app.services.document_processor import DocumentChunk

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A retrieved chunk plus its relevance score. Returned to the query layer."""
    chunk_id: str
    document_id: str
    filename: str
    content: str
    chunk_index: int
    relevance_score: float  # cosine similarity: 0.0 (unrelated) → 1.0 (identical)


class VectorStore:
    """
    ChromaDB-backed vector store.

    One ChromaDB collection holds all chunks from all documents. Documents are
    distinguished by their document_id stored in chunk metadata, which allows
    us to filter or delete by document without affecting other documents.
    """

    def __init__(self):
        # PersistentClient writes the vector index to disk at chroma_persist_dir.
        # On restart, ChromaDB loads the index back from disk — uploads survive
        # server restarts without needing to re-index documents.
        self._client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # get_or_create_collection is idempotent — safe to call on every startup.
        # cosine distance is the right metric for sentence embeddings.
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            "VectorStore ready — collection='%s' existing_chunks=%d persist_dir='%s'",
            settings.chroma_collection_name,
            self._collection.count(),
            settings.chroma_persist_dir,
        )

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        """
        Store a batch of chunks with their pre-computed embeddings.

        ChromaDB requires parallel lists for ids, embeddings, documents
        (the raw text), and metadatas. The metadata dict is what comes back
        at query time — it's how we know which file a retrieved chunk came from.

        Args:
            chunks:     DocumentChunk objects from the document processor
            embeddings: one float vector per chunk, same order as chunks
        """
        if not chunks:
            return

        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.content for c in chunks],
            metadatas=[
                {
                    "document_id": c.document_id,
                    "filename": c.filename,
                    "chunk_index": c.chunk_index,
                    "total_chunks": c.total_chunks,
                }
                for c in chunks
            ],
        )
        logger.info(
            "Stored %d chunks for document_id=%s filename=%s",
            len(chunks),
            chunks[0].document_id,
            chunks[0].filename,
        )

    def delete_document(self, document_id: str) -> int:
        """
        Remove all chunks belonging to a document from the vector index.

        ChromaDB's where filter lets us delete by metadata field without
        knowing individual chunk IDs.

        Returns:
            Number of chunks deleted.
        """
        # Count before deletion so we can return a meaningful number
        existing = self._collection.get(where={"document_id": document_id})
        count = len(existing["ids"])

        if count > 0:
            self._collection.delete(where={"document_id": document_id})
            logger.info("Deleted %d chunks for document_id=%s", count, document_id)
        else:
            logger.warning("No chunks found for document_id=%s — nothing deleted", document_id)

        return count

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def search(self, query_embedding: list[float], top_k: int) -> list[SearchResult]:
        """
        Find the top-K chunks most semantically similar to the query.

        ChromaDB returns results sorted by cosine distance (ascending).
        We convert distance to similarity so scores are intuitive:
          1.0 = perfect match, 0.0 = completely unrelated.

        Args:
            query_embedding: vector produced by EmbeddingService.embed_one()
            top_k:           number of results to return

        Returns:
            List of SearchResult, sorted by relevance_score descending.
        """
        total_chunks = self._collection.count()
        if total_chunks == 0:
            logger.warning("Vector store is empty — no documents have been indexed yet")
            return []

        # Never ask for more results than exist
        n_results = min(top_k, total_chunks)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        search_results = []
        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i]
            metadata = results["metadatas"][0][i]

            # cosine distance ∈ [0, 2]; similarity = 1 - distance → range [-1, 1]
            # We clamp to [0, 1] for user-facing scores: negative similarity means
            # "opposite direction" which is effectively "not relevant" (score = 0).
            similarity = round(max(0.0, 1.0 - distance), 4)

            search_results.append(
                SearchResult(
                    chunk_id=results["ids"][0][i],
                    document_id=metadata["document_id"],
                    filename=metadata["filename"],
                    content=results["documents"][0][i],
                    chunk_index=metadata["chunk_index"],
                    relevance_score=similarity,
                )
            )

        # Already sorted by ChromaDB (lowest distance first = highest similarity first)
        return search_results

    def count(self) -> int:
        """Total number of chunks currently stored."""
        return self._collection.count()
