"""
Retrieval quality metrics for the RAG pipeline.

After each query, we compute these metrics from the returned source chunks.
They answer a question the raw response doesn't: was the retrieval actually good?

METRICS EXPLAINED:

  mean_relevance — average cosine similarity across all retrieved chunks.
    A high mean (> 0.7) means the vector store consistently found relevant
    content. A low mean suggests the question doesn't match what's indexed,
    or the document set doesn't cover the topic.

  max_relevance — the best chunk's score.
    If max is low, there's nothing relevant in the index for this question.
    If max is high but mean is low, you got one good chunk and several weak
    ones — consider reducing top_k.

  min_relevance — the weakest chunk's score.
    If min is very low (< 0.3), you're sending irrelevant context to the LLM,
    which increases prompt length and can confuse generation. Filtering chunks
    below a relevance threshold before generation would help here.

  chunks_above_threshold — count of chunks meeting the target (0.7 cosine
    similarity, from the README performance targets). This is the primary
    quality gate: if 0 chunks clear the threshold, the answer is likely weak
    regardless of what the LLM says.

WHY NOT MEASURE ANSWER QUALITY HERE:
  Evaluating whether the *answer* is correct requires a reference answer to
  compare against — that's a different concern (offline evaluation with labeled
  test sets, not per-request runtime metrics). What we *can* measure at runtime
  is retrieval quality, which is a leading indicator of answer quality.
"""

from dataclasses import dataclass

from app.models.schemas import SourceChunk

RELEVANCE_THRESHOLD = 0.7  # from README performance targets


@dataclass
class RetrievalMetrics:
    """Quality metrics for a single RAG query's retrieval stage."""
    chunk_count: int
    mean_relevance: float     # average cosine similarity across retrieved chunks
    max_relevance: float      # highest-scoring chunk
    min_relevance: float      # lowest-scoring chunk
    chunks_above_threshold: int  # chunks with relevance >= RELEVANCE_THRESHOLD


def compute_retrieval_metrics(sources: list[SourceChunk]) -> RetrievalMetrics:
    """
    Compute retrieval quality metrics from the sources in a QueryResponse.

    Returns zero-value metrics when no chunks were retrieved (empty store).

    Args:
        sources: the SourceChunk list from QueryResponse.sources

    Returns:
        RetrievalMetrics with computed quality stats
    """
    if not sources:
        return RetrievalMetrics(
            chunk_count=0,
            mean_relevance=0.0,
            max_relevance=0.0,
            min_relevance=0.0,
            chunks_above_threshold=0,
        )

    scores = [s.relevance_score for s in sources]

    return RetrievalMetrics(
        chunk_count=len(scores),
        mean_relevance=round(sum(scores) / len(scores), 4),
        max_relevance=round(max(scores), 4),
        min_relevance=round(min(scores), 4),
        chunks_above_threshold=sum(1 for s in scores if s >= RELEVANCE_THRESHOLD),
    )
