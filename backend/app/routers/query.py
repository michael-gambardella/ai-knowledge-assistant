import logging

from fastapi import APIRouter, Request, HTTPException

from app.config import settings
from app.models.schemas import QueryRequest, QueryResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/query", tags=["Query"])


@router.post("/", response_model=QueryResponse)
async def query(request: Request, body: QueryRequest):
    """
    Core RAG endpoint: embed question → retrieve chunks → generate answer.

    Returns the answer with source citations and per-stage latency metrics
    (retrieval_ms, generation_ms, total_ms).
    """
    top_k = body.top_k if body.top_k is not None else settings.top_k_results

    try:
        response = request.app.state.rag_pipeline.query(
            question=body.question,
            top_k=top_k,
        )
    except Exception as e:
        logger.exception("RAG pipeline error for question=%r", body.question)
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

    return response
