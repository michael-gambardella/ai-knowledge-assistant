import logging

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse

from app.config import settings
from app.evaluation.metrics import compute_retrieval_metrics
from app.evaluation.logger import log_query_trace
from app.models.schemas import QueryRequest, QueryResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/query", tags=["Query"])


@router.post("/", response_model=QueryResponse)
async def query(request: Request, body: QueryRequest):
    """
    RAG endpoint (non-streaming): returns the complete answer in one response.
    Use /query/stream for real-time token delivery.
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

    try:
        metrics = compute_retrieval_metrics(response.sources)
        log_query_trace(body.question, response, metrics)
    except Exception:
        logger.exception("Evaluation layer error — query response unaffected")

    return response


@router.post("/stream")
async def query_stream(request: Request, body: QueryRequest):
    """
    RAG endpoint (streaming): delivers tokens via Server-Sent Events as they
    arrive from Claude, so the UI can render the answer in real time.

    SSE event sequence:
      data: {"type": "sources", "sources": [...], "retrieval_ms": N}
      data: {"type": "token",   "text": "..."}   (one per token)
      data: {"type": "done",    "generation_ms": N, "total_ms": N, "mean_relevance_score": N}
      data: {"type": "error",   "message": "..."}  (only on failure)

    WHY SSE OVER WEBSOCKETS:
      SSE is unidirectional (server → client) and works over a plain HTTP
      connection. WebSockets add bidirectional overhead we don't need. SSE
      also handles reconnection automatically in the browser and works through
      standard proxies and load balancers without special configuration.
    """
    top_k = body.top_k if body.top_k is not None else settings.top_k_results

    return StreamingResponse(
        request.app.state.rag_pipeline.stream(
            question=body.question,
            top_k=top_k,
        ),
        media_type="text/event-stream",
        headers={
            # Disable buffering so tokens reach the client immediately.
            # Without these, nginx/proxies may buffer the stream and the
            # client sees nothing until the response is complete — defeating
            # the point of streaming.
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
