from fastapi import APIRouter, HTTPException
from app.models.schemas import QueryRequest, QueryResponse

router = APIRouter(prefix="/query", tags=["Query"])


@router.post("/", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Core RAG endpoint: embed question → retrieve chunks → generate answer.

    Implemented in Step 3 (retrieval) and Step 4 (generation).
    """
    raise HTTPException(status_code=501, detail="RAG pipeline not yet implemented — coming in Steps 3 & 4")
