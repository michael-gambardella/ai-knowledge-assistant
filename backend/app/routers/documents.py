from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.schemas import DocumentUploadResponse, DocumentListResponse

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Accepts a document, processes it through the indexing pipeline:
    parse → chunk → embed → store in vector DB.

    Implemented in Step 2.
    """
    raise HTTPException(status_code=501, detail="Document pipeline not yet implemented — coming in Step 2")


@router.get("/", response_model=DocumentListResponse)
async def list_documents():
    """
    Lists all indexed documents. Implemented in Step 2.
    """
    raise HTTPException(status_code=501, detail="Not yet implemented — coming in Step 2")


@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """
    Removes a document and all its chunks from the vector DB. Implemented in Step 2.
    """
    raise HTTPException(status_code=501, detail="Not yet implemented — coming in Step 2")
