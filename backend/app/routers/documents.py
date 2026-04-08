import uuid
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, Request

from app.models.schemas import DocumentUploadResponse, DocumentListResponse
from app.services import document_registry

logger = logging.getLogger(__name__)

SUPPORTED_CONTENT_TYPES = {
    "application/pdf",
    "text/plain",
    "text/markdown",
}
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/upload", response_model=DocumentUploadResponse, status_code=201)
async def upload_document(request: Request, file: UploadFile = File(...)):
    """
    Indexing pipeline entry point: parse → chunk → (embed + store in Step 3).

    The processor is loaded once at startup and accessed via app.state to avoid
    re-initializing the splitter on every request.
    """
    # --- Validate file type ---
    content_type = file.content_type or ""
    if content_type not in SUPPORTED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{content_type}'. Upload a PDF, TXT, or Markdown file.",
        )

    # --- Read and validate file size ---
    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(file_bytes) / 1024 / 1024:.1f} MB). Maximum is 20 MB.",
        )
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # --- Process: parse + chunk ---
    document_id = str(uuid.uuid4())
    processor = request.app.state.document_processor

    try:
        chunks = processor.process(
            file_bytes=file_bytes,
            filename=file.filename,
            content_type=content_type,
            document_id=document_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # --- Register document metadata ---
    # In Step 3 this is also where we embed chunks and store them in ChromaDB.
    document_registry.register(
        document_id=document_id,
        filename=file.filename,
        chunk_count=len(chunks),
    )

    logger.info(
        "Upload complete: filename=%s id=%s chunks=%d",
        file.filename, document_id, len(chunks),
    )

    return DocumentUploadResponse(
        document_id=document_id,
        filename=file.filename,
        chunk_count=len(chunks),
        message=f"Successfully processed '{file.filename}' into {len(chunks)} chunks.",
    )


@router.get("/", response_model=DocumentListResponse)
async def list_documents():
    """Returns all indexed documents from the registry."""
    documents = document_registry.list_all()
    return DocumentListResponse(documents=documents, total=len(documents))


@router.delete("/{document_id}", status_code=204)
async def delete_document(document_id: str):
    """
    Removes a document from the registry.
    In Step 3, this will also delete its vectors from ChromaDB.
    """
    removed = document_registry.remove(document_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")
