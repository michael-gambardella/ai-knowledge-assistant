import logging
import time
import uuid

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.models.schemas import DocumentListResponse, DocumentUploadResponse
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
    Full indexing pipeline: parse → chunk → embed → store in vector DB.

    Each service (processor, embedder, vector store) is loaded once at startup
    and accessed via app.state, so no initialization cost per request.
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

    document_id = str(uuid.uuid4())
    processor = request.app.state.document_processor
    embedder = request.app.state.embedding_service
    vector_store = request.app.state.vector_store

    # --- Stage 1: Parse + chunk ---
    try:
        chunks = processor.process(
            file_bytes=file_bytes,
            filename=file.filename,
            content_type=content_type,
            document_id=document_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # --- Stage 2: Embed all chunks in one batched call ---
    # Batching is critical — embedding 50 chunks one at a time is ~50x slower
    # than embedding them together because the model processes them in parallel.
    t_embed_start = time.perf_counter()
    chunk_texts = [c.content for c in chunks]
    embeddings = embedder.embed(chunk_texts)
    embed_ms = round((time.perf_counter() - t_embed_start) * 1000, 1)

    # --- Stage 3: Store vectors in ChromaDB ---
    vector_store.add_chunks(chunks, embeddings)

    # --- Register document-level metadata ---
    document_registry.register(
        document_id=document_id,
        filename=file.filename,
        chunk_count=len(chunks),
    )

    logger.info(
        "Upload complete: filename=%s id=%s chunks=%d embed_ms=%s",
        file.filename, document_id, len(chunks), embed_ms,
    )

    return DocumentUploadResponse(
        document_id=document_id,
        filename=file.filename,
        chunk_count=len(chunks),
        message=f"Indexed '{file.filename}' — {len(chunks)} chunks embedded in {embed_ms}ms.",
    )


@router.get("/", response_model=DocumentListResponse)
async def list_documents():
    """Returns all indexed documents."""
    documents = document_registry.list_all()
    return DocumentListResponse(documents=documents, total=len(documents))


@router.delete("/{document_id}", status_code=204)
async def delete_document(document_id: str, request: Request):
    """
    Removes a document from both the vector store and the registry.
    Order matters: delete vectors first, then registry. If vector deletion
    fails, the registry entry stays intact and the document remains queryable.
    """
    if not document_registry.get(document_id):
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")

    vector_store = request.app.state.vector_store
    deleted_chunks = vector_store.delete_document(document_id)
    document_registry.remove(document_id)

    logger.info("Deleted document: id=%s chunks_removed=%d", document_id, deleted_chunks)
