import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import health, documents, query
from app.services.document_processor import DocumentProcessor
from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStore
from app.services.llm_client import LLMClient
from app.services.rag_pipeline import RAGPipeline

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: initialize all services before the server accepts requests.
    Shutdown: clean up resources.

    WHY LOAD EVERYTHING HERE:
    The embedding model (~80 MB) takes 1-3 seconds to load into memory. Loading
    it once at startup means every subsequent request gets a pre-warmed model.
    Loading it per-request would add 1-3s to every upload — unacceptable latency.

    The same principle applies to the ChromaDB client: opening a persistent DB
    connection has overhead that should be paid once, not per-request.

    Services are attached to app.state so any router or dependency can access
    them via request.app.state without needing global variables or singletons.
    """
    logger.info("Starting AI Knowledge Assistant (env=%s)", settings.app_env)

    # Order matters: EmbeddingService must be ready before VectorStore,
    # because VectorStore logs the current chunk count at init (informational).
    app.state.document_processor = DocumentProcessor()
    app.state.embedding_service = EmbeddingService()
    app.state.vector_store = VectorStore()
    app.state.llm_client = LLMClient()
    app.state.rag_pipeline = RAGPipeline(
        embedding_service=app.state.embedding_service,
        vector_store=app.state.vector_store,
        llm_client=app.state.llm_client,
    )

    logger.info("All services initialized — ready to accept requests")
    yield

    logger.info("Shutting down AI Knowledge Assistant")


app = FastAPI(
    title="AI Knowledge Assistant",
    description="Production RAG system: upload documents, ask questions, get answers with sources.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allows the React frontend to call this API during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(health.router)
app.include_router(documents.router)
app.include_router(query.router)
