import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import health, documents, query

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs setup logic before the server starts accepting requests,
    and teardown logic when it shuts down.

    This is where we'll initialize the embedding model and vector DB
    connection in later steps — loading them once at startup is far
    cheaper than loading them on every request.
    """
    logger.info("Starting AI Knowledge Assistant (env=%s)", settings.app_env)
    # Future: initialize embedding model, vector DB connection here
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
