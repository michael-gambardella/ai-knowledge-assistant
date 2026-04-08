"""
Lightweight document metadata store backed by a local JSON file.

Tracks which documents have been uploaded and indexed. This sits alongside
ChromaDB (which stores the actual vectors) — ChromaDB doesn't expose a clean
"list all documents" API, so we maintain our own index of document-level info.

In a production system this would be a Postgres table. Using a JSON file here
keeps the stack simple while the RAG pipeline is being built.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

REGISTRY_PATH = "./document_registry.json"


def _load() -> dict:
    if not os.path.exists(REGISTRY_PATH):
        return {}
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(data: dict) -> None:
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def register(document_id: str, filename: str, chunk_count: int) -> dict:
    """Add a document record. Returns the created record."""
    registry = _load()
    record = {
        "document_id": document_id,
        "filename": filename,
        "chunk_count": chunk_count,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }
    registry[document_id] = record
    _save(registry)
    logger.info("Registered document: id=%s filename=%s chunks=%d", document_id, filename, chunk_count)
    return record


def list_all() -> list[dict]:
    """Return all registered documents, newest first."""
    registry = _load()
    records = list(registry.values())
    records.sort(key=lambda r: r["uploaded_at"], reverse=True)
    return records


def get(document_id: str) -> Optional[dict]:
    """Return a single document record, or None if not found."""
    return _load().get(document_id)


def remove(document_id: str) -> bool:
    """Delete a document record. Returns True if it existed."""
    registry = _load()
    if document_id not in registry:
        return False
    del registry[document_id]
    _save(registry)
    logger.info("Removed document from registry: id=%s", document_id)
    return True
