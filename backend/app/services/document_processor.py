"""
Document processing pipeline: parse → chunk.

This module is responsible for turning raw uploaded files into a list of
text chunks that are ready to be embedded in Step 3. Two concerns live here:

  1. PARSING  — extract raw text from a file regardless of format
  2. CHUNKING — split that text into overlapping segments

Keeping these concerns in one service makes the pipeline easy to test and swap
(e.g. swapping pypdf for pdfplumber later changes only this file).
"""

import io
import logging
import uuid
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.config import settings

logger = logging.getLogger(__name__)

SUPPORTED_TYPES = {
    "application/pdf": "pdf",
    "text/plain": "txt",
    "text/markdown": "md",
}


@dataclass
class DocumentChunk:
    """
    A single chunk of text extracted from a document.

    This is the atomic unit the rest of the pipeline operates on.
    Metadata travels with the chunk so the vector DB can return source
    info at query time — without it, you can't show citations.
    """
    chunk_id: str
    document_id: str
    filename: str
    content: str
    chunk_index: int
    total_chunks: int
    char_count: int = field(init=False)

    def __post_init__(self):
        self.char_count = len(self.content)


class DocumentProcessor:
    """
    Stateless service: call process() with file bytes and get back chunks.

    Stateless means no instance variables change between calls — safe to
    use as a singleton loaded once at startup (which we do in main.py lifespan).
    """

    def __init__(self):
        # RecursiveCharacterTextSplitter tries to split on paragraph breaks,
        # then sentence breaks, then word breaks — in that order. This preserves
        # semantic coherence better than splitting on character count alone.
        #
        # chunk_size:    target number of characters per chunk
        # chunk_overlap: characters shared between adjacent chunks
        #
        # WHY OVERLAP MATTERS:
        # If a key sentence sits at the boundary between chunk 4 and chunk 5,
        # without overlap it appears truncated in both. With overlap, that
        # sentence appears in full in at least one chunk, so retrieval can find it.
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        logger.info(
            "DocumentProcessor initialized (chunk_size=%d, overlap=%d)",
            settings.chunk_size,
            settings.chunk_overlap,
        )

    def process(
        self,
        file_bytes: bytes,
        filename: str,
        content_type: str,
        document_id: str | None = None,
    ) -> list[DocumentChunk]:
        """
        Entry point: parse file bytes → split into chunks → return chunk list.

        Args:
            file_bytes:   raw bytes from the uploaded file
            filename:     original filename (used in metadata / citations)
            content_type: MIME type — determines which parser to use
            document_id:  optional stable ID; one is generated if not provided

        Returns:
            List of DocumentChunk objects, ready to be embedded.

        Raises:
            ValueError: if the file type is not supported or the file is empty
        """
        if content_type not in SUPPORTED_TYPES:
            raise ValueError(
                f"Unsupported file type '{content_type}'. "
                f"Supported: {', '.join(SUPPORTED_TYPES.keys())}"
            )

        doc_id = document_id or str(uuid.uuid4())
        logger.info("Processing document: filename=%s id=%s type=%s", filename, doc_id, content_type)

        # Step 1: Parse — extract raw text
        raw_text = self._parse(file_bytes, content_type, filename)

        if not raw_text.strip():
            raise ValueError(f"No text could be extracted from '{filename}'. Is the file empty or image-only?")

        logger.info("Extracted %d characters from '%s'", len(raw_text), filename)

        # Step 2: Chunk — split into overlapping segments
        chunks = self._chunk(raw_text, doc_id, filename)

        logger.info("Split '%s' into %d chunks", filename, len(chunks))
        return chunks

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse(self, file_bytes: bytes, content_type: str, filename: str) -> str:
        """Route to the correct parser based on MIME type."""
        file_format = SUPPORTED_TYPES[content_type]

        if file_format == "pdf":
            return self._parse_pdf(file_bytes, filename)
        else:
            # Plain text and markdown — decode and return
            return self._parse_text(file_bytes, filename)

    def _parse_pdf(self, file_bytes: bytes, filename: str) -> str:
        """
        Extract text from a PDF using pypdf.

        pypdf reads each page and returns its text content. Pages are joined
        with double newlines so the chunker can treat them as paragraph breaks.
        """
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            pages = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    pages.append(text.strip())
                else:
                    logger.debug("Page %d of '%s' yielded no text (may be image-based)", i + 1, filename)

            return "\n\n".join(pages)
        except Exception as e:
            raise ValueError(f"Failed to parse PDF '{filename}': {e}") from e

    def _parse_text(self, file_bytes: bytes, filename: str) -> str:
        """Decode plain text or markdown files, trying UTF-8 then latin-1."""
        for encoding in ("utf-8", "latin-1"):
            try:
                return file_bytes.decode(encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Could not decode '{filename}' as text.")

    def _chunk(self, text: str, document_id: str, filename: str) -> list[DocumentChunk]:
        """
        Split raw text into overlapping DocumentChunk objects.

        LangChain's RecursiveCharacterTextSplitter handles the splitting logic.
        We wrap each string segment in a DocumentChunk dataclass so metadata
        (document ID, filename, chunk position) stays attached to the content.
        """
        segments: list[str] = self._splitter.split_text(text)
        total = len(segments)

        return [
            DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                document_id=document_id,
                filename=filename,
                content=segment,
                chunk_index=i,
                total_chunks=total,
            )
            for i, segment in enumerate(segments)
        ]
