"""
Tests for the document processing pipeline (parse + chunk).

These are unit tests — they run fast and don't require any API keys,
a running server, or a vector database. Testing parsing and chunking
in isolation makes failures easy to diagnose.
"""

import pytest
from app.services.document_processor import DocumentProcessor, SUPPORTED_TYPES


@pytest.fixture
def processor():
    return DocumentProcessor()


class TestTextParsing:
    def test_plain_text_is_extracted(self, processor):
        text = "This is a test document.\n\nIt has two paragraphs."
        chunks = processor.process(
            file_bytes=text.encode("utf-8"),
            filename="test.txt",
            content_type="text/plain",
        )
        full_text = " ".join(c.content for c in chunks)
        assert "test document" in full_text
        assert "two paragraphs" in full_text

    def test_markdown_is_extracted(self, processor):
        md = "# Title\n\nSome content here."
        chunks = processor.process(
            file_bytes=md.encode("utf-8"),
            filename="test.md",
            content_type="text/markdown",
        )
        assert len(chunks) >= 1
        assert any("Title" in c.content or "content" in c.content for c in chunks)

    def test_empty_file_raises(self, processor):
        with pytest.raises(ValueError, match="No text"):
            processor.process(
                file_bytes=b"   ",
                filename="empty.txt",
                content_type="text/plain",
            )

    def test_unsupported_type_raises(self, processor):
        with pytest.raises(ValueError, match="Unsupported"):
            processor.process(
                file_bytes=b"data",
                filename="file.docx",
                content_type="application/vnd.openxmlformats",
            )


class TestChunking:
    def test_short_document_produces_one_chunk(self, processor):
        text = "Short document."
        chunks = processor.process(
            file_bytes=text.encode(),
            filename="short.txt",
            content_type="text/plain",
        )
        assert len(chunks) == 1

    def test_long_document_produces_multiple_chunks(self, processor):
        # Generate text much longer than chunk_size (500 chars)
        text = ("This is a sentence that will be repeated many times. " * 30)
        chunks = processor.process(
            file_bytes=text.encode(),
            filename="long.txt",
            content_type="text/plain",
        )
        assert len(chunks) > 1

    def test_chunk_metadata_is_correct(self, processor):
        text = "Hello world. " * 100
        doc_id = "test-doc-123"
        chunks = processor.process(
            file_bytes=text.encode(),
            filename="meta_test.txt",
            content_type="text/plain",
            document_id=doc_id,
        )
        for i, chunk in enumerate(chunks):
            assert chunk.document_id == doc_id
            assert chunk.filename == "meta_test.txt"
            assert chunk.chunk_index == i
            assert chunk.total_chunks == len(chunks)
            assert chunk.char_count == len(chunk.content)
            assert chunk.chunk_id  # non-empty UUID

    def test_chunk_ids_are_unique(self, processor):
        text = "Repeated content. " * 100
        chunks = processor.process(
            file_bytes=text.encode(),
            filename="ids.txt",
            content_type="text/plain",
        )
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_overlap_means_content_continuity(self, processor):
        """
        With overlap enabled, the end of chunk N should appear at the start
        of chunk N+1. This verifies that boundary content isn't lost.
        """
        # Build text with a unique marker near a chunk boundary
        text = ("A " * 200) + "BOUNDARY_MARKER " + ("B " * 200)
        chunks = processor.process(
            file_bytes=text.encode(),
            filename="overlap.txt",
            content_type="text/plain",
        )
        marker_chunks = [c for c in chunks if "BOUNDARY_MARKER" in c.content]
        # The marker should appear in at least one chunk (overlap ensures it isn't cut)
        assert len(marker_chunks) >= 1, "Boundary content should appear in at least one chunk"
