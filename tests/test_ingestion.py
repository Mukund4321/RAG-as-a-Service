import pytest
import numpy as np
from app.ingestion.chunker import chunk_text
from app.ingestion.document_loader import load_document


class TestChunker:
    def test_basic_chunking(self):
        text = "This is sentence one. This is sentence two. This is sentence three."
        chunks = chunk_text(text, chunk_size=10, overlap=2)
        assert len(chunks) >= 1
        assert all(isinstance(c, str) and len(c) > 0 for c in chunks)

    def test_empty_text(self):
        chunks = chunk_text("", chunk_size=512, overlap=64)
        assert chunks == []

    def test_overlap_preserves_context(self):
        sentences = [f"Sentence number {i} about topic A." for i in range(20)]
        text = " ".join(sentences)
        chunks = chunk_text(text, chunk_size=30, overlap=10)
        assert len(chunks) > 1
        # Verify no chunk is empty
        assert all(len(c.strip()) > 0 for c in chunks)

    def test_single_long_sentence(self):
        text = "word " * 600  # one long run-on
        chunks = chunk_text(text.strip() + ".", chunk_size=100, overlap=20)
        assert len(chunks) >= 1


class TestDocumentLoader:
    def test_txt_loading(self):
        content = b"Hello world. This is a test document."
        text = load_document("test.txt", content)
        assert "Hello world" in text

    def test_md_loading(self):
        content = b"# Header\n\nSome paragraph text."
        text = load_document("test.md", content)
        assert "Header" in text

    def test_unsupported_format_raises(self):
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_document("test.xyz", b"data")

    def test_utf8_handling(self):
        content = "Résumé naïve café".encode("utf-8")
        text = load_document("test.txt", content)
        assert "Résumé" in text
