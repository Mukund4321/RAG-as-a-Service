import io
from pathlib import Path


def load_document(filename: str, content: bytes) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return _load_pdf(content)
    elif ext in (".txt", ".md", ".rst"):
        return content.decode("utf-8", errors="replace")
    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported: .pdf, .txt, .md, .rst")


def _load_pdf(content: bytes) -> str:
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(content))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)
    except ImportError:
        raise RuntimeError("pypdf is required for PDF ingestion: pip install pypdf")
