# backend/ingest.py

import re
from pathlib import Path
from pypdf import PdfReader

# ── Tuning parameters ─────────────────────────────────────────────────
# Change these based on your document size and query type.
# Small focused docs  → CHUNK_SIZE 50–100,  OVERLAP 10–15
# Large reference docs → CHUNK_SIZE 300–500, OVERLAP 50–75
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80

# ─────────────────────────────────────────
# STEP 1: PARSE
# ─────────────────────────────────────────

def parse_file(file_path: str) -> str:
    """
    Accepts a file path, returns the full raw text as a single string.
    Supports .pdf and .txt files.
    
    Design note: this function's only job is extraction.
    It does NOT clean or chunk — that's the next step's responsibility.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if path.suffix.lower() == ".pdf":
        return _parse_pdf(path)
    elif path.suffix.lower() == ".txt":
        return _parse_txt(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")


def _parse_pdf(path: Path) -> str:
    """Extract text from every page of a PDF, joined with newlines."""
    reader = PdfReader(str(path))
    pages = []

    for page in reader.pages:
        text = page.extract_text()
        if text:  # Some pages (e.g. scanned images) return None
            pages.append(text)

    if not pages:
        raise ValueError(f"No extractable text found in {path.name}. "
                         "It may be a scanned PDF.")
    
    return "\n".join(pages)


def _parse_txt(path: Path) -> str:
    """Read a plain text file."""
    return path.read_text(encoding="utf-8", errors="ignore")


# ─────────────────────────────────────────
# STEP 2: CLEAN
# ─────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Normalise raw extracted text.
    
    Design note: we're not destroying information here —
    just making it consistent so chunking works predictably.
    """
    # Collapse 3+ newlines into 2 (preserve paragraph breaks, kill page noise)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Replace tabs with a single space
    text = re.sub(r'\t+', ' ', text)

    # Collapse multiple spaces into one
    text = re.sub(r' {2,}', ' ', text)

    # Strip leading/trailing whitespace from each line
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)

    return text.strip()


# ─────────────────────────────────────────
# STEP 3: CHUNK
# ─────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 50, overlap: int = 10) -> list[str]:
    """
    Split text into overlapping word-based chunks.
    
    Args:
        text:       The cleaned full-document text.
        chunk_size: How many words per chunk. 500 is a good default —
                    large enough for context, small enough to be precise.
        overlap:    How many words to repeat between consecutive chunks.
                    50 words (~2-3 sentences) prevents context gaps at boundaries.
    
    Returns:
        A list of string chunks.
    
    Design note: we split by words, not characters or tokens.
    Word counts are human-readable and easy to reason about.
    Sentence-transformers handle variable-length inputs fine.
    """
    words = text.split()

    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        # Move forward by (chunk_size - overlap) so the next chunk
        # starts 'overlap' words before the current chunk ended
        start += chunk_size - overlap

    return chunks


# ─────────────────────────────────────────
# PIPELINE: wire all three steps together
# ─────────────────────────────────────────

def ingest_document(file_path: str) -> list[str]:
    """
    The public API of this module.
    
    Given a file path, returns a list of clean, overlapping text chunks.
    This is the only function other modules need to import.
    
    Usage:
        from ingest import ingest_document
        chunks = ingest_document("documents/report.pdf")
    """
    raw_text = parse_file(file_path)
    clean = clean_text(raw_text)
    chunks = chunk_text(clean, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    print(f"[ingest] '{Path(file_path).name}' → {len(chunks)} chunks "
          f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks