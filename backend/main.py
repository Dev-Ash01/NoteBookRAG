# backend/main.py

import os
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pipeline import ingest_and_store, query_pipeline

# ── App init ──────────────────────────────────────────────────────────
app = FastAPI(
    title="Knowledge Base Search Engine",
    description="Upload documents and query them using RAG.",
    version="1.0.0"
)

# CORS lets your frontend (running on a different port) talk to this API.
# During development, we allow all origins.
# In production you'd restrict this to your actual frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Where uploaded files are temporarily stored before ingestion
UPLOAD_DIR = Path("documents")
UPLOAD_DIR.mkdir(exist_ok=True)


# ── Request / Response models ─────────────────────────────────────────
# Pydantic models do two things:
# 1. Validate incoming request data automatically
# 2. Document your API in the auto-generated Swagger UI
# Always define these — they make your API self-documenting.

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5   # how many chunks to retrieve — optional, defaults to 5

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[str]

class IngestResponse(BaseModel):
    filename: str
    chunks_stored: int
    message: str


# ── Endpoints ─────────────────────────────────────────────────────────

@app.get("/")
def root():
    """Health check — lets you confirm the server is running."""
    return {"status": "running", "message": "Knowledge Base API is live."}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    """
    Upload a PDF or TXT document to be ingested into the knowledge base.

    Design note: we save the file to disk first, then pass the path
    to ingest_and_store(). This keeps the file handling separate from
    the pipeline logic — ingest_and_store() doesn't need to know
    anything about HTTP uploads.
    """
    # Validate file type before doing any work
    allowed_extensions = {".pdf", ".txt"}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file_ext}'. Allowed: PDF, TXT"
        )

    # Save uploaded file to disk
    save_path = UPLOAD_DIR / file.filename
    with save_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run the full ingestion pipeline
    try:
        chunks_count = ingest_and_store(str(save_path))
    except ValueError as e:
        # e.g. scanned PDF with no extractable text
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

    return IngestResponse(
        filename=file.filename,
        chunks_stored=chunks_count,
        message=f"Successfully ingested '{file.filename}' into the knowledge base."
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the knowledge base with a natural language question.

    Returns a synthesized answer with the source chunks used.
    """
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:
        result = query_pipeline(
            question=request.question,
            top_k=request.top_k
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="No documents have been ingested yet. Upload a document first."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

    return QueryResponse(
        query=result["query"],
        answer=result["answer"],
        sources=result["sources"]
    )

