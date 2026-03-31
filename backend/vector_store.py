# backend/vector_store.py

import faiss
import numpy as np
import json
from pathlib import Path

# Where the index and chunk text will be saved on disk
INDEX_PATH = "vector_store/index.faiss"
CHUNKS_PATH = "vector_store/chunks.json"
DIMENSION = 384   # Must match all-MiniLM-L6-v2 output size


def _ensure_dir():
    Path("vector_store").mkdir(exist_ok=True)


def save_index(embeddings: np.ndarray, chunks: list[str]):
    """
    Build a FAISS index from embeddings and persist it alongside
    the original chunk texts.

    Why store chunks separately? FAISS only stores vectors — it has
    no concept of the text they came from. We maintain a parallel
    list where chunks[i] corresponds to the vector at index i.
    That's how we go from 'found vector #42' → 'here is the text'.
    """
    _ensure_dir()

    # IndexFlatL2 = exact search using L2 (Euclidean) distance.
    # "Flat" means no compression — fine for up to ~100k vectors.
    # For millions of vectors you'd switch to IndexIVFFlat,
    # but that complexity isn't needed here.
    index = faiss.IndexFlatL2(DIMENSION)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"[vector_store] Saved {index.ntotal} vectors → {INDEX_PATH}")


def load_index():
    """
    Load a previously saved FAISS index and chunk list from disk.
    Returns (index, chunks).
    """
    if not Path(INDEX_PATH).exists():
        raise FileNotFoundError(
            "No vector index found. Ingest at least one document first."
        )

    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"[vector_store] Loaded {index.ntotal} vectors")
    return index, chunks


def search(query_embedding: np.ndarray, top_k: int = 5) -> list[str]:
    """
    Find the top_k most relevant chunks for a query embedding.

    Returns a list of chunk strings, ordered by relevance (closest first).

    Design note: this function loads the index fresh on each call.
    For a production system you'd cache this in memory at startup.
    For learning purposes, loading each time is fine — it makes the
    read/write boundary explicit and easy to reason about.
    """
    index, chunks = load_index()

    # FAISS returns (distances, indices) — both shape (1, top_k)
    distances, indices = index.search(query_embedding, top_k)

    # indices[0] = the result row for our single query
    results = []
    for idx in indices[0]:
        if idx != -1:  # FAISS returns -1 when fewer results exist than top_k
            results.append(chunks[idx])

    return results