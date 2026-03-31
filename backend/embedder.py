# backend/embedder.py

from sentence_transformers import SentenceTransformer
import numpy as np

# Load once at module level — not inside a function.
# This is important: the model takes ~2 seconds to load.
# If you loaded it inside embed() it would reload on every call.
MODEL_NAME = "all-MiniLM-L6-v2"
_model = SentenceTransformer(MODEL_NAME)


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Convert a list of strings into a 2D numpy array of embeddings.
    Shape: (len(texts), 384)  — 384 dimensions per vector.

    Design note: we always return a numpy array, never a plain list.
    FAISS expects numpy float32 arrays. Enforcing this here means
    the vector store never has to worry about type conversion.
    """
    if not texts:
        raise ValueError("Cannot embed an empty list.")

    embeddings = _model.encode(
        texts,
        show_progress_bar=len(texts) > 50,  # Show progress for large batches
        convert_to_numpy=True
    )
    return embeddings.astype("float32")


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string.
    Returns shape: (1, 384) — a 2D array so FAISS search works directly.

    Design note: keeping query embedding separate from batch embedding
    makes the intent clear at the call site. When you read
    embed_query(q) you immediately know it's a single search input.
    """
    return embed_texts([query])