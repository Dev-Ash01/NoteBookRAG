# backend/pipeline.py  — add the query pipeline below ingest_and_store()

from ingest import ingest_document, CHUNK_SIZE, CHUNK_OVERLAP
from embedder import embed_texts, embed_query
from vector_store import save_index, search
from synthesizer import synthesize


def ingest_and_store(file_path: str):
    print(f"\n── Ingesting: {file_path} ──")
    print(f"   chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
    chunks = ingest_document(file_path)
    print(f"[pipeline] Embedding {len(chunks)} chunks...")
    embeddings = embed_texts(chunks)
    save_index(embeddings, chunks)
    print(f"[pipeline] Done. {len(chunks)} chunks ready for search.\n")
    return len(chunks)


def query_pipeline(question: str, top_k: int = 10) -> dict:
    """
    The full query path — one function, clear contract.

    Given a question string, returns:
    {
        "answer":  str,
        "sources": list[str],
        "query":   str
    }

    This is what your FastAPI /query endpoint will call directly.
    Notice it has zero knowledge of files, FAISS internals,
    or API keys — each module owns its own complexity.
    """
    # Step 1: embed the question
    query_vec = embed_query(question)

    # Step 2: retrieve the most relevant chunks
    relevant_chunks = search(query_vec, top_k=top_k)

    # Step 3: synthesize an answer
    result = synthesize(question, relevant_chunks)

    return result