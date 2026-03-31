# backend/test_embeddings.py

from embedder import embed_texts, embed_query
from vector_store import save_index, search
from pipeline import ingest_and_store

# ── Test 1: embedding shape is correct ──────────────────────────────
sample_chunks = [
    "The company revenue grew by 23 percent in Q3.",
    "The legal team reviewed all compliance documents.",
    "Machine learning models require large datasets.",
]
embeddings = embed_texts(sample_chunks)

print(f"Embedding shape: {embeddings.shape}")
# Should print: (3, 384)
assert embeddings.shape == (3, 384), "Shape wrong!"
print("Shape check: PASSED")

# ── Test 2: save and search round-trip ──────────────────────────────
save_index(embeddings, sample_chunks)

query_vec = embed_query("What were the financial results?")
results = search(query_vec, top_k=2)

print(f"\nQuery: 'What were the financial results?'")
print(f"Top result: {results[0]}")
# Should return the revenue chunk, not the legal or ML chunk
assert "revenue" in results[0].lower() or "percent" in results[0].lower()
print("Retrieval relevance check: PASSED")

# ── Test 3: run the full pipeline on your actual document ────────────
# Replace with the path to your PDF from Phase 1
ingest_and_store("../documents/Module 2 Part 1.pdf")
query_vec = embed_query("What is data cleaning?")
results = search(query_vec, top_k=3)

print("\nTop 3 results from your document:")
for i, r in enumerate(results, 1):
    print(f"\n[{i}] {r[:200]}...")