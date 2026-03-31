# backend/diagnose_retrieval.py

from embedder import embed_texts, embed_query
from vector_store import load_index
from ingest import ingest_document
import numpy as np

# ── DIAGNOSTIC 1: Look at your actual chunks ─────────────────────────
# Before blaming retrieval, check what the chunks actually contain.
# Bad chunks = retrieval will always fail, no matter how good the model is.

print("=" * 60)
print("DIAGNOSTIC 1: What do your chunks look like?")
print("=" * 60)

chunks = ingest_document("../documents/Module 2 Part 1.pdf")  # your actual file

print(f"\nTotal chunks: {len(chunks)}")
print(f"\n── First 3 chunks ──")
for i, chunk in enumerate(chunks[:3]):
    print(f"\n[Chunk {i}] ({len(chunk.split())} words)")
    print(chunk[:300])
    print("...")

# What to look for:
# GOOD: chunks contain coherent sentences and meaningful content
# BAD: chunks are full of ".......", page numbers, headers, garbled text


# ── DIAGNOSTIC 2: Does the relevant content even exist in any chunk? ──
# Search for the keyword manually — before involving embeddings at all.

print("\n" + "=" * 100)
print("DIAGNOSTIC 2: Manual keyword search in chunks")
print("=" * 100)

# Replace these with words you KNOW are in your document
keywords = ["cleaning", "data", "raw"]  # <-- change these

for keyword in keywords:
    matches = [i for i, c in enumerate(chunks) if keyword.lower() in c.lower()]
    print(f"\nKeyword '{keyword}' found in {len(matches)} chunks: {matches[:5]}")
    if matches:
        print(f"  Preview: {chunks[matches[0]][:200]}")

# What to look for:
# If your keyword exists in chunks but retrieval didn't return those chunks,
# the problem is the embedding model or query phrasing (Suspects 2 or 3).
# If the keyword isn't in ANY chunk, the problem is parsing/chunking (Suspect 1).


# ── DIAGNOSTIC 3: Check the actual similarity scores ──────────────────
# FAISS returns distances — lower = more similar. Let's see the numbers.

print("\n" + "=" * 60)
print("DIAGNOSTIC 3: Similarity scores for your query")
print("=" * 60)

index, stored_chunks = load_index()

your_query = "What is data cleaning?"   # <-- use your actual query
query_vec = embed_query(your_query)

# Get distances AND indices (we bypass our search() wrapper to see raw scores)
distances, indices = index.search(query_vec, k=5)

print(f"\nQuery: '{your_query}'")
print(f"\nTop 5 results with distances (lower = more similar):")
for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), 1):
    preview = stored_chunks[idx][:500].replace('\n', ' ')
    print(f"\n  [{rank}] Distance: {dist:.4f}")
    print(f"       {preview}...")

# What to look for:
# Distances below 0.5  = strong match  (good)
# Distances 0.5 - 1.0  = weak match
# Distances above 1.0  = essentially unrelated
# If ALL distances are above 1.0, your query phrasing is the problem.
# If scores look fine but wrong chunks come back, chunking is the problem.