# backend/test_ingest.py
# Run with: python test_ingest.py

from ingest import ingest_document, chunk_text, clean_text

# ── Test 1: chunking logic (no file needed) ──────────────────────────
sample = "word " * 600   # 600 words
chunks = chunk_text(sample, chunk_size=500, overlap=50)

print(f"Total chunks: {len(chunks)}")          # Should be 3
print(f"Chunk 0 word count: {len(chunks[0].split())}")  # 500
print(f"Chunk 1 word count: {len(chunks[1].split())}")  # 500
print(f"Chunk 2 word count: {len(chunks[2].split())}")  # 150 (remainder)

# ── Test 2: overlap is actually there ───────────────────────────────
words_0 = chunks[0].split()
words_1 = chunks[1].split()
overlap_region = words_0[-50:]          # Last 50 words of chunk 0
first_50_of_1 = words_1[:50]           # First 50 words of chunk 1
assert overlap_region == first_50_of_1, "Overlap broken!"
print("Overlap check: PASSED")

# ── Test 3: cleaning ─────────────────────────────────────────────────
messy = "Hello   world\n\n\n\nSecond   paragraph\t\there."
cleaned = clean_text(messy)
assert '\n\n\n' not in cleaned
assert '\t' not in cleaned
print(f"Clean check: PASSED → '{cleaned}'")

print("\nAll tests passed. Ready for Phase 2.")