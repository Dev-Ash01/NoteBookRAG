# backend/test_synthesis.py

from synthesizer import build_prompt, synthesize
from pipeline import query_pipeline, ingest_and_store

# ── Test 1: inspect the prompt before spending API credits ────────────
# Always do this first — it costs nothing and reveals prompt issues
print("=" * 60)
print("TEST 1: What does the actual prompt look like?")
print("=" * 60)

sample_chunks = [
    "The company revenue grew by 23 percent in Q3 2024.",
    "Growth was primarily driven by the Asia-Pacific expansion.",
    "The legal team reviewed compliance documents in October."
]

prompt = build_prompt("What drove the Q3 growth?", sample_chunks)
print(prompt)
# Read this carefully. Does it look like something that would
# produce a grounded, accurate answer? This is your gut-check.


# ── Test 2: does the LLM stay grounded? ──────────────────────────────
print("\n" + "=" * 60)
print("TEST 2: Does the LLM answer from context only?")
print("=" * 60)

# This question has NO answer in the chunks — LLM should say so
result = synthesize(
    query="What is the CEO's name?",
    chunks=sample_chunks
)
print(f"Answer: {result['answer']}")
# Should say it cannot find the answer — NOT invent a name


# ── Test 3: full end-to-end on your real document ─────────────────────
print("\n" + "=" * 60)
print("TEST 3: Full pipeline on your document")
print("=" * 60)

# First re-ingest to make sure index is fresh
ingest_and_store("../documents/ch 1.pdf")  # <-- your actual file

# Now ask a question you know the answer to from reading the doc
result = query_pipeline("How do you make the people aware of the importance of environment sustainability?")  # <-- real question

print(f"\nQuestion: {result['query']}")
print(f"\nAnswer:\n{result['answer']}")
print(f"\nSources used ({len(result['sources'])}):")
for i, source in enumerate(result['sources'], 1):
    print(f"\n  [{i}] {source[:150]}...")