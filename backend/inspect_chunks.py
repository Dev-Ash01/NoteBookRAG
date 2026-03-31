from ingest import ingest_document

source = "F:\My Data Analyst Projects\LLM Projects\Knowledge_base_Search_Engine\documents\Module 2 Part 1.pdf"
chunks = ingest_document(source)

print("total chunks:", len(chunks))
for i in range(min(4, len(chunks))):
    print(f"\n=== chunk {i} ===")
    print(chunks[i])