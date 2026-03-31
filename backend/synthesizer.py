# backend/synthesizer.py  — Groq version

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Available free models on Groq:
# "llama3-8b-8192"     — fast, good for simple RAG
# "llama3-70b-8192"    — slower, much better reasoning  ← use this
# "mixtral-8x7b-32768" — great for long contexts
MODEL = "llama-3.3-70b-versatile"
MAX_TOKENS = 1024


def build_prompt(query: str, chunks: list[str]) -> str:
    """Same prompt logic — nothing changes here."""
    numbered_chunks = ""
    for i, chunk in enumerate(chunks, 1):
        numbered_chunks += f"[{i}] {chunk.strip()}\n\n"

    prompt = f"""You are a helpful assistant answering questions strictly based on the provided document excerpts.

RULES:
- Answer ONLY using information from the document excerpts below.
- If the answer is not clearly present in the excerpts, respond with:
  "I could not find a clear answer to this in the provided documents."
- Be concise and direct — 2 to 4 sentences is ideal.
- At the end of your answer, cite which excerpt(s) you used, like: (Source: [1], [3])

DOCUMENT EXCERPTS:
{numbered_chunks}
QUESTION: {query}

ANSWER:"""
    return prompt


def synthesize(query: str, chunks: list[str]) -> dict:
    """
    Same interface as before — pipeline.py doesn't need to change at all.
    Only this file knows we switched from Anthropic to Groq.
    """
    if not chunks:
        return {
            "answer": "No relevant documents were found for your query.",
            "sources": [],
            "query": query
        }

    prompt = build_prompt(query, chunks)

    try:
        response = _client.chat.completions.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise assistant that answers only from provided context."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        answer = response.choices[0].message.content.strip()

        return {
            "answer": answer,
            "sources": chunks,
            "query": query
        }

    except Exception as e:
        return {
            "answer": f"API error: {str(e)}",
            "sources": [],
            "query": query
        }