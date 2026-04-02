"""
collect_rag_responses.py

Loads the FAISS index built by build_rag_index.py, retrieves the top-k most
relevant document chunks for each question, injects them as context into the
prompt, and queries all three baseline models (ChatGPT, Claude, Gemma).

Output: results/rag_responses.csv  (same schema as baseline_responses.csv)

Prerequisites:
  - Run Scripts/build_rag_index.py first to produce:
      data/rag_index.faiss
      data/rag_chunks.json
"""

import os
import csv
import json
import time
import numpy as np
from datetime import datetime, timezone

import faiss
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
from google import genai

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
anthropic_key = os.getenv("ANTHROPIC_API_KEY")
gemini_key = os.getenv("GEMINI_API_KEY")

if not openai_key:
    raise ValueError("Missing OPENAI_API_KEY in .env")
if not anthropic_key:
    raise ValueError("Missing ANTHROPIC_API_KEY in .env")
if not gemini_key:
    raise ValueError("Missing GEMINI_API_KEY in .env")

openai_client = OpenAI(api_key=openai_key)
anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
google_client = genai.Client(api_key=gemini_key)

OPENAI_MODEL = "gpt-4o-mini"
CLAUDE_MODEL = "claude-haiku-4-5"
GEMMA_MODEL = "gemma-3-27b-it"
EMBED_MODEL = "text-embedding-3-small"

INPUT_CSV = "data/questions_clean.csv"
OUTPUT_CSV = "results/rag_responses.csv"
INDEX_PATH = "data/rag_index.faiss"
CHUNKS_PATH = "data/rag_chunks.json"

TOP_K = 3  # number of chunks to retrieve per question

BASE_SYSTEM_PROMPT = (
    "You are answering technical AWS questions for an evaluation study. "
    "Answer clearly and directly. If you are uncertain, say so."
)


def build_rag_system_prompt(retrieved_chunks: list[dict]) -> str:
    """Prepend retrieved context to the system prompt."""
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        context_parts.append(
            f"[Source {i} – {chunk['topic']}]\n{chunk['text']}"
        )
    context_block = "\n\n".join(context_parts)
    return (
        f"{BASE_SYSTEM_PROMPT}\n\n"
        f"Use the following retrieved AWS documentation excerpts to inform your answer:\n\n"
        f"{context_block}"
    )


def embed_query(question: str) -> np.ndarray:
    """Embed a single question and return a normalized float32 vector."""
    response = openai_client.embeddings.create(model=EMBED_MODEL, input=[question])
    vec = np.array(response.data[0].embedding, dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(vec)
    return vec


def retrieve_chunks(
    question: str,
    index: faiss.Index,
    chunks: list[dict],
    top_k: int = TOP_K,
) -> list[dict]:
    """Embed the question, search the index, return the top-k chunks."""
    query_vec = embed_query(question)
    _, indices = index.search(query_vec, top_k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]


def ask_openai(question: str, system_prompt: str) -> tuple[str, str]:
    response = openai_client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
    )
    return response.output_text.strip(), OPENAI_MODEL


def ask_claude(question: str, system_prompt: str) -> tuple[str, str]:
    response = anthropic_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1000,
        system=system_prompt,
        messages=[{"role": "user", "content": question}],
    )
    text_parts = [
        block.text
        for block in response.content
        if getattr(block, "type", None) == "text"
    ]
    return "\n".join(text_parts).strip(), CLAUDE_MODEL


def ask_gemma(question: str, system_prompt: str) -> tuple[str, str]:
    prompt = f"{system_prompt}\n\nQuestion: {question}"
    response = google_client.models.generate_content(
        model=GEMMA_MODEL,
        contents=prompt,
    )
    return response.text.strip(), GEMMA_MODEL


def main():
    print("Loading RAG index and chunks...")

    if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(
            f"RAG index not found. Run Scripts/build_rag_index.py first.\n"
            f"  Expected: {INDEX_PATH}\n"
            f"            {CHUNKS_PATH}"
        )

    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Index loaded: {index.ntotal} vectors | {len(chunks)} chunks")

    os.makedirs("results", exist_ok=True)

    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        questions = list(csv.DictReader(f))

    print(f"Loaded {len(questions)} questions from {INPUT_CSV}\n")

    providers = [
        ("ChatGPT", ask_openai),
        ("Claude", ask_claude),
        ("Gemma", ask_gemma),
    ]

    rows = []

    for q in questions:
        question_id = q["question_id"]
        question_text = q["question"]

        # Retrieve relevant chunks once per question (shared across all models)
        retrieved = retrieve_chunks(question_text, index, chunks)
        system_prompt = build_rag_system_prompt(retrieved)
        sources = " | ".join(c["source"] for c in retrieved)

        for model_name, fn in providers:
            print(f"Running Q{question_id} with {model_name} (RAG)...")

            try:
                response_text, model_version = fn(question_text, system_prompt)
            except Exception as e:
                response_text = f"ERROR: {str(e)}"
                model_version = "unknown"

            rows.append({
                "question_id": question_id,
                "question": question_text,
                "model": model_name,
                "model_version": model_version,
                "rag_enabled": True,
                "prompt": question_text,
                "response": response_text,
                "retrieved_sources": sources,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "correctness": "",
                "impact": "",
                "likelihood": "",
                "risk_score": "",
                "notes": "",
            })

            time.sleep(1)

    fieldnames = [
        "question_id",
        "question",
        "model",
        "model_version",
        "rag_enabled",
        "prompt",
        "response",
        "retrieved_sources",
        "timestamp_utc",
        "correctness",
        "impact",
        "likelihood",
        "risk_score",
        "notes",
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} RAG responses -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
