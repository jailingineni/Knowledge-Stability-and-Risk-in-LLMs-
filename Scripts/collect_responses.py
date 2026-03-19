import os
import csv
import time
from datetime import datetime, timezone

from dotenv import load_dotenv
from openai import OpenAI
import anthropic
from google import genai

print("Script started...")

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

print("API keys loaded.")

openai_client = OpenAI(api_key=openai_key)
anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
google_client = genai.Client(api_key=gemini_key)

OPENAI_MODEL = "gpt-4o-mini"
CLAUDE_MODEL = "claude-haiku-4-5"
GEMMA_MODEL = "gemma-3-27b-it"

INPUT_CSV = "data/questions_clean.csv"
OUTPUT_CSV = "results/baseline_responses.csv"

SYSTEM_PROMPT = (
    "You are answering technical AWS questions for an evaluation study. "
    "Answer clearly and directly. If you are uncertain, say so."
)


def ask_openai(question):
    response = openai_client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    )
    return response.output_text.strip(), OPENAI_MODEL


def ask_claude(question):
    response = anthropic_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": question}],
    )

    text_parts = []
    for block in response.content:
        if getattr(block, "type", None) == "text":
            text_parts.append(block.text)

    return "\n".join(text_parts).strip(), CLAUDE_MODEL


def ask_gemma(question):
    prompt = f"{SYSTEM_PROMPT}\n\nQuestion: {question}"
    response = google_client.models.generate_content(
        model=GEMMA_MODEL,
        contents=prompt,
    )
    return response.text.strip(), GEMMA_MODEL


def main():
    print("Inside main()...")
    os.makedirs("results", exist_ok=True)

    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        questions = list(csv.DictReader(f))

    print(f"Loaded {len(questions)} questions from {INPUT_CSV}")

    rows = []

    providers = [
        ("ChatGPT", ask_openai),
        ("Claude", ask_claude),
        ("Gemma", ask_gemma),
    ]

    for q in questions:
        question_id = q["question_id"]
        question_text = q["question"]

        for model_name, fn in providers:
            print(f"Running Q{question_id} with {model_name}...")

            try:
                response_text, model_version = fn(question_text)
            except Exception as e:
                response_text = f"ERROR: {str(e)}"
                model_version = "unknown"

            rows.append({
                "question_id": question_id,
                "question": question_text,
                "model": model_name,
                "model_version": model_version,
                "rag_enabled": False,
                "prompt": question_text,
                "response": response_text,
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

    print(f"Saved {len(rows)} responses to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()