import os
import json
import time
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    raise ValueError("Missing OPENAI_API_KEY in .env")

client = OpenAI(api_key=openai_key)

# Input/output files
BASELINE_CSV = "results/baseline_responses.csv"
REFERENCE_CSV = "data/Questions Data Set - Sheet1 (1).csv"
OUTPUT_CSV = "results/baseline_responses_graded.csv"

# Change this if you want to grade against another column later
REFERENCE_TEXT_COLUMN = "Answer"

# Your reference CSV uses this as the ID column right now
REFERENCE_ID_COLUMN = "Unnamed: 0"

JUDGE_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """
You are grading technical AWS answers for a research dataset.

You will receive:
- a question
- a verified reference answer
- a model response

Your job:
- decide whether the model response is correct
- allow paraphrasing
- mark incorrect if the response contains a meaningful factual error
- mark incorrect if the response is outdated relative to the verified answer
- mark incorrect if it misses a key requirement needed to answer the question
- be strict

Return valid JSON only with this schema:
{
  "correctness": "Correct" or "Incorrect",
  "notes": "brief explanation"
}
""".strip()


def grade_response(question: str, reference_text: str, model_response: str) -> dict:
    user_prompt = f"""
Question:
{question}

Verified Reference:
{reference_text}

Model Response:
{model_response}
""".strip()

    response = client.responses.create(
        model=JUDGE_MODEL,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    text = response.output_text.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = {
            "correctness": "Incorrect",
            "notes": f"Grader returned non-JSON output: {text[:200]}"
        }

    correctness = parsed.get("correctness", "Incorrect")
    notes = parsed.get("notes", "")

    if correctness not in {"Correct", "Incorrect"}:
        correctness = "Incorrect"
        notes = f"Unexpected correctness label. Raw notes: {notes}"

    return {
        "correctness": correctness,
        "notes": notes
    }


def main():
    baseline_df = pd.read_csv(BASELINE_CSV)
    reference_df = pd.read_csv(REFERENCE_CSV)

    # Build question_id -> verified reference mapping
    reference_map = {}
    for _, row in reference_df.iterrows():
        qid = str(row[REFERENCE_ID_COLUMN]).strip()
        reference_text = str(row[REFERENCE_TEXT_COLUMN]).strip()
        reference_map[qid] = reference_text

    graded_correctness = []
    graded_notes = []

    for i, row in baseline_df.iterrows():
        question_id = str(row["question_id"]).strip()
        question = str(row["question"]).strip()
        response_text = str(row["response"]).strip()

        reference_text = reference_map.get(question_id, "")

        if not reference_text:
            graded_correctness.append("Missing Reference")
            graded_notes.append("No verified reference found for this question_id")
            continue

        print(f"Grading row {i+1}/{len(baseline_df)} | Q{question_id} | {row['model']}")

        try:
            result = grade_response(question, reference_text, response_text)
            graded_correctness.append(result["correctness"])
            graded_notes.append(result["notes"])
        except Exception as e:
            graded_correctness.append("Grading Error")
            graded_notes.append(str(e))

        time.sleep(0.5)

    baseline_df["correctness"] = graded_correctness
    baseline_df["notes"] = graded_notes

    baseline_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved graded file to {OUTPUT_CSV}")

    # Simple summary
    summary = (
        baseline_df.groupby(["model", "correctness"])
        .size()
        .unstack(fill_value=0)
    )
    print("\nSummary:")
    print(summary)


if __name__ == "__main__":
    main()