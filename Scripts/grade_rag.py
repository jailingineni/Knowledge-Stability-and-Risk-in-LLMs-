"""
grade_rag.py

Grades rag_responses.csv using the same LLM-as-judge approach as grade_baseline.py.
Then prints a side-by-side comparison of baseline vs RAG correctness per model.

Output: results/rag_responses_graded.csv
"""

import os
import json
import time
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

RAG_CSV        = "results/rag_responses.csv"
BASELINE_CSV   = "results/baseline_responses_graded.csv"
REFERENCE_CSV  = "data/Questions Data Set - Sheet1 (1).csv"
OUTPUT_CSV     = "results/rag_responses_graded.csv"

REFERENCE_TEXT_COLUMN = "Answer"
REFERENCE_ID_COLUMN   = "Unnamed: 0"
JUDGE_MODEL           = "gpt-4o-mini"

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


def grade_response(question, reference_text, model_response):
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
        return {"correctness": "Incorrect", "notes": f"Non-JSON output: {text[:200]}"}

    correctness = parsed.get("correctness", "Incorrect")
    notes = parsed.get("notes", "")
    if correctness not in {"Correct", "Incorrect"}:
        correctness = "Incorrect"
        notes = f"Unexpected label. Raw: {notes}"

    return {"correctness": correctness, "notes": notes}


def main():
    rag_df       = pd.read_csv(RAG_CSV)
    reference_df = pd.read_csv(REFERENCE_CSV)

    reference_map = {
        str(row[REFERENCE_ID_COLUMN]).strip(): str(row[REFERENCE_TEXT_COLUMN]).strip()
        for _, row in reference_df.iterrows()
    }

    graded_correctness, graded_notes = [], []

    for i, row in rag_df.iterrows():
        question_id   = str(row["question_id"]).strip()
        question      = str(row["question"]).strip()
        response_text = str(row["response"]).strip()
        reference     = reference_map.get(question_id, "")

        if not reference:
            graded_correctness.append("Missing Reference")
            graded_notes.append("No verified reference found")
            continue

        print(f"Grading {i+1}/{len(rag_df)} | Q{question_id} | {row['model']}")
        try:
            result = grade_response(question, reference, response_text)
            graded_correctness.append(result["correctness"])
            graded_notes.append(result["notes"])
        except Exception as e:
            graded_correctness.append("Grading Error")
            graded_notes.append(str(e))

        time.sleep(0.5)

    rag_df["correctness"] = graded_correctness
    rag_df["notes"]       = graded_notes
    rag_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved graded RAG results -> {OUTPUT_CSV}")

    # ── Comparison ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("BASELINE vs RAG CORRECTNESS COMPARISON")
    print("="*60)

    baseline_df = pd.read_csv(BASELINE_CSV)

    for model in ["ChatGPT", "Claude", "Gemma"]:
        b = baseline_df[baseline_df["model"] == model]
        r = rag_df[rag_df["model"] == model]

        b_correct = (b["correctness"] == "Correct").sum()
        r_correct = (r["correctness"] == "Correct").sum()
        total     = len(b)

        b_pct = b_correct / total * 100 if total else 0
        r_pct = r_correct / total * 100 if total else 0
        delta = r_pct - b_pct

        sign = "+" if delta >= 0 else ""
        print(f"\n{model}")
        print(f"  Baseline : {b_correct}/{total} correct ({b_pct:.0f}%)")
        print(f"  RAG      : {r_correct}/{total} correct ({r_pct:.0f}%)")
        print(f"  Change   : {sign}{delta:.0f}pp")

    print("\n" + "="*60)
    print("OVERALL (all models combined)")
    print("="*60)
    total_b = len(baseline_df)
    total_r = len(rag_df)
    b_all = (baseline_df["correctness"] == "Correct").sum()
    r_all = (rag_df["correctness"] == "Correct").sum()
    print(f"  Baseline : {b_all}/{total_b} correct ({b_all/total_b*100:.0f}%)")
    print(f"  RAG      : {r_all}/{total_r} correct ({r_all/total_r*100:.0f}%)")

    # ── Per-question breakdown ───────────────────────────────────────────────
    print("\n" + "="*60)
    print("PER-QUESTION BREAKDOWN (avg across models)")
    print("="*60)
    print(f"{'Q':>3}  {'Baseline':>10}  {'RAG':>6}  {'Change':>8}")
    print("-"*35)

    for qid in range(1, 26):
        b_q = baseline_df[baseline_df["question_id"] == qid]
        r_q = rag_df[rag_df["question_id"] == qid]
        b_c = (b_q["correctness"] == "Correct").sum()
        r_c = (r_q["correctness"] == "Correct").sum()
        n   = len(b_q)
        delta = r_c - b_c
        sign = "+" if delta > 0 else ("" if delta == 0 else "")
        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
        print(f" {qid:>2}  {b_c}/{n} correct  {r_c}/{n}   {arrow} {sign}{delta}")


if __name__ == "__main__":
    main()
