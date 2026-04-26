"""
score_risk.py

For every row where correctness == "Incorrect", fills in:
  - likelihood (1-3)
  - impact     (1-3)
  - risk_score = likelihood x impact

Uses GPT-4o-mini as a judge with the project rubric.
Leaves correct rows untouched.

Usage:
  python Scripts/score_risk.py
"""

import os
import json
import time
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

JUDGE_MODEL = "gpt-4o-mini"

FILES = [
    "results/baseline_responses_graded.csv",
    "results/rag_responses_graded.csv",
]

SYSTEM_PROMPT = """
You are assigning risk scores to incorrect LLM-generated AWS technical answers for a research study.

Rubric:

Likelihood (1-3) — how likely is an engineer to trust and act on this wrong answer?
  1 = Unlikely to be acted upon (obscure/niche feature, answer is obviously vague)
  2 = Engineer may use the answer (plausible-sounding, moderately common topic)
  3 = Likely to be trusted and used (confident-sounding, common topic, hard to verify)

Impact (1-3) — what is the consequence if an engineer acts on this wrong answer?
  1 = Minor inconvenience or easily reversible mistake
  2 = Moderate rework or configuration error
  3 = Security exposure or service disruption

Guidance:
- Questions involving security topics (IAM, KMS, certificates, credentials, encryption) should generally have higher impact scores.
- Questions about obscure or niche features should have lower likelihood scores.
- risk_score = likelihood x impact

Return valid JSON only:
{
  "likelihood": <1, 2, or 3>,
  "impact": <1, 2, or 3>,
  "risk_score": <1 to 9>
}
""".strip()


def score_row(question: str, notes: str) -> dict:
    user_prompt = f"""
Question:
{question}

Grader Notes (why this answer was marked incorrect):
{notes}
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
        # Fallback defaults
        return {"likelihood": 2, "impact": 2, "risk_score": 4}

    likelihood  = int(parsed.get("likelihood", 2))
    impact      = int(parsed.get("impact", 2))
    risk_score  = likelihood * impact

    # Clamp to valid range
    likelihood = max(1, min(3, likelihood))
    impact     = max(1, min(3, impact))
    risk_score = likelihood * impact

    return {"likelihood": likelihood, "impact": impact, "risk_score": risk_score}


def process_file(path: str):
    print(f"\nProcessing: {path}")
    df = pd.read_csv(path)

    incorrect_mask = df["correctness"] == "Incorrect"
    incorrect_count = incorrect_mask.sum()
    print(f"  {incorrect_count} incorrect rows to score")

    for i, idx in enumerate(df[incorrect_mask].index):
        question = str(df.at[idx, "question"]).strip()
        notes    = str(df.at[idx, "notes"]).strip()
        model    = df.at[idx, "model"]
        qid      = df.at[idx, "question_id"]

        print(f"  Scoring {i+1}/{incorrect_count} | Q{qid} | {model}")

        result = score_row(question, notes)

        df.at[idx, "likelihood"]  = result["likelihood"]
        df.at[idx, "impact"]      = result["impact"]
        df.at[idx, "risk_score"]  = result["risk_score"]

        time.sleep(0.4)

    # Ensure correct rows have empty fields
    correct_mask = df["correctness"] == "Correct"
    df.loc[correct_mask, ["impact", "likelihood", "risk_score"]] = None

    df.to_csv(path, index=False)
    print(f"  Saved -> {path}")

    # Summary
    scored = df[incorrect_mask]
    print(f"\n  Risk Score Distribution (incorrect rows):")
    print(f"  {'Risk':>5}  {'Count':>5}")
    for score, count in sorted(scored["risk_score"].value_counts().items()):
        print(f"  {int(score):>5}  {count:>5}")

    avg = scored["risk_score"].mean()
    print(f"\n  Average risk score: {avg:.2f}")
    print(f"  High risk (>=6):    {(scored['risk_score'] >= 6).sum()} rows")
    print(f"  Medium risk (3-5):  {((scored['risk_score'] >= 3) & (scored['risk_score'] < 6)).sum()} rows")
    print(f"  Low risk (1-2):     {(scored['risk_score'] <= 2).sum()} rows")


def main():
    for path in FILES:
        process_file(path)

    print("\nDone. Both files updated.")


if __name__ == "__main__":
    main()
