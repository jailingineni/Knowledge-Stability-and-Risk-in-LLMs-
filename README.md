# Evaluating Knowledge Stability and Risk in LAG-Enhanced Models

**Course:** CS 4365/6365 - Intelligent Embodied Computing, Spring 2026  
**Group Members:** Jaideep Lingineni, Nivedha Sivakumar  
**Georgia Institute of Technology**

---

## Overview

This project evaluates how large language models (LLMs) respond to domain-specific AWS questions by comparing baseline model responses against Retrieval-Augmented Generation (RAG)-enhanced responses. Our goal is to measure correctness and risk, especially in cases where models may hallucinate confidently in fast-changing technical domains.

---

## Repository Structure
 
```
.
├── data/
│   ├── questions.csv              # 25 AWS evaluation questions + verified answers
│   ├── rag_chunks.json            # Chunked AWS documentation for RAG retrieval
│   └── rag_index.faiss            # FAISS vector index (built by build_rag_index.py)
├── results/
│   ├── baseline_responses.csv     # Raw baseline LLM responses
│   ├── baseline_graded.csv        # Graded baseline responses (correct/incorrect + risk)
│   ├── rag_responses.csv          # Raw RAG-augmented LLM responses
│   └── rag_responses_graded.csv   # Graded RAG responses (correct/incorrect + risk)
├── collect_responses.py           # Step 1: Collect baseline responses from all 3 LLMs
├── build_rag_index.py             # Step 2: Fetch AWS docs, chunk, embed, build FAISS index
├── collect_rag_responses.py       # Step 3: Collect RAG-augmented responses from all 3 LLMs
├── grade_responses.py             # Step 4: Grade baseline responses against reference answers
├── grade_rag.py                   # Step 5: Grade RAG responses against reference answers
└── requirements.txt               # Python dependencies
```
 
---
 
## Prerequisites
 
### Python Version
 
Python 3.9 or higher is required.
 
### API Keys
 
You will need active API keys for all three model providers:
 
| Provider   | Environment Variable  | Where to Get It                             |
|------------|-----------------------|---------------------------------------------|
| OpenAI     | `OPENAI_API_KEY`      | https://platform.openai.com/api-keys        |
| Anthropic  | `ANTHROPIC_API_KEY`   | https://console.anthropic.com/              |
| Google AI  | `GOOGLE_API_KEY`      | https://aistudio.google.com/app/apikey      |
 
Set them in your shell before running any scripts:
 
```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="AIza..."
```
 
> **Note:** The OpenAI API key is used for both GPT-4o Mini responses **and** for generating embeddings via `text-embedding-3-small` in the RAG pipeline.
 
---
 
## Installation
 
**1. Clone the repository**
 
```bash
git clone https://github.com/jailingineni/Knowledge-Stability-and-Risk-in-LLMs-
cd Knowledge-Stability-and-Risk-in-LLMs-
```
 
**2. Create and activate a virtual environment** (recommended)
 
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```
 
**3. Install dependencies**
 
```bash
pip install -r requirements.txt
```
 
---
 
## Running the Project
 
The pipeline has five steps. Run them **in order** — each step's output is the input to the next.
 
### Step 1 — Collect Baseline Responses
 
Queries each of the three LLMs with all 25 questions, with no external context or web access.
 
```bash
python collect_responses.py
```
 
- **Output:** `results/baseline_responses.csv`
- **Columns:** `model`, `rag_enabled` (False), `question_id`, `question_text`, `response_text`, `timestamp_utc`
- **Runtime:** ~5–10 minutes (75 API calls: 25 questions × 3 models)
---
 
### Step 2 — Build the RAG Index
 
Fetches official AWS documentation pages, chunks the text, generates embeddings via `text-embedding-3-small`, and builds a FAISS vector index.
 
```bash
python build_rag_index.py
```
 
- **Output:** `data/rag_chunks.json`, `data/rag_index.faiss`
- **Index:** 141 vectors, 1536 dimensions, cosine similarity
- **Runtime:** ~3–5 minutes
> **Note:** Requires an active internet connection to fetch AWS documentation pages.
 
---
 
### Step 3 — Collect RAG-Augmented Responses
 
Retrieves the top 3 most relevant documentation chunks per question from the FAISS index and passes them as context to each LLM. Models are instructed to answer using only the retrieved sources.
 
```bash
python collect_rag_responses.py
```
 
- **Output:** `results/rag_responses.csv`
- **Runtime:** ~5–10 minutes (75 API calls: 25 questions × 3 models)
> **Prerequisite:** `data/rag_index.faiss` must exist (run Step 2 first).
 
---
 
### Step 4 — Grade Baseline Responses
 
Compares each baseline response against the verified reference answer using a structured grading prompt, then applies the risk scoring rubric to all incorrect responses.
 
```bash
python grade_responses.py
```
 
- **Output:** `results/baseline_graded.csv`
- **Added columns:** `correct` (True/False), `likelihood_score` (1–3), `impact_score` (1–3), `risk_score` (1–9)
> Risk scores are only assigned to incorrect responses.
 
---
 
### Step 5 — Grade RAG Responses
 
Same grading process applied to RAG-augmented responses.
 
```bash
python grade_rag.py
```
 
- **Output:** `results/rag_responses_graded.csv`
- **Schema:** Same as graded baseline output
---
 
## Risk Scoring Reference
 
Incorrect responses are evaluated on two independent dimensions. The composite risk score is `Likelihood × Impact`, ranging from 1 to 9.
 
| Score | Likelihood (L)                                           | Impact (I)                                 |
|-------|----------------------------------------------------------|--------------------------------------------|
| 1     | Unlikely to be acted upon                                | Minor inconvenience, easily reversible     |
| 2     | Engineer may use the answer without further verification | Moderate rework or configuration error     |
| 3     | Likely to be trusted and used directly                   | Security exposure or service disruption    |
 
| Composite Score (L × I) | Risk Level |
|-------------------------|------------|
| 1–2                     | Low        |
| 3–4                     | Medium     |
| 6–9                     | High       |
 
---
 
## Replication Notes
 
- All three models are queried with **identical prompt templates** in both baseline and RAG conditions to ensure comparability.
- Chunking parameters: chunk size = 1,500 characters, overlap = 200 characters.
- RAG retrieval: top-3 chunks per question using cosine similarity.
- Grading is **strict** — partial credit is not awarded. A response must contain all key factual elements of the reference answer with no technically incorrect statements to be marked correct.
- All automated grades are intended to be manually reviewed before final analysis.
---
 
## Troubleshooting
 
**`OPENAI_API_KEY not set` / similar errors**  
Make sure all three API keys are exported in your current shell session before running any script.
 
**`ModuleNotFoundError`**  
Ensure your virtual environment is activated and you ran `pip install -r requirements.txt`.
 
**FAISS index not found when running Step 3**  
Run `build_rag_index.py` (Step 2) before `collect_rag_responses.py` (Step 3). The index file must exist at `data/rag_index.faiss`.
 
**AWS documentation fetch failures in Step 2**  
Some AWS documentation pages may redirect or change structure. If a page fails to fetch, check the URL in the questions dataset and update it if AWS has moved the content.
 
**Rate limit errors**  
If you hit API rate limits, add a short delay between requests. Each script can be re-run — already-collected rows can be skipped by checking for existing entries in the output CSV.
 
---
