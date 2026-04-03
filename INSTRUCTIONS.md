#Instrctutions to Run the Project

## Prequistes
Have the following installed
- Python 3.10+
- pip
- Git

## Clone the Repository
- open terminal
- run: git clone https://github.com/your-repo/Knowledge-Stability-and-Risk-in-LLMs.git
cd Knowledge-Stability-and-Risk-in-LLMs

## Set Up Virtual Environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows


## Install dependencies
- pip install -r requirements.txt

## Configure API Keys
- Create .env file in your root directory and add the contents from below in you .env file
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
- Ensure .env is not commited to GitHub

## Prepare Dataset
- Verify dataset exists and ensure questions are formatted correctly
data/questions_clean.csv

## Colleect Baseline Responses
- Run the script and it should output a .csv file
python Scripts/collect_responses.py


## Grade Baseline Responses
- Run the script and it should output a .csv file
python Scripts/grade_baseline.py


## RAG Pipeline

### Overview
The RAG (Retrieval-Augmented Generation) pipeline works in two steps:
1. Build a vector index from the verified AWS documentation source URLs
2. Use that index to retrieve relevant context at query time and inject it into each model's prompt

### How It Works
For each question, the pipeline:
1. Embeds the question using OpenAI `text-embedding-3-small`
2. Searches the FAISS index for the top 3 most semantically similar document chunks
3. Injects those chunks into the system prompt before querying all 3 models
4. All 3 models (ChatGPT, Claude, Gemma) receive the same retrieved context

The source URLs come directly from the verified Answer column in the dataset, so the retrieved content is the same documentation that backs up the correct answers.

### Step 1 — Build the RAG Index (run once)
This script fetches all source URLs, chunks the text, embeds the chunks, and saves a FAISS index to disk.

```bash
python Scripts/build_rag_index.py
```

Output:
- `data/rag_index.faiss` — FAISS vector index (141 vectors)
- `data/rag_chunks.json` — chunk text and metadata (source URL, topic, question IDs)

### Step 2 — Collect RAG Responses
This script loads the index, retrieves the top 3 chunks per question, and queries all 3 models with the augmented prompt.

```bash
python Scripts/collect_rag_responses.py
```

Output:
- `results/rag_responses.csv` — 75 rows (25 questions x 3 models), `rag_enabled=True`

### Step 3 — Grade RAG Responses and Compare vs Baseline
This script grades the RAG responses using GPT-4o-mini as a judge (same approach as baseline grading) and prints a side-by-side comparison.

```bash
python Scripts/grade_rag.py
```

Output:
- `results/rag_responses_graded.csv` — graded RAG responses
- Terminal summary showing baseline vs RAG correctness per model and per question
