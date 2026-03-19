# Evaluating Knowledge Stability and Risk in LAG-Enhanced Models

**Course:** CS 4365/6365 - Intelligent Embodied Computing, Spring 2026  
**Group Members:** Jaideep Lingineni, Nivedha Sivakumar  
**Georgia Institute of Technology**

---

## Overview

This project evaluates how large language models (LLMs) respond to domain-specific AWS questions by comparing baseline model responses against Retrieval-Augmented Generation (RAG)-enhanced responses. Our goal is to measure correctness and risk, especially in cases where models may hallucinate confidently in fast-changing technical domains.

---

## Models

### Baseline Models (Implemented)
- GPT-4.0 Mini (API)
- Claude Haiku 4.5 (API)
- Gemma 3 (27B-IT)

> Note: We initially planned to use local models, but due to hardware limitations we switched to API-based models instead.

### Planned RAG Models
- GPT-4.0 Mini + RAG
- Claude Haiku 4.5 + RAG
- Gemma 3 (27B-IT) + RAG

---

## Methodology

### 1. Question Dataset
- Fixed dataset of AWS-related technical questions
- Focus on recently updated AWS features to test knowledge freshness

### 2. Baseline Response Collection
- A Python script queries all baseline models using their APIs
- Responses are stored in a CSV file for later evaluation

### 3. Evaluation
- Model responses are compared against verified answers from official AWS documentation
- An automated grading script classifies responses as correct or incorrect
- Results are manually reviewed to verify grading accuracy

### 4. Planned RAG Integration
- Add retrieval from official AWS documentation and related sources
- Re-run the same question set with retrieval support
- Compare RAG-enhanced performance against baseline results

---

## Project Structure

```text
Knowledge-Stability-and-Risk-in-LLMs/
│
├── data/
│   ├── questions_clean.csv
│   ├── response_template.csv
│   └── raw question datasets
│
├── results/
│   ├── baseline_responses.csv
│   └── baseline_results.csv
│
├── Scripts/
│   ├── collect_responses.py
│   └── grade_baseline.py
│
├── venv/
├── .env
├── .gitignore
├── requirements.txt
└── README.md