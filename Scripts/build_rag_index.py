"""
build_rag_index.py

Fetches the source URLs from the verified answer dataset, chunks the page text,
embeds each chunk using OpenAI text-embedding-3-small, and builds a FAISS
cosine-similarity index saved to disk.

Run once before collect_rag_responses.py.

Output:
  data/rag_index.faiss   - FAISS vector index
  data/rag_chunks.json   - chunk text + metadata (source URL, question_id)
"""

import os
import json
import time

import numpy as np
import requests
import faiss
from bs4 import BeautifulSoup
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBED_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 1500       # characters per chunk
CHUNK_OVERLAP = 200     # overlap between consecutive chunks

INDEX_PATH = "data/rag_index.faiss"
CHUNKS_PATH = "data/rag_chunks.json"

# ---------------------------------------------------------------------------
# Source URLs taken directly from the verified Answer column in the dataset.
# Each entry maps one or more question IDs to the page(s) that answer them.
# ---------------------------------------------------------------------------
AWS_DOCS = [
    # Q1 – ACM exportable public certificates
    {
        "question_ids": [1],
        "url": "https://aws.amazon.com/blogs/aws/aws-certificate-manager-introduces-exportable-public-ssl-tls-certificates-to-use-anywhere/",
        "topic": "ACM Exportable Public Certificates",
    },
    # Q2 – ACM max validity period (198 days, Feb 2026 change)
    {
        "question_ids": [2],
        "url": "https://aws.amazon.com/about-aws/whats-new/2026/02/aws-certificate-manager-updates-default/",
        "topic": "ACM Certificate Validity Update 2026",
    },
    # Q3 – EC2 nested virtualization (C8i, M8i, R8i)
    {
        "question_ids": [3],
        "url": "https://aws.amazon.com/about-aws/whats-new/2026/02/amazon-ec2-nested-virtualization-on-virtual/",
        "topic": "EC2 Nested Virtualization",
    },
    # Q4 – AWS Backup cross-region to air-gapped vaults (single-action copy 2026)
    {
        "question_ids": [4],
        "url": "https://aws.amazon.com/about-aws/whats-new/2026/02/aws-backup-adds-cross-region-database-snapshot-logically-air-gapped-vaults/",
        "topic": "AWS Backup Air-Gapped Cross-Region Copy",
    },
    # Q5 – Bedrock structured outputs / JSON Schema
    {
        "question_ids": [5],
        "url": "https://aws.amazon.com/blogs/machine-learning/structured-outputs-on-amazon-bedrock-schema-compliant-ai-responses/",
        "topic": "Bedrock Structured Outputs",
    },
    # Q6 – RDS/Aurora restore: modify backup settings during restore
    {
        "question_ids": [6],
        "url": "https://aws.amazon.com/about-aws/whats-new/2026/02/rds-aurora-backup-configuration-restoring-snapshots/",
        "topic": "RDS Aurora Backup Config During Restore",
    },
    # Q7 – IAM Identity Center multi-region replication + KMS requirement
    {
        "question_ids": [7],
        "url": "https://aws.amazon.com/blogs/aws/aws-iam-identity-center-now-supports-multi-region-replication-for-aws-account-access-and-application-use/",
        "topic": "IAM Identity Center Multi-Region Replication",
    },
    # Q8 – Lambda Java runtimes (Java 25 latest)
    {
        "question_ids": [8],
        "url": "https://docs.aws.amazon.com/lambda/latest/dg/lambda-runtimes.html",
        "topic": "Lambda Runtimes",
    },
    # Q9, Q10 – Private CA partitioned CRLs (100M certs per CA)
    {
        "question_ids": [9, 10],
        "url": "https://aws.amazon.com/blogs/security/aws-private-certificate-authority-now-supports-partitioned-crls/",
        "topic": "Private CA Partitioned CRLs",
    },
    # Q11 – CloudWatch unified log management (OCSF + OTel)
    {
        "question_ids": [11],
        "url": "https://aws.amazon.com/blogs/aws/amazon-cloudwatch-introduces-unified-data-management-and-analytics-for-operations-security-and-compliance/",
        "topic": "CloudWatch Unified Log Management",
    },
    # Q12 – Compute Optimizer EBS automation rules (gp2→gp3, io1→io2)
    {
        "question_ids": [12],
        "url": "https://aws.amazon.com/blogs/aws-cloud-financial-management/introducing-automated-amazon-ebs-volume-optimization-in-aws-compute-optimizer/",
        "topic": "Compute Optimizer EBS Automation",
    },
    # Q13 – Control Tower Controls Dedicated experience
    {
        "question_ids": [13],
        "url": "https://aws.amazon.com/blogs/aws/aws-control-tower-introduces-a-controls-dedicated-experience/",
        "topic": "Control Tower Controls Dedicated",
    },
    # Q14 – Organizations MPA baseline tests (every 90 days)
    {
        "question_ids": [14],
        "url": "https://aws.amazon.com/about-aws/whats-new/2026/03/multi-party-approval-team-baselining/",
        "topic": "Organizations Multi-Party Approval Baselining",
    },
    # Q15 – Redshift array_sort determinism for SUPER elements (two sources)
    {
        "question_ids": [15],
        "url": "https://aws.amazon.com/about-aws/whats-new/2026/03/amazon-redshift-nine-new-array-functions/",
        "topic": "Redshift New Array Functions",
    },
    {
        "question_ids": [15],
        "url": "https://docs.aws.amazon.com/redshift/latest/dg/super-overview.html",
        "topic": "Redshift SUPER Data Type Overview",
    },
    # Q16 – DataBrew resource types (Dataset, Recipe, Ruleset, Job)
    {
        "question_ids": [16],
        "url": "https://docs.aws.amazon.com/AWSCloudFormation/latest/TemplateReference/aws-resource-databrew-dataset.html",
        "topic": "DataBrew CloudFormation Resources",
    },
    # Q17 – Lightsail OpenClaw gateway token rotation / credential storage
    {
        "question_ids": [17],
        "url": "https://aws.amazon.com/blogs/aws/introducing-openclaw-on-amazon-lightsail-to-run-your-autonomous-private-ai-agents/",
        "topic": "Lightsail OpenClaw",
    },
    # Q18 – NAT Gateway Trusted Advisor inactive flag (32-day / no route)
    {
        "question_ids": [18],
        "url": "https://aws.amazon.com/about-aws/whats-new/2026/02/trusted-advisor-unused-nat-gateway-check/",
        "topic": "Trusted Advisor NAT Gateway Check",
    },
    # Q19 – Amazon Connect Health + EHR integration
    {
        "question_ids": [19],
        "url": "https://aws.amazon.com/blogs/industries/introducing-amazon-connect-health-agentic-ai-for-healthcare-built-for-the-people-who-deliver-it/",
        "topic": "Amazon Connect Health EHR Integration",
    },
    # Q20 – CloudWatch lock contention diagnostics + EventBridge integration
    {
        "question_ids": [20],
        "url": "https://aws.amazon.com/about-aws/whats-new/2026/02/amazon-cloudwatch-lock-contention-diagnostics-rds-postgresql/",
        "topic": "CloudWatch Lock Contention Diagnostics",
    },
    {
        "question_ids": [20],
        "url": "https://aws.amazon.com/about-aws/whats-new/2019/10/amazon-cloudwatch-sends-alarm-state-change-events-amazon-eventbridge/",
        "topic": "CloudWatch EventBridge Integration",
    },
    # Q21 – EKS node repair ignores DiskPressure/MemoryPressure/PIDPressure
    {
        "question_ids": [21],
        "url": "https://docs.aws.amazon.com/eks/latest/userguide/node-health.html",
        "topic": "EKS Node Health Auto Repair",
    },
    # Q22 – EKS managed node group: EC2 instances + Auto Scaling group created
    {
        "question_ids": [22],
        "url": "https://docs.aws.amazon.com/eks/latest/userguide/managed-node-groups.html",
        "topic": "EKS Managed Node Groups",
    },
    # Q23 – STS temporary credentials advantages
    {
        "question_ids": [23],
        "url": "https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_temp.html",
        "topic": "IAM Temporary Security Credentials (STS)",
    },
    # Q24 – KMS encryption context conditions in key policy
    {
        "question_ids": [24],
        "url": "https://docs.aws.amazon.com/kms/latest/developerguide/encrypt_context.html",
        "topic": "KMS Encryption Context",
    },
    # Q25 – Bedrock Projects access isolation via IAM / project ID
    {
        "question_ids": [25],
        "url": "https://docs.aws.amazon.com/bedrock/latest/userguide/projects.html",
        "topic": "Bedrock Projects",
    },
]


def fetch_page(url: str) -> str:
    """Fetch a page and return cleaned plain text."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")

    for tag in soup(["nav", "header", "footer", "script", "style", "aside", "noscript"]):
        tag.decompose()

    main = (
        soup.find("main")
        or soup.find("div", {"id": "main-content"})
        or soup.find("div", {"id": "aws-doc-page-content"})
        or soup.body
    )

    text = main.get_text(separator="\n", strip=True) if main else ""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines)


def chunk_text(text: str, url: str, topic: str, question_ids: list) -> list[dict]:
    """Split text into overlapping chunks with metadata."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({
                "text": chunk,
                "source": url,
                "topic": topic,
                "question_ids": question_ids,
            })
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using OpenAI."""
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]


def main():
    os.makedirs("data", exist_ok=True)

    all_chunks: list[dict] = []

    # Step 1: Fetch and chunk all source pages
    seen_urls: set[str] = set()
    for doc in AWS_DOCS:
        url = doc["url"]
        if url in seen_urls:
            continue
        seen_urls.add(url)

        print(f"Fetching [{doc['topic']}]")
        print(f"  {url}")
        try:
            text = fetch_page(url)
            chunks = chunk_text(text, url, doc["topic"], doc["question_ids"])
            all_chunks.extend(chunks)
            print(f"  -> {len(chunks)} chunks ({len(text):,} chars)")
        except Exception as e:
            print(f"  -> SKIPPED: {e}")
        time.sleep(0.8)

    print(f"\nTotal chunks collected: {len(all_chunks)}")

    if not all_chunks:
        print("No chunks collected — check network access. Exiting.")
        return

    # Step 2: Embed all chunks in batches of 100
    print("\nEmbedding chunks...")
    all_embeddings: list[list[float]] = []
    batch_size = 100

    for i in range(0, len(all_chunks), batch_size):
        batch_texts = [c["text"] for c in all_chunks[i: i + batch_size]]
        embeddings = embed_batch(batch_texts)
        all_embeddings.extend(embeddings)
        print(f"  Embedded {min(i + batch_size, len(all_chunks))}/{len(all_chunks)}")
        time.sleep(0.3)

    # Step 3: Build FAISS index (inner product on L2-normalized = cosine similarity)
    print("\nBuilding FAISS index...")
    dim = len(all_embeddings[0])
    embeddings_array = np.array(all_embeddings, dtype=np.float32)
    faiss.normalize_L2(embeddings_array)

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings_array)

    # Step 4: Save to disk
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"\nSaved FAISS index ({index.ntotal} vectors) -> {INDEX_PATH}")
    print(f"Saved chunk metadata ({len(all_chunks)} chunks)  -> {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
