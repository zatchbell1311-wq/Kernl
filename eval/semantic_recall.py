"""
semantic_recall.py - v0.3.0 Semantic Recall metric

Measures how much MEANING survives compression, not just patch count.
Addresses the critical finding that CRR is tautological (counts patches, not meaning).

How it works:
1. Process the 24-turn fintech dialogue through DSPM
2. Compress at various budgets (60, 100, 250, 400, 800)
3. For each budget, ask a judge LLM 10 factual questions using ONLY the compressed context
4. Score accuracy = Semantic Recall (SR)

Run:
    $env:GROQ_API_KEY = "gsk_..."
    python eval/semantic_recall.py
"""
import os
import sys
import time
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openai import OpenAI
from dspm import DSPMMemory

# ── Configuration ──
MODEL = "openai/gpt-oss-120b"
BUDGETS = [60, 100, 250, 400, 800]

# ── The 24-turn fintech dialogue ──
CONVERSATION = [
    ("user", "Building a fintech app. Stack: FastAPI, PostgreSQL 15, Redis. JWT auth, 15-min access tokens, 7-day refresh tokens."),
    ("asst", "FastAPI chosen. PostgreSQL 15 with async SQLAlchemy. Redis for token blacklist. JWT HS256, access 15min, refresh 7-day."),
    ("user", "PCI-DSS compliance mandatory. All card data must be tokenized. Never store raw card numbers."),
    ("asst", "PCI-DSS scope confirmed. Stripe tokenization for card data. Raw PANs never touch our servers."),
    ("user", "Rate limiting: 100 requests/min per user. 1000 requests/min per API key. Return 429 with Retry-After."),
    ("asst", "Token bucket in Redis. Per-user 100/min, per-key 1000/min. 429 + Retry-After header on breach."),
    ("user", "Revision: access token expiry changed from 15 minutes to 30 minutes after security review."),
    ("asst", "Access tokens updated to 30-min expiry. Refresh tokens stay at 7 days. Blacklist TTL updated accordingly."),
    ("user", "Database: max connection pool size 20. Query timeout 5 seconds. Slow query threshold 200ms for logging."),
    ("asst", "Pool size 20, max overflow 10, pool recycle 1800s. pg_statement_timeout 5000ms. Slow query log at 200ms."),
    ("user", "S3 for document storage. Pre-signed URLs, 1 hour expiry. Server-side encryption SSE-KMS. Private bucket only."),
    ("asst", "S3 private bucket, SSE-KMS, pre-signed URLs 1h expiry. Public access blocked at bucket policy level."),
    ("user", "Revision: S3 pre-signed URL expiry reduced from 1 hour to 15 minutes for security compliance."),
    ("asst", "Pre-signed URL TTL updated to 15 minutes. Existing URLs invalidated via lifecycle policy."),
    ("user", "Deployment: ECS Fargate, 0.5 vCPU, 1GB RAM per task. Min 2 tasks, max 10. Autoscale at 70% CPU."),
    ("asst", "Fargate 0.5vCPU/1GB. ALB with health checks. Autoscaling: min 2, max 10, target CPU 70%."),
    ("user", "Observability: structured JSON logs, request_id correlation. Prometheus metrics. Alert if p99 > 500ms for 5 min."),
    ("asst", "structlog JSON + request_id middleware. Prometheus via fastapi-instrumentator. Alertmanager: p99>500ms for 5min."),
    ("user", "Revision: autoscale CPU threshold changed from 70% to 60% after load testing showed bottlenecks."),
    ("asst", "Autoscaling target updated to 60% CPU. Scale-out cooldown 120s, scale-in cooldown 300s."),
    ("user", "GDPR: user data deletion within 30 days of request. Audit log of all deletions. Orders pseudonymized not deleted."),
    ("asst", "DELETE /users/{id}/gdpr endpoint. 30-day SLA. Orders pseudonymized, PII columns nulled. Deletion audit in S3."),
    ("user", "Final check: what are all the hard constraints and key decisions we made?"),
    ("asst", "Summary: PCI-DSS compliant, JWT 30min/7day, rate limits 100/1000 per min, pool 20, S3 15min URLs, Fargate 60% CPU, GDPR 30day deletion."),
]

# ── 10 factual questions with ground truth answers ──
QUESTIONS = [
    {
        "question": "What is the JWT access token expiry time?",
        "answer": "30 minutes",
        "keywords": ["30", "minute"],
    },
    {
        "question": "What is the JWT refresh token expiry time?",
        "answer": "7 days",
        "keywords": ["7", "day"],
    },
    {
        "question": "What is the rate limit per user?",
        "answer": "100 requests per minute",
        "keywords": ["100"],
    },
    {
        "question": "What is the rate limit per API key?",
        "answer": "1000 requests per minute",
        "keywords": ["1000"],
    },
    {
        "question": "What is the S3 pre-signed URL expiry time?",
        "answer": "15 minutes",
        "keywords": ["15", "minute"],
    },
    {
        "question": "What is the database connection pool size?",
        "answer": "20",
        "keywords": ["20"],
    },
    {
        "question": "What is the database query timeout?",
        "answer": "5 seconds",
        "keywords": ["5", "second"],
    },
    {
        "question": "What is the slow query log threshold?",
        "answer": "200ms",
        "keywords": ["200"],
    },
    {
        "question": "What is the autoscale CPU threshold?",
        "answer": "60%",
        "keywords": ["60"],
    },
    {
        "question": "What is the GDPR data deletion SLA?",
        "answer": "30 days",
        "keywords": ["30", "day"],
    },
]


def judge_answer(llm_client, model, context, question):
    """Ask the judge LLM to answer a question using ONLY the compressed context."""
    prompt = (
        f"You are a fact-extraction engine. Answer the question using ONLY the context below. "
        f"If the answer is not in the context, say 'NOT FOUND'. "
        f"Be concise - just the answer, no explanation.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        f"ANSWER:"
    )
    try:
        response = llm_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as e:
        return f"ERROR: {e}"


def score_answer(answer, keywords):
    """Check if all keywords are present in the answer (case-insensitive)."""
    answer_lower = answer.lower()
    if "not found" in answer_lower:
        return False
    return all(kw.lower() in answer_lower for kw in keywords)


def main():
    if not os.environ.get("GROQ_API_KEY"):
        print("ERROR: set $env:GROQ_API_KEY first")
        sys.exit(1)

    llm = OpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )

    print("=" * 70)
    print("DSPM v0.3.0 -- SEMANTIC RECALL METRIC")
    print("=" * 70)
    print()

    # ── Phase 1: Extract patches from the dialogue ──
    print("Phase 1: Extracting patches from 24-turn dialogue...")
    mem = DSPMMemory(
        budget=800,
        llm_client=llm,
        model=MODEL,
    )

    for idx, (role, text) in enumerate(CONVERSATION):
        patches = mem.add_turn(role, text)
        print(f"  Turn {idx:02d} | new={len(patches)} | memory={len(mem.all_patches)}")

    stats = mem.stats
    print()
    print(f"  Total patches: {stats['total_patches']}")
    print(f"  Critical total: {stats['critical_total']}")
    print(f"  Conversation tokens: {stats['conversation_tokens']}")
    print()

    # ── Phase 2: For each budget, compress and test recall ──
    print("=" * 70)
    print("Phase 2: Semantic Recall test across budgets")
    print("=" * 70)
    print()

    results = []
    for budget in BUDGETS:
        print(f"--- Budget: {budget} tokens ---")
        mem.set_budget(budget)
        context = mem.get_context(query="What are all constraints and key decisions?")

        correct = 0
        for i, q in enumerate(QUESTIONS):
            answer = judge_answer(llm, MODEL, context, q["question"])
            is_correct = score_answer(answer, q["keywords"])
            status = "✓" if is_correct else "✗"
            print(f"  Q{i+1} {status} | {q['question']}")
            print(f"       Expected: {q['answer']} | Got: {answer}")
            if is_correct:
                correct += 1

        sr = (correct / len(QUESTIONS)) * 100
        crr = mem.stats["crr"]
        ctx_tokens = mem.stats["context_tokens"]

        results.append({
            "budget": budget,
            "crr": crr,
            "sr": sr,
            "ctx_tokens": ctx_tokens,
            "correct": correct,
            "total": len(QUESTIONS),
        })
        print(f"  -> CRR: {crr:.1f}% | SR: {sr:.1f}% ({correct}/{len(QUESTIONS)})")
        print()

    # ── Phase 3: Summary table ──
    print("=" * 70)
    print("SUMMARY: CRR vs Semantic Recall (SR)")
    print("=" * 70)
    print()
    print(f"  {'Budget':>6} | {'CRR':>7} | {'SR':>7} | {'Ctx tok':>7} | {'Score':>6}")
    print("  " + "-" * 50)
    for r in results:
        print(f"  {r['budget']:>6} | {r['crr']:>6.1f}% | {r['sr']:>6.1f}% | {r['ctx_tokens']:>7} | {r['correct']}/{r['total']}")
    print()
    print("CRR = Critical Retention Rate (patch count - tautological)")
    print("SR  = Semantic Recall (actual meaning retained - honest metric)")
    print()
    print("If SR drops significantly at tight budgets while CRR stays 100%,")
    print("it proves CRR alone is insufficient for evaluating compression quality.")


if __name__ == "__main__":
    main()
