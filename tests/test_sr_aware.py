"""
test_sr_aware.py - Tests SR-aware budget selection

DSPM should automatically find the minimum budget where SR >= target.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openai import OpenAI
from dspm import DSPMMemory
from dspm.sr_budget import SRAwareBudgetSelector

MODEL = "openai/gpt-oss-120b"

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

QUESTIONS = [
    {"question": "What is the JWT access token expiry time?", "keywords": ["30", "minute"]},
    {"question": "What is the JWT refresh token expiry time?", "keywords": ["7", "day"]},
    {"question": "What is the rate limit per user?", "keywords": ["100"]},
    {"question": "What is the rate limit per API key?", "keywords": ["1000"]},
    {"question": "What is the S3 pre-signed URL expiry time?", "keywords": ["15", "minute"]},
    {"question": "What is the database connection pool size?", "keywords": ["20"]},
    {"question": "What is the database query timeout?", "keywords": ["5", "second"]},
    {"question": "What is the slow query log threshold?", "keywords": ["200"]},
    {"question": "What is the autoscale CPU threshold?", "keywords": ["60"]},
    {"question": "What is the GDPR data deletion SLA?", "keywords": ["30", "day"]},
]

def main():
    if not os.environ.get("GROQ_API_KEY"):
        print("ERROR: set $env:GROQ_API_KEY first")
        sys.exit(1)

    llm = OpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )

    print("=" * 70)
    print("DSPM v0.3.0 -- SR-AWARE BUDGET SELECTION TEST")
    print("=" * 70)
    print()

    # Create SR selector with hardcoded questions
    sr_selector = SRAwareBudgetSelector(
        llm_client=llm,
        model=MODEL,
        target_sr=0.8,
        questions=QUESTIONS,
    )

    # Create DSPMMemory with SR-aware mode
    mem = DSPMMemory(
        budget=800,
        llm_client=llm,
        model=MODEL,
        target_sr=0.8,
        ctx_window=8192,
        response_reserve=2000,
    )
    # Override the SR selector with our pre-configured one
    mem._sr_selector = sr_selector

    # Process the dialogue
    print("Phase 1: Processing 24-turn dialogue...")
    for idx, (role, text) in enumerate(CONVERSATION):
        patches = mem.add_turn(role, text)
        print(f"  Turn {idx:02d} | new={len(patches)} | memory={len(mem.all_patches)}")

    print()
    print(f"  Total patches: {len(mem.all_patches)}")
    print(f"  Critical total: {sum(1 for p in mem.all_patches if p.patch_type in {'constraint', 'decision'})}")
    print()

    # Get context with SR-aware budget selection
    print("=" * 70)
    print("Phase 2: SR-Aware Budget Selection")
    print(f"  Target SR: 80%")
    print("=" * 70)
    print()
    print("  Running binary search... (this takes ~5-7 minutes)")
    print()

    ctx = mem.get_context(query="What are all constraints and key decisions?")

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Optimal budget found: {mem.budget} tokens")
    print(f"  Semantic Recall (SR): {mem.last_sr:.1%}")
    print(f"  Context tokens: {mem.stats['context_tokens']}")
    print()
    print("  Compressed context:")
    print("-" * 70)
    print(ctx)
    print("-" * 70)
    print()

    if mem.last_sr >= 0.8:
        print("  PASS: SR >= 80% target met!")
    else:
        print(f"  PARTIAL: SR = {mem.last_sr:.1%} (target was 80%)")

if __name__ == "__main__":
    main()
