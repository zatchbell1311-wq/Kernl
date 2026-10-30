"""big_live_test.py -- 24-turn fintech conversation test (uses GROQ_API_KEY env var)."""
import os
from openai import OpenAI
from dspm import DSPMMemory

if not os.environ.get("GROQ_API_KEY"):
    print("ERROR: set $env:GROQ_API_KEY first")
    raise SystemExit(1)

llm = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

mem = DSPMMemory(
    budget=400,
    llm_client=llm,
    model="openai/gpt-oss-120b",
)

conversation = [
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

print("=" * 60)
print("DSPM v0.2.1 -- BIG LIVE TEST (24 turns)")
print("=" * 60)

for idx, (role, text) in enumerate(conversation):
    patches = mem.add_turn(role, text)
    critical = sum(1 for p in patches if p.patch_type in {"constraint", "decision"})
    print(f"Turn {idx:02d} | role={role:<4} | new={len(patches)} (crit={critical}) | memory={len(mem.all_patches)}")

print()
print("=" * 60)
print("FINAL STATS:")
print("=" * 60)
s = mem.stats
for k, v in s.items():
    print(f"  {k:<22}: {v}")
