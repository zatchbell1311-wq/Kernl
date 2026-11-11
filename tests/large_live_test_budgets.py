"""
large_live_test_budgets.py -- LIVE test at scale across multiple budgets.
24-turn fintech conversation with 3 revisions (JWT 15->30min, S3 1h->15min,
CPU 70%->60%). Extracts patches ONCE via the LLM, then re-compresses at
budgets [60, 100, 250, 400, 800] (no extra LLM calls).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dspm import DSPMMemory
from dspm.patch  import count_tokens
from dspm.engine import BudgetTooSmallError
from openai import OpenAI

if not os.environ.get("GROQ_API_KEY"):
    print("ERROR: set $env:GROQ_API_KEY first")
    sys.exit(1)

llm = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

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

BUDGETS = [60, 100, 250, 400, 800]
MODEL   = "openai/gpt-oss-120b"

def main():
    print("=" * 70)
    print("DSPM v0.2.1 -- LARGE LIVE TEST (24 turns, multi-budget)")
    print(f"  Model: {MODEL} on Groq")
    print("=" * 70)
    print()
    print("Phase 1: Extracting patches (LLM calls, one per turn)...")
    print("-" * 70)

    mem = DSPMMemory(budget=400, llm_client=llm, model=MODEL)
    for idx, (role, text) in enumerate(CONVERSATION):
        patches = mem.add_turn(role, text)
        crit = sum(1 for p in patches if p.patch_type in {"constraint", "decision"})
        print(f"  Turn {idx:02d} | role={role:<4} | new={len(patches)} (crit={crit}) | "
              f"memory={len(mem.all_patches)}")

    s = mem.stats
    print()
    print(f"  Total patches        : {s['total_patches']}")
    print(f"  Critical total       : {s['critical_total']}")
    print(f"  Conversation tokens  : {s['conversation_tokens']}")
    print(f"  LLM calls            : {s['llm_calls']}")
    print(f"  LLM errors           : {s['llm_errors']}")

    print()
    print("-" * 70)
    print("Phase 1b: Revision tracking check")
    print("-" * 70)
    crit_patches = mem.critical_patches
    revision_markers = []
    for p in crit_patches:
        pl = p.payload.lower()
        if any(kw in pl for kw in ["revis", "chang", "updat", "from", "->", "to"]):
            revision_markers.append(p.payload)
    print(f"  Patches mentioning a revision: {len(revision_markers)}")
    for r in revision_markers[:8]:
        print(f"    - {r}")

    print()
    print("=" * 70)
    print("Phase 2: Sweeping budgets (NO LLM calls)")
    print("=" * 70)
    print()
    print(f"  {'budget':>6} | {'crit':>6} | {'CRR':>7} | {'ctx':>5} | {'TRR':>7} | result")
    print("  " + "-" * 56)

    rows = []
    all_pass = True
    for b in BUDGETS:
        try:
            mem.set_budget(b)
            ctx = mem.get_context(query="What are all constraints and key decisions?")
            st  = mem.stats
            ctx_tok = count_tokens(ctx)
            trr  = (1 - ctx_tok / max(st["conversation_tokens"], 1)) * 100
            ok   = "PASS" if st["crr"] == 100.0 else "FAIL"
            if st["crr"] != 100.0:
                all_pass = False
            print(f"  {b:>6} | {st['critical_selected']:>3}/{st['critical_total']:<2} | "
                  f"{st['crr']:>6.1f}% | {ctx_tok:>5} | {trr:>6.1f}% | {ok}")
            rows.append({"budget": b, "ctx": ctx, "crr": st["crr"],
                         "ctx_tok": ctx_tok, "trr": trr})
        except BudgetTooSmallError as e:
            print(f"  {b:>6} |  --  |   --   |  --  |   --   | SKIP ({str(e)[:40]})")

    print()
    print("=" * 70)
    if all_pass:
        print("ALL PASS -- CRR = 100% at every budget tested.")
    else:
        print("FAIL -- at least one budget dropped a critical.")
        sys.exit(1)

    print()
    print("=" * 70)
    print("Phase 3: Context preview at each budget")
    print("=" * 70)
    for r in rows:
        print()
        print(f"--- budget={r['budget']} | ctx={r['ctx_tok']} tok | TRR={r['trr']:.1f}% ---")
        print(r["ctx"])

if __name__ == "__main__":
    main()
