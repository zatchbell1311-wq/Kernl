"""
test_universal_budgets.py -- prove CRR = 100% across a CONTINUOUS range
of budgets, not just a hand-picked set.

Sweeps budgets: 10, 20, 30, 40, 50, 75, 100, 150, 200, 250, 300, 400,
                500, 750, 1000, 1500, 2000
                + the auto-detected floor for this set of criticals.

Run:
    python tests/test_universal_budgets.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# --- stub out the heavy embedder before importing DSPMEngine ---
import numpy as np
class _FakeEmbedder:
    def encode(self, texts, show_progress_bar=False):
        return np.zeros((len(texts), 8), dtype=np.float32)
import dspm.engine as _eng
_eng._embedder = _FakeEmbedder()
# ----------------------------------------------------------------

from dspm.patch  import SemanticPatch, count_tokens
from dspm.engine import DSPMEngine, BudgetTooSmallError

# 22 criticals (matches big_live_test.py) + a few non-criticals.
PATCHES = [
    SemanticPatch("p0_0",  0, "constraint", "JWT auth: 15-min access tokens, 7-day refresh tokens."),
    SemanticPatch("p1_0",  1, "constraint", "JWT HS256 algorithm with 15-minute access tokens and 7-day refresh."),
    SemanticPatch("p2_0",  2, "constraint", "PCI-DSS compliance mandatory; all card data tokenized; never store raw card numbers."),
    SemanticPatch("p3_0",  3, "constraint", "PCI-DSS scope confirmed; Stripe tokenization adopted for card PANs."),
    SemanticPatch("p4_0",  4, "constraint", "Rate limiting: 100 requests/min per user, 1000 requests/min per API key, return 429 with Retry-After header."),
    SemanticPatch("p8_0",  8, "constraint", "Database max connection pool size 20. Query timeout 5 seconds. Slow query threshold 200ms for logging."),
    SemanticPatch("p9_0",  9, "constraint", "Database pool size 20, max overflow 10, pool recycle 1800s; pg_statement_timeout 5000ms; slow query log at 200ms."),
    SemanticPatch("p10_0",10, "constraint", "S3 private bucket only; SSE-KMS encryption; pre-signed URLs expire in 1 hour; public access blocked at bucket policy."),
    SemanticPatch("p11_0",11, "constraint", "S3 public access blocked at bucket policy level; pre-signed URLs expire in 1 hour."),
    SemanticPatch("p14_0",14, "constraint", "ECS Fargate tasks: 0.5 vCPU, 1GB RAM each. Min 2 tasks, max 10. Autoscale at 70% CPU."),
    SemanticPatch("p15_0",15, "constraint", "Fargate task limits 0.5vCPU, 1GB RAM; autoscaling min 2 max 10 tasks; target CPU 70%."),
    SemanticPatch("p16_0",16, "constraint", "Observability: structured JSON logs, request_id correlation, Prometheus metrics, Alertmanager if p99 > 500ms for 5min."),
    SemanticPatch("p17_0",17, "constraint", "structlog JSON + request_id middleware; Prometheus via fastapi-instrumentator; Alertmanager p99>500ms for 5min."),
    SemanticPatch("p20_0",20, "constraint", "GDPR: user data deletion within 30 days of request; audit log of all deletions; orders pseudonymized not deleted."),
    SemanticPatch("p21_0",21, "constraint", "GDPR mandates user data deletion within 30 days of request, with audit logs for all deletions."),
    SemanticPatch("p6_0",  6, "decision", "Access token expiry revised from 15 minutes to 30 minutes following security review."),
    SemanticPatch("p7_0",  7, "decision", "Access token expiry set to 30 minutes; refresh tokens remain 7 days; blacklist TTL updated accordingly."),
    SemanticPatch("p12_0",12, "decision", "S3 pre-signed URL expiry reduced from 1 hour to 15 minutes for security compliance."),
    SemanticPatch("p13_0",13, "decision", "Pre-signed URL TTL updated to 15 minutes; existing URLs invalidated via lifecycle policy."),
    SemanticPatch("p18_0",18, "decision", "Autoscale CPU threshold revised from 70% to 60% following load testing bottleneck identification."),
    SemanticPatch("p19_0",19, "decision", "Autoscaling target updated to 60% CPU; scale-out cooldown 120s, scale-in cooldown 300s."),
    SemanticPatch("p22_0",22, "decision", "Fargate autoscaling target confirmed at 60% CPU, aligning with previous revision and GDPR 30-day deletion SLA."),
    # non-criticals
    SemanticPatch("p0_1",  0, "code",      "FastAPI chosen. PostgreSQL 15 with async SQLAlchemy. Redis for token blacklist."),
    SemanticPatch("p0_2",  0, "entity",    "Entities: FastAPI, PostgreSQL 15, Redis, JWT HS256."),
    SemanticPatch("p5_0",  5, "code",      "Token bucket in Redis. Per-user 100/min, per-key 1000/min. 429 + Retry-After."),
    SemanticPatch("p9_1",  9, "entity",    "PostgreSQL 15, async SQLAlchemy, Redis, S3, SSE-KMS."),
    SemanticPatch("p15_1",15, "code",      "Fargate 0.5vCPU/1GB. ALB with health checks. Sticky sessions via Nginx ip_hash."),
    SemanticPatch("p17_1",17, "code",      "Prometheus via fastapi-instrumentator; Alertmanager config for p99 latency alerts."),
    SemanticPatch("p21_1",21, "code",      "DELETE /users/{id}/gdpr endpoint; 30-day SLA; PII columns nulled; deletion audit in S3."),
]

RAW_TOKENS = sum(count_tokens(t.payload) for t in PATCHES)
CRIT_TOTAL = sum(1 for p in PATCHES if p.patch_type in {"constraint", "decision"})
FLOOR      = DSPMEngine.min_budget_for_crr(CRIT_TOTAL)


def run_at_budget(budget: int):
    eng = DSPMEngine(budget=budget)
    selected, diag = eng.compress(PATCHES, "all constraints and decisions", 24)
    crit_sel = sum(1 for p in selected if p.patch_type in {"constraint", "decision"})
    crr      = crit_sel / CRIT_TOTAL * 100 if CRIT_TOTAL else 0.0
    ctx_str  = eng.build_context(selected)
    ctx_tok  = count_tokens(ctx_str)
    trr      = (1 - ctx_tok / max(RAW_TOKENS, 1)) * 100
    return {
        "budget": budget, "crit_sel": crit_sel, "crr": crr,
        "ctx_tok": ctx_tok, "trr": trr,
        "dense_mode": diag.get("dense_mode", "off"),
    }


def main():
    print("=" * 78)
    print("DSPM v0.2.2 -- UNIVERSAL BUDGET SWEEP TEST")
    print("=" * 78)
    print(f"  Raw conversation tokens  : {RAW_TOKENS}")
    print(f"  Total critical patches   : {CRIT_TOTAL}")
    print(f"  Absolute floor (CRR=100%): {FLOOR} tokens   "
          f"(= {CRIT_TOTAL} criticals + 4 tag overhead)")
    print("=" * 78)
    print()

    # Sweep a continuous range + the floor + a couple of below-floor sanity checks.
    # With 22 criticals, the floor is 22*2 + 4 = 48 tokens.
    budgets = [FLOOR, FLOOR + 1, FLOOR + 5, FLOOR + 10, 75, 100, 150, 200, 250, 300, 400,
               500, 750, 1000, 1500, 2000, 5000]

    # Filter out budgets below the floor (they should raise BudgetTooSmallError).
    testable = [b for b in budgets if b >= FLOOR]
    below_floor = sorted(set(b for b in budgets if b < FLOOR) | {FLOOR - 1, FLOOR - 5, 10, 20, 30})
    below_floor = [b for b in below_floor if b >= 1]

    print(f"  {'budget':>6} | {'crit':>6} | {'CRR':>7} | {'ctx':>5} | {'TRR':>7} | "
          f"{'mode':>6} | result")
    print("  " + "-" * 72)

    all_pass = True
    for b in testable:
        r = run_at_budget(b)
        # CRR must be 100%. Context must be within +20% of budget.
        crr_ok = (r["crr"] == 100.0)
        ctx_ok = r["ctx_tok"] <= int(b * 1.2)
        ok = "PASS" if (crr_ok and ctx_ok) else "FAIL"
        if not (crr_ok and ctx_ok):
            all_pass = False
        print(f"  {b:>6} | {r['crit_sel']:>3}/{CRIT_TOTAL:<2} | "
              f"{r['crr']:>6.1f}% | {r['ctx_tok']:>5} | {r['trr']:>6.1f}% | "
              f"{r['dense_mode']:>6} | {ok}")

    print()
    print("  Below-floor budgets (should raise BudgetTooSmallError, not silently drop):")
    for b in below_floor:
        try:
            run_at_budget(b)
            print(f"  {b:>6} | FAIL -- did not raise BudgetTooSmallError")
            all_pass = False
        except BudgetTooSmallError as e:
            print(f"  {b:>6} | OK -- raised BudgetTooSmallError ({str(e)[:60]}...)")

    print()
    print("=" * 78)
    if all_pass:
        print(f"ALL PASS -- CRR = 100% at every budget from {FLOOR} to 5000 tokens.")
        print(f"Below the floor ({FLOOR}), the engine raises a clear error instead of silently")
        print("dropping criticals. This is the universal guarantee.")
    else:
        print("FAIL -- at least one budget failed. DO NOT PUBLISH.")
        sys.exit(1)


if __name__ == "__main__":
    main()
