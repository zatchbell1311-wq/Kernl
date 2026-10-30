"""
small_live_test_budgets.py -- LIVE test across multiple budgets.
10-turn REST API conversation. Extracts patches ONCE via the LLM, then
re-compresses at budgets [60, 100, 250, 400] (no extra LLM calls).
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
    ("user", "Build a REST API. Use FastAPI with PostgreSQL 15. JWT auth, 24h expiry."),
    ("asst", "FastAPI chosen. PostgreSQL 15 with async SQLAlchemy. JWT HS256, 24h expiry, Redis cache 60s TTL."),
    ("user", "Add AWS S3 for images. Product schema: id, name, price Decimal(10,2), stock_count, image_url."),
    ("asst", "S3 pre-signed URLs, 15-min expiry, SSE-KMS. Pydantic v2 validates Decimal(10,2)."),
    ("user", "Revision: raise Redis TTL from 60s to 300s for listings. Keep stock_count at 60s."),
    ("asst", "Redis keys split: listings TTL 300s, stock_count TTL 60s with write-through invalidation."),
    ("user", "Security: replace 24h JWT with 15-min access + 30-day refresh tokens. Revoke via Redis."),
    ("asst", "Access tokens 15-min, refresh 30-day with reuse detection. Denylist in Redis keyed by jti."),
    ("user", "Rate limiting: 100 req/min per API key, burst 50. Return 429 with Retry-After, never 403."),
    ("asst", "Token-bucket via atomic Lua in Redis. 429 + Retry-After header on breach."),
]

BUDGETS = [60, 100, 250, 400]
MODEL   = "openai/gpt-oss-120b"

def main():
    print("=" * 70)
    print("DSPM v0.2.1 -- SMALL LIVE TEST (10 turns, multi-budget)")
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
            ctx = mem.get_context(query="final architecture constraints and decisions")
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
