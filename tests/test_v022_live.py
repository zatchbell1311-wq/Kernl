"""
test_v022_live.py -- verify v0.2.2 max_extraction_tokens parameter
works on a real LLM (Groq gpt-oss-120b).

Tests:
1. Default (2000) - backward compat with v0.2.1
2. Custom (800)   - simulates small-context local model
3. Small (400)    - aggressive limit, still must extract patches

Run:
    $env:GROQ_API_KEY = "gsk_..."
    python tests/test_v022_live.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dspm import DSPMMemory
from dspm.patch import count_tokens
from openai import OpenAI

if not os.environ.get("GROQ_API_KEY"):
    print("ERROR: set $env:GROQ_API_KEY first")
    sys.exit(1)

llm = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

# 8-turn conversation (smaller than large_live_test for speed)
CONVERSATION = [
    ("user", "Build a REST API. Use FastAPI with PostgreSQL 15. JWT auth, 24h expiry."),
    ("asst", "FastAPI chosen. PostgreSQL 15 with async SQLAlchemy. JWT HS256, 24h expiry, Redis cache 60s TTL."),
    ("user", "Add AWS S3 for images. Pre-signed URLs, 15-min expiry, SSE-KMS."),
    ("asst", "S3 pre-signed URLs, 15-min expiry, SSE-KMS. Pydantic v2 validates schemas."),
    ("user", "Revision: raise Redis TTL from 60s to 300s for listings. Keep stock_count at 60s."),
    ("asst", "Redis keys split: listings TTL 300s, stock_count TTL 60s with write-through invalidation."),
    ("user", "Security: replace 24h JWT with 15-min access + 30-day refresh tokens. Revoke via Redis."),
    ("asst", "Access tokens 15-min, refresh 30-day with reuse detection. Denylist in Redis keyed by jti."),
]

MODEL = "openai/gpt-oss-120b"

def run_test(label, max_tokens):
    print(f"\n{'='*60}")
    print(f"TEST: {label}")
    print(f"  max_extraction_tokens = {max_tokens}")
    print(f"  budget = 250")
    print(f"  model = {MODEL}")
    print(f"{'='*60}")

    mem = DSPMMemory(
        budget=250,
        llm_client=llm,
        model=MODEL,
        max_extraction_tokens=max_tokens,
    )

    for idx, (role, text) in enumerate(CONVERSATION):
        patches = mem.add_turn(role, text)

    s = mem.stats
    print(f"\nResults:")
    print(f"  Turns processed       : {s['turns']}")
    print(f"  Total patches         : {s['total_patches']}")
    print(f"  Critical total        : {s['critical_total']}")
    print(f"  LLM calls             : {s['llm_calls']}")
    print(f"  LLM errors            : {s['llm_errors']}")
    print(f"  Conversation tokens   : {s['conversation_tokens']}")
    print(f"  Context tokens (out)  : {s['context_tokens']}")

    # Pass/fail criteria
    print(f"\nVerdict:")
    if s["llm_errors"] > 0:
        print(f"  FAIL - {s['llm_errors']} LLM errors occurred")
        return False
    if s["total_patches"] == 0:
        print(f"  FAIL - no patches extracted")
        return False
    if s["critical_total"] == 0:
        print(f"  FAIL - no critical patches extracted")
        return False
    print(f"  PASS - {s['total_patches']} patches extracted, 0 errors")
    print(f"  PASS - {s['critical_total']} critical patches found")
    return True

def main():
    print("=" * 60)
    print("DSPM v0.2.2 LIVE TEST - max_extraction_tokens parameter")
    print("=" * 60)

    results = []
    results.append(("Default (2000)", run_test("Backward compat - default 2000", 2000)))
    results.append(("Custom (800)",   run_test("Small-context simulation - 800", 800)))
    results.append(("Aggressive (400)", run_test("Aggressive limit - 400", 400)))

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    all_pass = True
    for label, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {label:<20} : {status}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("ALL PASS - v0.2.2 patch verified on real Groq LLM.")
        print("Safe to publish and tell Piotr.")
    else:
        print("FAIL - at least one test failed. DO NOT publish.")
        sys.exit(1)

if __name__ == "__main__":
    main()
