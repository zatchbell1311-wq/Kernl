"""Test async_add_turns_batch - 2-pass parallel extraction."""
import os, sys, asyncio, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openai import AsyncOpenAI
from dspm import AsyncDSPMMemory

MODEL = "openai/gpt-oss-120b"

# 8-turn dialogue
TURNS = [
    ("user", "Build a REST API. Use FastAPI with PostgreSQL 15. JWT auth, 24h expiry."),
    ("asst", "FastAPI chosen. PostgreSQL 15 with async SQLAlchemy. JWT HS256, 24h expiry."),
    ("user", "Rate limiting: 100 requests/min per user. Return 429 with Retry-After."),
    ("asst", "Token bucket in Redis. Per-user 100/min. 429 + Retry-After header on breach."),
    ("user", "Revision: JWT expiry changed from 24h to 15-min access + 30-day refresh."),
    ("asst", "Access tokens 15-min, refresh 30-day with reuse detection. Denylist in Redis."),
    ("user", "Database: max connection pool size 20. Query timeout 5 seconds."),
    ("asst", "Pool size 20, max overflow 10, pool recycle 1800s. pg_statement_timeout 5000ms."),
]

async def main():
    if not os.environ.get("GROQ_API_KEY"):
        print("ERROR: set $env:GROQ_API_KEY first")
        return

    llm = AsyncOpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )

    print("=" * 60)
    print("DSPM v0.3.0 -- BATCH EXTRACTION TEST")
    print("=" * 60)
    print()

    mem = AsyncDSPMMemory(budget=250, llm_client=llm, model=MODEL)

    t = time.time()
    patches = await mem.async_add_turns_batch(TURNS)
    elapsed = time.time() - t

    print(f"Extracted {len(patches)} patches from {len(TURNS)} turns")
    print(f"Time: {elapsed:.1f}s (all 8 turns extracted in parallel)")
    print(f"LLM calls: {mem.stats['llm_calls']}")
    print(f"LLM errors: {mem.stats['llm_errors']}")
    print()

    ctx = await mem.async_get_context(query="Summarize key decisions and constraints")
    print(f"Compressed context ({mem.stats['context_tokens']} tokens):")
    print(ctx)
    print()
    print(f"CRR: {mem.stats['crr']}%")
    print(f"Total patches in memory: {mem.stats['total_patches']}")

asyncio.run(main())
