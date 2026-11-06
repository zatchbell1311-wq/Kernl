"""
test_async.py - Tests AsyncDSPMMemory with asyncio.gather()
Processes 3 dialogues concurrently to prove the async architecture works.
"""
import os
import sys
import asyncio
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openai import AsyncOpenAI
from dspm import AsyncDSPMMemory

MODEL = "openai/gpt-oss-120b"

# 3 short dialogues
DIALOGUE_1 = [
    ("user", "Build a REST API. Use FastAPI with PostgreSQL 15. JWT auth, 24h expiry."),
    ("asst", "FastAPI chosen. PostgreSQL 15 with async SQLAlchemy. JWT HS256, 24h expiry."),
    ("user", "Revision: JWT expiry changed from 24h to 15-min access + 30-day refresh."),
    ("asst", "Access tokens 15-min, refresh 30-day with reuse detection. Denylist in Redis."),
]

DIALOGUE_2 = [
    ("user", "Building a fintech app. JWT auth, 15-min access tokens, 7-day refresh tokens."),
    ("asst", "FastAPI chosen. JWT HS256, access 15min, refresh 7-day."),
    ("user", "Rate limiting: 100 requests/min per user. Return 429 with Retry-After."),
    ("asst", "Token bucket in Redis. Per-user 100/min. 429 + Retry-After header."),
]

DIALOGUE_3 = [
    ("user", "Database: max connection pool size 20. Query timeout 5 seconds."),
    ("asst", "Pool size 20, max overflow 10, pool recycle 1800s. pg_statement_timeout 5000ms."),
    ("user", "S3 for document storage. Pre-signed URLs, 1 hour expiry. SSE-KMS."),
    ("asst", "S3 private bucket, SSE-KMS, pre-signed URLs 1h expiry."),
]

async def process_dialogue(llm, name, dialogue):
    """Process one dialogue asynchronously."""
    print(f"  [{name}] Starting...")
    mem = AsyncDSPMMemory(budget=250, llm_client=llm, model=MODEL)
    
    for idx, (role, text) in enumerate(dialogue):
        patches = await mem.async_add_turn(role, text)
        print(f"  [{name}] Turn {idx} | new={len(patches)} | memory={len(mem.all_patches)}")
    
    ctx = await mem.async_get_context(query="What are the rules?")
    print(f"  [{name}] DONE - CRR: {mem.stats['crr']}%")
    return name, ctx

async def main():
    if not os.environ.get("GROQ_API_KEY"):
        print("ERROR: set $env:GROQ_API_KEY first")
        sys.exit(1)

    llm = AsyncOpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )

    print("=" * 60)
    print("DSPM v0.3.0 -- ASYNC TEST (3 dialogues concurrently)")
    print("=" * 60)
    print()

    t = time.time()
    
    # Run all 3 dialogues concurrently!
    results = await asyncio.gather(
        process_dialogue(llm, "Dialogue 1", DIALOGUE_1),
        process_dialogue(llm, "Dialogue 2", DIALOGUE_2),
        process_dialogue(llm, "Dialogue 3", DIALOGUE_3),
    )
    
    elapsed = time.time() - t
    print()
    print(f"Total time: {elapsed:.1f}s (3 dialogues processed concurrently)")
    print()
    
    for name, ctx in results:
        print(f"--- {name} ---")
        print(ctx[:200])
        print()

if __name__ == "__main__":
    asyncio.run(main())
