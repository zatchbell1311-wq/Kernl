import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dspm import DSPMEngine, make_client, extract_patches, merge_into_memory

TEST_CONVERSATION = [
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

def run_verify():
    print("=" * 55)
    print("DSPM v0.2.0 -- CRR Verification Test")
    print("=" * 55)

    client = make_client(provider="groq", model_override="openai/gpt-oss-120b")
    engine = DSPMEngine(budget=400)
    memory = []

    for idx, (role, text) in enumerate(TEST_CONVERSATION):
        patches = extract_patches(text, idx, client, dialogue_name="verify")
        merge_into_memory(memory, patches)
        print(f"  Turn {idx:02d} | new_patches={len(patches)} | memory_size={len(memory)}")

    critical_total = sum(1 for p in memory if p.patch_type in {"constraint", "decision"})
    selected, diag = engine.compress(memory, "final architecture constraints", len(TEST_CONVERSATION) - 1)
    critical_selected = sum(1 for p in selected if p.patch_type in {"constraint", "decision"})
    dspm_tokens = sum(p.token_cost for p in selected)
    raw_tokens = sum(len(t.split()) * 4 // 3 for _, t in TEST_CONVERSATION)

    crr = (critical_selected / critical_total * 100) if critical_total > 0 else 0.0
    trr = (1 - dspm_tokens / max(raw_tokens, 1)) * 100

    print()
    print(f"  Critical patches total    : {critical_total}")
    print(f"  Critical patches selected : {critical_selected}")
    print(f"  CRR                       : {crr:.1f}%")
    print(f"  DSPM tokens               : {dspm_tokens}")
    print(f"  Estimated TRR             : {trr:.1f}%")
    print()

    if crr < 100.0:
        print(f"FAIL -- CRR is {crr:.1f}%, expected 100%. DO NOT PUBLISH.")
        sys.exit(1)
    else:
        print("PASS -- CRR = 100%. Safe to publish.")
        print()
        print("Context preview:")
        print(engine.build_context(selected))

if __name__ == "__main__":
    run_verify()
