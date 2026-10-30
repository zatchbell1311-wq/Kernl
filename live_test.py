import os, sys
sys.path.insert(0, ".")

from dspm import DSPMEngine, make_client, extract_patches, merge_into_memory

LIVE_CONVERSATION = [
    ("user", "Build a real-time chat application. Use Node.js with Socket.io. MongoDB for message storage. JWT auth."),
    ("asst", "Node.js with Socket.io chosen. MongoDB with message TTL index 30 days. JWT HS256, 1h expiry."),
    ("user", "Add Redis for presence tracking. Online/offline status, last seen timestamp. Pub/sub for cross-server sync."),
    ("asst", "Redis pub/sub for presence. HSET user:{id} status online last_seen timestamp. TTL 300s on presence keys."),
    ("user", "Revision: JWT expiry too short. Change from 1h to 8h access tokens, add 7-day refresh tokens."),
    ("asst", "Access tokens 8h, refresh tokens 7-day with rotation. Blacklist in Redis on logout."),
    ("user", "Scale requirement: must handle 50K concurrent connections per server. Horizontal scaling with sticky sessions."),
    ("asst", "Socket.io with Redis adapter for horizontal scale. Sticky sessions via Nginx ip_hash. Target 50K per node."),
    ("user", "Message delivery guarantee: exactly-once delivery. Ack-based system, retry 3 times, then dead letter queue."),
    ("asst", "Client ack with 5s timeout, retry 3x exponential backoff, undelivered to DLQ in Redis after 3 failures."),
    ("user", "Security: end-to-end encryption for DMs. X25519 key exchange, AES-256-GCM for message encryption."),
    ("asst", "X25519 ECDH key exchange on connect. AES-256-GCM per message. Server never sees plaintext DMs."),
    ("user", "Revision: MongoDB TTL 30 days is too short for compliance. Change to 2 years retention."),
    ("asst", "MongoDB TTL index updated to 730 days (2 years). Compliance requirement documented."),
    ("user", "Rate limiting: 60 messages per minute per user. Sliding window in Redis. Exceed limit returns 429."),
    ("asst", "Sliding window rate limit in Redis sorted sets. 60 msg/min per user. 429 with Retry-After on breach."),
]

def run_live_test():
    print("=" * 60)
    print("DSPM v0.2.0 -- LIVE DATA TEST")
    print("=" * 60)

    client = make_client(provider="openrouter", model_override="meta-llama/llama-3.3-70b-instruct")
    engine = DSPMEngine(budget=400)
    memory = []

    for idx, (role, text) in enumerate(LIVE_CONVERSATION):
        patches = extract_patches(text, idx, client, dialogue_name="live_test")
        merge_into_memory(memory, patches)
        critical = sum(1 for p in patches if p.patch_type in {"constraint","decision"})
        print(f"  Turn {idx:02d} | new={len(patches)} (critical={critical}) | memory={len(memory)}")

    print()
    critical_total = sum(1 for p in memory if p.patch_type in {"constraint","decision"})
    selected, diag = engine.compress(memory, "final architecture with all constraints", len(LIVE_CONVERSATION)-1)
    critical_selected = sum(1 for p in selected if p.patch_type in {"constraint","decision"})

    crr = critical_selected / critical_total * 100 if critical_total > 0 else 0
    dspm_tok = sum(p.token_cost for p in selected)

    print(f"  Total memory patches : {len(memory)}")
    print(f"  Critical total       : {critical_total}")
    print(f"  Critical selected    : {critical_selected}")
    print(f"  CRR                  : {crr:.1f}%")
    print(f"  DSPM tokens used     : {dspm_tok} / 400 budget")
    print()

    if crr == 100.0:
        print("PASS -- CRR = 100%")
    else:
        print(f"FAIL -- CRR = {crr:.1f}%  DO NOT PUBLISH")

    print()
    print("=" * 60)
    print("COMPRESSED CONTEXT:")
    print("=" * 60)
    print(engine.build_context(selected))

if __name__ == "__main__":
    run_live_test()
