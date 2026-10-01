import os
from openai import OpenAI
from dspm import DSPMMemory

KEY = os.environ.get("GROQ_API_KEY", "")
assert KEY, "Set key first: $env:GROQ_API_KEY = 'gsk_...'"

llm = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=KEY)
memory = DSPMMemory(budget=250, llm_client=llm, model="openai/gpt-oss-20b")

print("== Turn 1 ==")
p1 = memory.add_turn("user", "Build an API with FastAPI. p95 latency must stay under 200ms. Use PostgreSQL 15.")
print(f"   extracted {len(p1)} patches")
for p in p1:
    print(f"   [{p.patch_type[:4].upper()}] {p.payload}")

print("== Turn 2 ==")
p2 = memory.add_turn("assistant", "FastAPI chosen with async SQLAlchemy; Redis cache TTL raised 60s to 300s for listings.")
print(f"   extracted {len(p2)} patches")
for p in p2:
    print(f"   [{p.patch_type[:4].upper()}] {p.payload}")

print("\n== Compressed context ==")
print(memory.get_context(query="what are the constraints?"))

print("\n== Stats ==")
print(memory.stats)

print("\n== VERDICT ==")
crits = memory.critical_patches
total = len(p1) + len(p2)
print(f"[{'PASS' if total >= 2 else 'FAIL'}] Extraction: {total} patches (need >= 2)")
print(f"[{'PASS' if len(crits) >= 1 else 'FAIL'}] Criticals found: {len(crits)}")
if crits:
    has_200 = any("200" in p.payload for p in crits)
    print(f"[{'PASS' if has_200 else 'FAIL'}] 'p95 under 200ms' constraint captured")
print(f"[{'PASS' if memory.stats['crr'] == 100 else 'FAIL'}] CRR guarantee: {memory.stats['crr']}%")
