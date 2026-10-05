import os, sys, time
from openai import OpenAI
from dspm import DSPMMemory
from dspm.patch import count_tokens

KEY = os.environ.get("GROQ_API_KEY", "")
assert KEY and "YOUR_KEY" not in KEY, "Set a real key: $env:GROQ_API_KEY = 'gsk_...'"

llm = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=KEY)

def pick_model():
    for m in ["openai/gpt-oss-20b", "qwen/qwen3.8-27b", "openai/gpt-oss-120b"]:
        try:
            r = llm.chat.completions.create(model=m,
                messages=[{"role": "user", "content": "Reply with exactly: OK"}],
                max_tokens=512, temperature=0.0)
            if (r.choices[0].message.content or "").strip():
                print(f"[model] using {m}")
                return m
        except Exception as e:
            print(f"[model] {m} unavailable: {str(e)[:60]}")
    sys.exit("No working model on Groq")

MODEL = pick_model()

# 18-turn adversarial conversation: early buried constraint, a revision,
# a verbatim restatement, dense competing criticals.
CONV = [
    ("user",      "We are building a payment-processing platform. Stack: FastAPI with PostgreSQL 15 and async SQLAlchemy."),
    ("user",      "Hard requirement, non-negotiable: monthly uptime SLA of 99.9%. This was promised to the customer."),
    ("user",      "Session cache TTL must be 60 seconds for security compliance."),
    ("assistant", "Stack locked: FastAPI, PostgreSQL 15, async SQLAlchemy. Redis for sessions with the 60s TTL."),
    ("user",      "Authentication: JWT with HS256. Access tokens 15 minutes, refresh tokens 30 days."),
    ("assistant", "JWT HS256 chosen; access tokens expire in 15 minutes; refresh tokens rotate every 30 days with reuse detection."),
    ("user",      "All PII must be encrypted at rest with AES-256. No exceptions."),
    ("assistant", "AES-256 at rest for all PII fields; keys managed in AWS KMS with quarterly rotation."),
    ("user",      "Anomaly detection: alert when transaction z-score exceeds 3 for 5 consecutive minutes."),
    ("assistant", "Anomaly rule: z-score > 3 sustained for 5 consecutive minutes triggers a P1 alert with 10-minute cooldown."),
    ("user",      "Audit logs must be retained for 7 years for SOC2 compliance."),
    ("assistant", "Audit retention 7 years: hot tier 90 days, then S3 Glacier Deep Archive, legal-hold flag overrides lifecycle."),
    ("user",      "Revision: raise the session cache TTL from 60 seconds to 300 seconds. The 60s value caused throttling."),
    ("assistant", "Session cache TTL updated to 300 seconds; the 60s setting is superseded."),
    ("user",      "Rate limiting: 1000 requests per minute per merchant ID, respond 429 with Retry-After."),
    ("assistant", "Token-bucket rate limit: 1000 req/min per merchant ID, 429 plus Retry-After on exhaustion."),
    ("user",      "Restating the hard requirement so it is not lost: monthly uptime SLA of 99.9% - non-negotiable."),
    ("assistant", "Confirmed constraints: 99.9% uptime SLA, AES-256 PII at rest, 7-year audit retention, 1000 req/min rate limit, 300s session TTL."),
]

raw_text = "\n".join(f"{r.upper()}: {t}" for r, t in CONV)
RAW = count_tokens(raw_text)
print(f"\nRaw conversation: {RAW} tokens across {len(CONV)} turns\n")

memory = DSPMMemory(budget=400, llm_client=llm, model=MODEL)

def add(role, text):
    for _ in (1, 2):
        try:
            return memory.add_turn(role, text)
        except Exception as e:
            print(f"   ! API error ({str(e)[:60]}) - retrying in 20s")
            time.sleep(20)
    return []

print("=== EXTRACTION (18 turns, ~2-4 min) ===")
for i, (role, text) in enumerate(CONV):
    p = add(role, text)
    if not p:  # reasoning models occasionally return empty - one nudge retry
        time.sleep(2)
        p = add(role, text)
    kinds = ",".join(sorted({q.patch_type[:4] for q in p})) or "-"
    print(f"  T{i:02d} [{role[:4]}] -> {len(p)} patches ({kinds})")
    time.sleep(3)

crits = memory.critical_patches
allp = memory.all_patches
print(f"\nMemory: {len(allp)} patches | {len(crits)} criticals")
for p in crits:
    print(f"  [{p.patch_type[:4].upper()}] {p.payload}")

QUERY = "What are the hard requirements, constraints, and final decisions?"

print("\n=== BUDGET FRONTIER ===")
hdr = f"{'budget':>6} | {'ctx':>4} | {'TRR':>6} | {'CRR':>10} | 99.9 alive"
print(hdr); print("-" * len(hdr))
results = {}
for b in [100, 150, 250, 400]:
    memory.engine.budget = b
    ctx = memory.get_context(query=QUERY)
    ct = count_tokens(ctx)
    sel = [p for p in memory.selected_patches if p.is_critical]
    crr = 100 * len(sel) // max(1, len(crits))
    trr = (1 - ct / RAW) * 100
    alive = "99.9" in ctx
    results[b] = (ct, crr, alive, ctx)
    print(f"{b:>6} | {ct:>4} | {trr:>5.1f}% | {len(sel):>2}/{len(crits):<2}={crr:>3}% | {'YES' if alive else 'NO '}")

ttl = [p for p in crits if "ttl" in p.payload.lower()]
old_only = [p for p in ttl if "60" in p.payload and "300" not in p.payload]
dup_count = sum(1 for p in crits if "99.9" in p.payload)

print("\n=== VERDICT (the hard checks) ===")
print(f"[{'PASS' if len(allp) >= 12 else 'FAIL'}] Extraction volume: {len(allp)} patches (need >= 12)")
print(f"[{'PASS' if len(crits) >= 6 else 'FAIL'}] Critical mass: {len(crits)} criticals (need >= 6)")
core = all(results[b][1] == 100 for b in [150, 250, 400])
print(f"[{'PASS' if core else 'FAIL'}] CRR = 100% at budgets 150/250/400")
print(f"[{'PASS' if results[100][1] >= 90 else 'FAIL'}] CRR >= 90% at brutal budget 100 (got {results[100][1]}%)")
early = all(results[b][2] for b in [150, 250, 400])
print(f"[{'PASS' if early else 'FAIL'}] Buried early constraint (99.9% uptime) survives every feasible budget")
rev = len(ttl) >= 1 and any("300" in p.payload for p in ttl) and not old_only
print(f"[{'PASS' if rev else 'FAIL'}] Revision supersession: TTL critical holds NEW value (300s), old 60s version gone")
print(f"[{'PASS' if dup_count == 1 else 'FAIL'}] Verbatim restatement collapsed to exactly 1 critical (found {dup_count})")
cap = all(results[b][0] <= b for b in results)
print(f"[{'PASS' if cap else 'FAIL'}] Hard budget cap respected at every budget")

print("\n=== FINAL COMPRESSED CONTEXT @ 250 ===")
print(results[250][3])
