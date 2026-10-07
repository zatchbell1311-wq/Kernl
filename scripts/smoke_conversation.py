"""v0.1.13 smoke test - CONVERSATION mode.

The gate the v0.1.10 release lacked: prompt changes were verified in
document mode only, and the config-!=constraint rule silently
misclassified conversation value directives ('set the webhook timeout
to 30 seconds') as code. This script runs the 8-turn webhook revision
conversation against a real model and verifies the full chain:
directive classified as a decision -> supersession fires -> (was X)
baked -> the story survives starvation trimming.

RELEASE RULE (since v0.1.13): extraction-layer changes require BOTH
scripts/smoke_dense.py AND this script green before twine.

Requires: GROQ_API_KEY set in your environment.
Note: extraction is stochastic - one FAIL deserves one re-run; two
consecutive FAILs means the rule is not holding, iterate before
shipping.
"""
import os
from openai import OpenAI
from dspm import DSPMMemory

KEY = os.environ.get("GROQ_API_KEY", "")
assert KEY, "Set key first: $env:GROQ_API_KEY = 'gsk_...' (fresh key)"

llm = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=KEY)
memory = DSPMMemory(budget=59, llm_client=llm, model="openai/gpt-oss-120b")

CONVERSATION = [
    ("user", "I'm building a fintech app. Non-negotiable: all payments must be PCI-DSS compliant, and max fee per transaction is 0.5%."),
    ("assistant", "Noted. For PCI-DSS compliance we'll need tokenized card storage and quarterly ASV scans. I'll track the 0.5% fee cap."),
    ("user", "We're using Stripe for payments. Set the webhook timeout to 30 seconds."),
    ("assistant", "Stripe integration configured with a 30-second webhook timeout and automatic retries on failure."),
    ("user", "Change the webhook timeout from 30 seconds to 10 seconds - 30 is too slow."),
    ("assistant", "Webhook timeout updated to 10 seconds; the 30s value is superseded."),
    ("user", "Also we need audit logs stored for exactly 5 years, and the dashboard must load in under 2 seconds."),
    ("assistant", "Audit retention set to 5 years. For the sub-2-second dashboard we'll use Redis caching with a CDN."),
]

print("Adding turns...")
for role, text in CONVERSATION:
    patches = memory.add_turn(role, text)
    print(f"  [{role[:4].upper():<4}] {len(patches)} patches")

webhook = [p.payload for p in memory.critical_patches if "webhook" in p.payload.lower()]
print(f"\nwebhook criticals: {len(webhook)}")
for w in webhook:
    print(f"  -> {w}")

ctx = memory.get_context(query="What is the current webhook timeout, and what value did it replace?")
print(f"\nrevision-question ctx:\n{ctx}")

print("\nVERDICT")
crit_ok = len(webhook) == 1
print(f"[{'PASS' if crit_ok else 'FAIL'}] exactly one webhook critical: {len(webhook)}")
if crit_ok:
    story_ok = ("10" in webhook[0] and "was" in webhook[0].lower()
                and "30" in webhook[0] and "(was 10)" not in webhook[0])
    print(f"[{'PASS' if story_ok else 'FAIL'}] revision story baked (10, was 30 - not (was 10))")
ctx_ok = "10" in ctx and "30" in ctx
print(f"[{'PASS' if ctx_ok else 'FAIL'}] both values survive trimming in ctx (10 and 30)")
crr_ok = memory.stats["crr"] == 100
print(f"[{'PASS' if crr_ok else 'FAIL'}] CRR 100%: {memory.stats['crr']}%")
print("\n(if CRR alone fails: critical-count bloat from restatement duplicates - the known v0.1.14 issue, not the prompt fix)")
