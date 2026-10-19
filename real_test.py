"""
Real-world DSPM test: an agent conversation where memory compression
actually matters. 8 turns, then compression at 3 budgets.

Requires: GROQ_API_KEY set in your environment.
"""
import os
from openai import OpenAI
from dspm import DSPMMemory

llm = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
)
memory = DSPMMemory(budget=250, llm_client=llm, model="openai/gpt-oss-20b")

# A realistic multi-turn conversation
conversation = [
    ("user", "I'm building a fintech app. Non-negotiable: all payments must be PCI-DSS compliant, and max fee per transaction is 0.5%."),
    ("assistant", "Noted. For PCI-DSS compliance we'll need tokenized card storage and quarterly ASV scans. I'll track the 0.5% fee cap."),
    ("user", "We're using Stripe for payments. Set the webhook timeout to 30 seconds."),
    ("assistant", "Stripe integration configured with a 30-second webhook timeout and automatic retries on failure."),
    ("user", "Change the webhook timeout from 30 seconds to 10 seconds — 30 is too slow."),
    ("assistant", "Webhook timeout updated to 10 seconds; the 30s value is superseded."),
    ("user", "Also we need audit logs stored for exactly 5 years, and the dashboard must load in under 2 seconds."),
    ("assistant", "Audit retention set to 5 years. For the sub-2-second dashboard we'll use Redis caching with a CDN."),
]

print("Adding turns...")
for role, text in conversation:
    patches = memory.add_turn(role, text)
    print(f"  [{role[:4]}] {len(patches)} patches extracted")

print(f"\nTotal patches: {len(memory.all_patches)}")
print(f"Criticals (constraints + decisions): {len(memory.critical_patches)}")

# THE TEST: compress at 3 budget levels
for budget in [100, 200, 300]:
    memory.set_budget(budget)  # NOT memory.engine.budget: get_context() re-syncs it
    context = memory.get_context(query="What are the compliance requirements and key settings?")
    crits_selected = sum(1 for p in memory.selected_patches if p.is_critical)
    crits_total = len(memory.critical_patches)
    print(f"\n--- Budget {budget} ---")
    print(f"Criticals retained: {crits_selected}/{crits_total}")
    print(f"CRR: {100 * crits_selected // max(1, crits_total)}%")
    print(f"Context length: {len(context)} chars")
    print(f"'PCI' in context: {'PCI' in context}")
    print(f"'0.5%' in context: {'0.5%' in context}")
    print(f"NEW value '10 seconds' in context: {'10 seconds' in context or '10s' in context}")
    print(f"OLD value '30 seconds' gone: {'30 seconds' not in context and '30s' not in context}")
    print(f"Context:\n{context}")
