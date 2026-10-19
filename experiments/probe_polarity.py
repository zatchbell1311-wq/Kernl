from dspm.memory import DSPMMemory
from dspm.patch import SemanticPatch

cases = [
    ("Must NOT use Redis for session storage", "redis session storage"),
    ("API latency must be under 2 seconds", "api latency"),
    ("Retries must be no more than 3 per request", "retries"),
    ("Audit logs must not be stored longer than 90 days", "audit logs"),
]
for text, query in cases:
    print("\n==", text)
    for budget in (4, 6, 8, 10, 14, 20, 40):
        m = DSPMMemory(budget=budget)
        m._merge_patch(SemanticPatch('p0_0', 0, 'constraint', text, []))
        print(f"  budget={budget:>2}  ->  {m.get_context(query=query)!r}")