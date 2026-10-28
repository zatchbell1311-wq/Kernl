"""
Verification script for Bug #4 (DSPM)

Bug #4 - crr=100 but PCI-DSS never classified as critical.

Run from the repo root with the dspm venv active:

    python scripts/verify_bug4.py
"""

from dspm import DSPMMemory
from dspm.patch import SemanticPatch


def banner(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def show_store(m, label):
    print(f"\n[{label}]")
    print(f"  total_patches      : {len(m.patches)}")
    print(f"  critical_total     : {len(m.critical_patches)}")
    s = m.stats
    print(f"  crr (pre-get_ctx)  : {s['crr']}  (context_built={s['context_built']})")
    print(f"  critical_by_role   : {s['critical_by_role']}")
    for i, p in enumerate(m.patches):
        marker = "[CRIT]" if p.is_critical else "      "
        print(f"    {marker} [{i}] type={p.patch_type!r:14s}  "
              f"is_critical={p.is_critical}  payload={p.payload!r}")


# -- Scenario 1: PCI-DSS as a constraint (the desired shape) ------
banner("Scenario 1 - PCI-DSS as a [constraint] (the happy path)")

m = DSPMMemory(budget=250)
m._merge_patch(SemanticPatch('p0', 0, 'constraint',
                             'All payments must be PCI-DSS compliant', []))
show_store(m, "PCI-DSS as [constraint]")
ctx = m.get_context(query='pci')
print(f"\nDelivered context:\n{ctx}")
s = m.stats
print(f"\ncrr (post-get_ctx) : {s['crr']}  (critical_total={s['critical_total']}, "
      f"critical_selected={s['critical_selected']})")


# -- Scenario 2: PCI-DSS as an entity (the bug shape) -------------
banner("Scenario 2 - PCI-DSS as an [entity] (the bug shape)")

m = DSPMMemory(budget=250)
m._merge_patch(SemanticPatch('p0', 0, 'entity',
                             'PCI-DSS compliance required', []))
show_store(m, "PCI-DSS as [entity]")
ctx = m.get_context(query='pci')
print(f"\nDelivered context:\n{ctx!r}")
s = m.stats
print(f"\ncrr (post-get_ctx) : {s['crr']}  (critical_total={s['critical_total']}, "
      f"critical_selected={s['critical_selected']})")
print(f"\n>> If crr=100 above AND PCI-DSS is type='entity', that's the bug:")
print(f">> vacuous 100% over zero criticals, PCI-DSS info invisible to CRR.")


# -- Scenario 3: PCI-DSS as entity + a real critical -------------
banner("Scenario 3 - PCI-DSS as [entity] + an unrelated [constraint]")

m = DSPMMemory(budget=250)
m._merge_patch(SemanticPatch('p0', 0, 'entity',
                             'PCI-DSS compliance required', []))
m._merge_patch(SemanticPatch('p1', 1, 'constraint',
                             'Webhook timeout must be 30 seconds', []))
show_store(m, "Mixed: PCI-DSS as entity + webhook as constraint")
ctx = m.get_context(query='pci')
print(f"\nDelivered context:\n{ctx}")
s = m.stats
print(f"\ncrr (post-get_ctx) : {s['crr']}  (critical_total={s['critical_total']}, "
      f"critical_selected={s['critical_selected']})")
print(f"\n>> PCI-DSS is non-critical, never in critical_total.")
print(f">> CRR's denominator only counts the webhook, so 100% looks healthy.")
print(f">> But PCI-DSS compliance info was never a critical - the metric lies.")


# -- Scenario 4: probe the patch_type vocabulary ----------------
banner("Scenario 4 - What patch_types exist and which are critical?")

try:
    from dspm.config import PATCH_TYPES
    print(f"\nPATCH_TYPES from config: {PATCH_TYPES}")
except ImportError as e:
    print(f"\nCould not import PATCH_TYPES from dspm.config: {e}")
    PATCH_TYPES = ['constraint', 'decision', 'entity', 'preference',
                   'context', 'code']

print(f"\nFor each type, is is_critical True or False?")
for t in PATCH_TYPES:
    try:
        p = SemanticPatch(f'id_{t}', 0, t, f'sample {t} payload', [])
        print(f"  type={t!r:14s}  is_critical={p.is_critical}")
    except Exception as e:
        print(f"  type={t!r:14s}  ERROR: {e}")


# -- Scenario 5: empty store (vacuous 100%) ---------------------
banner("Scenario 5 - Empty store: does crr report 100 vacuously?")

m = DSPMMemory(budget=250)
s = m.stats
print(f"\nEmpty store stats:")
print(f"  total_patches     : {s['total_patches']}")
print(f"  critical_total    : {s['critical_total']}")
print(f"  critical_selected : {s['critical_selected']}")
print(f"  crr               : {s['crr']}")
print(f"  context_built     : {s['context_built']}")
print(f"\n>> If crr=100 with critical_total=0, the formula returns 100.")
print(f">> This is the vacuous-100 case. Combined with PCI-DSS being")
print(f">> mistyped as entity, you get a healthy-looking metric that")
print(f">> hides the fact that PCI-DSS info was never critical.")


# -- Scenario 6: probe the extractor's prompt --------------------
banner("Scenario 6 - Inspect the extractor's system prompt for PCI-DSS classification")

try:
    from dspm.extractor import _build_system_prompt
    prompt = _build_system_prompt(dense=False)
    print(f"\nSystem prompt (non-dense) - looking for PCI-DSS / compliance guidance:")
    print(f"  Prompt length: {len(prompt)} chars")
    keywords = ['pci', 'compliance', 'constraint', 'decision', 'entity',
                'critical', 'security', 'regulatory']
    for kw in keywords:
        if kw.lower() in prompt.lower():
            idx = prompt.lower().find(kw.lower())
            start = max(0, idx - 40)
            end = min(len(prompt), idx + 80)
            snippet = prompt[start:end].replace('\n', ' ')
            print(f"  [{kw!r}] found: ...{snippet}...")
        else:
            print(f"  [{kw!r}] NOT in prompt")

    print(f"\n--- Full prompt (first 2000 chars) ---")
    print(prompt[:2000])
    if len(prompt) > 2000:
        print(f"\n... (truncated, total {len(prompt)} chars)")

except ImportError as e:
    print(f"\nCould not import _build_system_prompt from dspm.extractor: {e}")
    print(f"(This is fine - we'll need to see extractor.py directly.)")
except Exception as e:
    print(f"\nError inspecting prompt: {e}")


print("\n" + "=" * 70)
print("DONE - paste this entire output back to me for analysis.")
print("=" * 70)