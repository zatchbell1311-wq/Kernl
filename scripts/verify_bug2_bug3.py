"""
Verification script for Bugs #2 and #3 (DSPM)

Bug #2 - Duplicate rendering: 'Deadline updated to Monday' should
         supersede 'Deadline set to Friday', not render both.
Bug #3 - Non-numeric revision: the (was X) history should bake the
         day name correctly ('was Friday'), not show a numeric
         counter that flips to 'Monday'.

Run from the repo root with the dspm venv active:

    python scripts/verify_bug2_bug3.py
"""

from dspm import DSPMMemory
from dspm.patch import SemanticPatch


def banner(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def show_store(m, label):
    print(f"\n[{label}]")
    print(f"  turns              : {m.turns}")
    print(f"  total_patches      : {len(m.patches)}")
    print(f"  critical_patches   : {len(m.critical_patches)}")
    print(f"  crr (pre-get_ctx)  : {m.stats['crr']}")
    for i, p in enumerate(m.critical_patches):
        print(f"    [{i}] type={p.patch_type!r:14s} "
              f"is_critical={p.is_critical}  payload={p.payload!r}")


def show_context(m, query, label):
    ctx = m.get_context(query=query)
    print(f"\n[{label}] Delivered context (query={query!r}):")
    print(ctx if ctx else "(empty)")
    print(f"  crr (post-get_ctx) : {m.stats['crr']}")
    return ctx


# -- Bug #2: duplicate rendering ----------------------------------
banner("BUG #2 - Duplicate rendering (Deadline updated/moved to Monday)")

# Scenario A: same-type, days differ, 'updated' marker
m = DSPMMemory(budget=250)
m._merge_patch(SemanticPatch('p0', 0, 'decision',
                             'Deadline set to Friday', []))
m._merge_patch(SemanticPatch('p1', 1, 'decision',
                             'Deadline updated to Monday', []))
show_store(m, "Scenario A: set Friday -> update to Monday (same-type DEC)")
show_context(m, 'deadline', 'Scenario A delivered')

# Scenario B: same-type, days differ, 'moved' marker
m = DSPMMemory(budget=250)
m._merge_patch(SemanticPatch('p0', 0, 'decision',
                             'Deadline set to Friday', []))
m._merge_patch(SemanticPatch('p1', 1, 'decision',
                             'Deadline moved to Monday', []))
show_store(m, "Scenario B: set Friday -> move to Monday (same-type DEC)")

# Scenario C: same-type, days MATCH (restatement should collapse)
m = DSPMMemory(budget=250)
m._merge_patch(SemanticPatch('p0', 0, 'decision',
                             'Deadline updated to Monday', []))
m._merge_patch(SemanticPatch('p1', 1, 'decision',
                             'Deadline moved to Monday', []))
show_store(m, "Scenario C: updated to Monday + moved to Monday (same day)")

# Scenario D: cross-type (CON -> DEC), days differ
m = DSPMMemory(budget=250)
m._merge_patch(SemanticPatch('p0', 0, 'constraint',
                             'Deadline set to Friday', []))
m._merge_patch(SemanticPatch('p1', 1, 'decision',
                             'Deadline moved to Monday', []))
show_store(m, "Scenario D: CON Friday -> DEC Monday (cross-type)")

# Scenario E: scoped - 'Invoice deadline' vs 'Report deadline' (must NOT merge)
m = DSPMMemory(budget=250)
m._merge_patch(SemanticPatch('p0', 0, 'decision',
                             'Invoice deadline set to Friday', []))
m._merge_patch(SemanticPatch('p1', 1, 'decision',
                             'Report deadline moved to Monday', []))
show_store(m, "Scenario E: Invoice vs Report deadline (must stay 2)")


# -- Bug #3: non-numeric revision ---------------------------------
banner("BUG #3 - Non-numeric revision history (Friday -> Monday)")

# Single revision
m = DSPMMemory(budget=250)
m._merge_patch(SemanticPatch('p0', 0, 'decision',
                             'Deadline set to Friday', []))
m._merge_patch(SemanticPatch('p1', 1, 'decision',
                             'Deadline updated to Monday', []))
crits = [p.payload for p in m.critical_patches]
print(f"\nSingle revision result : {crits[0]!r}")
print(f"Expected               : 'Deadline updated to Monday (was Friday)'")
print(f"Contains 'Monday'      : {'Monday' in crits[0]}")
print(f"Contains '(was Friday)': {'(was Friday)' in crits[0]}")

# Chain: Fri -> Mon -> Wed
m = DSPMMemory(budget=250)
m._merge_patch(SemanticPatch('p0', 0, 'decision',
                             'Deadline set to Friday', []))
m._merge_patch(SemanticPatch('p1', 1, 'decision',
                             'Deadline updated to Monday', []))
m._merge_patch(SemanticPatch('p2', 2, 'decision',
                             'Deadline moved to Wednesday', []))
crits = [p.payload for p in m.critical_patches]
print(f"\nChain (Fri->Mon->Wed)  : {crits[0]!r}")
print(f"Expected               : 'Deadline moved to Wednesday (was Monday then Friday)'")

# What the dump_store journal looks like
m = DSPMMemory(budget=250)
m._merge_patch(SemanticPatch('p0', 0, 'decision',
                             'Deadline set to Friday', []))
m._merge_patch(SemanticPatch('p1', 1, 'decision',
                             'Deadline updated to Monday', []))
dump = m.dump_store(include_history=True)
print(f"\n--- dump_store(include_history=True) ---")
print(f"patches (active)      : {len(dump['patches'])}")
print(f"superseded log entries: {len(dump['superseded'])}")
for entry in dump['superseded']:
    print(f"  {entry}")


# -- Render check -------------------------------------------------
banner("RENDER CHECK - Does the delivered context show 1 or 2 deadline lines?")

m = DSPMMemory(budget=250)
m._merge_patch(SemanticPatch('p0', 0, 'decision',
                             'Deadline set to Friday', []))
m._merge_patch(SemanticPatch('p1', 1, 'decision',
                             'Deadline updated to Monday', []))
ctx = m.get_context(query='deadline')
print(f"\nFull delivered context:\n{ctx}")
lines = ctx.splitlines()
deadline_lines = [l for l in lines
                  if 'deadline' in l.lower()
                  or 'monday' in l.lower()
                  or 'friday' in l.lower()]
print(f"\nDeadline-related lines: {len(deadline_lines)}")
print(f"Expected              : exactly 1")
print(f"\nVERDICT:")
if len(deadline_lines) == 1:
    print("  BUG #2 + #3: FIXED - single superseded patch with (was Friday) baked in.")
else:
    print(f"  BUG #2 + #3: STILL BROKEN - {len(deadline_lines)} deadline lines rendered.")
    for i, l in enumerate(deadline_lines):
        print(f"    [{i}] {l!r}")


# -- Tight budget render check ------------------------------------
banner("TIGHT BUDGET - Does starvation trim lose the day name?")

m = DSPMMemory(budget=18)
m._merge_patch(SemanticPatch('p0', 0, 'decision',
                             'Deadline set to Friday', []))
m._merge_patch(SemanticPatch('p1', 1, 'decision',
                             'Deadline updated to Monday', []))
ctx = m.get_context(query='deadline')
print(f"\nBudget=18 delivered context:\n{ctx!r}")
print(f"\nVERDICT:")
if 'Monday' in ctx and 'Friday' in ctx:
    print("  Day names survive tight trim - good.")
elif 'Monday' in ctx:
    print("  Current day survives but history lost - partial bug.")
else:
    print("  Day names lost at tight budget - bug present.")

print("\n" + "=" * 70)
print("DONE - paste this entire output back to me for analysis.")
print("=" * 70)