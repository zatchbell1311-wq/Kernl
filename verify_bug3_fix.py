"""
Standalone verification for Bug #3 fixes (no LLM, no pytest needed).

Run from the repo root after copying the updated files:

    python scripts/verify_bug3_fix.py

Confirms:
  - Bug #3a: baked day names preserve capitalization ('Friday' not 'friday')
  - Bug #3b: revision chains preserve full day lineage ('Monday then Friday')
  - Bug #3 guard: same-day restatements don't bake
  - Bug #3 guard: scoped deadlines stay apart
"""

from dspm import DSPMMemory
from dspm.patch import SemanticPatch


def banner(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# Bug #3a: capitalization
banner("Bug #3a - Day name capitalization in (was X) span")
m = DSPMMemory(budget=250)
m._merge_patch(SemanticPatch('p0', 0, 'decision', 'Deadline set to Friday', []))
m._merge_patch(SemanticPatch('p1', 1, 'decision', 'Deadline updated to Monday', []))
crits = [p.payload for p in m.critical_patches]
print(f"  Result   : {crits[0]!r}")
print(f"  Expected : 'Deadline updated to Monday (was Friday)'")
assert '(was Friday)' in crits[0], f"FAIL: capitalization lost: {crits[0]!r}"
assert '(was friday)' not in crits[0], f"FAIL: lowercased leak: {crits[0]!r}"
print("  PASS: capitalization preserved")


# Bug #3b: chain
banner("Bug #3b - Full day lineage in chain (Fri -> Mon -> Wed)")
m = DSPMMemory(budget=250)
m._merge_patch(SemanticPatch('p0', 0, 'decision', 'Deadline set to Friday', []))
m._merge_patch(SemanticPatch('p1', 1, 'decision', 'Deadline updated to Monday', []))
m._merge_patch(SemanticPatch('p2', 2, 'decision', 'Deadline moved to Wednesday', []))
crits = [p.payload for p in m.critical_patches]
print(f"  Result   : {crits[0]!r}")
print(f"  Expected : 'Deadline moved to Wednesday (was Monday then Friday)'")
assert '(was Monday then Friday)' in crits[0], f"FAIL: chain truncated: {crits[0]!r}"
print("  PASS: full lineage chained")


# Bug #3 guard: same-day restatement
banner("Bug #3 guard - Same-day restatement (no baking)")
m = DSPMMemory(budget=250)
m._merge_patch(SemanticPatch('p0', 0, 'decision', 'Deadline updated to Monday', []))
m._merge_patch(SemanticPatch('p1', 1, 'decision', 'Deadline moved to Monday', []))
crits = [p.payload for p in m.critical_patches]
print(f"  Result   : {crits[0]!r}")
print(f"  Expected : 'Deadline moved to Monday' (no (was ...) span)")
assert '(was' not in crits[0], f"FAIL: same-day restatement baked: {crits[0]!r}"
print("  PASS: no baking for same-day restatement")


# Bug #3 guard: scoped deadlines
banner("Bug #3 guard - Scoped deadlines stay apart")
m = DSPMMemory(budget=250)
m._merge_patch(SemanticPatch('p0', 0, 'decision', 'Invoice deadline set to Friday', []))
m._merge_patch(SemanticPatch('p1', 1, 'decision', 'Report deadline moved to Monday', []))
crits = [p.payload for p in m.critical_patches]
print(f"  Result   : {len(crits)} critical(s)")
print(f"  Expected : 2 criticals (Invoice + Report)")
assert len(crits) == 2, f"FAIL: scoped deadlines merged: {crits}"
print("  PASS: scoped deadlines stay apart")


# Bug #4b: CRR honesty
banner("Bug #4b - CRR honest when no criticals post-build")
m = DSPMMemory(budget=250)
m._merge_patch(SemanticPatch('p0', 0, 'entity', 'PCI-DSS compliance required', []))
m._merge_patch(SemanticPatch('p1', 1, 'entity', 'HIPAA compliant', []))
print(f"  Pre-build crr  : {m.stats['crr']} (expected 100)")
assert m.stats['crr'] == 100, "FAIL: pre-build should be 100"
m.get_context(query='compliance')
s = m.stats
print(f"  Post-build crr : {s['crr']} (expected 0)")
print(f"  critical_total : {s['critical_total']} (expected 0)")
assert s['crr'] == 0, f"FAIL: vacuous 100% post-build: {s}"
print("  PASS: CRR honest about empty critical set")


# Bug #4a: prompt has compliance rule
banner("Bug #4a - Extractor prompt classifies compliance as constraint")
from dspm.extractor import _build_system_prompt
for dense in (True, False):
    prompt = _build_system_prompt(dense=dense)
    prompt_lower = prompt.lower()
    has_rule = ('pci-dss' in prompt_lower
                or 'hipaa' in prompt_lower
                or 'compliance' in prompt_lower)
    print(f"  dense={dense}: compliance rule present = {has_rule}")
    assert has_rule, f"FAIL: dense={dense} prompt lacks compliance rule"
print("  PASS: both prompts have compliance classification rule")


print("\n" + "=" * 70)
print("ALL VERIFICATIONS PASSED - copy the updated files to your repo")
print("=" * 70)
print("\nFiles to copy:")
print("  memory.py       -> src/dspm/memory.py")
print("  extractor.py    -> src/dspm/extractor.py")
print("  test_memory.py  -> tests/test_memory.py")