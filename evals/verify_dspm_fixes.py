#!/usr/bin/env python3
"""
DSPM v0.5.0 Fix Verification Script

This script checks that ALL fixes are actually in the SOURCE FILES
(engine.py, extractor.py, patch.py, memory.py, __init__.py, client.py, pyproject.toml)
— NOT just in the one-time fix scripts.

Run:  python verify_dspm_fixes.py

If everything passes, the fixes are permanent and you can safely delete
the fix scripts (fix_dspm_v04.py, fix_dspm_v05.py).
"""

import sys
from pathlib import Path

print("=" * 70)
print("DSPM v0.5.0 FIX VERIFICATION")
print("Checking that fixes are in SOURCE FILES (not fix scripts)")
print("=" * 70)
print()

# ============================================================
# Find the DSPM source directory
# ============================================================
candidates = [
    Path.cwd() / "dspm",
    Path.cwd(),
    Path.home() / "Downloads" / "dspm_v0.2.0_package" / "dspm",
    Path.home() / "Downloads" / "dspm_v0.2.0_package",
]

import site
for sp in site.getsitepackages():
    candidates.append(Path(sp) / "dspm")

dspm_dir = None
for c in candidates:
    if (c / "patch.py").exists() and (c / "engine.py").exists():
        dspm_dir = c
        break

if dspm_dir is None:
    print("ERROR: Could not find DSPM source files.")
    sys.exit(1)

print(f"Source location: {dspm_dir}")
print()

# ============================================================
# Verify each fix is in the actual source files
# ============================================================
passed = 0
failed = 0
total = 0

def check(description, file_path, search_text, should_be_present=True):
    """Check if search_text is in file_path."""
    global passed, failed, total
    total += 1
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"  [FAIL] {description}")
        print(f"         File not found: {file_path}")
        failed += 1
        return
    content = file_path.read_text(encoding="utf-8")
    is_present = search_text in content
    if should_be_present:
        ok = is_present
    else:
        ok = not is_present

    if ok:
        print(f"  [OK]   {description}")
        print(f"         File: {file_path.name}")
        passed += 1
    else:
        print(f"  [FAIL] {description}")
        print(f"         File: {file_path.name}")
        if should_be_present:
            print(f"         Expected to find: '{search_text[:60]}...'")
        else:
            print(f"         Should NOT contain: '{search_text[:60]}...'")
        failed += 1
    print()


# ── FIX 1: patch.py — SLOT in SHORT_TAGS ──
print("-" * 70)
print("FIX 1: patch.py — SLOT in SHORT_TAGS")
print("-" * 70)
check(
    "'slot': 'SLOT' in patch.py SHORT_TAGS",
    dspm_dir / "patch.py",
    '"slot":       "SLOT"'
)

# ── FIX 2: patch.py — _fingerprint no longer duplicated ──
print("-" * 70)
print("FIX 2: patch.py — _fingerprint no longer duplicated (only 1 return)")
print("-" * 70)
patch_content = (dspm_dir / "patch.py").read_text(encoding="utf-8")
fingerprint_count = patch_content.count("return hashlib.md5(normalized.encode()).hexdigest()[:16]")
total += 1
if fingerprint_count == 1:
    print(f"  [OK]   _fingerprint has exactly 1 return statement (was 3)")
    print(f"         File: patch.py")
    passed += 1
else:
    print(f"  [FAIL] _fingerprint has {fingerprint_count} return statements (expected 1)")
    print(f"         File: patch.py")
    failed += 1
print()

# ── FIX 3: engine.py — slot in BASE_BUDGET_FRACTIONS ──
print("-" * 70)
print("FIX 3: engine.py — slot in BASE_BUDGET_FRACTIONS")
print("-" * 70)
check(
    "'slot': 0.30 in engine.py budget",
    dspm_dir / "engine.py",
    '"slot":       0.30,'
)

check(
    "'code' reduced to 0.05 in engine.py budget",
    dspm_dir / "engine.py",
    '"code":       0.05,'
)

# ── FIX 4: engine.py — slot in CRITICAL_TYPES ──
print("-" * 70)
print("FIX 4: engine.py — slot in CRITICAL_TYPES")
print("-" * 70)
check(
    "'slot' in CRITICAL_TYPES",
    dspm_dir / "engine.py",
    'CRITICAL_TYPES = {"constraint", "decision", "slot"}'
)

# ── FIX 5: engine.py — SHADOW_THRESHOLD = 0.01 ──
print("-" * 70)
print("FIX 5: engine.py — SHADOW_THRESHOLD = 0.01")
print("-" * 70)
check(
    "SHADOW_THRESHOLD = 0.01",
    dspm_dir / "engine.py",
    "SHADOW_THRESHOLD = 0.01"
)

# ── FIX 6: engine.py — _shadow_select fills total budget ──
print("-" * 70)
print("FIX 6: engine.py — _shadow_select fills TOTAL budget (not per-type)")
print("-" * 70)
check(
    "_shadow_select uses total budget check",
    dspm_dir / "engine.py",
    "if total + p.token_cost <= self.budget:"
)

# ── FIX 7: extractor.py — max 15 items (not max 5) ──
print("-" * 70)
print("FIX 7: extractor.py — max 15 items (not max 5)")
print("-" * 70)
check(
    "'max 15 items' in extractor prompt",
    dspm_dir / "extractor.py",
    "max 15 items"
)

# Old "max 5 items" should be GONE
check(
    "'max 5 items' removed from extractor",
    dspm_dir / "extractor.py",
    "max 5 items",
    should_be_present=False
)

# ── FIX 8: extractor.py — 'max 1 per turn' removed ──
print("-" * 70)
print("FIX 8: extractor.py — 'max 1 per turn' removed")
print("-" * 70)
check(
    "'max 1 per turn' removed from extractor",
    dspm_dir / "extractor.py",
    "max 1 per turn",
    should_be_present=False
)

# ── FIX 9: extractor.py — slot in type_map ──
print("-" * 70)
print("FIX 9: extractor.py — slot in type_map")
print("-" * 70)
check(
    "'slot': 'slot' in type_map",
    dspm_dir / "extractor.py",
    '"slot": "slot",'
)

# ── FIX 10: extractor.py — attraction examples in prompt ──
print("-" * 70)
print("FIX 10: extractor.py — attraction examples in prompt")
print("-" * 70)
check(
    "attraction.type = museum example",
    dspm_dir / "extractor.py",
    "attraction.type = museum"
)

check(
    "attraction.type = boat example",
    dspm_dir / "extractor.py",
    "attraction.type = boat"
)

check(
    "attraction.type = college example",
    dspm_dir / "extractor.py",
    "attraction.type = college"
)

# ── FIX 11: __init__.py — version 0.5.0 ──
print("-" * 70)
print("FIX 11: __init__.py — version 0.5.0")
print("-" * 70)
check(
    "__version__ = '0.5.0' in __init__.py",
    dspm_dir / "__init__.py",
    '__version__ = "0.5.0"'
)

# ── FIX 12: memory.py — version 0.5.0 in save() ──
print("-" * 70)
print("FIX 12: memory.py — version 0.5.0 in save()")
print("-" * 70)
check(
    "version '0.5.0' in memory.py save()",
    dspm_dir / "memory.py",
    '"version"    : "0.5.0",'
)

# ── FIX 13: client.py — stealth/ox-alpha removed ──
print("-" * 70)
print("FIX 13: client.py — stealth/ox-alpha removed")
print("-" * 70)
check(
    "'stealth/ox-alpha' removed from client.py",
    dspm_dir / "client.py",
    "stealth/ox-alpha",
    should_be_present=False
)

# ── FIX 14: pyproject.toml — version 0.5.0 ──
print("-" * 70)
print("FIX 14: pyproject.toml — version 0.5.0")
print("-" * 70)
pyproject_path = dspm_dir.parent / "pyproject.toml"
check(
    "version '0.5.0' in pyproject.toml",
    pyproject_path,
    'version     = "0.5.0"'
)

# ============================================================
# ALSO: Verify by importing and checking runtime behavior
# ============================================================
print("=" * 70)
print("RUNTIME VERIFICATION — import DSPM and check")
print("=" * 70)
print()

try:
    import dspm
    print(f"  [OK]   import dspm works, version = {dspm.__version__}")
    print()

    from dspm.patch import SHORT_TAGS
    print(f"  SHORT_TAGS = {SHORT_TAGS}")
    if "slot" in SHORT_TAGS:
        print(f"  [OK]   'slot' is in SHORT_TAGS (value: {SHORT_TAGS['slot']})")
    else:
        print(f"  [FAIL] 'slot' NOT in SHORT_TAGS")
        failed += 1
    print()

    from dspm.engine import CRITICAL_TYPES, BASE_BUDGET_FRACTIONS
    print(f"  CRITICAL_TYPES = {CRITICAL_TYPES}")
    print(f"  BASE_BUDGET_FRACTIONS = {BASE_BUDGET_FRACTIONS}")
    if "slot" in CRITICAL_TYPES:
        print(f"  [OK]   'slot' is in CRITICAL_TYPES")
    else:
        print(f"  [FAIL] 'slot' NOT in CRITICAL_TYPES")
        failed += 1
    if "slot" in BASE_BUDGET_FRACTIONS:
        print(f"  [OK]   'slot' is in BASE_BUDGET_FRACTIONS (value: {BASE_BUDGET_FRACTIONS['slot']})")
    else:
        print(f"  [FAIL] 'slot' NOT in BASE_BUDGET_FRACTIONS")
        failed += 1
    print()

    from dspm.engine import SHADOW_THRESHOLD
    print(f"  SHADOW_THRESHOLD = {SHADOW_THRESHOLD}")
    if SHADOW_THRESHOLD == 0.01:
        print(f"  [OK]   SHADOW_THRESHOLD is 0.01 (was 0.05)")
    else:
        print(f"  [FAIL] SHADOW_THRESHOLD is {SHADOW_THRESHOLD} (expected 0.01)")
        failed += 1
    print()

    # Check extractor prompt
    from dspm.extractor import EXTRACTION_SYSTEM_PROMPT
    if "max 15 items" in EXTRACTION_SYSTEM_PROMPT:
        print(f"  [OK]   Extractor prompt says 'max 15 items'")
    else:
        print(f"  [FAIL] Extractor prompt does NOT say 'max 15 items'")
        failed += 1
    if "attraction.type = museum" in EXTRACTION_SYSTEM_PROMPT:
        print(f"  [OK]   Extractor prompt has attraction.type = museum example")
    else:
        print(f"  [FAIL] Extractor prompt missing attraction examples")
        failed += 1
    if "max 1 per turn" not in EXTRACTION_SYSTEM_PROMPT:
        print(f"  [OK]   'max 1 per turn' removed from extractor prompt")
    else:
        print(f"  [FAIL] 'max 1 per turn' still in extractor prompt")
        failed += 1
    print()

except Exception as e:
    print(f"  [FAIL] Could not import dspm: {e}")
    failed += 1

# ============================================================
# SUMMARY
# ============================================================
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()
print(f"  Total checks: {total + 6}")  # +6 for runtime checks
print(f"  Passed:       {passed}")
print(f"  Failed:       {failed}")
print()

if failed == 0:
    print("  *** ALL FIXES ARE IN THE SOURCE FILES ***")
    print()
    print("  The fixes are PERMANENT — they live in:")
    print(f"    {dspm_dir}/patch.py")
    print(f"    {dspm_dir}/engine.py")
    print(f"    {dspm_dir}/extractor.py")
    print(f"    {dspm_dir}/__init__.py")
    print(f"    {dspm_dir}/memory.py")
    print(f"    {dspm_dir}/client.py")
    print(f"    {dspm_dir.parent}/pyproject.toml")
    print()
    print("  The fix scripts (fix_dspm_v04.py, fix_dspm_v05.py) are NO LONGER NEEDED.")
    print("  They already applied their changes to the source files.")
    print("  You can safely delete them.")
    print()
    print("  To verify again after deleting, just re-run this script.")
else:
    print("  *** SOME FIXES ARE MISSING ***")
    print()
    print("  Do NOT delete the fix scripts yet. Re-run them:")
    print("    python fix_dspm_v04.py")
    print("    python fix_dspm_v05.py")
    print("  Then re-run this verification script.")
print()
