# release_prep_0_1_17.py
# DSPM v0.1.17 release prep - run ONCE from the project root.
# Never deletes anything; safe to re-run.
# 1) bumps 0.1.16 -> 0.1.17 in pyproject.toml (+ __init__.py if present)
# 2) puts the v0.1.17 entry on top of CHANGELOG.md
# 3) updates stale "0.1.16" mentions in README.md and prints any
#    T7 / adaptive-budgeting lines so you can check the wording
# 4) prints the exact command sheet for steps 3-9

import re
import sys
from pathlib import Path

OLD = "0.1.16"
NEW = "0.1.17"
TODAY = "2026-10-22"

root = Path.cwd()
pyproject = root / "pyproject.toml"

if not pyproject.exists():
    sys.exit("pyproject.toml not found in this folder. Open the terminal in the "
             "DSPM project root (the folder with pyproject.toml) and run again.")

text = pyproject.read_text(encoding="utf-8")

m = re.search(r'(?m)^\s*name\s*=\s*["\']([^"\']+)["\']', text)
PKG = m.group(1) if m else "dspm-memory"
print(f"[info] project root : {root}")
print(f"[info] package name : {PKG}  (used in the pip / PyPI commands)")

# ---------- 1) version bump ----------
have = re.search(r'(?m)^version\s*=\s*["\']([^"\']+)["\']', text)
cur = have.group(1) if have else None
if cur == NEW:
    print(f"[version] pyproject.toml already at {NEW} - nothing to do")
elif cur == OLD:
    text = re.sub(rf'(?m)^version\s*=\s*["\']{re.escape(OLD)}["\']',
                  f'version = "{NEW}"', text)
    pyproject.write_text(text, encoding="utf-8")
    print(f"[version] pyproject.toml bumped {OLD} -> {NEW}")
else:
    sys.exit(f'ERROR: expected version "{OLD}" in pyproject.toml but found "{cur}". '
             f'Open pyproject.toml, set  version = "{NEW}",  save, run this again.')

for cand in (root / "src" / "dspm" / "__init__.py", root / "dspm" / "__init__.py"):
    if cand.exists():
        t = cand.read_text(encoding="utf-8")
        t2, n = re.subn(rf'__version__\s*=\s*["\']{re.escape(OLD)}["\']',
                        f'__version__ = "{NEW}"', t)
        if n:
            cand.write_text(t2, encoding="utf-8")
            print(f"[version] {cand.relative_to(root)}: __version__ {OLD} -> {NEW}")
        elif "__version__" in t:
            print(f'[version] {cand.relative_to(root)}: __version__ is not "{OLD}" - '
                  f'set __version__ = "{NEW}" yourself')

# ---------- 2) CHANGELOG.md ----------
ENTRY = f"""## [{NEW}] - {TODAY}

### Fixed
- **Cross-type restatements collapse (review finding S1)** - CON<->DEC restatements across
  types now collapse (rule b3); the live smoke CRR failure they caused is gone.
- **Thin cross-type echoes (review finding S2)** - the old frozen fallback rule is removed and
  replaced by the bound-head bridge (rule b4): echoes sharing the same bound head with two or
  more shared non-numeric words, equal numbers, equal units and compatible polarity now mark
  the old fact stale. The previously frozen fee-pair tests were unfrozen and rewritten to
  assert the (b4) collapse.
- **Docs honesty (review fixes)** - **T7 adaptive budgeting is RESERVED, NOT WIRED**: ema_query
  is only ever reset, never updated, and ALPHA_EMA, SHADOW_THRESHOLD, DELTA_MIN_SAVING,
  BASE_BUDGET_FRACTIONS, CRITICAL_MAX_WORDS and REVISION_OVERLAP are unwired config - do not
  quote T7 as active behavior. T2 slot fusion keeps the highest (utility, turn_index), which
  in practice is the newest turn (utility is 0.0 at that stage).

### Added
- **Patch source roles** - extracted facts remember whether they came from the user or the
  assistant; roles survive save/load, stats exposes critical_by_role, and rendered criticals
  are annotated, e.g. "fee cap set to 0.5% as tracked (user)".
- **Split token accounting** - stats reports conversation_tokens (raw turns), patch_tokens
  (extracted) and context_tokens (delivered) separately.
- **Evaluation harness** (eval_harness.py) - five scenarios, all green (AGGREGATE 5/5):
  payments_echo, tier_trap, revision_chain, starvation_honesty, trim_quality; --live mode
  adds latency / token / cost accounting.
- **14 new tests** (12 memory + 2 persistence); full suite 94/94 green.
- **Verified replays** - budget 59: crr=100 with two spare capacity slots; variance probe
  delivers 7/7; starvation honesty reports store=8, delivered=3, crr=37 instead of
  over-delivering.

### Known issues (repro locked, fix queued for 0.1.18)
- **S3 trim quality** - at budget 59 the delivered revision line can lose the (was X) span
  form and the subject noun ("timeout 10 seconds, 30 seconds"); the current value always
  survives. Locked in as the trim_quality harness repro.
- **(b4) scope-noun limitation** - same-number scope-noun pairs (e.g. org-admins /
  max-users 500) merge by design; structured slots are the planned 0.1.18 fix. Documented by
  tier_trap's fifth split guard.
- Also queued for 0.1.18: rule (a) multi-overlapping-slot sweep; rule (c) polarity-free pairs.

"""

ch = root / "CHANGELOG.md"
if ch.exists() and f"## [{NEW}]" in ch.read_text(encoding="utf-8"):
    print("[changelog] v0.1.17 entry already present - nothing to do")
else:
    if ch.exists():
        old = ch.read_text(encoding="utf-8")
        lines = old.splitlines(keepends=True)
        if lines and lines[0].lstrip().startswith("#"):
            new_text = lines[0] + "\n" + ENTRY + "".join(lines[1:])
        else:
            new_text = ENTRY + old
    else:
        new_text = "# Changelog\n\n" + ENTRY
    ch.write_text(new_text, encoding="utf-8")
    print("[changelog] CHANGELOG.md updated - v0.1.17 entry added on top")

# ---------- 3) README.md ----------
rd = root / "README.md"
if rd.exists():
    t = rd.read_text(encoding="utf-8")
    n = t.count(OLD)
    if n:
        rd.write_text(t.replace(OLD, NEW), encoding="utf-8")
        print(f"[readme] README.md updated - {n} mention(s) of {OLD} -> {NEW}")
    else:
        print("[readme] README.md has no '0.1.16' mentions - nothing to do")
    hits = [(i + 1, ln.strip()) for i, ln in enumerate(t.splitlines())
            if re.search(r"T7|adaptive budget|highest utility", ln, re.I)]
    if hits:
        print("[readme] README lines mentioning T7 / adaptive budgeting / highest utility:")
        for i, ln in hits:
            print(f"    line {i}: {ln}")
        print("[readme] if any of these claim T7/T2 as an ACTIVE feature, change the")
        print("[readme] sentence to the reserved-not-wired wording (see CHANGELOG entry).")
    else:
        print("[readme] no T7 / adaptive-budgeting mentions found - wording is clean")
else:
    print("[readme] README.md not found - skipped")

# ---------- command sheet ----------
sep = "=" * 66
print()
print(sep)
print("PREP DONE - your remaining commands (PowerShell, project root):")
print(sep)
print(f"""
3) One-time tool install (skip if you already have build + twine):
   python -m pip install --upgrade build twine

4) Build (clean any old dist first so only 0.1.17 gets uploaded):
   if (Test-Path dist) {{ Remove-Item -Recurse -Force dist }}
   python -m build

5) Check the built files (both must say PASSED):
   python -m twine check dist/*

6) Dry-run upload to TestPyPI (account is SEPARATE - register on test.pypi.org):
   python -m twine upload --repository testpypi dist/*
   -> username: __token__   password: your TestPyPI API token

7) Install-check from TestPyPI in a throwaway venv (no activation needed):
   python -m venv $env:TEMP\\dspm_testenv
   $env:TEMP\\dspm_testenv\\Scripts\\python.exe -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ {PKG}=={NEW}
   $env:TEMP\\dspm_testenv\\Scripts\\python.exe -c "import dspm; from importlib.metadata import version; print('TestPyPI OK:', version('{PKG}'))"

8) Real upload to PyPI (0.1.16 was never published, so 0.1.17 just becomes newest):
   python -m twine upload dist/*
   -> username: __token__   password: your PyPI API token

9) Git commit, tag, push:
   git branch --show-current
   git add pyproject.toml CHANGELOG.md README.md
   git add src tests scripts ":(exclude)*.f2bak" ":(exclude)*.f3bak" ":(exclude)*.f5bak"
   git status
   git commit -m "Release v0.1.17"
   git tag -a v0.1.17 -m "DSPM v0.1.17"
   git push origin v0.1.17
   git push origin v0.1.16-dev
""")
print("The moment the 20-run loop is green: revoke the temporary GROQ")
print("key in the Groq console. Budget 59 stays untouched.")