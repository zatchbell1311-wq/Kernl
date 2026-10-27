# F4_cross_type_collapse.py -- v0.1.17 finding S1 (live conversation smoke):
# cross-type CON<->DEC same-value restatements never collapsed -- the user
# states a requirement that extracts as [CON], the assistant echo extracts
# as [DEC], and no supersession rule compares across types without a
# revision verb. The live smoke store bloated to 8-10 criticals and CRR
# fell to 70-87% at its 59-token budget (delivery caps at 7 criticals).
#
# Three coordinated fixes to memory.py:
#   S1-a  rule (b3): cross-type restatement collapse -- same slot anchor,
#        >=2 shared non-numeric content words, identical numbers, identical
#        units, compatible bound classes, and at least one directional or
#        exact bound (stricter than same-type (b2), because a cross-type
#        merge is riskier). Known residual, frozen by test: thin
#        restatements sharing ONE non-numeric word ('Maximum fee ... 0.5%'
#        vs 'fee cap ... 0.5%') stay -- collapsing them would also merge
#        tier-scoped facts with identical numbers.
#   S1-b 'sub' joins _BOUND_UP and _bound_class splits hyphens like
#        spaces, so 'sub-2-second' reads as an upper bound.
#   S1-c _units_in splits hyphens like spaces, so 'sub-2-second' carries
#        the unit 'second' (previously it looked unitless and the unit
#        check blocked the collapse).
#
# Save in the project root and run:
#     & $py F4_cross_type_collapse.py
#
# Flow: PHASE 0 locate+versions -> PHASE 1 ensure the 9 regression tests
# -> PHASE 2 RED baseline (4 must fail: dashboard collapse, retention
# collapse, sub-prefix bound, hyphen units; 5 guards already pass)
# -> PHASE 3 anchored patches -> PHASE 4 GREEN 9/9 -> PHASE 5 full gate
# (80 expected = 71 + 9) -> PHASE 6 verify slices + a replay of the EXACT
# live DIAG store (7 criticals -> 6, CRR 100 at budget 59).
#
# Idempotent; partial states handled; originals backed up ONCE to
# <file>.f4bak. All anchors are ASCII-only (no em-dash lines matched).

import os
import py_compile
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

NEW9 = [
    "test_cross_type_dashboard_restatement_collapses",
    "test_cross_type_retention_restatement_collapses",
    "test_cross_type_thin_shared_restatement_stays",
    "test_cross_type_contradictory_bounds_not_collapsed",
    "test_cross_type_same_subject_pages_not_collapsed",
    "test_cross_type_plain_values_not_collapsed",
    "test_cross_type_revision_supersedes_unchanged",
    "test_sub_prefix_reads_as_upper_bound",
    "test_units_scan_hyphenated_tokens",
]
NODES = ["tests/test_memory.py::" + n for n in NEW9]


def die(msg):
    print()
    print("F4 ABORT -- " + msg)
    print("Each file is written only after ALL its anchors validate, so a")
    print(".f4bak restore should not be needed. Paste this output back as-is.")
    sys.exit(1)


def file_text(fp):
    with open(fp, encoding="utf-8") as f:
        return f.read()


def write_text(fp, s):
    crlf = b"\r\n" in open(fp, "rb").read()[:4000]
    with open(fp, "w", encoding="utf-8",
              newline=("\r\n" if crlf else "\n")) as f:
        f.write(s)


def backup(fp):
    bak = fp + ".f4bak"
    if not os.path.exists(bak):
        shutil.copyfile(fp, bak)


def hr(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def run_pytest(args, capture):
    if capture:
        r = subprocess.run([sys.executable, "-m", "pytest"] + args,
                           cwd=ROOT, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        print(r.stdout[-4000:])
        if r.stderr.strip():
            print("stderr tail:", r.stderr[-600:])
        return r
    return subprocess.run([sys.executable, "-m", "pytest"] + args, cwd=ROOT)


def locate_pkg():
    for cand in (os.path.join(ROOT, "dspm"),
                 os.path.join(ROOT, "src", "dspm")):
        if os.path.isfile(os.path.join(cand, "patch.py")):
            return cand
    die("could not locate the dspm package -- tried dspm/ and src/dspm/ "
        "under " + ROOT)


PKG = locate_pkg()
P = {
    "memory":    os.path.join(PKG, "memory.py"),
    "init":      os.path.join(PKG, "__init__.py"),
    "tests":     os.path.join(ROOT, "tests", "test_memory.py"),
    "pyproject": os.path.join(ROOT, "pyproject.toml"),
}


def find_version(fp):
    if not os.path.isfile(fp):
        return None
    text = file_text(fp)
    return (re.search(r'__version__\s*=\s*"([^"]+)"', text)
            or re.search(r'^version\s*=\s*"([^"]+)"', text, re.M))


def patch_file(path, edits, soft=False):
    src = file_text(path)
    plan = []
    for old, new in edits:
        if new in src:
            print("  [skip: already applied] " + os.path.basename(path))
            continue
        if old not in src:
            if soft:
                print("  [WARN: anchor missing, edit skipped] "
                      + os.path.basename(path) + " :: "
                      + old[:80].replace("\n", "\\n"))
                continue
            die("anchor missing in %s:\n%s" % (path, old[:300]))
        plan.append((old, new))
    if not plan:
        return
    for old, new in plan:
        src = src.replace(old, new, 1)
    backup(path)
    write_text(path, src)
    if path.endswith(".py"):
        py_compile.compile(path, doraise=True)
    print("  patched + compiled: " + path)


# ---------------------------------------------------------------------------
# PHASE 0
# ---------------------------------------------------------------------------

hr("PHASE 0 -- PACKAGE + VERSIONS (expect 0.1.17 everywhere)")
print("package dir : " + PKG)
for label, key in (("dspm/__init__.py", "init"),
                   ("pyproject.toml", "pyproject")):
    m = find_version(P[key])
    print("%-18s %s" % (label, m.group(1) if m else "<no version line>"))

# ---------------------------------------------------------------------------
# PHASE 1 -- the nine S1 regression tests
# ---------------------------------------------------------------------------

TESTS9 = '''
# ---------------------------------------------------------------------------
# v0.1.17 finding S1 (live conversation smoke): cross-type CON<->DEC
# restatements collapsed; "sub-" reads as an upper bound; units scan
# hyphenated tokens. Live evidence (DIAG dump): 'Maximum fee per
# transaction is limited to 0.5%' [CON] + 'fee cap set to 0.5% as tracked'
# [DEC], and 'dashboard must load in under 2 seconds' [DEC] +
# 'Dashboard latency must be sub-2-second' [CON], coexisted as duplicate
# criticals; the store bloated to 8-10 and CRR fell to 70-87% at the
# smoke's 59-token budget.
# ---------------------------------------------------------------------------


def test_cross_type_dashboard_restatement_collapses():
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('s0', 6, 'decision',
                                 'dashboard must load in under 2 seconds', []))
    m._merge_patch(SemanticPatch('s1', 7, 'constraint',
                                 'Dashboard latency must be sub-2-second', []))
    crits = [p.payload for p in m.critical_patches]
    assert len(crits) == 1, \\
        f"cross-type dashboard restatement should collapse: {crits}"


def test_cross_type_retention_restatement_collapses():
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('s0', 0, 'decision',
                                 'Audit retention set to 5 years', []))
    m._merge_patch(SemanticPatch('s1', 1, 'constraint',
                                 'Audit logs stored for exactly 5 years', []))
    crits = [p.payload for p in m.critical_patches]
    assert len(crits) == 1, \\
        f"cross-type retention restatement should collapse: {crits}"


def test_cross_type_thin_shared_restatement_stays():
    """Known residual, frozen deliberately: the live fee pair shares only
    ONE non-numeric word ('fee'); collapsing it would also merge
    tier-scoped facts ('Fee capped 0.5% cloud tier' vs 'premium tier')."""
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('s0', 0, 'constraint',
                                 'Maximum fee per transaction is limited to 0.5%', []))
    m._merge_patch(SemanticPatch('s1', 1, 'decision',
                                 'fee cap set to 0.5% as tracked', []))
    assert len(m.critical_patches) == 2


def test_cross_type_contradictory_bounds_not_collapsed():
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('s0', 0, 'decision',
                                 'Audit retention must not exceed 5 years', []))
    m._merge_patch(SemanticPatch('s1', 1, 'constraint',
                                 'Audit retention at least 5 years', []))
    assert len(m.critical_patches) == 2, \\
        f"contradictory cross-type bounds merged: {[p.payload for p in m.critical_patches]}"


def test_cross_type_same_subject_pages_not_collapsed():
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('s0', 0, 'constraint',
                                 'Dashboard loads in under 2 seconds', []))
    m._merge_patch(SemanticPatch('s1', 1, 'decision',
                                 'Homepage loads in under 2 seconds', []))
    assert len(m.critical_patches) == 2, \\
        f"different subjects merged: {[p.payload for p in m.critical_patches]}"


def test_cross_type_plain_values_not_collapsed():
    """Both-plain guard: no directional/exact bound, one shared word --
    'Budget 100 for cloud' vs 'Budget 100 for training' must stay apart."""
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('s0', 0, 'constraint',
                                 'Budget 100 for cloud', []))
    m._merge_patch(SemanticPatch('s1', 1, 'decision',
                                 'Budget 100 for training', []))
    assert len(m.critical_patches) == 2


def test_cross_type_revision_supersedes_unchanged():
    """(b3) must not disturb (c): a real cross-type revision with changed
    numbers still supersedes and bakes history."""
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('s0', 0, 'constraint',
                                 'Webhook timeout must be 30 seconds', []))
    m._merge_patch(SemanticPatch('s1', 1, 'decision',
                                 'Updated webhook timeout changed to 10 seconds instead', []))
    crits = [p.payload for p in m.critical_patches]
    assert len(crits) == 1, crits
    assert '(was' in crits[0] and '30' in crits[0], crits


def test_sub_prefix_reads_as_upper_bound():
    from dspm.memory import _bound_class
    assert _bound_class("Dashboard latency must be sub-2-second") == "upper"
    # sanity: prior vocabulary unchanged
    assert _bound_class("retention must not exceed 5 years") == "upper"
    assert _bound_class("retention exactly 5 years") == "exact"


def test_units_scan_hyphenated_tokens():
    from dspm.memory import _units_in
    assert _units_in("Dashboard latency must be sub-2-second") == frozenset({"second"})
    assert _units_in("24-hour window") == frozenset({"hour"})
    # sanity: plain tokens unchanged
    assert _units_in("timeout 30 seconds") == frozenset({"second"})
'''

hr("PHASE 1 -- S1 REGRESSION TESTS PRESENT?")
tsrc = file_text(P["tests"])
present = [n for n in NEW9 if ("def " + n + "(") in tsrc]
if len(present) == 9:
    print("[skip: all nine S1 tests already in tests/test_memory.py]")
elif present:
    die("only some of the nine S1 tests are present (%s) -- paste the "
        "tail of tests/test_memory.py back" % ", ".join(present))
else:
    backup(P["tests"])
    write_text(P["tests"], tsrc.rstrip("\n") + "\n\n\n" + TESTS9.strip("\n") + "\n")
    print("  tests appended: 9 new (S1 regressions + guards)")

# ---------------------------------------------------------------------------
# PHASE 2 -- RED baseline
# ---------------------------------------------------------------------------

hr("PHASE 2 -- RED BASELINE: 4 must fail (collapse x2, sub-prefix, hyphen units)")
base = run_pytest(["-q"] + NODES, capture=True)
if base.returncode == 0:
    print("NOTE: all nine already pass -- fixes already applied? The patch")
    print("phase below is idempotent and will simply skip.")
else:
    m = re.search(r"(\d+) failed", base.stdout)
    n = int(m.group(1)) if m else -1
    if not (1 <= n <= 4):
        die("unexpected baseline result (%r failed of 9) -- expected at "
            "most the 4 known-red tests. Paste the output above" % n)
    print("RED BASELINE OK: %d/4 known-red tests fail on the unfixed "
          "tree (the other %d guards already pass)." % (n, 9 - n))

# ---------------------------------------------------------------------------
# PHASE 3 -- patches (memory.py only)
# ---------------------------------------------------------------------------

hr("PHASE 3 -- S1-a/b/c: memory.py (CORE)")

EDITS_MEMORY = [
    # S1-b: "sub" joins the upper-bound vocabulary
    (
        '_BOUND_UP = {\n'
        '    "less", "fewer", "under", "below", "within", "most", "max",\n'
        '    "maximum", "cap", "capped", "up",\n'
        '}',
        '# v0.1.17 (finding S1): "sub" added -- hyphenated compounds like\n'
        '# "sub-2-second" are upper bounds now that _bound_class splits\n'
        '# hyphens like spaces (live smoke evidence).\n'
        '_BOUND_UP = {\n'
        '    "less", "fewer", "under", "below", "within", "most", "max",\n'
        '    "maximum", "cap", "capped", "up", "sub",\n'
        '}',
    ),
    # S1-b: _bound_class tokenizes hyphens like spaces
    (
        '    ws = [w.lower().strip(".,;()") for w in text.split()]',
        '    # v0.1.17 (finding S1): hyphens split like spaces so\n'
        '    # "sub-2-second" reads as bound word + value + unit.\n'
        '    ws = [w.lower().strip(".,;()") for w in text.replace("-", " ").split()]',
    ),
    # S1-c: _units_in scans hyphenated tokens
    (
        '    units = set()\n'
        '    for w in text.lower().split():\n'
        '        canon = _UNIT_ALIASES.get(w.strip(".,;)"))\n'
        '        if canon:\n'
        '            units.add(canon)\n'
        '    return frozenset(units)',
        '    units = set()\n'
        '    # v0.1.17 (finding S1): hyphens split like spaces --\n'
        '    # "sub-2-second" carries the unit "second"; without this,\n'
        '    # hyphenated values looked unitless and restatement collapse\n'
        '    # compared them against nothing.\n'
        '    for w in text.lower().replace("-", " ").split():\n'
        '        canon = _UNIT_ALIASES.get(w.strip(".,;)"))\n'
        '        if canon:\n'
        '            units.add(canon)\n'
        '    return frozenset(units)',
    ),
    # S1-a: rule (b3) after (c2) in the cross-type branch
    (
        "                    # (c2) terse cross-type revision (mirrors a2)\n"
        "                    if marker and new_head and new_head == _slot_head(existing.payload) \\\n"
        "                            and new_nums != _numbers(existing.payload):\n"
        "                        stale.append(existing)          # (c2) terse cross-type\n"
        "                        continue",
        "                    # (c2) terse cross-type revision (mirrors a2)\n"
        "                    if marker and new_head and new_head == _slot_head(existing.payload) \\\n"
        "                            and new_nums != _numbers(existing.payload):\n"
        "                        stale.append(existing)          # (c2) terse cross-type\n"
        "                        continue\n"
        "                    # v0.1.17 (finding S1): (b3) cross-type same-value\n"
        "                    # restatement. Live smoke evidence: the user states a\n"
        "                    # requirement that extracts as [CON] ('dashboard must\n"
        "                    # load in under 2 seconds') and the assistant echo\n"
        "                    # extracts as [DEC] ('Dashboard latency must be\n"
        "                    # sub-2-second') -- no rule compared across types\n"
        "                    # without a revision verb, so both coexisted and the\n"
        "                    # store bloated to 8-10 criticals (CRR 70-87% at the\n"
        "                    # 59-token smoke budget). Stricter than same-type (b2):\n"
        "                    # additionally requires at least one directional or\n"
        "                    # exact bound, because cross-type merges are riskier\n"
        "                    # ('Budget 100 for cloud' vs 'Budget 100 for training'\n"
        "                    # is plain/plain and must stay apart). Known residual,\n"
        "                    # frozen by test: thin restatements sharing ONE\n"
        "                    # non-numeric word ('Maximum fee ... 0.5%' vs 'fee cap\n"
        "                    # ... 0.5%') stay -- collapsing them would also merge\n"
        "                    # tier-scoped facts with identical numbers.\n"
        "                    bc_new = _bound_class(patch.payload)\n"
        "                    bc_old = _bound_class(existing.payload)\n"
        "                    if (new_head and new_head == _slot_head(existing.payload)\n"
        "                            and len({w for w in shared if not re.search(r\"\\d\", w)}) >= 2\n"
        "                            and new_nums and new_nums == _numbers(existing.payload)\n"
        "                            and _units_in(patch.payload) == _units_in(existing.payload)\n"
        "                            and _polarity_compatible(bc_new, bc_old)\n"
        "                            and not (bc_new == \"plain\" and bc_old == \"plain\")):\n"
        "                        stale.append(existing)          # (b3) cross-type restatement\n"
        "                        continue",
    ),
    # merge-rules comment block gains (b3)
    (
        '    #   (c2) cross-type + same anchor + revision verb + changed numbers',
        '    #   (c2) cross-type + same anchor + revision verb + changed numbers\n'
        '    #   (b3) cross-type CON<->DEC same-value restatement: same anchor,\n'
        '    #        >=2 shared non-numeric words, identical numbers + units,\n'
        '    #        compatible bounds, at least one directional/exact bound\n'
        '    #        (v0.1.17 finding S1 -- live smoke showed user [CON] +\n'
        '    #        assistant [DEC] echoes coexisting as duplicate criticals)',
    ),
]
patch_file(P["memory"], EDITS_MEMORY)

# ---------------------------------------------------------------------------
# PHASE 4 -- GREEN re-check
# ---------------------------------------------------------------------------

hr("PHASE 4 -- GREEN RE-CHECK: 9/9")
green = run_pytest(["-q"] + NODES, capture=True)
if green.returncode != 0:
    die("GREEN check failed -- the fixes did not land correctly. Paste "
        "everything back; do NOT re-run F4 yet.")
print("GREEN: 9/9 pass after patching.")

# ---------------------------------------------------------------------------
# PHASE 5 -- full gate
# ---------------------------------------------------------------------------

hr("PHASE 5 -- FULL GATE: pytest tests/ -q   (80 expected = 71 + 9)")
coll = run_pytest(["tests/", "-q", "--collect-only"], capture=True)
m = re.search(r"(\d+) tests? collected", coll.stdout)
total = int(m.group(1)) if m else -1
print("collected: %d (expected 80 = 71 + 9; a different count is fine if "
      "you added/removed tests yourself)" % total)

gate = run_pytest(["tests/", "-q"], capture=False)
if gate.returncode != 0:
    print()
    print("F4: FULL GATE FAILED -- the patches ARE on disk. Do NOT re-run F4")
    print("(it is idempotent, but I need the failure text first). Paste")
    print("everything back as-is.")
    sys.exit(2)

# ---------------------------------------------------------------------------
# PHASE 6 -- verify slices + live DIAG-store replay
# ---------------------------------------------------------------------------

VERIFY = '''
import inspect
import dspm
from dspm import DSPMMemory
from dspm.memory import _bound_class, _units_in
from dspm.patch import SemanticPatch


def show(title, src, marker, count):
    lines = src.splitlines()
    for i, line in enumerate(lines):
        if marker in line:
            print("=" * 70)
            print(title)
            print("=" * 70)
            for out in lines[i:i + count]:
                print(out)
            print()
            return
    print("MISSING MARKER: " + title + " -> " + marker)


print("dspm.__version__ =", dspm.__version__)
show("S1-a -- rule (b3)", inspect.getsource(DSPMMemory._merge_patch),
     "(b3)", 12)
print("S1-b bound probe:", _bound_class("Dashboard latency must be sub-2-second"))
print("S1-c units probe:", _units_in("Dashboard latency must be sub-2-second"),
      _units_in("24-hour window"))

# Replay the EXACT live DIAG store (7 criticals, budget 59) and confirm
# the predicted post-fix shape: dashboard pair merges, fee residual stays.
m = DSPMMemory(budget=59)


def mk(pid, turn, t, pl):
    return SemanticPatch(patch_id=pid, turn_index=turn, patch_type=t,
                         payload=pl, dependencies=[])


for pid, turn, t, pl in [
    ("d0", 0, "constraint",
     "All payments must be PCI-DSS compliant as a non-negotiable requirement"),
    ("d1", 0, "constraint", "Maximum fee per transaction is limited to 0.5%"),
    ("d2", 1, "decision", "fee cap set to 0.5% as tracked"),
    ("d3", 5, "decision",
     "Webhook timeout updated to 10 seconds, superseding previous 30 seconds value (was 30)"),
    ("d4", 6, "decision", "dashboard must load in under 2 seconds"),
    ("d5", 7, "decision", "Audit retention set to 5 years"),
    ("d6", 7, "constraint", "Dashboard latency must be sub-2-second"),
]:
    m._merge_patch(mk(pid, turn, t, pl))
print("DIAG-store replay: %d criticals (live run had 7; fee residual "
      "stays by design, dashboard pair merges)" % len(m.critical_patches))
ctx = m.get_context("webhook timeout")
s = m.stats
print("replayed smoke at budget 59: critical_total=%d critical_selected=%d "
      "crr=%d" % (s["critical_total"], s["critical_selected"], s["crr"]))
'''

hr("PHASE 6 -- VERIFY SLICES + DIAG-STORE REPLAY")
subprocess.run([sys.executable, "-c", VERIFY], cwd=ROOT)

hr("F4 COMPLETE")
print("Rule (b3) in, vocabulary and unit scan fixed, guards green, full")
print("gate green. Backups: memory.py.f4bak + tests .f4bak (delete when")
print("satisfied).")
print("NEXT -- re-run the conversation smoke several times:")
print('  1..5 | ForEach-Object { & $py scripts\\smoke_conversation.py 2>&1 | Select-String "PASS|FAIL" }')
print("Expected: 4x PASS every run (store ~6 criticals, all delivered at")
print("budget 59). The fee duplicate may still appear in the STORE (the")
print("documented residual) but no longer costs CRR. If any run still")
print("fails, run diag_smoke_criticals.py again and paste the DIAG block --")
print("the next evidence round targets whatever pairs remain.")
