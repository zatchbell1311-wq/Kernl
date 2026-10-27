# F7_live_echo_collapse.py -- v0.1.17 20-run live gate, round 2. F6 was
# green in isolation (102 tests, offline harness 5/5, (b5) verified on
# the fixture wording) but the live loop still failed 6/20 runs, all
# CRR under-reporting: 87/77/87/87/70/87. Two signatures, and both are
# echo criticals whose LIVE word order breaks the slot-head anchor
# that every earlier bridge rule keys on:
#
#   Signature B (all 6 failing runs): the audit retention fact
#   extracted TWICE -- noun-first as 'Audit retention set to 5
#   years' and verb-first as 'Retain audit logs for exactly 5 years'
#   -- delivered as '[DEC] 5 years' + '[DEC] exactly 5'. Root cause:
#   (b2) requires equal slot heads ('audit' vs 'retain') and (b)
#   needs >=4 shared words (the pair shares 2). A stray patch, and
#   no existing rule's word overlap catches it.
#
#   Signature A (runs 5/10/12/17): '[CON] Dashboard sub-2-second' +
#   '[DEC] sub-2-second dashboard' both delivered. The stored
#   payloads share the compound token AND the subject noun, but at
#   least one led with a confirmation verb ('Confirmed sub-2-second
#   dashboard', 'Keep Dashboard latency sub-2-second'), so (b5)'s
#   BIDIRECTIONAL slot-head containment failed -- 'confirmed' and
#   'keep' never appear in the other payload. Every other (b5)
#   condition passes: shared digit anchor, identical numbers, units,
#   bounds, proximity.
#
# Rule (b6) -- the shared-subject echo bridge (same- OR cross-type) --
# anchors on what the pair SHARES instead of where each payload
# starts: a shared subject noun (>=4 chars, not bound/unit/filler),
# identical numbers, identical units, compatible bounds, never
# plain/plain, <=2 turns, plus a residue guard that keeps scoped
# facts apart: every unshared content word must be a confirmation
# verb (closed _ECHO_VERBS set), a slot-family twin of the other side
# (retain~retention), or property-positioned behind the shared
# subject with only non-prepositional filler between. Platinum/silver
# fees, dashboard/homepage, cloud/training budgets and audit-vs-
# payment logs all stay separate (guards below freeze each).
#
# Save in the project root and run:
#     & $py F7_live_echo_collapse.py
#
# Flow: PHASE 0 locate+versions -> PHASE 1 tests (14 new) -> PHASE 2
# RED baseline (7 of 14 must fail: the echo merges + both store
# replays; the 7 guards already pass) -> PHASE 3 anchored memory.py
# patches -> PHASE 4 GREEN 14/14 -> PHASE 5 full gate (116 expected =
# 102 + 14) -> PHASE 6 slices + merge probes + run replays -> PHASE 6b
# offline harness (AGGREGATE: 5/5).
#
# Idempotent; partial states handled; originals backed up ONCE to
# <file>.f7bak. All anchors are ASCII-only. memory.py-only patch --
# the engine half of F6 is untouched.

import os
import py_compile
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

NEW14 = [
    # reds (fail pre-patch): the two live signatures + replays
    "test_shared_subject_retention_echo_collapses",
    "test_shared_subject_retention_echo_symmetric",
    "test_shared_subject_dashboard_echo_dirty_heads_collapses",
    "test_shared_subject_dashboard_echo_one_dirty_head_collapses",
    "test_same_type_dashboard_echo_under2_collapses",
    "test_run5_store_replay_collapses_all_echoes",
    "test_run8_store_replay_collapses_all_echoes",
    # guards (pass pre and post): the freeze set
    "test_shared_subject_bridge_keeps_scoped_budgets",
    "test_shared_subject_bridge_keeps_scoped_dashboards",
    "test_shared_subject_bridge_keeps_platinum_silver_fees",
    "test_shared_subject_bridge_keeps_dashboard_homepage",
    "test_shared_subject_bridge_keeps_retention_vs_periodic_review",
    "test_shared_subject_bridge_requires_shared_subject_noun",
    "test_shared_subject_bridge_edge_guards",
]
NODES = ["tests/test_memory.py::" + n for n in NEW14]


def die(msg):
    print()
    print("F7 ABORT -- " + msg)
    print("Each file is written only after ALL its anchors validate, so a")
    print(".f7bak restore should not be needed. Paste this output back as-is.")
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
    bak = fp + ".f7bak"
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
    "engine":    os.path.join(PKG, "engine.py"),
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
# PHASE 1 -- the fourteen F7 tests
# ---------------------------------------------------------------------------

TESTS_F7 = '''
# ---------------------------------------------------------------------------
# v0.1.17 20-run live gate, round 2 (F7): 6/20 runs failed CRR at
# 87/77/70 -- two live-extraction echo signatures no offline suite
# could produce. (i) The audit retention fact arrived twice -- noun
# first ('Audit retention set to 5 years', head 'audit') and verb
# first ('Retain audit logs for exactly 5 years', head 'retain') --
# and survived as TWO decisions in 6/6 failing runs ('[DEC] 5 years'
# + '[DEC] exactly 5'); (b2) requires equal heads, (b) needs >=4
# shared words, so nothing collapsed it. (ii) Dashboard echoes led
# with confirmation verbs ('Confirmed sub-2-second dashboard' -- head
# 'confirmed' absent from the other payload), so (b5)'s
# bidirectional head containment failed while every other condition
# passed. Rule (b6) bridges same- and cross-type echoes by what the
# pair SHARES: subject noun + identical value/unit/bound safety +
# a residue guard that keeps scoped facts apart. The payloads below
# are pinned to the observed run-4/5/8 ctx lines by the trim
# assertions (the failing runs starved every line to the 2-word
# floor: '[DEC] exactly 5', '[DEC] 5 years', '[CON] Dashboard
# sub-2-second', '[DEC] sub-2-second dashboard', '[DEC] under 2').
# ---------------------------------------------------------------------------


def test_shared_subject_retention_echo_collapses():
    """Live runs 4/5/8/10/12/15/17 (every FAIL) delivered BOTH
    '[DEC] 5 years' and '[DEC] exactly 5' -- the audit retention fact
    extracted twice. The trim assertions pin the stored payloads to
    those exact ctx lines."""
    from dspm.engine import DSPMEngine
    eng = DSPMEngine(budget=59)
    assert eng._trim_numeric_first(
        'Audit retention set to 5 years', 2) == '5 years'
    assert eng._trim_numeric_first(
        'Retain audit logs for exactly 5 years', 2) == 'exactly 5'
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('s0', 7, 'decision',
                                 'Audit retention set to 5 years', []))
    m._merge_patch(SemanticPatch('s1', 7, 'decision',
                                 'Retain audit logs for exactly 5 years', []))
    crits = [p.payload for p in m.critical_patches]
    assert len(crits) == 1, \\
        f"verb-first retention echo should collapse: {crits}"


def test_shared_subject_retention_echo_symmetric():
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('s0', 7, 'decision',
                                 'Retain audit logs for exactly 5 years', []))
    m._merge_patch(SemanticPatch('s1', 7, 'decision',
                                 'Audit retention set to 5 years', []))
    assert len(m.critical_patches) == 1, \\
        f"symmetric retention echo should collapse: " \\
        f"{[p.payload for p in m.critical_patches]}"


def test_shared_subject_dashboard_echo_dirty_heads_collapses():
    """Run 5's delivered pair: '[CON] Dashboard sub-2-second' +
    '[DEC] sub-2-second dashboard' -- BOTH live payloads led with a
    confirmation verb, so (b5)'s bidirectional slot-head containment
    failed in both directions ('keep' and 'confirmed' each absent
    from the other payload) while every other (b5) condition
    passed."""
    from dspm.engine import DSPMEngine
    eng = DSPMEngine(budget=59)
    assert eng._trim_numeric_first(
        'Keep Dashboard latency sub-2-second', 2) == 'Dashboard sub-2-second'
    assert eng._trim_numeric_first(
        'Confirmed sub-2-second dashboard', 2) == 'sub-2-second dashboard'
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('s0', 7, 'constraint',
                                 'Keep Dashboard latency sub-2-second', []))
    m._merge_patch(SemanticPatch('s1', 8, 'decision',
                                 'Confirmed sub-2-second dashboard', []))
    assert len(m.critical_patches) == 1, \\
        f"double-dirty-head dashboard echo should collapse: " \\
        f"{[p.payload for p in m.critical_patches]}"


def test_shared_subject_dashboard_echo_one_dirty_head_collapses():
    """The single-dirty-head variant: the constraint keeps its clean
    'dashboard' head, the echo still leads with 'confirmed'."""
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('s0', 7, 'constraint',
                                 'Dashboard latency must be sub-2-second', []))
    m._merge_patch(SemanticPatch('s1', 8, 'decision',
                                 'Confirmed sub-2-second dashboard', []))
    assert len(m.critical_patches) == 1, \\
        f"one-dirty-head dashboard echo should collapse: " \\
        f"{[p.payload for p in m.critical_patches]}"


def test_same_type_dashboard_echo_under2_collapses():
    """Run 8's shape (signature B run, no dashboard pair visible in
    ctx): the delivered line was '[DEC] under 2' (the t6 requirement)
    and the evicted 8th critical was its same-type echo -- (b2)
    needs equal heads ('dashboard' vs 'sub'), (b) needs >=4 shared
    words, so both stayed and the store read 8."""
    from dspm.engine import DSPMEngine
    eng = DSPMEngine(budget=59)
    assert eng._trim_numeric_first(
        'dashboard must load in under 2 seconds', 2) == 'under 2'
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('s0', 6, 'decision',
                                 'dashboard must load in under 2 seconds', []))
    m._merge_patch(SemanticPatch('s1', 7, 'decision',
                                 'sub-2-second dashboard confirmed', []))
    assert len(m.critical_patches) == 1, \\
        f"same-type under-2 echo should collapse: " \\
        f"{[p.payload for p in m.critical_patches]}"


def test_run5_store_replay_collapses_all_echoes():
    """Store-level replay of the CRR 77% run: six facts + the
    retention echo + two head-dirty dashboard echoes. Pre-F7 the
    store bloats to 9 and the delivered ctx shows the duplicate
    lines; post-F7 every echo collapses, the store reads 6 and CRR
    is 100 at the smoke's budget."""
    m = DSPMMemory(budget=59)
    rows = [
        ('r0', 0, 'constraint',
         'All payments must be PCI-DSS compliant as a non-negotiable requirement'),
        ('r1', 0, 'constraint', 'Maximum fee per transaction is limited to 0.5%'),
        ('r2', 1, 'decision', 'fee cap set to 0.5% as tracked'),
        ('r3', 3, 'decision', 'automatic retries on failure enabled'),
        ('r4', 5, 'decision',
         'Webhook timeout updated to 10 seconds, superseding the previous 30 seconds value (was 30)'),
        ('r5', 6, 'constraint', 'dashboard load time must be under 2 seconds'),
        ('r6', 7, 'decision', 'Audit retention set to 5 years'),
        ('r7', 7, 'decision', 'Retain audit logs for exactly 5 years'),
        ('r8', 7, 'constraint', 'Keep Dashboard latency sub-2-second'),
        ('r9', 7, 'decision', 'Confirmed sub-2-second dashboard'),
    ]
    for pid, turn, t, pl in rows:
        m._merge_patch(SemanticPatch(pid, turn, t, pl, []))
    ctx = m.get_context('webhook timeout')
    s = m.stats
    assert s['critical_total'] == 6, s
    assert s['crr'] == 100, s
    assert sum('sub-2-second' in l for l in ctx.splitlines()) == 1, ctx
    assert 'under 2' not in ctx, ctx
    retention = [l for l in ctx.splitlines()
                 if ('exactly' in l or 'retention' in l
                     or 'audit' in l or 'years' in l)]
    assert len(retention) == 1, ctx


def test_run8_store_replay_collapses_all_echoes():
    """Store-level replay of the CRR 87% run 8: six facts with the
    dashboard requirement as a t6 decision ('[DEC] under 2' was the
    delivered line), plus the retention echo and the same-type
    dashboard echo -- the evicted 8th critical. Post-F7 the store
    reads 6 and CRR is 100."""
    m = DSPMMemory(budget=59)
    rows = [
        ('r0', 0, 'constraint',
         'All payments must be PCI-DSS compliant as a non-negotiable requirement'),
        ('r1', 0, 'constraint', 'Maximum fee per transaction is limited to 0.5%'),
        ('r2', 1, 'decision', 'fee cap set to 0.5% as tracked'),
        ('r3', 3, 'decision', 'automatic retries on failure enabled'),
        ('r4', 5, 'decision',
         'Webhook timeout updated to 10 seconds, superseding the previous 30 seconds value (was 30)'),
        ('r5', 6, 'decision', 'dashboard must load in under 2 seconds'),
        ('r6', 7, 'decision', 'Audit retention set to 5 years'),
        ('r7', 7, 'decision', 'Retain audit logs for exactly 5 years'),
        ('r8', 7, 'decision', 'sub-2-second dashboard confirmed'),
    ]
    for pid, turn, t, pl in rows:
        m._merge_patch(SemanticPatch(pid, turn, t, pl, []))
    ctx = m.get_context('webhook timeout')
    s = m.stats
    assert s['critical_total'] == 6, s
    assert s['crr'] == 100, s
    assert 'under 2' not in ctx, ctx
    assert sum('sub-2-second' in l for l in ctx.splitlines()) == 1, ctx


def test_shared_subject_bridge_keeps_scoped_budgets():
    """Scope nouns stay scopes: 'Cloud' before the shared subject and
    'for training' after it (a preposition blocks the
    property-position tolerance -- 'for <scope>' is never a
    property)."""
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('s0', 0, 'constraint',
                                  'Cloud budget cap 100', []))
    m._merge_patch(SemanticPatch('s1', 1, 'decision',
                                 'budget cap 100 for training', []))
    assert len(m.critical_patches) == 2


def test_shared_subject_bridge_keeps_scoped_dashboards():
    """Same value, same subject, different audiences: a scope noun
    BEFORE the shared subject ('Guest dashboard...', 'Admin
    dashboard...') is still a scope -- the residue guard rejects it
    by position. (The 'for admins/for guests' shape with a bound-word
    echo head is the FROZEN (b4) canary behavior and merges -- that
    limitation predates F7.)"""
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('s0', 0, 'decision',
                                  'Guest dashboard sub-2-second', []))
    m._merge_patch(SemanticPatch('s1', 1, 'decision',
                                  'Admin dashboard sub-2-second', []))
    assert len(m.critical_patches) == 2


def test_shared_subject_bridge_keeps_platinum_silver_fees():
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('s0', 0, 'constraint',
                                  'Maximum platinum fee is 0.5%', []))
    m._merge_patch(SemanticPatch('s1', 1, 'decision',
                                 'silver tier fee cap 0.5%', []))
    assert len(m.critical_patches) == 2


def test_shared_subject_bridge_keeps_dashboard_homepage():
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('s0', 0, 'constraint',
                                  'Dashboard loads in under 2 seconds', []))
    m._merge_patch(SemanticPatch('s1', 1, 'decision',
                                 'Homepage loads in under 2 seconds', []))
    assert len(m.critical_patches) == 2


def test_shared_subject_bridge_keeps_retention_vs_periodic_review():
    """Storing for 5 years vs reviewing every 5 years: same subject,
    same value -- different facts. Blocked by the plain/plain gate
    ('every' is not a bound word)."""
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('s0', 0, 'decision',
                                 'Audit retention set to 5 years', []))
    m._merge_patch(SemanticPatch('s1', 1, 'decision',
                                 'Review audit logs every 5 years', []))
    assert len(m.critical_patches) == 2


def test_shared_subject_bridge_requires_shared_subject_noun():
    """Property nouns never anchor the bridge: 'logs' is shared but is
    a _PROPERTY_NOUN (like 'time'/'latency'/'response'), so this pair
    has NO shared subject and stays two facts. (The closer pair
    'Retain audit logs for exactly 5 years' + 'Retain payment logs
    for exactly 5 years' already merges today via rule (b)'s >=4
    shared words -- pre-existing, canary-frozen behavior.)"""
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('s0', 0, 'decision',
                                  'Audit logs retained 5 years', []))
    m._merge_patch(SemanticPatch('s1', 1, 'decision',
                                  'Retain payment logs for exactly 5 years', []))
    assert len(m.critical_patches) == 2


def test_shared_subject_bridge_edge_guards():
    m1 = DSPMMemory(budget=250)
    m1._merge_patch(SemanticPatch('s0', 0, 'decision',
                                  'Audit retention set to 5 years', []))
    m1._merge_patch(SemanticPatch('s1', 5, 'decision',
                                  'Retain audit logs for exactly 5 years', []))
    assert len(m1.critical_patches) == 2, "beyond two turns must stay"
    m2 = DSPMMemory(budget=250)
    m2._merge_patch(SemanticPatch('s0', 0, 'decision',
                                  'Retain audit logs for exactly 5 years', []))
    m2._merge_patch(SemanticPatch('s1', 1, 'decision',
                                  'Retain audit logs for exactly 3 years', []))
    assert len(m2.critical_patches) == 2, "different values must stay"
    m3 = DSPMMemory(budget=250)
    m3._merge_patch(SemanticPatch('s0', 0, 'decision',
                                  'Retain audit logs for exactly 5 years', []))
    m3._merge_patch(SemanticPatch('s1', 1, 'decision',
                                  'Retain audit logs for exactly 5 months', []))
    assert len(m3.critical_patches) == 2, "different units must stay"
    m4 = DSPMMemory(budget=250)
    m4._merge_patch(SemanticPatch('s0', 0, 'constraint',
                                  'Keep Dashboard latency sub-2-second', []))
    m4._merge_patch(SemanticPatch('s1', 1, 'decision',
                                  'Confirmed sub-2-minute dashboard', []))
    assert len(m4.critical_patches) == 2, "unit mismatch inside the compound"
'''

hr("PHASE 1 -- F7 TESTS PRESENT?")
tsrc = file_text(P["tests"])
present = [n for n in NEW14 if ("def " + n + "(") in tsrc]
if len(present) == len(NEW14):
    print("[skip: all fourteen F7 tests already in tests/test_memory.py]")
elif present:
    die("only some of the fourteen F7 tests are present (%s) -- paste "
        "the tail of tests/test_memory.py back" % ", ".join(present))
else:
    backup(P["tests"])
    write_text(P["tests"], tsrc.rstrip("\n") + "\n\n\n"
               + TESTS_F7.strip("\n") + "\n")
    print("  tests appended: 14 new (7 echo reds + 7 guards)")

# ---------------------------------------------------------------------------
# PHASE 2 -- RED baseline: 7 of 14 must fail
# ---------------------------------------------------------------------------

hr("PHASE 2 -- RED BASELINE: 7 must fail (5 echo merges + 2 replays)")
base = run_pytest(["-q"] + NODES, capture=True)
if base.returncode == 0:
    print("NOTE: all fourteen already pass -- fixes already applied? The")
    print("patch phase below is idempotent and will simply skip.")
else:
    m = re.search(r"(\d+) failed", base.stdout)
    n = int(m.group(1)) if m else -1
    if not (1 <= n <= 7):
        die("unexpected baseline result (%r failed of 14) -- expected at "
            "most the 7 known-red tests. Paste the output above" % n)
    print("RED BASELINE OK: %d/7 known-red tests fail on the unfixed tree "
          "(the other %d guards already pass)." % (n, 14 - n))

# ---------------------------------------------------------------------------
# PHASE 3 -- anchored patches (memory.py only)
# ---------------------------------------------------------------------------

hr("PHASE 3 -- memory.py: (b6) vocab + method + two call sites")

VOCAB_F7 = '''# v0.1.17 F7 (20-run live gate, round 2): confirmation/stasis verbs a
# live extractor leads echo payloads with. Both round-2 failure
# signatures are echoes whose FIRST significant token is the verb, not
# the subject noun -- 'Retain audit logs for exactly 5 years' (head
# 'retain' vs the original's 'audit') and 'Confirmed sub-2-second
# dashboard' (head 'confirmed', absent from the other payload) --
# which is exactly what broke the slot-head anchors of (b2)/(b3)/(b5).
# In the (b6) residue test these verbs restate, they never scope.
_ECHO_VERBS = frozenset({
    "confirmed", "confirm", "confirms", "keep", "keeps", "kept",
    "lock", "locks", "locked", "target", "targets", "ensure",
    "ensures", "agree", "agreed", "remain", "remains", "stay",
    "stays", "hold", "holds", "track", "tracked", "noted",
    "accepted", "approve", "approved", "enforce", "enforced",
    "apply", "applies", "applied", "retain", "retains", "retained",
})

# v0.1.17 F7: morphological slot twins -- 'retention' and 'retain' are
# the same slot to any reader, never to a string set. Deliberately a
# closed family list: a wrong merge costs more than a missed merge.
_SLOT_FAMILIES = (
    frozenset({"retain", "retention", "retained", "retaining"}),
)

# v0.1.17 F7: small measurement/filing nouns that ride along with a
# subject ('dashboard load time', 'audit logs', 'dashboard latency').
# They never anchor the bridge as the shared subject, but they may
# sit between the subject and a later residue word -- and 'records'
# BEFORE the subject ('No more than 500 records per page') still
# reads as a scope word, because position carries the meaning there.
_PROPERTY_NOUNS = frozenset({
    "time", "load", "loads", "latency", "response", "speed",
    "duration", "period", "logs", "log", "data", "records", "record",
})

# v0.1.17 F7: a preposition between the shared subject and an unshared
# noun marks a SCOPE, not a property -- 'budget for cloud' vs
# 'dashboard latency'. The (b6) property-position tolerance never
# crosses one of these.
_BRIDGE_BLOCKERS = frozenset({
    "for", "to", "of", "in", "on", "at", "by", "with", "from", "per",
})


def _slot_family_eq(a: str, b: str) -> bool:
    """v0.1.17 F7: True when two words are morphological twins of one
    slot ('retain' ~ 'retention')."""
    for fam in _SLOT_FAMILIES:
        if a in fam and b in fam:
            return True
    return False'''

METHOD_F7 = '''    def _echo_bridge(self, patch: SemanticPatch,
                     existing: SemanticPatch) -> bool:
        """v0.1.17 F7 (20-run live gate, round 2): (b6) shared-subject
        same-value echo bridge -- same- OR cross-type.

        F6 was green in isolation (102 tests, harness 5/5) yet the
        live loop still failed 6/20 runs, all CRR under-reporting,
        with two signatures -- echo criticals whose live word order
        breaks the slot-head anchor every earlier bridge keys on:

          'Audit retention set to 5 years' + 'Retain audit logs for
          exactly 5 years'  (heads 'audit' vs 'retain': (b2) requires
          head equality, (b) needs >=4 shared words -- the pair
          survived as TWO decisions in 6/6 failing runs, delivered as
          '[DEC] 5 years' + '[DEC] exactly 5')

          'Dashboard latency must be sub-2-second' + 'Confirmed
          sub-2-second dashboard'  (the echo leads with the
          confirmation verb, so (b5)'s bidirectional head containment
          fails; with a verb-first constraint BOTH directions fail)

        (b6) anchors on what the pair SHARES instead of where each
        payload starts. All required: identical non-empty numbers,
        identical units, compatible bounds, never plain/plain, <=2
        turns, and a shared SUBJECT NOUN (>=4 chars, purely alpha,
        not stopword/revision marker/bound word/unit/echo verb/
        property noun). The residue guard then keeps scoped facts
        apart: every unshared content word must be a confirmation
        verb (_ECHO_VERBS), a morphological twin of the other side
        ('retention' ~ 'retain'), or sit in property position after
        the shared subject with only non-blocker stopwords, echo
        verbs, property nouns, digits or <=3-char non-bound tokens
        between. 'platinum/silver fee', 'dashboard vs homepage',
        'cloud vs training budgets', 'audit vs payment logs' and the
        no-more-than/page-size pair all stay separate.
        """
        if abs(patch.turn_index - existing.turn_index) > 2:
            return False
        new_nums = _numbers(patch.payload)
        if not new_nums or new_nums != _numbers(existing.payload):
            return False
        if _units_in(patch.payload) != _units_in(existing.payload):
            return False
        bc_new = _bound_class(patch.payload)
        bc_old = _bound_class(existing.payload)
        if not _polarity_compatible(bc_new, bc_old):
            return False
        if bc_new == "plain" and bc_old == "plain":
            return False
        a_words = re.findall(r"\\w+", patch.payload.lower())
        b_words = re.findall(r"\\w+", existing.payload.lower())
        a_set, b_set = set(a_words), set(b_words)
        shared = a_set & b_set
        subjects = {w for w in shared
                    if len(w) >= 4 and w.isalpha()
                    and w not in _STOPWORDS and w not in _REVISION_MARKERS
                    and w not in _BOUND_UP and w not in _BOUND_DOWN
                    and w not in _BOUND_EXACT and w not in _POLARITY_NEG
                    and w not in _UNIT_ALIASES and w not in _ECHO_VERBS
                    and w not in _PROPERTY_NOUNS}
        if not subjects:
            return False
        for words, other in ((a_words, b_set), (b_words, a_set)):
            subj_pos = [i for i, w in enumerate(words) if w in subjects]
            for i, w in enumerate(words):
                if (len(w) < 4 or not w.isalpha() or w in shared
                        or w in _STOPWORDS or w in _REVISION_MARKERS
                        or w in _ECHO_VERBS or w in _BOUND_UP
                        or w in _BOUND_DOWN or w in _BOUND_EXACT
                        or w in _POLARITY_NEG or w in _UNIT_ALIASES):
                    continue
                if any(_slot_family_eq(w, o) for o in other):
                    continue
                ok = False
                for sp in subj_pos:
                    if i > sp and all(
                            (t in _STOPWORDS and t not in _BRIDGE_BLOCKERS)
                            or t in _ECHO_VERBS or t in _PROPERTY_NOUNS
                            or any(c.isdigit() for c in t)
                            or (len(t) <= 3 and t not in _BOUND_UP
                                and t not in _BOUND_DOWN
                                and t not in _BOUND_EXACT)
                            for t in words[sp + 1:i]):
                        ok = True
                        break
                if not ok:
                    return False
        return True

'''

EDITS_MEMORY = [
    # module docstring: the F7 paragraph
    (
        'F6 half fixes the floor and the minimal story.\n'
        '"""',
        'F6 half fixes the floor and the minimal story.\n'
        '\n'
        '20-run live gate, round 2 (F7): F6 was green in isolation\n'
        '(102 tests, harness 5/5) but the live loop still failed 6/20\n'
        'runs, all CRR under-reporting (87/77/87/87/70/87). Both\n'
        'signatures are echo criticals whose live word order breaks\n'
        'the slot-head anchor every earlier bridge keys on. (i) The\n'
        'audit retention fact arrived twice -- noun first (\'Audit\n'
        'retention set to 5 years\', head \'audit\') and verb first\n'
        '(\'Retain audit logs for exactly 5 years\', head \'retain\') --\n'
        'and survived as TWO decisions in 6/6 failing runs (\'[DEC] 5\n'
        'years\' + \'[DEC] exactly 5\'). (ii) Dashboard echoes led\n'
        'with confirmation verbs (\'Confirmed sub-2-second dashboard\'\n'
        '-- head \'confirmed\', absent from the other payload), so\n'
        '(b5)\'s bidirectional head containment never fired. Rule\n'
        '(b6) bridges same- and cross-type echoes by what the pair\n'
        'SHARES instead of where each payload starts: a shared\n'
        'subject noun, identical numbers/units, compatible bounds,\n'
        'never plain/plain, within two turns -- plus a residue guard\n'
        '(every unshared content word must be a confirmation verb, a\n'
        'slot-family twin like retain~retention, or property-positioned\n'
        'behind the subject with only non-prepositional filler\n'
        'between). Scoped pairs -- platinum vs silver fee, dashboard\n'
        'vs homepage, cloud vs training budgets, audit vs payment\n'
        'logs -- stay separate.\n'
        '"""',
    ),
    # merge-rules comment block gains (b6)
    (
        '    #   (b5) cross-type word-order echo: BOTH slot heads present\n'
        '    #        in the other payload\'s word set + a shared\n'
        '    #        digit-bearing token as the value anchor;\n'
        '    #        numbers/units/bounds/proximity safety as (b3)/(b4)\n'
        '    #        (v0.1.17 20-run gate mode 1 -- \'Dashboard\n'
        '    #        sub-2-second\' [CON] + \'sub-2-second dashboard\' [DEC]\n'
        '    #        bloated the store to 8; CRR 87% on 3/20 runs)',
        '    #   (b5) cross-type word-order echo: BOTH slot heads present\n'
        '    #        in the other payload\'s word set + a shared\n'
        '    #        digit-bearing token as the value anchor;\n'
        '    #        numbers/units/bounds/proximity safety as (b3)/(b4)\n'
        '    #        (v0.1.17 20-run gate mode 1 -- \'Dashboard\n'
        '    #        sub-2-second\' [CON] + \'sub-2-second dashboard\' [DEC]\n'
        '    #        bloated the store to 8; CRR 87% on 3/20 runs)\n'
        '    #   (b6) shared-subject same-value echo (same- OR cross-type,\n'
        '    #        v0.1.17 F7): value safety as (b3)/(b5) + a shared\n'
        '    #        subject noun >=4 chars, never plain/plain, <=2 turns,\n'
        '    #        and NO conflicting scope noun in the residue --\n'
        '    #        unshared content words must be echo verbs, slot-family\n'
        '    #        twins (retain~retention), or property-positioned\n'
        '    #        behind the subject (round 2: 6/20 runs failed CRR\n'
        '    #        87-70% on \'Audit retention set to 5 years\' + \'Retain\n'
        '    #        audit logs for exactly 5 years\' and on\n'
        '    #        confirmation-verb dashboard echoes)',
    ),
    # residue vocab + family helper after _head_direction
    (
        '    if head in _BOUND_EXACT:\n'
        '        return "exact"\n'
        '    return "plain"',
        '    if head in _BOUND_EXACT:\n'
        '        return "exact"\n'
        '    return "plain"\n'
        '\n'
        '\n'
        + VOCAB_F7,
    ),
    # (b6) same-type call site, after (b2)
    (
        '                        stale.append(existing)          # (b2) restatement collapse\n'
        '                        continue\n'
        '                else:',
        '                        stale.append(existing)          # (b2) restatement collapse\n'
        '                        continue\n'
        '                    # v0.1.17 F7 (20-run live gate, round 2): (b6)\n'
        '                    # shared-subject echo bridge, same-type site.\n'
        '                    # Round 2 failed 6/20 runs on echo pairs whose\n'
        '                    # live word order breaks every head anchor:\n'
        '                    # \'Audit retention set to 5 years\' + \'Retain\n'
        '                    # audit logs for exactly 5 years\' (heads \'audit\'\n'
        '                    # vs \'retain\') survived as TWO decisions in 6/6\n'
        '                    # failing runs. (b6) anchors on the shared\n'
        '                    # subject noun + value safety + residue guard\n'
        '                    # instead -- see _echo_bridge.\n'
        '                    if self._echo_bridge(patch, existing):\n'
        '                        stale.append(existing)          # (b6) shared-subject echo bridge\n'
        '                        continue\n'
        '                else:',
    ),
    # (b6) cross-type call site, after (b5)
    (
        '                        stale.append(existing)          # (b5) compound-token bridge\n'
        '                        continue\n'
        '\n'
        '        # bake replaced values into the surviving payload',
        '                        stale.append(existing)          # (b5) compound-token bridge\n'
        '                        continue\n'
        '                    # v0.1.17 F7: (b6) shared-subject echo bridge,\n'
        '                    # cross-type site -- the confirmation-verb echo\n'
        '                    # shapes that slip (b5)\'s bidirectional head\n'
        '                    # containment (\'Keep Dashboard latency\n'
        '                    # sub-2-second\' + \'Confirmed sub-2-second\n'
        '                    # dashboard\': neither \'keep\' nor \'confirmed\'\n'
        '                    # appears in the other payload, while every\n'
        '                    # other (b5) condition passes).\n'
        '                    if self._echo_bridge(patch, existing):\n'
        '                        stale.append(existing)          # (b6) shared-subject echo bridge\n'
        '                        continue\n'
        '\n'
        '        # bake replaced values into the surviving payload',
    ),
    # the _echo_bridge method, before _jaccard
    (
        '    def _jaccard(self, left: str, right: str) -> float:',
        METHOD_F7 + '    def _jaccard(self, left: str, right: str) -> float:',
    ),
]
patch_file(P["memory"], EDITS_MEMORY)

# ---------------------------------------------------------------------------
# PHASE 4 -- GREEN re-check
# ---------------------------------------------------------------------------

hr("PHASE 4 -- GREEN RE-CHECK: 14/14")
green = run_pytest(["-q"] + NODES, capture=True)
if green.returncode != 0:
    die("GREEN check failed -- the (b6) rule did not land correctly. "
        "Paste everything back; do NOT re-run F7 yet.")
print("GREEN: 14/14 pass after patching.")

# ---------------------------------------------------------------------------
# PHASE 5 -- full gate
# ---------------------------------------------------------------------------

hr("PHASE 5 -- FULL GATE: pytest tests/ -q   (116 expected = 102 + 14)")
coll = run_pytest(["tests/", "-q", "--collect-only"], capture=True)
m = re.search(r"(\d+) tests? collected", coll.stdout)
total = int(m.group(1)) if m else -1
print("collected: %d (expected 116 = 102 + 14; a different count is fine "
      "if you added/removed tests yourself)" % total)

gate = run_pytest(["tests/", "-q"], capture=False)
if gate.returncode != 0:
    print()
    print("F7: FULL GATE FAILED -- the patches ARE on disk. Do NOT re-run")
    print("F7 (it is idempotent, but I need the failure text first). Paste")
    print("everything back as-is -- especially any legacy merge test that")
    print("changed; (b6) is a strict fallback after (b2)/(b3)/(b4)/(b5),")
    print("so a changed legacy test means a residue-guard leak to name.")
    sys.exit(2)

# ---------------------------------------------------------------------------
# PHASE 6 -- verify slices + merge probes + run replays
# ---------------------------------------------------------------------------

VERIFY = r'''
import inspect
import dspm
from dspm import DSPMMemory
from dspm.engine import DSPMEngine
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


def mk(pid, turn, t, pl):
    return SemanticPatch(patch_id=pid, turn_index=turn, patch_type=t,
                         payload=pl, dependencies=[])


def probe(label, rows, expect):
    m = DSPMMemory(budget=250)
    for r in rows:
        m._merge_patch(mk(*r))
    got = len(m.critical_patches)
    print("%-56s -> %d criticals (%s)"
          % (label, got, "OK" if got == expect else "MISMATCH"))


print("dspm.__version__ =", dspm.__version__)
show("round 2 -- rule (b6) method", inspect.getsource(DSPMMemory._echo_bridge),
     "(b6) shared-subject", 8)
show("round 2 -- (b6) call sites", inspect.getsource(DSPMMemory._merge_patch),
     "(b6) shared-subject echo bridge", 4)

print("-- the two live signatures now collapse:")
probe("retention echo (runs 4/5/8/10/12/15/17 'exactly 5')", [
    ("b0", 7, "decision", "Audit retention set to 5 years"),
    ("b1", 7, "decision", "Retain audit logs for exactly 5 years")], 1)
probe("retention echo, symmetric direction", [
    ("b1", 0, "decision", "Retain audit logs for exactly 5 years"),
    ("b0", 1, "decision", "Audit retention set to 5 years")], 1)
probe("dashboard echo, both heads dirty (run 5 pair)", [
    ("c0", 7, "constraint", "Keep Dashboard latency sub-2-second"),
    ("c1", 8, "decision", "Confirmed sub-2-second dashboard")], 1)
probe("dashboard echo, one dirty head", [
    ("d0", 7, "constraint", "Dashboard latency must be sub-2-second"),
    ("d1", 8, "decision", "Confirmed sub-2-second dashboard")], 1)
probe("same-type under-2 echo (run 8 pair)", [
    ("e0", 6, "decision", "dashboard must load in under 2 seconds"),
    ("e1", 7, "decision", "sub-2-second dashboard confirmed")], 1)
probe("t6 load-time variant + verb-first echo", [
    ("f0", 6, "constraint", "dashboard load time must be under 2 seconds"),
    ("f1", 7, "constraint", "Keep Dashboard latency sub-2-second")], 1)
print("-- the freeze set still holds:")
probe("platinum/silver tier fees still stay", [
    ("t0", 0, "constraint", "Maximum platinum fee is 0.5%"),
    ("t1", 1, "decision", "silver tier fee cap 0.5%")], 2)
probe("gold/silver tiers still stay", [
    ("g0", 0, "constraint", "Gold tier fee capped at 0.5%"),
    ("g1", 1, "decision", "Silver tier fee capped at 0.5%")], 2)
probe("plain budget scopes still stay (cloud/training)", [
    ("c0", 0, "constraint", "Budget 100 for cloud"),
    ("c1", 1, "decision", "Budget 100 for training")], 2)
probe("scoped budget variants still stay", [
    ("s0", 0, "constraint", "Cloud budget cap 100"),
    ("s1", 1, "decision", "budget cap 100 for training")], 2)
probe("dashboard/homepage still stay", [
    ("d0", 0, "constraint", "Dashboard loads in under 2 seconds"),
    ("d1", 1, "decision", "Homepage loads in under 2 seconds")], 2)
probe("'no more than' pair still stays", [
    ("n0", 0, "constraint", "No more than 500 records per page"),
    ("n1", 1, "decision", "page size capped at 500")], 2)
probe("scope-before-subject dashboards still stay (Guest/Admin)", [
    ("a0", 0, "decision", "Guest dashboard sub-2-second"),
    ("a1", 1, "decision", "Admin dashboard sub-2-second")], 2)
probe("no-shared-subject log pair still stays (property nouns)", [
    ("l0", 0, "decision", "Audit logs retained 5 years"),
    ("l1", 1, "decision", "Retain payment logs for exactly 5 years")], 2)
probe("retention vs periodic review still stays", [
    ("r0", 0, "decision", "Audit retention set to 5 years"),
    ("r1", 1, "decision", "Review audit logs every 5 years")], 2)
probe("different values (5y vs 3y) still stay", [
    ("v0", 0, "decision", "Retain audit logs for exactly 5 years"),
    ("v1", 1, "decision", "Retain audit logs for exactly 3 years")], 2)
probe("different units (years vs months) still stay", [
    ("u0", 0, "decision", "Retain audit logs for exactly 5 years"),
    ("u1", 1, "decision", "Retain audit logs for exactly 5 months")], 2)
probe("beyond two turns still stays", [
    ("p0", 0, "decision", "Audit retention set to 5 years"),
    ("p1", 5, "decision", "Retain audit logs for exactly 5 years")], 2)
print("-- pre-existing merges untouched:")
probe("canary scope-noun pair still merges (documented)", [
    ("k0", 0, "constraint", "Max users 500 per org"),
    ("k1", 1, "decision", "org admins cap 500")], 1)
probe("'for admins/guests' dashboards merge via (b4) (frozen canary)", [
    ("a0", 0, "constraint", "Dashboard sub-2-second for admins"),
    ("a1", 1, "decision", "sub-2-second dashboard for guests")], 1)
probe("'retain audit/payment logs' merge via (b) (pre-existing)", [
    ("l0", 0, "decision", "Retain audit logs for exactly 5 years"),
    ("l1", 1, "decision", "Retain payment logs for exactly 5 years")], 1)
probe("fee pair still collapses ((b4) untouched)", [
    ("f1", 0, "constraint", "Maximum fee per transaction is limited to 0.5%"),
    ("f2", 1, "decision", "fee cap set to 0.5% as tracked")], 1)
probe("run-9 reversed dashboard echo still collapses ((b5) untouched)", [
    ("b0", 7, "constraint", "Dashboard latency must be sub-2-second"),
    ("b1", 8, "decision", "sub-2-second dashboard confirmed")], 1)

eng = DSPMEngine(budget=59)
print("trim fidelity (failing-run ctx lines):",
      repr(eng._trim_numeric_first("Audit retention set to 5 years", 2)),
      repr(eng._trim_numeric_first("Retain audit logs for exactly 5 years", 2)),
      repr(eng._trim_numeric_first("Keep Dashboard latency sub-2-second", 2)),
      repr(eng._trim_numeric_first("Confirmed sub-2-second dashboard", 2)),
      repr(eng._trim_numeric_first("dashboard must load in under 2 seconds", 2)))

BASE = [
    ("r0", 0, "constraint",
     "All payments must be PCI-DSS compliant as a non-negotiable requirement"),
    ("r1", 0, "constraint", "Maximum fee per transaction is limited to 0.5%"),
    ("r2", 1, "decision", "fee cap set to 0.5% as tracked"),
    ("r3", 3, "decision", "automatic retries on failure enabled"),
    ("r4", 5, "decision",
     "Webhook timeout updated to 10 seconds, superseding the previous 30 seconds value (was 30)"),
]

RUN5 = BASE + [
    ("r5", 6, "constraint", "dashboard load time must be under 2 seconds"),
    ("r6", 7, "decision", "Audit retention set to 5 years"),
    ("r7", 7, "decision", "Retain audit logs for exactly 5 years"),
    ("r8", 7, "constraint", "Keep Dashboard latency sub-2-second"),
    ("r9", 7, "decision", "Confirmed sub-2-second dashboard"),
]
m = DSPMMemory(budget=59)
for r in RUN5:
    m._merge_patch(mk(*r))
ctx = m.get_context("webhook timeout")
s = m.stats
print("run-5 replay: store=%d delivered=%d crr=%d (expect 6, 6, 100)"
      % (s["critical_total"], s["critical_selected"], s["crr"]))
for line in ctx.splitlines():
    print("   " + line)

RUN8 = BASE + [
    ("r5", 6, "decision", "dashboard must load in under 2 seconds"),
    ("r6", 7, "decision", "Audit retention set to 5 years"),
    ("r7", 7, "decision", "Retain audit logs for exactly 5 years"),
    ("r8", 7, "decision", "sub-2-second dashboard confirmed"),
]
m2 = DSPMMemory(budget=59)
for r in RUN8:
    m2._merge_patch(mk(*r))
m2.get_context("webhook timeout")
s2 = m2.stats
print("run-8 replay: store=%d delivered=%d crr=%d (expect 6, 6, 100)"
      % (s2["critical_total"], s2["critical_selected"], s2["crr"]))

DIAG = [
    ("d0", 0, "constraint",
     "All payments must be PCI-DSS compliant as a non-negotiable requirement"),
    ("d1", 0, "constraint", "Maximum fee per transaction is limited to 0.5%"),
    ("d2", 1, "decision", "fee cap set to 0.5% as tracked"),
    ("d3", 5, "decision",
     "Webhook timeout updated to 10 seconds, superseding previous 30 seconds value (was 30)"),
    ("d4", 6, "decision", "dashboard must load in under 2 seconds"),
    ("d5", 7, "decision", "Audit retention set to 5 years"),
    ("d6", 7, "constraint", "Dashboard latency must be sub-2-second"),
]
m3 = DSPMMemory(budget=59)
for r in DIAG:
    m3._merge_patch(mk(*r))
m3.get_context("webhook timeout")
s3 = m3.stats
print("DIAG-store replay (paren form): store=%d delivered=%d crr=%d "
      "(expect 5, 5, 100)"
      % (s3["critical_total"], s3["critical_selected"], s3["crr"]))

m4 = DSPMMemory(budget=59)
for r in DIAG + [
    ("v1", 8, "decision", "homepage response under 2 seconds"),
    ("v2", 9, "decision", "PCI compliance tracked"),
]:
    m4._merge_patch(mk(*r))
m4.get_context("webhook timeout")
s4 = m4.stats
print("variance probe (+2 variants, store %d): delivered=%d crr=%d "
      "(expect 7, 7, 100)"
      % (s4["critical_total"], s4["critical_selected"], s4["crr"]))

m5 = DSPMMemory(budget=59)
for r in DIAG + [
    ("x1", 7, "decision", "Retain audit logs for exactly 5 years"),
    ("x2", 8, "decision", "homepage response under 2 seconds"),
    ("x3", 9, "decision", "PCI compliance tracked"),
]:
    m5._merge_patch(mk(*r))
m5.get_context("webhook timeout")
s5 = m5.stats
print("variance probe round 2 (+retention echo +2 variants, store %d): "
      "delivered=%d crr=%d (expect 7, 7, 100)"
      % (s5["critical_total"], s5["critical_selected"], s5["crr"]))
'''

hr("PHASE 6 -- VERIFY SLICES + MERGE PROBES + RUN REPLAYS")
subprocess.run([sys.executable, "-c", VERIFY], cwd=ROOT)

# ---------------------------------------------------------------------------
# PHASE 6b -- offline harness (review point 5)
# ---------------------------------------------------------------------------

hr("PHASE 6b -- OFFLINE HARNESS: expect AGGREGATE: 5/5")
harn = None
for cand in (os.path.join(ROOT, "eval_harness.py"),
             os.path.join(ROOT, "scripts", "eval_harness.py")):
    if os.path.isfile(cand):
        harn = cand
        break
if harn is None:
    print("[skip: eval_harness.py not found next to the project root or in")
    print(" scripts/ -- run it yourself before the live loop; offline it")
    print(" must read AGGREGATE: 5/5]")
else:
    r = subprocess.run([sys.executable, harn], cwd=ROOT,
                       capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    out = r.stdout or ""
    print(out[-3000:])
    if "AGGREGATE: 5/5" not in out:
        print()
        print("F7: HARNESS NOT 5/5 -- the patches are on disk. Do NOT")
        print("re-run F7. Paste this block back as-is.")
        sys.exit(2)
    print("harness gate: AGGREGATE 5/5. The tier_trap 5th split guard and")
    print("the trim_quality S3 report are the documented pre-existing")
    print("outputs; nothing changed there.")

hr("F7 COMPLETE")
print("Rule (b6) collapses both live-gate round-2 signatures: the")
print("verb-first retention echo ('Audit retention set to 5 years' +")
print("'Retain audit logs for exactly 5 years') and the")
print("confirmation-verb dashboard echoes -- the store stays at 6")
print("against the 7-critical delivery cap. Backups: .f7bak next to")
print("memory.py and tests/test_memory.py (delete when satisfied).")
print()
print("NEXT -- re-gate in this order, offline first:")
print("  1. full gate:  pytest tests/ -q        (116 expected = 102 + 14)")
print("  2. harness:    & $py eval_harness.py   (AGGREGATE: 5/5 -- already")
print("     run in PHASE 6b if this script found it)")
print()
print("Then the LIVE 20-run loop (keep your current short-lived key;")
print("revoke it only after the loop is green):")
print()
print('  1..20 | ForEach-Object { "== RUN $_ =="; & $py scripts\\smoke_conversation.py 2>&1; Start-Sleep 2 } | Tee-Object -FilePath smoke_20run_log.txt | Select-String -CaseSensitive "^== RUN|^\\[FAIL\\]"')
print()
print("Green = twenty '== RUN n ==' headers and ZERO [FAIL] lines (the")
print("4x PASS lines are inside smoke_20run_log.txt). On ANY [FAIL]:")
print("  a. run scripts\\diag_smoke_criticals.py (the v2 in the project")
print("     root prints the DIAG block AND the uncovered-pair scan)")
print("  b. paste the DIAG block, the failing run's full section from")
print("     smoke_20run_log.txt, AND the pre-trim payload dump your")
print("     smoke now prints (memory.patches before trimming) -- round 3")
print("     evidence.")
print()
print("After green, resume the publish checklist IN THIS ORDER:")
print("  1. README T7 fix FIRST (line 79: '7-stage pipeline ... adaptive")
print("     budgeting' -> 6-stage + '(reserved, not wired in this")
print("     release)'; line 212: 'T1-T7 pipeline' -> 'T1-T6 pipeline")
print("     (T7 reserved)'). The dist you already built is STALE -- PyPI")
print("     renders the README from the package, so rebuild after this.")
print("  2. CHANGELOG v0.1.17 'Fixed' section gains the (b6) bullet")
print("     (paste-ready text was handed over with F7).")
print("  3. if (Test-Path dist) { Remove-Item -Recurse -Force dist }")
print("     python -m build")
print("  4. python -m twine check dist/*   (both files must say PASSED)")
print("  5. TestPyPI dry run + the throwaway-venv install check")
print("  6. python -m twine upload dist/*   (real PyPI)")
print('  7. git add pyproject.toml CHANGELOG.md README.md')
print('     git add src tests scripts ":(exclude)*.f2bak" ":(exclude)*.f3bak" ":(exclude)*.f4bak" ":(exclude)*.f5bak" ":(exclude)*.f6bak" ":(exclude)*.f7bak"')
print('     git status')
print('     git commit -m "Release v0.1.17"')
print('     git tag -a v0.1.17 -m "DSPM v0.1.17"')
print('     git push origin v0.1.17')
print('     git push origin v0.1.16-dev')
print("  8. Revoke the temporary GROQ key. Budget 59 stays untouched.")
