# F6_live_gate_fixes.py -- v0.1.17 20-run live gate: the flake hunt caught
# two failure modes the offline suites structurally could not (both are
# LIVE-EXTRACTION shapes):
#
#   Mode 1 -- CRR 87% on 3/20 runs (9, 11, 20). The same dashboard
#   latency fact arrived twice: 'Dashboard sub-2-second' [CON] and
#   'sub-2-second dashboard' [DEC] -- a WORD-ORDER REVERSAL. The pair's
#   distinctive shared token ('sub-2-second') carries a digit, so the
#   (b3) ">=2 shared NON-numeric words" guard counted only 'dashboard',
#   and (b4) needs a bound-word slot head. The pair never collapsed,
#   the store bloated to 8 against the 7-critical delivery cap, and CRR
#   read 87% (7/8 delivered). Rule (b5) bridges order-insensitive
#   cross-type echoes: BOTH slot heads present in the other payload's
#   word set (a scope noun on one side only, as in the tier guards,
#   never bridges), a shared digit-bearing token as the value anchor,
#   plus every (b3)/(b4) safety: identical numbers, identical units,
#   compatible bounds, never plain/plain, within two turns.
#
#   Mode 2 -- trim truncation on 1/20 runs (19). A comma-form revision
#   ('webhook timeout updated to 10 seconds, was 30 seconds' -- no
#   '(was' marker) floored at 2 words, 'was' scored as filler, and the
#   2-word allowance delivered '[DEC] 10 seconds,' with the replaced
#   value GONE while CRR stayed 100. Engine fixes: _word_floor
#   recognizes comma-form history, and the minimal-story path
#   normalizes it to the canonical '10 (was 30)'; a '(was 30 seconds)'
#   span with the unit inside the parens also compacts at the floor
#   instead of falling through to the valueless ladder result.
#
# Save in the project root and run:
#     & $py F6_live_gate_fixes.py
#
# Flow: PHASE 0 locate+versions -> PHASE 1 tests (8 new) -> PHASE 2 RED
# baseline (5 of 8 must fail: bridge x2, floor, comma-story, store) ->
# PHASE 3 anchored patches (memory.py + engine.py) -> PHASE 4 GREEN 8/8
# -> PHASE 5 full gate (102 expected = 94 + 8) -> PHASE 6 slices +
# probes + run-19/DIAG replays -> PHASE 6b offline harness
# (AGGREGATE: 5/5).
#
# Idempotent; partial states handled; originals backed up ONCE to
# <file>.f6bak. All anchors are ASCII-only (no em-dash lines matched).

import os
import py_compile
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

NEW8 = [
    "test_compound_token_bridge_collapses_reversed_pair",
    "test_compound_token_bridge_symmetric_direction",
    "test_compound_token_bridge_keeps_different_values",
    "test_compound_token_bridge_keeps_different_units",
    "test_compound_token_bridge_requires_echo_proximity",
    "test_word_floor_covers_comma_form_history",
    "test_comma_form_revision_trims_to_canonical_story",
    "test_starved_store_keeps_both_revision_values_comma_form",
]
NODES = ["tests/test_memory.py::" + n for n in NEW8]


def die(msg):
    print()
    print("F6 ABORT -- " + msg)
    print("Each file is written only after ALL its anchors validate, so a")
    print(".f6bak restore should not be needed. Paste this output back as-is.")
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
    bak = fp + ".f6bak"
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
# PHASE 1 -- the eight F6 tests
# ---------------------------------------------------------------------------

TESTS_F6 = '''
# ---------------------------------------------------------------------------
# v0.1.17 20-run live gate (F6): two live-extraction failure modes the
# offline suites could not produce. Mode 1 (runs 9/11/20, CRR 87%): the
# same dashboard latency fact arrived as 'Dashboard sub-2-second' [CON]
# and 'sub-2-second dashboard' [DEC] -- a word-order reversal whose
# shared digit-bearing compound the (b3) non-numeric word count
# excluded, so the pair never collapsed and the store bloated to 8
# against the 7-critical delivery cap. Rule (b5) bridges it. Mode 2
# (run 19): a comma-form revision ('... 10 seconds, was 30 seconds')
# fell to the 2-word floor -- no '(was' marker -- and the 2-word
# allowance delivered '[DEC] 10 seconds,' with the replaced value
# gone. The comma form now floors at 3 and the minimal story
# normalizes to '10 (was 30)'.
# ---------------------------------------------------------------------------


def test_compound_token_bridge_collapses_reversed_pair():
    """Live run-9 shape: the DEC echo leads with the value compound, the
    CON leads with the subject noun -- heads in reversed order, and the
    pair shares only ONE non-numeric word, below (b3)'s >=2 guard."""
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('s0', 7, 'constraint',
                                 'Dashboard latency must be sub-2-second', []))
    m._merge_patch(SemanticPatch('s1', 8, 'decision',
                                 'sub-2-second dashboard confirmed', []))
    crits = [p.payload for p in m.critical_patches]
    assert len(crits) == 1, \\
        f"reversed dashboard echo should collapse: {crits}"


def test_compound_token_bridge_symmetric_direction():
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('s0', 0, 'decision',
                                 'sub-2-second dashboard confirmed', []))
    m._merge_patch(SemanticPatch('s1', 1, 'constraint',
                                 'Dashboard latency must be sub-2-second', []))
    assert len(m.critical_patches) == 1, \\
        f"symmetric reversed echo should collapse: {[p.payload for p in m.critical_patches]}"


def test_compound_token_bridge_keeps_different_values():
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('s0', 0, 'constraint',
                                 'Dashboard latency must be sub-2-second', []))
    m._merge_patch(SemanticPatch('s1', 1, 'decision',
                                 'sub-3-second dashboard confirmed', []))
    assert len(m.critical_patches) == 2


def test_compound_token_bridge_keeps_different_units():
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('s0', 0, 'constraint',
                                 'Dashboard latency must be sub-2-second', []))
    m._merge_patch(SemanticPatch('s1', 1, 'decision',
                                 'sub-2-minute dashboard confirmed', []))
    assert len(m.critical_patches) == 2


def test_compound_token_bridge_requires_echo_proximity():
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('s0', 0, 'constraint',
                                 'Dashboard latency must be sub-2-second', []))
    m._merge_patch(SemanticPatch('s1', 5, 'decision',
                                 'sub-2-second dashboard confirmed', []))
    assert len(m.critical_patches) == 2


def test_word_floor_covers_comma_form_history():
    """Run 19's root cause: a comma-form revision carried no '(was'
    marker, floored at 2, and the 2-word allowance lost the replaced
    value entirely."""
    from dspm.engine import DSPMEngine
    eng = DSPMEngine(budget=59)
    assert eng._word_floor('webhook timeout updated to 10 seconds (was 30)') == 3
    assert eng._word_floor(
        'webhook timeout updated to 10 seconds, was 30 seconds') == 3
    assert eng._word_floor(
        'webhook timeout updated to 10 seconds, previously 30 seconds') == 3
    assert eng._word_floor('dashboard latency must be sub-2-second') == 2
    assert eng._word_floor('webhook timeout was updated to 10 seconds') == 2


def test_comma_form_revision_trims_to_canonical_story():
    from dspm.engine import DSPMEngine
    eng = DSPMEngine(budget=59)
    out = eng._trim_numeric_first(
        'webhook timeout updated to 10 seconds, was 30 seconds', 3)
    assert out == '10 (was 30)', out
    out2 = eng._trim_numeric_first(
        'webhook timeout updated to 10 seconds, previously 30 seconds', 3)
    assert out2 == '10 (was 30)', out2
    # the classic paren form keeps its story
    out3 = eng._trim_numeric_first(
        'webhook timeout updated to 10 seconds (was 30)', 3)
    assert out3 == '10 (was 30)', out3


def test_starved_store_keeps_both_revision_values_comma_form():
    """Store-level replay of live run 19: budget 59, seven criticals
    after the merges, the revision in comma form. The delivered context
    must keep BOTH the current value and the replaced one."""
    m = DSPMMemory(budget=59)
    rows = [
        ('r0', 0, 'constraint',
         'All payments must be PCI-DSS compliant as a non-negotiable requirement'),
        ('r1', 0, 'constraint', 'Maximum fee per transaction is limited to 0.5%'),
        ('r2', 1, 'decision', 'fee cap set to 0.5% as tracked'),
        ('r3', 5, 'decision',
         'webhook timeout updated to 10 seconds, was 30 seconds'),
        ('r4', 6, 'decision', 'dashboard must load in under 2 seconds'),
        ('r5', 7, 'decision', 'Audit retention set to 5 years'),
        ('r6', 7, 'constraint', 'Dashboard latency must be sub-2-second'),
        ('r7', 8, 'decision', 'homepage response under 2 seconds'),
        ('r8', 9, 'decision', 'PCI compliance tracked'),
    ]
    for pid, turn, t, pl in rows:
        m._merge_patch(SemanticPatch(pid, turn, t, pl, []))
    ctx = m.get_context('webhook timeout')
    assert '10' in ctx, ctx
    assert '30' in ctx, ctx
    s = m.stats
    assert s['crr'] == 100, s
'''

hr("PHASE 1 -- F6 TESTS PRESENT?")
tsrc = file_text(P["tests"])
present = [n for n in NEW8 if ("def " + n + "(") in tsrc]
if len(present) == len(NEW8):
    print("[skip: all eight F6 tests already in tests/test_memory.py]")
elif present:
    die("only some of the eight F6 tests are present (%s) -- paste the "
        "tail of tests/test_memory.py back" % ", ".join(present))
else:
    backup(P["tests"])
    write_text(P["tests"], tsrc.rstrip("\n") + "\n\n\n" + TESTS_F6.strip("\n") + "\n")
    print("  tests appended: 8 new (2 bridge reds, 3 bridge guards, 3 trim)")

# ---------------------------------------------------------------------------
# PHASE 2 -- RED baseline: 5 of 8 must fail
# ---------------------------------------------------------------------------

hr("PHASE 2 -- RED BASELINE: 5 must fail (bridge x2, floor, story, store)")
base = run_pytest(["-q"] + NODES, capture=True)
if base.returncode == 0:
    print("NOTE: all eight already pass -- fixes already applied? The patch")
    print("phase below is idempotent and will simply skip.")
else:
    m = re.search(r"(\d+) failed", base.stdout)
    n = int(m.group(1)) if m else -1
    if not (1 <= n <= 5):
        die("unexpected baseline result (%r failed of 8) -- expected at "
            "most the 5 known-red tests. Paste the output above" % n)
    print("RED BASELINE OK: %d/5 known-red tests fail on the unfixed tree "
          "(the other %d guards already pass)." % (n, 8 - n))

# ---------------------------------------------------------------------------
# PHASE 3 -- anchored patches
# ---------------------------------------------------------------------------

hr("PHASE 3 -- MODE 1 (b5) memory.py + MODE 2 engine.py: core files")

EDITS_MEMORY = [
    # module docstring: the F6 paragraph
    (
        'prose is droppable -- stats separates conversation/patch/\n'
        'context tokens (review point 4), patches carry source_role\n'
        'through persistence (review point 2), and the smoke\'s stale\n'
        'v0.1.14 hint is retired.\n'
        '"""',
        'prose is droppable -- stats separates conversation/patch/\n'
        'context tokens (review point 4), patches carry source_role\n'
        'through persistence (review point 2), and the smoke\'s stale\n'
        'v0.1.14 hint is retired.\n'
        '\n'
        '20-run live gate (F6): the loop caught two live-extraction\n'
        'shapes the offline suites could not. Mode 1 -- the same\n'
        'dashboard latency fact arrived as a word-order reversal\n'
        '(\'Dashboard sub-2-second\' [CON] vs \'sub-2-second dashboard\'\n'
        '[DEC]); its distinctive shared token carries a digit, so the\n'
        '(b3) non-numeric word count never reached two and the store\n'
        'bloated to 8 against the 7-critical delivery cap (CRR 87% on\n'
        '3/20 runs). Rule (b5) bridges order-insensitive echoes under\n'
        'every (b3)/(b4) safety. Mode 2 -- comma-form revision history\n'
        'lost the replaced value at the 2-word floor; the engine-side\n'
        'F6 half fixes the floor and the minimal story.\n'
        '"""',
    ),
    # merge-rules comment block gains (b5)
    (
        '    #        the store at 6 against the 7-critical delivery cap;\n'
        '    #        6/20 live runs tipped to 8 and CRR fell to 87%)',
        '    #        the store at 6 against the 7-critical delivery cap;\n'
        '    #        6/20 live runs tipped to 8 and CRR fell to 87%)\n'
        '    #   (b5) cross-type word-order echo: BOTH slot heads present\n'
        '    #        in the other payload\'s word set + a shared\n'
        '    #        digit-bearing token as the value anchor;\n'
        '    #        numbers/units/bounds/proximity safety as (b3)/(b4)\n'
        '    #        (v0.1.17 20-run gate mode 1 -- \'Dashboard\n'
        '    #        sub-2-second\' [CON] + \'sub-2-second dashboard\' [DEC]\n'
        '    #        bloated the store to 8; CRR 87% on 3/20 runs)',
    ),
    # rule (b5) after (b4)
    (
        '                        stale.append(existing)          # (b4) thin echo bridge\n'
        '                        continue',
        '                        stale.append(existing)          # (b4) thin echo bridge\n'
        '                        continue\n'
        '                    # v0.1.17 20-run live gate (mode 1 -- runs\n'
        '                    # 9/11/20 at CRR 87%): (b5) compound-token\n'
        '                    # bridge. The same dashboard latency fact\n'
        '                    # arrived as \'Dashboard sub-2-second\' [CON]\n'
        '                    # and \'sub-2-second dashboard\' [DEC] -- a\n'
        '                    # word-order reversal. Its distinctive shared\n'
        '                    # token (\'sub-2-second\') carries a digit, so\n'
        '                    # the >=2 shared NON-numeric guard in (b3)\n'
        '                    # counted only \'dashboard\' and the pair never\n'
        '                    # collapsed; the store bloated to 8 against\n'
        '                    # the 7-critical delivery cap. (b5) bridges\n'
        '                    # order-insensitive echoes: BOTH slot heads\n'
        '                    # present in the other payload\'s word set (a\n'
        '                    # scope noun on one side only, as in the tier\n'
        '                    # guards, never bridges), a shared\n'
        '                    # digit-bearing token as the value anchor,\n'
        '                    # plus every (b3)/(b4) safety: identical\n'
        '                    # numbers, identical units, compatible bounds,\n'
        '                    # never plain/plain, <=2 turns.\n'
        '                    old_words = set(re.findall(r"\\w+", existing.payload.lower()))\n'
        '                    new_words = set(re.findall(r"\\w+", patch.payload.lower()))\n'
        '                    if (len(old_words & new_words) >= 2\n'
        '                            and any(re.search(r"\\d", w) for w in old_words & new_words)\n'
        '                            and new_nums == _numbers(existing.payload)\n'
        '                            and old_head and old_head in new_words\n'
        '                            and new_head and new_head in old_words\n'
        '                            and _units_in(patch.payload) == _units_in(existing.payload)\n'
        '                            and _polarity_compatible(bc_new, bc_old)\n'
        '                            and not (bc_new == "plain" and bc_old == "plain")\n'
        '                            and abs(patch.turn_index - existing.turn_index) <= 2):\n'
        '                        stale.append(existing)          # (b5) compound-token bridge\n'
        '                        continue',
    ),
]
patch_file(P["memory"], EDITS_MEMORY, soft=True)

EDITS_ENGINE = [
    # module docstring: the F6 paragraph
    (
        'experiments/bench_kv_cache.py).\n'
        '"""',
        'experiments/bench_kv_cache.py).\n'
        '\n'
        'v0.1.17 F6 (20-run live gate, run 19): comma-form revision\n'
        'history. The live extractor also emits "10 seconds, was 30\n'
        'seconds" -- no "(was" marker. Those payloads floored at 2 words,\n'
        '"was" scored as filler, and a 2-word allowance delivered "[DEC]\n'
        '10 seconds," with the replaced value gone entirely. _word_floor\n'
        'recognizes the comma form now, the minimal-story path normalizes\n'
        'it to the canonical "10 (was 30)", and a "(was 30 seconds)" span\n'
        'with the unit inside the parens compacts to "10 (was 30)" at\n'
        'the floor instead of falling through to the valueless ladder\n'
        'result.\n'
        '"""',
    ),
    # history-marker vocabulary next to the negation binder's skip words
    (
        '    _BIND_SKIP = (\n'
        '        "be", "is", "are", "was", "were", "been", "being", "the",\n'
        '        "a", "an", "to", "in", "on", "at", "for", "of", "or", "and",\n'
        '        "than", "must", "should", "will", "shall", "may", "might",\n'
        '        "can", "could", "our", "any", "all",\n'
        '    )',
        '    _BIND_SKIP = (\n'
        '        "be", "is", "are", "was", "were", "been", "being", "the",\n'
        '        "a", "an", "to", "in", "on", "at", "for", "of", "or", "and",\n'
        '        "than", "must", "should", "will", "shall", "may", "might",\n'
        '        "can", "could", "our", "any", "all",\n'
        '    )\n'
        '    # v0.1.17 F6: prose history markers that may directly precede\n'
        '    # the replaced value in a comma-form revision ("10 seconds,\n'
        '    # was 30 seconds", "... previously 30 seconds"). The floor and\n'
        '    # the minimal-story path both use them.\n'
        '    _HISTORY_MARKERS = (\n'
        '        "was", "were", "previous", "previously", "prior",\n'
        '        "earlier", "formerly", "superseded",\n'
        '    )',
    ),
    # _word_floor: comma-form history floors at 3
    (
        '        return 3 if "(was" in payload else 2',
        '        # v0.1.17 F6 (20-run live gate, run 19): the live\n'
        '        # extractor also emits comma-form history ("10 seconds,\n'
        '        # was 30 seconds") -- no "(was" marker, so it floored at\n'
        '        # 2, the 2-word allowance delivered "10 seconds," and the\n'
        '        # replaced value was gone.\n'
        '        if "(was" in payload:\n'
        '            return 3\n'
        '        if re.search(r"\\b(?:was|were|previous(?:ly)?|prior|earlier|"\n'
        '                     r"formerly|superseded)\\s+\\S*\\d",\n'
        '                     payload, re.IGNORECASE):\n'
        '            return 3\n'
        '        return 2',
    ),
    # minimal-story path: comma-form trigger + canonical normalization
    (
        '        if max_words <= 3 and any(w.lower().startswith("(was") for w in words):\n'
        '            was_start = next(i for i, w in enumerate(words)\n'
        '                             if w.lower().startswith("(was"))\n'
        '            j = was_start\n'
        '            while j < len(words) and not words[j].endswith(")"):\n'
        '                j += 1\n'
        '            was_end = min(j, len(words) - 1)\n'
        '            span = words[was_start:was_end + 1]',
        '        # v0.1.17 F6 (20-run live gate, run 19): revision history\n'
        '        # comes in two shapes -- the baked paren form "(was 30)"\n'
        '        # and the comma form a live extractor emits ("10 seconds,\n'
        '        # was 30 seconds"). Only the paren form was recognized,\n'
        '        # so comma-form payloads missed the minimal-story floor\n'
        '        # and the ladder kept the current value while dropping\n'
        '        # the replaced one entirely ("[DEC] 10 seconds,").\n'
        '        was_start = None\n'
        '        paren_form = False\n'
        '        for i, w in enumerate(words):\n'
        '            if w.lower().startswith("(was"):\n'
        '                was_start, paren_form = i, True\n'
        '                break\n'
        '        if was_start is None:\n'
        '            for i, w in enumerate(words):\n'
        '                if (w.lower().strip(".,;:") in self._HISTORY_MARKERS\n'
        '                        and i + 1 < len(words)\n'
        '                        and re.search(r"\\d", words[i + 1])):\n'
        '                    was_start, paren_form = i, False\n'
        '                    break\n'
        '        if max_words <= 3 and was_start is not None:\n'
        '            if paren_form:\n'
        '                j = was_start\n'
        '                while j < len(words) and not words[j].endswith(")"):\n'
        '                    j += 1\n'
        '                was_end = min(j, len(words) - 1)\n'
        '                span = words[was_start:was_end + 1]\n'
        '            else:\n'
        '                # comma form normalizes to the canonical story\n'
        '                # tokens: "was 30 seconds" -> "(was 30)"\n'
        '                span = ["(was", words[was_start + 1].rstrip(".,;:)") + ")"]\n'
        '            if len(span) > 2:\n'
        '                # the history unit is droppable: "10 (was 30\n'
        '                # seconds)" costs four words, "10 (was 30)"\n'
        '                # carries the same revision story in three.\n'
        '                span = [span[0], span[1].rstrip(".,;:)") + ")"]',
    ),
]
patch_file(P["engine"], EDITS_ENGINE)

# ---------------------------------------------------------------------------
# PHASE 4 -- GREEN re-check
# ---------------------------------------------------------------------------

hr("PHASE 4 -- GREEN RE-CHECK: 8/8")
green = run_pytest(["-q"] + NODES, capture=True)
if green.returncode != 0:
    die("GREEN check failed -- the fixes did not land correctly. Paste "
        "everything back; do NOT re-run F6 yet.")
print("GREEN: 8/8 pass after patching.")

# ---------------------------------------------------------------------------
# PHASE 5 -- full gate
# ---------------------------------------------------------------------------

hr("PHASE 5 -- FULL GATE: pytest tests/ -q   (102 expected = 94 + 8)")
coll = run_pytest(["tests/", "-q", "--collect-only"], capture=True)
m = re.search(r"(\d+) tests? collected", coll.stdout)
total = int(m.group(1)) if m else -1
print("collected: %d (expected 102 = 94 + 8; a different count is fine if "
      "you added/removed tests yourself)" % total)

gate = run_pytest(["tests/", "-q"], capture=False)
if gate.returncode != 0:
    print()
    print("F6: FULL GATE FAILED -- the patches ARE on disk. Do NOT re-run")
    print("F6 (it is idempotent, but I need the failure text first). Paste")
    print("everything back as-is -- especially any legacy trim test that")
    print("changed; the story path now also fires for comma-form payloads.")
    sys.exit(2)

# ---------------------------------------------------------------------------
# PHASE 6 -- verify slices + probes + replays
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
    print("%-52s -> %d criticals (%s)"
          % (label, got, "OK" if got == expect else "MISMATCH"))


print("dspm.__version__ =", dspm.__version__)
show("mode 1 -- rule (b5)", inspect.getsource(DSPMMemory._merge_patch),
     "(b5) compound-token", 14)
show("mode 2 -- comma-form floor", inspect.getsource(DSPMEngine._word_floor),
     "comma-form", 10)
show("mode 2 -- comma-form story",
     inspect.getsource(DSPMEngine._trim_numeric_first),
     "comma form normalizes", 12)

probe("run-9 shape: reversed dashboard echo collapses", [
    ("b0", 7, "constraint", "Dashboard latency must be sub-2-second"),
    ("b1", 8, "decision", "sub-2-second dashboard confirmed")], 1)
probe("reversed echo, symmetric direction", [
    ("b1", 0, "decision", "sub-2-second dashboard confirmed"),
    ("b0", 1, "constraint", "Dashboard latency must be sub-2-second")], 1)
probe("different values (2s vs 3s) stay", [
    ("v0", 0, "constraint", "Dashboard latency must be sub-2-second"),
    ("v1", 1, "decision", "sub-3-second dashboard confirmed")], 2)
probe("different units (second vs minute) stay", [
    ("u0", 0, "constraint", "Dashboard latency must be sub-2-second"),
    ("u1", 1, "decision", "sub-2-minute dashboard confirmed")], 2)
probe("beyond two turns stays", [
    ("p0", 0, "constraint", "Dashboard latency must be sub-2-second"),
    ("p1", 5, "decision", "sub-2-second dashboard confirmed")], 2)
probe("platinum/silver tier fees still stay (b3/b4 guards)", [
    ("t0", 0, "constraint", "Maximum platinum fee is 0.5%"),
    ("t1", 1, "decision", "silver tier fee cap 0.5%")], 2)
probe("gold/silver tiers still stay", [
    ("g0", 0, "constraint", "Gold tier fee capped at 0.5%"),
    ("g1", 1, "decision", "Silver tier fee capped at 0.5%")], 2)
probe("plain budget scopes still stay (cloud/training)", [
    ("c0", 0, "constraint", "Budget 100 for cloud"),
    ("c1", 1, "decision", "Budget 100 for training")], 2)
probe("dashboard/homepage still stay", [
    ("d0", 0, "constraint", "Dashboard loads in under 2 seconds"),
    ("d1", 1, "decision", "Homepage loads in under 2 seconds")], 2)
probe("canary scope-noun pair still merges (documented)", [
    ("k0", 0, "constraint", "Max users 500 per org"),
    ("k1", 1, "decision", "org admins cap 500")], 1)
probe("'no more than' head mismatch still stays", [
    ("n0", 0, "constraint", "No more than 500 records per page"),
    ("n1", 1, "decision", "page size capped at 500")], 2)
probe("fee pair still collapses ((b4) untouched)", [
    ("f1", 0, "constraint", "Maximum fee per transaction is limited to 0.5%"),
    ("f2", 1, "decision", "fee cap set to 0.5% as tracked")], 1)

eng = DSPMEngine(budget=59)
print("floor probes (expect 3 3 3 2 2):",
      eng._word_floor("webhook timeout updated to 10 seconds (was 30)"),
      eng._word_floor("webhook timeout updated to 10 seconds, was 30 seconds"),
      eng._word_floor(
          "webhook timeout updated to 10 seconds, previously 30 seconds"),
      eng._word_floor("dashboard latency must be sub-2-second"),
      eng._word_floor("webhook timeout was updated to 10 seconds"))
print("comma-form story   (expect '10 (was 30)'):",
      repr(eng._trim_numeric_first(
          "webhook timeout updated to 10 seconds, was 30 seconds", 3)))
print("previously story   (expect '10 (was 30)'):",
      repr(eng._trim_numeric_first(
          "webhook timeout updated to 10 seconds, previously 30 seconds", 3)))
print("paren-with-unit    (expect '10 (was 30)'):",
      repr(eng._trim_numeric_first(
          "webhook timeout updated to 10 seconds (was 30 seconds)", 3)))
print("paren classic      (expect '10 (was 30)'):",
      repr(eng._trim_numeric_first(
          "webhook timeout updated to 10 seconds (was 30)", 3)))
print("comma form at 2 words falls to the ladder (run 19's old shape):",
      repr(eng._trim_numeric_first(
          "webhook timeout updated to 10 seconds, was 30 seconds", 2)))

RUN19 = [
    ("r0", 0, "constraint",
     "All payments must be PCI-DSS compliant as a non-negotiable requirement"),
    ("r1", 0, "constraint", "Maximum fee per transaction is limited to 0.5%"),
    ("r2", 1, "decision", "fee cap set to 0.5% as tracked"),
    ("r3", 5, "decision",
     "webhook timeout updated to 10 seconds, was 30 seconds"),
    ("r4", 6, "decision", "dashboard must load in under 2 seconds"),
    ("r5", 7, "decision", "Audit retention set to 5 years"),
    ("r6", 7, "constraint", "Dashboard latency must be sub-2-second"),
    ("r7", 8, "decision", "homepage response under 2 seconds"),
    ("r8", 9, "decision", "PCI compliance tracked"),
]
m = DSPMMemory(budget=59)
for r in RUN19:
    m._merge_patch(mk(*r))
ctx = m.get_context("webhook timeout")
line = next((l for l in ctx.splitlines() if "10" in l), "")
s = m.stats
print("run-19 replay: store=%d delivered=%d crr=%d"
      % (s["critical_total"], s["critical_selected"], s["crr"]))
print("delivered webhook line:", repr(line))
print("both values in ctx (expect True):", "10" in ctx and "30" in ctx)

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
m2 = DSPMMemory(budget=59)
for r in DIAG:
    m2._merge_patch(mk(*r))
m2.get_context("webhook timeout")
s2 = m2.stats
print("DIAG-store replay (paren form): store=%d delivered=%d crr=%d "
      "(expect 5, 5, 100)"
      % (s2["critical_total"], s2["critical_selected"], s2["crr"]))

m3 = DSPMMemory(budget=59)
for r in DIAG + [
    ("v1", 8, "decision", "homepage response under 2 seconds"),
    ("v2", 9, "decision", "PCI compliance tracked"),
]:
    m3._merge_patch(mk(*r))
m3.get_context("webhook timeout")
s3 = m3.stats
print("variance probe (+2 variants, store %d): delivered=%d crr=%d "
      "(expect 7, 7, 100)"
      % (s3["critical_total"], s3["critical_selected"], s3["crr"]))
'''

hr("PHASE 6 -- VERIFY SLICES + REPLAYS")
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
        print("F6: HARNESS NOT 5/5 -- the patches are on disk. Do NOT")
        print("re-run F6. Paste this block back as-is.")
        sys.exit(2)
    print("harness gate: AGGREGATE 5/5. trim_quality still REPORTS the")
    print("mid-range span-form loss (allowance >= 4 ladder shape) -- that")
    print("remains the documented S3 / 0.1.18 item; the floor case that")
    print("failed live run 19 is fixed.")

hr("F6 COMPLETE")
print("Rule (b5) collapses the reversed dashboard echo (store 8 -> 7 =")
print("the delivery cap), and comma-form revision history floors at 3")
print("with the canonical '10 (was 30)' story -- both live-gate failure")
print("modes close. Backups: .f6bak next to every patched file (delete")
print("when satisfied).")
print()
print("NEXT -- re-gate in this order, offline first:")
print("  1. full gate:  pytest tests/ -q        (102 expected = 94 + 8)")
print("  2. harness:    & $py eval_harness.py   (AGGREGATE: 5/5 -- already")
print("     run in PHASE 6b if this script found it)")
print()
print("Then the LIVE 20-run loop (keep your current short-lived key;")
print("revoke it only after the loop is green):")
print()
print('  1..20 | ForEach-Object { "== RUN $_ =="; & $py scripts\\smoke_conversation.py 2>&1; Start-Sleep 2 } | Tee-Object -FilePath smoke_20run_log.txt | Select-String -CaseSensitive "^== RUN|^\\[FAIL\\]"')
print()
print("Green = twenty '== RUN n ==' headers and ZERO [FAIL] lines (the")
print("4x PASS lines are inside smoke_20run_log.txt). On ANY [FAIL]: run")
print("scripts\\diag_smoke_criticals.py and paste the DIAG block plus the")
print("failing run's full section from the log -- next evidence round.")
print()
print("After green, resume the publish checklist IN THIS ORDER:")
print("  1. README T7 fix FIRST (line 79: '7-stage pipeline ... adaptive")
print("     budgeting' -> 6-stage + '(reserved, not wired in this")
print("     release)'; line 212: 'T1-T7 pipeline' -> 'T1-T6 pipeline")
print("     (T7 reserved)'). The dist you already built is STALE -- PyPI")
print("     renders the README from the package, so rebuild after this.")
print("  2. if (Test-Path dist) { Remove-Item -Recurse -Force dist }")
print("     python -m build")
print("  3. python -m twine check dist/*   (both files must say PASSED)")
print("  4. TestPyPI dry run + the throwaway-venv install check")
print("  5. python -m twine upload dist/*   (real PyPI)")
print('  6. git add pyproject.toml CHANGELOG.md README.md')
print('     git add src tests scripts ":(exclude)*.f2bak" ":(exclude)*.f3bak" ":(exclude)*.f4bak" ":(exclude)*.f5bak" ":(exclude)*.f6bak"')
print('     git status')
print('     git commit -m "Release v0.1.17"')
print('     git tag -a v0.1.17 -m "DSPM v0.1.17"')
print('     git push origin v0.1.17')
print('     git push origin v0.1.16-dev')
print("  7. Revoke the temporary GROQ key. Budget 59 stays untouched.")