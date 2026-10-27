# F3_defect_fixes.py -- v0.1.17 code-review defect batch: findings F1-F4
# (dependency remap, unit aliases, "can't" polarity, bake recount),
# docstring honesty (F5/F6), and the version split F2 left behind
# (pyproject.toml already says 0.1.17, __init__.py still says 0.1.16).
#
# Save in the project root (next to F2_review_fixes.py) and run:
#     & $py F3_defect_fixes.py
#
# Flow:
#   PHASE 0  locate the package (dspm/ or src/dspm/), report versions
#   PHASE 1  make sure the four F1-F4 regression tests exist (append if
#            missing -- you already added them, so this will skip)
#   PHASE 2  RED BASELINE: the four tests must FAIL on the unfixed tree,
#            proving they are real regression tests, not vacuous ones
#   PHASE 3  anchored patches: F1 (extractor two-pass region replace),
#            F2+F3+F4 (memory.py), F5/F6 (engine/config docstrings),
#            version reconciliation (__init__.py 0.1.16 -> 0.1.17)
#   PHASE 4  GREEN re-check of the four tests (4/4 must pass)
#   PHASE 5  full pytest gate (71 expected: 67 + 4)
#   PHASE 6  verify slices + live runtime probes
#
# Expect roughly 6-9 minutes on your machine (two targeted pytest runs
# plus the full gate; your last full gate alone was 4:29).
# Idempotent: re-runs skip anything already applied, and a partially
# applied state (some fixes in, some not) is handled cleanly -- the
# baseline accepts 1-4 failures and only the missing patches apply.
# Every original is backed up ONCE to <file>.f3bak. Core edits abort on
# a missing anchor BEFORE the file is written; cosmetic edits (engine/
# config docstrings, version strings) warn and continue.

import os
import py_compile
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

NEW4 = [
    "test_extractor_remaps_forward_dependency_references",
    "test_unit_aliases_distinguish_milliseconds_from_seconds",
    "test_cant_is_recognized_as_negative_polarity",
    "test_baked_revision_recounts_token_cost",
]
NODES = ["tests/test_memory.py::" + n for n in NEW4]


def die(msg):
    print()
    print("F3 ABORT -- " + msg)
    print("Each file is written only after ALL its anchors validate, so a")
    print(".f3bak restore should not be needed. Paste this output back as-is.")
    sys.exit(1)


def file_text(fp):
    with open(fp, encoding="utf-8") as f:
        return f.read()


def write_text(fp, s):
    # preserve the file's existing newline convention (CRLF on Windows)
    crlf = b"\r\n" in open(fp, "rb").read()[:4000]
    with open(fp, "w", encoding="utf-8",
              newline=("\r\n" if crlf else "\n")) as f:
        f.write(s)


def backup(fp):
    bak = fp + ".f3bak"
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


# ---------------------------------------------------------------------------
# PHASE 0 -- locate the package, report state
# ---------------------------------------------------------------------------

def locate_pkg():
    for cand in (os.path.join(ROOT, "dspm"),
                 os.path.join(ROOT, "src", "dspm")):
        if os.path.isfile(os.path.join(cand, "patch.py")):
            return cand
    die("could not locate the dspm package -- tried dspm/ and src/dspm/ "
        "under " + ROOT)


PKG = locate_pkg()
P = {
    "extractor": os.path.join(PKG, "extractor.py"),
    "memory":    os.path.join(PKG, "memory.py"),
    "engine":    os.path.join(PKG, "engine.py"),
    "config":    os.path.join(PKG, "config.py"),
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


hr("PHASE 0 -- PACKAGE LOCATION + VERSION STATE")
print("package dir : " + PKG)
for label, key in (("dspm/__init__.py", "init"),
                   ("pyproject.toml", "pyproject")):
    m = find_version(P[key])
    print("%-18s %s" % (label, m.group(1) if m else "<no version line>"))


def patch_file(path, edits, soft=False):
    """Apply a list of (old, new) anchored edits to one file. All anchors
    validate before anything is written. `soft=True` warns and skips a
    missing anchor instead of aborting (for cosmetic edits)."""
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
# PHASE 1 -- ensure the four regression tests exist
# ---------------------------------------------------------------------------

TESTS4 = '''
# ---------------------------------------------------------------------------
# v0.1.17 code-review findings F1-F4: dependency remap, unit aliases,
# "can't" polarity, bake recount. RED on the unfixed tree -- F3 turns
# them green. (Header added by F3_defect_fixes.py; the test bodies are
# the ones reviewed in the code-review round.)
# ---------------------------------------------------------------------------

def test_extractor_remaps_forward_dependency_references():
    from dspm.extractor import extract_turn
    content = json.dumps([
        {"patch_id": "child", "patch_type": "decision",
         "payload": "Deploy after approval", "dependencies": ["parent"]},
        {"patch_id": "parent", "patch_type": "constraint",
         "payload": "Approval is required", "dependencies": []},
    ])
    patches = extract_turn(KwargCapturingClient(content), 'test-model',
                           'Approval is required, then deploy.', 5)
    assert [p.patch_id for p in patches] == ["p5_0", "p5_1"]
    assert patches[0].dependencies == ["p5_1"]


def test_unit_aliases_distinguish_milliseconds_from_seconds():
    from dspm.memory import _units_in
    assert _units_in("webhook timeout 100 ms") == frozenset({"ms"})
    assert _units_in("webhook timeout 100 s") == frozenset({"second"})
    assert _units_in("timeout 30 s") == _units_in("timeout 30 seconds")


def test_cant_is_recognized_as_negative_polarity():
    from dspm.memory import _bound_class
    assert _bound_class("retention can't exceed 5 years") == "upper"
    assert _bound_class("retention at least 5 years") == "lower"


def test_baked_revision_recounts_token_cost():
    from dspm.patch import count_tokens
    mem = DSPMMemory(budget=400)
    mem._merge_patch(_mk_crit("c0", "webhook timeout 30 seconds", 0))
    mem._merge_patch(_mk_crit("c1", "webhook timeout 20 seconds", 1))
    rev = _mk_crit("c2", "webhook timeout updated to 10 seconds", 2)
    mem._merge_patch(rev)
    assert rev.token_cost == count_tokens(rev.to_prompt_str())
'''

hr("PHASE 1 -- REGRESSION TESTS PRESENT?")
tsrc = file_text(P["tests"])
present = [n for n in NEW4 if ("def " + n + "(") in tsrc]
if len(present) == 4:
    print("[skip: all four F1-F4 tests already in tests/test_memory.py]")
elif present:
    die("only some of the four F1-F4 tests are present (%s) -- paste the "
        "tail of tests/test_memory.py back" % ", ".join(present))
else:
    if "_mk_crit" not in tsrc:
        die("tests/test_memory.py lacks the _mk_crit helper -- F2's test "
            "block must be present first")
    backup(P["tests"])
    write_text(P["tests"], tsrc.rstrip("\n") + "\n\n\n" + TESTS4.strip("\n") + "\n")
    print("  tests appended: 4 new (F1-F4 regressions)")

# ---------------------------------------------------------------------------
# PHASE 2 -- RED baseline
# ---------------------------------------------------------------------------

hr("PHASE 2 -- RED BASELINE: the four regressions on the UNFIXED tree")
base = run_pytest(["-q"] + NODES, capture=True)
if base.returncode == 0:
    print("NOTE: all four already pass -- fixes already applied? The patch")
    print("phase below is idempotent and will simply skip.")
else:
    m = re.search(r"(\d+) failed", base.stdout)
    n = int(m.group(1)) if m else -1
    if not (1 <= n <= 4):
        die("unexpected baseline result (%r failed) -- paste the output "
            "above" % n)
    print("RED BASELINE OK: %d/4 fail on the unfixed tree -- they are real "
          "regression tests, not vacuous ones." % n)

# ---------------------------------------------------------------------------
# PHASE 3 -- patches
# ---------------------------------------------------------------------------

# ---- F1: extractor two-pass construction (REGION replace) -----------------
# The old loop copies the LLM's dependency strings verbatim while
# regenerating patch ids as p{turn}_{k} -- LLM ids can never equal the
# stored ids, so every live dependency dangles and T4/W_DEP are dead.
# Region-replace (not line-anchor) so the em-dash comment inside the old
# loop never has to match exactly.
F1_NEW = '''    parsed = parse_extraction(raw)

    # v0.1.17 (finding F1): two-pass construction. The prompt asks the LLM
    # for its own patch_ids, but stored ids are regenerated as p{turn}_{k}
    # -- copying the LLM's dependency strings verbatim left every live
    # dependency dangling (LLM ids can never equal the regenerated scheme),
    # so T4 causal pruning never pruned and W_DEP contributed a flat 0 to
    # every utility score. Pass 1 validates and fingerprint-dedups; pass 2a
    # builds the LLM-id -> stored-id map COMPLETE before pass 2b rewrites
    # the dependencies, so forward references resolve. Unknown refs stay
    # dangling (downstream already tolerates them); duplicate LLM ids map
    # to the last occurrence. The id sequence over kept patches is
    # unchanged, so persistence roundtrips and existing tests are safe.
    accepted: List[Dict[str, Any]] = []
    seen_fingerprints: set = set()
    for item in parsed:
        if not isinstance(item, dict):
            continue
        p_type = str(item.get('patch_type') or item.get('type') or '').lower().strip()
        if p_type not in PATCH_TYPES:
            continue
        payload = str(item.get('payload') or item.get('text') or '')
        if not payload.strip():
            continue
        probe = SemanticPatch(patch_id="probe", turn_index=turn_index,
                              patch_type=p_type, payload=payload,
                              dependencies=[])
        if probe.fingerprint in seen_fingerprints:
            continue
        seen_fingerprints.add(probe.fingerprint)
        accepted.append(item)

    id_map: Dict[str, str] = {}
    for k, item in enumerate(accepted):
        llm_id = str(item.get('patch_id') or '').strip()
        if llm_id:
            id_map[llm_id] = f"p{turn_index}_{k}"

    patches: List[SemanticPatch] = []
    for k, item in enumerate(accepted):
        # v0.1.10: accepts a list of dependency ids or a single string;
        # anything else is []. v0.1.17 (F1): ids are remapped through
        # id_map so same-turn references actually link.
        raw_deps = item.get('dependencies')
        if isinstance(raw_deps, list):
            deps = [id_map.get(str(x), str(x)) for x in raw_deps]
        elif isinstance(raw_deps, str):
            deps = [id_map.get(raw_deps, raw_deps)]
        else:
            deps = []
        patches.append(SemanticPatch(
            patch_id=f"p{turn_index}_{k}",
            turn_index=turn_index,
            patch_type=str(item.get('patch_type') or item.get('type') or '').lower().strip(),
            payload=str(item.get('payload') or item.get('text') or ''),
            dependencies=deps,
        ))

    return patches'''


def patch_f1_extractor(path):
    src = file_text(path)
    if "id_map" in src:
        print("  [skip: F1 already applied] extractor.py")
        return
    START = "    parsed = parse_extraction(raw)\n"
    END = "\n    return patches"
    i = src.find(START)
    if i == -1:
        die("F1: start marker 'parsed = parse_extraction(raw)' missing in "
            "extractor.py")
    j = src.find(END, i)
    if j == -1:
        die("F1: end marker 'return patches' missing in extractor.py")
    region = src[i:j + len(END)]
    for landmark in ("seen_fingerprints",
                     'patch_id = f"p{turn_index}_{len(patches)}"',
                     "dependencies=deps"):
        if landmark not in region:
            die("F1: expected landmark %r not inside the replaced region -- "
                "extractor.py differs from the reviewed tree. Paste the "
                "construction loop back." % landmark)
    backup(path)
    write_text(path, src[:i] + F1_NEW + src[j + len(END):])
    py_compile.compile(path, doraise=True)
    print("  patched + compiled: extractor.py (F1 region)")


hr("PHASE 3a -- F1: extractor dependency remap (CORE)")
patch_f1_extractor(P["extractor"])

patch_file(P["extractor"], [(
    'BOTH smoke tests green before release.\n"""',
    'BOTH smoke tests green before release.\n'
    '\n'
    'v0.1.17: role threading (review point 2) -- the speaker label rides\n'
    'the turn line, telling binding user directives from assistant\n'
    'restatements. Finding F1: LLM-declared dependency ids are remapped\n'
    'to the stored p{turn}_{k} ids via a two-pass construction --\n'
    'previously deps were copied verbatim and always dangled, leaving T4\n'
    'and W_DEP dead. Extraction-layer change: both smoke gates must be\n'
    'green before release.\n'
    '"""',
)])

# ---- F2 + F3 + F4: memory.py ------------------------------------------------

hr("PHASE 3b -- F2/F3/F4: memory.py (CORE)")
EDITS_MEMORY = [
    # F2a: explicit unit-alias table replaces the rstrip("s") set
    (
        '_TIME_SIZE_UNITS = {\n'
        '    "ms", "millisecond", "second", "sec", "minute", "min",\n'
        '    "hour", "hr", "day", "week", "month", "year",\n'
        '    "kb", "mb", "gb", "tb", "%",\n'
        '}',
        '# v0.1.17 (finding F2): explicit alias map. The old rstrip("s")\n'
        '# membership check mapped "ms" -> "m" and "s" -> "" (both MISSED\n'
        '# the set), so "100 ms" vs "100 s" collapsed as a restatement\n'
        '# (a millisecond/second revision silently merged) and "30 s" vs\n'
        '# "30 seconds" survived as duplicate criticals. Spelled-out forms\n'
        '# canonicalize exactly as before, so collapse behavior on\n'
        '# existing transcripts is unchanged.\n'
        '_UNIT_ALIASES = {\n'
        '    "ms": "ms", "millisecond": "ms", "milliseconds": "ms",\n'
        '    "s": "second", "sec": "second", "secs": "second",\n'
        '    "second": "second", "seconds": "second",\n'
        '    "min": "minute", "mins": "minute", "minute": "minute",\n'
        '    "minutes": "minute",\n'
        '    "hr": "hour", "hrs": "hour", "hour": "hour", "hours": "hour",\n'
        '    "d": "day", "day": "day", "days": "day",\n'
        '    "w": "week", "week": "week", "weeks": "week",\n'
        '    "mo": "month", "month": "month", "months": "month",\n'
        '    "y": "year", "yr": "year", "year": "year", "years": "year",\n'
        '    "kb": "kb", "mb": "mb", "gb": "gb", "tb": "tb", "%": "%",\n'
        '}',
    ),
    # F2b: _units_in body reads the alias map
    (
        '    units = set()\n'
        '    for w in text.lower().split():\n'
        '        clean = w.strip(".,;)")\n'
        '        if clean.rstrip("s") in _TIME_SIZE_UNITS:\n'
        '            units.add(clean.rstrip("s"))\n'
        '    return frozenset(units)',
        '    units = set()\n'
        '    for w in text.lower().split():\n'
        '        canon = _UNIT_ALIASES.get(w.strip(".,;)"))\n'
        '        if canon:\n'
        '            units.add(canon)\n'
        '    return frozenset(units)',
    ),
    # F3: "can't" joins the negation vocabulary (engine list already has it)
    (
        '_POLARITY_NEG = {\n'
        '    "not", "no", "never", "without", "cannot", "mustn\'t",\n'
        '    "don\'t", "doesn\'t", "won\'t", "cant", "mustnt",\n'
        '}',
        '_POLARITY_NEG = {\n'
        '    "not", "no", "never", "without", "cannot", "can\'t", "mustn\'t",\n'
        '    "don\'t", "doesn\'t", "won\'t", "cant", "mustnt",\n'
        '}',
    ),
    # F4: the bake mutates the payload -- it must recount, like every
    # other mutation site (T3, both trim loops) does
    (
        '                chain = _dedup_chain(replaced_vals)[-3:]\n'
        '                patch.payload = f"{patch.payload} (was {\' then \'.join(chain)})"',
        '                chain = _dedup_chain(replaced_vals)[-3:]\n'
        '                patch.payload = f"{patch.payload} (was {\' then \'.join(chain)})"\n'
        '                # v0.1.17 (finding F4): token_cost must reflect the baked\n'
        '                # payload -- every other mutation site (T3 delta encoding,\n'
        '                # both trim loops) recounts; this was the one exception,\n'
        '                # so T5\'s cost penalty undershot baked criticals.\n'
        '                patch.recount()',
    ),
    # v0.1.17 module-docstring paragraph (memory.py)
    (
        'prefix-cache-friendly output (see experiments/bench_kv_cache.py).\n"""',
        'prefix-cache-friendly output (see experiments/bench_kv_cache.py).\n'
        '\n'
        'v0.1.17: four reviewer points plus four code-review findings, all\n'
        'riding this unpublished release. Review points: stats gains\n'
        'context_built (CRR is measured after get_context, not assumed)\n'
        'and selected (delivered patch count); add_turn threads the speaker\n'
        'role into the extraction prompt; the (was X) bake keeps the three\n'
        'most recently replaced values (was [:2]). Findings: F2 -- _units_in\n'
        'uses an explicit unit-alias map (the old rstrip("s") check mapped\n'
        '"ms" to "m" and "s" to "", so ms/s revisions collapsed and\n'
        's/seconds duplicated); F3 -- "can\'t" joins the negation vocabulary\n'
        'so bound classes keep upper and lower bounds from merging; F4 --\n'
        'the bake recounts token_cost. The dependency-id remap (F1) lives in\n'
        'extractor.py.\n'
        '"""',
    ),
]
patch_file(P["memory"], EDITS_MEMORY)

# ---- F5/F6: docstring honesty (engine.py + config.py, cosmetic) -------------

hr("PHASE 3c -- F5/F6: docstring honesty (cosmetic, warn-only)")
EDITS_ENGINE = [
    (
        '"""T7: allocate per-type budgets using the EMA-like query type signal."""',
        '"""T7: RESERVED pass-through. The adaptive per-type allocation\n'
        '        described in the package design is NOT wired: ema_query is\n'
        '        only ever reset (reset_ema), never updated, and ALPHA_EMA,\n'
        '        SHADOW_THRESHOLD, DELTA_MIN_SAVING, BASE_BUDGET_FRACTIONS,\n'
        '        CRITICAL_MAX_WORDS and REVISION_OVERLAP are unwired config.\n'
        '        Do not quote T7 as active behavior."""',
    ),
    (
        '"""T2: fuse duplicate slot keys for non-critical patches by highest utility and turn index."""',
        '"""T2: fuse duplicate slot keys for non-critical patches, keeping\n'
        '        the highest (utility, turn_index) -- in practice the newest\n'
        '        turn, because utility is still 0.0 for every patch at this\n'
        '        stage (T5 scores later in the pipeline)."""',
    ),
]
patch_file(P["engine"], EDITS_ENGINE, soft=True)

patch_file(P["config"], [(
    'ALPHA_EMA = 0.5',
    '# Reserved for the T7 adaptive-budgeting stage -- NOT wired yet\n'
    '# (engine._adaptive_budgeting is a pass-through; ema_query is never\n'
    '# updated). Do not quote these as active behavior.\n'
    'ALPHA_EMA = 0.5',
)], soft=True)

# ---- version reconciliation -------------------------------------------------

hr("PHASE 3d -- version reconciliation (cosmetic, warn-only)")
patch_file(P["init"], [(
    '__version__ = "0.1.16"',
    '__version__ = "0.1.17"',
)], soft=True)
patch_file(P["pyproject"], [(
    'version = "0.1.16"',
    'version = "0.1.17"',
)], soft=True)

init_ver = find_version(P["init"])
proj_ver = find_version(P["pyproject"])
print("versions now: __init__.py=%s  pyproject.toml=%s"
      % (init_ver.group(1) if init_ver else "?",
         proj_ver.group(1) if proj_ver else "?"))

# ---------------------------------------------------------------------------
# PHASE 4 -- GREEN re-check
# ---------------------------------------------------------------------------

hr("PHASE 4 -- GREEN RE-CHECK: the four regressions on the FIXED tree")
green = run_pytest(["-q"] + NODES, capture=True)
if green.returncode != 0:
    die("GREEN check failed -- the fixes did not land correctly. Paste "
        "everything back; do NOT re-run F3 yet.")
print("GREEN: 4/4 pass after patching.")

# ---------------------------------------------------------------------------
# PHASE 5 -- full gate
# ---------------------------------------------------------------------------

hr("PHASE 5 -- FULL GATE: pytest tests/ -q   (71 expected = 67 + 4)")
coll = run_pytest(["tests/", "-q", "--collect-only"], capture=True)
m = re.search(r"(\d+) tests? collected", coll.stdout)
total = int(m.group(1)) if m else -1
print("collected: %d (expected 71; a different count is fine if you "
      "added/removed tests yourself)" % total)

gate = run_pytest(["tests/", "-q"], capture=False)
if gate.returncode != 0:
    print()
    print("F3: FULL GATE FAILED -- the patches ARE on disk. Do NOT re-run F3")
    print("(it is idempotent, but I need the failure text first). Paste")
    print("everything back as-is.")
    sys.exit(2)

# ---------------------------------------------------------------------------
# PHASE 6 -- verify slices + live probes
# ---------------------------------------------------------------------------

VERIFY = '''
import inspect
import json
import dspm
from dspm.extractor import extract_turn
from dspm.memory import DSPMMemory, _units_in, _bound_class
from dspm.patch import SemanticPatch, count_tokens


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
show("F1 -- two-pass construction (id_map)", inspect.getsource(extract_turn),
     "id_map", 5)
show("F2 -- alias-map _units_in", inspect.getsource(_units_in),
     "def _units_in", 8)
print("F2 probe:", _units_in("webhook timeout 100 ms"),
      _units_in("webhook timeout 100 s"),
      _units_in("timeout 30 s") == _units_in("timeout 30 seconds"))
print("F3 probe:", _bound_class("retention can't exceed 5 years"),
      _bound_class("retention at least 5 years"))
show("F4 -- bake recounts", inspect.getsource(DSPMMemory._merge_patch),
     "patch.recount()", 4)


class MockResp:
    def __init__(self, content):
        msg = type("Msg", (), {"content": content})()
        self.choices = [type("Ch", (), {"message": msg})()]


CONTENT = json.dumps([
    {"patch_id": "child", "patch_type": "decision",
     "payload": "Deploy after approval", "dependencies": ["parent"]},
    {"patch_id": "parent", "patch_type": "constraint",
     "payload": "Approval is required", "dependencies": []},
])


class Client:
    def __init__(self):
        class C:
            @staticmethod
            def create(**kwargs):
                return MockResp(CONTENT)
        class Ch:
            completions = C()
        self.chat = Ch()


ps = extract_turn(Client(), "test-model", "Approval is required, then deploy.", 5)
print("F1 probe ids:", [p.patch_id for p in ps], "deps:", ps[0].dependencies)


def mk(pid, payload, turn):
    return SemanticPatch(patch_id=pid, turn_index=turn,
                         patch_type="constraint", payload=payload,
                         dependencies=[])


mem = DSPMMemory(budget=400)
mem._merge_patch(mk("c0", "webhook timeout 30 seconds", 0))
mem._merge_patch(mk("c1", "webhook timeout 20 seconds", 1))
rev = mk("c2", "webhook timeout updated to 10 seconds", 2)
mem._merge_patch(rev)
print("F4 probe payload:", rev.payload)
print("F4 probe recount:", rev.token_cost == count_tokens(rev.to_prompt_str()))
'''

hr("PHASE 6 -- VERIFY SLICES + LIVE PROBES")
subprocess.run([sys.executable, "-c", VERIFY], cwd=ROOT)

hr("F3 COMPLETE")
print("All four code-review defects fixed and proven: RED baseline -> "
      "patch -> 4/4 green -> full gate.")
print("Backups: *.f3bak next to each patched file (delete them once you "
      "are satisfied).")
print("REMAINING BEFORE RELEASE (see the review notes):")
print("  1. BOTH smoke gates (extraction layer changed -- v0.1.13 rule):")
print("       & $py scripts/smoke_conversation.py")
print("       & $py scripts/<document smoke>   (whichever your tree uses)")
print("  2. README/changelog: add the v0.1.17 entry; scrub any claim of")
print("     T7 'adaptive budgeting' or T2 'by highest utility' as ACTIVE.")
print("  3. Manual hygiene: delete the DUPLICATED test blocks in")
print("     tests/test_memory.py (test_dense_prompt_classifies_config_")
print("     correctly + test_add_document_chunks_on_sentences are each")
print("     defined twice -- the second copies are dead weight); dedupe")
print("     .gitignore; add *.f2bak, *.f3bak and the scratch scripts.")
