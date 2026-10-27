# F7a_docstring_anchor_repair.py -- one-time repair for the F7 abort.
#
# F7 validates every anchor before writing anything; it stopped safely
# because memory.py's module docstring does not end with the note F6 was
# supposed to leave there. This script writes ONLY that docstring
# paragraph, in the exact canonical wording, after checking all six F7
# anchors (read from your own F7_live_echo_collapse.py -- zero drift).
# No code is touched. Backup: memory.py.f7abak. Idempotent. When unsure,
# it changes NOTHING and prints evidence to paste back instead.
#
# Save in the project root and run:
#     & $py F7a_docstring_anchor_repair.py

import ast
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def die(msg):
    print()
    print("F7a ABORT -- " + msg)
    sys.exit(1)


def hr(t):
    print()
    print("=" * 70)
    print(t)
    print("=" * 70)


pkg = None
for cand in (os.path.join(ROOT, "dspm"), os.path.join(ROOT, "src", "dspm")):
    if os.path.isfile(os.path.join(cand, "memory.py")):
        pkg = cand
        break
if not pkg:
    die("could not find memory.py under " + ROOT)
MEM = os.path.join(pkg, "memory.py")
F7 = os.path.join(ROOT, "F7_live_echo_collapse.py")
if not os.path.isfile(F7):
    die("could not find F7_live_echo_collapse.py in the project root")

# the canonical F6 docstring paragraph -- exactly the text F6 writes
P6_LINES = [
    "20-run live gate (F6): the loop caught two live-extraction",
    "shapes the offline suites could not. Mode 1 -- the same",
    "dashboard latency fact arrived as a word-order reversal",
    "('Dashboard sub-2-second' [CON] vs 'sub-2-second dashboard'",
    "[DEC]); its distinctive shared token carries a digit, so the",
    "(b3) non-numeric word count never reached two and the store",
    "bloated to 8 against the 7-critical delivery cap (CRR 87% on",
    "3/20 runs). Rule (b5) bridges order-insensitive echoes under",
    "every (b3)/(b4) safety. Mode 2 -- comma-form revision history",
    "lost the replaced value at the 2-word floor; the engine-side",
    "F6 half fixes the floor and the minimal story.",
]
ANCHOR0 = P6_LINES[-1] + '\n"""'
NAMES = ["docstring tail", "(b5) comment block", "_head_direction tail",
         "(b2) call site", "(b5) call site", "_jaccard def"]

try:
    f7tree = ast.parse(open(F7, encoding="utf-8").read())
    anchors = []
    for node in f7tree.body:
        if (isinstance(node, ast.Assign)
                and any(getattr(t, "id", "") == "EDITS_MEMORY"
                        for t in node.targets)):
            for elt in node.value.elts:
                if isinstance(elt, ast.Tuple):
                    anchors.append(ast.literal_eval(elt.elts[0]))
    if not anchors:
        die("found EDITS_MEMORY in F7 but could not read its anchors")
except Exception as e:
    die("could not parse F7_live_echo_collapse.py (%s)" % e)

norm = open(MEM, encoding="utf-8").read()
raw = open(MEM, encoding="utf-8", newline="").read()
lines = raw.splitlines(keepends=True)
NL = "\r\n" if (lines and lines[0].endswith("\r\n")) else "\n"


def report(tag):
    print(tag)
    ok = True
    for i, a in enumerate(anchors):
        hit = a in norm
        ok = ok and hit
        label = NAMES[i] if i < len(NAMES) else "?"
        if hit:
            print("  [FOUND  ] %d  %s" % (i, label))
        else:
            print("  [MISSING] %d  %s  anchor starts: %r"
                  % (i, label, a[:90]))
    return ok


hr("F7 ANCHOR CHECK -- BEFORE")
all_ok = report("all six F7 anchors F7 will look for:")

if all_ok:
    print()
    print("Nothing to repair -- every F7 anchor already validates.")
    print("Just re-run:  & $py F7_live_echo_collapse.py")
    sys.exit(0)

if "20-run live gate, round 2 (F7)" in norm:
    print()
    print("F7's own docstring paragraph is already present -- your F7")
    print("run already wrote memory.py once. Do NOT repair; just re-run")
    print("F7 (it will skip the applied edits and continue):")
    print("    & $py F7_live_echo_collapse.py")
    sys.exit(0)

if anchors[0] in norm:
    die("the docstring anchor is fine but ANOTHER anchor is missing -- "
        "paste this whole output back; do not run F7 yet")

try:
    tree = ast.parse(raw)
except SyntaxError as e:
    die("memory.py does not parse (%s)" % e)
doc = None
if tree.body and isinstance(tree.body[0], ast.Expr):
    v = getattr(tree.body[0].value, "value", None)
    if isinstance(v, str):
        doc = tree.body[0]
if doc is None:
    die("no module docstring found in memory.py -- paste this back")
S, E = doc.lineno, doc.end_lineno
if E <= S:
    die("module docstring is a single line -- paste this back")
close_txt = lines[E - 1].strip()
if close_txt not in ('"""', "'''"):
    die("the docstring closing line is not a plain delimiter (%r) -- "
        "paste this back" % close_txt)

hr("YOUR CURRENT DOCSTRING TAIL (last 8 lines, exact)")
for L in lines[max(0, E - 9):E]:
    print(repr(L))

start = None
for i in range(S - 1, E - 1):
    if "20-run live gate" in lines[i]:
        start = i
        break
cut_from = start if start is not None else E - 1
para = [L + NL for L in P6_LINES]
rebuilt = lines[:cut_from]
if rebuilt and rebuilt[-1].strip():
    rebuilt.append(NL)
rebuilt += para
rebuilt += lines[E - 1:]
raw_new = "".join(rebuilt)

bak = MEM + ".f7abak"
if not os.path.exists(bak):
    shutil.copyfile(MEM, bak)
with open(MEM, "w", encoding="utf-8", newline="") as f:
    f.write(raw_new)
print()
print("  [docstring paragraph restored -- backup at %s]" % bak)

norm = open(MEM, encoding="utf-8").read()
hr("F7 ANCHOR CHECK -- AFTER")
ok = report("all six F7 anchors F7 will look for:")

hr("YOUR NEW DOCSTRING TAIL (last 13 lines)")
lines2 = open(MEM, encoding="utf-8", newline="").read().splitlines(True)
for L in lines2[-13:]:
    print(L.rstrip("\r\n"))

print()
if ok:
    print("All six anchors now validate. Re-run F7:")
    print("    & $py F7_live_echo_collapse.py")
    print("Expected from there: GREEN 14/14 -> full gate 116 passed ->")
    print("probes OK -> harness AGGREGATE: 5/5.")
else:
    print("Some anchors are STILL missing. Do NOT run F7. Paste this")
    print("whole output back and the next repair will be exact.")