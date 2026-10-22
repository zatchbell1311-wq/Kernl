# F2_review_fixes.py -- v0.1.17 review-fix batch (review points 1-5).
# Save in the project root (next to _diag_review.py) and run:
#     & $py F2_review_fixes.py
# Expect ~4 minutes total: probe -> patches -> tests appended -> verify
# slices -> full pytest gate (67 expected). Every edit anchors on exact
# source text; a missing anchor aborts BEFORE that file is written.
# Re-runs are safe (idempotent). Originals backed up once to <file>.f2bak.

import os
import py_compile
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def die(msg):
    print()
    print("F2 ABORT -- " + msg)
    print("Each file is written only after ALL its anchors validate, so a")
    print(".f2bak restore should not be needed. Paste this output back as-is.")
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
    bak = fp + ".f2bak"
    if not os.path.exists(bak):
        shutil.copyfile(fp, bak)


def patch_file(rel, edits):
    fp = os.path.join(ROOT, *rel.split("/"))
    src = file_text(fp)
    plan = []
    for old, new in edits:
        if new in src:
            print("  [skip: already applied] " + rel)
            continue
        if old not in src:
            die("anchor missing in %s:\n%s" % (rel, old[:300]))
        plan.append((old, new))
    if not plan:
        return
    for old, new in plan:
        src = src.replace(old, new, 1)
    backup(fp)
    write_text(fp, src)
    py_compile.compile(fp, doraise=True)
    print("  patched + compiled: " + rel)


# ---------------------------------------------------------------------------
# FIX-1 (review point 2): role threading. add_turn(role, text) accepted the
# role and dropped it; extract_turn never saw who spoke. The role now rides
# the turn line as "[role]". Trailing keyword arg with default 'user' keeps
# every positional caller (smoke scripts included) working. Classification
# rules are untouched -- information only, zero semantic drift.
# ---------------------------------------------------------------------------

EDITS_EXTRACTOR = [
    (
        "def extract_turn(llm_client, model: str, turn_text: str, turn_index: int,\n"
        "                 recent_context: str = '', dense: bool = False,\n"
        "                 max_tokens: Optional[int] = None,\n"
        "                 reasoning_effort: Optional[str] = None) -> List[SemanticPatch]:",
        "def extract_turn(llm_client, model: str, turn_text: str, turn_index: int,\n"
        "                 recent_context: str = '', dense: bool = False,\n"
        "                 max_tokens: Optional[int] = None,\n"
        "                 reasoning_effort: Optional[str] = None,\n"
        "                 role: str = 'user') -> List[SemanticPatch]:",
    ),
    (
        r'''    user_content = (
        f"Recent context:\n{recent_context}\n\nTurn {turn_index}:\n{turn_text}"
        if recent_context else f"Turn {turn_index}:\n{turn_text}"
    )''',
        r'''    # v0.1.17 (review point 2): the speaker role rides on the turn line so
    # the extractor can tell a binding user directive ("always return JSON")
    # from an assistant restatement. add_turn(role, ...) used to accept the
    # role and then drop it on the floor. Information-only: classification
    # rules are unchanged, so behavior on existing transcripts is identical
    # apart from the added label.
    user_content = (
        f"Recent context:\n{recent_context}\n\nTurn {turn_index}:\n[{role}] {turn_text}"
        if recent_context else f"Turn {turn_index}:\n[{role}] {turn_text}"
    )''',
    ),
    (
        'supports the parameter."""',
        'supports the parameter.\n'
        '      role -- v0.1.17: speaker label for the turn\n'
        "      ('user'/'assistant'/...), threaded into the prompt so the\n"
        '      extractor knows who spoke; add_turn() previously accepted\n'
        '      the role and silently dropped it."""',
    ),
]

EDITS_ADDTURN = [
    (
        "        dense = self.dense or numeric_density(text) >= DENSE_AUTO_THRESHOLD\n"
        "        patches = extract_turn(self.llm_client, self.model, text, self.turns,",
        "        dense = self.dense or numeric_density(text) >= DENSE_AUTO_THRESHOLD\n"
        "        # v0.1.17 (review point 2): role threaded through -- it was\n"
        "        # accepted here and then dropped on the floor.\n"
        "        patches = extract_turn(self.llm_client, self.model, text, self.turns,\n"
        "                               role=role,",
    ),
]

# ---------------------------------------------------------------------------
# FIX-2/3 (review points 1, 3, 4): stats delivery flag + bake cap.
#   * stats gains "context_built" and "selected". CRR before the first
#     get_context() is an ASSUMPTION (critical_selected falls back to
#     critical_total) -- now it is visible. An empty selection (trr=100,
#     crr=0) is visible at a glance instead of hiding behind a perfect
#     reduction number. TRR formula deliberately unchanged: clamping a
#     negative reduction would be metric gaming; raw/context stay exposed.
#   * _merge_patch baked only the first TWO replaced values; one revision
#     superseding three stale criticals deleted the third value with no
#     trace. Now: dedup first, keep the three most recently added.
# ---------------------------------------------------------------------------

EDITS_STATS_AND_BAKE = [
    (
        '        """Return a summary of memory health, selected-critical retention, and token reduction rate."""',
        '        """Return a summary of memory health, selected-critical retention, and token reduction rate.\n'
        '\n'
        '        v0.1.17 (review points 1+4): `context_built` is the delivery flag\n'
        '        -- CRR is only MEASURED after the first get_context(); before\n'
        '        that it is assumed (critical_selected falls back to\n'
        '        critical_total). `selected` is the delivered patch count\n'
        '        (store size pre-build), so an empty selection is visible\n'
        '        next to trr instead of hiding behind a perfect reduction\n'
        '        number. TRR keeps its raw formula: clamping a negative\n'
        '        reduction would be metric gaming, and raw_tokens /\n'
        '        context_tokens stay exposed for audit.\n'
        '        """',
    ),
    (
        '        return {\n            "turns": self.turns,',
        '        # v0.1.17 (review points 1+4): selected mirrors the\n'
        '        # critical_selected fallback -- pre-build it reports the\n'
        '        # store size, post-build the delivered size. context_built\n'
        '        # lets anyone quoting "100% CRR" show the number was\n'
        '        # measured on a delivered context, not assumed.\n'
        '        selected_count = len(self.selected_patches) if self._context_built else len(self.patches)\n'
        '        return {\n            "turns": self.turns,',
    ),
    (
        '            "crr": crr,\n            "budget": self.budget,',
        '            "crr": crr,\n'
        '            "context_built": self._context_built,\n'
        '            "selected": selected_count,\n'
        '            "budget": self.budget,',
    ),
    (
        '                chain = _dedup_chain(replaced_vals[:2])',
        '                # v0.1.17 (review point 3): the [:2] cap silently deleted the\n'
        '                # third-oldest value when ONE revision superseded three\n'
        '                # or more stale criticals (repro: webhook-timeout\n'
        '                # entries at 30/20/15 seconds, then "webhook timeout\n'
        '                # updated to 10 seconds" -- the 15 vanished without a\n'
        '                # trace). Keep the three most recently added replaced\n'
        '                # values; deeper history is dropped, and the payload\n'
        '                # cap plus the was-span floors in the engine bound the\n'
        '                # result.\n'
        '                chain = _dedup_chain(replaced_vals)[-3:]',
    ),
]

# ---------------------------------------------------------------------------
# FIX-4 (review point 5): live benchmark latency/token reporting.
# ---------------------------------------------------------------------------

CLASS_TEXT = '''class UsageRecordingClient:
    """v0.1.17 (review point 5): pass-through proxy recording per-call
    latency and token usage, so live mode can REPORT extractor cost next
    to accuracy instead of asserting it. Wraps any OpenAI-compatible
    client; a missing .usage attribute counts as zero."""

    def __init__(self, inner):
        self._inner = inner
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.latencies = []
        owner = self

        class Completions:
            @staticmethod
            def create(**kwargs):
                from time import perf_counter
                t0 = perf_counter()
                resp = owner._inner.chat.completions.create(**kwargs)
                owner.latencies.append(perf_counter() - t0)
                owner.calls += 1
                usage = getattr(resp, "usage", None)
                if usage is not None:
                    owner.input_tokens += getattr(usage, "prompt_tokens", 0) or 0
                    owner.output_tokens += getattr(usage, "completion_tokens", 0) or 0
                return resp

        class Chat:
            completions = Completions()

        self.chat = Chat()

    def __getattr__(self, name):
        if name == "_inner":
            raise AttributeError(name)
        return getattr(self._inner, name)

    def report(self, label):
        lat = sorted(self.latencies)
        p50 = lat[len(lat) // 2] if lat else 0.0
        total = sum(self.latencies)
        print("%-18s calls %3d   p50 %6.2fs   total %7.1fs   tokens in %7d / out %7d"
              % (label, self.calls, p50, total,
                 self.input_tokens, self.output_tokens))'''

EDITS_BENCH = [
    (
        "def live(args):\n    from dspm import DSPMMemory",
        CLASS_TEXT + "\n\n\ndef live(args):\n    from dspm import DSPMMemory",
    ),
    (
        '    if not args.model:\n'
        '        sys.exit("live mode needs --model (your extractor/judge model name)")\n'
        '    client = _client(args)',
        '    if not args.model:\n'
        '        sys.exit("live mode needs --model (your extractor/judge model name)")\n'
        '    client = _client(args)\n'
        '    # v0.1.17 (review point 5): three probes -- extractor (DSPM\n'
        '    # overhead), rolling summary (baseline overhead), judge (shared\n'
        '    # evaluation overhead).\n'
        '    extract_probe = UsageRecordingClient(client)\n'
        '    summarize_probe = UsageRecordingClient(client)\n'
        '    judge_probe = UsageRecordingClient(client)',
    ),
    (
        "        mem = DSPMMemory(budget=args.budget, llm_client=client, model=args.model)",
        "        mem = DSPMMemory(budget=args.budget, llm_client=extract_probe, model=args.model)",
    ),
    (
        "            rolling = llm_summarize(client, args.model, rolling, role, text)",
        "            rolling = llm_summarize(summarize_probe, args.model, rolling, role, text)",
    ),
    (
        '        s_d = llm_judge(client, args.model, ctx, d["ground_truth"])',
        '        s_d = llm_judge(judge_probe, args.model, ctx, d["ground_truth"])',
    ),
    (
        '        s_r = llm_judge(client, args.model, rolling, d["ground_truth"])',
        '        s_r = llm_judge(judge_probe, args.model, rolling, d["ground_truth"])',
    ),
    (
        '    print(f"mean               DSPM {fmt(d_scores)}{extra}   "\n'
        '          f"rolling-summary {fmt(r_scores)}{extra_r}   "\n'
        '          f"({len(d_scores)} dialogues, budget {args.budget})")',
        '    print(f"mean               DSPM {fmt(d_scores)}{extra}   "\n'
        '          f"rolling-summary {fmt(r_scores)}{extra_r}   "\n'
        '          f"({len(d_scores)} dialogues, budget {args.budget})")\n'
        '    # v0.1.17 (review point 5): latency/token accounting for the\n'
        '    # efficiency table -- extractor vs rolling-summary overhead,\n'
        '    # judge shared by both arms.\n'
        '    extract_probe.report("extractor")\n'
        '    summarize_probe.report("rolling-summary")\n'
        '    judge_probe.report("judge")',
    ),
]

EDITS_VERSION = [
    ('version = "0.1.16"', 'version = "0.1.17"'),
]

# ---------------------------------------------------------------------------
# New regression tests (appended to tests/test_memory.py).
# ---------------------------------------------------------------------------

TESTS_TEXT = '''


# ---------------------------------------------------------------------------
# v0.1.17 review-fix regressions (F2): role threading, stats delivery flag,
# and the bake cap for supersession chains deeper than two.
# ---------------------------------------------------------------------------

def _mk_crit(pid, payload, turn):
    """Manual critical patch mirroring the MockLLMClient JSON field set."""
    return SemanticPatch(
        patch_id=pid, turn_index=turn, patch_type="constraint",
        payload=payload, dependencies=[], utility=0.9,
        fingerprint="", slot_key="", is_delta=False, delta_base="",
        causal_depth=0)


def test_role_threaded_into_extractor_prompt():
    llm = PromptCapturingClient()
    mem = DSPMMemory(llm_client=llm, model='test-model', budget=250)
    mem.add_turn('assistant', 'I will always return JSON output.')
    assert '[assistant]' in llm.user_messages[0], llm.user_messages[0]


def test_role_defaults_to_user_when_extracting_directly():
    from dspm.extractor import extract_turn
    llm = KwargCapturingClient()
    extract_turn(llm, 'test-model', 'Always return JSON.', 0)
    assert '[user]' in llm.calls[0]["messages"][1]["content"]


def test_stats_exposes_delivery_flag_and_selected_count():
    llm = MockLLMClient()
    mem = DSPMMemory(llm_client=llm, model='test-model', budget=250)
    mem.add_turn('user', 'Return JSON and use POST /v1/items for creation.')
    s = mem.stats
    assert s["context_built"] is False
    assert s["selected"] == s["total_patches"]      # pre-build: store size
    mem.get_context(query="overview")
    s2 = mem.stats
    assert s2["context_built"] is True
    assert s2["selected"] <= s2["total_patches"]    # post-build: delivered


def test_single_revision_bakes_three_superseded_values():
    mem = DSPMMemory(budget=400)
    mem._merge_patch(_mk_crit("c0", "webhook timeout 30 seconds", 0))
    mem._merge_patch(_mk_crit("c1", "webhook timeout 20 seconds", 1))
    mem._merge_patch(_mk_crit("c2", "webhook timeout 15 seconds", 2))
    rev = _mk_crit("c3", "webhook timeout updated to 10 seconds", 3)
    mem._merge_patch(rev)
    assert "(was" in rev.payload, rev.payload
    for old_val in ("30", "20", "15"):
        assert old_val in rev.payload, rev.payload   # v0.1.16 lost the 15
    assert len([p for p in mem.all_patches if p.is_critical]) == 1
'''

# ---------------------------------------------------------------------------
# Verification slices (run in a fresh interpreter so imports are patched).
# ---------------------------------------------------------------------------

VERIFY = '''
import inspect
from dspm.extractor import extract_turn
from dspm import DSPMMemory


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


show("VERIFY 1 -- extract_turn signature (role appended)",
     inspect.getsource(extract_turn), "def extract_turn", 5)
show("VERIFY 2 -- user_content (role on the turn line)",
     inspect.getsource(extract_turn), "user_content = (", 9)
show("VERIFY 3 -- add_turn threads role",
     inspect.getsource(DSPMMemory.add_turn), "patches = extract_turn", 5)
show("VERIFY 4 -- stats selected_count",
     inspect.getsource(DSPMMemory.stats.fget), "selected_count", 3)
show("VERIFY 5 -- stats dict (context_built, selected)",
     inspect.getsource(DSPMMemory.stats.fget), '"context_built"', 3)
show("VERIFY 6 -- bake cap",
     inspect.getsource(DSPMMemory._merge_patch), "chain = _dedup_chain", 2)
print("=" * 70)
print("VERIFY 7 -- add_document (paste this: decides whether document")
print("chunks need a 'document' role label in a follow-up)")
print("=" * 70)
print(inspect.getsource(DSPMMemory.add_document))
'''

# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

print("=" * 70)
print("F2 REVIEW-FIX BATCH  (v0.1.16 -> v0.1.17)")
print("=" * 70)

# -- 0. probe the manual-patch construction the new tests rely on --------
sys.path.insert(0, ROOT)
try:
    from dspm.patch import SemanticPatch
    _p = SemanticPatch(
        patch_id="probe", turn_index=0, patch_type="constraint",
        payload="probe payload 30 seconds", dependencies=[], utility=0.9,
        fingerprint="", slot_key="", is_delta=False, delta_base="",
        causal_depth=0)
    print("probe: SemanticPatch OK  fingerprint=%r  is_critical=%r"
          % (getattr(_p, "fingerprint", "<none>"),
             getattr(_p, "is_critical", "<none>")))
except TypeError as exc:
    die("SemanticPatch rejected the helper kwargs -- paste this back:\n%r"
        % (exc,))

print("\n-- FIX-1 (review point 2): role threading ---------------------------")
patch_file("dspm/extractor.py", EDITS_EXTRACTOR)
patch_file("dspm/memory.py", EDITS_ADDTURN)

print("\n-- FIX-2/3 (review points 1, 3, 4): stats flag + bake cap -------------")
patch_file("dspm/memory.py", EDITS_STATS_AND_BAKE)

print("\n-- FIX-4 (review point 5): live benchmark probes -----------------------")
patch_file("benchmark_compare.py", EDITS_BENCH)
patch_file("pyproject.toml", EDITS_VERSION)

print("\n-- regression tests ----------------------------------------------------")
tfp = os.path.join(ROOT, "tests", "test_memory.py")
tsrc = file_text(tfp)
if "test_single_revision_bakes_three_superseded_values" in tsrc:
    print("  [skip: tests already appended]")
else:
    if not tsrc.endswith("\n"):
        tsrc += "\n"
    backup(tfp)
    write_text(tfp, tsrc + TESTS_TEXT)
    print("  tests appended: tests/test_memory.py (4 new)")

print("\n-- verification (fresh import of the patched tree) --------------------")
subprocess.run([sys.executable, "-c", VERIFY], cwd=ROOT)

print("\n" + "=" * 70)
print("FULL GATE -- pytest tests/ -q   (63 old + 4 new = 67 expected; ~3 min)")
print("=" * 70)
gate = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"],
                      cwd=ROOT)
if gate.returncode != 0:
    print("\nF2: gate FAILED -- do NOT re-run F2 yet (it is idempotent, but I")
    print("need the failure text first). Paste everything back as-is.")
    sys.exit(2)

print("\nF2 COMPLETE -- v0.1.17 applied, 67/67 green.")
print("Paste everything from 'F2 REVIEW-FIX BATCH' down.")