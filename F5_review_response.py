# ==================== Part 1 of 2 (paste both parts into ONE file) ====================

# F5_review_response.py -- v0.1.17 finding S2 + the external review.
#
# SCOPE (which review points this batch answers):
#   R2  source_role rides on the patches: stored, persisted, counted in
#       stats, renderable behind render_source_roles (off by default --
#       it costs tokens). The F2 batch threaded the role into the
#       extractor PROMPT; the store still could not tell a user
#       requirement from an assistant echo.
#   R3  rule (b4) is the LAST word-overlap extension: a bound-word slot
#       head bridging a thin cross-type echo, with a canary test that
#       freezes the remaining limitation (same-value bounds sharing
#       only a scope noun still merge). The structured-slot rewrite
#       stays planned.
#   R4  stats separates conversation_tokens (input turns) from
#       patch_tokens (extraction output; raw_tokens stays as alias) and
#       context_tokens (delivery). Notebooks persist input tokens.
#   R1  (CRR on empty selection) was already fixed in your F2 batch;
#       F5 does not touch it.
#   R5  eval_harness.py is NOT in this file -- it ships only after you
#       confirm F5 works on your tree. It measures store + delivery
#       layers offline; --live measures extraction reliability,
#       latency, cost. No judge model, no accuracy claims.
#
# NOT COVERED (open _merge_patch issues, on the record):
#   (a) rule (a) can still sweep several overlapping slots in one
#       revision (history is baked; slot granularity is lost).
#   (b) rule (c) still has no polarity check.
#   F5 touches neither (a) nor (c); both are queued for the next batch.
#
# ROOT CAUSE THIS FIXES (finding S2): after F4, the 20-run live loop
# still failed CRR at 87% (= 7/8 delivered) on 6/20 runs. The
# deliberately frozen fee pair ('Maximum fee ... 0.5%' [CON] + 'fee cap
# ... 0.5%' [DEC]) kept the NOMINAL store at 6 against the smoke's
# 7-critical delivery cap, so any two extraction variants tipped a run
# to 8 and one fell out of the budget. (b4) merges that pair WITHOUT
# merging the tier-scoped facts the freeze was protecting.
#
# REAL-TREE SAFETY (your machine, not my sandbox):
#   - PHASE 0 checks EVERY anchor (core + cosmetic) against your files
#     and prints ALL missing anchors at once; NOTHING is written if any
#     core anchor is missing. Missing core anchors come with the exact
#     PowerShell Select-String command to print that region so you can
#     paste it back -- no guessing from my copy.
#   - Cosmetic edits (docstrings/comments) are SOFT: warn and skip.
#     Your tree skipped the F3 docstring paragraphs, so no core anchor
#     depends on them. Core anchors are functional code only: base
#     v0.1.16 code, F2-written lines, or exact bytes your green F4 run
#     wrote.
#   - Version is already 0.1.17 everywhere; this script NEVER bumps it.
#
# All expected numbers are PREDICTIONS from my sandbox tree, not
# results: 15 F5 test nodes, 9 red pre-patch, full gate 80 + 14 = 94,
# DIAG replay 7 -> 5 criticals at crr 100, +2 variance probe 7/7.
#
# Save in the project root and run:
#     & $py F5_review_response.py
#
# Flow: PHASE 0 locate+versions+FULL ANCHOR PREFLIGHT -> PHASE 1 tests
# (unfreeze fee pair, append the rest) -> PHASE 2 RED baseline ->
# PHASE 3 anchored patches -> PHASE 4 GREEN 15/15 -> PHASE 5 full gate
# -> PHASE 6 probes. Idempotent; originals backed up ONCE to <file>.f5bak.

import os
import py_compile
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def die(msg):
    print()
    print("F5 ABORT -- " + msg)
    print("Paste this output back as-is. Do not re-run until we have "
          "looked at it.")
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
    bak = fp + ".f5bak"
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
    for cand in (os.path.join(ROOT, "src", "dspm"),
                 os.path.join(ROOT, "dspm")):
        if os.path.isfile(os.path.join(cand, "patch.py")):
            return cand
    die("could not locate the dspm package -- tried src/dspm/ and dspm/ "
        "under " + ROOT)


PKG = locate_pkg()
P = {
    "memory":    os.path.join(PKG, "memory.py"),
    "patch":     os.path.join(PKG, "patch.py"),
    "persist":   os.path.join(PKG, "persistence.py"),
    "init":      os.path.join(PKG, "__init__.py"),
    "tests":     os.path.join(ROOT, "tests", "test_memory.py"),
    "ptests":    os.path.join(ROOT, "tests", "test_persistence.py"),
    "smoke":     os.path.join(ROOT, "scripts", "smoke_conversation.py"),
    "pyproject": os.path.join(ROOT, "pyproject.toml"),
}

CORE = True
SOFT = False


def patch_file(path, edits):
    """Apply (old, new, core, hint) tuples. Only runs after PHASE 0's
    preflight validated every anchor against the real tree."""
    src = file_text(path)
    plan = []
    for old, new, core, _hint in edits:
        if new in src:
            print("  [skip: already applied] " + os.path.basename(path))
            continue
        if old not in src:
            if core:
                die("core anchor vanished between preflight and write:\n"
                    + path + " ::\n" + old[:300])
            print("  [cosmetic skip: anchor not in your copy] "
                  + os.path.basename(path) + " :: "
                  + old.splitlines()[0][:70])
            continue
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
# PHASE 1 constants -- the fee-pair unfreeze + the F5 test blocks
# ---------------------------------------------------------------------------

FROZEN_OLD = '''def test_cross_type_thin_shared_restatement_stays():
    """Known residual, frozen deliberately: the live fee pair shares only
    ONE non-numeric word ('fee'); collapsing it would also merge
    tier-scoped facts ('Fee capped 0.5% cloud tier' vs 'premium tier')."""
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('s0', 0, 'constraint',
                                 'Maximum fee per transaction is limited to 0.5%', []))
    m._merge_patch(SemanticPatch('s1', 1, 'decision',
                                 'fee cap set to 0.5% as tracked', []))
    assert len(m.critical_patches) == 2'''

FROZEN_NEW = '''def test_cross_type_thin_shared_fee_pair_collapses():
    """v0.1.17 finding S2 unfreezes this pair. The 20-run live loop kept
    failing CRR at 87% (7/8 delivered) on 30% of runs: this frozen fee
    pair held the nominal store at 6 against the smoke's 7-critical
    delivery cap, so two extraction variants tipped a run to 8. Rule
    (b4) bridges it -- bound-word slot head ('Maximum ...') agreeing
    with its own direction, shared subject noun as the other slot head,
    within two turns. The tier danger the freeze guarded against is now
    separately frozen by test_bound_head_bridge_keeps_tier_scoped_fees
    and test_bound_head_bridge_keeps_gold_silver_tiers."""
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('s0', 0, 'constraint',
                                 'Maximum fee per transaction is limited to 0.5%', []))
    m._merge_patch(SemanticPatch('s1', 1, 'decision',
                                 'fee cap set to 0.5% as tracked', []))
    assert len(m.critical_patches) == 1, \\
        f"bound-head bridge should collapse the fee echo: {[p.payload for p in m.critical_patches]}"'''

TM_NEW = [
    "test_cross_type_thin_shared_fee_pair_collapses",
    "test_fee_cap_bridge_symmetric_direction",
    "test_bound_head_bridge_canary_scope_noun_merges",
    "test_bound_head_bridge_keeps_tier_scoped_fees",
    "test_bound_head_bridge_keeps_gold_silver_tiers",
    "test_bound_head_bridge_requires_echo_proximity",
    "test_bound_head_bridge_keeps_different_values",
    "test_bound_head_bridge_keeps_contradictory_bounds",
    "test_no_more_than_head_direction_mismatch_stays",
    "test_source_role_rides_on_add_turn",
    "test_render_source_roles_flag",
    "test_stats_reports_critical_roles",
    "test_stats_separates_conversation_patch_context_tokens",
]
TP_NEW = ["test_source_role_roundtrips_through_save_load",
          "test_conversation_tokens_roundtrip"]

TESTS_TM = '''
# ---------------------------------------------------------------------------
# v0.1.17 finding S2 (20-run live loop: 6/20 runs at CRR 87% = 7/8) plus
# the external review's actionable points. (b4) bridges thin cross-type
# echoes via a bound-word slot head; the bake stops telling a value
# twice; source_role and the three-way token split answer review points
# 2 and 4. CRR-on-empty-selection (review point 1) was fixed in the F2
# batch and is already pinned above.
# ---------------------------------------------------------------------------


def test_fee_cap_bridge_symmetric_direction():
    """The bridge is direction-agnostic: DEC first, CON echo second."""
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('s0', 0, 'decision',
                                 'fee cap set to 0.5% as tracked', []))
    m._merge_patch(SemanticPatch('s1', 1, 'constraint',
                                 'Maximum fee per transaction is limited to 0.5%', []))
    assert len(m.critical_patches) == 1, \\
        f"symmetric fee echo should collapse: {[p.payload for p in m.critical_patches]}"


def test_bound_head_bridge_canary_scope_noun_merges():
    """KNOWN LIMITATION, frozen deliberately (review point 3): word overlap
    cannot tell 'Max users 500 per org' from 'org admins cap 500' -- same
    value, same direction, shared scope noun 'org', bound-word head 'max'.
    (b4) merges this pair. The structured slot supersession planned for
    the next batch is the fix; when it lands, this canary is the test to flip."""
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('s0', 0, 'constraint',
                                 'Max users 500 per org', []))
    m._merge_patch(SemanticPatch('s1', 1, 'decision',
                                 'org admins cap 500', []))
    assert len(m.critical_patches) == 1, \\
        f"canary: (b4) merges scope-noun pairs (documented): {[p.payload for p in m.critical_patches]}"


def test_bound_head_bridge_keeps_tier_scoped_fees():
    """The case the old freeze protected: bound-word head, but the other
    side's head ('silver') is NOT the shared subject noun -- the pair
    shares only 'fee' and the qualifiers differ. Must stay apart."""
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('s0', 0, 'constraint',
                                 'Maximum platinum fee is 0.5%', []))
    m._merge_patch(SemanticPatch('s1', 1, 'decision',
                                 'silver tier fee cap 0.5%', []))
    assert len(m.critical_patches) == 2, \\
        f"tier-scoped fees merged: {[p.payload for p in m.critical_patches]}"


def test_bound_head_bridge_keeps_gold_silver_tiers():
    """No bound-word head on either side -> no bridge, regardless of the
    identical value and direction."""
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('s0', 0, 'constraint',
                                 'Gold tier fee capped at 0.5%', []))
    m._merge_patch(SemanticPatch('s1', 1, 'decision',
                                 'Silver tier fee capped at 0.5%', []))
    assert len(m.critical_patches) == 2


def test_bound_head_bridge_requires_echo_proximity():
    """Beyond two turns the bridge closes: extraction echoes are adjacent;
    distant restatements with thin overlap are not worth the risk."""
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('s0', 0, 'constraint',
                                 'Maximum fee per transaction is limited to 0.5%', []))
    m._merge_patch(SemanticPatch('s1', 5, 'decision',
                                 'fee cap set to 0.5% as tracked', []))
    assert len(m.critical_patches) == 2


def test_bound_head_bridge_keeps_different_values():
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('s0', 0, 'constraint',
                                 'Maximum fee per transaction is limited to 0.5%', []))
    m._merge_patch(SemanticPatch('s1', 1, 'decision',
                                 'fee cap set to 1% as tracked', []))
    assert len(m.critical_patches) == 2


def test_bound_head_bridge_keeps_contradictory_bounds():
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('s0', 0, 'constraint',
                                 'Maximum fee 0.5%', []))
    m._merge_patch(SemanticPatch('s1', 1, 'decision',
                                 'fee floor 0.5%', []))
    assert len(m.critical_patches) == 2, \\
        f"contradictory bounds merged: {[p.payload for p in m.critical_patches]}"


def test_no_more_than_head_direction_mismatch_stays():
    """'No more than 500 records per page' IS an upper bound, but its
    slot head 'more' is a LOWER-bound word inside an upper-bound
    sentence -- the head-direction agreement requirement exists exactly
    so negated bounds never bridge. Same fact, kept apart: conservative
    is correct for a rule this deep in the risk curve."""
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('s0', 0, 'constraint',
                                 'No more than 500 records per page', []))
    m._merge_patch(SemanticPatch('s1', 1, 'decision',
                                 'page size capped at 500', []))
    assert len(m.critical_patches) == 2


def test_source_role_rides_on_add_turn():
    """Review point 2, storage half: the F2 batch threaded the role into
    the extractor prompt; it must also ride on the extracted patches."""
    llm = MockLLMClient()
    mem = DSPMMemory(llm_client=llm, model='test-model', budget=250)
    patches = mem.add_turn('assistant', 'I will always return JSON output.')
    assert patches, "extraction returned no patches"
    assert all(p.source_role == 'assistant' for p in patches), \\
        [p.source_role for p in patches]


def test_render_source_roles_flag():
    """Role attribution renders only when asked for -- it costs tokens."""
    m = DSPMMemory(budget=250)
    p = SemanticPatch('r0', 0, 'constraint', 'fee cap set to 0.5% as tracked', [])
    p.source_role = 'user'
    m._merge_patch(p)
    assert '(user)' not in m.get_context('fee')       # off by default
    m2 = DSPMMemory(budget=250, render_source_roles=True)
    p2 = SemanticPatch('r1', 0, 'constraint', 'fee cap set to 0.5% as tracked', [])
    p2.source_role = 'user'
    m2._merge_patch(p2)
    assert '(user)' in m2.get_context('fee')


def test_stats_reports_critical_roles():
    m = DSPMMemory(budget=250)
    p = SemanticPatch('a0', 0, 'constraint', 'PCI-DSS compliance required', [])
    p.source_role = 'user'
    m._merge_patch(p)
    d = SemanticPatch('a1', 1, 'decision', 'dashboard loads in under 2 seconds', [])
    d.source_role = 'assistant'
    m._merge_patch(d)
    assert m.stats['critical_by_role'] == {'user': 1, 'assistant': 1}


def test_stats_separates_conversation_patch_context_tokens():
    """Review point 4: conversation_tokens counts the RAW turn text fed
    to add_turn; patch_tokens (alias raw_tokens) counts the extracted
    patches; context_tokens counts the delivery."""
    from dspm.patch import count_tokens
    llm = MockLLMClient()
    mem = DSPMMemory(llm_client=llm, model='test-model', budget=250)
    t1 = 'Return JSON and use POST /v1/items for creation.'
    t2 = 'All API responses must include a request id header.'
    mem.add_turn('user', t1)
    mem.add_turn('assistant', t2)
    s = mem.stats
    assert s['conversation_tokens'] == count_tokens(t1) + count_tokens(t2), s
    assert s['patch_tokens'] == s['raw_tokens'] > 0
    mem.get_context('json')
    s2 = mem.stats
    assert s2['context_tokens'] > 0
    assert s2['patch_tokens'] >= s2['context_tokens'], s2
'''

TESTS_TP = '''

class TestRoleAndTokenAccounting:
    """v0.1.17 review points 2+4 through the notebook format."""

    def test_source_role_roundtrips_through_save_load(self, tmp_path):
        m1 = DSPMMemory(budget=250)
        p = _mk_patch("p0_0", 0, "constraint", "PCI-DSS compliance required")
        p.source_role = "user"
        m1._merge_patch(p)
        d = _mk_patch("p1_0", 1, "decision", "fee cap set to 0.5% as tracked")
        d.source_role = "assistant"
        m1._merge_patch(d)
        path = str(tmp_path / "mem.json")
        m1.save(path)
        m2 = DSPMMemory(budget=250)
        assert m2.load(path) == 2
        roles = {p.payload: p.source_role for p in m2.all_patches}
        assert roles["PCI-DSS compliance required"] == "user", roles
        assert roles["fee cap set to 0.5% as tracked"] == "assistant", roles

    def test_conversation_tokens_roundtrip(self, tmp_path):
        m1 = DSPMMemory(budget=250)
        m1._conversation_tokens = 137
        m1._merge_patch(_mk_patch("p0_0", 0, "constraint", "must work offline"))
        path = str(tmp_path / "mem.json")
        m1.save(path)
        m2 = DSPMMemory(budget=250)
        m2.load(path)
        assert m2._conversation_tokens == 137, m2._conversation_tokens
'''

# ---------------------------------------------------------------------------
# PHASE 3 constants -- every edit is (old, new, core, hint)
# ---------------------------------------------------------------------------

DOC_OLD = '"""\n\nfrom __future__ import annotations'
DOC_NEW = (
    'v0.1.17 (unpublished, F5 batch): rule (b4) bridges thin cross-type\n'
    'extraction echoes -- a bound-word slot head that agrees with its own\n'
    "bound direction, the other side's slot head being the shared subject\n"
    'noun, within two turns. The frozen fee pair held the nominal store\n'
    "at 6 against the smoke's 7-critical delivery cap, so 6/20 live runs\n"
    'tipped to 8 and CRR fell to 87%. A canary test freezes the remaining\n'
    'word-overlap limitation (same-value bounds sharing only a scope noun\n'
    'still merge; structured slot supersession is the planned next step,\n'
    'reviewer point 3). Same batch: patches carry source_role through the\n'
    'store and the notebook (reviewer point 2), stats splits conversation\n'
    '/ patch / context tokens (reviewer point 4), and the (was X) span\n'
    'stays the machine-readable revision story the starvation trim\n'
    'protects -- prose is droppable, the span is not.\n'
    '"""\n\nfrom __future__ import annotations'
)

RULES_OLD = (
    '    #   (b3) cross-type CON<->DEC same-value restatement: same anchor,\n'
    '    #        >=2 shared non-numeric words, identical numbers + units,\n'
    '    #        compatible bounds, at least one directional/exact bound\n'
    '    #        (v0.1.17 finding S1 -- live smoke showed user [CON] +\n'
    '    #        assistant [DEC] echoes coexisting as duplicate criticals)'
)
RULES_NEW = RULES_OLD + (
    '\n    #   (b4) cross-type thin echo: bound-word slot head on one side,\n'
    '    #        shared subject noun as the other slot head, <=2 turns\n'
    '    #        apart (v0.1.17 finding S2 -- the frozen fee pair held\n'
    '    #        the store at 6 against the 7-critical delivery cap;\n'
    '    #        6/20 live runs tipped to 8 and CRR fell to 87%)'
)

B3COMMENT_OLD = (
    "                    # is plain/plain and must stay apart). Known residual,\n"
    "                    # frozen by test: thin restatements sharing ONE\n"
    "                    # non-numeric word ('Maximum fee ... 0.5%' vs 'fee cap\n"
    "                    # ... 0.5%') stay -- collapsing them would also merge\n"
    "                    # tier-scoped facts with identical numbers."
)
B3COMMENT_NEW = (
    "                    # is plain/plain and must stay apart). The thin-restatement\n"
    "                    # freeze on this rule was lifted in finding S2 -- see (b4)\n"
    "                    # below for the guarded bridge that replaced it."
)

B3BLOCK_OLD = (
    '                        stale.append(existing)          # (b3) cross-type restatement\n'
    '                        continue'
)
B3BLOCK_NEW = r'''                        stale.append(existing)          # (b3) cross-type restatement
                        continue
                    # v0.1.17 (finding S2): (b4) bound-head bridge for thin
                    # cross-type echoes. The 20-run live loop still failed
                    # CRR 87% (7/8 delivered) on 30% of runs: the frozen
                    # fee pair held the nominal store at 6 against a
                    # 7-critical delivery cap, so two extraction variants
                    # tipped a run to 8. The fee pair's shape: one side
                    # leads with a BOUND WORD ('Maximum fee ... 0.5%') and
                    # the other leads with the shared subject noun
                    # ('fee cap set to 0.5%'). Bridge conditions, all
                    # required: the bound-word head agrees with its OWN
                    # payload's bound direction ('No more than ...' does
                    # NOT bridge -- its head 'more' is a lower-bound word
                    # inside an upper-bound sentence), the other side's
                    # slot head IS a shared content word, and the pair
                    # sits within two turns (extraction echoes are
                    # adjacent; tier-scoped facts stated turns apart are
                    # not). Canary-frozen limitation: same-value bounds
                    # sharing only a scope noun still merge ('Max users
                    # 500 per org' + 'org admins cap 500') -- structured
                    # slot supersession is the planned next fix
                    # (reviewer point 3).
                    old_head = _slot_head(existing.payload)
                    shared_nonnum = {w for w in shared if not re.search(r"\d", w)}
                    if (new_nums and new_nums == _numbers(existing.payload)
                            and _units_in(patch.payload) == _units_in(existing.payload)
                            and _polarity_compatible(bc_new, bc_old)
                            and not (bc_new == "plain" and bc_old == "plain")
                            and abs(patch.turn_index - existing.turn_index) <= 2
                            and ((new_head in _BOUND_HEAD_WORDS
                                  and _head_direction(new_head) == bc_new
                                  and old_head in shared_nonnum)
                                 or (old_head in _BOUND_HEAD_WORDS
                                     and _head_direction(old_head) == bc_old
                                     and new_head in shared_nonnum))):
                        stale.append(existing)          # (b4) thin echo bridge
                        continue'''

COMPAT_OLD = '    return (a, b) in _BOUND_COMPAT or (b, a) in _BOUND_COMPAT'
COMPAT_NEW = '''    return (a, b) in _BOUND_COMPAT or (b, a) in _BOUND_COMPAT


# v0.1.17 (finding S2): bound words that can serve as a slot head and
# bridge a thin cross-type echo (rule (b4)).
_BOUND_HEAD_WORDS = frozenset(_BOUND_UP | _BOUND_DOWN | _BOUND_EXACT)


def _head_direction(head: str) -> str:
    """Direction class implied by a bound-word slot head. The head must
    agree with its OWN payload's bound class to bridge: 'No more than 5'
    is an upper bound whose head 'more' is a lower-bound word, so it
    never bridges (guarded by test_no_more_than_head_direction_mismatch_stays)."""
    if head in _BOUND_UP:
        return "upper"
    if head in _BOUND_DOWN:
        return "lower"
    if head in _BOUND_EXACT:
        return "exact"
    return "plain"'''

SIG_OLD = (
    '    def __init__(self, budget: int = DEFAULT_BUDGET, llm_client: Optional[Any] = None,\n'
    '                 model: str = "gpt-4o-mini", dense: bool = False,\n'
    '                 stable_order: bool = False, **kwargs: Any):'
)
SIG_NEW = (
    '    def __init__(self, budget: int = DEFAULT_BUDGET, llm_client: Optional[Any] = None,\n'
    '                 model: str = "gpt-4o-mini", dense: bool = False,\n'
    '                 stable_order: bool = False, render_source_roles: bool = False,\n'
    '                 **kwargs: Any):'
)

STATE_OLD = (
    '        self.selected_patches: List[SemanticPatch] = []\n'
    '        self._last_context = ""\n'
    '        self._context_built = False'
)
STATE_NEW = '''        self.selected_patches: List[SemanticPatch] = []
        self._last_context = ""
        self._context_built = False
        # v0.1.17 (review points 2+4): source attribution + input-token
        # accounting. _conversation_tokens counts the RAW turn text fed
        # into add_turn (the extraction INPUT), separate from
        # patch_tokens (extraction OUTPUT) and context_tokens (the
        # delivered result) -- the reviewer's point: raw_tokens measured
        # extracted patches and was easy to misread as conversation size.
        self._conversation_tokens = 0
        # v0.1.17 (review point 2): optional '(role)' suffix on delivered
        # lines. Off by default -- it costs ~2 tokens per delivered line
        # and budgets were sized without it.
        self.render_source_roles = bool(render_source_roles)'''

ADDTURN_OLD = (
    '        for p in patches:\n'
    '            self._merge_patch(p)\n'
    '        self.turns += 1'
)
ADDTURN_NEW = '''        # v0.1.17 (review point 4): the raw turn text is the extraction
        # INPUT -- counted whether or not extraction succeeds.
        self._conversation_tokens += count_tokens(text)
        for p in patches:
            # v0.1.17 (review point 2, data path): attribution rides on
            # the patch. The extractor PROMPT has carried the role since
            # the F2 batch; the store could not tell a user requirement
            # from an assistant echo.
            p.source_role = role
            self._merge_patch(p)
        self.turns += 1'''

RENDER_OLD = (
    '        context = self.engine.build_context(selected)\n'
    '        self._last_context = context\n'
    '        return context'
)
RENDER_NEW = '''        context = self.engine.build_context(selected)
        # v0.1.17 (review point 2): optional role attribution on delivered
        # lines, when render_source_roles=True and the patch carries a
        # role. Rendered AFTER selection, so it never changes what the
        # engine picks -- only what the reader sees about who said it.
        if self.render_source_roles:
            context = "\\n".join(
                f"{p.to_prompt_str()} ({p.source_role})"
                if p.source_role else p.to_prompt_str()
                for p in selected) or context
        self._last_context = context
        return context'''

STATSDOC_OLD = (
    '        clamping a negative reduction would be metric gaming, and\n'
    '        raw_tokens / context_tokens stay exposed for audit.'
)
STATSDOC_NEW = STATSDOC_OLD + (
    '\n        v0.1.17 (review point 4): conversation_tokens (raw turn\n'
    '        text fed to add_turn) and patch_tokens (alias of\n'
    '        raw_tokens -- the EXTRACTED patches) make the three-way\n'
    '        split explicit, so nobody mistakes patch tokens for\n'
    '        conversation size again.'
)

KEYS_OLD = (
    '            "raw_tokens": raw_tokens,\n'
    '            "context_tokens": context_tokens,\n'
    '            "trr": trr,\n'
    '        }'
)
KEYS_NEW = '''            "conversation_tokens": self._conversation_tokens,
            "raw_tokens": raw_tokens,
            "patch_tokens": raw_tokens,
            "context_tokens": context_tokens,
            "trr": trr,
            # v0.1.17 (review point 2): attribution census for criticals
            # -- 'unattributed' covers pre-0.1.17 notebooks.
            "critical_by_role": dict(sorted(roles.items())),
        }'''

ROLES_OLD = (
    '        selected_count = len(self.selected_patches) if self._context_built else len(self.patches)\n'
    '        return {'
)
ROLES_NEW = '''        selected_count = len(self.selected_patches) if self._context_built else len(self.patches)
        roles: Dict[str, int] = {}
        for p in self.critical_patches:
            key = p.source_role or "unattributed"
            roles[key] = roles.get(key, 0) + 1
        return {'''

RESET_OLD = (
    '        self.turns = 0\n'
    '        self._last_context = ""\n'
    '        self._context_built = False\n'
    '        self.engine.reset_ema()'
)
RESET_NEW = (
    '        self.turns = 0\n'
    '        self._last_context = ""\n'
    '        self._context_built = False\n'
    '        self._conversation_tokens = 0\n'
    '        self.engine.reset_ema()'
)

EDITS_MEMORY = [
    (DOC_OLD, DOC_NEW, SOFT, 'from __future__ import annotations'),
    (RULES_OLD, RULES_NEW, SOFT, '(b3) cross-type CON'),
    (B3COMMENT_OLD, B3COMMENT_NEW, SOFT, 'Known residual,'),
    (B3BLOCK_OLD, B3BLOCK_NEW, CORE, 'cross-type restatement'),
    (COMPAT_OLD, COMPAT_NEW, CORE, '_BOUND_COMPAT'),
    (SIG_OLD, SIG_NEW, CORE, 'def __init__(self, budget'),
    (STATE_OLD, STATE_NEW, CORE, '_context_built = False'),
    (ADDTURN_OLD, ADDTURN_NEW, CORE, 'for p in patches:'),
    (RENDER_OLD, RENDER_NEW, CORE, 'build_context(selected)'),
    (STATSDOC_OLD, STATSDOC_NEW, SOFT, 'metric gaming'),
    (KEYS_OLD, KEYS_NEW, CORE, '"raw_tokens"'),
    (ROLES_OLD, ROLES_NEW, CORE, 'selected_count = len'),
    (RESET_OLD, RESET_NEW, CORE, 'reset_ema()'),
]

EDITS_PATCH = [
    ('from typing import List\n', 'from typing import List, Optional\n',
     CORE, 'from typing import'),
    ('(dataclasses.field, config.PATCH_TYPES) were removed.\n"""',
     '''(dataclasses.field, config.PATCH_TYPES) were removed.

v0.1.17: source_role field -- the speaker label captured at add_turn()
time, so a saved notebook distinguishes user requirements from
assistant echoes (review point 2, storage half). Optional with a None
default: pre-0.1.17 notebooks and manual test patches load as
unattributed.
"""''', SOFT, 'PATCH_TYPES) were removed'),
    ('    causal_depth: int = 0\n\n    def __post_init__(self) -> None:',
     '''    causal_depth: int = 0
    # v0.1.17 (review point 2, storage half): speaker attribution, set by
    # add_turn() right after extraction. None = manual patch or
    # pre-0.1.17 notebook (loads as unattributed).
    source_role: Optional[str] = None

    def __post_init__(self) -> None:''', CORE, 'causal_depth'),
]

EDITS_PERSIST = [
    ('        "dependencies": list(p.dependencies),\n    }',
     '''        "dependencies": list(p.dependencies),
        # v0.1.17 (review point 2): speaker attribution rides the record;
        # old readers ignore unknown fields, old files load as None.
        "source_role": getattr(p, "source_role", None),
    }''', CORE, 'list(p.dependencies)'),
    ('            dependencies=[str(x) for x in deps],\n        )',
     '''            dependencies=[str(x) for x in deps],
            source_role=(str(d["source_role"]) if d.get("source_role") else None),
        )''', CORE, 'for x in deps]'),
    ('    payload = {"schema": _SCHEMA_VERSION, "turns": memory.turns, "patches": records}',
     '''    # v0.1.17 (review point 4): record the raw input tokens of the turns
    # that produced this notebook, so a merged session can still report
    # what the conversation cost.
    payload = {"schema": _SCHEMA_VERSION, "turns": memory.turns,
               "conversation_tokens": getattr(memory, "_conversation_tokens", 0),
               "patches": records}''', CORE, '_SCHEMA_VERSION'),
    ('    if isinstance(data, dict) and data.get("turns"):\n'
     '        memory.turns = max(memory.turns, int(data["turns"]))\n'
     '    return loaded',
     '''    if isinstance(data, dict) and data.get("turns"):
        memory.turns = max(memory.turns, int(data["turns"]))
    # v0.1.17 (review point 4): merge semantics -- loaded notebooks add
    # their recorded input tokens to this session's counter.
    if isinstance(data, dict) and data.get("conversation_tokens"):
        base = getattr(memory, "_conversation_tokens", 0)
        memory._conversation_tokens = base + int(data["conversation_tokens"])
    return loaded''', CORE, 'data.get("turns")'),
    ('Unused typing.List import removed. No schema change.\n"""',
     '''Unused typing.List import removed. No schema change.

v0.1.17: patch records carry source_role (review point 2 -- speaker
attribution survives save/load), and the notebook header carries
conversation_tokens (review point 4 -- the raw input size that produced
it; loads additively under merge semantics). Both are optional keys:
pre-0.1.17 files load as unattributed / zero.
"""''', SOFT, 'Unused typing.List import removed'),
]

EDITS_SMOKE = [
    ("critical-count bloat from restatement duplicates - the known "
     "v0.1.14 issue, not the prompt fix",
     "critical-count bloat from uncollapsed duplicates -- the "
     "v0.1.17 (b3)/(b4) rules track it; if this repeats, run "
     "diag_smoke_criticals.py and paste the DIAG block",
     SOFT, 'the known v0.1.14 issue'),
    ("the known v0.1.14 issue, not the prompt fix",
     "a v0.1.17 residual -- run diag_smoke_criticals.py for the "
     "store dump",
     SOFT, 'v0.1.14 issue, not the prompt'),
]

# ==================== Part 2 of 2 (append below Part 1) ====================

# ---------------------------------------------------------------------------
# PHASE 0 -- package + versions + FULL ANCHOR PREFLIGHT (writes NOTHING)
# ---------------------------------------------------------------------------

hr("PHASE 0 -- PACKAGE + VERSIONS + FULL ANCHOR PREFLIGHT")
print("package dir : " + PKG)
for label, key in (("dspm/__init__.py", "init"),
                   ("pyproject.toml", "pyproject")):
    m = find_version(P[key]) if False else None
for label, key in (("dspm/__init__.py", "init"),
                   ("pyproject.toml", "pyproject")):
    m = None
    if os.path.isfile(P[key]):
        t = file_text(P[key])
        m = (re.search(r'__version__\s*=\s*"([^"]+)"', t)
             or re.search(r'^version\s*=\s*"([^"]+)"', t, re.M))
    print("%-18s %s" % (label, m.group(1) if m else "<no version line>"))
    if m and m.group(1) != "0.1.17":
        print("  NOTE: expected 0.1.17. This script never bumps versions;")
        print("  continue only if that is your intended release line.")

CORE_MISSING = []
SOFT_MISSING = []


def preflight(path, edits, label):
    if not os.path.isfile(path):
        print("  %-28s FILE MISSING" % label)
        CORE_MISSING.append((label, path, "<the file itself is missing>"))
        return
    src = file_text(path)
    ok = applied = miss_c = miss_s = 0
    for old, new, core, _hint in edits:
        if new in src:
            applied += 1
        elif old in src:
            ok += 1
        elif core:
            miss_c += 1
            CORE_MISSING.append((label, path, old))
        else:
            miss_s += 1
            SOFT_MISSING.append((label, path, old))
    print("  %-28s %d ok, %d already applied, %d cosmetic-missing (skip)"
          % (label, ok, applied, miss_s))


preflight(P["memory"], EDITS_MEMORY, "memory.py core+soft")
preflight(P["patch"], EDITS_PATCH, "patch.py core+soft")
preflight(P["persist"], EDITS_PERSIST, "persistence.py core+soft")
if os.path.isfile(P["smoke"]):
    preflight(P["smoke"], EDITS_SMOKE, "smoke hint (soft)")
else:
    print("  [skip: scripts/smoke_conversation.py not found]")

# tests are not old->new edits; check their preconditions explicitly
tsrc = file_text(P["tests"]) if os.path.isfile(P["tests"]) else ""
ptsrc = file_text(P["ptests"]) if os.path.isfile(P["ptests"]) else ""
if not tsrc:
    CORE_MISSING.append(("tests/test_memory.py", P["tests"],
                         "<the file itself is missing>"))
elif ("def " + TM_NEW[0] + "(") in tsrc:
    print("  %-28s fee pair already unfrozen by a previous F5 run"
          % "tests/test_memory.py")
elif FROZEN_OLD in tsrc:
    print("  %-28s fee-pair unfreeze anchor ok" % "tests/test_memory.py")
elif "def test_cross_type_thin_shared_restatement_stays(" in tsrc:
    CORE_MISSING.append(("fee-pair unfreeze", P["tests"],
                         FROZEN_OLD))
else:
    CORE_MISSING.append(("fee-pair unfreeze", P["tests"],
                          "<neither the old frozen fee test nor the F5 "
                          "replacement found>"))
if tsrc:
    present_m = [n for n in TM_NEW[1:] if ("def " + n + "(") in tsrc]
    if len(present_m) == len(TM_NEW[1:]):
        print("  %-28s all 12 F5 memory tests already present"
              % "tests/test_memory.py")
    elif not present_m:
        print("  %-28s clean tail -- 12 tests will append"
              % "tests/test_memory.py")
    else:
        CORE_MISSING.append(("F5 memory-test block", P["tests"],
                             "<PARTIAL block: %s present>"
                             % ", ".join(present_m)))
if not ptsrc:
    CORE_MISSING.append(("tests/test_persistence.py", P["ptests"],
                         "<the file itself is missing>"))
elif all(("def " + n + "(") in ptsrc for n in TP_NEW):
    print("  %-28s both F5 persistence tests already present"
          % "tests/test_persistence.py")
elif not any(("def " + n + "(") in ptsrc for n in TP_NEW):
    print("  %-28s clean tail -- 2 tests will append"
          % "tests/test_persistence.py")
else:
    CORE_MISSING.append(("F5 persistence-test block", P["ptests"],
                         "<PARTIAL block>"))

if SOFT_MISSING:
    print()
    print("  cosmetic anchors not found (these edits will warn + skip):")
    for label, path, old in SOFT_MISSING:
        print("    - %s :: %s" % (label, old.splitlines()[0][:70]))
if CORE_MISSING:
    print()
    print("MISSING CORE ANCHORS -- NOTHING HAS BEEN WRITTEN:")
    for label, path, old in CORE_MISSING:
        print()
        print("  [%s] %s" % (label, path))
        print("    expected region starts with:")
        for ln in old.splitlines()[:4]:
            print("      " + ln[:92])
        frag = old.splitlines()[0].strip()[:44].replace("'", "")
        if frag.startswith("<"):
            print("    (no text to search for this one -- paste the file)")
        else:
            print("    print it on your machine and paste the output:")
            print("      Select-String -Path '%s' -SimpleMatch -Pattern "
                  "'%s' -Context 6,10"
                  % (path.replace(ROOT + os.sep, ""), frag))
    die("fix the anchors above (or paste their real text back to me), "
        "then re-run. No file has been modified.")
print()
print("ALL ANCHORS OK -- proceeding. Backups: .f5bak, written once per file.")

# ---------------------------------------------------------------------------
# PHASE 1 -- tests: unfreeze the fee pair, append the S2 + review blocks
# ---------------------------------------------------------------------------

hr("PHASE 1 -- S2 + review tests")
tsrc = file_text(P["tests"])
ptsrc = file_text(P["ptests"])
unfroze = False
if "def test_cross_type_thin_shared_restatement_stays(" in tsrc:
    if FROZEN_OLD not in tsrc:
        die("the old frozen fee test is present but its body differs "
            "from the expected text -- paste the output of:\n"
            "  Select-String -Path 'tests\\test_memory.py' -SimpleMatch "
            "-Pattern 'thin_shared' -Context 2,18")
    backup(P["tests"])
    tsrc = tsrc.replace(FROZEN_OLD, FROZEN_NEW, 1)
    write_text(P["tests"], tsrc)
    unfroze = True
    print("  unfrozen: fee-pair residual test now asserts the (b4) collapse")
elif ("def " + TM_NEW[0] + "(") in tsrc:
    unfroze = True          # already unfrozen by a previous F5 run
if not unfroze:
    die("neither the old frozen fee test nor its F5 replacement was found "
        "in tests/test_memory.py")
rest = TM_NEW[1:]
present_m = [n for n in rest if ("def " + n + "(") in tsrc]
if len(present_m) == len(rest):
    print("[skip: all %d F5 memory tests already in tests/test_memory.py]"
          % len(rest))
elif present_m:
    die("only some F5 memory tests are present (%s) -- paste the tail of "
        "tests/test_memory.py back" % ", ".join(present_m))
else:
    backup(P["tests"])
    write_text(P["tests"],
               tsrc.rstrip("\n") + "\n\n\n" + TESTS_TM.strip("\n") + "\n")
    print("  tests appended: %d new to tests/test_memory.py" % len(rest))
present_p = [n for n in TP_NEW if ("def " + n + "(") in ptsrc]
if len(present_p) == len(TP_NEW):
    print("[skip: both F5 persistence tests already in tests/test_persistence.py]")
elif present_p:
    die("only some F5 persistence tests are present (%s)" % ", ".join(present_p))
else:
    backup(P["ptests"])
    write_text(P["ptests"],
               ptsrc.rstrip("\n") + "\n" + TESTS_TP.strip("\n") + "\n")
    print("  tests appended: %d new to tests/test_persistence.py" % len(TP_NEW))

NODES = (["tests/test_memory.py::" + n for n in TM_NEW]
         + ["tests/test_persistence.py::TestRoleAndTokenAccounting::" + n
            for n in TP_NEW])

# ---------------------------------------------------------------------------
# PHASE 2 -- RED baseline
# ---------------------------------------------------------------------------

hr("PHASE 2 -- RED BASELINE (PREDICTION from my sandbox: 9 of 15 fail "
   "pre-patch: bridge x3, role x4, tokens x2; guards x6 pass)")
base = run_pytest(["-q"] + NODES, capture=True)
if base.returncode == 0:
    print("NOTE: all fifteen already pass -- fixes already applied? The patch")
    print("phase below is idempotent and will simply skip.")
else:
    m = re.search(r"(\d+) failed", base.stdout)
    n = int(m.group(1)) if m else -1
    if not (1 <= n <= 9):
        die("unexpected baseline result (%r failed of 15) -- expected at "
            "most the 9 known-red tests. Paste the output above" % n)
    print("RED BASELINE OK: %d known-red tests fail on the unfixed tree "
          "(the other %d guards already pass)." % (n, 15 - n))

# ---------------------------------------------------------------------------
# PHASE 3 -- anchored patches
# ---------------------------------------------------------------------------

hr("PHASE 3 -- S2/(b4) + source_role + token split: core files")
patch_file(P["memory"], EDITS_MEMORY)
patch_file(P["patch"], EDITS_PATCH)
patch_file(P["persist"], EDITS_PERSIST)
if os.path.isfile(P["smoke"]):
    patch_file(P["smoke"], EDITS_SMOKE)
else:
    print("  [skip: scripts/smoke_conversation.py not found]")

# ---------------------------------------------------------------------------
# PHASE 4 -- GREEN re-check
# ---------------------------------------------------------------------------

hr("PHASE 4 -- GREEN RE-CHECK: 15/15")
green = run_pytest(["-q"] + NODES, capture=True)
if green.returncode != 0:
    die("GREEN check failed -- the fixes did not land correctly. Paste "
        "everything back; do NOT re-run F5 yet.")
print("GREEN: 15/15 pass after patching.")

# ---------------------------------------------------------------------------
# PHASE 5 -- full gate
# ---------------------------------------------------------------------------

hr("PHASE 5 -- FULL GATE: pytest tests/ -q   (PREDICTION: 94 = your 80 + 14)")
coll = run_pytest(["tests/", "-q", "--collect-only"], capture=True)
m = re.search(r"(\d+) tests? collected", coll.stdout)
total = int(m.group(1)) if m else -1
print("collected: %d (PREDICTION from my sandbox: 94 = your 80 + 14 -- "
      "12 appended to test_memory.py, 2 to test_persistence.py, the fee "
      "test rewritten in place. A different count is fine if you added "
      "tests yourself; paste the number either way)" % total)

gate = run_pytest(["tests/", "-q"], capture=False)
if gate.returncode != 0:
    print()
    print("F5: FULL GATE FAILED -- the patches ARE on disk. Do NOT re-run F5")
    print("(it is idempotent, but I need the failure text first). Paste")
    print("everything back as-is.")
    sys.exit(2)

# ---------------------------------------------------------------------------
# PHASE 6 -- verify slices + probes (expectations are PREDICTIONS from my
# sandbox; a MISMATCH line is information for the next evidence round,
# not a failure)
# ---------------------------------------------------------------------------

VERIFY = r'''
import inspect
import dspm
from dspm import DSPMMemory
from dspm.memory import _head_direction
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


def mk(pid, turn, t, pl, role=None):
    p = SemanticPatch(patch_id=pid, turn_index=turn, patch_type=t,
                      payload=pl, dependencies=[])
    if role:
        p.source_role = role
    return p


def probe(label, rows, expect):
    m = DSPMMemory(budget=250)
    for r in rows:
        m._merge_patch(mk(*r))
    got = len(m.critical_patches)
    print("%-52s -> %d criticals (%s)"
          % (label, got, "OK" if got == expect else "MISMATCH"))


print("dspm.__version__ =", dspm.__version__)
show("S2 -- rule (b4)", inspect.getsource(DSPMMemory._merge_patch), "(b4)", 14)
print("head-direction probes:",
      _head_direction("maximum"), _head_direction("more"),
      _head_direction("cap"), _head_direction("exactly"))

probe("fee pair CON t0 + DEC t1 (live DIAG payloads)", [
    ("d1", 0, "constraint", "Maximum fee per transaction is limited to 0.5%"),
    ("d2", 1, "decision", "fee cap set to 0.5% as tracked")], 1)
probe("fee pair reversed (DEC t0, CON t1)", [
    ("d2", 0, "decision", "fee cap set to 0.5% as tracked"),
    ("d1", 1, "constraint", "Maximum fee per transaction is limited to 0.5%")], 1)
probe("tier-scoped platinum/silver (the old freeze's case)", [
    ("t0", 0, "constraint", "Maximum platinum fee is 0.5%"),
    ("t1", 1, "decision", "silver tier fee cap 0.5%")], 2)
probe("gold/silver tiers, no bound-word head", [
    ("g0", 0, "constraint", "Gold tier fee capped at 0.5%"),
    ("g1", 1, "decision", "Silver tier fee capped at 0.5%")], 2)
probe("distant fee restatement (t0 vs t5)", [
    ("d1", 0, "constraint", "Maximum fee per transaction is limited to 0.5%"),
    ("d2", 5, "decision", "fee cap set to 0.5% as tracked")], 2)
probe("'no more than' head-direction mismatch", [
    ("n0", 0, "constraint", "No more than 500 records per page"),
    ("n1", 1, "decision", "page size capped at 500")], 2)

m2 = DSPMMemory(budget=250)
m2._merge_patch(mk("v0", 0, "decision", "webhook timeout set to 30 seconds"))
m2._merge_patch(mk("v1", 1, "decision", "webhook timeout updated to 10 seconds"))
print("plain revision still bakes history:  ",
      [p.payload for p in m2.critical_patches])

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

m = DSPMMemory(budget=59)
for r in DIAG:
    m._merge_patch(mk(*r))
print("DIAG-store replay: %d criticals (PREDICTION: 5 -- the fee AND "
      "dashboard pairs both merge now; the live run had 7 then 6)"
      % len(m.critical_patches))
m.get_context("webhook timeout")
s = m.stats
print("replayed smoke at budget 59: critical_total=%d critical_selected=%d "
      "crr=%d (PREDICTION: 5/5 at crr 100)"
      % (s["critical_total"], s["critical_selected"], s["crr"]))

m3 = DSPMMemory(budget=59)
for r in DIAG + [
    ("v1", 8, "decision", "homepage response under 2 seconds"),
    ("v2", 9, "decision", "PCI compliance tracked"),
]:
    m3._merge_patch(mk(*r))
m3.get_context("webhook timeout")
s3 = m3.stats
print("variance probe (+2 extraction variants, store %d): "
      "critical_selected=%d crr=%d  (PREDICTION: store 7, delivered 7, "
      "crr 100 -- nominal 5 + delivery cap 7 = two spare slots)"
      % (s3["critical_total"], s3["critical_selected"], s3["crr"]))

mr = DSPMMemory(budget=250)
mr._conversation_tokens = count_tokens("Set the fee cap to 0.5 percent.")
mr._merge_patch(mk("x0", 0, "constraint",
                   "fee cap set to 0.5% as tracked", role="user"))
mr.get_context("fee")
sr = mr.stats
print("stats probe: conversation_tokens=%d patch_tokens=%d "
      "context_tokens=%d critical_by_role=%s"
      % (sr["conversation_tokens"], sr["patch_tokens"],
         sr["context_tokens"], sr["critical_by_role"]))
mr2 = DSPMMemory(budget=250, render_source_roles=True)
mr2._merge_patch(mk("x1", 0, "constraint",
                    "fee cap set to 0.5% as tracked", role="user"))
print("role-render probe:", repr(mr2.get_context("fee")))
'''

hr("PHASE 6 -- VERIFY SLICES + DIAG-STORE REPLAY")
subprocess.run([sys.executable, "-c", VERIFY], cwd=ROOT)

hr("F5 COMPLETE")
print("Rule (b4) bridges the fee pair (nominal store 5 vs delivery cap 7),")
print("source_role rides the patches and the notebook, stats splits the")
print("three token numbers. Backups: .f5bak next to every patched file.")
print()
print("NOT covered by F5, on the record: rule (a) can still sweep several")
print("overlapping slots in one revision; rule (c) still has no polarity")
print("check. Both are queued for the next batch.")
print()
print("NEXT -- after the gate is green, the live confirmation loop (fresh")
print("SHORT-LIVED key, revoke it afterwards):")
print()
print('  $sec = Read-Host "key" -AsSecureString')
print("  $env:GROQ_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($sec))")
print('  1..20 | ForEach-Object { & $py scripts\\smoke_conversation.py 2>&1 | Select-String -CaseSensitive "PASS|FAIL" }')
print()
print("Prediction: 4x PASS on all 20 runs. On ANY FAIL: run your current")
print("diag_smoke_criticalis.py if you have it -- your original diag still")
print("dumps the store -- and paste the DIAG block; the next evidence")
print("round targets whatever pairs remain.")
print()
print("Then paste the PHASE 5 gate tail + PHASE 6 block back. After you")
print("confirm F5 works, I send eval_harness.py (review point 5): it")
print("measures the STORE + DELIVERY layers offline (collapse precision,")
print("fact recall in the delivered context, revision correctness, CRR")
print("honesty) and, in --live mode, extraction reliability + latency +")
print("cost. It does NOT judge answer accuracy and uses no judge model.")
print("Then the publish checklist: README/changelog v0.1.17 (keep the T7")
print("'reserved, not wired' wording), python -m build, twine check,")
print("TestPyPI dry run, PyPI, git tag v0.1.17.")