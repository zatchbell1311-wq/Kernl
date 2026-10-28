"""
Verification script for Bug #6 (DSPM)

Bug #6 - TRR (Token Reduction Rate) is computed on patch_tokens
         (extracted patches) instead of conversation_tokens (raw turn
         text fed to add_turn). The metric underreports actual token
         usage because patch_tokens << conversation_tokens in real
         conversations.

Run from the repo root with the dspm venv active:

    python scripts/verify_bug6.py
"""

import inspect
import json
from dspm import DSPMMemory


def banner(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


class _FakeLLM:
    """Mock LLM that returns small patches per turn."""

    def __init__(self):
        self._patches = [
            {"patch_type": "decision", "payload": "Use Stripe for payments.",
             "patch_id": "x1", "dependencies": []},
            {"patch_type": "constraint", "payload": "Return JSON output always.",
             "patch_id": "x2", "dependencies": []},
        ]
        self.content = json.dumps(self._patches)

        class MockResponse:
            def __init__(self, content):
                self.choices = [type('Choice', (), {
                    'message': type('Message', (), {'content': content}),
                    'finish_reason': 'stop',
                })]

        class Completions:
            def __init__(self, owner):
                self.owner = owner

            def create(self, **kwargs):
                return MockResponse(self.owner.content)

        class Chat:
            def __init__(self, owner):
                self.owner = owner
                self.completions = Completions(owner)

        self.chat = Chat(self)


def show_current_trr_formula():
    """Show the current TRR formula and what it measures."""
    banner("CURRENT TRR FORMULA - What it measures today")
    src = inspect.getsource(DSPMMemory.stats.fget)
    for i, line in enumerate(src.splitlines(), 1):
        if ('trr' in line.lower() or 'raw_tokens' in line
                or 'context_tokens' in line
                or 'conversation_tokens' in line):
            print(f"  L{i:4d}: {line.rstrip()}")
    print("\n  Current formula: trr = 100 - (context_tokens / raw_tokens) * 100")
    print("  raw_tokens     = sum of patch token costs (EXTRACTED patches)")
    print("  context_tokens = delivered context size")
    print("  >> Bug: should use conversation_tokens (RAW turn text), not raw_tokens")


def show_bug_in_action():
    """Reproduce the bug with a concrete example."""
    banner("BUG REPRO - TRR underreports with current formula")

    m = DSPMMemory(budget=250, llm_client=_FakeLLM())
    t1 = ("We need to build a payment system. Use Stripe. Return JSON. "
          "Use POST /v1/items. Require auth on every endpoint.")
    t2 = ("The webhook timeout must be 30 seconds. Audit retention is 5 years. "
          "Maximum fee is 0.5%. Deadline is Friday.")
    m.add_turn('user', t1)
    m.add_turn('assistant', t2)

    ctx = m.get_context(query='system requirements')
    s = m.stats

    print(f"\n  conversation_tokens : {s['conversation_tokens']}  (raw turn text fed to add_turn)")
    print(f"  patch_tokens (raw)  : {s['patch_tokens']}  (extracted patches)")
    print(f"  context_tokens      : {s['context_tokens']}  (delivered context)")
    print(f"  trr (current)       : {s['trr']}%")

    trr_correct = 100 - int((s['context_tokens'] / max(1, s['conversation_tokens'])) * 100)
    print(f"\n  trr (should be)     : {trr_correct}%  (using conversation_tokens as denominator)")
    print(f"\n  >> Current TRR ({s['trr']}%) understates reduction.")
    print(f"  >> Correct TRR ({trr_correct}%) shows real compression: {s['conversation_tokens']}")
    print(f"     conversation tokens -> {s['context_tokens']} context tokens.")


def show_proposed_fixes():
    """Show three candidate fixes so the user can pick one."""
    banner("PROPOSED FIXES - Three options for the TRR formula")

    print("""
  Option A - Minimal (just swap the denominator):
    trr = 100 - (context_tokens / max(1, conversation_tokens)) * 100
    Rationale: TRR should measure how much we compressed the
    CONVERSATION down to the delivered context. The current
    formula measures compression of extracted patches, which
    is meaningless (patches are already a reduction).

  Option B - Add trr_by_patch as a secondary metric:
    Keep current trr (rename to trr_by_patch for clarity).
    Add trr_by_conversation as the new primary metric.
    Rationale: Both numbers are useful -- patch compression
    shows extractor efficiency, conversation compression
    shows end-to-end system efficiency. Backward compatible.

  Option C - Replace + expose both raw numbers:
    Replace trr with conversation-based formula.
    Expose context_tokens, conversation_tokens, patch_tokens
    separately (already done in v0.1.17).
    Rationale: Simplest, most honest. Breaking change for
    anyone reading stats['trr'] expecting patch-based.
    """)


def check_existing_tests():
    """Find existing tests that assert specific TRR values."""
    banner("EXISTING TESTS - Which tests assert TRR values?")
    try:
        with open('tests/test_memory.py', 'r', encoding='utf-8') as f:
            content = f.read()
        lines = content.splitlines()
        found = False
        for i, line in enumerate(lines, 1):
            if 'trr' in line.lower():
                found = True
                start = max(0, i - 3)
                end = min(len(lines), i + 2)
                print(f"\n  Around line {i}:")
                for j in range(start, end):
                    marker = ">>>" if j == i - 1 else "   "
                    print(f"    {marker} L{j+1:4d}: {lines[j]}")
        if not found:
            print("\n  No tests assert TRR values -- safe to change the formula.")
    except FileNotFoundError:
        print("  tests/test_memory.py not found")


def show_stats_keys():
    """Show all keys in the stats dict so we know what's exposed."""
    banner("STATS DICT - All exposed keys")
    m = DSPMMemory(budget=250, llm_client=_FakeLLM())
    m.add_turn('user', 'Test turn with some content.')
    s = m.stats
    print(f"\n  All stats keys ({len(s)} total):")
    for k, v in sorted(s.items()):
        print(f"    {k:30s} = {v!r}")


def main():
    show_current_trr_formula()
    show_bug_in_action()
    show_proposed_fixes()
    check_existing_tests()
    show_stats_keys()

    print("\n" + "=" * 70)
    print("DONE - paste this entire output back to me for analysis.")
    print("=" * 70)
    print("\nTell me which fix option you prefer:")
    print("  A - minimal swap (breaking change)")
    print("  B - add trr_by_conversation alongside existing trr (backward compat)")
    print("  C - replace + expose both numbers (breaking, most honest)")


if __name__ == '__main__':
    main()