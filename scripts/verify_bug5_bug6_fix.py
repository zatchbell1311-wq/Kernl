"""
Standalone verification for Bug #5 and Bug #6 fixes (no LLM, no pytest).

Run after copying the updated files:

    python scripts/verify_bug5_bug6_fix.py
"""

import json
from dspm import DSPMMemory
from dspm.patch import count_tokens


def banner(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ── Bug #5: truncation handling ─────────────────────────────────

banner("BUG #5 - Truncation classification + no-wasted-retry")

from dspm.extractor import extract_turn


class LengthMockResponse:
    def __init__(self):
        self.choices = [type('Choice', (), {
            'message': type('Message', (), {'content': ''}),
            'finish_reason': 'length',
        })]


class TruncatingClient:
    """Simulates gpt-oss-20b burning output budget on chain-of-thought."""
    def __init__(self):
        class C:
            @staticmethod
            def create(**kwargs):
                return LengthMockResponse()
        class Ch:
            completions = C()
        self.chat = Ch()


telemetry = {}
patches = extract_turn(TruncatingClient(), 'openai/gpt-oss-20b',
                       'We use Stripe.', 0, telemetry=telemetry)
print(f"  patches             : {len(patches)}")
print(f"  telemetry           : {telemetry}")
print(f"  expected            : llm_calls=1, llm_length_finish=1, "
      f"llm_empty=0, llm_retries=0, llm_fallbacks=1")
assert telemetry.get('llm_calls') == 1, \
    f"FAIL: should make only 1 call (no retry): {telemetry}"
assert telemetry.get('llm_length_finish') == 1, \
    f"FAIL: should bump llm_length_finish: {telemetry}"
assert telemetry.get('llm_empty', 0) == 0, \
    f"FAIL: should NOT bump llm_empty: {telemetry}"
assert telemetry.get('llm_retries', 0) == 0, \
    f"FAIL: should NOT retry (deterministic): {telemetry}"
print("  PASS: truncation classified correctly, no wasted retry")


# Bug #5 guard: genuine empty content still retries
banner("BUG #5 GUARD - Genuine empty content still retries")


class MockResponse:
    def __init__(self, content, finish_reason="stop"):
        self.choices = [type('Choice', (), {
            'message': type('Message', (), {'content': content}),
            'finish_reason': finish_reason,
        })]


class EmptyThenOK:
    def __init__(self):
        self.calls = 0
        owner = self

        class C:
            @staticmethod
            def create(**kwargs):
                owner.calls += 1
                if owner.calls == 1:
                    return MockResponse("")
                return MockResponse(json.dumps(
                    [{'patch_type': 'entity',
                      'payload': 'Stripe', 'patch_id': 'x'}]))
        class Ch:
            completions = C()
        self.chat = Ch()


telemetry = {}
patches = extract_turn(EmptyThenOK(), 'openai/gpt-oss-20b',
                       'We use Stripe.', 0, telemetry=telemetry)
print(f"  patches             : {len(patches)}")
print(f"  telemetry           : {telemetry}")
assert len(patches) == 1, f"FAIL: should retry and succeed: {patches}"
assert telemetry.get('llm_empty', 0) == 1, f"FAIL: should bump llm_empty: {telemetry}"
assert telemetry.get('llm_retries', 0) == 1, f"FAIL: should retry once: {telemetry}"
print("  PASS: genuine empty content still retries")


# ── Bug #6: TRR by conversation ──────────────────────────────────

banner("BUG #6 - trr_by_conversation uses correct denominator")


class FakeLLM:
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


m = DSPMMemory(budget=250, llm_client=FakeLLM())
t1 = ("We need to build a payment system. Use Stripe. Return JSON. "
      "Use POST /v1/items. Require auth on every endpoint.")
t2 = ("The webhook timeout must be 30 seconds. Audit retention is 5 years. "
      "Maximum fee is 0.5%. Deadline is Friday.")
m.add_turn('user', t1)
m.add_turn('assistant', t2)
m.get_context(query='system requirements')
s = m.stats

print(f"  conversation_tokens : {s['conversation_tokens']}")
print(f"  patch_tokens (raw)  : {s['patch_tokens']}")
print(f"  context_tokens      : {s['context_tokens']}")
print(f"  trr (patch-based)   : {s['trr']}%")
print(f"  trr_by_conversation : {s['trr_by_conversation']}%")

# Both keys must be present (Option B - backward compat)
assert 'trr' in s, "FAIL: trr key missing"
assert 'trr_by_conversation' in s, "FAIL: trr_by_conversation key missing"

# trr_by_conversation must use conversation_tokens
expected = 100 - int(
    (s['context_tokens'] / max(1, s['conversation_tokens'])) * 100)
assert s['trr_by_conversation'] == expected, \
    f"FAIL: trr_by_conversation wrong: expected {expected}, got {s['trr_by_conversation']}"

# When patch_tokens != conversation_tokens, the two TRRs must differ
if s['patch_tokens'] != s['conversation_tokens']:
    assert s['trr'] != s['trr_by_conversation'], \
        f"FAIL: trr and trr_by_conversation should differ: {s}"

print("  PASS: trr_by_conversation uses correct denominator")
print(f"  >> Old TRR ({s['trr']}%) understated reduction.")
print(f"  >> New TRR ({s['trr_by_conversation']}%) shows real compression.")


print("\n" + "=" * 70)
print("ALL VERIFICATIONS PASSED")
print("=" * 70)
print("\nNext steps:")
print("  1. Copy the updated files to your repo:")
print("     memory.py       -> src/dspm/memory.py")
print("     extractor.py    -> src/dspm/extractor.py")
print("     test_memory.py  -> tests/test_memory.py")
print("  2. Run: python -m pytest tests -q --basetemp=\"$env:TEMP\\dspm-pytest\"")
print("  3. Expected: 179 passed (162 original + 17 new)")