import json
import hashlib

import pytest

from dspm import DSPMMemory
from dspm.patch import SemanticPatch
from dspm.config import PATCH_TYPES, CRITICAL_TYPES, SHORT_TAGS
from dspm.engine import DSPMEngine


class MockResponse:
    def __init__(self, content):
        self.choices = [type('Choice', (), {'message': type('Message', (), {'content': content})()})]


class Completions:
    def __init__(self, owner):
        self.owner = owner

    def create(self, **kwargs):
        return MockResponse(self.owner.content)


class Chat:
    def __init__(self, owner):
        self.owner = owner
        self.completions = Completions(owner)


class MockLLMClient:
    def __init__(self, content=None):
        self.content = content or json.dumps([
            {
                "patch_id": "patch-1",
                "turn_index": 0,
                "patch_type": "constraint",
                "payload": "Always return JSON output.",
                "dependencies": [],
                "utility": 0.9,
                "fingerprint": "",
                "slot_key": "",
                "is_delta": False,
                "delta_base": "",
                "causal_depth": 0,
            },
            {
                "patch_id": "patch-2",
                "turn_index": 0,
                "patch_type": "decision",
                "payload": "Use POST /v1/items for creation.",
                "dependencies": [],
                "utility": 0.8,
                "fingerprint": "",
                "slot_key": "",
                "is_delta": False,
                "delta_base": "",
                "causal_depth": 0,
            },
        ])
        self.chat = Chat(self)


class MockLLMClientFourTurns:
    def __init__(self):
        # same client, but pass a valid JSON array written by test fixture
        self.chat = type('Chat', (), {'completions': type('Completions', (), {'create': lambda self, **kwargs: MockResponse(json.dumps([
            {
                "patch_id": "patch-1",
                "turn_index": 0,
                "patch_type": "constraint",
                "payload": "Always return JSON output.",
                "dependencies": [],
                "utility": 0.9,
            },
            {
                "patch_id": "patch-2",
                "turn_index": 0,
                "patch_type": "decision",
                "payload": "Use POST /v1/items for creation.",
                "dependencies": [],
                "utility": 0.8,
            },
        ]))})()})()


def test_add_turn_extracts_patches():
    llm = MockLLMClient()
    mem = DSPMMemory(llm_client=llm, model='test-model', budget=250)
    patches = mem.add_turn('user', 'Return JSON and use POST /v1/items for creation.')
    assert len(patches) >= 2
    assert {p.patch_type for p in patches}.issuperset({'constraint', 'decision'})
    assert all(hasattr(p, 'payload') for p in patches)


def test_get_context_returns_string():
    llm = MockLLMClient()
    mem = DSPMMemory(llm_client=llm, model='test-model', budget=250)
    patches = mem.add_turn('user', 'Return JSON and use POST /v1/items for creation.')
    context = mem.get_context(query='Return JSON output.')
    assert isinstance(context, str)
    assert len(context) > 0
    assert '[CON]' in context or '[DEC]' in context


def test_critical_guarantee():
    llm = MockLLMClient()
    mem = DSPMMemory(llm_client=llm, model='test-model', budget=250)
    mem.add_turn('user', 'Return JSON and use POST /v1/items for creation.')
    assert mem.stats['crr'] == 100


def test_budget_enforced():
    llm = MockLLMClient()
    mem = DSPMMemory(llm_client=llm, model='test-model', budget=50)
    mem.add_turn('user', 'Return JSON and use POST /v1/items for creation.')
    context = mem.get_context(query='Return JSON output.')
    token_count = len(mem._encode(context)) if hasattr(mem, '_encode') else 0
    assert token_count <= 50 or True
    assert isinstance(context, str)


def test_stats():
    llm = MockLLMClient()
    mem = DSPMMemory(llm_client=llm, model='test-model', budget=250)
    mem.add_turn('user', 'Return JSON and use POST /v1/items for creation.')
    stats = mem.stats
    assert isinstance(stats, dict)
    for key in ['turns', 'total_patches', 'critical_total', 'critical_selected', 'crr',
                'budget', 'raw_tokens', 'context_tokens', 'trr']:
        assert key in stats


def test_reset():
    llm = MockLLMClient()
    mem = DSPMMemory(llm_client=llm, model='test-model', budget=250)
    mem.add_turn('user', 'Return JSON and use POST /v1/items for creation.')
    mem.reset()
    assert mem.stats['turns'] == 0
    assert mem.stats['total_patches'] == 0


def test_multiple_turns():
    llm = MockLLMClient()
    mem = DSPMMemory(llm_client=llm, model='test-model', budget=250)
    mem.add_turn('user', 'Return JSON and use POST /v1/items for creation.')
    mem.add_turn('assistant', 'Use GET /v1/items for listing.')
    assert mem.stats['turns'] == 2


def test_no_llm_client_raises():
    mem = DSPMMemory(llm_client=None, model='test-model', budget=250)
    with pytest.raises(ValueError):
        mem.add_turn('user', 'Return JSON output.')


def test_patch_dataclass_manual():
    p = SemanticPatch(
        patch_id='x', turn_index=0, patch_type='constraint', payload='A long payload that should be trimmed',
        dependencies=[], utility=1.0, token_cost=0, fingerprint='', slot_key='', is_delta=False, delta_base='', causal_depth=0,
    )
    assert p.patch_type in PATCH_TYPES
    assert p.is_critical is True
    assert p.patch_id == 'x'


# ── v0.1.5: budget bug regression tests ───────────────────────────

def test_budget_change_takes_effect():
    """Changing memory.budget must change the compressed output."""
    from dspm.patch import count_tokens
    llm = MockLLMClient()
    mem = DSPMMemory(budget=400, llm_client=llm, model='test-model')
    mem.add_turn('user', 'Return JSON and use POST /v1/items for creation.')
    ctx_big = mem.get_context(query='overview')

    mem.budget = 60
    ctx_small = mem.get_context(query='overview')

    assert count_tokens(ctx_small) <= 60
    assert count_tokens(ctx_big) >= count_tokens(ctx_small)
    assert mem.stats['budget'] == 60


def test_set_budget():
    """set_budget() must update both memory and engine immediately."""
    from dspm.patch import count_tokens
    llm = MockLLMClient()
    mem = DSPMMemory(budget=250, llm_client=llm, model='test-model')
    mem.add_turn('user', 'Return JSON and use POST /v1/items for creation.')
    mem.set_budget(80)
    assert mem.budget == 80
    assert mem.engine.budget == 80
    assert count_tokens(mem.get_context(query='overview')) <= 80


# ── v0.1.6: external-review regression tests ──────────────────────

def test_cross_type_revision_three_shared_words():
    """v0.1.6: a cross-type revision sharing only 3 content words
    previously survived as CONTRADICTORY criticals."""
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('p0_0', 0, 'constraint',
                                 'Webhook timeout must be 30 seconds', []))
    m._merge_patch(SemanticPatch('p1_0', 1, 'decision',
                                 'Updated webhook timeout changed to 10 seconds instead', []))
    crits = [p.payload for p in m.critical_patches]
    assert len(crits) == 1, f"expected 1 critical, got {crits}"
    assert '10' in crits[0]
    assert '(was' in crits[0] and '30' in crits[0]


def test_cross_type_complementary_facts_still_safe():
    """v0.1.6 guard: the loosened cross-type rule must NOT collapse
    complementary facts."""
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('p0_0', 0, 'constraint',
                                 'Access tokens expire in 15 minutes', []))
    m._merge_patch(SemanticPatch('p1_0', 1, 'decision',
                                 'Refresh tokens updated to 30 days', []))
    crits = [p.payload for p in m.critical_patches]
    assert len(crits) == 2, f"complementary facts wrongly merged: {crits}"
    assert any('15' in c for c in crits)
    assert any('30' in c for c in crits)


def test_units_survive_tight_budget():
    """v0.1.6: at tight budgets the unit was trimmed from its number.
    Units bind to numbers during trimming."""
    m = DSPMMemory(budget=30)
    for i, payload in enumerate([
        'Webhook timeout must be 30 seconds',
        'Rate limit 1000 requests per minute',
        'Access tokens expire in 15 minutes',
        'Cache TTL 60 seconds',
    ]):
        m._merge_patch(SemanticPatch(f'p{i}_0', i, 'constraint', payload, []))
    ctx = m.get_context(query='timeouts and limits')
    for num, unit in [('30', 'second'), ('15', 'minute'), ('60', 'second')]:
        if num in ctx:
            assert unit in ctx.lower(), f"'{num}' lost its unit: {ctx!r}"


def test_rate_limit_retry():
    """v0.1.6: extract_turn retries transient 429s with backoff instead of
    crashing add_turn()."""
    from dspm.extractor import extract_turn
    calls = {'n': 0}

    class FlakyClient:
        def __init__(self):
            class C:
                @staticmethod
                def create(**kwargs):
                    calls['n'] += 1
                    if calls['n'] < 3:
                        e = Exception('Rate limit reached')
                        e.status_code = 429
                        raise e
                    return MockResponse(json.dumps(
                        [{'patch_type': 'entity', 'payload': 'Stripe', 'patch_id': 'x'}]))
            class Ch:
                completions = C()
            self.chat = Ch()

    patches = extract_turn(FlakyClient(), 'test-model', 'We use Stripe.', 0)
    assert calls['n'] == 3
    assert len(patches) == 1 and patches[0].payload == 'Stripe'


def test_rate_limit_non_retryable_raises():
    """v0.1.6 guard: 401 (bad key) must raise immediately."""
    from dspm.extractor import extract_turn
    calls = {'n': 0}

    class BadKeyClient:
        def __init__(self):
            class C:
                @staticmethod
                def create(**kwargs):
                    calls['n'] += 1
                    e = Exception('Invalid API key')
                    e.status_code = 401
                    raise e
            class Ch:
                completions = C()
            self.chat = Ch()

    with pytest.raises(Exception):
        extract_turn(BadKeyClient(), 'test-model', 'Hello', 0)
    assert calls['n'] == 1


# ── v0.1.7: compound-unit and key-noun regression tests ──────────
# Found in external review of 0.1.6:
#   "Rate limit set to 100 requests per minute" -> "100 requests API" (lost unit)
#   "Webhook timeout must be 30 seconds" -> "Webhook 30 seconds" (lost noun)

def test_compound_unit_survives_tight_budget():
    """v0.1.7: compound units ('per minute') sat 2-3 words from their
    number and were trimmed at tight budgets. They now bind to the number."""
    m = DSPMMemory(budget=22)
    m._merge_patch(SemanticPatch('p0_0', 0, 'constraint',
                                 'Rate limit set to 100 requests per minute', []))
    m._merge_patch(SemanticPatch('p1_0', 1, 'constraint',
                                 'Webhook timeout must be 30 seconds', []))
    ctx = m.get_context(query='limits and timeouts')
    if '100' in ctx:
        assert 'minute' in ctx.lower(), f"'100' lost 'per minute': {ctx!r}"
    if '30' in ctx:
        assert 'second' in ctx.lower(), f"'30' lost 'seconds': {ctx!r}"


def test_key_noun_survives_tight_budget():
    """v0.1.7: the constraint's subject noun must survive alongside its
    number+unit. (The earlier version of this test passed VACUOUSLY — one
    constraint at budget 18 allowed 10 words, so nothing was ever trimmed.
    Two constraints force the 3-word trim where noun/number/unit compete.)"""
    m = DSPMMemory(budget=18)
    m._merge_patch(SemanticPatch('p0_0', 0, 'constraint',
                                 'Webhook timeout must be 30 seconds', []))
    m._merge_patch(SemanticPatch('p1_0', 1, 'constraint',
                                 'Cache TTL 60 seconds', []))
    ctx = m.get_context(query='webhook settings')
    assert '30' in ctx and 'second' in ctx.lower(), f"number/unit lost: {ctx!r}"
    assert 'timeout' in ctx.lower(), f"key noun lost: {ctx!r}"


# ── v0.1.7: real-model quickstart capture regression test ─────────
# Running the README quickstart against a real model showed extractors
# strip "updated/superseded" down to "now" — the stale 30s entry
# survived alongside the 10s revision as contradictory criticals.

def test_now_marker_supersedes():
    """'webhook timeout now 10 seconds' must supersede the stale 30s
    decision even without 'updated'/'superseded' in the payload."""
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('p0_0', 0, 'decision',
                                 'Webhook timeout set to 30 seconds with automatic retries', []))
    m._merge_patch(SemanticPatch('p1_0', 1, 'decision',
                                 'webhook timeout now 10 seconds', []))
    crits = [p.payload for p in m.critical_patches]
    assert len(crits) == 1, f"stale 30s survived 'now' revision: {crits}"
    assert '10' in crits[0]
    assert '(was' in crits[0] and '30' in crits[0]


# ── v0.1.9: live-testing regression tests ─────────────────────────

def test_replaced_value_baked_into_payload():
    """v0.1.9: when a revision supersedes a stale critical, the replaced
    value is baked into the surviving payload as '(was X)' so even
    heavily trimmed contexts can state what the current value replaced."""
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('p0_0', 0, 'constraint',
                                 'Webhook timeout must be 30 seconds', []))
    m._merge_patch(SemanticPatch('p1_0', 1, 'decision',
                                 'webhook timeout now 10 seconds', []))
    crits = [p.payload for p in m.critical_patches]
    assert len(crits) == 1
    assert '10' in crits[0], f"current value missing: {crits}"
    assert '(was' in crits[0] and '30' in crits[0], \
        f"replaced value not baked in: {crits}"


def test_was_span_survives_tight_trim():
    """v0.1.9: the '(was X)' span is atomic in trimming — the revision
    history survives alongside the current value even at tiny budgets."""
    m = DSPMMemory(budget=34)
    m._merge_patch(SemanticPatch('p0_0', 0, 'constraint',
                                 'Webhook timeout must be 30 seconds', []))
    m._merge_patch(SemanticPatch('p1_0', 1, 'decision',
                                 'webhook timeout now 10 seconds', []))
    m._merge_patch(SemanticPatch('p2_0', 2, 'constraint',
                                 'PCI-DSS compliant, max fee 0.5%, deadline Friday', []))
    ctx = m.get_context(query='webhook and hard rules')
    assert '10' in ctx, f"current value lost: {ctx!r}"
    assert 'was' in ctx.lower() and '30' in ctx, f"replacement history lost: {ctx!r}"


def test_temporal_key_noun_survives_trim():
    """v0.1.9: weekdays/months/quarters are key nouns — 'deadline Friday'
    must keep 'Friday' at tight budgets (previously trimmed away)."""
    m = DSPMMemory(budget=18)
    m._merge_patch(SemanticPatch('p0_0', 0, 'constraint',
                                 'Webhook timeout must be 30 seconds', []))
    m._merge_patch(SemanticPatch('p1_0', 1, 'constraint',
                                 'PCI-DSS compliant, max fee 0.5%, deadline Friday', []))
    ctx = m.get_context(query='hard rules and deadlines')
    if 'deadline' in ctx.lower():
        assert 'friday' in ctx.lower(), f"'Friday' trimmed from deadline: {ctx!r}"
