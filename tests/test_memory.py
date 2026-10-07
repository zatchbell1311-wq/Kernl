import json

import pytest

from dspm import DSPMMemory
from dspm.patch import SemanticPatch
from dspm.config import PATCH_TYPES


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


# v0.1.10: replaces MockLLMClientFourTurns (defined since v0.1.0, never
# referenced by any test — dead code). Records the system prompt and user
# message of every LLM call, so the auto-dense and add_document tests can
# verify what the extractor was actually asked to do.
class PromptCapturingClient:
    def __init__(self, content=None):
        self.content = content or json.dumps([
            {"patch_type": "entity", "payload": "Noted.", "patch_id": "x"}
        ])
        self.prompts = []
        self.user_messages = []
        owner = self

        class Completions:
            @staticmethod
            def create(**kwargs):
                owner.prompts.append(kwargs["messages"][0]["content"])
                owner.user_messages.append(kwargs["messages"][1]["content"])
                return MockResponse(owner.content)

        class Chat:
            completions = Completions()

        self.chat = Chat()


# v0.1.11: records the FULL kwargs of every LLM call so tests can assert
# on what the extractor actually sent (max_tokens, reasoning_effort).
class KwargCapturingClient:
    def __init__(self, content=None):
        self.content = content or json.dumps([
            {"patch_type": "entity", "payload": "Stripe.", "patch_id": "x"}
        ])
        self.calls = []
        owner = self

        class Completions:
            @staticmethod
            def create(**kwargs):
                owner.calls.append(kwargs)
                return MockResponse(owner.content)

        class Chat:
            completions = Completions()

        self.chat = Chat()


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
    """v0.1.10: this test previously passed VACUOUSLY — it asserted
    `token_count <= 50 or True` against an `_encode()` method that
    doesn't exist, i.e. it asserted nothing. It now genuinely asserts
    the compressed context fits the budget."""
    from dspm.patch import count_tokens
    llm = MockLLMClient()
    mem = DSPMMemory(llm_client=llm, model='test-model', budget=50)
    mem.add_turn('user', 'Return JSON and use POST /v1/items for creation.')
    context = mem.get_context(query='Return JSON output.')
    assert isinstance(context, str)
    assert count_tokens(context) <= 50, f"budget exceeded: {context!r}"


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


# ── v0.1.10: auto-dense + document ingestion regression tests ────
# The 70%/85% document-comparison failure: extraction ran the standard
# 5-patch funnel because numeric_density() was defined but never called
# (dead code) and the dense flag was manual-only.

def test_auto_dense_triggers_on_fact_heavy_turn():
    """v0.1.10: a fact-heavy turn (numeric density >= threshold) must get
    the dense extraction funnel even when the memory was created with
    dense=False. This is the dead-code fix for the document-test failure."""
    llm = PromptCapturingClient()
    mem = DSPMMemory(budget=250, llm_client=llm, model='test-model', dense=False)
    mem.add_turn('user',
                 'p95 latency 200ms, fee 0.5%, timeout 30 seconds, '
                 'budget 250 tokens, retries 3')
    assert llm.prompts, "no prompt was captured"
    assert 'FACT-DENSE' in llm.prompts[0], \
        f"fact-heavy turn did not trigger dense mode: {llm.prompts[0][:80]!r}"


def test_auto_dense_not_triggered_on_plain_prose():
    """v0.1.10 guard: prose without numbers must NOT trigger the dense
    funnel — auto-dense is density-driven, not always-on."""
    llm = PromptCapturingClient()
    mem = DSPMMemory(budget=250, llm_client=llm, model='test-model', dense=False)
    mem.add_turn('user',
                 'We should probably use a queue here and think about '
                 'the overall design together before deciding anything')
    assert llm.prompts, "no prompt was captured"
    assert 'FACT-DENSE' not in llm.prompts[0], "dense mode triggered on plain prose"


def test_dense_prompt_classifies_config_correctly():
    """v0.1.10: the dense prompt must tell the model that experimental/
    configuration parameters are NOT constraints (live document testing
    misclassified config-table rows as [CON], consuming protected budget
    while real results competed for scraps), and that each numeric
    result gets its OWN patch."""
    from dspm.extractor import _build_system_prompt
    prompt = _build_system_prompt(dense=True)
    assert 'NOT constraints' in prompt, \
        "dense prompt lacks the config-is-not-a-constraint rule"
    assert 'OWN patch' in prompt, \
        "dense prompt lacks the one-stat-per-patch rule"


def test_add_document_chunks_on_sentences():
    """v0.1.10: add_document splits on sentence boundaries — never
    mid-sentence or mid-table-row — and feeds each chunk as one turn."""
    llm = PromptCapturingClient()
    mem = DSPMMemory(budget=250, llm_client=llm, model='test-model')
    doc = " ".join(
        f"Requirement {i} states that the p95 latency must stay "
        f"under {100 + i} milliseconds."
        for i in range(1, 31))
    n_chunks = mem.add_document(doc, chunk_tokens=60)
    assert n_chunks >= 2, f"expected multiple chunks, got {n_chunks}"
    assert mem.turns == n_chunks, "turns must equal chunk count"
    for text in llm.user_messages:
        assert text.rstrip().endswith('.'), \
            f"chunk split mid-sentence: ...{text[-40:]!r}"


def test_recency_prefers_recent_noncritical():
    """v0.1.10: recency actually decays. The old formula exp(-lambda *
    max(0, 0 - turn_index)) was identically 1.0, so W_RECENCY never
    affected ranking. With two equal-cost non-critical patches from
    different turns and a budget that fits only one, the RECENT patch
    must survive. (This test fails on v0.1.9, where the tie kept the
    older patch.)"""
    m = DSPMMemory(budget=12)
    m._merge_patch(SemanticPatch('p0_0', 0, 'entity',
                                 'Alpha service handles user profiles', []))
    m._merge_patch(SemanticPatch('p9_0', 9, 'entity',
                                 'Beta service handles user profiles', []))
    ctx = m.get_context(query='user profiles')
    assert 'Beta' in ctx, f"recent patch lost (recency not decaying?): {ctx!r}"
    assert 'Alpha' not in ctx, f"older patch beat the recent one: {ctx!r}"


# ── v0.1.11: reasoning-model extractor hardening regression tests ─
# From three rounds of live document testing: reasoning models (gpt-oss)
# spend output tokens on chain-of-thought BEFORE the answer, so a
# 2000-token budget could return HTTP 200 with EMPTY content — and
# v0.1.10 only retried on exceptions, so the chunk silently extracted
# 0 patches (the missing-numbers failure mode).

def test_extract_retries_on_empty_content():
    """v0.1.11: HTTP 200 with EMPTY content must retry with backoff.
    v0.1.10 accepted it silently after one call — 0-patch chunks."""
    from dspm.extractor import extract_turn
    calls = {'n': 0}

    class EmptyThenContentClient:
        def __init__(self):
            class C:
                @staticmethod
                def create(**kwargs):
                    calls['n'] += 1
                    if calls['n'] < 3:
                        return MockResponse("")   # reasoning burned the budget
                    return MockResponse(json.dumps(
                        [{'patch_type': 'entity', 'payload': 'Stripe', 'patch_id': 'x'}]))
            class Ch:
                completions = C()
            self.chat = Ch()

    patches = extract_turn(EmptyThenContentClient(), 'test-model', 'We use Stripe.', 0)
    assert calls['n'] == 3, f"empty content must be retried; got {calls['n']} call(s)"
    assert len(patches) == 1 and patches[0].payload == 'Stripe'


def test_extract_all_empty_returns_zero_patches():
    """v0.1.11 guard: if every attempt returns empty content, the turn
    degrades gracefully to 0 patches — add_turn() must never crash on a
    degraded (but successful) response."""
    from dspm.extractor import extract_turn
    calls = {'n': 0}

    class AlwaysEmptyClient:
        def __init__(self):
            class C:
                @staticmethod
                def create(**kwargs):
                    calls['n'] += 1
                    return MockResponse("")
            class Ch:
                completions = C()
            self.chat = Ch()

    patches = extract_turn(AlwaysEmptyClient(), 'test-model', 'We use Stripe.', 0)
    assert calls['n'] == 3
    assert patches == []


def test_reasoning_effort_auto_for_gpt_oss():
    """v0.1.11: gpt-oss models automatically get reasoning_effort="low"
    (extraction is structured output, not a reasoning task) and the
    raised output budget EXTRACT_MAX_TOKENS."""
    from dspm.extractor import extract_turn
    from dspm.config import EXTRACT_MAX_TOKENS
    llm = KwargCapturingClient()
    extract_turn(llm, 'openai/gpt-oss-120b', 'We use Stripe.', 0)
    assert len(llm.calls) == 1
    assert llm.calls[0].get('reasoning_effort') == 'low'
    assert llm.calls[0]['max_tokens'] == EXTRACT_MAX_TOKENS


def test_no_reasoning_effort_for_standard_models():
    """v0.1.11 guard: non-gpt-oss models must NOT receive the
    reasoning_effort kwarg — providers reject unknown parameters with a
    non-retryable 400."""
    from dspm.extractor import extract_turn
    llm = KwargCapturingClient()
    extract_turn(llm, 'llama-3.1-8b-instant', 'We use Stripe.', 0)
    assert len(llm.calls) == 1
    assert 'reasoning_effort' not in llm.calls[0]


def test_reasoning_effort_rejection_falls_back():
    """v0.1.11: a 400 that arrives WHILE reasoning_effort was sent may be
    the provider rejecting the kwarg — it must be stripped and retried,
    not treated as fatal (400 is otherwise non-retryable)."""
    from dspm.extractor import extract_turn
    calls = []

    class RejectingKwargClient:
        def __init__(self):
            class C:
                @staticmethod
                def create(**kwargs):
                    calls.append(kwargs)
                    if len(calls) == 1:
                        e = Exception('Unsupported parameter: reasoning_effort')
                        e.status_code = 400
                        raise e
                    return MockResponse(json.dumps(
                        [{'patch_type': 'entity', 'payload': 'Stripe', 'patch_id': 'x'}]))
            class Ch:
                completions = C()
            self.chat = Ch()

    patches = extract_turn(RejectingKwargClient(), 'openai/gpt-oss-120b', 'We use Stripe.', 0)
    assert len(patches) == 1
    assert len(calls) == 2
    assert calls[0].get('reasoning_effort') == 'low'
    assert 'reasoning_effort' not in calls[1]


def test_custom_client_typeerror_falls_back():
    """v0.1.11: a custom OpenAI-compatible client whose create() doesn't
    accept reasoning_effort raises TypeError — the kwarg is stripped and
    the call retried, keeping the package's any-client-object promise."""
    from dspm.extractor import extract_turn
    calls = []

    class NoKwargClient:
        def __init__(self):
            class C:
                @staticmethod
                def create(**kwargs):
                    calls.append(kwargs)
                    if 'reasoning_effort' in kwargs:
                        raise TypeError(
                            "create() got an unexpected keyword argument 'reasoning_effort'")
                    return MockResponse(json.dumps(
                        [{'patch_type': 'entity', 'payload': 'Stripe', 'patch_id': 'x'}]))
            class Ch:
                completions = C()
            self.chat = Ch()

    patches = extract_turn(NoKwargClient(), 'openai/gpt-oss-20b', 'We use Stripe.', 0)
    assert len(patches) == 1
    assert len(calls) == 2
    assert 'reasoning_effort' not in calls[1]