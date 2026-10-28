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


# ── v0.1.12: revision-story regression tests ─────────────────────
# From the live webhook rematch (conversation mode, 70% removal):
#   Bug 1 — the (was X) baking took the stale patch's FIRST numeric
#   word, which is its CURRENT value when the stale patch is itself a
#   revision: "… 10 seconds (was 30)" superseded by a restated "10
#   seconds" baked a wrong "(was 10)" instead of carrying "(was 30)".
#   Bug 2 — at starvation budgets (8 criticals, 59 tokens → 2-word
#   allowances) the trim ladder kept "10 seconds" and dropped the
#   was-span entirely; the answer model then fabricated a replacement.

def test_revision_of_revision_keeps_history():
    """v0.1.12: a revision-of-a-revision must bake the HISTORY, not the
    stale patch's current value. The live trace: 30 → '10 (was 30)' →
    restated 10 baked '(was 10)' — wrong. It must carry '(was 30)'."""
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('p0_0', 0, 'decision',
                                 'Webhook timeout set to 30 seconds', []))
    m._merge_patch(SemanticPatch('p1_0', 1, 'decision',
                                 'webhook timeout now 10 seconds', []))
    first = [p.payload for p in m.critical_patches]
    assert len(first) == 1
    assert '(was 30)' in first[0], f"first revision baked wrong: {first[0]!r}"
    # a restatement of the SAME revision arrives (assistant confirming)
    m._merge_patch(SemanticPatch('p2_0', 2, 'decision',
                                 'Webhook Timeout updated to 10 seconds, '
                                 'superseding previous 30 seconds value', []))
    crits = [p.payload for p in m.critical_patches]
    assert len(crits) == 1, f"restatement should supersede: {crits}"
    assert '10' in crits[0]
    assert '(was 10)' not in crits[0], \
        f"baked the stale's CURRENT value, not its history: {crits[0]!r}"
    assert 'was' in crits[0].lower() and '30' in crits[0], \
        f"revision history lost: {crits[0]!r}"


def test_revision_story_survives_critical_starvation():
    """v0.1.12: the live rematch scenario — 8 criticals at a 59-token
    budget give 2-word trim allowances. The ladder kept '10 seconds' and
    dropped the '(was 30)' span, so the replaced value vanished from the
    context and the answer model fabricated a replacement. The story
    form '10 (was 30)' must now survive."""
    m = DSPMMemory(budget=59)
    m._merge_patch(SemanticPatch('p0_0', 0, 'constraint',
                                 'Payments must be PCI compliant', []))
    m._merge_patch(SemanticPatch('p0_1', 0, 'constraint',
                                 'Max fee 0.5%', []))
    m._merge_patch(SemanticPatch('p2_0', 2, 'decision',
                                 'Webhook timeout set to 30 seconds', []))
    m._merge_patch(SemanticPatch('p3_0', 3, 'decision',
                                 'webhook timeout now 10 seconds', []))
    m._merge_patch(SemanticPatch('p6_0', 6, 'constraint',
                                 'Audit retention 5 years', []))
    m._merge_patch(SemanticPatch('p6_1', 6, 'constraint',
                                 'Dashboard loads in 2 seconds', []))
    m._merge_patch(SemanticPatch('p1_0', 1, 'decision',
                                 'Use Stripe for payments', []))
    m._merge_patch(SemanticPatch('p7_0', 7, 'decision',
                                 'Redis caching with CDN', []))
    m._merge_patch(SemanticPatch('p8_0', 8, 'decision',
                                 'Tokenized ASV storage', []))
    assert len(m.critical_patches) == 8, \
        f"expected 8 criticals (webhook pair collapsed): {len(m.critical_patches)}"
    ctx = m.get_context(query='webhook timeout')
    assert '10' in ctx, f"current value lost: {ctx!r}"
    assert 'was' in ctx.lower() and '30' in ctx, \
        f"revision story lost under starvation: {ctx!r}"


def test_revision_chain_bakes_full_history():
    """v0.1.12: a genuine new revision over a previously-revised patch
    must chain the FULL lineage: 60 -> 300 -> 120 bakes
    '(was 300 then 60)', not just the last replaced value."""
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('p0_0', 0, 'decision',
                                 'Cache TTL set to 60 seconds', []))
    m._merge_patch(SemanticPatch('p1_0', 1, 'decision',
                                 'cache TTL now 300 seconds', []))
    m._merge_patch(SemanticPatch('p2_0', 2, 'decision',
                                 'cache TTL updated to 120 seconds', []))
    crits = [p.payload for p in m.critical_patches]
    assert len(crits) == 1, f"chain should collapse to one critical: {crits}"
    assert '120' in crits[0], f"current value missing: {crits[0]!r}"
    assert '300' in crits[0] and '60' in crits[0], \
        f"full revision lineage not chained: {crits[0]!r}"


# ── v0.1.13: value-directive classification regression test ──────
# From the live webhook rematch (round 2): 'webhook timeout set to 30
# seconds' — a value DIRECTIVE — was classified as [CODE], because the
# v0.1.10 config-≠-constraint rule (written from document-mode
# evidence) pattern-matched it as a 'configuration parameter'. The 30
# then never entered the critical lineage, so the revision history had
# nothing to bake. Auto-dense routes number-bearing conversation turns
# through the dense prompt — both prompts must be safe in both modes.

def test_prompt_value_directives_are_decisions():
    """v0.1.13: both prompts must classify speaker value directives
    ('set the timeout to 30 seconds', 'updated to 10 seconds') as
    DECISIONS, not code — even when the parameter sounds like
    configuration. The dense prompt additionally names the
    directive-vs-description distinction, and the document-mode config
    rule must survive the refinement."""
    from dspm.extractor import _build_system_prompt
    for dense in (True, False):
        prompt = _build_system_prompt(dense=dense)
        assert 'is a DECISION' in prompt, \
            f"dense={dense}: value directives must be classified as decisions"
    dense_prompt = _build_system_prompt(dense=True)
    assert 'DIRECTIVES' in dense_prompt, \
        "dense prompt lacks the directive-vs-description rule"
    assert 'NOT constraints' in dense_prompt, \
        "dense prompt lost the config-description rule in the refinement"


# ── v0.1.14: readable trimming + baking dedup regression tests ───
# From the baseline experiment (Oct 11): DSPM tied a rolling summary
# at equal budgets (78% each) — the guarantee works, nothing is lost,
# but the 2-3 word payloads are too fragmented for a judge to
# recognize ("encryption AES-256 at" is unreadable). Key nouns are
# now priority 50 (was 30), a post-fill check guarantees the
# constraint's subject survives alongside its number, and (was X)
# chains are deduplicated.

def test_key_noun_outranks_number_adjacent():
    """v0.1.14: key nouns (priority 50) now outrank number-adjacent
    words (priority 40) — 'rate 100' is more readable than
    '100 requests' when the budget forces a choice. On v0.1.13,
    'requests' (40) beat 'rate'/'limit' (30); now they don't."""
    from dspm.engine import DSPMEngine
    engine = DSPMEngine()
    result = engine._trim_numeric_first(
        "rate limit 100 requests per minute", 2)
    # v0.1.13: "100 requests" (requests at 40 > rate at 30)
    # v0.1.14: "rate 100" or "limit 100" (rate/limit at 50 > requests at 40)
    assert ('rate' in result or 'limit' in result), \
        f"key noun lost to number-adjacent word: {result!r}"
    assert '100' in result, f"number lost: {result!r}"


def test_baking_chain_no_consecutive_duplicates():
    """v0.1.14: the (was X) chain never contains consecutive duplicate
    values — the baseline experiment showed '(was 20 then 20)'. Chains
    are deduplicated in _replaced_value and in the merge join."""
    from dspm.memory import _dedup_chain
    # unit test: consecutive duplicates are removed
    assert _dedup_chain(["20", "20"]) == ["20"]
    assert _dedup_chain(["300", "300", "60"]) == ["300", "60"]
    # non-consecutive duplicates are preserved
    assert _dedup_chain(["20", "30", "20"]) == ["20", "30", "20"]
    # behavioral test: a revision chain produces no consecutive duplicates
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('p0_0', 0, 'decision',
                                 'Cache TTL set to 60 seconds', []))
    m._merge_patch(SemanticPatch('p1_0', 1, 'decision',
                                 'cache TTL now 300 seconds', []))
    m._merge_patch(SemanticPatch('p2_0', 2, 'decision',
                                 'cache TTL updated to 120 seconds', []))
    crits = [p.payload for p in m.critical_patches]
    assert len(crits) == 1
    import re as _re
    m2 = _re.search(r"\(was ([^)]+)\)", crits[0])
    if m2:
        chain = m2.group(1).split(" then ")
        for i in range(1, len(chain)):
            assert chain[i] != chain[i - 1], \
                f"consecutive duplicate in chain: {chain}"


# ── v0.1.15: restatement collapse regression tests ───────────────
# From the baseline experiments (Oct 11): same-type restatements (user
# states, assistant confirms) produced duplicate criticals competing
# for budget — "audit retention 5 years" AND "audit logs stored for
# exactly 5 years" coexisted as two [CON] patches. Rule (b2) collapses
# them. Rule (b) gains the same unit check (fixes the latent
# "100 per minute" vs "100 per hour" false-positive from the audit).

def test_restatement_collapse():
    """v0.1.15: user states a fact, assistant confirms it with different
    phrasing — same type, same value, same unit → ONE critical, not
    two duplicates eating budget. (Fails on v0.1.14: both coexist.)"""
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('p0_0', 0, 'constraint',
                                 'Audit retention set to 5 years', []))
    m._merge_patch(SemanticPatch('p1_0', 1, 'constraint',
                                 'Audit logs stored for exactly 5 years', []))
    crits = [p.payload for p in m.critical_patches]
    assert len(crits) == 1, \
        f"restatement not collapsed (v0.1.15 fix): {crits}"
    assert '5' in crits[0], f"value missing: {crits}"
    assert '(was' not in crits[0], \
        f"no baking for same-value restatement: {crits}"


def test_different_units_not_collapsed():
    """v0.1.15 guard: same number + same subject but different
    time/size units are complementary facts, not restatements.
    '100 per minute' ≠ '100 per hour'."""
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('p0_0', 0, 'constraint',
                                 'Rate limit 100 requests per minute', []))
    m._merge_patch(SemanticPatch('p1_0', 1, 'constraint',
                                 'Rate cap 100 requests per hour', []))
    crits = [p.payload for p in m.critical_patches]
    assert len(crits) == 2, \
        f"different units wrongly collapsed: {crits}"
    assert any('minute' in c.lower() for c in crits)
    assert any('hour' in c.lower() for c in crits)


def test_different_subjects_not_collapsed():
    """v0.1.15 guard: same number + same head word but different
    qualifiers (cloud vs training) are different facts with only 2
    shared content words — below the ≥3 threshold."""
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('p0_0', 0, 'constraint',
                                 'Budget 100 for cloud', []))
    m._merge_patch(SemanticPatch('p1_0', 1, 'constraint',
                                 'Budget 100 for training', []))
    crits = [p.payload for p in m.critical_patches]
    assert len(crits) == 2, \
        f"different subjects wrongly collapsed: {crits}"
    assert any('cloud' in c.lower() for c in crits)
    assert any('training' in c.lower() for c in crits)


# ── v0.1.16: polarity-preserving trimming + bound-aware collapse ─
# Reviewer finding (post-v0.1.15): "100% CRR means the memory items
# remain, but important words like 'not' or 'less than' could still
# be lost." All six confirmed by scripts/repro_polarity_bug.py on
# v0.1.15: the trim ladder scored negations and comparators as filler
# (10), so "dashboard must load in under 2 seconds" trimmed to
# "dashboard 2 seconds", "Payments must not be stored in plaintext"
# lost "not" at the 2-word floor (meaning INVERTED), and (b2)
# collapsed "retention must not exceed 5 years" (upper) with
# "retention at least 5 years" (lower) — one bound silently deleted.

def test_negation_survives_starvation_trim():
    """v0.1.16: 'not' must survive a 3-word trim together with the word
    it negates — 'Payments must not' bans nothing, and 'Payments
    stored' bans the opposite of the constraint."""
    from dspm.engine import DSPMEngine
    engine = DSPMEngine()
    result = engine._trim_numeric_first(
        "Payments must not be stored in plaintext", 3)
    assert "not" in result.lower(), f"negation lost: {result!r}"
    assert "stored" in result.lower() or "plaintext" in result.lower(), \
        f"negated word lost, negation is noise: {result!r}"


def test_negation_binds_at_two_word_floor():
    """v0.1.16: at the 2-word floor the negation binds to its verb —
    'never stored' — instead of losing 'never' entirely (v0.1.15 gave
    'Cards must', inverting the constraint)."""
    from dspm.engine import DSPMEngine
    engine = DSPMEngine()
    result = engine._trim_numeric_first(
        "Cards must never be stored unencrypted", 2)
    assert "never" in result.lower(), f"negation lost at floor: {result!r}"
    assert "stored" in result.lower(), f"negated word lost: {result!r}"


def test_comparator_less_than_survives_trim():
    """v0.1.16: 'less than' binds to its number — 'latency 100
    milliseconds' from 'latency less than 100 milliseconds' is a
    different constraint (upper bound became an exact value)."""
    from dspm.engine import DSPMEngine
    engine = DSPMEngine()
    result = engine._trim_numeric_first(
        "latency less than 100 milliseconds", 3)
    assert "less than" in result.lower(), f"comparator lost: {result!r}"
    assert "100" in result, f"value lost: {result!r}"


def test_comparator_under_survives_trim():
    """v0.1.16: 'under 2 seconds' is one atomic span — 'dashboard 2
    seconds' reads as 'loads IN 2 seconds', not 'under'."""
    from dspm.engine import DSPMEngine
    engine = DSPMEngine()
    result = engine._trim_numeric_first(
        "dashboard must load in under 2 seconds", 3)
    assert "under" in result.lower(), f"comparator lost: {result!r}"
    assert "2" in result and "second" in result.lower(), \
        f"value/unit lost: {result!r}"


def test_comparator_degrades_gracefully():
    """v0.1.16: when the full span cannot fit, a comparator span
    degrades to comparator+number (direction outranks the unit) —
    never to a bare number with its direction silently gone."""
    from dspm.engine import DSPMEngine
    engine = DSPMEngine()
    result = engine._trim_numeric_first(
        "dashboard must load in under 2 seconds", 2)
    assert "under" in result.lower() and "2" in result, \
        f"comparator dropped entirely at 2 words: {result!r}"


def test_no_more_than_span_survives():
    """v0.1.16: 'no more than 3' is one span. v0.1.15 produced the
    garbled 'no more 3 retries' ('than' was filler)."""
    from dspm.engine import DSPMEngine
    engine = DSPMEngine()
    result = engine._trim_numeric_first(
        "allow no more than 3 retries", 4)
    assert "no more than" in result.lower() and "3" in result, \
        f"'no more than' span broken: {result!r}"


def test_bound_class_classification():
    """v0.1.16 unit test: _bound_class reads bound direction correctly,
    including 'not exceed' (upper) vs bare 'exceed' (lower)."""
    from dspm.memory import _bound_class
    assert _bound_class("retention must not exceed 5 years") == "upper"
    assert _bound_class("retention no more than 5 years") == "upper"
    assert _bound_class("retention at least 5 years") == "lower"
    assert _bound_class("latency must exceed 100 ms") == "lower"
    assert _bound_class("retention exactly 5 years") == "exact"
    assert _bound_class("retention 5 years") == "plain"
    assert _bound_class("must not be stored") == "neg"


def test_contradictory_bounds_not_collapsed():
    """v0.1.16: an upper bound and a lower bound on the same slot, same
    number, same unit are DIFFERENT constraints — v0.1.15's (b2)
    collapsed them into one critical, silently deleting a bound."""
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('p0_0', 0, 'constraint',
                                 'Audit retention must not exceed 5 years', []))
    m._merge_patch(SemanticPatch('p1_0', 1, 'constraint',
                                 'Audit retention at least 5 years', []))
    crits = [p.payload for p in m.critical_patches]
    assert len(crits) == 2, \
        f"contradictory bounds collapsed (v0.1.16 fix): {crits}"
    assert any('exceed' in c.lower() for c in crits)
    assert any('least' in c.lower() for c in crits)


def test_same_direction_bounds_still_collapse():
    """v0.1.16 guard: restatements with the SAME bound direction still
    collapse — 'no more than 5 years' and 'must not exceed 5 years'
    are the same fact."""
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('p0_0', 0, 'constraint',
                                 'Audit retention no more than 5 years', []))
    m._merge_patch(SemanticPatch('p1_0', 1, 'constraint',
                                 'Audit retention must not exceed 5 years', []))
    crits = [p.payload for p in m.critical_patches]
    assert len(crits) == 1, \
        f"same-direction bounds should collapse: {crits}"
    assert '5' in crits[0]


def test_polarity_words_survive_end_to_end():
    """v0.1.16 end-to-end: at a starvation budget the compressed
    context keeps 'not', 'under', and 'less than' — CRR was already
    100 on v0.1.15, but the MEANING was not."""
    m = DSPMMemory(budget=40)
    m._merge_patch(SemanticPatch('p0_0', 0, 'constraint',
                                 'Dashboard must load in under 2 seconds', []))
    m._merge_patch(SemanticPatch('p1_0', 1, 'constraint',
                                 'Payments must not be stored in plaintext', []))
    m._merge_patch(SemanticPatch('p2_0', 2, 'constraint',
                                 'Latency less than 100 milliseconds', []))
    m._merge_patch(SemanticPatch('p3_0', 3, 'constraint',
                                 'Rate limit 100 requests per minute', []))
    ctx = m.get_context(query='hard requirements')
    assert 'not' in ctx.lower(), f"negation lost end-to-end: {ctx!r}"
    assert 'under' in ctx.lower(), f"comparator lost end-to-end: {ctx!r}"
    assert 'less than' in ctx.lower(), f"comparator lost end-to-end: {ctx!r}"
    assert m.stats['crr'] == 100


def test_crr_reports_zero_when_budget_excludes_all_criticals():
    m = DSPMMemory(budget=4)
    m._merge_patch(SemanticPatch('p0_0', 0, 'constraint', 'Must NOT use Redis', []))
    assert m.stats['crr'] == 100
    ctx = m.get_context(query='redis')
    assert ctx == ''
    assert m.stats['crr'] == 0


def test_reset_resets_crr_contract():
    m = DSPMMemory(budget=4)
    m._merge_patch(SemanticPatch('p0_0', 0, 'constraint', 'Must NOT use Redis', []))
    m.get_context(query='redis')
    assert m.stats['crr'] == 0
    m.reset()
    m._merge_patch(SemanticPatch('p1_0', 0, 'constraint', 'Use Postgres', []))
    assert m.stats['crr'] == 100


# ---------------------------------------------------------------------------
# v0.1.17 review-fix regressions: role threading, stats delivery flag,
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


# ---------------------------------------------------------------------------
# v0.1.17 finding S1 (live conversation smoke): cross-type CON<->DEC
# restatements collapsed; "sub-" reads as an upper bound; units scan
# hyphenated tokens. Live evidence (DIAG dump): 'Maximum fee per
# transaction is limited to 0.5%' [CON] + 'fee cap set to 0.5% as tracked'
# [DEC], and 'dashboard must load in under 2 seconds' [DEC] +
# 'Dashboard latency must be sub-2-second' [CON], coexisted as duplicate
# criticals; the store bloated to 8-10 and CRR fell to 70-87% at the
# smoke's 59-token budget.
# ---------------------------------------------------------------------------


def test_cross_type_dashboard_restatement_collapses():
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('s0', 6, 'decision',
                                 'dashboard must load in under 2 seconds', []))
    m._merge_patch(SemanticPatch('s1', 7, 'constraint',
                                 'Dashboard latency must be sub-2-second', []))
    crits = [p.payload for p in m.critical_patches]
    assert len(crits) == 1, \
        f"cross-type dashboard restatement should collapse: {crits}"


def test_cross_type_retention_restatement_collapses():
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('s0', 0, 'decision',
                                 'Audit retention set to 5 years', []))
    m._merge_patch(SemanticPatch('s1', 1, 'constraint',
                                 'Audit logs stored for exactly 5 years', []))
    crits = [p.payload for p in m.critical_patches]
    assert len(crits) == 1, \
        f"cross-type retention restatement should collapse: {crits}"


def test_cross_type_thin_shared_fee_pair_collapses():
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
    assert len(m.critical_patches) == 1, \
        f"bound-head bridge should collapse the fee echo: {[p.payload for p in m.critical_patches]}"


def test_cross_type_contradictory_bounds_not_collapsed():
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('s0', 0, 'decision',
                                 'Audit retention must not exceed 5 years', []))
    m._merge_patch(SemanticPatch('s1', 1, 'constraint',
                                 'Audit retention at least 5 years', []))
    assert len(m.critical_patches) == 2, \
        f"contradictory cross-type bounds merged: {[p.payload for p in m.critical_patches]}"


def test_cross_type_same_subject_pages_not_collapsed():
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('s0', 0, 'constraint',
                                 'Dashboard loads in under 2 seconds', []))
    m._merge_patch(SemanticPatch('s1', 1, 'decision',
                                 'Homepage loads in under 2 seconds', []))
    assert len(m.critical_patches) == 2, \
        f"different subjects merged: {[p.payload for p in m.critical_patches]}"


def test_cross_type_plain_values_not_collapsed():
    """Both-plain guard: no directional/exact bound, one shared word --
    'Budget 100 for cloud' vs 'Budget 100 for training' must stay apart."""
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('s0', 0, 'constraint',
                                 'Budget 100 for cloud', []))
    m._merge_patch(SemanticPatch('s1', 1, 'decision',
                                 'Budget 100 for training', []))
    assert len(m.critical_patches) == 2


def test_cross_type_revision_supersedes_unchanged():
    """(b3) must not disturb (c): a real cross-type revision with changed
    numbers still supersedes and bakes history."""
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('s0', 0, 'constraint',
                                 'Webhook timeout must be 30 seconds', []))
    m._merge_patch(SemanticPatch('s1', 1, 'decision',
                                 'Updated webhook timeout changed to 10 seconds instead', []))
    crits = [p.payload for p in m.critical_patches]
    assert len(crits) == 1, crits
    assert '(was' in crits[0] and '30' in crits[0], crits


def test_sub_prefix_reads_as_upper_bound():
    from dspm.memory import _bound_class
    assert _bound_class("Dashboard latency must be sub-2-second") == "upper"
    # sanity: prior vocabulary unchanged
    assert _bound_class("retention must not exceed 5 years") == "upper"
    assert _bound_class("retention exactly 5 years") == "exact"


def test_units_scan_hyphenated_tokens():
    from dspm.memory import _units_in
    assert _units_in("Dashboard latency must be sub-2-second") == frozenset({"second"})
    assert _units_in("24-hour window") == frozenset({"hour"})
    # sanity: plain tokens unchanged
    assert _units_in("timeout 30 seconds") == frozenset({"second"})


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
    assert len(m.critical_patches) == 1, \
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
    assert len(m.critical_patches) == 1, \
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
    assert len(m.critical_patches) == 2, \
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
    assert len(m.critical_patches) == 2, \
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
    assert all(p.source_role == 'assistant' for p in patches), \
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
    assert len(crits) == 1, \
        f"reversed dashboard echo should collapse: {crits}"


def test_compound_token_bridge_symmetric_direction():
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('s0', 0, 'decision',
                                 'sub-2-second dashboard confirmed', []))
    m._merge_patch(SemanticPatch('s1', 1, 'constraint',
                                 'Dashboard latency must be sub-2-second', []))
    assert len(m.critical_patches) == 1, \
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
    assert len(crits) == 1, \
        f"verb-first retention echo should collapse: {crits}"


def test_shared_subject_retention_echo_symmetric():
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('s0', 7, 'decision',
                                 'Retain audit logs for exactly 5 years', []))
    m._merge_patch(SemanticPatch('s1', 7, 'decision',
                                 'Audit retention set to 5 years', []))
    assert len(m.critical_patches) == 1, \
        f"symmetric retention echo should collapse: " \
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
    assert len(m.critical_patches) == 1, \
        f"double-dirty-head dashboard echo should collapse: " \
        f"{[p.payload for p in m.critical_patches]}"


def test_shared_subject_dashboard_echo_one_dirty_head_collapses():
    """The single-dirty-head variant: the constraint keeps its clean
    'dashboard' head, the echo still leads with 'confirmed'."""
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('s0', 7, 'constraint',
                                 'Dashboard latency must be sub-2-second', []))
    m._merge_patch(SemanticPatch('s1', 8, 'decision',
                                 'Confirmed sub-2-second dashboard', []))
    assert len(m.critical_patches) == 1, \
        f"one-dirty-head dashboard echo should collapse: " \
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
    assert len(m.critical_patches) == 1, \
        f"same-type under-2 echo should collapse: " \
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


# ── Issue #2 (GitHub): extractor-rewrite supersession gap ─────────
# Live evidence (gpt-oss-120b): the extractor rewrote 'TTL' to
# 'Time-to-live (TTL)' and added filler 'as specified'. The new
# patch's slot head became 'time' (first word of 'Time-to-live'),
# which doesn't appear in the old patch's content words, so (a3)'s
# _head_anchored check fails and the stale 60s patch survives
# alongside the new 120s patch. Both criticals render, the store
# bloats, CRR is reported as 50% on this slot.


def test_issue2_ttl_dec_to_dec_extractor_rewrite_supersedes():
    """Issue #2 (gpt-oss-120b shape, same-type): the extractor rewrites
    'TTL' to 'Time-to-live (TTL)' and adds 'as specified'. The new
    [DEC] patch's slot head is 'time', absent from the old [DEC]
    'Cache TTL must be 60 seconds' content words, so (a3) misses.
    Shared acronym 'TTL' should bridge."""
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('p0_0', 0, 'decision',
                                 'Cache TTL must be 60 seconds', []))
    m._merge_patch(SemanticPatch('p1_0', 1, 'decision',
                                 'Time-to-live (TTL) setting updated to 120 seconds as specified', []))
    crits = [p.payload for p in m.critical_patches]
    assert len(crits) == 1, f"stale 60s survived extractor rewrite: {crits}"
    assert '120' in crits[0]
    assert '(was' in crits[0] and '60' in crits[0]


def test_issue2_ttl_con_to_dec_extractor_rewrite_supersedes():
    """Issue #2 (gpt-oss-120b shape, cross-type): same extractor
    rewrite, but old patch is a [CON] (covers the case where the
    original requirement was a constraint and the revision is a
    decision)."""
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('p0_0', 0, 'constraint',
                                 'Cache TTL must be 60 seconds', []))
    m._merge_patch(SemanticPatch('p1_0', 1, 'decision',
                                 'Time-to-live (TTL) setting updated to 120 seconds as specified', []))
    crits = [p.payload for p in m.critical_patches]
    assert len(crits) == 1, f"stale 60s survived extractor rewrite: {crits}"
    assert '120' in crits[0]
    assert '(was' in crits[0] and '60' in crits[0]


# ── Bug #2 regression: deadline supersession renders a single line ──

def test_bug2_deadline_supersession_renders_single_line():
    """Bug #2 (duplicate rendering): 'Deadline set to Friday' followed
    by 'Deadline updated to Monday' must render as a SINGLE line in
    the delivered context, with the old day baked as (was Friday).
    Pre-fix: both old and new patches rendered simultaneously,
    bloating the context with duplicate deadline lines."""
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('p0_0', 0, 'decision',
                                 'Deadline set to Friday', []))
    m._merge_patch(SemanticPatch('p1_0', 1, 'decision',
                                 'Deadline updated to Monday', []))
    # Store check: only one critical survives
    crits = [p.payload for p in m.critical_patches]
    assert len(crits) == 1, f"stale Friday survived: {crits}"
    # Render check: exactly one deadline-related line in context
    ctx = m.get_context(query='deadline')
    deadline_lines = [l for l in ctx.splitlines()
                      if 'deadline' in l.lower()
                      or 'monday' in l.lower()
                      or 'friday' in l.lower()]
    assert len(deadline_lines) == 1, \
        f"duplicate rendering (Bug #2): {deadline_lines}"
    assert 'Monday' in ctx
    # Cross-type variant
    m2 = DSPMMemory(budget=250)
    m2._merge_patch(SemanticPatch('p0_0', 0, 'constraint',
                                  'Deadline set to Friday', []))
    m2._merge_patch(SemanticPatch('p1_0', 1, 'decision',
                                  'Deadline moved to Monday', []))
    assert len(m2.critical_patches) == 1, \
        f"cross-type deadline not superseded: {[p.payload for p in m2.critical_patches]}"
    # Scoped variant: different deadlines stay apart
    m3 = DSPMMemory(budget=250)
    m3._merge_patch(SemanticPatch('p0_0', 0, 'decision',
                                  'Invoice deadline set to Friday', []))
    m3._merge_patch(SemanticPatch('p1_0', 1, 'decision',
                                  'Report deadline moved to Monday', []))
    assert len(m3.critical_patches) == 2, \
        f"scoped deadlines wrongly merged: {[p.payload for p in m3.critical_patches]}"


# ── Bug #3: non-numeric revision (Friday -> Monday) ──────────────
# Two sub-bugs found via scripts/verify_bug2_bug3.py:
#   3a - capitalization lost in the baked (was X) span
#   3b - chain truncation: 'Mon (was Fri)' superseded by 'Wed'
#        baked only '(was Fri)', dropping 'Mon' from the lineage

def test_bug3a_day_revision_bakes_capitalized_day():
    """Bug #3a: the (was X) span must preserve the day name's original
    capitalization. Pre-fix: 'Deadline set to Friday' superseded by
    'Deadline updated to Monday' baked '(was friday)' (lowercased)
    because _day_value() lowercases via _normalize()."""
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('p0_0', 0, 'decision',
                                 'Deadline set to Friday', []))
    m._merge_patch(SemanticPatch('p1_0', 1, 'decision',
                                 'Deadline updated to Monday', []))
    crits = [p.payload for p in m.critical_patches]
    assert len(crits) == 1
    assert '(was Friday)' in crits[0], \
        f"day name lowercased in baked history: {crits[0]!r}"
    assert '(was friday)' not in crits[0], \
        f"lowercased day name leaked into history: {crits[0]!r}"


def test_bug3b_day_revision_chain_bakes_full_lineage():
    """Bug #3b: a revision of a previously-revised day patch must
    chain the full lineage. Pre-fix: 'Deadline updated to Monday
    (was Friday)' superseded by 'Deadline moved to Wednesday' baked
    only '(was Friday)', silently dropping 'Monday' from the chain
    because _replaced_value's numeric chain logic never fires for
    day-only patches (no digits -> current is None)."""
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('p0_0', 0, 'decision',
                                 'Deadline set to Friday', []))
    m._merge_patch(SemanticPatch('p1_0', 1, 'decision',
                                 'Deadline updated to Monday', []))
    m._merge_patch(SemanticPatch('p2_0', 2, 'decision',
                                 'Deadline moved to Wednesday', []))
    crits = [p.payload for p in m.critical_patches]
    assert len(crits) == 1, f"chain should collapse to one critical: {crits}"
    assert 'Wednesday' in crits[0], f"current value missing: {crits[0]!r}"
    assert '(was Monday then Friday)' in crits[0], \
        f"full day lineage not chained: {crits[0]!r}"


def test_bug3_day_revision_same_day_restatement_no_baking():
    """Bug #3 guard: a same-day restatement ('updated to Monday' +
    'moved to Monday') must NOT bake a (was X) span — it's the
    same value, not a revision. The restatement check inside
    _replaced_value must short-circuit and return history (None)
    when the new patch mentions the stale's current day."""
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('p0_0', 0, 'decision',
                                 'Deadline updated to Monday', []))
    m._merge_patch(SemanticPatch('p1_0', 1, 'decision',
                                 'Deadline moved to Monday', []))
    crits = [p.payload for p in m.critical_patches]
    assert len(crits) == 1
    assert '(was' not in crits[0], \
        f"same-day restatement should not bake: {crits[0]!r}"


def test_bug3_day_revision_scoped_deadlines_stay_apart():
    """Bug #3 guard: scoped deadlines ('Invoice deadline' vs 'Report
    deadline') must NOT supersede each other — they're different slots.
    The _day_revision head-containment guard must reject pairs whose
    new patch's slot head doesn't appear in the stale's content words."""
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('p0_0', 0, 'decision',
                                 'Invoice deadline set to Friday', []))
    m._merge_patch(SemanticPatch('p1_0', 1, 'decision',
                                 'Report deadline moved to Monday', []))
    crits = [p.payload for p in m.critical_patches]
    assert len(crits) == 2, \
        f"scoped deadlines wrongly merged: {crits}"


# ── Bug #4: crr=100 but PCI-DSS never critical ───────────────────
# Two root causes found via scripts/verify_bug4.py:
#   4a - extractor prompt has no compliance/security classification
#        rule, so PCI-DSS gets typed as [entity] (non-critical)
#   4b - CRR formula returns 100 vacuously when critical_total == 0

def test_bug4a_extractor_prompt_classifies_compliance_as_constraint():
    """Bug #4a: the system prompt must tell the LLM that regulatory,
    compliance, and security requirements (PCI-DSS, HIPAA, GDPR, SOC 2,
    ISO 27001, FedRAMP, etc.) are CONSTRAINTS, not entities — even when
    they are named things. They are non-negotiable specs the system must
    satisfy. Pre-fix: the prompt mentioned 'entity holds named things'
    with no compliance rule, so PCI-DSS was typed as [entity]
    (non-critical)."""
    from dspm.extractor import _build_system_prompt
    for dense in (True, False):
        prompt = _build_system_prompt(dense=dense)
        prompt_lower = prompt.lower()
        assert ('pci-dss' in prompt_lower
                or 'hipaa' in prompt_lower
                or 'compliance' in prompt_lower), \
            f"dense={dense}: prompt lacks compliance classification rule"
        # The rule must explicitly say compliance requirements are
        # constraints, not just mention the keywords
        assert 'constraint' in prompt_lower, \
            f"dense={dense}: prompt must reference constraint type"


def test_bug4b_crr_honest_when_no_criticals_post_build():
    """Bug #4b: when get_context() has been called but the store has
    zero critical patches, CRR must NOT report 100 vacuously. Pre-fix:
    crr=100 hid the fact that PCI-DSS info was never classified as
    critical (it was typed as [entity], which is non-critical). The
    honest value is 0 -- we tried to deliver criticals but there were
    none to deliver."""
    m = DSPMMemory(budget=250)
    # Simulate PCI-DSS mistyped as [entity] (the bug shape)
    m._merge_patch(SemanticPatch('p0_0', 0, 'entity',
                                 'PCI-DSS compliance required', []))
    m._merge_patch(SemanticPatch('p1_0', 1, 'entity',
                                 'HIPAA compliant', []))
    # Pre-build: CRR is assumed 100 (backward compat)
    assert m.stats['crr'] == 100, "pre-build should assume 100"
    assert m.stats['context_built'] is False
    # Post-build: CRR must be honest about the empty critical set
    m.get_context(query='compliance')
    s = m.stats
    assert s['context_built'] is True
    assert s['critical_total'] == 0
    assert s['crr'] == 0, \
        f"vacuous 100% post-build hides missing criticals: {s}"


def test_bug4b_crr_pre_build_still_100_for_backward_compat():
    """Bug #4b guard: pre-build, CRR still reports 100 (assumed
    perfect) for backward compatibility. The honesty fix only kicks
    in post-build, because pre-build we haven't measured anything yet."""
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('p0_0', 0, 'entity',
                                 'PCI-DSS compliance required', []))
    s = m.stats
    assert s['context_built'] is False
    assert s['critical_total'] == 0
    assert s['crr'] == 100, "pre-build should assume 100"


def test_bug4b_crr_still_100_with_real_criticals():
    """Bug #4b guard: when critical_total > 0, CRR formula is unchanged.
    The fix only affects the critical_total == 0 post-build case."""
    m = DSPMMemory(budget=250)
    m._merge_patch(SemanticPatch('p0_0', 0, 'constraint',
                                 'Webhook timeout must be 30 seconds', []))
    assert m.stats['crr'] == 100  # pre-build
    m.get_context(query='webhook')
    s = m.stats
    assert s['critical_total'] == 1
    assert s['crr'] == 100, f"real criticals should still report 100: {s}"


# ── Bug #5: partial truncation is parsed instead of discarded ─────────

def test_bug5_partial_length_content_is_not_discarded():
    """Non-empty content returned with finish_reason='length' must reach
    parse_extraction. Partial JSON may parse or fail, but it must not be
    treated as an empty truncation or retried."""
    from types import SimpleNamespace
    from dspm.extractor import extract_turn

    class PartialLengthClient:
        def __init__(self):
            self.calls = 0

            class Completions:
                def create(inner_self, **kwargs):
                    self.calls += 1
                    return SimpleNamespace(choices=[SimpleNamespace(
                        message=SimpleNamespace(content='[{"patch_type":"entity"'),
                        finish_reason="length",
                    )])

            self.chat = SimpleNamespace(completions=Completions())

    llm = PartialLengthClient()
    telemetry = {}

    # Use a direct extraction call so the test can observe a parse attempt
    # without depending on DSPMMemory's private telemetry callback wiring.
    patches = extract_turn(llm, 'test-model', 'We use Stripe.', 0, telemetry=telemetry)
    assert patches == []  # the deliberately incomplete JSON cannot recover
    assert telemetry.get('llm_length_finish', 0) == 0, \
        f"partial truncation should NOT bump llm_length_finish: {telemetry}"
    assert telemetry.get('llm_calls') == 1, \
        f"partial content should not retry: {telemetry}"


# ── Bug #6: TRR computed on patch_tokens, not conversation_tokens ───
# Root cause confirmed via scripts/verify_bug6.py: a 55-token
# conversation extracting to 16 patch tokens and delivering 16 context
# tokens reported trr=0% (looks like zero reduction) when the real
# end-to-end compression was 71%.
# Fix (Option B): add trr_by_conversation alongside existing trr.
# Backward compatible: existing trr key still present, just renamed
# semantically to "patch-compression TRR".

def test_bug6_trr_by_conversation_uses_correct_denominator():
    """Bug #6: trr_by_conversation uses conversation_tokens as the
    denominator (raw turn text fed to add_turn), not patch_tokens
    (extracted patches)."""
    from dspm.patch import count_tokens
    llm = MockLLMClient()
    mem = DSPMMemory(budget=250, llm_client=llm, model='test-model')
    t1 = 'Return JSON and use POST /v1/items for creation.'
    t2 = 'All API responses must include a request id header.'
    mem.add_turn('user', t1)
    mem.add_turn('assistant', t2)
    mem.get_context('json')
    s = mem.stats
    expected_conv = count_tokens(t1) + count_tokens(t2)
    assert s['conversation_tokens'] == expected_conv, \
        f"conversation_tokens wrong: {s}"
    expected_trr_conv = 100 - int(
        (s['context_tokens'] / max(1, s['conversation_tokens'])) * 100)
    assert s['trr_by_conversation'] == expected_trr_conv, \
        f"trr_by_conversation wrong: expected {expected_trr_conv}, got {s['trr_by_conversation']}"
    assert s['trr'] != s['trr_by_conversation'] or s['patch_tokens'] == s['conversation_tokens'], \
        f"trr and trr_by_conversation should differ when patch_tokens != conversation_tokens: {s}"


def test_bug6_trr_key_still_present_for_backward_compat():
    """The existing trr key remains available alongside the new metric."""
    llm = MockLLMClient()
    mem = DSPMMemory(budget=250, llm_client=llm, model='test-model')
    mem.add_turn('user', 'Return JSON output.')
    s = mem.stats
    assert 'trr' in s, "trr key must remain for backward compat"
    assert 'trr_by_conversation' in s, "trr_by_conversation key must be added"


def test_bug6_trr_by_conversation_zero_when_no_conversation():
    """No conversation input means the end-to-end reduction is undefined
    and is reported as zero."""
    m = DSPMMemory(budget=250)
    s = m.stats
    assert s['conversation_tokens'] == 0
    assert s['trr_by_conversation'] == 0, \
        f"trr_by_conversation should be 0 with no conversation: {s}"


def test_bug6_trr_by_conversation_100_when_context_empty():
    """An empty delivered context with non-empty conversation input means
    full end-to-end token reduction."""
    m = DSPMMemory(budget=250)
    m._conversation_tokens = 50
    m.get_context(query='whatever')
    s = m.stats
    assert s['context_tokens'] == 0, s
    assert s['trr_by_conversation'] == 100, \
        f"empty context + non-empty conversation should be 100%: {s}"
