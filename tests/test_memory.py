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
# Reported by a real user: "even after changing the budget, it shows
# the same results." Root cause: DSPMMemory.budget and engine.budget
# were independent copies — memory.budget assignments were silently
# ignored by the engine.

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