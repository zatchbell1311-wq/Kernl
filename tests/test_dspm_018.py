"""Offline regression coverage for the v0.1.18 instrumentation batch."""

import json
from types import SimpleNamespace

from dspm.engine import DSPMEngine
from dspm.memory import DSPMMemory
from dspm.patch import SemanticPatch, count_tokens


CON_BOUND = {
    "patch_type": "constraint",
    "payload": "Audit retention must not exceed 90 days",
    "patch_id": "llm-a",
    "dependencies": [],
}
DEC_PLAIN_ECHO = {
    "patch_type": "decision",
    "payload": "Updated audit retention to 90 days",
    "patch_id": "llm-b",
    "dependencies": [],
}
DEC_300 = {
    "patch_type": "decision",
    "payload": "Escalation timeout set to 300 seconds",
    "patch_id": "llm-c",
    "dependencies": [],
}
DEC_120 = {
    "patch_type": "decision",
    "payload": "Escalation timeout changed to 120 seconds",
    "patch_id": "llm-d",
    "dependencies": [],
}
BOUND_PAYLOAD = "Audit retention must not exceed 90 days"


class FakeLLM:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        result = self.script.pop(0)
        content = json.dumps(result if isinstance(result, list) else [result])
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=content), finish_reason="stop")])


def test_supersession_history_is_journaled():
    mem = DSPMMemory(llm_client=FakeLLM(script=[DEC_300, DEC_120]))
    mem.add_turn("user", "set the escalation timeout to 300 seconds")
    mem.add_turn("user", "change the escalation timeout to 120 seconds")
    dump = mem.dump_store(include_history=True)

    assert len(dump["patches"]) == 1
    assert dump["patches"][0]["status"] == "active"
    assert len(dump["superseded"]) == 1
    assert dump["superseded"][0]["status"] == "superseded"
    assert dump["superseded"][0]["patch_id"] == "p0_0"
    assert dump["superseded"][0]["superseded_by"] == "p1_0"


def test_bound_wording_guard_off_keeps_0117_behavior():
    mem = DSPMMemory(llm_client=FakeLLM(script=[CON_BOUND, DEC_PLAIN_ECHO]))
    mem.add_turn("user", "Audit retention must not exceed 90 days.")
    mem.add_turn("assistant", "Updated: audit retention 90 days.")

    crits = mem.critical_patches
    assert len(crits) == 1
    assert "not exceed" not in crits[0].payload


def test_bound_wording_guard_on_preserves_bound():
    mem = DSPMMemory(
        llm_client=FakeLLM(script=[CON_BOUND, DEC_PLAIN_ECHO]),
        protect_bound_wording=True,
    )
    mem.add_turn("user", "Audit retention must not exceed 90 days.")
    mem.add_turn("assistant", "Updated: audit retention 90 days.")

    crits = mem.critical_patches
    assert len(crits) == 1
    assert "not exceed" in crits[0].payload
    dump = mem.dump_store(include_history=True)
    assert "bound_wording_guard" in [d["kind"] for d in dump["deduped"]]


def test_operator_phrase_survives_moderate_trim():
    engine = DSPMEngine(budget=12)
    patch = SemanticPatch("p1", 0, "constraint", BOUND_PAYLOAD, [])
    selected, _diag = engine.compress([patch], "", 0)
    ctx = engine.build_context(selected)
    assert "not exceed" in ctx
    assert "90" in ctx
    assert "day" in ctx


def test_starvation_two_word_floor_freezes_current_limitation():
    engine = DSPMEngine(budget=60)
    assert engine._trim_numeric_first(BOUND_PAYLOAD, 2) == "90 days"


def test_budget_hard_cap_respected():
    patches = [SemanticPatch(f"p{i}", i, "constraint",
                             f"limit {i} threshold {i} seconds", [])
               for i in range(30)]
    engine = DSPMEngine(budget=40)
    selected, _diag = engine.compress(patches, "", 0)
    assert count_tokens(engine.build_context(selected)) <= 40


def test_dump_store_consistent_with_stats_and_calls():
    llm = FakeLLM(script=[DEC_300, DEC_120])
    mem = DSPMMemory(llm_client=llm)
    mem.add_turn("user", "set the escalation timeout to 300 seconds")
    mem.add_turn("user", "change the escalation timeout to 120 seconds")
    dump = mem.dump_store(include_history=True)

    assert len(dump["patches"]) == len(mem.patches)
    ids = [record["patch_id"] for record in dump["patches"]]
    assert len(ids) == len(set(ids))
    stats = mem.stats
    assert stats["llm_calls"] == 2 == len(llm.calls)
    assert stats["llm_fallbacks"] == 0


def test_add_turn_idempotent_and_get_context_deterministic():
    mem = DSPMMemory(llm_client=FakeLLM(script=[DEC_300, DEC_300]))
    mem.add_turn("user", "set the escalation timeout to 300 seconds")
    mem.add_turn("user", "set the escalation timeout to 300 seconds")
    assert len(mem.patches) == 1
    context1 = mem.get_context("what is the timeout?")
    context2 = mem.get_context("what is the timeout?")
    assert context1 == context2
