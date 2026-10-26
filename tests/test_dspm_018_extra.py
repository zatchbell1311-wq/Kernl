"""UNTESTED -- written against the pasted dspm-memory 0.1.18 source
(config.py, extractor.py, memory.py, tests/test_dspm_018.py); NOT executed
by the author. New offline regression coverage complementing
tests/test_dspm_018.py. Deterministic, network-free, Windows-safe.

Coverage (item numbers match the deliverable-A spec):
  1.  strict_llm=True + empty completion            -> RuntimeError, exact counters
  2.  strict_llm=True + finish_reason="length"
      with empty content                             -> RuntimeError, exact counters
  3.  strict_llm=True + malformed non-JSON output   -> RuntimeError, exact counters
  4.  strict_llm=False + empty completion           -> 0-patch fallback, add_turn continues
  5.  a legitimate "[]" response                     -> NOT a parse failure, NOT a fallback
  6.  clean 3-turn run                               -> exact llm_calls, all failure counters 0
  7.  mixed script (empty then good, one turn)      -> retries counted, no fallback
  8.  max_tokens floor: max_tokens=50 -> 512 (direct extract_turn);
      default DSPMMemory -> 4000; max_tokens_floor=6000 -> 6000
  9.  reasoning_param_mode "top" / "extra_body" / "off" for a gpt-oss model
  10. feedback_context=False -> no "Recent context:" block in the prompt
  11. dump_store(include_history=True) after 300, duplicate 300, 120:
      supersession journal + exact_same_type dedup journal

Design notes (all taken from the pasted source, not assumptions):
  - FakeLLM mirrors tests/test_dspm_018.py's script/calls/SimpleNamespace
    pattern, extended with one capability: a (content, finish_reason)
    tuple entry is returned verbatim so tests can script empty
    completions and finish_reason="length". Dict/list entries keep the
    original JSON-encoded, finish_reason="stop" behavior.
  - extract_turn sleeps time.sleep(2)/time.sleep(4) between retry
    attempts; every test that can hit a retry monkeypatches
    dspm.extractor.time.sleep and asserts the recorded durations
    instead of waiting.
  - Empty-completion scripts carry exactly three entries because the
    retry loop makes up to 3 attempts before the strict raise / lenient
    fallback; the FakeLLM pops one entry per attempt.
  - No test calls get_context() or any engine method, so the
    sentence-transformers load suspected behind the slow existing suite
    is never triggered. mem.stats and mem.dump_store are the only
    observation surfaces.
"""

import json
from types import SimpleNamespace

import pytest

import dspm.extractor
from dspm.config import EXTRACT_MAX_TOKENS, EXTRACT_MIN_TOKENS
from dspm.extractor import extract_turn
from dspm.memory import DSPMMemory


# ---------------------------------------------------------------------------
# Fixtures -- payloads mirror tests/test_dspm_018.py so scenarios stay
# comparable across the two files. WEBHOOK_DEC is new: a third, clearly
# distinct fact for multi-turn runs (no overlap with the other payloads
# under any merge rule).
# ---------------------------------------------------------------------------

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
CON_BOUND = {
    "patch_type": "constraint",
    "payload": "Audit retention must not exceed 90 days",
    "patch_id": "llm-a",
    "dependencies": [],
}
WEBHOOK_DEC = {
    "patch_type": "decision",
    "payload": "Webhook retries capped at 5 attempts",
    "patch_id": "llm-e",
    "dependencies": [],
}

GPT_OSS_MODEL = "openai/gpt-oss-20b"

EMPTY_STOP = ("", "stop")
EMPTY_LENGTH = ("", "length")


class FakeLLM:
    """Same script/calls/SimpleNamespace pattern as tests/test_dspm_018.py,
    extended with one capability: a (content, finish_reason) tuple entry is
    returned verbatim, so tests can script empty completions,
    finish_reason="length" truncations, and malformed raw text. Dict and
    list entries keep the original JSON-encoding behavior with
    finish_reason="stop"."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        result = self.script.pop(0)
        if isinstance(result, tuple):
            content, finish_reason = result
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason=finish_reason)])
        content = json.dumps(result if isinstance(result, list) else [result])
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=content), finish_reason="stop")])


def _patch_sleep(monkeypatch):
    """Neutralize the retry backoff and record it instead of waiting.

    extract_turn calls time.sleep(2) and time.sleep(4) between its up-to-3
    attempts (extractor.py); monkeypatching dspm.extractor.time.sleep keeps
    the tests fast and lets them assert the backoff actually ran.
    """
    sleeps = []
    monkeypatch.setattr(dspm.extractor.time, "sleep",
                        lambda seconds: sleeps.append(seconds))
    return sleeps


def _all_llm_counters(stats):
    """The seven v0.1.18 telemetry keys as a flat tuple, in the fixed order
    (llm_calls, llm_empty, llm_errors, llm_retries, llm_length_finish,
    llm_parse_failures, llm_fallbacks) -- the same order as the dict
    literal in DSPMMemory.__init__."""
    return (stats["llm_calls"], stats["llm_empty"], stats["llm_errors"],
            stats["llm_retries"], stats["llm_length_finish"],
            stats["llm_parse_failures"], stats["llm_fallbacks"])


# ---------------------------------------------------------------------------
# Items 1-3: strict_llm=True turns each failure class into a RuntimeError
# ---------------------------------------------------------------------------

def test_strict_llm_empty_completion_raises(monkeypatch):
    """Item 1: strict + empty completion -> RuntimeError. The retry loop
    runs its full 3 attempts first (2s/4s backoff), then llm_fallbacks
    counts the silent fallback that strict refuses."""
    sleeps = _patch_sleep(monkeypatch)
    llm = FakeLLM(script=[EMPTY_STOP, EMPTY_STOP, EMPTY_STOP])
    mem = DSPMMemory(llm_client=llm, strict_llm=True)
    with pytest.raises(RuntimeError, match="returned empty content"):
        mem.add_turn("user", "set the escalation timeout to 300 seconds")
    # the turn aborted before bookkeeping: no turn counted, nothing stored
    assert mem.stats["turns"] == 0
    assert mem.stats["total_patches"] == 0
    assert _all_llm_counters(mem.stats) == (3, 3, 0, 2, 0, 0, 1)
    assert len(llm.calls) == 3
    assert sleeps == [2, 4]


def test_strict_llm_length_finish_empty_content_raises(monkeypatch):
    """Item 2: strict + finish_reason="length" with empty content ->
    RuntimeError from the truncation branch. Empty content keeps the retry
    loop running, so counters match item 1 plus llm_length_finish=1."""
    sleeps = _patch_sleep(monkeypatch)
    llm = FakeLLM(script=[EMPTY_LENGTH, EMPTY_LENGTH, EMPTY_LENGTH])
    mem = DSPMMemory(llm_client=llm, strict_llm=True)
    with pytest.raises(RuntimeError, match="output was truncated"):
        mem.add_turn("user", "set the escalation timeout to 300 seconds")
    assert mem.stats["turns"] == 0
    assert _all_llm_counters(mem.stats) == (3, 3, 0, 2, 1, 0, 1)
    assert sleeps == [2, 4]


def test_strict_llm_malformed_output_raises(monkeypatch):
    """Item 3: strict + non-JSON output -> RuntimeError from the parse
    branch. Non-empty content breaks the retry loop on the FIRST attempt:
    one call, one parse failure, no backoff."""
    sleeps = _patch_sleep(monkeypatch)
    llm = FakeLLM(script=[("sorry, I cannot produce the patches right now", "stop")])
    mem = DSPMMemory(llm_client=llm, strict_llm=True)
    with pytest.raises(RuntimeError, match="is not a parseable JSON array"):
        mem.add_turn("user", "set the escalation timeout to 300 seconds")
    assert mem.stats["turns"] == 0
    assert _all_llm_counters(mem.stats) == (1, 0, 0, 0, 0, 1, 1)
    assert sleeps == []


# ---------------------------------------------------------------------------
# Items 4-5: lenient mode (default) degrades without raising
# ---------------------------------------------------------------------------

def test_lenient_empty_completion_falls_back_and_continues(monkeypatch):
    """Item 4: default strict_llm=False + empty completion -> no raise,
    0-patch turn, the same retry+fallback counters as the strict path;
    a following good turn stores its patch normally."""
    sleeps = _patch_sleep(monkeypatch)
    llm = FakeLLM(script=[EMPTY_STOP, EMPTY_STOP, EMPTY_STOP, DEC_300])
    mem = DSPMMemory(llm_client=llm)  # strict_llm defaults to LLM_STRICT_DEFAULT=False
    patches = mem.add_turn("user", "set the escalation timeout to 300 seconds")
    assert patches == []
    assert mem.stats["turns"] == 1
    assert _all_llm_counters(mem.stats) == (3, 3, 0, 2, 0, 0, 1)
    assert sleeps == [2, 4]

    second = mem.add_turn("user", "confirmed: the escalation timeout is 300 seconds")
    assert len(second) == 1
    assert len(mem.patches) == 1
    assert mem.stats["turns"] == 2
    assert _all_llm_counters(mem.stats) == (4, 3, 0, 2, 0, 0, 1)


def test_legitimate_empty_array_is_not_a_parse_failure():
    """Item 5: a raw "[]" response means "no patches in this turn" and must
    NOT touch llm_parse_failures or llm_fallbacks (extractor.py's
    legit_empty check)."""
    llm = FakeLLM(script=[[]])
    mem = DSPMMemory(llm_client=llm)
    patches = mem.add_turn("user", "this turn carries no extractable facts")
    assert patches == []
    assert mem.stats["turns"] == 1
    assert _all_llm_counters(mem.stats) == (1, 0, 0, 0, 0, 0, 0)


# ---------------------------------------------------------------------------
# Items 6-7: exact counter accounting across turns
# ---------------------------------------------------------------------------

def test_clean_run_counts_calls_with_zero_failures(monkeypatch):
    """Item 6: three good turns, three distinct facts -> exactly
    llm_calls=3, every failure counter zero, no backoff sleeps."""
    sleeps = _patch_sleep(monkeypatch)
    llm = FakeLLM(script=[DEC_300, CON_BOUND, WEBHOOK_DEC])
    mem = DSPMMemory(llm_client=llm)
    mem.add_turn("user", "set the escalation timeout to 300 seconds")
    mem.add_turn("user", "Audit retention must not exceed 90 days.")
    mem.add_turn("user", "cap webhook retries at 5 attempts")
    assert mem.stats["turns"] == 3
    assert len(mem.patches) == 3
    assert _all_llm_counters(mem.stats) == (3, 0, 0, 0, 0, 0, 0)
    assert len(llm.calls) == 3
    assert sleeps == []
    # default model gpt-4o-mini is not a gpt-oss model: no reasoning kwargs
    assert "reasoning_effort" not in llm.calls[0]
    assert "extra_body" not in llm.calls[0]


def test_mixed_empty_then_good_retries_without_fallback(monkeypatch):
    """Item 7: an empty completion followed by a good one INSIDE one turn ->
    the retry succeeds, llm_retries counts it, no fallback is recorded,
    and a later clean turn leaves the counters untouched."""
    sleeps = _patch_sleep(monkeypatch)
    llm = FakeLLM(script=[EMPTY_STOP, DEC_300, WEBHOOK_DEC])
    mem = DSPMMemory(llm_client=llm)
    patches = mem.add_turn("user", "set the escalation timeout to 300 seconds")
    assert len(patches) == 1
    assert _all_llm_counters(mem.stats) == (2, 1, 0, 1, 0, 0, 0)
    assert sleeps == [2]

    mem.add_turn("user", "cap webhook retries at 5 attempts")
    assert len(mem.patches) == 2
    assert _all_llm_counters(mem.stats) == (3, 1, 0, 1, 0, 0, 0)


# ---------------------------------------------------------------------------
# Item 8: max_tokens floor (config.EXTRACT_MIN_TOKENS = 512)
# ---------------------------------------------------------------------------

def test_max_tokens_floor_bumps_low_direct_request():
    """Item 8a: extract_turn(max_tokens=50) -> the client receives the
    512-token floor, not 50 (reasoning-model empty-completion guard)."""
    llm = FakeLLM(script=[DEC_300])
    telemetry = {}
    patches = extract_turn(llm, "gpt-4o-mini",
                           "set the escalation timeout to 300 seconds", 0,
                           max_tokens=50, telemetry=telemetry)
    assert len(patches) == 1
    assert llm.calls[0]["max_tokens"] == EXTRACT_MIN_TOKENS
    assert telemetry["llm_calls"] == 1


def test_max_tokens_default_and_floor_via_memory():
    """Item 8b: DSPMMemory never forwards a max_tokens (add_turn passes
    none), so the default request is EXTRACT_MAX_TOKENS=4000, already above
    the 512 floor; the only public-API case where the floor binds is
    max_tokens_floor > 4000, which RAISES the request above the default."""
    llm_default = FakeLLM(script=[DEC_300])
    DSPMMemory(llm_client=llm_default).add_turn(
        "user", "set the escalation timeout to 300 seconds")
    assert llm_default.calls[0]["max_tokens"] == EXTRACT_MAX_TOKENS
    assert EXTRACT_MAX_TOKENS >= EXTRACT_MIN_TOKENS

    llm_floored = FakeLLM(script=[DEC_300])
    DSPMMemory(llm_client=llm_floored, max_tokens_floor=6000).add_turn(
        "user", "set the escalation timeout to 300 seconds")
    assert llm_floored.calls[0]["max_tokens"] == 6000


# ---------------------------------------------------------------------------
# Item 9: reasoning_param_mode channels for a gpt-oss model
# ---------------------------------------------------------------------------

def test_reasoning_param_mode_default_sends_top_level_kwarg():
    """Item 9: reasoning_param_mode=None falls back to config
    REASONING_PARAM_MODE="top" -> reasoning_effort="low" as a top-level
    kwarg for a gpt-oss model, nothing nested."""
    llm = FakeLLM(script=[DEC_300])
    mem = DSPMMemory(llm_client=llm, model=GPT_OSS_MODEL)
    mem.add_turn("user", "set the escalation timeout to 300 seconds")
    assert llm.calls[0]["reasoning_effort"] == "low"
    assert "extra_body" not in llm.calls[0]


def test_reasoning_param_mode_extra_body_nests_kwarg():
    """Item 9: reasoning_param_mode="extra_body" -> reasoning_effort nested
    in extra_body, no top-level kwarg."""
    llm = FakeLLM(script=[DEC_300])
    mem = DSPMMemory(llm_client=llm, model=GPT_OSS_MODEL,
                     reasoning_param_mode="extra_body")
    mem.add_turn("user", "set the escalation timeout to 300 seconds")
    assert llm.calls[0]["extra_body"] == {"reasoning_effort": "low"}
    assert "reasoning_effort" not in llm.calls[0]


def test_reasoning_param_mode_off_omits_kwarg():
    """Item 9: reasoning_param_mode="off" -> the reasoning parameter is
    never sent in any form."""
    llm = FakeLLM(script=[DEC_300])
    mem = DSPMMemory(llm_client=llm, model=GPT_OSS_MODEL,
                     reasoning_param_mode="off")
    mem.add_turn("user", "set the escalation timeout to 300 seconds")
    assert "reasoning_effort" not in llm.calls[0]
    assert "extra_body" not in llm.calls[0]


# ---------------------------------------------------------------------------
# Item 10: feedback_context flag gates the prompt's recent-context block
# ---------------------------------------------------------------------------

def test_feedback_context_flag_gates_recent_context_block():
    """Item 10: with feedback_context=True, a set _last_context
    rides into the next extraction prompt under a 'Recent context:' block;
    with feedback_context=False it never does. _last_context is set
    directly instead of calling get_context() so this file stays
    engine-free (avoiding the suspected sentence-transformers load)."""
    ctx = "[DEC] Escalation timeout set to 300 seconds"

    llm_on = FakeLLM(script=[DEC_300, WEBHOOK_DEC])
    mem_on = DSPMMemory(llm_client=llm_on, feedback_context=True)
    mem_on.add_turn("user", "set the escalation timeout to 300 seconds")
    first_prompt = llm_on.calls[0]["messages"][1]["content"]
    assert "Recent context:" not in first_prompt  # _last_context empty on turn 0
    mem_on._last_context = ctx
    mem_on.add_turn("user", "cap webhook retries at 5 attempts")
    second_prompt_on = llm_on.calls[1]["messages"][1]["content"]
    assert "Recent context:" in second_prompt_on
    assert ctx in second_prompt_on
    assert "[user] cap webhook retries at 5 attempts" in second_prompt_on

    llm_off = FakeLLM(script=[DEC_300, WEBHOOK_DEC])
    mem_off = DSPMMemory(llm_client=llm_off, feedback_context=False)
    mem_off.add_turn("user", "set the escalation timeout to 300 seconds")
    mem_off._last_context = ctx
    mem_off.add_turn("user", "cap webhook retries at 5 attempts")
    second_prompt_off = llm_off.calls[1]["messages"][1]["content"]
    assert "Recent context:" not in second_prompt_off
    assert ctx not in second_prompt_off
    assert "[user] cap webhook retries at 5 attempts" in second_prompt_off
    # both runs stored the same two distinct facts
    assert len(mem_on.patches) == 2
    assert len(mem_off.patches) == 2


# ---------------------------------------------------------------------------
# Item 11: dump_store(include_history=True) journals both merge histories
# ---------------------------------------------------------------------------

def test_dump_store_journals_supersession_and_exact_duplicate():
    """Item 11: 300 -> duplicate 300 -> 120 leaves ONE active patch with
    the baked '(was 300)', one exact_same_type dedup journal entry, and
    one supersession journal entry."""
    llm = FakeLLM(script=[DEC_300, DEC_300, DEC_120])
    mem = DSPMMemory(llm_client=llm)
    mem.add_turn("user", "set the escalation timeout to 300 seconds")
    mem.add_turn("user", "confirmed: escalation timeout stays at 300 seconds")
    mem.add_turn("user", "change the escalation timeout to 120 seconds")

    dump = mem.dump_store(include_history=True)
    assert dump["turns"] == 3

    assert len(dump["patches"]) == 1
    active = dump["patches"][0]
    assert active["patch_id"] == "p2_0"
    assert active["status"] == "active"
    assert "changed to 120 seconds" in active["payload"]
    assert "(was 300)" in active["payload"]

    assert len(dump["superseded"]) == 1
    gone = dump["superseded"][0]
    assert gone["patch_id"] == "p0_0"
    assert gone["superseded_by"] == "p2_0"
    assert gone["status"] == "superseded"

    assert len(dump["deduped"]) == 1
    dup = dump["deduped"][0]
    assert dup["kind"] == "exact_same_type"
    assert dup["kept"] == "p0_0"
    assert dup["dropped"] == "p1_0"
    assert dup["status"] == "merged"
    assert dup["at_turn"] == 1

    assert dump["llm_telemetry"]["llm_calls"] == 3
    assert dump["llm_telemetry"]["llm_fallbacks"] == 0