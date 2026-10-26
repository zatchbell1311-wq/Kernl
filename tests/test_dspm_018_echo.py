"""Offline regression tests for the v0.1.18 echo-suppression variants
(guard v2 -- task 2b).

Task 2 context (diag_out_r2, budget=60, feedback_context=True, gpt-oss-20b):
turn 20 reported a pod restart -- "[report-gen] pod restarted after OOM;
limit raised to 2Gi." -- yet the extractor emitted five [DEC] patches
built out of the fed-back CP12 context. Four were partial echoes /
confabulations ("set alerts to 50", "set within to 2", "set duration to
90 days"); the fifth, "Connect retry updated
to 120 (was 300)" (two trimmed context lines, "Connect retry" + "120 (was
300)", stitched together), superseded the real p7_0 via rule (a) and the
real p2_0 via rule (a2), silently deleting two live facts. The r1 runs with
feedback_context=False never produced the pair.

Guard v1 (task 2) dropped only the fusion echo: its single 0.8
context-coverage bar let the model's glue words ("set ... to 50") dilute
every partial echo below the drop threshold, so the store still bloated.
Guard v2 replaces coverage percentages with attribution counts
(_echo_guard_reason in memory.py): rule 1 drops verbatim context
fragments, rule 2 drops payloads whose attributable tokens all come from
the context and none from the turn, rule 3 drops payloads supported by
neither shown text (counted separately in llm_unattributed_drops).

These tests replay that exact scenario against a canned LLM (no network):

  * echo_guard OFF (default) must reproduce the kill -- characterization
    of the current bug, frozen so a future rule change that masks it
    fails loudly here;
  * echo_guard ON must drop the four context echoes, retain the real pod
    update, avoid supersession, and preserve both victims;
  * the false-positive guards are frozen: a genuine update stated in the
    turn survives (turn-token share), and a terse anaphoric update
    ("Bump it to 130 ...") survives too (rule 2 requires ZERO turn
    tokens);
  * the rule-3 cost is frozen honestly: a heavily paraphrased genuine
    update sharing no vocabulary with either shown text is dropped --
    the documented price of determinism;
  * feedback_fence ON must wrap the fed-back context in the do-not-extract
    markers inside whatever prompt the extractor builds; OFF (default) must
    leave the prompt unfenced while still feeding the raw context.

Everything here is offline and deterministic: no LLM, no network, no
engine budget math beyond what _merge_patch itself does. Suite
expectation when this file replaces the task-2 version: 138 existing
+ 12 = 150 (the task-2 file's 8 tests are superseded by these 12).
"""

import json

from dspm import DSPMMemory
from dspm.patch import SemanticPatch
from dspm.memory import (
    _FEEDBACK_FENCE_OPEN,
    _FEEDBACK_FENCE_CLOSE,
    _FEEDBACK_FENCE_RULES,
)

# --- the exact r2 shapes (payload strings copied verbatim from the
# check_turn20 paste-back; context string copied verbatim from the b60
# CP12 store dump; turn text is the literal message 20 of the corpus) ---

CP12_CONTEXT_B60 = (
    "[DEC] 50 alerts / [CON] within 2 / [DEC] severity-1 incidents / "
    "[DEC] 90 days / [CON] 90 days. / [DEC] Connect retry / "
    "[DEC] 120 (was 300) / [DEC] Automatic retries"
)

TURN20_TEXT = "[report-gen] pod restarted after OOM; limit raised to 2Gi."

P2_0 = "Connect Slack workspace acme-ops. Failed alert deliveries must retry automatically."
P7_0 = "Escalation timeout updated to 120 seconds (was 300)"
P20_3 = "Connect retry updated to 120 (was 300)"


def _patch(pid, turn, ptype, payload):
    return {
        "patch_id": pid,
        "turn_index": turn,
        "patch_type": ptype,
        "payload": payload,
        "dependencies": [],
        "utility": 0.8,
        "fingerprint": "",
        "slot_key": "",
        "is_delta": False,
        "delta_base": "",
        "causal_depth": 0,
    }


TURN20_PATCHES = [
    _patch("p20_0", 20, "decision", "set alerts to 50"),
    _patch("p20_1", 20, "decision", "set within to 2"),
    _patch("p20_2", 20, "decision", "set duration to 90 days"),
    _patch("p20_3", 20, "decision", P20_3),
    _patch("p20_4", 20, "decision", "pod limit raised to 2Gi"),
]


class MockResponse:
    def __init__(self, content):
        self.choices = [type("Choice", (), {"message": type("Message", (), {"content": content})()})]


class RecordingCompletions:
    def __init__(self, owner):
        self.owner = owner

    def create(self, **kwargs):
        self.owner.calls.append(kwargs)
        return MockResponse(json.dumps(self.owner.patches))


class RecordingChat:
    def __init__(self, owner):
        self.owner = owner
        self.completions = RecordingCompletions(owner)


class RecordingClient:
    """OpenAI-style fake: chat.completions.create(**kwargs) -> .choices[0].message.content."""

    def __init__(self, patches):
        self.patches = patches
        self.calls = []
        self.chat = RecordingChat(self)


def _prompt_text(client):
    """Every string the extractor passed to create(), concatenated."""
    out = []

    def walk(obj):
        if isinstance(obj, str):
            out.append(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                walk(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                walk(v)

    for kwargs in client.calls:
        walk(kwargs)
    return " ".join(out)


def _seeded_memory(client, **kwargs):
    """Store state as it was at turn 20 of the r2 b60 run: p2_0 and p7_0
    active, CP12 context rendered, 20 turns ingested."""
    kwargs.setdefault("feedback_context", True)
    m = DSPMMemory(budget=60, llm_client=client, **kwargs)
    for pid, tidx, ptype, payload in (
        ("p2_0", 2, "decision", P2_0),
        ("p7_0", 7, "decision", P7_0),
    ):
        p = SemanticPatch(pid, tidx, ptype, payload, [])
        p.source_role = "user"
        m._merge_patch(p)
    m.turns = 20
    m._last_context = CP12_CONTEXT_B60
    return m


# -- turn-20 replay: characterization of the bug (defaults) -----------------

def test_turn20_replay_guard_off_reproduces_echo_kill():
    """With defaults (echo_guard off) the exact r2 turn-20 extraction kills
    p7_0 via rule (a) and p2_0 via rule (a2). Frozen as the bug report: if
    a future merge-rule change masks this failure, this test fails loudly."""
    m = _seeded_memory(RecordingClient(TURN20_PATCHES))
    added = m.add_turn("assistant", TURN20_TEXT)
    assert len(added) == 5
    payloads = {p.payload for p in m.patches}
    assert P20_3 in payloads                    # the echo entered the store
    assert P2_0 not in payloads and P7_0 not in payloads   # both victims dead
    sup = m.dump_store(include_history=True)["superseded"]
    assert len(sup) == 2
    killer = [p for p in m.patches if p.payload == P20_3][0]
    assert {e["payload"] for e in sup} == {P2_0, P7_0}
    assert all(e["superseded_by"] == killer.patch_id for e in sup)
    assert m.stats["llm_echo_drops"] == 0       # guard off: nothing dropped
    assert m.stats["llm_unattributed_drops"] == 0


def test_turn20_replay_guard_on_drops_echoes_keeps_real_patch_and_victims():
    """The four context echoes are dropped while the pod update from the
    turn survives; the fused context echo cannot supersede either victim."""
    m = _seeded_memory(RecordingClient(TURN20_PATCHES), echo_guard=True)
    added = m.add_turn("assistant", TURN20_TEXT)
    assert [p.payload for p in added] == ["pod limit raised to 2Gi"]
    payloads = {p.payload for p in m.patches}
    assert not ({"set alerts to 50", "set within to 2", "set duration to 90 days"} & payloads)
    assert P20_3 not in payloads
    assert P2_0 in payloads and P7_0 in payloads
    assert m.dump_store(include_history=True)["superseded"] == []
    assert m.stats["llm_echo_drops"] == 4
    assert m.stats["llm_unattributed_drops"] == 0


def test_guard_drops_partial_echo_fragments():
    """The bloat fix, isolated: the three partial echoes v1 kept ('set
    alerts to 50' etc.) each carry >= 2 context-only content tokens and
    ZERO turn tokens, so rule 2 drops them all. Glue words ('set', 'to')
    live in neither text and no longer dilute any coverage bar."""
    client = RecordingClient([
        _patch("p21_0", 21, "decision", "set alerts to 50"),
        _patch("p21_1", 21, "decision", "set within to 2"),
        _patch("p21_2", 21, "decision", "set duration to 90 days"),
    ])
    m = _seeded_memory(client, echo_guard=True)
    added = m.add_turn("assistant", TURN20_TEXT)
    assert added == []
    assert m.stats["llm_echo_drops"] == 3
    assert m.stats["llm_unattributed_drops"] == 0


def test_guard_drops_unattributed_payload():
    """Rule 3, isolated: the invented payload shares no content token with
    the turn or the fed-back context -- the extractor was shown
    exactly two texts and the payload is supported by neither. It is a
    confabulation, and it is counted separately from context echoes."""
    client = RecordingClient([
        _patch("p21_0", 21, "decision", "billing region migrated to frankfurt cluster"),
    ])
    m = _seeded_memory(client, echo_guard=True)
    added = m.add_turn("assistant", TURN20_TEXT)
    assert added == []
    assert m.stats["llm_unattributed_drops"] == 1
    assert m.stats["llm_echo_drops"] == 0


# -- false-positive guards: what the guard must NOT drop -------------------

def test_guard_keeps_genuine_update_stated_in_turn():
    """A payload overlapping the context must survive when the turn itself
    states the change: 4 of 5 content tokens come from the turn text, so
    rule 2's ZERO-turn-token leg fails and the patch is kept. This is the
    guard's main false-positive risk, frozen off."""
    client = RecordingClient([
        _patch("p21_0", 21, "decision", "alert threshold raised to 80 (was 50)"),
    ])
    m = DSPMMemory(budget=60, llm_client=client, echo_guard=True)
    m.turns = 21
    m._last_context = "[DEC] 50 alerts per month included"
    added = m.add_turn("user", "Raise the alert threshold from 50 to 80 per month.")
    assert len(added) == 1
    assert m.stats["llm_echo_drops"] == 0
    assert m.stats["llm_unattributed_drops"] == 0


def test_guard_keeps_terse_anaphoric_update():
    """A terse turn that renames the item by pronoun ('Bump it to 130')
    still yields a payload naming the item ('connect retry updated to
    130'). The payload shares connect/retry/130 with the turn, so rule 2
    (which requires ZERO turn tokens) keeps it -- even though 120 and
    the item name also appear in the fed-back context. This is why rule
    2 keys on turn tokens rather than a context-coverage bar."""
    client = RecordingClient([
        _patch("p21_0", 21, "decision", "connect retry updated to 130 (was 120)"),
    ])
    m = _seeded_memory(client, echo_guard=True)
    added = m.add_turn("user", "Bump it to 130 for the connect retry.")
    assert len(added) == 1
    assert m.stats["llm_echo_drops"] == 0
    assert m.stats["llm_unattributed_drops"] == 0


def test_guard_drops_heavily_paraphrased_feedback_documented_residual():
    """Rule 3's documented cost, frozen loudly: a genuine update phrased
    with entirely different vocabulary than the turn ('Cut the alert
    volume in half' -> 'user wants fewer notifications') shares zero
    content tokens with both shown texts and is dropped. Deterministic
    attribution cannot see synonymy. Mitigation, not cure: the feedback
    fence asks the model to restate the turn's nouns and numbers, which
    restores turn tokens and survival. If you fix this with a smarter
    classifier (embeddings, LLM judge), delete this test with the rule."""
    client = RecordingClient([
        _patch("p21_0", 21, "decision", "user wants fewer notifications"),
    ])
    m = _seeded_memory(client, echo_guard=True)
    added = m.add_turn("user", "Cut the alert volume in half.")
    assert added == []                         # the drop is expected here
    assert m.stats["llm_unattributed_drops"] == 1


def test_guard_drops_verbatim_context_fragment():
    """Rule 1: a payload copied word-for-word from the fed-back context
    that the turn never states is an echo regardless of token counts --
    including short payloads rule 2's >=2-token bar is too conservative
    to touch."""
    client = RecordingClient([
        _patch("p21_0", 21, "decision", "Automatic retries"),
    ])
    m = _seeded_memory(client, echo_guard=True)
    added = m.add_turn("assistant", TURN20_TEXT)
    assert added == []
    assert m.stats["llm_echo_drops"] == 1
    assert m.stats["llm_unattributed_drops"] == 0


def test_guard_inert_without_feedback_context():
    """With feedback_context=False there is nothing to echo: the guard must
    not drop anything even when the store once rendered a context (recent
    context is never fed, so the extractor cannot copy it) -- and rule 3
    must not fire either, because the whole guard is gated on the flag."""
    m = _seeded_memory(RecordingClient(TURN20_PATCHES),
                       echo_guard=True, feedback_context=False)
    added = m.add_turn("assistant", TURN20_TEXT)
    assert len(added) == 5
    assert m.stats["llm_echo_drops"] == 0
    assert m.stats["llm_unattributed_drops"] == 0


# -- prompt fence -----------------------------------------------------------

def test_fence_wraps_feedback_context_in_prompt():
    """feedback_fence=True: the fed-back context must reach the extractor
    wrapped in the do-not-extract markers, with the rules text riding along
    inside recent_context (no extractor.py change is involved)."""
    client = RecordingClient(TURN20_PATCHES)
    m = _seeded_memory(client, feedback_fence=True)
    m.add_turn("assistant", TURN20_TEXT)
    assert len(client.calls) == 1              # no retries: clean JSON
    assert m.stats["llm_calls"] == 1           # telemetry mutated in place
    prompt = _prompt_text(client)
    assert _FEEDBACK_FENCE_OPEN in prompt
    assert _FEEDBACK_FENCE_CLOSE in prompt
    assert _FEEDBACK_FENCE_RULES in prompt
    assert "Connect retry" in prompt           # raw context still fed back


def test_default_prompt_has_no_fence_and_context_still_fed():
    """Defaults unchanged: no fence markers in the prompt, but the raw
    fed-back context is still there (feedback_context stays True). If the
    'Connect retry' assertion ever fails, the extractor stopped passing
    recent_context through -- that would be a different bug."""
    client = RecordingClient(TURN20_PATCHES)
    m = _seeded_memory(client)
    m.add_turn("assistant", TURN20_TEXT)
    prompt = _prompt_text(client)
    assert _FEEDBACK_FENCE_OPEN not in prompt
    assert _FEEDBACK_FENCE_CLOSE not in prompt
    assert "Connect retry" in prompt


# -- defaults ----------------------------------------------------------------

def test_new_flags_default_off_and_telemetry_keys_exposed():
    """Construction without the new kwargs must preserve existing
    behavior: feedback off by default, both mitigations off, and both new
    telemetry keys present at zero in stats (they flow through
    result.update())."""
    m = DSPMMemory(budget=60)
    assert m.feedback_context is False
    assert m.feedback_fence is False
    assert m.echo_guard is False
    assert m.stats["llm_echo_drops"] == 0
    assert m.stats["llm_unattributed_drops"] == 0
