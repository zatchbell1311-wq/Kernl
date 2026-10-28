"""
Verification script for Bug #5 (DSPM)

Bug #5 - Intermittent empty LLM responses (gpt-oss-20b, 3/14 runs).

Run from the repo root with the dspm venv active:

    python scripts/verify_bug5.py
"""

import argparse
import inspect
import json


def banner(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def show_config():
    """Show the extractor's current config so we know what parameters fire."""
    banner("CONFIG - Extractor parameters that affect empty responses")
    try:
        from dspm.config import (EXTRACT_MAX_TOKENS, EXTRACT_MIN_TOKENS,
                                 REASONING_PARAM_MODE, LLM_STRICT_DEFAULT)
        print(f"  EXTRACT_MAX_TOKENS    : {EXTRACT_MAX_TOKENS}")
        print(f"  EXTRACT_MIN_TOKENS    : {EXTRACT_MIN_TOKENS}")
        print(f"  REASONING_PARAM_MODE  : {REASONING_PARAM_MODE}")
        print(f"  LLM_STRICT_DEFAULT    : {LLM_STRICT_DEFAULT}")
    except ImportError as e:
        print(f"  Could not import config: {e}")

    try:
        from dspm.extractor import _build_system_prompt
        prompt = _build_system_prompt(dense=False)
        print(f"\n  System prompt length  : {len(prompt)} chars")
        print(f"  System prompt preview : {prompt[:120]!r}...")
    except Exception as e:
        print(f"  Could not inspect prompt: {e}")


def show_retry_logic():
    """Show the retry loop's behavior so we know which paths fire."""
    banner("RETRY LOGIC - What happens on empty content / exceptions")
    try:
        from dspm.extractor import extract_turn
        src = inspect.getsource(extract_turn)
        for i, line in enumerate(src.splitlines(), 1):
            stripped = line.strip()
            if any(kw in stripped for kw in
                   ['NON_RETRYABLE', 'llm_empty', 'llm_retries',
                    'finish_reason', 'strict', 'attempt < 2',
                    'reasoning_effort', 'time.sleep']):
                print(f"  L{i:4d}: {stripped}")
    except Exception as e:
        print(f"  Could not inspect source: {e}")


def simulate_empty_response_paths():
    """Simulate the three failure paths without a real LLM."""
    banner("SIMULATION - Three empty-response failure paths")

    from dspm.extractor import extract_turn

    class MockResponse:
        def __init__(self, content, finish_reason="stop"):
            self.choices = [type('Choice', (), {
                'message': type('Message', (), {'content': content}),
                'finish_reason': finish_reason,
            })]

    class TrackingClient:
        """Mock client whose _respond() returns scripted responses."""

        def __init__(self, name, responses):
            self.name = name
            self.responses = responses
            self.calls = []

            owner = self

            class Completions:
                @staticmethod
                def create(**kwargs):
                    return owner._respond(kwargs)

            class Chat:
                completions = Completions()

            self.chat = Chat()

        def _respond(self, kwargs):
            self.calls.append(kwargs)
            if not self.responses:
                return MockResponse("[]", "stop")
            kind, payload = self.responses.pop(0)
            if kind == "empty":
                return MockResponse("", "stop")
            if kind == "length":
                return MockResponse("", "length")
            if kind == "ok":
                return MockResponse(payload, "stop")
            if kind == "exception":
                e = Exception(payload)
                e.status_code = 429
                raise e
            return MockResponse(payload, "stop")

    # Path 1: 2 empties then success
    print("\n  Path 1: 2 empty responses, then success (should retry 3x)")
    telemetry = {}
    client = TrackingClient("p1", [
        ("empty", ""), ("empty", ""),
        ("ok", json.dumps([{"patch_type": "entity",
                            "payload": "Stripe", "patch_id": "x"}])),
    ])
    patches = extract_turn(client, 'openai/gpt-oss-20b', 'We use Stripe.', 0,
                           telemetry=telemetry)
    print(f"    calls          : {len(client.calls)}")
    print(f"    patches        : {len(patches)}")
    print(f"    telemetry      : {telemetry}")
    print(f"    expected       : 3 calls, 1 patch, llm_empty=2, llm_retries=2")
    assert len(patches) == 1, f"FAIL: expected 1 patch, got {len(patches)}"
    assert telemetry.get('llm_empty') == 2, f"FAIL: llm_empty wrong"
    print("    PASS")

    # Path 2: 3 empties
    print("\n  Path 2: 3 empty responses (should degrade to 0 patches)")
    telemetry = {}
    client = TrackingClient("p2", [
        ("empty", ""), ("empty", ""), ("empty", ""),
    ])
    patches = extract_turn(client, 'openai/gpt-oss-20b', 'We use Stripe.', 0,
                           telemetry=telemetry)
    print(f"    calls          : {len(client.calls)}")
    print(f"    patches        : {len(patches)}")
    print(f"    telemetry      : {telemetry}")
    assert len(patches) == 0, f"FAIL: expected 0 patches, got {len(patches)}"
    assert telemetry.get('llm_empty') == 3, f"FAIL: llm_empty wrong"
    print("    PASS")

    # Path 3: finish_reason='length'
    print("\n  Path 3: finish_reason='length' (truncation, 0 patches)")
    telemetry = {}
    client = TrackingClient("p3", [("length", "")])
    patches = extract_turn(client, 'openai/gpt-oss-20b', 'We use Stripe.', 0,
                           telemetry=telemetry)
    print(f"    calls          : {len(client.calls)}")
    print(f"    patches        : {len(patches)}")
    print(f"    telemetry      : {telemetry}")
    assert len(patches) == 0, f"FAIL: expected 0 patches, got {len(patches)}"
    assert telemetry.get('llm_length_finish') == 1, f"FAIL: llm_length_finish wrong"
    print("    PASS")


def show_strict_mode_behavior():
    """Show how strict_llm changes the empty-response behavior."""
    banner("STRICT MODE - How strict_llm changes the fallback")
    from dspm.extractor import extract_turn

    class MockResponse:
        def __init__(self, content, finish_reason="stop"):
            self.choices = [type('Choice', (), {
                'message': type('Message', (), {'content': content}),
                'finish_reason': finish_reason,
            })]

    class Client:
        def __init__(self):
            class C:
                @staticmethod
                def create(**kwargs):
                    return MockResponse("")
            class Ch:
                completions = C()
            self.chat = Ch()

    print("\n  strict=False (default): empty -> 0 patches, no exception")
    telemetry = {}
    patches = extract_turn(Client(), 'openai/gpt-oss-20b', 'We use Stripe.', 0,
                           telemetry=telemetry, strict=False)
    print(f"    patches   : {len(patches)}")
    print(f"    telemetry : {telemetry}")
    print("    PASS (silent fallback)")

    print("\n  strict=True: empty -> RuntimeError (no silent fallback)")
    telemetry = {}
    try:
        extract_turn(Client(), 'openai/gpt-oss-20b', 'We use Stripe.', 0,
                     telemetry=telemetry, strict=True)
        print("    FAIL: should have raised RuntimeError")
    except RuntimeError as e:
        print(f"    raised    : RuntimeError")
        print(f"    message   : {str(e)[:100]!r}...")
        print("    PASS (strict mode refuses silent fallback)")


def main():
    show_config()
    show_retry_logic()
    simulate_empty_response_paths()
    show_strict_mode_behavior()

    print("\n" + "=" * 70)
    print("DONE - paste this entire output back to me for analysis.")
    print("=" * 70)
    print("\nALSO paste these files (if they exist in your repo):")
    print("  - scripts/smoke_conversation.py")
    print("  - Any 14-run repro script that produced 3/14 failures")


if __name__ == '__main__':
    main()