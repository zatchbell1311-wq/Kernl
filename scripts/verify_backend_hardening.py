"""
Standalone verification for Bug #7a, #7b, #8 (v0.1.22 backend hardening).

Run after copying the updated files:

    python scripts/verify_backend_hardening.py
"""

import asyncio
import json
import os
import tempfile
import threading
import time

from dspm import DSPMMemory
from dspm.patch import SemanticPatch
from dspm.persistence import save_memory, load_memory, CorruptNotebookError


def banner(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ── Bug #8: silent corruption ───────────────────────────────────

banner("BUG #8 - Silent corruption / data loss")

with tempfile.TemporaryDirectory() as tmpdir:
    path = os.path.join(tmpdir, "mem.json")

    # Test 1: corrupt file is quarantined
    print("\n  Test 1: corrupt file quarantined (not overwritten)")
    with open(path, "w", encoding="utf-8") as f:
        f.write('{"patches": [INVALID JSON')
    mem = DSPMMemory(budget=250)
    loaded = load_memory(mem, path)
    assert loaded == 0, f"should return 0 for corrupt: {loaded}"
    assert not os.path.exists(path), "corrupt file should be quarantined"
    quarantined = [f for f in os.listdir(tmpdir) if f.startswith("mem.json.corrupt.")]
    assert len(quarantined) == 1, f"expected 1 quarantine file, got {quarantined}"
    print(f"    PASS: quarantined to {quarantined[0]}")

    # Test 2: strict mode raises
    print("\n  Test 2: strict mode raises CorruptNotebookError")
    with open(path, "w", encoding="utf-8") as f:
        f.write('{"patches": [INVALID')
    mem2 = DSPMMemory(budget=250)
    try:
        load_memory(mem2, path, strict=True)
        print("    FAIL: should have raised")
    except CorruptNotebookError as e:
        print(f"    PASS: raised CorruptNotebookError: {str(e)[:80]}...")

    # Test 3: backup created on save
    print("\n  Test 3: save creates .bak before overwrite")
    mem3 = DSPMMemory(budget=250)
    mem3._merge_patch(SemanticPatch('p0', 0, 'constraint',
                                    'Webhook timeout 30 seconds', []))
    save_memory(mem3, path)
    mem4 = DSPMMemory(budget=250)
    mem4._merge_patch(SemanticPatch('p0', 0, 'constraint',
                                    'Webhook timeout 10 seconds', []))
    save_memory(mem4, path)
    backup = path + ".bak"
    assert os.path.exists(backup), "backup file missing"
    with open(backup) as f:
        bak_data = json.load(f)
    bak_payloads = [p["payload"] for p in bak_data["patches"]]
    assert any("30 seconds" in p for p in bak_payloads), \
        f"backup should contain old state: {bak_payloads}"
    print(f"    PASS: backup at {backup} contains old state")

print("\n  All Bug #8 tests passed.")


# ── Bug #7a: concurrent save race ──────────────────────────────

banner("BUG #7a - Concurrent save race condition")

with tempfile.TemporaryDirectory() as tmpdir:
    path = os.path.join(tmpdir, "mem.json")
    mem0 = DSPMMemory(budget=250)
    save_memory(mem0, path)

    mem_a = DSPMMemory(budget=250)
    mem_a._merge_patch(SemanticPatch('a', 0, 'constraint',
                                     'Constraint A from thread A', []))
    mem_b = DSPMMemory(budget=250)
    mem_b._merge_patch(SemanticPatch('b', 0, 'constraint',
                                     'Constraint B from thread B', []))

    errors = []

    def save_thread(mem, name):
        try:
            save_memory(mem, path)
        except Exception as e:
            errors.append((name, e))

    t1 = threading.Thread(target=save_thread, args=(mem_a, "A"))
    t2 = threading.Thread(target=save_thread, args=(mem_b, "B"))
    t1.start(); t2.start(); t1.join(); t2.join()

    assert not errors, f"concurrent saves failed: {errors}"
    with open(path) as f:
        final = json.load(f)
    payloads = [p["payload"] for p in final["patches"]]
    assert len(payloads) == 1, \
        f"expected 1 patch (one writer won), got {len(payloads)}"
    print(f"  PASS: concurrent saves serialized cleanly, final state valid")
    print(f"  Final payload: {payloads[0]!r}")


# ── Bug #7b: async API ─────────────────────────────────────────

banner("BUG #7b - Async API (async_add_turn)")


class SyncMockClient:
    """Sync client -- async_add_turn wraps in asyncio.to_thread."""
    def __init__(self):
        self.content = json.dumps([
            {"patch_type": "constraint", "payload": "Always return JSON.",
             "patch_id": "p1", "dependencies": []},
        ])
        class MockResponse:
            def __init__(self, content):
                self.choices = [type('Choice', (), {
                    'message': type('Message', (), {'content': content}),
                    'finish_reason': 'stop',
                })]
        class Completions:
            @staticmethod
            def create(**kwargs):
                return MockResponse(SyncMockClient.content if False else None)
        # Fix the closure
        class Completions2:
            @staticmethod
            def create(**kwargs):
                return MockResponse(SyncMockClient().content)
        class Chat:
            completions = Completions2()
        self.chat = Chat()


class AsyncMockClient:
    """Async client -- async_add_turn awaits create() directly."""
    def __init__(self):
        class MockResponse:
            def __init__(self, content):
                self.choices = [type('Choice', (), {
                    'message': type('Message', (), {'content': content}),
                    'finish_reason': 'stop',
                })]

        content = json.dumps([
            {"patch_type": "constraint", "payload": "Always return JSON.",
             "patch_id": "p1", "dependencies": []},
        ])

        class AsyncCompletions:
            @staticmethod
            async def create(**kwargs):
                return MockResponse(content)
        class AsyncChat:
            completions = AsyncCompletions()
        self.chat = AsyncChat()


# Test 1: async with sync client (to_thread path)
print("\n  Test 1: async_add_turn with sync client (to_thread path)")


async def test_sync_client():
    client = SyncMockClient()
    mem = DSPMMemory(budget=250, llm_client=client, model='test')
    patches = await mem.async_add_turn('user', 'Return JSON output.')
    return patches


patches = asyncio.run(test_sync_client())
assert len(patches) >= 1, f"sync client path failed: {patches}"
print(f"    PASS: extracted {len(patches)} patch(es)")


# Test 2: async with async client (native await path)
print("\n  Test 2: async_add_turn with async client (native await)")


async def test_async_client():
    client = AsyncMockClient()
    mem = DSPMMemory(budget=250, llm_client=client, model='test')
    patches = await mem.async_add_turn('user', 'Return JSON output.')
    return patches


patches = asyncio.run(test_async_client())
assert len(patches) >= 1, f"async client path failed: {patches}"
print(f"    PASS: extracted {len(patches)} patch(es)")


# Test 3: concurrent calls overlap (the whole point)
print("\n  Test 3: two concurrent async_add_turn calls overlap")


class SlowAsyncClient:
    def __init__(self):
        class MockResponse:
            def __init__(self, content):
                self.choices = [type('Choice', (), {
                    'message': type('Message', (), {'content': content}),
                    'finish_reason': 'stop',
                })]
        content = json.dumps([
            {"patch_type": "entity", "payload": "Noted.",
             "patch_id": "x", "dependencies": []},
        ])

        class AsyncCompletions:
            @staticmethod
            async def create(**kwargs):
                await asyncio.sleep(0.5)
                return MockResponse(content)
        class AsyncChat:
            completions = AsyncCompletions()
        self.chat = AsyncChat()


async def test_concurrent():
    client = SlowAsyncClient()
    mem1 = DSPMMemory(budget=250, llm_client=client, model='m')
    mem2 = DSPMMemory(budget=250, llm_client=client, model='m')
    start = time.time()
    results = await asyncio.gather(
        mem1.async_add_turn('user', 'Turn 1'),
        mem2.async_add_turn('user', 'Turn 2'),
    )
    return results, time.time() - start


results, elapsed = asyncio.run(test_concurrent())
assert elapsed < 0.95, \
    f"FAIL: async calls serialized ({elapsed:.2f}s >= 1.0s)"
print(f"    PASS: both calls completed in {elapsed:.2f}s (concurrent)")


print("\n" + "=" * 70)
print("ALL VERIFICATIONS PASSED")
print("=" * 70)
print("\nFiles to copy to your repo:")
print("  persistence.py  -> src/dspm/persistence.py")
print("  extractor.py    -> src/dspm/extractor.py")
print("  memory.py       -> src/dspm/memory.py")
print("  pyproject.toml  -> pyproject.toml (bumps version to 0.1.22)")
print("  test_backend_hardening.py -> tests/test_backend_hardening.py")
print("\nThen run:")
print("  pip install portalocker>=2.7")
print("  python -m pytest tests -q --basetemp=\"$env:TEMP\\dspm-pytest\"")
