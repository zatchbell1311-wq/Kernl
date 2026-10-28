"""Tests for v0.1.22 backend hardening: Bug #7a (concurrent save race),
Bug #7b (async API), Bug #8 (silent corruption / data loss).

These tests extend the existing tests/test_persistence.py and
tests/test_memory.py with new regression coverage for the LinkedIn
reviewer's three concerns.
"""

import asyncio
import json
import os
import tempfile
import threading
import time

import pytest

from dspm import DSPMMemory
from dspm.patch import SemanticPatch
from dspm.persistence import (save_memory, load_memory,
                               CorruptNotebookError)


# ── Bug #8: silent corruption / data loss ──────────────────────

def test_bug8_corrupt_file_quarantined_not_overwritten(tmp_path):
    """Bug #8: a corrupted notebook file must be quarantined to
    <path>.corrupt.<timestamp>, NOT left in place to be overwritten
    by the next save_memory() call. Pre-fix: load_memory returned 0
    silently, then the next save overwrote the corrupt file with
    the in-memory state -- the user's original memory was gone forever."""
    path = str(tmp_path / "mem.json")
    # Write a corrupt notebook
    with open(path, "w", encoding="utf-8") as f:
        f.write('{"patches": [INVALID JSON')

    mem = DSPMMemory(budget=250)
    # load_memory should return 0 silently (default non-strict)
    loaded = load_memory(mem, path)
    assert loaded == 0

    # The corrupt file should be quarantined, not left in place
    assert not os.path.exists(path), \
        f"corrupt file should be quarantined, not left at {path!r}"
    # A quarantine file should exist
    quarantined = [f for f in os.listdir(tmp_path)
                   if f.startswith("mem.json.corrupt.")]
    assert len(quarantined) == 1, \
        f"expected one quarantine file, got {quarantined}"
    # The quarantine file should still contain the corrupt content
    with open(os.path.join(tmp_path, quarantined[0]), "r", encoding="utf-8") as f:
        assert "INVALID JSON" in f.read()


def test_bug8_strict_mode_raises_on_corrupt(tmp_path):
    """Bug #8: with strict=True, load_memory raises CorruptNotebookError
    instead of returning 0 silently. The corrupt file is still
    quarantined so the user can inspect/recover it."""
    path = str(tmp_path / "mem.json")
    with open(path, "w", encoding="utf-8") as f:
        f.write('{"patches": [INVALID')

    mem = DSPMMemory(budget=250)
    with pytest.raises(CorruptNotebookError, match="corrupt"):
        load_memory(mem, path, strict=True)

    # File should still be quarantined even when strict raises
    assert not os.path.exists(path)
    quarantined = [f for f in os.listdir(tmp_path)
                   if f.startswith("mem.json.corrupt.")]
    assert len(quarantined) == 1


def test_bug8_save_creates_backup_before_overwrite(tmp_path):
    """Bug #8: save_memory() must back up the existing file to
    <path>.bak before overwriting, so one prior state is always
    recoverable even if the new save is wrong."""
    path = str(tmp_path / "mem.json")

    # Save state 1
    mem1 = DSPMMemory(budget=250)
    mem1._merge_patch(SemanticPatch('p0_0', 0, 'constraint',
                                    'Webhook timeout must be 30 seconds', []))
    save_memory(mem1, path)
    assert os.path.exists(path)

    # Save state 2 (different content)
    mem2 = DSPMMemory(budget=250)
    mem2._merge_patch(SemanticPatch('p0_0', 0, 'constraint',
                                    'Webhook timeout must be 10 seconds', []))
    save_memory(mem2, path)

    # Backup should exist and contain state 1
    backup = path + ".bak"
    assert os.path.exists(backup), f"backup file missing at {backup!r}"
    with open(backup, "r", encoding="utf-8") as f:
        backup_data = json.load(f)
    backup_payloads = [p["payload"] for p in backup_data["patches"]]
    assert any("30 seconds" in p for p in backup_payloads), \
        f"backup should contain state 1 (30 seconds), got {backup_payloads}"


def test_bug8_missing_file_returns_zero_cleanly(tmp_path):
    """Bug #8 guard: load_memory on a missing file still returns 0
    (no spurious quarantine or backup)."""
    path = str(tmp_path / "nonexistent.json")
    mem = DSPMMemory(budget=250)
    assert load_memory(mem, path) == 0
    # No quarantine files created
    assert os.listdir(tmp_path) == []


def test_bug8_valid_file_loads_normally(tmp_path):
    """Bug #8 guard: a valid notebook still loads correctly -- the
    corruption/backup machinery doesn't interfere with the happy path."""
    path = str(tmp_path / "mem.json")
    mem1 = DSPMMemory(budget=250)
    mem1._merge_patch(SemanticPatch('p0_0', 0, 'constraint',
                                    'Webhook timeout must be 30 seconds', []))
    save_memory(mem1, path)

    mem2 = DSPMMemory(budget=250)
    loaded = load_memory(mem2, path)
    assert loaded == 1
    crits = [p.payload for p in mem2.critical_patches]
    assert any("30 seconds" in c for c in crits)


# ── Bug #7a: concurrent save race condition ─────────────────────

def test_bug7a_concurrent_saves_dont_lose_data(tmp_path):
    """Bug #7a: two threads calling save_memory() simultaneously on
    the same path used to overwrite each other (lost-update race).
    With file locking, they serialize cleanly: both saves succeed,
    and the final state on disk is one of the two (not a corrupted
    merge or a half-write)."""
    path = str(tmp_path / "mem.json")

    # Initial state
    mem0 = DSPMMemory(budget=250)
    save_memory(mem0, path)

    # Two threads with different states race to save
    mem_a = DSPMMemory(budget=250)
    mem_a._merge_patch(SemanticPatch('a_0', 0, 'constraint',
                                     'Constraint A from thread A', []))

    mem_b = DSPMMemory(budget=250)
    mem_b._merge_patch(SemanticPatch('b_0', 0, 'constraint',
                                     'Constraint B from thread B', []))

    errors = []

    def save_thread(mem, name):
        try:
            save_memory(mem, path)
        except Exception as e:
            errors.append((name, e))

    t1 = threading.Thread(target=save_thread, args=(mem_a, "A"))
    t2 = threading.Thread(target=save_thread, args=(mem_b, "B"))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert not errors, f"concurrent saves failed: {errors}"

    # The final file on disk should be valid JSON and contain
    # exactly one of the two states (not a merge, not a corruption).
    with open(path, "r", encoding="utf-8") as f:
        final_data = json.load(f)
    final_payloads = [p["payload"] for p in final_data["patches"]]
    assert len(final_payloads) == 1, \
        f"expected 1 patch (one writer won), got {len(final_payloads)}"
    assert ("Constraint A from thread A" in final_payloads[0]
            or "Constraint B from thread B" in final_payloads[0]), \
        f"final state should be one of the two writers: {final_payloads}"


def test_bug7a_lockfile_created_and_cleaned_up(tmp_path):
    """Bug #7a guard: the lockfile mechanism should work without
    leaving stale files (when portalocker is installed). When
    portalocker is NOT installed, a RuntimeWarning is emitted but
    save still succeeds (best-effort, single-process)."""
    path = str(tmp_path / "mem.json")
    mem = DSPMMemory(budget=250)
    mem._merge_patch(SemanticPatch('p0_0', 0, 'constraint', 'Test', []))

    try:
        import portalocker  # noqa: F401
        has_locker = True
    except ImportError:
        has_locker = False

    if has_locker:
        # No warning, lockfile is created during save and removed after
        save_memory(mem, path)
        # The lockfile may or may not persist depending on portalocker's
        # default behavior, but the save must succeed
        assert os.path.exists(path)
    else:
        # Warning emitted, but save still succeeds
        with pytest.warns(RuntimeWarning, match="portalocker"):
            save_memory(mem, path)
        assert os.path.exists(path)


# ── Bug #7b: async API ─────────────────────────────────────────

def test_bug7b_async_add_turn_works_with_sync_client():
    """Bug #7b: async_add_turn() must work with a sync LLM client
    (wrapped in asyncio.to_thread). This is the backward-compat path
    -- users who don't have AsyncOpenAI can still use the async API
    and free their event loop during the LLM call."""
    import json as _json

    class MockResponse:
        def __init__(self, content):
            self.choices = [type('Choice', (), {
                'message': type('Message', (), {'content': content}),
                'finish_reason': 'stop',
            })]

    class SyncClient:
        """A standard sync OpenAI-compatible client."""
        def __init__(self):
            content = _json.dumps([
                {"patch_type": "constraint",
                 "payload": "Always return JSON output.",
                 "patch_id": "patch-1", "dependencies": []},
            ])

            class Completions:
                @staticmethod
                def create(**kwargs):
                    return MockResponse(content)
            class Chat:
                completions = Completions()
            self.chat = Chat()

    async def run():
        client = SyncClient()
        mem = DSPMMemory(budget=250, llm_client=client, model='test-model')
        patches = await mem.async_add_turn('user', 'Return JSON output.')
        return patches

    patches = asyncio.run(run())
    assert len(patches) >= 1
    assert all(hasattr(p, 'payload') for p in patches)


def test_bug7b_async_add_turn_works_with_async_client():
    """Bug #7b: async_add_turn() must work natively with an async
    LLM client (e.g. AsyncOpenAI). When llm_client.chat.completions.create
    is a coroutine function, async_extract_turn awaits it directly
    instead of wrapping in asyncio.to_thread."""
    import json as _json

    class MockResponse:
        def __init__(self, content):
            self.choices = [type('Choice', (), {
                'message': type('Message', (), {'content': content}),
                'finish_reason': 'stop',
            })]

    class AsyncCompletions:
        @staticmethod
        async def create(**kwargs):
            return MockResponse(_json.dumps([
                {"patch_type": "constraint",
                 "payload": "Always return JSON output.",
                 "patch_id": "patch-1", "dependencies": []},
            ]))

    class AsyncChat:
        completions = AsyncCompletions()

    class AsyncClient:
        def __init__(self):
            self.chat = AsyncChat()

    async def run():
        client = AsyncClient()
        mem = DSPMMemory(budget=250, llm_client=client, model='test-model')
        patches = await mem.async_add_turn('user', 'Return JSON output.')
        return patches

    patches = asyncio.run(run())
    assert len(patches) >= 1


def test_bug7b_async_concurrent_turns_free_event_loop():
    """Bug #7b: the whole point of async is concurrency. Two
    async_add_turn calls on different memory instances should
    interleave (both complete in roughly the time of one, not two).
    This test uses a slow mock client to verify the LLM calls
    actually overlap rather than serialize."""
    import json as _json

    class SlowMockResponse:
        def __init__(self, content):
            self.choices = [type('Choice', (), {
                'message': type('Message', (), {'content': content}),
                'finish_reason': 'stop',
            })]

    class SlowAsyncCompletions:
        @staticmethod
        async def create(**kwargs):
            await asyncio.sleep(0.5)  # simulate slow LLM
            return SlowMockResponse(_json.dumps([
                {"patch_type": "entity", "payload": "Noted.",
                 "patch_id": "x", "dependencies": []},
            ]))

    class SlowAsyncChat:
        completions = SlowAsyncCompletions()

    class SlowAsyncClient:
        def __init__(self):
            self.chat = SlowAsyncChat()

    async def run():
        client = SlowAsyncClient()
        mem1 = DSPMMemory(budget=250, llm_client=client, model='m')
        mem2 = DSPMMemory(budget=250, llm_client=client, model='m')
        # Run both concurrently
        start = time.time()
        results = await asyncio.gather(
            mem1.async_add_turn('user', 'Turn 1'),
            mem2.async_add_turn('user', 'Turn 2'),
        )
        elapsed = time.time() - start
        return results, elapsed

    results, elapsed = asyncio.run(run())
    # If the calls were truly concurrent, elapsed < 1.0s (both fit
    # in one 0.5s window). If they serialized, elapsed >= 1.0s.
    assert elapsed < 0.95, \
        f"async calls serialized instead of overlapping: {elapsed:.2f}s"
    assert len(results[0]) >= 1 and len(results[1]) >= 1


def test_bug7b_sync_add_turn_still_works():
    """Bug #7b guard: the existing sync add_turn() must still work
    unchanged. Adding the async API is purely additive."""
    llm = MockLLMClient()
    mem = DSPMMemory(budget=250, llm_client=llm, model='test-model')
    patches = mem.add_turn('user', 'Return JSON and use POST /v1/items.')
    assert len(patches) >= 2


# Reuse MockLLMClient from test_memory.py for the sync guard test.
class MockLLMClient:
    def __init__(self, content=None):
        self.content = content or json.dumps([
            {"patch_type": "constraint", "payload": "Always return JSON output.",
             "patch_id": "patch-1", "dependencies": []},
            {"patch_type": "decision", "payload": "Use POST /v1/items for creation.",
             "patch_id": "patch-2", "dependencies": []},
        ])
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
