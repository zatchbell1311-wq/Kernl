"""Persistence tests - all offline, no LLM client needed."""
import json
import os

import pytest

from dspm import DSPMMemory
from dspm.patch import SemanticPatch


def _mk_patch(pid, turn, ptype, payload):
    return SemanticPatch(patch_id=pid, turn_index=turn,
                         patch_type=ptype, payload=payload, dependencies=[])


class TestSaveLoad:
    def test_save_creates_valid_json(self, tmp_path):
        m = DSPMMemory(budget=250)
        m._merge_patch(_mk_patch("p0_0", 0, "constraint", "must work offline"))
        m._merge_patch(_mk_patch("p1_0", 1, "decision", "deadline Oct 20"))
        path = str(tmp_path / "mem.json")
        n = m.save(path)
        assert n == 2
        with open(path) as f:
            data = json.load(f)
        assert data["schema"] == 1
        assert len(data["patches"]) == 2

    def test_roundtrip_restores_criticals(self, tmp_path):
        m1 = DSPMMemory(budget=250)
        m1._merge_patch(_mk_patch("p0_0", 0, "constraint", "p95 under 200ms"))
        m1._merge_patch(_mk_patch("p0_1", 0, "decision", "use PostgreSQL 15"))
        path = str(tmp_path / "mem.json")
        m1.save(path)

        m2 = DSPMMemory(budget=250)
        loaded = m2.load(path)
        assert loaded == 2
        assert m2.stats["critical_total"] == 2
        payloads = {p.payload for p in m2.all_patches}
        assert "p95 under 200ms" in payloads
        assert "use PostgreSQL 15" in payloads

    def test_load_missing_file_returns_zero(self, tmp_path):
        m = DSPMMemory(budget=250)
        assert m.load(str(tmp_path / "nope.json")) == 0

    def test_loaded_memory_compresses_and_keeps_crr(self, tmp_path):
        """End-to-end: save -> load -> get_context with NO llm_client."""
        m1 = DSPMMemory(budget=250)
        m1._merge_patch(_mk_patch("p0_0", 0, "constraint", "must work offline"))
        m1._merge_patch(_mk_patch("p0_1", 0, "constraint", "deadline Oct 20"))
        m1._merge_patch(_mk_patch("p1_0", 1, "decision", "use FastAPI"))
        path = str(tmp_path / "mem.json")
        m1.save(path)

        m2 = DSPMMemory(budget=250)          # NO llm_client needed
        m2.load(path)
        ctx = m2.get_context(query="hard requirements")
        assert "offline" in ctx.lower()
        assert "oct 20" in ctx.lower()
        assert m2.stats["crr"] == 100

    def test_cross_chat_revision_supersedes_on_load(self, tmp_path):
        """v0.1.3 supersession across sessions: Chat 2 revises a deadline
        loaded from Chat 1's file - the stale value must vanish."""
        m1 = DSPMMemory(budget=250)
        m1._merge_patch(_mk_patch("p0_0", 0, "constraint", "deadline Oct 20"))
        path = str(tmp_path / "mem.json")
        m1.save(path)

        m2 = DSPMMemory(budget=250)
        m2.load(path)
        m2._merge_patch(_mk_patch("p5_0", 5, "constraint",
                                  "deadline updated to Nov 5"))
        crits = [p.payload for p in m2.critical_patches]
        assert len(crits) == 1
        assert "Nov 5" in crits[0]
        assert not any("Oct 20" in c for c in crits)

    def test_load_skips_malformed_records(self, tmp_path):
        path = str(tmp_path / "bad.json")
        with open(path, "w") as f:
            json.dump({"schema": 1, "patches": [
                {"patch_id": "a", "turn_index": 0,
                 "patch_type": "constraint", "payload": "ok"},
                {"garbage": True},
                [1, 2, 3],
            ]}, f)
        m = DSPMMemory(budget=250)
        loaded = m.load(path)
        assert loaded == 1

    def test_load_corrupt_json_returns_zero(self, tmp_path):
        """v0.1.10: a truncated/corrupted notebook file returns 0 instead
        of raising — matching the defensive posture of _record_to_patch.
        (Atomic writes make this rare, but load() pointing at any damaged
        file shouldn't crash the app.)"""
        path = str(tmp_path / "corrupt.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write('{"schema": 1, "patches": [trunc')
        m = DSPMMemory(budget=250)
        assert m.load(path) == 0

    def test_atomic_save_leaves_no_tmp(self, tmp_path):
        m = DSPMMemory(budget=250)
        m._merge_patch(_mk_patch("p0_0", 0, "entity", "Stripe"))
        path = str(tmp_path / "mem.json")
        m.save(path)
        assert not os.path.exists(path + ".tmp")

    def test_overwrite_save(self, tmp_path):
        m = DSPMMemory(budget=250)
        m._merge_patch(_mk_patch("p0_0", 0, "entity", "Stripe"))
        path = str(tmp_path / "mem.json")
        m.save(path)
        m._merge_patch(_mk_patch("p1_0", 1, "entity", "Redis"))
        m.save(path)
        m2 = DSPMMemory(budget=250)
        assert m2.load(path) == 2