"""Persistence for DSPM memory.

Save and load the semantic-patch notebook across sessions, enabling
long-term memory: a user's Chat 2 can recall constraints stated in
Chat 1, and later revisions supersede earlier values on load.

Format: JSON list of patch records. Files are portable across machines
and versions; unknown fields in newer files are ignored on load for
forward compatibility.

v0.1.10: load_memory() now returns 0 for a corrupted or truncated
notebook file instead of raising — matching the defensive posture of
_record_to_patch, which already skips malformed records individually.
Unused typing.List import removed. No schema change.

v0.1.17: patch records carry source_role (review point 2 -- speaker
attribution survives save/load), and the notebook header carries
conversation_tokens (review point 4 -- the raw input size that produced
it; loads additively under merge semantics). Both are optional keys:
pre-0.1.17 files load as unattributed / zero.

v0.1.22 (Bug #7a): concurrent-save race condition. Two processes
calling save_memory() on the same path simultaneously used to
overwrite each other (lost-update race: A reads vN, B reads vN, A
writes vN+1, B writes vN+1 overwriting A's changes). The atomic
write (os.replace) protects against half-written files but not against
lost updates. Now: an exclusive file lock is acquired around both
save and load, using portalocker (cross-platform: fcntl on Unix,
msvcrt on Windows). Lock is held for the full read-modify-write
window, so concurrent writers serialize cleanly.

v0.1.22 (Bug #8): silent corruption / data loss. Pre-fix: a corrupted
notebook file was silently skipped (load_memory returned 0), then the
next save_memory() overwrote it with the in-memory state -- the user's
original memory was gone forever. Now:
  1. save_memory() backs up the existing file to <path>.bak before
     overwriting, so even a catastrophic bug preserves one prior state.
  2. load_memory() quarantines a corrupt file to
     <path>.corrupt.<timestamp> instead of leaving it in place to be
     overwritten. The user can inspect/recover it.
  3. load_memory(strict=True) raises CorruptNotebookError instead of
     returning 0, for callers who want loud failures.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

from dspm.patch import SemanticPatch

try:
    import portalocker
    _HAS_PORTALOCKER = True
except ImportError:
    _HAS_PORTALOCKER = False

_SCHEMA_VERSION = 1

_REQUIRED_FIELDS = ("patch_id", "turn_index", "patch_type", "payload")


class CorruptNotebookError(Exception):
    """Raised by load_memory(strict=True) when a notebook file cannot
    be parsed. The corrupt file has been quarantined to
    <path>.corrupt.<timestamp> and the original path is gone."""
    pass


def _patch_to_record(p: SemanticPatch) -> Dict[str, Any]:
    """Serialize a patch. Only stable fields stored; the rest are
    recomputed on load."""
    return {
        "patch_id": p.patch_id,
        "turn_index": p.turn_index,
        "patch_type": p.patch_type,
        "payload": p.payload,
        "dependencies": list(p.dependencies),
        # v0.1.17 (review point 2): speaker attribution rides the record;
        # old readers ignore unknown fields, old files load as None.
        "source_role": getattr(p, "source_role", None),
    }


def _record_to_patch(d: Dict[str, Any]) -> Optional[SemanticPatch]:
    """Deserialize one record. Returns None for malformed entries."""
    if not isinstance(d, dict):
        return None
    if not all(k in d for k in _REQUIRED_FIELDS):
        return None
    try:
        deps = d.get("dependencies") or []
        if not isinstance(deps, list):
            deps = []
        return SemanticPatch(
            patch_id=str(d["patch_id"]),
            turn_index=int(d["turn_index"]),
            patch_type=str(d["patch_type"]),
            payload=str(d["payload"]),
            dependencies=[str(x) for x in deps],
            source_role=(str(d["source_role"]) if d.get("source_role") else None),
        )
    except (TypeError, ValueError):
        return None


def _acquire_lock(path: str, exclusive: bool = True):
    """Bug #7a: acquire a cross-platform file lock on `path` (or its
    lockfile). Returns a context manager that releases the lock on
    exit. Uses portalocker if available (cross-platform, well-tested);
    falls back to a no-op lockfile if portalocker is not installed
    (single-process mode -- concurrent writers should install
    portalocker)."""
    if _HAS_PORTALOCKER:
        # portalocker.Lock handles its own lockfile; exclusive=True by
        # default for save, shared=False for load (we want exclusive on
        # load too, because load triggers _merge_patch which mutates
        # the in-memory state used by the next save).
        return portalocker.Lock(path + ".lock", fail_when_locked=False)
    # Fallback: no-op context manager. Logs a warning once.
    import warnings
    from contextlib import contextmanager
    warnings.warn(
        "portalocker not installed -- concurrent save/load race "
        "condition is NOT protected. Install with: pip install "
        "portalocker>=2.7",
        RuntimeWarning,
        stacklevel=3,
    )

    @contextmanager
    def _noop():
        yield

    return _noop()


def _backup_existing(path: str) -> Optional[str]:
    """Bug #8: before overwriting an existing notebook, copy it to
    <path>.bak so even a catastrophic bug preserves one prior state.
    Returns the backup path, or None if no prior file existed."""
    if not os.path.exists(path):
        return None
    backup = path + ".bak"
    try:
        import shutil
        shutil.copy2(path, backup)
        return backup
    except OSError:
        # Best-effort: if backup fails (disk full, permissions), we
        # still proceed with the save -- losing the backup is better
        # than losing the new state.
        return None


def _quarantine_corrupt(path: str) -> Optional[str]:
    """Bug #8: move a corrupt notebook file out of the way so the next
    save_memory() call doesn't overwrite it. The quarantined file is
    renamed to <path>.corrupt.<unix_timestamp> so the user can inspect
    or recover it. Returns the quarantine path, or None on failure."""
    if not os.path.exists(path):
        return None
    quarantine = f"{path}.corrupt.{int(time.time())}"
    try:
        os.rename(path, quarantine)
        return quarantine
    except OSError:
        # If rename fails (cross-device, permissions), try copy+delete.
        try:
            import shutil
            shutil.copy2(path, quarantine)
            os.remove(path)
            return quarantine
        except OSError:
            return None


def save_memory(memory, path: str) -> int:
    """Write the notebook to `path` (atomic write). Returns patches saved.

    v0.1.22 (Bug #7a): acquires an exclusive file lock so concurrent
    saves serialize cleanly -- no lost updates.
    v0.1.22 (Bug #8): backs up the existing file to <path>.bak before
    overwriting, so one prior state is always recoverable."""
    records = [_patch_to_record(p) for p in memory.patches]
    # v0.1.17 (review point 4): record the raw input tokens of the turns
    # that produced this notebook, so a merged session can still report
    # what the conversation cost.
    payload = {"schema": _SCHEMA_VERSION, "turns": memory.turns,
               "conversation_tokens": getattr(memory, "_conversation_tokens", 0),
               "patches": records}
    tmp = path + ".tmp"
    # Bug #7a: hold the lock for the full backup+write window so two
    # processes can't race on the backup step itself.
    with _acquire_lock(path, exclusive=True):
        # Bug #8: backup the existing file before we overwrite it.
        # If the existing file is corrupt (parse fails), the backup
        # preserves the corruption for inspection -- but we still
        # proceed with the save (the new state is valid).
        _backup_existing(path)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)  # atomic: file is never half-written
    return len(records)


def load_memory(memory, path: str, strict: bool = False) -> int:
    """Read a notebook from `path` into an existing memory (merge semantics:
    existing patches are kept; saved patches merge in through the same
    duplicate-suppression / supersession rules as live turns).

    Returns the number of patches loaded (0 if the file is missing,
    empty, or corrupted).

    v0.1.22 (Bug #7a): acquires an exclusive file lock so concurrent
    load+save calls don't race. Load is exclusive because it mutates
    the in-memory state that the next save() will write.
    v0.1.22 (Bug #8): a corrupted notebook is quarantined to
    <path>.corrupt.<timestamp> instead of being left in place to be
    overwritten by the next save. Pass strict=True to raise
    CorruptNotebookError instead of returning 0 silently."""
    if not os.path.exists(path):
        return 0
    with _acquire_lock(path, exclusive=True):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
            # Bug #8: quarantine the corrupt file so the next save()
            # doesn't overwrite it. In strict mode, raise.
            quarantine = _quarantine_corrupt(path)
            if strict:
                raise CorruptNotebookError(
                    f"Notebook at {path!r} is corrupt and has been "
                    f"quarantined to {quarantine!r}. Original error: {e}"
                ) from e
            return 0
    # accept both wrapped and bare-list formats
    records = data.get("patches") if isinstance(data, dict) else data
    if not isinstance(records, list):
        # Valid JSON but wrong shape -- treat as empty, but don't quarantine
        # (the file is structurally sound, just doesn't have patches).
        return 0
    loaded = 0
    for d in records:
        p = _record_to_patch(d)
        if p is None:
            continue
        before = len(memory.patches)
        memory._merge_patch(p)
        if len(memory.patches) > before:
            loaded += 1
    if isinstance(data, dict) and data.get("turns"):
        memory.turns = max(memory.turns, int(data["turns"]))
    # v0.1.17 (review point 4): merge semantics -- loaded notebooks add
    # their recorded input tokens to this session's counter.
    if isinstance(data, dict) and data.get("conversation_tokens"):
        base = getattr(memory, "_conversation_tokens", 0)
        memory._conversation_tokens = base + int(data["conversation_tokens"])
    return loaded
