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
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from dspm.patch import SemanticPatch

_SCHEMA_VERSION = 1

_REQUIRED_FIELDS = ("patch_id", "turn_index", "patch_type", "payload")


def _patch_to_record(p: SemanticPatch) -> Dict[str, Any]:
    """Serialize a patch. Only stable fields stored; the rest are
    recomputed on load."""
    return {
        "patch_id": p.patch_id,
        "turn_index": p.turn_index,
        "patch_type": p.patch_type,
        "payload": p.payload,
        "dependencies": list(p.dependencies),
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
        )
    except (TypeError, ValueError):
        return None


def save_memory(memory, path: str) -> int:
    """Write the notebook to `path` (atomic write). Returns patches saved."""
    records = [_patch_to_record(p) for p in memory.patches]
    payload = {"schema": _SCHEMA_VERSION, "turns": memory.turns, "patches": records}
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)  # atomic: file is never half-written
    return len(records)


def load_memory(memory, path: str) -> int:
    """Read a notebook from `path` into an existing memory (merge semantics:
    existing patches are kept; saved patches merge in through the same
    duplicate-suppression / supersession rules as live turns).

    Returns the number of patches loaded (0 if the file is missing,
    empty, or corrupted)."""
    if not os.path.exists(path):
        return 0
    # v0.1.10: a truncated or corrupted notebook returns 0 rather than
    # raising — matches _record_to_patch, which already skips malformed
    # records individually. Atomic writes make this rare in practice,
    # but a user pointing load() at a damaged file shouldn't get a crash.
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return 0
    # accept both wrapped and bare-list formats
    records = data.get("patches") if isinstance(data, dict) else data
    if not isinstance(records, list):
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
    return loaded