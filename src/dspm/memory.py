"""User-facing DSPM memory API."""

from __future__ import annotations

import copy
import json
import re
from typing import Any, Dict, List, Optional, Sequence

from dspm.config import PATCH_TYPES, CRITICAL_TYPES, DEFAULT_BUDGET, REVISION_OVERLAP
from dspm.engine import DSPMEngine
from dspm.extractor import extract_turn
from dspm.patch import SemanticPatch, count_tokens
from dspm.persistence import save_memory, load_memory

# v0.1.3: common function words excluded from revision matching.
_STOPWORDS = {
    "the", "a", "an", "to", "from", "for", "of", "in", "on", "at", "by", "as",
    "and", "or", "with", "must", "be", "is", "are", "was", "were", "will",
    "shall", "should", "this", "that", "it", "its", "all", "no", "not",
    "when", "if", "then", "than", "so", "we", "you",
}

# v0.1.3: verbs that signal a payload REVISES an earlier value. Revisions in
# real conversations announce themselves; complementary facts do not.
_REVISION_MARKERS = {
    "update", "updated", "change", "changed", "revise", "revised",
    "supersede", "superseded", "replace", "replaced", "instead",
    "raise", "raised", "lower", "lowered", "increase", "increased",
    "decrease", "decreased", "reduce", "reduced", "switch", "switched",
    "migrate", "migrated", "revert", "reverted",
}


def _normalize(word: str) -> str:
    """Lowercase + light stemming so 'sessions'~'session' and '30s'~'30' match."""
    w = word.lower()
    if len(w) > 3 and w.endswith("s"):
        return w[:-1]
    if len(w) > 2 and w.endswith("s") and any(c.isdigit() for c in w[:-1]):
        return w[:-1]          # "30s" -> "30"
    return w


def _content_words(text: str) -> set:
    """Robust content-word set: \\w+ tokens, stopwords removed, plurals normalized."""
    return {_normalize(w) for w in re.findall(r"\w+", text.lower())
            if w not in _STOPWORDS and len(w) > 1}


def _numbers(text: str) -> frozenset:
    """All numeric values in a payload (used for same-value collapse)."""
    return frozenset(re.findall(r"\d+(?:\.\d+)?", text))


def _has_revision_marker(text: str) -> bool:
    return bool(_REVISION_MARKERS & {w.lower() for w in re.findall(r"\w+", text)})


def _slot_head(text: str) -> str:
    """First meaningful topic word: >=3 chars, not a stopword, not a revision
    marker. Empty string if none found."""
    for w in re.findall(r"\w+", text.lower()):
        if len(w) >= 3 and w not in _STOPWORDS and w not in _REVISION_MARKERS:
            return _normalize(w)
    return ""


class DSPMMemory:
    """Main user-facing memory object for DSPM.

    Example:
        >>> memory = DSPMMemory(budget=250, llm_client=None, model="gpt-4o-mini")
        >>> memory.add_turn("user", "Return every API result as JSON and require auth.")
        >>> context = memory.get_context("What must the contract remember?")
        >>> print(context)
    """

    def __init__(self, budget: int = DEFAULT_BUDGET, llm_client: Optional[Any] = None, model: str = "gpt-4o-mini", **kwargs: Any):
        self.budget = budget
        self.llm_client = llm_client
        self.model = model
        self.engine = DSPMEngine(budget=budget)
        self.patches: List[SemanticPatch] = []
        self.turns = 0
        self.selected_patches: List[SemanticPatch] = []
        self._last_context = ""

    def add_turn(self, role: str, text: str) -> List[SemanticPatch]:
        """Add a conversation turn and return the newly extracted patches."""
        if self.llm_client is None:
            raise ValueError("llm_client is required to call add_turn()")

        patches = extract_turn(self.llm_client, self.model, text, self.turns, recent_context=self._last_context)
        for p in patches:
            self._merge_patch(p)
        self.turns += 1
        return patches

    # Supersession rules (accumulated v0.1.3 → v0.1.6):
    #   (a)  same type + >=3 shared words + revision verb
    #   (a2) same type + same topic anchor + revision verb + changed numbers
    #   (b)  same type + >=4 shared words + IDENTICAL number sets (restatement)
    #   (c)  cross-type + >=4 shared words, OR (v0.1.6) >=3 + revision verb
    #   (c2) cross-type + same anchor + revision verb + changed numbers
    # FIXED (v0.1.6): the cross-type rule required >=4 shared words with no
    # marker relaxation. A user's real test — constraint "Webhook timeout
    # must be 30 seconds" revised by decision "Updated webhook timeout
    # changed to 10 seconds instead" — shares only 3 content words, so BOTH
    # survived as contradictory criticals. Cross-type revisions now fire at
    # >=3 shared words when a revision verb is present (mirroring rule a),
    # plus a terse cross-type path (c2) mirroring a2.
    def _merge_patch(self, patch: SemanticPatch) -> None:
        """Merge a patch into memory with duplicate suppression and critical revision supersession."""
        # exact duplicate suppression (same type, same payload)
        for existing in self.patches:
            if existing.patch_type == patch.patch_type and existing.payload == patch.payload:
                return
        # cross-type exact duplicate criticals (same words, different type)
        if patch.is_critical:
            for existing in self.patches:
                if existing.is_critical and existing.fingerprint == patch.fingerprint:
                    return
        # critical supersession — collect ALL stale entries, then remove
        if patch.is_critical:
            new_words = _content_words(patch.payload)
            new_nums = _numbers(patch.payload)
            marker = _has_revision_marker(patch.payload)
            new_head = _slot_head(patch.payload)
            stale = []
            for existing in self.patches:
                if not existing.is_critical:
                    continue
                if existing.turn_index > patch.turn_index:
                    continue  # never let an older patch supersede a newer one
                shared = new_words & _content_words(existing.payload)
                same_type = existing.patch_type == patch.patch_type
                if same_type:
                    if marker and len(shared) >= 3:
                        stale.append(existing)          # (a) revision — rich overlap
                        continue
                    if marker and new_head and new_head == _slot_head(existing.payload) \
                            and new_nums != _numbers(existing.payload):
                        stale.append(existing)          # (a2) terse revision
                        continue
                    if len(shared) >= 4 and new_nums and new_nums == _numbers(existing.payload):
                        stale.append(existing)          # (b) same-value restatement
                        continue
                else:
                    # (c) cross-type revision — >=4 words, or >=3 with marker (v0.1.6)
                    if len(shared) >= 4 or (marker and len(shared) >= 3):
                        stale.append(existing)          # (c) cross-type revision
                        continue
                    # (c2) terse cross-type revision (v0.1.6, mirrors a2)
                    if marker and new_head and new_head == _slot_head(existing.payload) \
                            and new_nums != _numbers(existing.payload):
                        stale.append(existing)          # (c2) terse cross-type
                        continue
            for s in stale:
                self.patches.remove(s)
        self.patches.append(patch)

    def _jaccard(self, left: str, right: str) -> float:
        """Return the Jaccard overlap between canonicalized word sets."""
        a = set(left.lower().split())
        b = set(right.lower().split())
        if not a and not b:
            return 1.0
        return len(a & b) / len(a | b) if (a | b) else 0.0

    def _content_overlap(self, left: str, right: str) -> int:
        """Count shared normalized non-stopword tokens."""
        return len(_content_words(left) & _content_words(right))

    # FIXED (v0.1.5): engine budget synced from self.budget on every call.
    def get_context(self, query: str = "") -> str:
        """Return the compressed context string produced by the DSPM engine."""
        self.engine.budget = self.budget
        selected, diagnostics = self.engine.compress(self.patches, query, self.turns)
        self.selected_patches = selected
        context = self.engine.build_context(selected)
        self._last_context = context
        return context

    @property
    def critical_patches(self) -> List[SemanticPatch]:
        """List all constraint and decision patches stored in memory."""
        return [p for p in self.patches if p.is_critical]

    @property
    def all_patches(self) -> List[SemanticPatch]:
        """Return the list of every patch currently stored in memory."""
        return list(self.patches)

    @property
    def stats(self) -> Dict[str, Any]:
        """Return a summary of memory health, selected-critical retention, and token reduction rate."""
        critical_total = len(self.critical_patches)
        if self.selected_patches:
            critical_selected = len([p for p in self.selected_patches if p.is_critical])
        else:
            critical_selected = critical_total
        raw_tokens = sum(count_tokens(p.to_prompt_str()) for p in self.patches)
        context_tokens = count_tokens(self._last_context) if self._last_context else 0
        crr = 100 if critical_total == 0 else int((critical_selected / critical_total) * 100)
        trr = 100 - int((context_tokens / max(1, raw_tokens)) * 100) if raw_tokens else 0
        return {
            "turns": self.turns,
            "total_patches": len(self.patches),
            "critical_total": critical_total,
            "critical_selected": critical_selected,
            "crr": crr,
            "budget": self.budget,
            "raw_tokens": raw_tokens,
            "context_tokens": context_tokens,
            "trr": trr,
        }

    def reset(self) -> None:
        """Clear all stored memory and reset the context state."""
        self.patches.clear()
        self.selected_patches.clear()
        self.turns = 0
        self._last_context = ""
        self.engine.reset_ema()

    def set_budget(self, budget: int) -> None:
        """Update the token budget. Takes effect on the next get_context() call."""
        self.budget = int(budget)
        self.engine.budget = self.budget

    # ── v0.1.4: persistence ─────────────────────────────────────────

    def save(self, path: str) -> int:
        """Persist the memory notebook to a JSON file (atomic write)."""
        return save_memory(self, path)

    def load(self, path: str) -> int:
        """Load a notebook from a JSON file into this memory (merge semantics)."""
        return load_memory(self, path)