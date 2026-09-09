"""User-facing DSPM memory API.

The memory object provides a high-level API for adding conversation turns,
extracting semantic patches from an LLM-compatible client, storing them,
compressing them, and returning a context string within a token budget.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Optional, Sequence

from dspm.config import PATCH_TYPES, CRITICAL_TYPES, DEFAULT_BUDGET, REVISION_OVERLAP
from dspm.engine import DSPMEngine
from dspm.extractor import extract_turn
from dspm.patch import SemanticPatch, count_tokens


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
        """Add a conversation turn and return the newly extracted patches.

        The method raises ValueError if no LLM client is supplied, because
        extraction requires an llm_client.chat.completions.create call.
        """
        if self.llm_client is None:
            raise ValueError("llm_client is required to call add_turn()")

        patches = extract_turn(self.llm_client, self.model, text, self.turns, recent_context=self._last_context)
        for p in patches:
            self._merge_patch(p)
        self.turns += 1
        return patches

    def _merge_patch(self, patch: SemanticPatch) -> None:
        """Merge a patch into memory with duplicate suppression and critical revision superseding."""
        for existing in self.patches:
            if existing.patch_type == patch.patch_type and existing.payload == patch.payload:
                return
        # critical superseding by overlap threshold
        if patch.is_critical:
            for existing in self.patches:
                if existing.is_critical and existing.patch_type == patch.patch_type:
                    overlap = self._jaccard(existing.payload, patch.payload)
                    if overlap >= REVISION_OVERLAP:
                        self.patches.remove(existing)
                        break
        self.patches.append(patch)

    def _jaccard(self, left: str, right: str) -> float:
        """Return the Jaccard overlap between canonicalized word sets."""
        a = set(left.lower().split())
        b = set(right.lower().split())
        if not a and not b:
            return 1.0
        return len(a & b) / len(a | b) if (a | b) else 0.0

    def get_context(self, query: str = "") -> str:
        """Return the compressed context string produced by the DSPM engine."""
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
        context_tokens = sum(count_tokens(p.to_prompt_str()) for p in self.selected_patches)
        crr = 100 if critical_total == 0 else int((critical_selected / critical_total) * 100)
        # token reduction rate; a rough deterministic estimate
        trr = 100 - int((context_tokens / max(1, raw_tokens)) * 100) if raw_tokens else 0
        return {
            "turns": self.turns,
            "total_patches": len(self.patches),
            "critical_total": critical_total,
            "critical_selected": critical_selected,
            "crr": crr,
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
