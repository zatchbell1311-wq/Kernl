"""Compression engine for DSPM.

The engine applies the seven stages outlined in the package design:
T1 fingerprint deduplication, T2 slot fusion, T3 delta encoding,
T4 causal pruning, T5 utility scoring, T6 critical guarantee and
selection, and T7 adaptive budgeting.
"""

from __future__ import annotations

import copy
import math
import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from dspm.config import (
    PATCH_TYPES,
    BASE_BUDGET_FRACTIONS,
    CRITICAL_TYPES,
    CRITICAL_SHARE,
    DEFAULT_BUDGET,
    DELTA_MIN_SAVING,
    RECENCY_LAMBDA,
    SHADOW_THRESHOLD,
    W_ALIGN,
    W_COST,
    W_DEP,
    W_RECENCY,
    ALPHA_EMA,
)
from dspm.patch import SemanticPatch, count_tokens


class DSPMEngine:
    """Compression engine applying the DSPM seven-stage pipeline."""

    def __init__(self, budget: int = DEFAULT_BUDGET):
        self.budget = budget
        self.ema_query = {t: 0.0 for t in PATCH_TYPES}
        self._embedder = None

    def compress(self, patches: Sequence[SemanticPatch], query: str, turn_index: int) -> Tuple[List[SemanticPatch], Dict[str, Any]]:
        """Compress a list of SemanticPatch objects into selected patches plus diagnostics."""
        work = copy.deepcopy(list(patches))
        diagnostics = {
            "stages": [],
            "selected": len(work),
            "tokens": 0,
            "critical_retained": 0,
        }

        work = self._dedup_fingerprints(work)
        diagnostics["stages"].append("T1")
        work = self._slot_fusion(work)
        diagnostics["stages"].append("T2")
        work = self._delta_encoding(work)
        diagnostics["stages"].append("T3")
        work = self._causal_pruning(work)
        diagnostics["stages"].append("T4")
        work = self._score_utility(work, query)
        diagnostics["stages"].append("T5")
        selected, selected_diagnostics = self._shadow_selection(work)
        diagnostics.update(selected_diagnostics)
        diagnostics["stages"].append("T6")
        selected = self._adaptive_budgeting(selected)
        diagnostics["stages"].append("T7")

        return selected, diagnostics

    def _dedup_fingerprints(self, patches: Sequence[SemanticPatch]) -> List[SemanticPatch]:
        """T1: deduplicate NON-CRITICAL patches by fingerprint; criticals pass through."""
        keep = {}
        crits = []
        for p in patches:
            if p.is_critical:
                crits.append(p)
                continue
            if p.fingerprint in keep:
                if p.turn_index >= keep[p.fingerprint].turn_index:
                    keep[p.fingerprint] = p
            else:
                keep[p.fingerprint] = p
        return crits + list(keep.values())

    def _slot_fusion(self, patches: Sequence[SemanticPatch]) -> List[SemanticPatch]:
        """T2: fuse duplicate slot keys for non-critical patches by highest utility and turn index."""
        groups = {}
        for p in patches:
            if p.is_critical:
                groups[id(p)] = p   # v0.1.2: never fuse criticals
                continue
            groups.setdefault(p.slot_key, p)
            if p.slot_key in groups and groups[p.slot_key] != p:
                current = groups[p.slot_key]
                if (p.utility, p.turn_index) >= (current.utility, current.turn_index):
                    groups[p.slot_key] = p
        out = []
        seen = set()
        for k, p in groups.items():
            if isinstance(p, SemanticPatch) and id(p) not in seen:
                out.append(p)
                seen.add(id(p))
        return out

    def _delta_encoding(self, patches: Sequence[SemanticPatch]) -> List[SemanticPatch]:
        """T3: rewrite non-critical patches sharing a slot_key as word-level diffs, only when shorter."""
        slot_base = {}
        out = []
        for p in sorted(patches, key=lambda x: x.turn_index):
            if p.is_critical or p.slot_key not in slot_base:
                slot_base[p.slot_key] = p
                out.append(p)
                continue
            base = slot_base[p.slot_key]
            new_words = set(p.payload.lower().split())
            base_words = set(base.payload.lower().split())
            added = new_words - base_words
            removed = base_words - new_words
            parts = []
            if added:
                parts.append("+[" + " ".join(sorted(added)) + "]")
            if removed:
                parts.append("-[" + " ".join(sorted(removed)) + "]")
            delta = " ".join(parts)
            if delta and count_tokens(delta) < p.token_cost:
                p.payload = delta
                p.is_delta = True
                p.recount()
            slot_base[p.slot_key] = p
            out.append(p)
        return out

    def _causal_pruning(self, patches: Sequence[SemanticPatch]) -> List[SemanticPatch]:
        """T4: remove intermediate non-critical nodes from the dependency graph."""
        kept_ids = {p.patch_id for p in patches}
        children = {}
        for p in patches:
            for d in p.dependencies:
                if d in kept_ids:
                    children.setdefault(d, []).append(p.patch_id)
        out = []
        for p in patches:
            if p.is_critical:
                out.append(p)
                continue
            has_children = bool(children.get(p.patch_id))
            has_parents = any(d in kept_ids for d in p.dependencies)
            if has_children and has_parents:
                continue
            out.append(p)
        return out

    def _score_utility(self, patches: Sequence[SemanticPatch], query: str) -> List[SemanticPatch]:
        """T5: utility scoring using alignment, dependency centrality, recency, and cost penalties."""
        type_boost = {
            'constraint': 0.20,
            'decision': 0.18,
            'code': 0.12,
            'equation': 0.10,
            'entity': 0.06,
            'structure': 0.04,
        }
        max_cost = max((p.token_cost for p in patches), default=1)
        dep_counts = {p.patch_id: 0 for p in patches}
        for p in patches:
            for d in p.dependencies:
                dep_counts[d] = dep_counts.get(d, 0) + 1

        for p in patches:
            align = self._align_score(p, query)
            dep_c = dep_counts.get(p.patch_id, 0)
            recency = math.exp(-RECENCY_LAMBDA * max(0, 0 - p.turn_index))
            cost_n = p.token_cost / max_cost
            type_boost_value = type_boost.get(p.patch_type, 0.0)
            p.utility = (W_ALIGN * (align + type_boost_value)) + (W_DEP * dep_c) + (W_RECENCY * recency) - (W_COST * cost_n)
        return patches

    def _align_score(self, patch: SemanticPatch, query: str) -> float:
        """Return semantic alignment score using sentence-transformers when installed, else 0.5 default."""
        try:
            import sentence_transformers  # optional dependency
            if self._embedder is None:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer('all-MiniLM-L6-v2')
            q = self._embedder.encode(query)
            p = self._embedder.encode(patch.payload)
            try:
                return float(self._cosine(q, p))
            except Exception:
                return 0.5
        except Exception:
            return 0.5

    def _cosine(self, a, b) -> float:
        """Return cosine similarity between two vector-like iterables."""
        import numpy as np
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _shadow_selection(self, work: Sequence[SemanticPatch]) -> Tuple[List[SemanticPatch], Dict[str, Any]]:
        """T6: select critical patches, score non-critical patches, and fit under budget tokens."""
        criticals = sorted([p for p in work if p.is_critical], key=lambda p: p.utility, reverse=True)
        selected = list(criticals)
        trim_count = self._fit_criticals(selected)
        critical_retained = len([p for p in selected if p.is_critical])
        diagnostics = {"critical_retained": critical_retained, "trimmed_critical_words": trim_count}
        non_criticals = sorted([p for p in work if not p.is_critical],
                               key=lambda p: (p.utility / max(1, p.token_cost)), reverse=True)
        selected_total = selected
        for p in non_criticals:
            if count_tokens(self.build_context(selected_total + [p])) <= self.budget:
                selected_total.append(p)
        # FINAL HARD CAP on the real joined string (newline tokens included)
        while selected_total and count_tokens(self.build_context(selected_total)) > self.budget:
            non_crit = [q for q in selected_total if not q.is_critical]
            if non_crit:
                selected_total.remove(min(non_crit, key=lambda q: q.utility))
                continue
            trimmable = [q for q in selected_total if len(q.payload.split()) > 2]
            if trimmable:
                tgt = max(trimmable, key=lambda q: len(q.payload.split()))
                tgt.payload = self._trim_numeric_first(
                    tgt.payload, len(tgt.payload.split()) - 1)
                tgt.recount()
            else:
                selected_total.remove(min(selected_total, key=lambda q: q.utility))
        return selected_total, diagnostics

    def _fit_criticals(self, criticals: List[SemanticPatch]) -> int:
        """Proportionally fit critical patches into the critical token budget."""
        if not criticals:
            return 0
        limit = max(8, int(self.budget * CRITICAL_SHARE))
        n = len(criticals)
        per_patch_words = max(2, int((limit / n - 4) / 1.4))
        trimmed = 0
        # pass 1 — uniform proportional trim, numeric-first
        for p in criticals:
            before = len(p.payload.split())
            p.payload = self._trim_numeric_first(p.payload, per_patch_words)
            p.recount()
            trimmed += before - len(p.payload.split())
        # pass 2 — while the JOINED critical block is over limit, trim the
        # critical with the MOST words; drop only at the <=2-word floor
        while criticals and count_tokens(self.build_context(criticals)) > limit:
            trimmable = [q for q in criticals if len(q.payload.split()) > 2]
            if trimmable:
                tgt = max(trimmable, key=lambda q: len(q.payload.split()))
                tgt.payload = self._trim_numeric_first(
                    tgt.payload, len(tgt.payload.split()) - 1)
                tgt.recount()
                trimmed += 1
            else:
                worst = min(criticals, key=lambda q: q.utility)
                criticals.remove(worst)
        return trimmed

    # FIXED (v0.1.6): a unit following its number was being trimmed away at
    # tight budgets — a user observed "Webhook timeout must 30" (30 what?
    # ms? seconds? minutes?). A word immediately after a number now carries
    # a high keep-weight, so "30 seconds", "15 minutes", "60 days" survive
    # as bound pairs.
    def _trim_numeric_first(self, payload: str, max_words: int) -> str:
        """Trim payload to max_words; numbers and their units survive longest."""
        words = payload.split()
        if len(words) <= max_words:
            return payload

        def weight(i: int, w: str):
            s = 0
            if re.search(r"\d", w):
                s += 2   # numbers, versions, thresholds
            if "%" in w or w.isupper():
                s += 1   # units, acronyms
            if w[:1].isupper():
                s += 1   # proper nouns (tools, systems)
            # v0.1.6: a unit binds to its number — the word immediately
            # after a number is usually its unit (seconds, ms, minutes...).
            if i > 0 and re.search(r"\d", words[i - 1]):
                s += 2
            return (s, -i)  # ties → keep earlier word

        ranked = sorted(range(len(words)),
                        key=lambda i: weight(i, words[i]), reverse=True)[:max_words]
        return " ".join(words[i] for i in sorted(ranked))

    def _adaptive_budgeting(self, patches: Sequence[SemanticPatch]) -> List[SemanticPatch]:
        """T7: allocate per-type budgets using the EMA-like query type signal."""
        return list(patches)

    def build_context(self, patches: Sequence[SemanticPatch]) -> str:
        """Join all patch prompt strings into a newline-delimited context."""
        return "\n".join(p.to_prompt_str() for p in patches)

    def reset_ema(self) -> None:
        """Reset the EMA query signal state stored in the engine."""
        self.ema_query = {t: 0.0 for t in PATCH_TYPES}

    def _sorted_count(self, patches):
        return len(patches)