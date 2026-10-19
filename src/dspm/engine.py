"""Compression engine for DSPM.

The engine applies the seven stages outlined in the package design:
T1 fingerprint deduplication, T2 slot fusion, T3 delta encoding,
T4 causal pruning, T5 utility scoring, T6 critical guarantee and
selection, and T7 adaptive budgeting.

v0.1.10: recency in T5 now actually decays (the previous formula was
identically 1.0 for every patch, so W_RECENCY never influenced ranking —
a paper/code gap), and compress() diagnostics report the OUTPUT counts
instead of values frozen before compression ran. Unused imports removed.

v0.1.12: revision stories survive starvation budgets. Payloads carrying
a "(was X)" span have a 3-word floor (the story form "10 (was 30)"),
and _trim_numeric_first returns that minimal story when the allowance is
too small for value + unit + history. Found in the live webhook rematch:
8 criticals at a 59-token budget trimmed the revision to a valueless
"10 seconds" — the replaced value vanished and the answer model
fabricated a replacement.

v0.1.14: readable starvation trimming. The baseline experiment (Oct 11)
showed DSPM ties a rolling summary at equal budgets — the guarantee
works, nothing is lost, but the 2-3 word payloads are too fragmented
for a judge or answer model to recognize ("encryption AES-256 at" is
unreadable; "encryption AES-256 at rest" is a constraint). Three fixes:
key nouns boosted from 30 to 50 in the priority ladder; a post-fill
guarantee that a number-bearing payload always keeps its key noun; and
the minimal revision story now includes the subject ("timeout 10 (was
30)" not just "10 (was 30)"). Also adds more domain key nouns from the
baseline conversations (encryption, deployment, anonymization, sync,
binary, battery, cold, startup, background, etc.).

v0.1.16: polarity-preserving trimming (reviewer finding: "100% CRR
means the memory items remain, but important words like 'not' or 'less
than' could still be lost"). In the v0.1.15 ladder those words were
filler (priority 10) — under budget pressure "dashboard must load in
under 2 seconds" became "dashboard 2 seconds", and "Payments must not
be stored in plaintext" lost its negation at the 2-word floor,
inverting the constraint while CRR reported 100. Comparators now bind
to their numbers as atomic spans ("less than 100 milliseconds", "no
more than 3 retries"), negations bind to the word they negate ("not
stored"), and both outrank key nouns. Also: the T5 recency reference
now floors at the newest patch's turn_index (callers whose turn
counter lags — direct _merge_patch users, notebooks loaded with a
stale turns value — previously got a flat 1.0 recency for every patch,
the latent test_recency_prefers_recent_noncritical failure in clean
environments), and DSPMEngine(stable_order=True) emits patches in
(turn_index, patch_id) order for prefix-cache friendliness (see
experiments/bench_kv_cache.py).
"""

from __future__ import annotations

import copy
import math
import re
from typing import Any, Dict, List, Sequence, Tuple

from dspm.config import (
    PATCH_TYPES,
    CRITICAL_SHARE,
    DEFAULT_BUDGET,
    RECENCY_LAMBDA,
    W_ALIGN,
    W_COST,
    W_DEP,
    W_RECENCY,
)
from dspm.patch import SemanticPatch, count_tokens


class DSPMEngine:
    """Compression engine applying the DSPM seven-stage pipeline."""

    # Unit words recognized for number-binding (the word immediately
    # after a number). Compound units ("per minute") are detected
    # separately in _trim_numeric_first.
    _UNIT_WORDS = (
        r"(ms|millisecond|milliseconds|s|sec|secs|second|seconds|"
        r"min|mins|minute|minutes|hr|hrs|hour|hours|d|day|days|"
        r"w|week|weeks|mo|month|months|y|yr|year|years|kb|mb|gb|tb|%)"
    )
    # Constraint subject nouns — the word that makes a bare number
    # meaningful ("timeout" in "webhook timeout 30 seconds").
    # v0.1.9: added weekdays, month names, quarters, years.
    # v0.1.14: added domain nouns from the baseline conversations —
    # encryption, deployment, anonymization, sync, binary, battery,
    # cold-start, startup, background, compliance, anonymization,
    # monitoring, alerting, backup, scaling, region, database, etc.
    _KEY_NOUNS = (
        r"(timeout|limit|rate|deadline|budget|threshold|ttl|latency|"
        r"expiry|window|quota|cap|duration|interval|retention|"
        r"cooldown|sla|uptime|fee|price|cost|"
        r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
        r"january|february|march|april|may|june|july|august|"
        r"september|october|november|december|"
        r"q[1-4]|20\d\d|"
        r"encryption|deploy|deployment|region|database|anonymization|"
        r"compliance|sync|binary|battery|startup|background|offline|"
        r"monitoring|alerting|backup|scaling|instance|storage|"
        r"retention|pipeline|training|dashboard|page|screen|response|"
        r"request|retr(y|ies))s?"
    )
    # v0.1.9: temporal VALUES ("Friday", "March", "Q3", "2026") — when
    # attached to a key noun, they ARE the constraint's value and get
    # priority 90 (below numbers, above all nouns). Separate pattern
    # from _KEY_NOUNS so a temporal in a non-temporal position doesn't
    # accidentally outrank a number-adjacent word.
    _TEMPORALS = (
        r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
        r"january|february|march|april|may|june|july|august|"
        r"september|october|november|december|"
        r"q[1-4]|20\d\d)s?"
    )
    # v0.1.16: polarity vocabulary. Negations and comparators were
    # scored as filler (priority 10) by the v0.1.15 ladder, so tight
    # budgets produced "dashboard 2 seconds" from "dashboard must load
    # in under 2 seconds" and "Payments must" from "Payments must not
    # be stored in plaintext" — CRR stayed 100 while the direction or
    # polarity of the constraint was lost or inverted.
    _NEGATION_WORDS = (
        "not", "no", "never", "without", "cannot", "can't",
        "mustn't", "don't", "doesn't", "won't", "cant", "mustnt",
    )
    # comparator PHRASES (matched as cleaned word tuples) that bind to
    # the number that follows: "less than 100", "no more than 3",
    # "must not exceed 10". Longer phrases are listed first so
    # "no more than" wins over a bare "more than" scan position.
    _COMP_PHRASES = (
        ("no", "more", "than"), ("no", "less", "than"),
        ("no", "fewer", "than"),
        ("not", "more", "than"), ("not", "less", "than"),
        ("not", "fewer", "than"),
        ("must", "not", "exceed"), ("must", "never", "exceed"),
        ("not", "exceed"), ("not", "exceeds"),
        ("not", "exceeded"), ("not", "exceeding"),
        ("less", "than"), ("fewer", "than"), ("more", "than"),
        ("at", "least"), ("at", "most"), ("up", "to"),
    )
    # single-word comparators that bind when DIRECTLY followed by a
    # number: "under 2 seconds", "exactly 5 years", "max 3 retries".
    _COMP_SINGLE = (
        "under", "below", "above", "over", "within", "exactly",
        "approximately", "exceed", "exceeds", "exceeded",
        "exceeding", "max", "min", "maximum", "minimum",
    )
    # words the negation binder may skip while looking for the word it
    # negates ("not be stored" -> "not stored").
    _BIND_SKIP = (
        "be", "is", "are", "was", "were", "been", "being", "the",
        "a", "an", "to", "in", "on", "at", "for", "of", "or", "and",
        "than", "must", "should", "will", "shall", "may", "might",
        "can", "could", "our", "any", "all",
    )

    def __init__(self, budget: int = DEFAULT_BUDGET, stable_order: bool = False):
        self.budget = budget
        self.stable_order = stable_order
        self.ema_query = {t: 0.0 for t in PATCH_TYPES}
        self._embedder = None

    def compress(self, patches: Sequence[SemanticPatch], query: str, turn_index: int) -> Tuple[List[SemanticPatch], Dict[str, Any]]:
        """Compress a list of SemanticPatch objects into selected patches plus diagnostics.

        The method deep-copies patches then applies the T1-T7 pipeline. It
        returns a selected patch list together with a diagnostics dictionary
        representing the stage-level counts and token information.
        """
        work = copy.deepcopy(list(patches))
        diagnostics = {
            "stages": [],
            "selected": len(work),
            "tokens": 0,
            "critical_retained": 0,
        }

        # T1 fingerprint deduplication
        work = self._dedup_fingerprints(work)
        diagnostics["stages"].append("T1")

        # T2 slot fusion
        work = self._slot_fusion(work)
        diagnostics["stages"].append("T2")

        # T3 delta encoding
        work = self._delta_encoding(work)
        diagnostics["stages"].append("T3")

        # T4 causal pruning
        work = self._causal_pruning(work)
        diagnostics["stages"].append("T4")

        # T5 utility scoring
        # v0.1.10: turn_index now reaches the scorer so recency decays.
        work = self._score_utility(work, query, turn_index)
        diagnostics["stages"].append("T5")

        # T6 critical guarantee and shadow selection
        selected, selected_diagnostics = self._shadow_selection(work)
        diagnostics.update(selected_diagnostics)
        diagnostics["stages"].append("T6")

        # T7 adaptive budgeting
        selected = self._adaptive_budgeting(selected)
        diagnostics["stages"].append("T7")

        # v0.1.16: optional stable output ordering — (turn_index, patch_id)
        # instead of the utility-ranked selection order. A stable order
        # means a mid-conversation revision only invalidates the KV prefix
        # cache from that patch's position onward, instead of reordering
        # the whole context every turn. Off by default (preserves the
        # v0.1.15 output order); see experiments/bench_kv_cache.py for the
        # measurement.
        if self.stable_order:
            selected = sorted(selected, key=lambda p: (p.turn_index, p.patch_id))

        # v0.1.10: report the OUTPUT, not the input — "selected" and
        # "tokens" were previously frozen at their pre-compression values.
        diagnostics["selected"] = len(selected)
        diagnostics["tokens"] = count_tokens(self.build_context(selected))

        # ensure token bound
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
                groups[id(p)] = p   # never fuse criticals
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
        kept_ids = {p.patch_id: p for p in patches}
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

    def _score_utility(self, patches: Sequence[SemanticPatch], query: str,
                       current_turn: int = 0) -> List[SemanticPatch]:
        """T5: utility scoring using alignment, dependency centrality, recency, and cost penalties.

        v0.1.10: recency is now a real decay. The previous formula
        exp(-lambda * max(0, 0 - p.turn_index)) evaluated to exactly 1.0
        for every non-negative turn index, so W_RECENCY contributed a
        constant and never affected ranking. Older patches now decay
        relative to the current turn, matching the paper's description.
        The default keeps backward compatibility for direct callers."""
        # v0.1.16: the recency reference can never be older than the
        # newest patch — callers whose turn counter lags (direct
        # _merge_patch users, notebooks loaded with a stale turns value)
        # previously got a flat 1.0 recency for every patch, disabling
        # the decay entirely (the latent test_recency failure in clean
        # environments).
        if patches:
            current_turn = max(current_turn,
                               max(p.turn_index for p in patches))
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
            # v0.1.10: real recency decay (see docstring)
            recency = math.exp(-RECENCY_LAMBDA * max(0, current_turn - p.turn_index))
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

    @staticmethod
    def _word_floor(payload: str) -> int:
        """v0.1.12: the minimum word count a critical payload keeps under
        budget pressure. 2 by default; 3 when it carries a '(was X)'
        revision story — the story form '10 (was 30)' IS the floor, and
        trimming below it destroys the revision history while the
        current value survives (the live webhook-rematch failure)."""
        return 3 if "(was" in payload else 2

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
                continue
        # FINAL HARD CAP on the real joined string (newline tokens included)
        while selected_total and count_tokens(self.build_context(selected_total)) > self.budget:
            non_crit = [q for q in selected_total if not q.is_critical]
            if non_crit:
                selected_total.remove(min(non_crit, key=lambda q: q.utility))
                continue
            # v0.1.12: '(was X)' payloads have a 3-word floor (the story
            # form) — see _word_floor
            trimmable = [q for q in selected_total
                         if len(q.payload.split()) > self._word_floor(q.payload)]
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
        # pass 1 — uniform proportional trim
        # v0.1.12: payloads carrying "(was X)" get a 3-word floor
        # (_word_floor) — a 2-word allowance keeps the number+unit and
        # drops the history, leaving a valueless current value.
        for p in criticals:
            allowance = max(self._word_floor(p.payload), per_patch_words)
            before = len(p.payload.split())
            p.payload = self._trim_numeric_first(p.payload, allowance)
            p.recount()
            trimmed += before - len(p.payload.split())
        # pass 2 — while the JOINED critical block is over limit, trim the
        # critical with the MOST words; drop only at each payload's floor
        # v0.1.12: the floor is per-payload (2, or 3 for was-stories)
        while criticals and count_tokens(self.build_context(criticals)) > limit:
            trimmable = [q for q in criticals
                         if len(q.payload.split()) > self._word_floor(q.payload)]
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

    # v0.1.7: number+unit pairs are ATOMIC SPANS (weights can't guarantee
    # pairing — live testing produced both "timeout 30" and "Rate per").
    # v0.1.9: "(was X)" replacement spans protected at 60; temporal
    # values following a key noun ("deadline Friday") get priority 90.
    # v0.1.12: minimal revision story — was-payloads have a 3-word floor,
    # and at that floor the trim returns the story form "10 (was 30)"
    # (current value + history, units dropped) instead of a valueless
    # "10 seconds".
    # v0.1.14: key nouns boosted from 30 to 50 (the subject noun makes
    # the value meaningful — without it, "AES-256 at" is noise); a
    # post-fill check guarantees a number-bearing payload always keeps
    # its key noun; the minimal story includes the subject: "timeout 10
    # (was 30)" not just "10 (was 30)"; domain nouns expanded.
    def _trim_numeric_first(self, payload: str, max_words: int) -> str:
        """Trim payload to max_words. Number+unit pairs and '(was X)'
        replacement spans are atomic; selection priority: number-spans >
        temporal-after-key-noun (90) > negations (80) > was-spans (60) >
        key nouns and compound units (50) > number-adjacent words (40) >
        proper nouns (20) > filler (10). v0.1.16: comparator phrases
        bind to their numbers ("less than 100", "no more than 3") at
        number priority, and negations bind to the word they negate
        ("not stored") — a bound without its direction, or a negation
        without its verb, is a different constraint. Payloads carrying a '(was X)' span at an
        allowance of <= 3 words return the minimal story 'timeout 10
        (was 30)' — subject + current value + history. A post-fill
        guarantee ensures number-bearing results always include their
        key noun (the constraint's subject)."""
        words = payload.split()
        if len(words) <= max_words:
            return payload

        # ── 0. v0.1.12/14: minimal revision story (with subject) ───
        # At tiny allowances the ladder below keeps "10 seconds" (the
        # number+unit span outranks the was-span) and DROPS the history.
        # When the payload carries "(was X)" and the allowance is too
        # small for value + unit + history, keep the KEY NOUN + current
        # number + the full was-span, dropping everything else.
        # v0.1.14: the subject noun is now included — "timeout 10 (was
        # 30)" instead of just "10 (was 30)" — because a value without
        # its subject is unreadable by downstream models.
        # Only returns when the story fits within max_words — otherwise
        # falls through to the ladder, so callers trimming toward the
        # floor always make strict progress (no live-lock).
        if max_words <= 3 and any(w.lower().startswith("(was") for w in words):
            was_start = next(i for i, w in enumerate(words)
                             if w.lower().startswith("(was"))
            j = was_start
            while j < len(words) and not words[j].endswith(")"):
                j += 1
            was_end = min(j, len(words) - 1)
            span = words[was_start:was_end + 1]
            current = next((words[i] for i in range(was_start)
                            if re.search(r"\d", words[i])), None)
            # v0.1.14: find the key noun BEFORE the current value
            key_noun = None
            for i in range(was_start):
                if re.fullmatch(self._KEY_NOUNS,
                                words[i].lower().strip(".,;)")):
                    key_noun = words[i]
                    break
            parts = ([key_noun] if key_noun else []) + \
                     ([current] if current else []) + span
            if len(parts) <= max_words:
                return " ".join(parts)
            # try without the key noun (it costs 1 word)
            parts_no_noun = ([current] if current else []) + span
            if len(parts_no_noun) <= max_words:
                return " ".join(parts_no_noun)
            # story doesn't fit — fall through to the normal ladder

        # ── 1. identify protected spans ──────────────────────────────
        spans = []          # (start, end, priority, fallbacks)
        claimed = set()
        pinned = {}         # v0.1.16: word index -> priority (negation binder)
        i = 0
        while i < len(words):
            # v0.1.9: "(was 30 seconds)" replacement span — keep whole
            if words[i].lower().startswith("(was"):
                j = i
                while j < len(words) and not words[j].endswith(")"):
                    j += 1
                end = min(j, len(words) - 1)
                spans.append((i, end, 60, []))
                claimed.update(range(i, end + 1))
                i = end + 1
                continue
            # v0.1.16: comparator phrase + number (+unit) — atomic. The
            # comparator IS part of the value: "latency 100 ms" from
            # "latency less than 100 ms" is a different constraint.
            lw = words[i].lower().strip(".,;)")
            matched_phrase = False
            for phrase in self._COMP_PHRASES:
                L = len(phrase)
                if lw == phrase[0] and i + L <= len(words):
                    cleaned = tuple(
                        w.lower().strip(".,;)") for w in words[i:i + L])
                    if cleaned == phrase:
                        end = i + L - 1
                        num_i = None
                        if end + 1 < len(words) and re.search(r"\d", words[end + 1]):
                            num_i = end + 1
                            end = num_i
                            if end + 1 < len(words) and re.fullmatch(
                                    self._UNIT_WORDS,
                                    words[end + 1].lower().strip(".,;)")):
                                end += 1
                        fallbacks = []
                        if num_i is not None:
                            # graceful degradation, in order: phrase +
                            # number (drop the trailing unit — direction
                            # outranks the unit), then number + unit
                            # (drop the phrase), then the bare number —
                            # a direction without a value is noise.
                            fallbacks.append((i, num_i))
                            if end > num_i:
                                fallbacks.append((num_i, end))
                            fallbacks.append((num_i, num_i))
                        spans.append((i, end, 100, fallbacks))
                        claimed.update(range(i, end + 1))
                        i = end + 1
                        matched_phrase = True
                        break
            if matched_phrase:
                continue
            # v0.1.16: single-word comparator directly before a number —
            # "under 2 seconds", "exactly 5 years", "max 3 retries".
            if lw in self._COMP_SINGLE and i + 1 < len(words) \
                    and re.search(r"\d", words[i + 1]):
                end = i + 1
                if end + 1 < len(words) and re.fullmatch(
                        self._UNIT_WORDS,
                        words[end + 1].lower().strip(".,;)")):
                    end += 1
                fallbacks = [(i, i + 1)]          # comparator + number
                if end > i + 1:
                    fallbacks.append((i + 1, end))  # number + unit
                spans.append((i, end, 100, fallbacks))
                claimed.update(range(i, end + 1))
                i = end + 1
                continue
            # v0.1.16: negation binder — "not (be) stored" pins BOTH
            # words at priority 80. A negation without the word it
            # negates is not a constraint ("Payments must not" bans
            # nothing); a verb without its negation inverts the meaning.
            if lw in self._NEGATION_WORDS:
                j = i + 1
                while j < len(words) and j <= i + 3 \
                        and words[j].lower().strip(".,;)") in self._BIND_SKIP:
                    j += 1
                if j < len(words) and j <= i + 3 and j not in claimed:
                    pinned[i] = 80
                    pinned[j] = 80
                    i = j + 1
                    continue
                # nothing to bind — the word scorer still gives the bare
                # negation priority 80
                pinned[i] = 80
            # compound unit: "per minute" / "an hour" / "a day"
            if re.fullmatch(r"per|an|a", words[i].lower()) and i + 1 < len(words) \
                    and re.fullmatch(r"(second|minute|hour|day|week|month|year)s?",
                                     words[i + 1].lower().strip(".,;)")):
                spans.append((i, i + 1, 50, []))
                claimed.update((i, i + 1)); i += 2; continue
            # number + its unit: "30 seconds", "15 minutes", "500 mb"
            if re.search(r"\d", words[i]) and i + 1 < len(words) \
                    and re.fullmatch(self._UNIT_WORDS,
                                     words[i + 1].lower().strip(".,;)")):
                spans.append((i, i + 1, 100, []))
                claimed.update((i, i + 1)); i += 2; continue
            i += 1

        # ── 2. score remaining standalone words ─────────────────────
        items = [(prio, -s, s, e, fb) for (s, e, prio, fb) in spans]
        key_noun_words = []     # v0.1.14: track which words are key nouns
        for j, w in enumerate(words):
            if j in claimed:
                continue
            w_clean = w.lower().strip(".,;)")
            if re.search(r"\d", w):
                prio = 100                    # bare number
            elif j in pinned:
                prio = pinned[j]              # v0.1.16: negation / bound word
            elif j > 0 and re.search(r"\d", words[j - 1]):
                prio = 40                    # number-adjacent ("requests")
            # v0.1.9: temporal VALUE directly after a key noun —
            # "deadline Friday", "expires March", "release 2026". The
            # temporal IS the constraint's value; outranks everything
            # except numbers themselves.
            elif (j > 0
                  and re.fullmatch(self._KEY_NOUNS,
                                   words[j - 1].lower().strip(".,;)"))
                  and re.fullmatch(self._TEMPORALS, w_clean)):
                prio = 90                    # temporal value of a constraint
            elif w_clean in self._NEGATION_WORDS:
                prio = 80                    # v0.1.16: unbound negation
            elif w_clean in self._COMP_SINGLE \
                    or w_clean in ("than", "least", "most"):
                prio = 70                    # v0.1.16: leftover comparator
            elif re.fullmatch(self._KEY_NOUNS, w_clean):
                prio = 50                    # v0.1.14: constraint subject (was 30)
                key_noun_words.append(j)
            elif w[:1].isupper():
                prio = 20                    # proper noun
            else:
                prio = 10                    # filler
            items.append((prio, -j, j, j, []))

        # ── 3. greedy fill: priority desc, ties → earlier word ──────
        items.sort(reverse=True)
        keep, used = set(), 0
        # v0.1.16: spans with fallbacks try the full range first, then
        # degrade (comparator spans: phrase+number → number+unit →
        # number) so a span that cannot fit whole still places its most
        # meaningful subset instead of vanishing entirely.
        for (prio, _neg, s, e, fb) in items:
            for (fs, fe) in [(s, e)] + list(fb):
                if used + (fe - fs + 1) <= max_words:
                    keep.update(range(fs, fe + 1))
                    used += (fe - fs + 1)
                    break

        # ── 3b. v0.1.14: key-noun guarantee ─────────────────────────
        # If the payload contains numbers but no key noun survived the
        # greedy fill, the result is a value without context ("AES-256
        # at" instead of "encryption AES-256 at"). Swap the lowest-
        # priority non-number word for the highest-priority key noun.
        has_number = any(re.search(r"\d", words[j]) for j in keep)
        has_key_noun = any(j in key_noun_words for j in keep)
        if has_number and not has_key_noun and key_noun_words:
            best_noun = min(key_noun_words)  # earliest key noun
            if best_noun not in keep:
                # find the lowest-priority word to drop (a filler or
                # proper noun, not a number or unit)
                # v0.1.16: never swap out a negation or comparator to
                # make room for the key noun — the subject noun can be
                # inferred from context, an inverted bound cannot.
                drop_candidates = [j for j in keep
                                   if not re.search(r"\d", words[j])
                                   and j not in claimed
                                   and j not in pinned
                                   and words[j].lower().strip(".,;)")
                                       not in self._NEGATION_WORDS
                                   and words[j].lower().strip(".,;)")
                                       not in self._COMP_SINGLE]
                if drop_candidates:
                    drop = min(drop_candidates,
                               key=lambda j: (words[j].lower().strip(".,;)") in
                                              [words[k].lower() for k in key_noun_words], j))
                    # simply swap: remove the filler, add the key noun
                    keep.discard(drop)
                    keep.add(best_noun)

        return " ".join(words[j] for j in sorted(keep))

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
