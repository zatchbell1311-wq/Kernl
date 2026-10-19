"""User-facing DSPM memory API.

v0.1.10: auto-dense extraction — add_turn() now checks
numeric_density(text) >= DENSE_AUTO_THRESHOLD and routes fact-heavy turns
through the dense extraction funnel even when dense=False. (v0.1.9 defined
numeric_density but never called it, so document tests at 70%/85% removal
silently ran the 5-patch funnel and lost most numeric facts at extraction.)
Also adds add_document(): sentence-aware chunking for document ingestion.

v0.1.12: (was X) baking is history-aware. A stale patch that is itself a
revision ("... 10 seconds (was 30)") previously gave up its CURRENT value
when superseded — the baking took the stale payload's first numeric word,
so a restated revision baked a wrong "(was 10)" instead of carrying
"(was 30)" forward. It now detects restatements of the same revision and
carries the history, and chains full revision lineages ("(was 300 then
60)"). Found in the live webhook rematch.

v0.1.14: baking chains are deduplicated — the baseline experiment (Oct 11)
showed "(was 20 then 20)" (the same value baked twice) in the surviving
payload. _replaced_value now deduplicates the chain before joining.

v0.1.15: restatement collapse. The baseline experiment showed duplicate
criticals eating the budget — "audit retention 5 years" AND "audit logs
stored for exactly 5 years" coexisting as two [CON] patches. A new rule
(b2) collapses same-type restatements when they share a slot head, ≥2
shared non-numeric content words, identical numbers, and identical
time/size units. Rule (b) gains the same unit check (fixing the latent
"rate 100 per minute" vs "rate 100 per hour" false-positive flagged in the
v0.1.13 audit).

v0.1.16: bound-aware restatement collapse (reviewer finding: "100% CRR
means the memory items remain, but important words like 'not' or 'less
than' could still be lost"). Same numbers + same units + same slot head
does NOT mean the same fact when the bound direction differs: "retention
must not exceed 5 years" (upper) and "retention at least 5 years" (lower)
share every token the v0.1.15 rules compared, so (b2) collapsed them into
ONE critical, silently deleting a bound. Rules (b)/(b2) now require
compatible bound/polarity classes (_bound_class): same-direction bounds
still collapse, plain≈exact stays compatible (v0.1.15 behavior), and a
bound never merges with a different or missing bound. Conservative by
design — an unsolved duplicate is a budget cost; a merged contradiction
is a wrong answer. Also passes stable_order through to the engine for
prefix-cache-friendly output (see scripts/bench_kv_cache.py).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from dspm.config import DEFAULT_BUDGET, DENSE_AUTO_THRESHOLD
from dspm.engine import DSPMEngine
from dspm.extractor import extract_turn, numeric_density
from dspm.patch import SemanticPatch, count_tokens
from dspm.persistence import save_memory, load_memory

# v0.1.3: common function words excluded from revision matching.
_STOPWORDS = {
    "the", "a", "an", "to", "from", "for", "of", "in", "on", "at", "by", "as",
    "and", "or", "with", "must", "be", "is", "are", "was", "were", "will",
    "shall", "should", "this", "that", "it", "its", "all", "no", "not",
    "when", "if", "then", "than", "so", "we", "you",
}

# Verbs that signal a payload REVISES an earlier value. Revisions in
# real conversations announce themselves; complementary facts do not.
# v0.1.7: added "now", "moved" — live testing showed extractors often
# strip "updated"/"superseded" down to "timeout now 10 seconds".
_REVISION_MARKERS = {
    "update", "updated", "change", "changed", "revise", "revised",
    "supersede", "superseded", "replace", "replaced", "instead",
    "raise", "raised", "lower", "lowered", "increase", "increased",
    "decrease", "decreased", "reduce", "reduced", "switch", "switched",
    "migrate", "migrated", "revert", "reverted",
    "now", "moved",
}

# v0.1.15: time/size unit words for the restatement collapse safety
# check — "5 years" and "5 years" are the same fact; "100 per minute"
# and "100 per hour" are not, even though the number matches.
_TIME_SIZE_UNITS = {
    "ms", "millisecond", "second", "sec", "minute", "min",
    "hour", "hr", "day", "week", "month", "year",
    "kb", "mb", "gb", "tb", "%",
}

# v0.1.16: bound/polarity vocabulary for restatement compatibility.
# Same numbers + same units + same slot head does NOT mean same fact
# when the bound direction differs: "must not exceed 5 years" (upper)
# and "at least 5 years" (lower) share every token the v0.1.15 rules
# compared, so (b2) collapsed them — silently deleting a bound.
_BOUND_UP = {
    "less", "fewer", "under", "below", "within", "most", "max",
    "maximum", "cap", "capped", "up",
}
_BOUND_DOWN = {
    "more", "over", "above", "least", "min", "minimum", "floor",
}
_BOUND_EXACT = {"exactly", "precisely", "equal"}
_POLARITY_NEG = {
    "not", "no", "never", "without", "cannot", "mustn't",
    "don't", "doesn't", "won't", "cant", "mustnt",
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


# v0.1.15: extract time/size units from ALL words in the payload.
# v1.0 version only checked words IMMEDIATELY after numbers — missed "100
# requests per minute" (the unit is 3 words away from the number), so
# "per minute" and "per hour" looked identical. Now checks every word.
def _units_in(text: str) -> frozenset:
    """Extract ALL time/size units in the payload. Safety check for
    restatement collapse — '5 years' and '5 years' are the same fact;
    '100 per minute' and '100 per hour' are not."""
    units = set()
    for w in text.lower().split():
        clean = w.strip(".,;)")
        if clean.rstrip("s") in _TIME_SIZE_UNITS:
            units.add(clean.rstrip("s"))
    return frozenset(units)


# v0.1.16: classify a payload's bound direction / polarity, so
# value-matching collapse rules (b)/(b2) never merge contradictory
# constraints. Index-based so "not exceed" is read as ONE signal
# (upper) rather than negation + a bare "exceed" (which alone is a
# LOWER bound: "must exceed 100" = strictly greater).
def _bound_class(text: str) -> str:
    """Return 'upper' (at most / not exceed), 'lower' (at least /
    exceed), 'exact', 'neg' (bare negation, no bound), 'plain', or
    'mixed' (contradictory markers)."""
    ws = [w.lower().strip(".,;()") for w in text.split()]
    up = down = exact = neg = False
    i = 0
    while i < len(ws):
        w = ws[i]
        if w in _POLARITY_NEG:
            neg = True
            nxt = ws[i + 1] if i + 1 < len(ws) else ""
            if nxt in ("more", "over", "above") or nxt.startswith("exceed"):
                up = True
                i += 2
                continue
            if nxt in ("less", "fewer", "under", "below"):
                down = True
                i += 2
                continue
            i += 1
            continue
        if w in _BOUND_UP:
            up = True
        elif w in _BOUND_DOWN:
            down = True
        elif w.startswith("exceed"):
            down = True        # bare "exceed X" = strictly greater
        elif w in _BOUND_EXACT:
            exact = True
        i += 1
    if up and down:
        return "mixed"
    if up:
        return "upper"
    if down:
        return "lower"
    if exact:
        return "exact"
    if neg:
        return "neg"
    return "plain"


_BOUND_COMPAT = {
    ("plain", "plain"), ("plain", "exact"), ("exact", "exact"),
    ("upper", "upper"), ("lower", "lower"), ("mixed", "mixed"),
    ("neg", "neg"),
}


def _polarity_compatible(a: str, b: str) -> bool:
    """Can two payloads with identical numbers and units be the same
    fact? Conservative: only identical classes (plus plain~exact, the
    v0.1.15 behavior) collapse. Upper vs lower never merge; a bounded
    statement never merges with an unbounded one."""
    return (a, b) in _BOUND_COMPAT or (b, a) in _BOUND_COMPAT


# v0.1.12: the numeric value inside a single word, for comparing values
# across payloads ("30)" -> "30", "0.5%," -> "0.5").
def _first_number(word: str) -> Optional[str]:
    """The numeric value inside a single word ("30)" -> "30", "0.5%," -> "0.5")."""
    m = re.search(r"\d+(?:\.\d+)?", word)
    return m.group() if m else None


# v0.1.14: deduplicate a chain of values ("20 then 20" -> "20",
# "300 then 60 then 60" -> "300 then 60"). Found in the baseline
# experiment: "Pool size updated to 50 connections (was 20 then 20)".
def _dedup_chain(values: List[str]) -> List[str]:
    """Remove consecutive duplicates from a value chain, preserving order."""
    out = []
    for v in values:
        if not out or v != out[-1]:
            out.append(v)
    return out


# v0.1.12: the value a superseded patch gives up, for (was X) baking.
# v0.1.14: chains are deduplicated before joining.
def _replaced_value(stale_payload: str, new_current: Optional[str]) -> Optional[str]:
    """The value a superseded patch gives up, for (was X) baking.

    - plain stale patch ("timeout 30 seconds") -> its current value, "30"
    - stale revision restated by the new patch (same current value) ->
      the stale's HISTORY: "... 10 (was 30)" gives "30"
    - stale revision genuinely superseded (different value) -> the full
      lineage: "... 300 (was 60)" gives "300 then 60"
    - v0.1.14: chains are deduplicated — "300 then 300 then 60" gives
      "300 then 60" (the same value isn't baked twice)
    Returns None when the stale patch carries no numbers at all."""
    m = re.search(r"\(was ([^)]+)\)", stale_payload)
    history = m.group(1).strip() if m else None
    base = re.sub(r"\(was [^)]+\)", "", stale_payload)
    s_nums = [w for w in base.split() if re.search(r"\d", w)]
    current = s_nums[0] if s_nums else None
    current_val = _first_number(current) if current else None
    if current_val is not None and new_current is not None \
            and current_val == new_current:
        return history
    if current and history:
        chain = _dedup_chain([current, history])
        return " then ".join(chain)
    return current or history


class DSPMMemory:
    """Main user-facing memory object for DSPM.

    Example:
        >>> memory = DSPMMemory(budget=250, llm_client=None, model="gpt-4o-mini")
        >>> memory.add_turn("user", "Return every API result as JSON and require auth.")
        >>> context = memory.get_context("What must the contract remember?")
        >>> print(context)
    """

    def __init__(self, budget: int = DEFAULT_BUDGET, llm_client: Optional[Any] = None,
                 model: str = "gpt-4o-mini", dense: bool = False,
                 stable_order: bool = False, **kwargs: Any):
        self.budget = budget
        self.llm_client = llm_client
        self.model = model
        self.dense = dense
        # v0.1.16: prefix-cache-friendly output ordering — see
        # scripts/bench_kv_cache.py for why it matters.
        self.engine = DSPMEngine(budget=budget, stable_order=stable_order)
        self.patches: List[SemanticPatch] = []
        self.turns = 0
        self.selected_patches: List[SemanticPatch] = []
        self._last_context = ""
        self._context_built = False

    def add_turn(self, role: str, text: str) -> List[SemanticPatch]:
        """Add a conversation turn and return the newly extracted patches."""
        if self.llm_client is None:
            raise ValueError("llm_client is required to call add_turn()")

        dense = self.dense or numeric_density(text) >= DENSE_AUTO_THRESHOLD
        patches = extract_turn(self.llm_client, self.model, text, self.turns,
                               recent_context=self._last_context,
                               dense=dense)
        for p in patches:
            self._merge_patch(p)
        self.turns += 1
        return patches

    def add_document(self, text: str, chunk_tokens: int = 400) -> int:
        """Ingest a document as sentence-aware chunks. Returns chunk count."""
        if not text or not text.strip():
            return 0
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        chunks: List[str] = []
        cur: List[str] = []
        cur_toks = 0
        for s in sentences:
            s_toks = count_tokens(s)
            if cur and cur_toks + s_toks > chunk_tokens:
                chunks.append(" ".join(cur))
                cur, cur_toks = [], 0
            cur.append(s)
            cur_toks += s_toks
        if cur:
            chunks.append(" ".join(cur))
        for ch in chunks:
            self.add_turn("user", ch)
        return len(chunks)

    # Supersession rules (accumulated v0.1.3 → v0.1.16):
    #   (a)  same type + >=3 shared words + revision verb
    #   (a2) same type + same topic anchor + revision verb + changed numbers
    #   (b)  same type + >=4 shared words + IDENTICAL numbers + IDENTICAL units
    #        + compatible bound classes (v0.1.16)
    #   (b2) same type + same slot head + >=2 shared NON-NUMERIC words
    #        + IDENTICAL numbers + IDENTICAL units + compatible bound
    #        classes (v0.1.16)
    #   (c)  cross-type + >=4 shared words, OR >=3 + revision verb
    #   (c2) cross-type + same anchor + revision verb + changed numbers
    #
    # v0.1.15: rule (b2) collapses same-type restatements (user states,
    # assistant confirms — "audit retention 5 years" vs "audit logs stored
    # exactly 5 years" previously coexisted as duplicate criticals eating
    # the budget). Rule (b) gains the same unit check (fixes the latent
    # "100 per minute" vs "100 per hour" false-positive).
    # v0.1.16: (b)/(b2) gain the bound/polarity compatibility check —
    # "retention must not exceed 5 years" (upper) vs "retention at least
    # 5 years" (lower) previously collapsed to one critical, silently
    # deleting a bound.
    def _merge_patch(self, patch: SemanticPatch) -> None:
        """Merge a patch into memory with duplicate suppression and critical
        revision supersession."""
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
        stale = []
        if patch.is_critical:
            new_words = _content_words(patch.payload)
            new_nums = _numbers(patch.payload)
            marker = _has_revision_marker(patch.payload)
            new_head = _slot_head(patch.payload)
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
                    # v0.1.15: (b) same-value restatement now requires
                    # identical units (fixes "rate 100 per minute" vs
                    # "rate 100 per hour" false-positive); v0.1.16 adds
                    # bound/polarity compatibility ("at most 5" is not
                    # "at least 5").
                    if len(shared) >= 4 and new_nums and new_nums == _numbers(existing.payload) \
                            and _units_in(patch.payload) == _units_in(existing.payload) \
                            and _polarity_compatible(_bound_class(patch.payload),
                                                     _bound_class(existing.payload)):
                        stale.append(existing)          # (b) same-value restatement
                        continue
                    # v0.1.15: (b2) restatement collapse — same topic
                    # anchor + >=2 shared NON-NUMERIC words + identical
                    # numbers + identical units; v0.1.16 adds bound
                    # compatibility (upper ≠ lower).
                    if (new_head and new_head == _slot_head(existing.payload)
                            and len({w for w in shared if not re.search(r"\d", w)}) >= 2
                            and new_nums and new_nums == _numbers(existing.payload)
                            and _units_in(patch.payload) == _units_in(existing.payload)
                            and _polarity_compatible(_bound_class(patch.payload),
                                                     _bound_class(existing.payload))):
                        stale.append(existing)          # (b2) restatement collapse
                        continue
                else:
                    # (c) cross-type revision — >=4 words, or >=3 with marker
                    if len(shared) >= 4 or (marker and len(shared) >= 3):
                        stale.append(existing)          # (c) cross-type revision
                        continue
                    # (c2) terse cross-type revision (mirrors a2)
                    if marker and new_head and new_head == _slot_head(existing.payload) \
                            and new_nums != _numbers(existing.payload):
                        stale.append(existing)          # (c2) terse cross-type
                        continue

        # bake replaced values into the surviving payload
        if stale and patch.is_critical:
            patch_base = re.sub(r"\(was [^)]+\)", "", patch.payload)
            patch_nums = [w for w in patch_base.split() if re.search(r"\d", w)]
            new_current = _first_number(patch_nums[0]) if patch_nums else None
            replaced_vals = []
            for s in stale:
                val = _replaced_value(s.payload, new_current)
                if val:
                    replaced_vals.append(val)
            if replaced_vals and "(was" not in patch.payload:
                chain = _dedup_chain(replaced_vals[:2])
                patch.payload = f"{patch.payload} (was {' then '.join(chain)})"

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

    def get_context(self, query: str = "") -> str:
        """Return the compressed context string produced by the DSPM engine."""
        self.engine.budget = self.budget
        selected, diagnostics = self.engine.compress(self.patches, query, self.turns)
        self.selected_patches = selected
        self._context_built = True
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
        if self._context_built:
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
            "dense": self.dense,
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
        self._context_built = False
        self.engine.reset_ema()

    def set_budget(self, budget: int) -> None:
        """Update the token budget. Takes effect on the next get_context() call."""
        self.budget = int(budget)
        self.engine.budget = self.budget

    def save(self, path: str) -> int:
        """Persist the memory notebook to a JSON file (atomic write)."""
        return save_memory(self, path)

    def load(self, path: str) -> int:
        """Load a notebook from a JSON file into this memory (merge semantics)."""
        return load_memory(self, path)
