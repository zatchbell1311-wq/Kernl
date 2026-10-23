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
prefix-cache-friendly output (see experiments/bench_kv_cache.py).

v0.1.16 (review fold): the four reviewer findings are part of this
unpublished release, not a separate 0.1.17: (1) stats.context_built —
CRR is only measured after the first get_context(); before that,
critical_selected falls back to critical_total; (2) add_turn threads
the speaker role into the extraction prompt (it was accepted and then
dropped); (3) the (was X) chain keeps the three most recently replaced
values (the old [:2] cap silently deleted the third-oldest value when
one revision superseded three stale criticals); (4) stats.selected
reports delivered patches post-build, store size pre-build.
v0.1.17 (unpublished, F5 batch): rule (b4) bridges thin cross-type
extraction echoes -- a bound-word slot head that agrees with its own
bound direction, the other side's slot head being the shared subject
noun, within two turns. The frozen fee pair held the nominal store
at 6 against the smoke's 7-critical delivery cap, so 6/20 live runs
tipped to 8 and CRR fell to 87%. A canary test freezes the remaining
word-overlap limitation (same-value bounds sharing only a scope noun
still merge; structured slot supersession is the planned next step,
reviewer point 3). Same batch: patches carry source_role through the
store and the notebook (reviewer point 2), stats splits conversation
/ patch / context tokens (reviewer point 4), and the (was X) span
stays the machine-readable revision story the starvation trim
protects -- prose is droppable, the span is not.

20-run live gate (F6): the loop caught two live-extraction
shapes the offline suites could not. Mode 1 -- the same
dashboard latency fact arrived as a word-order reversal
('Dashboard sub-2-second' [CON] vs 'sub-2-second dashboard'
[DEC]); its distinctive shared token carries a digit, so the
(b3) non-numeric word count never reached two and the store
bloated to 8 against the 7-critical delivery cap (CRR 87% on
3/20 runs). Rule (b5) bridges order-insensitive echoes under
every (b3)/(b4) safety. Mode 2 -- comma-form revision history
lost the replaced value at the 2-word floor; the engine-side
F6 half fixes the floor and the minimal story.

20-run live gate, round 2 (F7): F6 was green in isolation
(102 tests, harness 5/5) but the live loop still failed 6/20
runs, all CRR under-reporting (87/77/87/87/70/87). Both
signatures are echo criticals whose live word order breaks
the slot-head anchor every earlier bridge keys on. (i) The
audit retention fact arrived twice -- noun first ('Audit
retention set to 5 years', head 'audit') and verb first
('Retain audit logs for exactly 5 years', head 'retain') --
and survived as TWO decisions in 6/6 failing runs ('[DEC] 5
years' + '[DEC] exactly 5'). (ii) Dashboard echoes led
with confirmation verbs ('Confirmed sub-2-second dashboard'
-- head 'confirmed', absent from the other payload), so
(b5)'s bidirectional head containment never fired. Rule
(b6) bridges same- and cross-type echoes by what the pair
SHARES instead of where each payload starts: a shared
subject noun, identical numbers/units, compatible bounds,
never plain/plain, within two turns -- plus a residue guard
(every unshared content word must be a confirmation verb, a
slot-family twin like retain~retention, or property-positioned
behind the subject with only non-prepositional filler
between). Scoped pairs -- platinum vs silver fee, dashboard
vs homepage, cloud vs training budgets, audit vs payment
logs -- stay separate.
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
_UNIT_ALIASES = {
    "ms": "ms", "millisecond": "ms", "milliseconds": "ms",
    "s": "second", "sec": "second", "secs": "second",
    "second": "second", "seconds": "second",
    "min": "minute", "mins": "minute", "minute": "minute", "minutes": "minute",
    "hr": "hour", "hrs": "hour", "hour": "hour", "hours": "hour",
    "d": "day", "day": "day", "days": "day",
    "w": "week", "week": "week", "weeks": "week",
    "mo": "month", "month": "month", "months": "month",
    "y": "year", "yr": "year", "year": "year", "years": "year",
    "kb": "kb", "mb": "mb", "gb": "gb", "tb": "tb", "%": "%",
}

# v0.1.16: bound/polarity vocabulary for restatement compatibility.
# Same numbers + same units + same slot head does NOT mean same fact
# when the bound direction differs: "must not exceed 5 years" (upper)
# and "at least 5 years" (lower) share every token the v0.1.15 rules
# compared, so (b2) collapsed them — silently deleting a bound.
# v0.1.17 (finding S1): "sub" added -- hyphenated compounds like
# "sub-2-second" are upper bounds now that _bound_class splits
# hyphens like spaces (live smoke evidence).
_BOUND_UP = {
    "less", "fewer", "under", "below", "within", "most", "max",
    "maximum", "cap", "capped", "up", "sub",
}
_BOUND_DOWN = {
    "more", "over", "above", "least", "min", "minimum", "floor",
}
_BOUND_EXACT = {"exactly", "precisely", "equal"}
_POLARITY_NEG = {
    "not", "no", "never", "without", "cannot", "can't", "mustn't",
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
    # v0.1.17 (finding S1): hyphens split like spaces --
    # "sub-2-second" carries the unit "second"; without this,
    # hyphenated values looked unitless and restatement collapse
    # compared them against nothing.
    for w in text.lower().replace("-", " ").split():
        canon = _UNIT_ALIASES.get(w.strip(".,;)"))
        if canon:
            units.add(canon)
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
    # v0.1.17 (finding S1): hyphens split like spaces so
    # "sub-2-second" reads as bound word + value + unit.
    ws = [w.lower().strip(".,;()") for w in text.replace("-", " ").split()]
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


# v0.1.17 (finding S2): bound words that can serve as a slot head and
# bridge a thin cross-type echo (rule (b4)).
_BOUND_HEAD_WORDS = frozenset(_BOUND_UP | _BOUND_DOWN | _BOUND_EXACT)


def _head_direction(head: str) -> str:
    """Direction class implied by a bound-word slot head. The head must
    agree with its OWN payload's bound class to bridge: 'No more than 5'
    is an upper bound whose head 'more' is a lower-bound word, so it
    never bridges (guarded by test_no_more_than_head_direction_mismatch_stays)."""
    if head in _BOUND_UP:
        return "upper"
    if head in _BOUND_DOWN:
        return "lower"
    if head in _BOUND_EXACT:
        return "exact"
    return "plain"


# v0.1.17 F7 (20-run live gate, round 2): confirmation/stasis verbs a
# live extractor leads echo payloads with. Both round-2 failure
# signatures are echoes whose FIRST significant token is the verb, not
# the subject noun -- 'Retain audit logs for exactly 5 years' (head
# 'retain' vs the original's 'audit') and 'Confirmed sub-2-second
# dashboard' (head 'confirmed', absent from the other payload) --
# which is exactly what broke the slot-head anchors of (b2)/(b3)/(b5).
# In the (b6) residue test these verbs restate, they never scope.
_ECHO_VERBS = frozenset({
    "confirmed", "confirm", "confirms", "keep", "keeps", "kept",
    "lock", "locks", "locked", "target", "targets", "ensure",
    "ensures", "agree", "agreed", "remain", "remains", "stay",
    "stays", "hold", "holds", "track", "tracked", "noted",
    "accepted", "approve", "approved", "enforce", "enforced",
    "apply", "applies", "applied", "retain", "retains", "retained",
})

# v0.1.17 F7: morphological slot twins -- 'retention' and 'retain' are
# the same slot to any reader, never to a string set. Deliberately a
# closed family list: a wrong merge costs more than a missed merge.
_SLOT_FAMILIES = (
    frozenset({"retain", "retention", "retained", "retaining"}),
)

# v0.1.17 F7: small measurement/filing nouns that ride along with a
# subject ('dashboard load time', 'audit logs', 'dashboard latency').
# They never anchor the bridge as the shared subject, but they may
# sit between the subject and a later residue word -- and 'records'
# BEFORE the subject ('No more than 500 records per page') still
# reads as a scope word, because position carries the meaning there.
_PROPERTY_NOUNS = frozenset({
    "time", "load", "loads", "latency", "response", "speed",
    "duration", "period", "logs", "log", "data", "records", "record",
})

# v0.1.17 F7: a preposition between the shared subject and an unshared
# noun marks a SCOPE, not a property -- 'budget for cloud' vs
# 'dashboard latency'. The (b6) property-position tolerance never
# crosses one of these.
_BRIDGE_BLOCKERS = frozenset({
    "for", "to", "of", "in", "on", "at", "by", "with", "from", "per",
})


def _slot_family_eq(a: str, b: str) -> bool:
    """v0.1.17 F7: True when two words are morphological twins of one
    slot ('retain' ~ 'retention')."""
    for fam in _SLOT_FAMILIES:
        if a in fam and b in fam:
            return True
    return False


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
                 stable_order: bool = False, render_source_roles: bool = False,
                 **kwargs: Any):
        self.budget = budget
        self.llm_client = llm_client
        self.model = model
        self.dense = dense
        # v0.1.16: prefix-cache-friendly output ordering — see
        # experiments/bench_kv_cache.py for why it matters.
        self.engine = DSPMEngine(budget=budget, stable_order=stable_order)
        self.patches: List[SemanticPatch] = []
        self.turns = 0
        self.selected_patches: List[SemanticPatch] = []
        self._last_context = ""
        self._context_built = False
        # v0.1.17 (review points 2+4): source attribution + input-token
        # accounting. _conversation_tokens counts the RAW turn text fed
        # into add_turn (the extraction INPUT), separate from
        # patch_tokens (extraction OUTPUT) and context_tokens (the
        # delivered result) -- the reviewer's point: raw_tokens measured
        # extracted patches and was easy to misread as conversation size.
        self._conversation_tokens = 0
        # v0.1.17 (review point 2): optional '(role)' suffix on delivered
        # lines. Off by default -- it costs ~2 tokens per delivered line
        # and budgets were sized without it.
        self.render_source_roles = bool(render_source_roles)

    def add_turn(self, role: str, text: str) -> List[SemanticPatch]:
        """Add a conversation turn and return the newly extracted patches."""
        if self.llm_client is None:
            raise ValueError("llm_client is required to call add_turn()")

        dense = self.dense or numeric_density(text) >= DENSE_AUTO_THRESHOLD
        # v0.1.16 (review point 2): role threaded through -- it was accepted
        # here and then dropped on the floor.
        patches = extract_turn(self.llm_client, self.model, text, self.turns,
                               role=role,
                               recent_context=self._last_context,
                               dense=dense)
        # v0.1.17 (review point 4): the raw turn text is the extraction
        # INPUT -- counted whether or not extraction succeeds.
        self._conversation_tokens += count_tokens(text)
        for p in patches:
            # v0.1.17 (review point 2, data path): attribution rides on
            # the patch. The extractor PROMPT has carried the role since
            # the F2 batch; the store could not tell a user requirement
            # from an assistant echo.
            p.source_role = role
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
    #   (b3) cross-type CON<->DEC same-value restatement: same anchor,
    #        >=2 shared non-numeric words, identical numbers + units,
    #        compatible bounds, at least one directional/exact bound
    #        (v0.1.17 finding S1 -- live smoke showed user [CON] +
    #        assistant [DEC] echoes coexisting as duplicate criticals)
    #   (b4) cross-type thin echo: bound-word slot head on one side,
    #        shared subject noun as the other slot head, <=2 turns
    #        apart (v0.1.17 finding S2 -- the frozen fee pair held
    #        the store at 6 against the 7-critical delivery cap;
    #        6/20 live runs tipped to 8 and CRR fell to 87%)
    #   (b5) cross-type word-order echo: BOTH slot heads present
    #        in the other payload's word set + a shared
    #        digit-bearing token as the value anchor;
    #        numbers/units/bounds/proximity safety as (b3)/(b4)
    #        (v0.1.17 20-run gate mode 1 -- 'Dashboard
    #        sub-2-second' [CON] + 'sub-2-second dashboard' [DEC]
    #        bloated the store to 8; CRR 87% on 3/20 runs)
    #   (b6) shared-subject same-value echo (same- OR cross-type,
    #        v0.1.17 F7): value safety as (b3)/(b5) + a shared
    #        subject noun >=4 chars, never plain/plain, <=2 turns,
    #        and NO conflicting scope noun in the residue --
    #        unshared content words must be echo verbs, slot-family
    #        twins (retain~retention), or property-positioned
    #        behind the subject (round 2: 6/20 runs failed CRR
    #        87-70% on 'Audit retention set to 5 years' + 'Retain
    #        audit logs for exactly 5 years' and on
    #        confirmation-verb dashboard echoes)
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
                    # v0.1.17 F7 (20-run live gate, round 2): (b6)
                    # shared-subject echo bridge, same-type site.
                    # Round 2 failed 6/20 runs on echo pairs whose
                    # live word order breaks every head anchor:
                    # 'Audit retention set to 5 years' + 'Retain
                    # audit logs for exactly 5 years' (heads 'audit'
                    # vs 'retain') survived as TWO decisions in 6/6
                    # failing runs. (b6) anchors on the shared
                    # subject noun + value safety + residue guard
                    # instead -- see _echo_bridge.
                    if self._echo_bridge(patch, existing):
                        stale.append(existing)          # (b6) shared-subject echo bridge
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
                    # v0.1.17 (finding S1): (b3) cross-type same-value
                    # restatement. Live smoke evidence: the user states a
                    # requirement that extracts as [CON] ('dashboard must
                    # load in under 2 seconds') and the assistant echo
                    # extracts as [DEC] ('Dashboard latency must be
                    # sub-2-second') -- no rule compared across types
                    # without a revision verb, so both coexisted and the
                    # store bloated to 8-10 criticals (CRR 70-87% at the
                    # 59-token smoke budget). Stricter than same-type (b2):
                    # additionally requires at least one directional or
                    # exact bound, because cross-type merges are riskier
                    # ('Budget 100 for cloud' vs 'Budget 100 for training'
                    # is plain/plain and must stay apart). The thin-restatement
                    # freeze on this rule was lifted in finding S2 -- see (b4)
                    # below for the guarded bridge that replaced it.
                    bc_new = _bound_class(patch.payload)
                    bc_old = _bound_class(existing.payload)
                    if (new_head and new_head == _slot_head(existing.payload)
                            and len({w for w in shared if not re.search(r"\d", w)}) >= 2
                            and new_nums and new_nums == _numbers(existing.payload)
                            and _units_in(patch.payload) == _units_in(existing.payload)
                            and _polarity_compatible(bc_new, bc_old)
                            and not (bc_new == "plain" and bc_old == "plain")):
                        stale.append(existing)          # (b3) cross-type restatement
                        continue
                    # v0.1.17 (finding S2): (b4) bound-head bridge for thin
                    # cross-type echoes. The 20-run live loop still failed
                    # CRR 87% (7/8 delivered) on 30% of runs: the frozen
                    # fee pair held the nominal store at 6 against a
                    # 7-critical delivery cap, so two extraction variants
                    # tipped a run to 8. The fee pair's shape: one side
                    # leads with a BOUND WORD ('Maximum fee ... 0.5%') and
                    # the other leads with the shared subject noun
                    # ('fee cap set to 0.5%'). Bridge conditions, all
                    # required: the bound-word head agrees with its OWN
                    # payload's bound direction ('No more than ...' does
                    # NOT bridge -- its head 'more' is a lower-bound word
                    # inside an upper-bound sentence), the other side's
                    # slot head IS a shared content word, and the pair
                    # sits within two turns (extraction echoes are
                    # adjacent; tier-scoped facts stated turns apart are
                    # not). Canary-frozen limitation: same-value bounds
                    # sharing only a scope noun still merge ('Max users
                    # 500 per org' + 'org admins cap 500') -- structured
                    # slot supersession is the planned next fix
                    # (reviewer point 3).
                    old_head = _slot_head(existing.payload)
                    shared_nonnum = {w for w in shared if not re.search(r"\d", w)}
                    if (new_nums and new_nums == _numbers(existing.payload)
                            and _units_in(patch.payload) == _units_in(existing.payload)
                            and _polarity_compatible(bc_new, bc_old)
                            and not (bc_new == "plain" and bc_old == "plain")
                            and abs(patch.turn_index - existing.turn_index) <= 2
                            and ((new_head in _BOUND_HEAD_WORDS
                                  and _head_direction(new_head) == bc_new
                                  and old_head in shared_nonnum)
                                 or (old_head in _BOUND_HEAD_WORDS
                                     and _head_direction(old_head) == bc_old
                                     and new_head in shared_nonnum))):
                        stale.append(existing)          # (b4) thin echo bridge
                        continue
                    # v0.1.17 20-run live gate (mode 1 -- runs
                    # 9/11/20 at CRR 87%): (b5) compound-token
                    # bridge. The same dashboard latency fact
                    # arrived as 'Dashboard sub-2-second' [CON]
                    # and 'sub-2-second dashboard' [DEC] -- a
                    # word-order reversal. Its distinctive shared
                    # token ('sub-2-second') carries a digit, so
                    # the >=2 shared NON-numeric guard in (b3)
                    # counted only 'dashboard' and the pair never
                    # collapsed; the store bloated to 8 against
                    # the 7-critical delivery cap. (b5) bridges
                    # order-insensitive echoes: BOTH slot heads
                    # present in the other payload's word set (a
                    # scope noun on one side only, as in the tier
                    # guards, never bridges), a shared
                    # digit-bearing token as the value anchor,
                    # plus every (b3)/(b4) safety: identical
                    # numbers, identical units, compatible bounds,
                    # never plain/plain, <=2 turns.
                    old_words = set(re.findall(r"\w+", existing.payload.lower()))
                    new_words = set(re.findall(r"\w+", patch.payload.lower()))
                    if (len(old_words & new_words) >= 2
                            and any(re.search(r"\d", w) for w in old_words & new_words)
                            and new_nums == _numbers(existing.payload)
                            and old_head and old_head in new_words
                            and new_head and new_head in old_words
                            and _units_in(patch.payload) == _units_in(existing.payload)
                            and _polarity_compatible(bc_new, bc_old)
                            and not (bc_new == "plain" and bc_old == "plain")
                            and abs(patch.turn_index - existing.turn_index) <= 2):
                        stale.append(existing)          # (b5) compound-token bridge
                        continue
                    # v0.1.17 F7: (b6) shared-subject echo bridge,
                    # cross-type site -- the confirmation-verb echo
                    # shapes that slip (b5)'s bidirectional head
                    # containment ('Keep Dashboard latency
                    # sub-2-second' + 'Confirmed sub-2-second
                    # dashboard': neither 'keep' nor 'confirmed'
                    # appears in the other payload, while every
                    # other (b5) condition passes).
                    if self._echo_bridge(patch, existing):
                        stale.append(existing)          # (b6) shared-subject echo bridge
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
                # v0.1.16 (review point 3): the [:2] cap silently deleted the
                # third-oldest value when ONE revision superseded three or
                # more stale criticals (repro: webhook-timeout entries at
                # 30/20/15 seconds, then "webhook timeout updated to 10
                # seconds" -- the 15 vanished without a trace). Keep the
                # three most recently added replaced values; deeper history
                # is dropped, and the payload cap plus the was-span floors
                # in the engine bound the result.
                chain = _dedup_chain(replaced_vals)[-3:]
                patch.payload = f"{patch.payload} (was {' then '.join(chain)})"
                patch.recount()   # token_cost must reflect the baked payload

        for s in stale:
            self.patches.remove(s)
        self.patches.append(patch)

    def _echo_bridge(self, patch: SemanticPatch,
                     existing: SemanticPatch) -> bool:
        """v0.1.17 F7 (20-run live gate, round 2): (b6) shared-subject
        same-value echo bridge -- same- OR cross-type.

        F6 was green in isolation (102 tests, harness 5/5) yet the
        live loop still failed 6/20 runs, all CRR under-reporting,
        with two signatures -- echo criticals whose live word order
        breaks the slot-head anchor every earlier bridge keys on:

          'Audit retention set to 5 years' + 'Retain audit logs for
          exactly 5 years'  (heads 'audit' vs 'retain': (b2) requires
          head equality, (b) needs >=4 shared words -- the pair
          survived as TWO decisions in 6/6 failing runs, delivered as
          '[DEC] 5 years' + '[DEC] exactly 5')

          'Dashboard latency must be sub-2-second' + 'Confirmed
          sub-2-second dashboard'  (the echo leads with the
          confirmation verb, so (b5)'s bidirectional head containment
          fails; with a verb-first constraint BOTH directions fail)

        (b6) anchors on what the pair SHARES instead of where each
        payload starts. All required: identical non-empty numbers,
        identical units, compatible bounds, never plain/plain, <=2
        turns, and a shared SUBJECT NOUN (>=4 chars, purely alpha,
        not stopword/revision marker/bound word/unit/echo verb/
        property noun). The residue guard then keeps scoped facts
        apart: every unshared content word must be a confirmation
        verb (_ECHO_VERBS), a morphological twin of the other side
        ('retention' ~ 'retain'), or sit in property position after
        the shared subject with only non-blocker stopwords, echo
        verbs, property nouns, digits or <=3-char non-bound tokens
        between. 'platinum/silver fee', 'dashboard vs homepage',
        'cloud vs training budgets', 'audit vs payment logs' and the
        no-more-than/page-size pair all stay separate.
        """
        if abs(patch.turn_index - existing.turn_index) > 2:
            return False
        new_nums = _numbers(patch.payload)
        if not new_nums or new_nums != _numbers(existing.payload):
            return False
        if _units_in(patch.payload) != _units_in(existing.payload):
            return False
        bc_new = _bound_class(patch.payload)
        bc_old = _bound_class(existing.payload)
        if not _polarity_compatible(bc_new, bc_old):
            return False
        if bc_new == "plain" and bc_old == "plain":
            return False
        a_words = re.findall(r"\w+", patch.payload.lower())
        b_words = re.findall(r"\w+", existing.payload.lower())
        a_set, b_set = set(a_words), set(b_words)
        shared = a_set & b_set
        subjects = {w for w in shared
                    if len(w) >= 4 and w.isalpha()
                    and w not in _STOPWORDS and w not in _REVISION_MARKERS
                    and w not in _BOUND_UP and w not in _BOUND_DOWN
                    and w not in _BOUND_EXACT and w not in _POLARITY_NEG
                    and w not in _UNIT_ALIASES and w not in _ECHO_VERBS
                    and w not in _PROPERTY_NOUNS}
        if not subjects:
            return False
        for words, other in ((a_words, b_set), (b_words, a_set)):
            subj_pos = [i for i, w in enumerate(words) if w in subjects]
            for i, w in enumerate(words):
                if (len(w) < 4 or not w.isalpha() or w in shared
                        or w in _STOPWORDS or w in _REVISION_MARKERS
                        or w in _ECHO_VERBS or w in _BOUND_UP
                        or w in _BOUND_DOWN or w in _BOUND_EXACT
                        or w in _POLARITY_NEG or w in _UNIT_ALIASES):
                    continue
                if any(_slot_family_eq(w, o) for o in other):
                    continue
                ok = False
                for sp in subj_pos:
                    if i > sp and all(
                            (t in _STOPWORDS and t not in _BRIDGE_BLOCKERS)
                            or t in _ECHO_VERBS or t in _PROPERTY_NOUNS
                            or any(c.isdigit() for c in t)
                            or (len(t) <= 3 and t not in _BOUND_UP
                                and t not in _BOUND_DOWN
                                and t not in _BOUND_EXACT)
                            for t in words[sp + 1:i]):
                        ok = True
                        break
                if not ok:
                    return False
        return True

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
        # v0.1.17 (review point 2): optional role attribution on delivered
        # lines, when render_source_roles=True and the patch carries a
        # role. Rendered AFTER selection, so it never changes what the
        # engine picks -- only what the reader sees about who said it.
        if self.render_source_roles:
            context = "\n".join(
                f"{p.to_prompt_str()} ({p.source_role})"
                if p.source_role else p.to_prompt_str()
                for p in selected) or context
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
        """Return a summary of memory health, selected-critical retention, and token reduction rate.

        v0.1.16 (review points 1+4): `context_built` is the delivery flag
        -- CRR is only MEASURED after the first get_context(); before that
        it is assumed (critical_selected falls back to critical_total).
        `selected` is the delivered patch count (store size pre-build), so
        an empty selection is visible next to trr instead of hiding
        behind a perfect reduction number. TRR keeps its raw formula:
        clamping a negative reduction would be metric gaming, and
        raw_tokens / context_tokens stay exposed for audit.
        v0.1.17 (review point 4): conversation_tokens (raw turn
        text fed to add_turn) and patch_tokens (alias of
        raw_tokens -- the EXTRACTED patches) make the three-way
        split explicit, so nobody mistakes patch tokens for
        conversation size again.
        """
        critical_total = len(self.critical_patches)
        if self._context_built:
            critical_selected = len([p for p in self.selected_patches if p.is_critical])
        else:
            critical_selected = critical_total
        raw_tokens = sum(count_tokens(p.to_prompt_str()) for p in self.patches)
        context_tokens = count_tokens(self._last_context) if self._last_context else 0
        crr = 100 if critical_total == 0 else int((critical_selected / critical_total) * 100)
        trr = 100 - int((context_tokens / max(1, raw_tokens)) * 100) if raw_tokens else 0
        # v0.1.16 (review points 1+4): selected mirrors the
        # critical_selected fallback -- pre-build it reports the store
        # size, post-build the delivered size. context_built lets anyone
        # quoting "100% CRR" show the number was measured on a delivered
        # context, not assumed.
        selected_count = len(self.selected_patches) if self._context_built else len(self.patches)
        roles: Dict[str, int] = {}
        for p in self.critical_patches:
            key = p.source_role or "unattributed"
            roles[key] = roles.get(key, 0) + 1
        return {
            "turns": self.turns,
            "total_patches": len(self.patches),
            "critical_total": critical_total,
            "critical_selected": critical_selected,
            "crr": crr,
            "context_built": self._context_built,
            "selected": selected_count,
            "budget": self.budget,
            "dense": self.dense,
            "conversation_tokens": self._conversation_tokens,
            "raw_tokens": raw_tokens,
            "patch_tokens": raw_tokens,
            "context_tokens": context_tokens,
            "trr": trr,
            # v0.1.17 (review point 2): attribution census for criticals
            # -- 'unattributed' covers pre-0.1.17 notebooks.
            "critical_by_role": dict(sorted(roles.items())),
        }

    def reset(self) -> None:
        """Clear all stored memory and reset the context state."""
        self.patches.clear()
        self.selected_patches.clear()
        self.turns = 0
        self._last_context = ""
        self._context_built = False
        self._conversation_tokens = 0
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
