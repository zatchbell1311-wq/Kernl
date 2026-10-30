# dspm/engine.py  --  UNIVERSAL v4.5
#
# Goal: CRR = 100% at ANY budget the user sets.
#
# Compression ladder (escalates only as far as needed):
#   Stage 1  proportional numeric-first trim         (target word count from budget/n)
#   Stage 2  iteratively trim longest down to 1 word
#   Stage 3  ultra-compact per patch                 (keyword + numbers + units, ~3 tokens)
#   Stage 4  dense-merge                             (one line per type, semicolon-separated)
#   Stage 5  ultra-dense                             (single token per critical, ONE line)
#
# The absolute physical floor for N criticals is N+4 tokens:
#   ~1 token per critical payload + 2 tag tokens per type ([CON] + [DEC]).
# Below that floor it is mathematically impossible to represent N distinct items,
# so the engine raises a clear BudgetTooSmallError instead of silently dropping CRR.
#
# Companion test: tests/test_universal_budgets.py sweeps budgets 10..2000 in steps.
import re
import math
from collections import defaultdict, Counter
from copy import deepcopy
# sentence_transformers is imported lazily in _get_embedder() to keep
# import-time lightweight and allow tests to inject a stub embedder.
from sklearn.metrics.pairwise import cosine_similarity

from .patch import SemanticPatch, count_tokens, _tag

EMBED_MODEL = "all-MiniLM-L6-v2"

ALPHA_EMA        = 0.5
SHADOW_THRESHOLD = 0.05
RECENCY_LAMBDA   = 0.15
DELTA_MIN_SAVING = 1
W_ALIGN, W_DEP, W_RECENCY, W_COST = 0.45, 0.20, 0.15, 0.20
REVISION_OVERLAP = 0.40

BASE_BUDGET_FRACTIONS = {
    "constraint": 0.30,
    "decision":   0.25,
    "code":       0.20,
    "equation":   0.10,
    "entity":     0.10,
    "structure":  0.05,
}
PATCH_TYPES    = list(BASE_BUDGET_FRACTIONS.keys())
CRITICAL_TYPES = {"constraint", "decision"}
CRITICAL_SHARE = 1.0

# Tag overhead in tokens: [CON] + [DEC] = ~4 tokens
TAG_OVERHEAD_TOKENS = 4

_embedder = None  # lazy-loaded on first use


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


class BudgetTooSmallError(RuntimeError):
    """Raised when the budget is physically too small to fit all criticals.

    The absolute floor is N_criticals + TAG_OVERHEAD_TOKENS tokens.
    Below that, N distinct items cannot be represented in N-1 tokens.
    """
    pass


class DSPMEngine:
    def __init__(self, budget: int = 400):
        if budget < 1:
            raise ValueError(f"budget must be >= 1, got {budget}")
        self.budget    = budget
        self.ema_query = {t: 0.0 for t in PATCH_TYPES}
        # Per-compress state:
        self._dense_mode = "off"   # off | dense | ultra
        self._last_diag  = {}

    # ── public API ────────────────────────────────────────────────
    def compress(self, patches, query: str, turn_index: int):
        if not patches:
            return [], {}
        working = deepcopy(patches)          # never mutate caller's patches
        diag    = {}
        self._dense_mode = "off"
        self._update_ema(query)
        working, diag["t1_removed"] = self._fingerprint_dedup(working)
        working, diag["t2_removed"] = self._slot_fusion(working)
        working, diag["t3_deltas"]  = self._delta_encode(working)
        working, diag["t4_removed"] = self._causal_prune(working)
        working                      = self._score_utility(working, query, turn_index)
        type_budgets                 = self._adaptive_budget()
        selected, diag["t6_removed"] = self._shadow_select(working, type_budgets)
        diag["selected_count"] = len(selected)
        diag["total_tokens"]   = sum(p.token_cost for p in selected)
        diag["type_budgets"]   = type_budgets
        diag["dense_mode"]     = self._dense_mode
        diag["floor_tokens"]   = self._floor_for(len([p for p in patches
                                                      if p.patch_type in CRITICAL_TYPES]))
        self._last_diag = diag
        return selected, diag

    def build_context(self, patches) -> str:
        """Build the prompt-context string. Honours dense/ultra-dense modes."""
        if self._dense_mode == "ultra":
            # ONE line for ALL criticals: "[C] word1 word2 ... wordN"
            # plus non-criticals on subsequent lines (if budget allows).
            crits  = [p for p in patches if p.patch_type in CRITICAL_TYPES]
            others = [p for p in patches if p.patch_type not in CRITICAL_TYPES]
            words  = []
            for ptype in ("constraint", "decision"):
                grp = [p for p in crits if p.patch_type == ptype]
                if grp:
                    tag = "C" if ptype == "constraint" else "D"
                    words.append(f"[{tag}]")
                    words.extend(p.payload for p in grp)
            lines = [" ".join(words)]
            for p in others:
                lines.append(p.to_prompt_str())
            return "\n".join(lines)

        if self._dense_mode == "dense":
            # One line per critical type, semicolon-separated payloads.
            crits  = [p for p in patches if p.patch_type in CRITICAL_TYPES]
            others = [p for p in patches if p.patch_type not in CRITICAL_TYPES]
            lines = []
            for ptype in ("constraint", "decision"):
                grp = [p for p in crits if p.patch_type == ptype]
                if grp:
                    merged = "; ".join(p.payload for p in grp)
                    lines.append(f"[{_tag(ptype)}] {merged}")
            for p in others:
                lines.append(p.to_prompt_str())
            return "\n".join(lines)

        # Normal mode: one line per patch.
        return "\n".join(p.to_prompt_str() for p in patches)

    def reset_ema(self):
        self.ema_query = {t: 0.0 for t in PATCH_TYPES}

    @staticmethod
    def min_budget_for_crr(n_criticals: int) -> int:
        """Conservative lower bound on the budget (in tokens) at which
        CRR = 100% is achievable AND the context fits within budget.

        Uses 2.5 tokens per critical (empirical safety margin: payloads like
        'Fargate', '1000', 'Access' are 2 tokens in cl100k_base; some are 3)
        + 4 tag overhead.

        Below this floor the engine raises BudgetTooSmallError instead of
        silently dropping criticals or overflowing the budget.
        """
        return max(1, int(n_criticals * 2.5) + TAG_OVERHEAD_TOKENS)

    def _floor_for(self, n_criticals: int) -> int:
        return self.min_budget_for_crr(n_criticals)

    # ── T1 fingerprint dedup ──────────────────────────────────────
    def _fingerprint_dedup(self, patches):
        kept, pos_of, removed = [], {}, 0
        for p in patches:
            if p.patch_type in CRITICAL_TYPES:
                kept.append(p)
                continue
            idx = pos_of.get(p.fingerprint)
            if idx is None:
                pos_of[p.fingerprint] = len(kept)
                kept.append(p)
            else:
                if p.turn_index > kept[idx].turn_index:
                    kept[idx] = p
                removed += 1
        return kept, removed

    # ── T2 slot fusion ────────────────────────────────────────────
    def _slot_fusion(self, patches):
        slots = defaultdict(list)
        for p in patches:
            if p.patch_type not in CRITICAL_TYPES:
                slots[p.slot_key].append(p)
        winners = {id(max(sl, key=lambda q: (q.utility, q.turn_index))) for sl in slots.values()}
        fused   = [p for p in patches if p.patch_type in CRITICAL_TYPES or id(p) in winners]
        return fused, len(patches) - len(fused)

    # ── T3 delta encoding ─────────────────────────────────────────
    def _delta_encode(self, patches):
        slot_base, delta_count = {}, 0
        for p in sorted(patches, key=lambda x: x.turn_index):
            sk = p.slot_key
            if p.patch_type in CRITICAL_TYPES or sk not in slot_base:
                slot_base[sk] = p
                continue
            base          = slot_base[sk]
            delta_payload = self._compute_delta(p.payload, base.payload)
            if count_tokens(delta_payload) <= p.token_cost - DELTA_MIN_SAVING:
                p.payload    = delta_payload
                p.is_delta   = True
                p.delta_base = base.patch_id
                p.recount()
                delta_count += 1
            slot_base[sk] = p
        return patches, delta_count

    @staticmethod
    def _compute_delta(new_text, base_text):
        nw, bw = set(new_text.lower().split()), set(base_text.lower().split())
        parts  = []
        if nw - bw: parts.append("+[" + " ".join(sorted(nw - bw)) + "]")
        if bw - nw: parts.append("-[" + " ".join(sorted(bw - nw)) + "]")
        return " ".join(parts) if parts else new_text

    # ── T4 causal prune ───────────────────────────────────────────
    def _causal_prune(self, patches):
        id_map   = {p.patch_id: p for p in patches}
        children = defaultdict(list)
        for p in patches:
            for dep in p.dependencies:
                if dep in id_map:
                    children[dep].append(p.patch_id)
        depth, visited = {}, set()
        queue = [p.patch_id for p in patches if not p.dependencies]
        for pid in queue:
            depth[pid] = 0
        visited.update(queue)
        while queue:
            nxt = []
            for pid in queue:
                for child in children.get(pid, []):
                    if child not in visited:
                        depth[child] = depth[pid] + 1
                        visited.add(child)
                        nxt.append(child)
            queue = nxt
        pruned, removed = [], 0
        for p in patches:
            d = depth.get(p.patch_id, 0)
            if d > 0 and children.get(p.patch_id) and p.patch_type not in CRITICAL_TYPES:
                removed += 1
            else:
                p.causal_depth = d
                pruned.append(p)
        return pruned, removed

    # ── utility scoring ───────────────────────────────────────────
    def _score_utility(self, patches, query, turn_index):
        if not patches:
            return patches
        texts       = [query] + [p.payload for p in patches]
        embs        = _get_embedder().encode(texts, show_progress_bar=False)
        q_emb, p_embs = embs[0], embs[1:]
        dep_counts  = Counter(dep for p in patches for dep in p.dependencies)
        max_dep     = max(dep_counts.values(), default=1)
        max_cost    = max(p.token_cost for p in patches) or 1
        type_boost  = {
            "constraint": 0.20, "decision": 0.18, "code": 0.12,
            "equation":   0.10, "entity":   0.06, "structure": 0.04,
        }
        for i, p in enumerate(patches):
            cos_sim   = float(cosine_similarity([q_emb], [p_embs[i]])[0][0])
            align     = cos_sim + type_boost.get(p.patch_type, 0.0)
            dep_c     = dep_counts.get(p.patch_id, 0) / max_dep
            recency   = math.exp(-RECENCY_LAMBDA * max(turn_index - p.turn_index, 0))
            cost_n    = p.token_cost / max_cost
            p.utility = W_ALIGN * align + W_DEP * dep_c + W_RECENCY * recency - W_COST * cost_n
        return patches

    def _update_ema(self, query):
        q = query.lower()
        for t in PATCH_TYPES:
            self.ema_query[t] = (
                ALPHA_EMA * (1.0 if t in q else 0.0)
                + (1 - ALPHA_EMA) * self.ema_query[t]
            )

    def _adaptive_budget(self):
        raw = {
            t: BASE_BUDGET_FRACTIONS[t] * (1 + ALPHA_EMA * self.ema_query[t])
            for t in PATCH_TYPES
        }
        tw = sum(raw.values())
        return {t: max(5, int(self.budget * raw[t] / tw)) for t in PATCH_TYPES}

    # ── T6 shadow select ──────────────────────────────────────────
    def _shadow_select(self, patches, type_budgets):
        criticals    = sorted(
            [p for p in patches if p.patch_type in CRITICAL_TYPES],
            key=lambda p: p.utility, reverse=True,
        )
        others       = [p for p in patches if p.patch_type not in CRITICAL_TYPES]
        pos_sum      = sum(max(p.utility, 0.0) for p in others) or 1e-9
        kept_others  = [
            p for p in others
            if p.utility <= 0.0 or (p.utility / pos_sum) >= SHADOW_THRESHOLD
        ]
        selected     = list(criticals)
        self._fit_criticals(selected)             # tiered, never drops
        used         = {t: 0 for t in PATCH_TYPES}
        for p in selected:
            used[p.patch_type] = used.get(p.patch_type, 0) + p.token_cost
        kept_others.sort(key=lambda p: p.utility / max(p.token_cost, 1), reverse=True)
        total = sum(p.token_cost for p in selected)
        for p in kept_others:
            if total >= self.budget:
                break
            if p.token_cost <= type_budgets.get(p.patch_type, 0) - used.get(p.patch_type, 0):
                selected.append(p)
                used[p.patch_type] += p.token_cost
                total              += p.token_cost
        while len(selected) > 1 and sum(p.token_cost for p in selected) > self.budget:
            nc     = [q for q in selected if q.patch_type not in CRITICAL_TYPES]
            if not nc:
                break  # never drop a critical
            victim = min(nc, key=lambda q: q.utility)
            selected.remove(victim)
        return selected, len(patches) - len(selected)

    # ── numeric-first trimmer ─────────────────────────────────────
    @staticmethod
    def _trim_numeric_first(payload, max_words):
        words = payload.split()
        if len(words) <= max_words:
            return payload

        def weight(i, w):
            s = 0
            if re.search(r"\d", w):         s += 2
            if "%" in w or w.isupper():     s += 1
            if w[:1].isupper():             s += 1
            return (s, -i)

        idx = sorted(range(len(words)),
                     key=lambda i: weight(i, words[i]),
                     reverse=True)[:max_words]
        return " ".join(words[i] for i in sorted(idx))

    # ── v4.5 ultra-compact: numbers + units + 1 keyword ──────────
    @staticmethod
    def _ultra_compact(payload):
        """JWT tokens expire after 15 minutes  ->  JWT 15 minutes
           S3 pre-signed URL expiry 1 hour      ->  S3 1 hour
           Rate limit 100 req/min per user      ->  100 req min user
        """
        words = payload.split()
        if not words:
            return payload
        keep, keyword = [], None
        for w in words:
            low = w.lower().rstrip(".,;:")
            if re.search(r"\d", w) or "%" in w:
                keep.append(low)
            elif low in {"s", "h", "m", "min", "max", "kb", "mb", "gb",
                         "ms", "sec", "hour", "day", "week", "year",
                         "req", "rpm", "rps", "vcpu", "ram", "cpu"}:
                keep.append(low)
            elif keyword is None and w[:1].isupper() and len(w) >= 2:
                keyword = w
        parts = ([keyword] if keyword else []) + keep
        return " ".join(parts[:4]) if parts else words[0]

    # ── v4.5 single-token key extraction ─────────────────────────
    @staticmethod
    def _single_token_key(payload):
        """Reduce a payload to its single most distinguishing token.

        Priority: number > ALLCAPS acronym > Capitalized word > first word.
        'JWT auth: 15-min access'    -> 'JWT'
        'Rate limiting 100 req/min'  -> '100'
        'GDPR mandates deletion'     -> 'GDPR'
        """
        words = payload.split()
        if not words:
            return "?"
        # 1. First standalone number with a unit context
        for w in words:
            if re.search(r"\d", w):
                return w.rstrip(".,;:")
        # 2. First ALLCAPS acronym (>= 2 chars)
        for w in words:
            stripped = w.rstrip(".,;:")
            if len(stripped) >= 2 and stripped.isupper():
                return stripped
        # 3. First Capitalized word
        for w in words:
            stripped = w.rstrip(".,;:")
            if stripped[:1].isupper():
                return stripped
        return words[0].rstrip(".,;:")

    # ── v4.5 universal tiered critical fit ───────────────────────
    def _fit_criticals(self, selected):
        """Universal 5-stage compression. NEVER drops a critical.

        Stage 1: proportional numeric-first trim (target word count from budget/n)
        Stage 2: iteratively trim longest down to 1 word
        Stage 3: incremental ultra-compact (keyword + numbers + units)
        Stage 4: dense-merge  (one line per type, semicolon-separated)
        Stage 5: ultra-dense  (single token per critical, ONE line for all)

        Stage 5 is the universal floor: N criticals + ~4 tag tokens.
        Below that, raise BudgetTooSmallError — it is physically impossible.
        """
        crits = [p for p in selected if p.patch_type in CRITICAL_TYPES]
        if not crits:
            return

        n     = len(crits)
        limit = max(8, int(self.budget * CRITICAL_SHARE))
        floor = self._floor_for(n)   # absolute minimum for CRR=100%

        # If even ultra-dense can't fit, raise — never silently drop.
        if self.budget < floor:
            raise BudgetTooSmallError(
                f"Budget {self.budget} is below the absolute floor "
                f"({floor} tokens for {n} criticals). "
                f"Use DSPMEngine.min_budget_for_crr({n}) to compute the minimum."
            )

        # ── Stage 1: proportional trim ──
        target_words = max(2, int((limit / n - 4) / 1.4))
        for p in crits:
            p.payload = self._trim_numeric_first(p.payload, target_words)
            p.recount()

        if sum(p.token_cost for p in crits) <= limit:
            self._dense_mode = "off"
            return

        # ── Stage 2: iteratively trim longest ──
        for _ in range(n * 4):
            if sum(p.token_cost for p in crits) <= limit:
                break
            trimmable = [q for q in crits if len(q.payload.split()) > 1]
            if not trimmable:
                break
            tgt = max(trimmable, key=lambda q: len(q.payload.split()))
            tgt.payload = self._trim_numeric_first(
                tgt.payload, max(1, len(tgt.payload.split()) - 1)
            )
            tgt.recount()

        if sum(p.token_cost for p in crits) <= limit:
            self._dense_mode = "off"
            return

        # ── Stage 3: incrementally ultra-compact the longest patches ──
        for p in crits:
            p._ultra_compacted = False
        for _ in range(n * 2):
            if sum(p.token_cost for p in crits) <= limit:
                break
            candidates = [q for q in crits if not q._ultra_compacted]
            if not candidates:
                break
            tgt = max(candidates, key=lambda q: q.token_cost)
            tgt.payload = self._ultra_compact(tgt.payload)
            tgt.recount()
            tgt._ultra_compacted = True

        # Re-check whether individual-patch form now fits.
        # If yes, we're done — no dense mode needed.
        per_patch_tokens = sum(p.token_cost for p in crits)
        if per_patch_tokens <= limit:
            self._dense_mode = "off"
            return

        # ── Stage 4: dense-merge ──
        # Compute the actual dense-merge cost: 2 lines (one per type),
        # each = tag + semicolon-separated payloads.
        # Estimate via tiktoken on a synthetic dense string.
        dense_str = self._synth_dense_str(crits)
        dense_tok = count_tokens(dense_str)
        if dense_tok <= limit:
            self._dense_mode = "dense"
            return

        # ── Stage 5: ultra-dense ──
        # Reduce each critical to a single distinguishing token, merge into ONE line.
        for p in crits:
            p.payload = self._single_token_key(p.payload)
            p.recount()
        ultra_str = self._synth_ultra_str(crits)
        ultra_tok = count_tokens(ultra_str)
        if ultra_tok <= limit:
            self._dense_mode = "ultra"
            return

        # Ultra-dense still overflows budget. This means even single-token
        # keys for N criticals exceed the budget. Raise rather than silently
        # overflow or drop CRR.
        raise BudgetTooSmallError(
            f"Budget {self.budget} cannot fit {n} criticals even in ultra-dense mode "
            f"(needs {ultra_tok} tokens, floor was estimated at {floor}). "
            f"Use DSPMEngine.min_budget_for_crr({n}) = "
            f"{self.min_budget_for_crr(n)} as the safe minimum, "
            f"or reduce the number of criticals."
        )

    @staticmethod
    def _synth_dense_str(crits):
        lines = []
        for ptype in ("constraint", "decision"):
            grp = [p for p in crits if p.patch_type == ptype]
            if grp:
                merged = "; ".join(p.payload for p in grp)
                lines.append(f"[{_tag(ptype)}] {merged}")
        return "\n".join(lines)

    @staticmethod
    def _synth_ultra_str(crits):
        words = []
        for ptype in ("constraint", "decision"):
            grp = [p for p in crits if p.patch_type == ptype]
            if grp:
                tag = "C" if ptype == "constraint" else "D"
                words.append(f"[{tag}]")
                words.extend(p.payload for p in grp)
        return " ".join(words)
