"""Configuration constants and default budget policies for DSPM.

This module centralizes the patch taxonomy, default budget constants, and
budget-sharing parameters used by the compression engine.
"""

PATCH_TYPES = ["constraint", "decision", "code", "equation", "entity", "structure"]
CRITICAL_TYPES = {"constraint", "decision"}
SHORT_TAGS = {
    "constraint": "CON",
    "decision": "DEC",
    "code": "CODE",
    "equation": "EQ",
    "entity": "ENT",
    "structure": "STR",
}

DEFAULT_BUDGET = 250
CRITICAL_SHARE = 1.0
CRITICAL_MAX_WORDS = 12
REVISION_OVERLAP = 0.40

W_ALIGN = 0.45
W_DEP = 0.20
W_RECENCY = 0.15
W_COST = 0.20

ALPHA_EMA = 0.5
SHADOW_THRESHOLD = 0.05
RECENCY_LAMBDA = 0.15
DELTA_MIN_SAVING = 1

BASE_BUDGET_FRACTIONS = {
    "constraint": 0.30,
    "decision": 0.25,
    "code": 0.20,
    "equation": 0.08,
    "entity": 0.08,
    "structure": 0.09,
}

MAX_PAYLOAD_CHARS = 200

# v0.1.10: auto-dense extraction — the fraction of digit-bearing words at
# which add_turn() routes a turn through the dense extraction funnel
# (up to 10 patches) even when the memory was created with dense=False.
# 0.05 ≈ one number per 20 words. Deliberately sensitive: the dense prompt
# is a permissive superset of the standard one, so triggering it on a
# number-bearing conversational turn is harmless, while MISSING it on a
# fact-dense document chunk loses most numeric facts at extraction (the
# 70%/85% document-test failure mode).
DENSE_AUTO_THRESHOLD = 0.05

# v0.1.11: extraction output budget, in completion tokens.
# v0.1.0–v0.1.10 hardcoded 2000 inside extract_turn. Live document testing
# (three comparison rounds with gpt-oss models) showed reasoning models
# spend output tokens on chain-of-thought BEFORE the visible content — a
# 2000-token budget can be consumed entirely by reasoning, returning HTTP
# 200 with EMPTY content and silently leaving the chunk with 0 patches.
# 4000 covers low-effort reasoning plus a full 10-patch dense extraction.
# Override per call with extract_turn(max_tokens=...) for providers that
# cap completions below this.
EXTRACT_MAX_TOKENS = 4000