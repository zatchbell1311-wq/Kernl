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

# v0.1.10: auto-dense extraction -- the fraction of digit-bearing words at
# which add_turn() routes a turn through the dense extraction funnel
# (up to 10 patches) even when the memory was created with dense=False.
DENSE_AUTO_THRESHOLD = 0.05

# v0.1.11: extraction output budget, in completion tokens.
EXTRACT_MAX_TOKENS = 4000

# v0.1.18 (instrumentation batch): LLM-call observability knobs.
# LLM_STRICT_DEFAULT=False keeps the v0.1.17 silent-fallback behavior;
# DSPMMemory(strict_llm=True) turns empty / finish_reason=="length" /
# unparseable extraction output into a RuntimeError instead of a silent
# 0-patch turn.
LLM_STRICT_DEFAULT = False

# v0.1.18: floor for extraction max_tokens. Reasoning models (gpt-oss et
# al.) spend completion tokens on hidden chain-of-thought before any
# visible text, so a small max_tokens returns HTTP 200 with EMPTY content.
# The floor only binds when a caller passes an explicit max_tokens below
# it; the default EXTRACT_MAX_TOKENS=4000 is unaffected.
EXTRACT_MIN_TOKENS = 512

# v0.1.18: where reasoning_effort travels in the request:
#   "top"        -- v0.1.11 behavior: top-level kwarg (Groq accepts this)
#   "extra_body" -- nested in extra_body, for stricter OpenAI-compatible
#                   providers that reject unknown top-level kwargs
#   "off"        -- never send it
REASONING_PARAM_MODE = "top"
