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
