"""LLM extraction helpers for DSPM.

The extraction layer is intentionally dependency-light: it accepts a
user-supplied OpenAI-compatible client object and returns SemanticPatch
objects without requiring the caller to install an LLM SDK package.

v0.1.10: `numeric_density` is now public (memory.add_turn calls it to
auto-enable dense mode for fact-heavy turns), and the dense-mode prompt
is hardened against two document-mode failure modes found in live
testing: experimental/config parameters misclassified as constraints
(fake criticals consuming the protected budget), and multiple statistics
merged into single payloads (numbers lost before any compression stage
ever ran).

v0.1.11: extractor hardening for reasoning models, from three rounds of
live document testing: the output budget rises 2000 -> 4000 tokens
(reasoning models spend output tokens on chain-of-thought before the
answer), HTTP-200-but-empty responses are retried like transient errors
(previously only exceptions retried, so an empty response silently
produced a 0-patch chunk), and gpt-oss models automatically run at low
reasoning effort — with graceful fallbacks if the provider or a custom
client rejects the reasoning_effort kwarg.

v0.1.13: value directives are decisions. A live conversation-mode test
showed 'webhook timeout set to 30 seconds' classified as [CODE] — the
v0.1.10 config-≠-constraint rule (written from document-mode evidence)
over-applied to speaker directives, so the replaced value never entered
the critical lineage and the revision history had nothing to bake. Both
prompts now distinguish value DIRECTIVES (always decisions) from
detached configuration DESCRIPTIONS (code/structure). New live gate:
scripts/smoke_conversation.py — extraction-layer changes now require
BOTH smoke tests green before release.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional

from dspm.patch import SemanticPatch
from dspm.config import PATCH_TYPES, EXTRACT_MAX_TOKENS


def _strip_code_fences(text: str) -> str:
    """Remove an outer json code fence wrapper from an LLM response."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```json\s*", "", text, flags=re.I)
        text = re.sub(r"^```\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _balanced_json_array(text: str) -> Optional[str]:
    """Return the first balanced JSON array substring if one is present."""
    stripped = _strip_code_fences(text)
    left = stripped.find("[")
    if left == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx in range(left, len(stripped)):
        ch = stripped[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    return stripped[left:idx + 1]
    return None


def _repair_json(text: str) -> str:
    """Repair common JSON shape issues emitted by LLMs, such as smart quotes and trailing commas."""
    text = text.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'").replace("\u2018", "'")
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text


def parse_extraction(raw_text: str) -> List[Dict[str, Any]]:
    """Parse an LLM raw JSON response into dictionaries.

    The parser tries the following robust strategies in sequence: direct
    JSON parse, code-fence stripping, balanced bracket extraction, and
    string repair. It returns an empty list on failure.
    """
    candidates = []
    text = raw_text.strip()
    candidates.append(text)
    candidates.append(_strip_code_fences(text))
    balanced = _balanced_json_array(text)
    if balanced:
        candidates.append(balanced)
    for candidate in candidates:
        candidate = _repair_json(candidate)
        try:
            data = json.loads(candidate)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                if isinstance(data.get("patches"), list):
                    return data["patches"]
        except Exception:
            pass
    return []


# v0.1.10: renamed from _numeric_density and made PUBLIC — memory.add_turn
# now calls it to auto-enable dense mode for fact-heavy turns. It was dead
# code in v0.1.9 (defined but never called): the manual dense flag was too
# easy to forget, and document runs silently used the 5-patch funnel.
def numeric_density(turn_text: str) -> float:
    """Fraction of words containing digits. 0.05 ≈ one number per 20 words."""
    words = turn_text.split()
    if not words:
        return 0.0
    return sum(1 for w in words if re.search(r"\d", w)) / len(words)


# v0.1.10: the dense prompt gains three rules, each mapped to a failure
# observed in live document testing:
#   - config/experimental params are NOT constraints (they were extracted
#     as [CON]/[DEC] and consumed the protected critical budget)
#   - one numeric result per payload (multiple stats per payload meant
#     the 200-char cap or trimming destroyed the rest)
#   - tables: one patch per row (results rows were summarized away)
#
# v0.1.13: the config rule is refined after a live conversation-mode
# failure (webhook rematch, round 2): 'webhook timeout set to 30 seconds'
# — a value DIRECTIVE — was classified as [CODE] because it pattern-
# matched 'configuration parameter', so the 30 never entered the
# critical lineage and the (was X) baking had nothing to carry. The rule
# now distinguishes SPEAKER DIRECTIVES (decisions, always) from DETACHED
# CONFIGURATION DESCRIPTIONS (a document reporting its own settings —
# code/structure). Auto-dense routes number-bearing conversation turns
# through this dense prompt, so it must be safe in BOTH modes.
def _build_system_prompt(dense: bool = False) -> str:
    if dense:
        return (
            "You are a semantic patch extractor. Extract semantic patches from the turn as a raw JSON array. "
            "This is FACT-DENSE content (document/spec/notes): extract up to 10 patches, prioritizing every "
            "distinct numeric value, threshold, date, and named specification — each with its exact value. "
            "Each element must be a JSON object with exactly these keys: "
            '"patch_type" (one of: constraint, decision, code, equation, entity, structure), '
            '"payload" (an 8-20 word note preserving every number, version, threshold AND UNIT verbatim; '
            'when the turn revises an earlier value, KEEP the revision verb in the payload, '
            'e.g. "webhook timeout updated to 10 seconds" not just "webhook timeout 10 seconds"), '
            '"patch_id" (a unique string), '
            '"dependencies" (a list of patch_ids this depends on; empty list if none). '
            "Rules: constraint max 2 per turn. decision max 2 per turn, extract only the NEW value for revisions. "
            "IMPORTANT: distinguish value DIRECTIVES from configuration DESCRIPTIONS. "
            "A speaker instructing, setting, or changing a value — 'set the webhook timeout "
            "to 30 seconds', 'change the fee to 0.5%', 'the timeout must be 10 seconds' — is a "
            "DECISION, even when the parameter sounds like configuration. Constraints are "
            "non-negotiable REQUIREMENTS on the system being built (deadlines, limits, "
            "compliance rules). Only detached descriptions of a document's or experiment's "
            "own settings (a study's temperature, a table's parameters, model names, budget "
            "values used in an evaluation) are NOT constraints — store those as code or "
            "structure patches. "
            "Each distinct numeric result (score, percentage, p-value, threshold, date) gets its "
            "OWN patch with the value and its unit/label verbatim; never merge multiple statistics "
            "into one payload. When the content contains a table, extract one patch per row, "
            "preserving the row label and its values verbatim. "
            "code holds implementation detail. equation holds formulas and numeric results. "
            "entity holds named things. structure holds schemas and workflows. "
            "Reply with a raw JSON array only — no markdown fences, no commentary."
        )
    return (
        "You are a semantic patch extractor. Extract at most 5 semantic patches from the turn as a raw JSON array. "
        "Each element must be a JSON object with exactly these keys: "
        '"patch_type" (one of: constraint, decision, code, equation, entity, structure), '
        '"payload" (an 8-20 word note preserving every number, version, threshold AND UNIT verbatim; '
        'when the turn revises an earlier value, KEEP the revision verb in the payload, '
        'e.g. "webhook timeout updated to 10 seconds" not just "webhook timeout 10 seconds"), '
        '"patch_id" (a unique string), '
        '"dependencies" (a list of patch_ids this depends on; empty list if none). '
        "Rules: constraint max 1 per turn, truly non-negotiable specs only. "
        "decision max 1 per turn, extract only the NEW value for revisions. "
        "A statement that sets, changes, or decides a parameter value ('set the timeout "
        "to 30 seconds', 'timeout updated to 10 seconds') is a DECISION, not code — even "
        "when it sounds like a configuration parameter. "
        "code holds implementation detail. equation holds formulas. entity holds named things. "
        "structure holds schemas and workflows. "
        "Reply with a raw JSON array only — no markdown fences, no commentary."
    )


def extract_turn(llm_client, model: str, turn_text: str, turn_index: int,
                 recent_context: str = '', dense: bool = False,
                 max_tokens: Optional[int] = None,
                 reasoning_effort: Optional[str] = None) -> List[SemanticPatch]:
    """Extract semantic patches from a conversation turn using an LLM client.

    `dense=True` widens the extraction funnel — more patches allowed,
    constraint/decision caps relaxed — for fact-heavy turns where the
    standard 5-patch cap loses most numeric facts. memory.add_turn
    decides this automatically via numeric_density(); callers may still
    force it with the flag.

    v0.1.11 parameters (both optional, fully backward compatible):
      max_tokens — output budget; defaults to config.EXTRACT_MAX_TOKENS
                   (4000). Pass a lower value for providers that cap
                   completion tokens below 4000.
      reasoning_effort — None (default) auto-detects: "low" for gpt-oss
                   models (extraction is a structured-output task, not a
                   reasoning task), nothing for other models. Pass
                   "low"/"medium"/"high" to force it for any model that
                   supports the parameter."""
    if llm_client is None:
        raise ValueError("llm_client is required to extract semantic patches")

    system_prompt = _build_system_prompt(dense=dense)
    user_content = (
        f"Recent context:\n{recent_context}\n\nTurn {turn_index}:\n{turn_text}"
        if recent_context else f"Turn {turn_index}:\n{turn_text}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    # v0.1.11: build the request kwargs. reasoning_effort is only sent
    # when explicitly requested or auto-detected for gpt-oss models —
    # other providers reject unknown kwargs with a non-retryable 400.
    if max_tokens is None:
        max_tokens = EXTRACT_MAX_TOKENS
    request_kwargs = {"max_tokens": max_tokens}
    if reasoning_effort is None and "gpt-oss" in model.lower():
        reasoning_effort = "low"
    if reasoning_effort:
        request_kwargs["reasoning_effort"] = reasoning_effort

    # v0.1.11 retry loop. Three failure classes now retry (up to 3
    # attempts, 2s/4s backoff between them):
    #   1. transient exceptions (429 rate limits, 5xx, network) — as before
    #   2. HTTP 200 with EMPTY content — NEW: reasoning models can burn
    #      the whole output budget on chain-of-thought and return no
    #      text; v0.1.10 accepted this silently, producing 0-patch
    #      chunks (the missing-numbers failure in live document testing)
    #   3. rejection of our reasoning_effort kwarg — NEW: a 400 while the
    #      kwarg is set, or a TypeError from a custom client whose
    #      create() doesn't accept it, strips the kwarg and retries once
    #      before treating the error as fatal
    # Non-retryable errors (400/401/403/404, outside the kwarg caveat)
    # still raise immediately — retrying a bad key or dead model is
    # pointless. If every attempt yields empty content, the turn degrades
    # gracefully to 0 patches instead of crashing add_turn().
    NON_RETRYABLE = {400, 401, 403, 404}
    response = None
    last_exc = None
    raw = ""
    for attempt in range(3):
        try:
            response = llm_client.chat.completions.create(
                model=model, messages=messages, temperature=0.0,
                **request_kwargs)
            # v0.1.11: reading the response INSIDE the try also makes odd
            # response shapes (missing .choices/.message) retryable
            # instead of crashing after the loop.
            raw = (response.choices[0].message.content or "").strip()
            if raw:
                break
        except Exception as e:
            last_exc = e
            code = getattr(e, 'status_code', None)
            if code in NON_RETRYABLE:
                if request_kwargs.get("reasoning_effort") and code == 400:
                    request_kwargs.pop("reasoning_effort")
                    continue
                raise
            if request_kwargs.get("reasoning_effort") and isinstance(e, TypeError):
                request_kwargs.pop("reasoning_effort")
                continue
        if attempt < 2:
            time.sleep(2 * (attempt + 1))   # 2s, 4s — between attempts only
    if response is None:
        raise last_exc

    parsed = parse_extraction(raw)

    patches: List[SemanticPatch] = []
    seen_fingerprints: set = set()
    for item in parsed:
        if not isinstance(item, dict):
            continue
        p_type = str(item.get('patch_type') or item.get('type') or '').lower().strip()
        if p_type not in PATCH_TYPES:
            continue
        payload = str(item.get('payload') or item.get('text') or '')
        if not payload.strip():
            continue
        patch_id = f"p{turn_index}_{len(patches)}"

        # v0.1.10: same behavior as before, readable form — accepts a
        # list of dependency ids or a single string; anything else is [].
        raw_deps = item.get('dependencies')
        if isinstance(raw_deps, list):
            deps = [str(x) for x in raw_deps]
        elif isinstance(raw_deps, str):
            deps = [raw_deps]
        else:
            deps = []

        patch = SemanticPatch(
            patch_id=patch_id,
            turn_index=turn_index,
            patch_type=p_type,
            payload=payload,
            dependencies=deps,
        )

        if patch.fingerprint in seen_fingerprints:
            continue
        seen_fingerprints.add(patch.fingerprint)
        patches.append(patch)

    return patches