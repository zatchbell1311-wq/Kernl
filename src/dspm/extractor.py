"""LLM extraction helpers for DSPM.

The extraction layer is intentionally dependency-light: it accepts a
user-supplied OpenAI-compatible client object and returns SemanticPatch
objects without requiring the caller to install an LLM SDK package.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, Iterable, List, Optional

from dspm.patch import SemanticPatch
from dspm.config import PATCH_TYPES, SHORT_TAGS


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


# v0.1.9: density metric for dense mode. A turn with many numeric facts
# (documents, specs, research notes) needs a wider extraction funnel
# than a conversational turn; the fixed 5-patch cap was shown in live
# document testing to capture only a fraction of the facts.
def _numeric_density(turn_text: str) -> float:
    """Fraction of words containing digits. 0.05 ≈ one number per 20 words."""
    words = turn_text.split()
    if not words:
        return 0.0
    return sum(1 for w in words if re.search(r"\d", w)) / len(words)


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
        "code holds implementation detail. equation holds formulas. entity holds named things. "
        "structure holds schemas and workflows. "
        "Reply with a raw JSON array only — no markdown fences, no commentary."
    )


def extract_turn(llm_client, model: str, turn_text: str, turn_index: int,
                 recent_context: str = '', dense: bool = False) -> List[SemanticPatch]:
    """Extract semantic patches from a conversation turn using an LLM client.

    v0.1.9: `dense=True` widens the extraction funnel — more patches
    allowed, constraint/decision caps relaxed — for fact-heavy turns
    where the standard 5-patch cap loses most numeric facts."""
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

    # Retry with backoff on transient errors (429 rate limits, 5xx,
    # network). Non-retryable HTTP errors (400/401/403/404) raise
    # immediately — retrying a bad key or dead model is pointless.
    NON_RETRYABLE = {400, 401, 403, 404}
    response = None
    last_exc = None
    for attempt in range(3):
        try:
            response = llm_client.chat.completions.create(
                model=model, messages=messages, temperature=0.0, max_tokens=2000)
            break
        except Exception as e:
            last_exc = e
            if getattr(e, 'status_code', None) in NON_RETRYABLE:
                raise
            time.sleep(2 * (attempt + 1))   # 2s, 4s
    if response is None:
        raise last_exc

    raw = response.choices[0].message.content or ""
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

        patch = SemanticPatch(
            patch_id=patch_id,
            turn_index=turn_index,
            patch_type=p_type,
            payload=payload,
            dependencies=[str(x) for x in (item.get('dependencies') or [])
                          if isinstance(item.get('dependencies'), list)] or
                         ([str(item['dependencies'])] if isinstance(item.get('dependencies'), str) else []),
        )

        if patch.fingerprint in seen_fingerprints:
            continue
        seen_fingerprints.add(patch.fingerprint)
        patches.append(patch)

    return patches