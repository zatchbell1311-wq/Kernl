"""LLM extraction helpers for DSPM.

The extraction layer is intentionally dependency-light: it accepts a
user-supplied OpenAI-compatible client object and returns SemanticPatch
objects without requiring the caller to install an LLM SDK package.
"""

from __future__ import annotations

import json
import re
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


# FIXED (v0.1.1): the prompt never specified the JSON keys, so models
# commonly returned {"type": ...} or free-form objects that the parser
# then discarded. The schema is now stated explicitly, including the
# payload-length and numbers-verbatim requirements from the paper.
def _build_system_prompt() -> str:
    return (
        "You are a semantic patch extractor. Extract at most 5 semantic patches from the turn as a raw JSON array. "
        "Each element must be a JSON object with exactly these keys: "
        '"patch_type" (one of: constraint, decision, code, equation, entity, structure), '
        '"payload" (an 8-20 word note preserving every number, version, and threshold verbatim), '
        '"patch_id" (a unique string), '
        '"dependencies" (a list of patch_ids this depends on; empty list if none). '
        "Rules: constraint max 1 per turn, truly non-negotiable specs only. "
        "decision max 1 per turn, extract only the NEW value for revisions. "
        "code holds implementation detail. equation holds formulas. entity holds named things. "
        "structure holds schemas and workflows. "
        "Reply with a raw JSON array only — no markdown fences, no commentary."
    )


def extract_turn(llm_client, model: str, turn_text: str, turn_index: int, recent_context: str = '') -> List[SemanticPatch]:
    """Extract semantic patches from a conversation turn using an LLM client.

    The function sends a structured prompt to the LLM, parses the response,
    and materializes SemanticPatch objects while filtering invalid types and
    de-duplicating patch records within the same turn.
    """
    if llm_client is None:
        raise ValueError("llm_client is required to extract semantic patches")

    system_prompt = _build_system_prompt()
    user_content = (
        f"Recent context:\n{recent_context}\n\nTurn {turn_index}:\n{turn_text}"
        if recent_context else f"Turn {turn_index}:\n{turn_text}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    # FIXED (v0.1.1): some reasoning models return content=None with HTTP
    # 200; `or ""` prevents a TypeError and lets the parser return [].
    response = llm_client.chat.completions.create(model=model, messages=messages, temperature=0.0, max_tokens=2000)
    raw = response.choices[0].message.content or ""
    parsed = parse_extraction(raw)

    patches: List[SemanticPatch] = []
    seen_fingerprints: set = set()
    for item in parsed:
        if not isinstance(item, dict):
            continue
        # FIXED (v0.1.1): accept "type" as a fallback key — most models
        # return {"type": ...} even when the prompt asks for "patch_type".
        p_type = str(item.get('patch_type') or item.get('type') or '').lower().strip()
        if p_type not in PATCH_TYPES:
            continue
        payload = str(item.get('payload') or item.get('text') or '')
        if not payload.strip():
            continue
        # FIXED (v0.1.1): deterministic per-turn id (hash() is not stable
        # across processes) — mirrors the Colab pipeline's p{turn}_{i}.
        patch_id = str(item.get('patch_id') or f"p{turn_index}_{len(patches)}")

        # FIXED (v0.1.1): only pass the fields the LLM should control.
        # Fingerprint/slot_key/token_cost were previously taken from the
        # LLM's (often empty/garbage) values; SemanticPatch.__post_init__
        # now computes them authoritatively.
        patch = SemanticPatch(
            patch_id=patch_id,
            turn_index=turn_index,
            patch_type=p_type,
            payload=payload,
            dependencies=[str(x) for x in (item.get('dependencies') or [])
                          if isinstance(item.get('dependencies'), list)] or
                         ([str(item['dependencies'])] if isinstance(item.get('dependencies'), str) else []),
        )

        # FIXED (v0.1.1): dedupe by fingerprint (same content) instead of
        # by LLM-supplied id — ids are unreliable across turns.
        if patch.fingerprint in seen_fingerprints:
            continue
        seen_fingerprints.add(patch.fingerprint)
        patches.append(patch)

    return patches