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
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
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


def extract_turn(llm_client, model: str, turn_text: str, turn_index: int, recent_context: str = '') -> List[SemanticPatch]:
    """Extract semantic patches from a conversation turn using an LLM client.

    The function sends a structured prompt to the LLM, parses the response,
    and materializes SemanticPatch objects while filtering invalid types and
    de-duplicating patch records within the same turn.
    """
    if llm_client is None:
        raise ValueError("llm_client is required to extract semantic patches")

    system_prompt = (
        "You are a semantic patch extractor. Extract at most 5 semantic patches from the turn. "
        "Allowed patch types exactly: constraint, decision, code, equation, entity, structure. "
        "Rules: constraint max 1 patch, truly non-negotiable specs only. "
        "decision max 1 patch, extract new decision values for revisions. "
        "code holds implementation detail. equation holds formulas. entity holds named things. "
        "structure holds schemas, records, or module boundaries. Reply with a raw JSON array only, no markdown fences."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Recent context:\n{recent_context}\n\nTurn:\n{turn_text}"},
    ]

    response = llm_client.chat.completions.create(model=model, messages=messages, temperature=0.0, max_tokens=2000)
    raw = response.choices[0].message.content
    parsed = parse_extraction(raw)

    patches: List[SemanticPatch] = []
    seen_ids: set = set()
    for item in parsed:
        if not isinstance(item, dict):
            continue
        p_type = str(item.get('patch_type', '')).lower().strip()
        if p_type not in PATCH_TYPES:
            continue
        payload = str(item.get('payload') or item.get('text') or '')
        if not payload.strip():
            continue
        # materialize a unique id
        patch_id = str(item.get('patch_id') or f"patch-{turn_index}-{len(patches)}-{hash(payload)}")
        if patch_id in seen_ids:
            continue
        seen_ids.add(patch_id)
        dependencies = item.get('dependencies') or []
        if not isinstance(dependencies, list):
            dependencies = []
        patch = SemanticPatch(
            patch_id=patch_id,
            turn_index=turn_index,
            patch_type=p_type,
            payload=payload,
            dependencies=[str(x) for x in dependencies],
            utility=float(item.get('utility') or 0.0),
            token_cost=0,
            fingerprint=item.get('fingerprint') or '',
            slot_key=item.get('slot_key') or '',
            is_delta=bool(item.get('is_delta') or False),
            delta_base=item.get('delta_base') or '',
            causal_depth=int(item.get('causal_depth') or 0),
        )
        patches.append(patch)

    return patches
