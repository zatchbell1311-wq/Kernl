# dspm/extractor.py — Patch extraction + memory merge (ported from DSPM v4.3 notebook)
import re
import json
import time
from typing import List

from .patch  import SemanticPatch
from .engine import PATCH_TYPES
from .client import FatalBackendError

EXTRACTION_TEMPERATURE  = 0.0
MAX_EXTRACTION_TOKENS   = 2000
API_SLEEP_BETWEEN_CALLS = 2
EXTRACT_DEBUG           = False
REVISION_OVERLAP        = 0.40

EXTRACTION_SYSTEM_PROMPT = """You are a precision semantic extraction engine for long-context LLM memory management.
Extract ONLY the load-bearing semantic units from the turn as a JSON array (max 5 items).

Each object MUST have exactly these keys:
"type" : one of [constraint, decision, code, equation, entity, structure]
"payload" : 8-20 word note. ALWAYS include every number, version, threshold, percentage, unit and named tool VERBATIM — numeric values are NEVER dropped for brevity, even if the payload runs longer.
"dependencies" : list of patch_ids this patch logically depends on (can be empty)
"patch_id" : unique string "p{turn}_{index}" using the CURRENT TURN INDEX given below

Extraction rules:
- constraints : hard requirements, limits, non-negotiable specs. Use SPARINGLY: max 1 per turn, and MERGE aggressively — across a long conversation keep the TOTAL constraints+decisions under ~20.
- decisions   : architecture-level choices and REVISIONS that supersede earlier values. Max 1 per turn.
- code        : technology stack, libraries, APIs, implementation choices
- equation    : formulas, thresholds, numerical specs (keep every operand and unit)
- entity      : named systems, tools, organisations, acronyms
- structure   : schemas, data formats, workflow steps, architectural patterns
- MERGE related facts into ONE consolidated patch; never split a single requirement across patches
- If a turn REVISES an earlier value, extract the NEW value as a decision/constraint — never extract both old and new

OUTPUT FORMAT: raw JSON array only. No markdown fences, no reasoning, no commentary.
If nothing is extractable, return []."""

JSON_REMINDER = (
    "\n\nREMINDER: your previous reply was not a bare JSON array. "
    "Respond NOW with ONLY the JSON array — no fences, no prose, no thinking out loud."
)

ERROR_LOG: List[dict] = []


# ── JSON parsing helpers ──────────────────────────────────────────
def _strip_code_fences(text: str) -> str:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    return text.strip()


def _balanced_json_array(text: str):
    start = text.find("[")
    if start == -1:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:         esc = False
            elif c == "\\": esc = True
            elif c == '"':  in_str = False
        else:
            if c == '"':   in_str = True
            elif c == "[": depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


def _repair_json(text: str) -> str:
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    return re.sub(r",\s*([\]}])", r"\1", text)


def _try_parse_array(raw):
    if raw is None:
        return None
    text = _strip_code_fences(raw)
    try:
        v = json.loads(text)
        if isinstance(v, list):
            return v
        if isinstance(v, dict):
            for val in v.values():
                if isinstance(val, list):
                    return val
        return None
    except json.JSONDecodeError:
        pass
    cand = _balanced_json_array(text)
    if cand:
        for t in (cand, _repair_json(cand)):
            try:
                v = json.loads(t)
                if isinstance(v, list):
                    return v
            except json.JSONDecodeError:
                continue
    return None


# ── main extractor ────────────────────────────────────────────────
def extract_patches(
    turn_text: str,
    turn_index: int,
    llm_client,
    recent_context: str = "",
    dialogue_name: str = "unknown",
    max_tokens: int = MAX_EXTRACTION_TOKENS,
) -> List[SemanticPatch]:
    user_content = (
        f"RECENT CONTEXT:\n{recent_context}\n\n"
        f"CURRENT TURN INDEX: {turn_index} (use it in patch_id)\n"
        f"CURRENT TURN:\n{turn_text}"
    )
    for attempt in range(3):
        try:
            response = llm_client.chat(
                messages=[
                    {"role": "system",
                     "content": EXTRACTION_SYSTEM_PROMPT + ("" if attempt == 0 else JSON_REMINDER)},
                    {"role": "user", "content": user_content},
                ],
                temperature=EXTRACTION_TEMPERATURE,
                max_tokens=max_tokens,
            )
        except FatalBackendError:
            raise
        except Exception as e:
            ERROR_LOG.append({"dialogue": dialogue_name, "turn": turn_index,
                               "error_type": "api_error", "message": str(e)[:300]})
            time.sleep(2)
            continue

        raw = (response.choices[0].message.content or "").strip()
        arr = _try_parse_array(raw)
        if arr is None:
            time.sleep(1)
            continue

        patches, seen_fp, skipped = [], set(), 0
        for d in arr:
            if not isinstance(d, dict):
                skipped += 1; continue
            if not all(k in d for k in ("type", "payload", "patch_id")):
                skipped += 1; continue
            if d["type"] not in PATCH_TYPES:
                skipped += 1; continue
            candidate = SemanticPatch(
                patch_id     = f"p{turn_index}_{len(patches)}",
                turn_index   = turn_index,
                patch_type   = d["type"],
                payload      = str(d["payload"])[:200],
                dependencies = [str(dep) for dep in d.get("dependencies", [])
                                if isinstance(dep, (str, int))],
            )
            if candidate.fingerprint in seen_fp:
                skipped += 1; continue
            seen_fp.add(candidate.fingerprint)
            patches.append(candidate)

        time.sleep(API_SLEEP_BETWEEN_CALLS)
        return patches

    ERROR_LOG.append({"dialogue": dialogue_name, "turn": turn_index,
                       "error_type": "extraction_failure", "message": "All 3 attempts failed"})
    return []


def merge_into_memory(memory: List[SemanticPatch], new_patches: List[SemanticPatch]) -> None:
    """Merge new patches into the global memory list in-place."""
    CRITICAL_TYPES = {"constraint", "decision"}
    for p in new_patches:
        if any(m.fingerprint == p.fingerprint for m in memory):
            continue
        if p.patch_type in CRITICAL_TYPES:
            nw = set(p.payload.lower().split())
            for m in [q for q in memory
                      if q.patch_type in CRITICAL_TYPES and q.slot_key == p.slot_key]:
                ow = set(m.payload.lower().split())
                if len(nw & ow) / max(len(nw | ow), 1) >= REVISION_OVERLAP:
                    memory.remove(m)
        memory.append(p)
