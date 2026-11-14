# dspm/extractor.py - Patch extraction + memory merge (v0.2.3)
import re
import json
import time
import logging
from typing import List

from .patch  import SemanticPatch
from .engine import PATCH_TYPES
from .client import FatalBackendError


class ExtractionError(RuntimeError):
    """Raised when all extraction attempts fail. Caller must know the turn was dropped."""
    pass

EXTRACTION_TEMPERATURE  = 0.0
MAX_EXTRACTION_TOKENS   = 2000
API_SLEEP_BETWEEN_CALLS = 0  # v0.2.3: was 2s, now 0 by default
EXTRACT_DEBUG           = False
REVISION_OVERLAP = 0.60

EXTRACTION_SYSTEM_PROMPT = """You are a precision semantic extraction engine for long-context LLM memory management.
Extract load-bearing semantic units from the turn as a JSON array (max 15 items).

Each object MUST have exactly these keys:
"type" : one of [constraint, decision, slot, code, equation, entity, structure]
"payload" : 8-20 word note. ALWAYS include every number, version, threshold, percentage, unit and named tool VERBATIM.
"dependencies" : list of patch_ids this patch logically depends on (can be empty)
"patch_id" : unique string "p{turn}_{index}" using the CURRENT TURN INDEX given below

TWO-TIER EXTRACTION RULES:

TIER 1 - CRITICAL (MERGE AGGRESSIVELY, keep TOTAL under 20 across conversation):
- constraint : THE MOST IMPORTANT hard requirement/limit from this turn. Max 1 per turn.
               MERGE related constraints into ONE consolidated patch.
               Across a long conversation, keep TOTAL constraints+decisions under ~20.
- decision   : THE MOST IMPORTANT architecture-level choice or REVISION from this turn. Max 1 per turn.
               If a turn REVISES an earlier value, extract ONLY the NEW value, never the old.
               MERGE related decisions into ONE consolidated patch.

TIER 2 - NON-CRITICAL (extract ALL that apply):
- slot       : slot-value pairs from user requests. Format: "service.slot = value"
               Extract EVERY slot-value pair the user mentions, even if there are many.
- code       : technology stack, libraries, APIs, implementation choices
- equation   : formulas, thresholds, numerical specs (keep every operand and unit)
- entity     : named systems, tools, organisations, acronyms
- structure  : schemas, data formats, workflow steps, architectural patterns

CRITICAL: The max 1 per turn + MERGE rule for [constraint] and [decision] is ESSENTIAL.
It keeps the critical patch count low enough for the compression engine to maintain
100% CPI at any budget. Extract ALL slot-value pairs as [slot] instead.

SLOT EXTRACTION EXAMPLES:

Restaurant slots:
  "I want chinese food in the centre" -> [SLOT] restaurant.food = chinese, [SLOT] restaurant.area = centre

Attraction slots — ALWAYS extract the type and area:
  "I'm looking for a museum in the centre" -> [SLOT] attraction.type = museum, [SLOT] attraction.area = centre
  "Are there any boat rides?" -> [SLOT] attraction.type = boat
  "I want to visit a college" -> [SLOT] attraction.type = college
  "Multiple sports in the south" -> [SLOT] attraction.type = multiple sports, [SLOT] attraction.area = south

Hotel slots:
  "I need a 4-star hotel in the east" -> [SLOT] hotel.stars = 4, [SLOT] hotel.area = east

Train slots:
  "Train to Cambridge arriving by 21:15" -> [SLOT] train.destination = cambridge, [SLOT] train.arriveBy = 21:15

Taxi slots:
  "Taxi from Ashley Hotel to Dojo Noodle Bar" -> [SLOT] taxi.departure = ashley hotel, [SLOT] taxi.destination = dojo noodle bar

OUTPUT FORMAT: raw JSON array only. No markdown fences, no reasoning, no commentary.
If nothing is extractable, return []."""

JSON_REMINDER = (
    "\n\nREMINDER: your previous reply was not a bare JSON array. "
    "Respond NOW with ONLY the JSON array - no fences, no prose, no thinking out loud."
)

logger = logging.getLogger(__name__)


# -- JSON parsing helpers --
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


# -- main extractor --
def extract_patches(
    turn_text: str,
    turn_index: int,
    llm_client,
    recent_context: str = "",
    dialogue_name: str = "unknown",
    max_tokens: int = MAX_EXTRACTION_TOKENS,
    error_log: list = None,  # v0.2.3: per-instance, was global ERROR_LOG
) -> List[SemanticPatch]:
    if error_log is None:
        error_log = []
    user_content = (
        f"RECENT CONTEXT:\n{recent_context}\n\n"
        f"CURRENT TURN INDEX: {turn_index} (use it in patch_id)\n"
        f"CURRENT TURN:\n{turn_text}"
    )
    backoff = [1, 2, 5]  # v0.2.3: single retry layer, 3 attempts total
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
            raise  # Auth errors fail immediately, no retry
        except Exception as e:
            error_log.append({"dialogue": dialogue_name, "turn": turn_index,
                              "error_type": "api_error", "message": str(e)[:300]})
            logger.warning(f"extract_patches turn {turn_index} attempt {attempt} failed: {type(e).__name__}: {str(e)[:200]}")
            if attempt < 2:
                time.sleep(backoff[attempt])
            continue

        raw = (response.choices[0].message.content or "").strip()
        if EXTRACT_DEBUG:
            print(f"\n[EXTRACT_DEBUG] turn={turn_index} raw:\n{raw[:500]}\n")
        arr = _try_parse_array(raw)
        if arr is None:
            error_log.append({"dialogue": dialogue_name, "turn": turn_index,
                              "error_type": "parse_failure", "message": "JSON parse failed"})
            logger.warning(f"extract_patches turn {turn_index} attempt {attempt}: JSON parse failed")
            if attempt < 2:
                time.sleep(backoff[attempt])
            continue

        patches, seen_fp, skipped = [], set(), 0
        for d in arr:
            if not isinstance(d, dict):
                skipped += 1; continue
            if not all(k in d for k in ("type", "payload", "patch_id")):
                skipped += 1; continue
            # v0.2.3: normalize type (case-insensitive, strip trailing 's')
            norm_type = str(d["type"]).lower().rstrip("s")
            type_map = {
                "constraint": "constraint",
                "constrain": "constraint",
                "decision": "decision",
                "slot": "slot",
                "code": "code",
                "equation": "equation",
                "entity": "entity",
                "structure": "structure",
            }
            if norm_type not in type_map:
                skipped += 1; continue
            canonical_type = type_map[norm_type]

            candidate = SemanticPatch(
                patch_id     = f"p{turn_index}_{len(patches)}",
                turn_index   = turn_index,
                patch_type   = canonical_type,  # v0.2.3: use canonical type
                payload      = str(d["payload"])[:200],
                dependencies = [str(dep) for dep in d.get("dependencies", [])
                                if isinstance(dep, (str, int))],
            )
            if candidate.fingerprint in seen_fp:
                skipped += 1; continue
            seen_fp.add(candidate.fingerprint)
            patches.append(candidate)

        if API_SLEEP_BETWEEN_CALLS > 0:  # v0.2.3: only sleep if explicitly set
            time.sleep(API_SLEEP_BETWEEN_CALLS)
        return patches

    error_log.append({"dialogue": dialogue_name, "turn": turn_index,
                      "error_type": "extraction_failure", "message": "All 3 attempts failed"})
    logger.error(f"extract_patches turn {turn_index}: all 3 attempts failed")
    raise ExtractionError(
        f"Extraction failed for turn {turn_index} in dialogue '{dialogue_name}' after 3 attempts. "
        f"Check mem.error_log for details. The turn's content was NOT stored."
    )


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
                overlap = len(nw & ow) / max(len(nw | ow), 1)
                if overlap >= REVISION_OVERLAP:
                    # v0.6.1: Don't merge if both have different numbers (Baidik #5)
                    import re as _re
                    p_nums = set(_re.findall(r'\d+[.%a-z]*', p.payload.lower()))
                    m_nums = set(_re.findall(r'\d+[.%a-z]*', m.payload.lower()))
                    if p_nums and m_nums and p_nums != m_nums:
                        continue  # Different numbers = different rules
                    memory.remove(m)
        memory.append(p)
