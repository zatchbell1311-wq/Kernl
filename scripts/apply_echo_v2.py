import ast
import re
import shutil
import sys
from pathlib import Path

PATH = Path("src/dspm/memory.py")

B1_FIND = r'''    return current or history


class DSPMMemory:'''

B1_REPL = r'''    return current or history


# v0.1.18 (echo guard v2, task 2b): fed-back-context echo suppression.
#
# The turn-20 r2 failure (diag_out_r2, budget=60, feedback_context=True,
# gpt-oss-20b): the extractor emitted five [DEC] payloads built out of
# the fed-back CP12 context. Four were partial echoes ("set alerts to
# 50", "set within to 2", "set duration to 90 days", "pod limit raised
# to 2Gi") that only bloated the store; the fifth -- "Connect retry
# updated to 120 (was 300)", fused from two trimmed context lines --
# superseded the live p7_0 via rule (a) and the live p2_0 via rule (a2),
# silently deleting two facts.
#
# Rule 1 (verbatim echo): the normalized payload is a substring of the
# fed-back context and not of the turn -> "echo".
# Rule 2 (fusion/partial echo): the payload shares >= 2 content tokens
# with the context and ZERO with the current turn -> "echo".
# Rule 3 (zero attribution): the payload shares no content token with
# the turn OR the context, and has >= 3 content tokens -> "unattributed".
#
# Telemetry: rule 1+2 drops count in llm_echo_drops, rule 3 drops in
# llm_unattributed_drops. Opt-in via DSPMMemory(echo_guard=True);
# defaults are unchanged.

_FEEDBACK_FENCE_OPEN = (
    "=== FEEDBACK CONTEXT (REPLAY OF YOUR OWN PRIOR SUMMARY -- "
    "FOR CONTINUITY ONLY, DO NOT EXTRACT FROM IT) BEGIN ==="
)
_FEEDBACK_FENCE_CLOSE = "=== FEEDBACK CONTEXT (DO NOT EXTRACT FROM IT) END ==="
_FEEDBACK_FENCE_RULES = (
    "The block above between the BEGIN/END markers is your own prior "
    "summary of this conversation, replayed so you know what is already "
    "stored. It is NOT part of the current turn. Never copy, quote, "
    "reword, or derive any patch payload from it -- not even partially, "
    "not even fused with current-turn content. Every patch must be fully "
    "supported by the current turn text alone. If the current turn "
    "states nothing new, return an empty JSON array. When a patch "
    "updates an existing item, restate the item's name and value in the "
    "payload so the change is traceable to this turn."
)


def _guard_content_tokens(text: str) -> set:
    """Content tokens for the echo guard: the file's own _content_words
    plus bare 1-char numbers. Digits are matched on word boundaries ONLY,
    so the '2' inside a compound like '2Gi' is never split off."""
    toks = _content_words(text)
    toks.update(re.findall(r"\b\d+(?:\.\d+)?\b", text.lower()))
    return toks


def _echo_guard_reason(payload: str, turn_text: str,
                       context_text: str) -> Optional[str]:
    """Classify one freshly extracted payload against the current turn
    text and the fed-back context. Returns None (keep), "echo" (drop,
    counted in llm_echo_drops), or "unattributed" (drop, counted in
    llm_unattributed_drops)."""
    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").lower()).strip()

    p_norm, t_norm, c_norm = _norm(payload), _norm(turn_text), _norm(context_text)
    if not p_norm:
        return None
    # Rule 1: verbatim context fragment the turn never states.
    if p_norm in c_norm and p_norm not in t_norm:
        return "echo"
    p_toks = _guard_content_tokens(payload)
    if not p_toks:
        return None
    turn_toks = _guard_content_tokens(turn_text)
    ctx_toks = _guard_content_tokens(context_text)
    in_turn = len(p_toks & turn_toks)
    in_ctx_only = len(p_toks & ctx_toks - turn_toks)
    # Rule 2: fusion/partial echo -- everything attributable is context.
    if in_ctx_only >= 2 and in_turn == 0:
        return "echo"
    # Rule 3: zero attribution -- supported by neither text it was shown.
    if in_turn + in_ctx_only == 0 and len(p_toks) >= 3:
        return "unattributed"
    return None


class DSPMMemory:'''

B2_FIND = r'''                 feedback_context: bool = True,
                 reasoning_param_mode: Optional[str] = None,'''

B2_REPL = r'''                 feedback_context: bool = True,
                 feedback_fence: bool = False,
                 echo_guard: bool = False,
                 reasoning_param_mode: Optional[str] = None,'''

B3_FIND = r'''        self.feedback_context = bool(feedback_context)
        self.reasoning_param_mode = reasoning_param_mode'''

B3_REPL = r'''        self.feedback_context = bool(feedback_context)
        # v0.1.18 (task 2/2b): opt-in echo suppressors. Defaults preserve
        # existing behavior exactly.
        self.feedback_fence = bool(feedback_fence)
        self.echo_guard = bool(echo_guard)
        self.reasoning_param_mode = reasoning_param_mode'''

B4_FIND = r'''            "llm_parse_failures": 0, "llm_fallbacks": 0,
        }'''

B4_REPL = r'''            "llm_parse_failures": 0, "llm_fallbacks": 0,
            "llm_echo_drops": 0, "llm_unattributed_drops": 0,
        }'''

B5_FIND = r'''        recent = self._last_context if self.feedback_context else ""
        patches = extract_turn(self.llm_client, self.model, text, self.turns,'''

B5_REPL = r'''        recent = self._last_context if self.feedback_context else ""
        # v0.1.18 (task 2): opt-in prompt fence -- wrap the fed-back
        # context in do-not-extract markers plus rules.
        if self.feedback_fence and recent:
            recent = (f"{_FEEDBACK_FENCE_OPEN}\n{recent}\n"
                      f"{_FEEDBACK_FENCE_CLOSE}\n{_FEEDBACK_FENCE_RULES}")
        patches = extract_turn(self.llm_client, self.model, text, self.turns,'''

B6_FIND = r'''                               reasoning_param_mode=self.reasoning_param_mode,
                               max_tokens_floor=self.max_tokens_floor)'''

B6_REPL = r'''                               reasoning_param_mode=self.reasoning_param_mode,
                               max_tokens_floor=self.max_tokens_floor)
        # v0.1.18 (echo guard v2): drop payloads echoed out of the
        # fed-back context, and confabulated payloads, BEFORE the merge
        # rules can supersede anything with them. Inert when
        # feedback_context is off.
        if self.echo_guard and self.feedback_context and self._last_context:
            kept: List[SemanticPatch] = []
            for p in patches:
                reason = _echo_guard_reason(p.payload, text, self._last_context)
                if reason == "echo":
                    self._llm_telemetry["llm_echo_drops"] += 1
                elif reason == "unattributed":
                    self._llm_telemetry["llm_unattributed_drops"] += 1
                else:
                    kept.append(p)
            patches = kept'''

BLOCKS = [("B1", B1_FIND, B1_REPL), ("B2", B2_FIND, B2_REPL),
          ("B3", B3_FIND, B3_REPL), ("B4", B4_FIND, B4_REPL),
          ("B5", B5_FIND, B5_REPL), ("B6", B6_FIND, B6_REPL)]


def main():
    raw = PATH.read_bytes().decode("utf-8")
    crlf = "\r\n" in raw
    text = raw.replace("\r\n", "\n")
    if "echo_guard" in text or "feedback_fence" in text:
        sys.exit("ABORT: echo_guard/feedback_fence already present in memory.py")
    for name, find, _ in BLOCKS:
        n = text.count(find)
        if n != 1:
            sys.exit(f"ABORT: anchor {name} matched {n} times (need exactly 1). Nothing was changed.")
    for name, find, repl in BLOCKS:
        text = text.replace(find, repl)
    ast.parse(text)
    if not re.search(r"^import re\b|^import .*\bre\b", text, re.M):
        print("WARNING: 'import re' not found at module level; check memory.py imports.")
    shutil.copy(PATH, str(PATH) + ".bak_pre_echo")
    out = text.replace("\n", "\r\n") if crlf else text
    PATH.write_bytes(out.encode("utf-8"))
    print("OK: all 6 blocks applied; backup at", str(PATH) + ".bak_pre_echo")


main()