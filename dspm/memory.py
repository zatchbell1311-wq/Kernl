# dspm/memory.py — DSPMMemory high-level wrapper
import json
import math
import time
from typing import List, Optional

from .patch     import SemanticPatch, count_tokens
from .engine    import DSPMEngine
from .extractor import extract_patches, merge_into_memory, ERROR_LOG


class DSPMMemory:
    """
    High-level wrapper around DSPMEngine.

    Simple usage:
        from openai import OpenAI
        from dspm import DSPMMemory

        llm = OpenAI(api_key="sk-...")
        mem = DSPMMemory(budget=400, llm_client=llm, model="gpt-4o-mini")

        mem.add_turn("user", "Always return JSON. Max fee 0.5%.")
        mem.add_turn("user", "Change max fee to 0.3%.")

        print(mem.get_context())
        print(mem.stats)
    """

    def __init__(
        self,
        budget: int = 400,
        llm_client=None,
        model: str = "gpt-4o-mini",
    ):
        self.budget      = budget
        self._llm_client = llm_client
        self._model      = model
        self._engine     = DSPMEngine(budget=budget)
        self._memory: List[SemanticPatch] = []
        self._turn_index = 0
        self._last_selected: List[SemanticPatch] = []
        self._last_query   = ""

        # token counters
        self._conversation_tokens = 0
        self._patch_tokens        = 0
        self._context_tokens      = 0

        # llm telemetry
        self._llm_calls   = 0
        self._llm_errors  = 0

    # ── core methods ──────────────────────────────────────────────

    def add_turn(self, role: str, text: str) -> List[SemanticPatch]:
        """
        Add one conversation turn and extract patches from it.
        Returns the list of new patches extracted this turn.
        """
        if self._llm_client is None:
            raise RuntimeError(
                "DSPMMemory needs an llm_client to extract patches. "
                "Pass llm_client=OpenAI(...) when creating DSPMMemory."
            )

        self._conversation_tokens += count_tokens(text)

        # Build a thin LLM client shim that uses self._llm_client
        # but exposes the .chat() interface extractor expects
        shim = _LLMClientShim(self._llm_client, self._model)

        calls_before = shim.n_calls
        new_patches  = extract_patches(
            turn_text      = text,
            turn_index     = self._turn_index,
            llm_client     = shim,
            recent_context = self._build_recent_context(),
            dialogue_name  = "dspm_memory",
        )
        self._llm_calls  += shim.n_calls - calls_before
        self._llm_errors += shim.n_errors

        merge_into_memory(self._memory, new_patches)
        self._patch_tokens = sum(p.token_cost for p in self._memory)
        self._turn_index  += 1

        return new_patches

    def get_context(self, query: str = "") -> str:
        """
        Returns the compressed context string under the token budget.
        Call this when you need to inject memory into an LLM prompt.
        """
        self._last_query    = query
        self._engine.budget = self.budget
        selected, _         = self._engine.compress(
            self._memory, query or "summarise", self._turn_index - 1
        )
        self._last_selected  = selected
        self._context_tokens = sum(p.token_cost for p in selected)
        return self._engine.build_context(selected)

    def set_budget(self, budget: int) -> None:
        """Update the token budget at any time."""
        self.budget          = budget
        self._engine.budget  = budget

    def reset(self) -> None:
        """Clear all memory and counters."""
        self._memory          = []
        self._turn_index      = 0
        self._last_selected   = []
        self._last_query      = ""
        self._conversation_tokens = 0
        self._patch_tokens        = 0
        self._context_tokens      = 0
        self._llm_calls           = 0
        self._llm_errors          = 0
        self._engine.reset_ema()

    # ── persistence ───────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save memory to a JSON file."""
        data = {
            "version"    : "0.2.0",
            "budget"     : self.budget,
            "turn_index" : self._turn_index,
            "patches"    : [_patch_to_dict(p) for p in self._memory],
        }
        # atomic write — write to .tmp then rename
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        import os
        os.replace(tmp, path)

    def load(self, path: str, strict: bool = False) -> None:
        """
        Load memory from a JSON file saved with save().
        Merges into existing memory (same semantics as merge_into_memory).
        strict=True raises on corrupt files instead of skipping.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            if strict:
                raise CorruptNotebookError(f"Failed to load {path}: {e}") from e
            return

        patches = []
        for d in data.get("patches", []):
            try:
                patches.append(_dict_to_patch(d))
            except Exception:
                if strict:
                    raise
                continue

        merge_into_memory(self._memory, patches)
        self._patch_tokens = sum(p.token_cost for p in self._memory)
        if data.get("turn_index", 0) > self._turn_index:
            self._turn_index = data["turn_index"]

    # ── properties ────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        """
        Returns a dict of key metrics.
        Use trr_by_conversation for real-world compression savings.
        """
        critical_total    = sum(1 for p in self._memory
                                if p.patch_type in {"constraint", "decision"})
        critical_selected = sum(1 for p in self._last_selected
                                if p.patch_type in {"constraint", "decision"})

        crr = 100.0
        if self._last_selected:
            crr = (critical_selected / critical_total * 100
                   if critical_total > 0 else 0.0)

        trr_by_conversation = 0.0
        if self._conversation_tokens > 0:
            trr_by_conversation = round(
                (1 - self._context_tokens / self._conversation_tokens) * 100, 1
            )

        return {
            "turns"                : self._turn_index,
            "total_patches"        : len(self._memory),
            "critical_total"       : critical_total,
            "critical_selected"    : critical_selected,
            "crr"                  : round(crr, 1),
            "trr_by_conversation"  : trr_by_conversation,
            "budget"               : self.budget,
            "conversation_tokens"  : self._conversation_tokens,
            "patch_tokens"         : self._patch_tokens,
            "context_tokens"       : self._context_tokens,
            "context_built"        : len(self._last_selected) > 0,
            "llm_calls"            : self._llm_calls,
            "llm_errors"           : self._llm_errors,
        }

    @property
    def critical_patches(self) -> List[SemanticPatch]:
        """All critical patches currently in memory."""
        return [p for p in self._memory
                if p.patch_type in {"constraint", "decision"}]

    @property
    def all_patches(self) -> List[SemanticPatch]:
        """All patches currently in memory."""
        return list(self._memory)

    # ── internals ─────────────────────────────────────────────────

    def _build_recent_context(self) -> str:
        recent = self._memory[-8:]
        return "\n".join(p.to_prompt_str() for p in recent)


# ── helpers ───────────────────────────────────────────────────────

class CorruptNotebookError(Exception):
    pass


class _LLMClientShim:
    """
    Thin shim so DSPMMemory can use any OpenAI-compatible client
    (openai.OpenAI, groq.Groq, etc.) with the extract_patches interface.
    """
    def __init__(self, client, model: str):
        self._client  = client
        self._model   = model
        self.n_calls  = 0
        self.n_errors = 0

    def chat(self, max_retries: int = 5, **kwargs):
        kwargs["model"] = self._model
        backoff = [1, 2, 5, 15, 30]
        last_exc = None
        for attempt in range(max_retries):
            try:
                self.n_calls += 1
                resp = self._client.chat.completions.create(**kwargs)
                content = (resp.choices[0].message.content or "").strip()
                if not content:
                    kwargs["max_tokens"] = int(
                        kwargs.get("max_tokens", 256)
                    ) * 2
                    continue
                return resp
            except Exception as e:
                last_exc = e
                self.n_errors += 1
                print(f"  DEBUG error attempt {attempt}: {type(e).__name__}: {str(e)[:200]}")
                if attempt < max_retries - 1:
                    time.sleep(backoff[min(attempt, 4)])
        raise RuntimeError(
            f"LLM backend failed after {max_retries} attempts"
        ) from last_exc


def _patch_to_dict(p: SemanticPatch) -> dict:
    return {
        "patch_id"    : p.patch_id,
        "turn_index"  : p.turn_index,
        "patch_type"  : p.patch_type,
        "payload"     : p.payload,
        "dependencies": p.dependencies,
        "utility"     : p.utility,
        "is_delta"    : p.is_delta,
        "delta_base"  : p.delta_base,
        "causal_depth": p.causal_depth,
    }


def _dict_to_patch(d: dict) -> SemanticPatch:
    return SemanticPatch(
        patch_id     = d["patch_id"],
        turn_index   = d["turn_index"],
        patch_type   = d["patch_type"],
        payload      = d["payload"],
        dependencies = d.get("dependencies", []),
        utility      = d.get("utility", 0.0),
        is_delta     = d.get("is_delta", False),
        delta_base   = d.get("delta_base", ""),
        causal_depth = d.get("causal_depth", 0),
    )
