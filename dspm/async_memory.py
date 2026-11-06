"""
dspm/async_memory.py - v0.3.0 AsyncDSPMMemory

Truly async version of DSPMMemory. Uses AsyncOpenAI for non-blocking LLM calls.
Allows multiple DSPMMemory instances to run concurrently via asyncio.gather().

Usage:
    import asyncio
    from openai import AsyncOpenAI
    from dspm import AsyncDSPMMemory

    async def main():
        llm = AsyncOpenAI(api_key="sk-...")
        mem = AsyncDSPMMemory(budget=400, llm_client=llm, model="gpt-4o-mini")
        
        await mem.async_add_turn("user", "Always return JSON. Max fee 0.5%.")
        ctx = await mem.async_get_context(query="What are the rules?")
        print(ctx)

    asyncio.run(main())
"""
import asyncio
import logging
from typing import List

from .patch import SemanticPatch, count_tokens
from .engine import DSPMEngine
from .extractor import async_extract_patches, merge_into_memory
from .client import FatalBackendError

logger = logging.getLogger(__name__)


class AsyncLLMClientShim:
    """
    Async shim so AsyncDSPMMemory can use any AsyncOpenAI-compatible client.
    chat() is a coroutine, as expected by async_extract_patches.
    """
    def __init__(self, client, model: str):
        self._client = client
        self._model = model
        self.n_calls = 0
        self.n_errors = 0

    async def chat(self, **kwargs):
        kwargs["model"] = self._model
        self.n_calls += 1

        try:
            resp = await self._client.chat.completions.create(**kwargs)
        except Exception as e:
            self.n_errors += 1
            status = getattr(e, "status_code", None)
            if status in (400, 401, 403, 404):
                raise FatalBackendError(f"HTTP {status}: {str(e)[:200]}") from e
            raise

        content = (resp.choices[0].message.content or "").strip()
        if not content:
            new_max = int(kwargs.get("max_tokens", 256)) * 2
            kwargs["max_tokens"] = new_max
            self.n_calls += 1
            try:
                resp = await self._client.chat.completions.create(**kwargs)
            except Exception as e:
                self.n_errors += 1
                status = getattr(e, "status_code", None)
                if status in (400, 401, 403, 404):
                    raise FatalBackendError(f"HTTP {status}: {str(e)[:200]}") from e
                raise
            content = (resp.choices[0].message.content or "").strip()
            if not content:
                raise RuntimeError("LLM returned empty content after max_tokens retry")

        return resp


class AsyncDSPMMemory:
    """
    Async high-level wrapper around DSPMEngine.
    """
    def __init__(
        self,
        budget=400,
        llm_client=None,
        model: str = "gpt-4o-mini",
        max_extraction_tokens: int = 2000,
        ctx_window: int = 8192,
        response_reserve: int = 2000,
    ):
        if budget == "auto":
            upper = ctx_window - response_reserve - 500
            budget = max(100, int(upper * 0.25))
            
        self.budget = int(budget)
        self._llm_client = llm_client
        self._model = model
        self._max_extraction_tokens = max_extraction_tokens
        self._engine = DSPMEngine(budget=self.budget)
        self._memory: List[SemanticPatch] = []
        self._turn_index = 0
        self._last_selected: List[SemanticPatch] = []
        self._last_query = ""
        
        self._conversation_tokens = 0
        self._patch_tokens = 0
        self._context_tokens = 0
        self._llm_calls = 0
        self._llm_errors = 0
        self._error_log: List[dict] = []

    async def async_add_turn(self, role: str, text: str) -> List[SemanticPatch]:
        """
        Async version of add_turn. Extracts patches without blocking the event loop.
        """
        if self._llm_client is None:
            raise RuntimeError("AsyncDSPMMemory needs an llm_client (AsyncOpenAI).")

        self._conversation_tokens += count_tokens(text)

        shim = AsyncLLMClientShim(self._llm_client, self._model)
        
        try:
            new_patches = await async_extract_patches(
                turn_text=text,
                turn_index=self._turn_index,
                llm_client=shim,
                recent_context=self._build_recent_context(),
                dialogue_name="dspm_async_memory",
                max_tokens=self._max_extraction_tokens,
                error_log=self._error_log,
            )
        except Exception:
            self._llm_errors += shim.n_errors
            raise

        self._llm_calls += shim.n_calls
        self._llm_errors += shim.n_errors

        merge_into_memory(self._memory, new_patches)
        self._patch_tokens = sum(p.token_cost for p in self._memory)
        self._turn_index += 1

        return new_patches

    async def async_add_turns_batch(self, turns: list) -> List[SemanticPatch]:
        """
        v0.3.0: 2-pass batch extraction for long dialogues.
        
        Pass 1: Extract all turns in parallel (no recent_context).
        Pass 2: Sequential merge for revisions/dedup.
        
        Tradeoff: Loses recent_context continuity between turns.
        ~3-5x speedup on long dialogues (50+ turns).
        
        Args:
            turns: List of (role, text) tuples
            
        Returns:
            List of all extracted patches.
        """
        if self._llm_client is None:
            raise RuntimeError("AsyncDSPMMemory needs an llm_client (AsyncOpenAI).")

        # ── Pass 1: Extract all turns in parallel ──
        async def extract_one(turn_idx, text):
            self._conversation_tokens += count_tokens(text)
            shim = AsyncLLMClientShim(self._llm_client, self._model)
            try:
                patches = await async_extract_patches(
                    turn_text=text,
                    turn_index=turn_idx,
                    llm_client=shim,
                    recent_context="",  # No context in batch mode
                    dialogue_name="dspm_batch",
                    max_tokens=self._max_extraction_tokens,
                    error_log=self._error_log,
                )
            except Exception:
                self._llm_errors += shim.n_errors
                raise
            self._llm_calls += shim.n_calls
            self._llm_errors += shim.n_errors
            return turn_idx, patches

        # Run all extractions concurrently
        tasks = [
            extract_one(idx, text)
            for idx, (role, text) in enumerate(turns)
        ]
        results = await asyncio.gather(*tasks)

        # ── Pass 2: Sequential merge for revisions/dedup ──
        all_patches = []
        for turn_idx, patches in sorted(results, key=lambda x: x[0]):
            merge_into_memory(self._memory, patches)
            all_patches.extend(patches)
            self._turn_index = turn_idx + 1

        self._patch_tokens = sum(p.token_cost for p in self._memory)
        return all_patches

    async def async_get_context(self, query: str = "") -> str:
        """
        Returns the compressed context string. Compression is CPU-bound and fast,
        so we just run it directly.
        """
        self._last_query = query
        self._engine.budget = self.budget
        
        selected, _ = self._engine.compress(
            self._memory, query or "summarise", self._turn_index - 1
        )
        
        self._last_selected = selected
        self._context_tokens = sum(p.token_cost for p in selected)
        return self._engine.build_context(selected)

    def _build_recent_context(self) -> str:
        recent = self._memory[-8:]
        return "\n".join(p.to_prompt_str() for p in recent)

    @property
    def stats(self) -> dict:
        critical_total = sum(1 for p in self._memory if p.patch_type in {"constraint", "decision"})
        critical_selected = sum(1 for p in self._last_selected if p.patch_type in {"constraint", "decision"})

        crr = 100.0
        if self._last_selected:
            crr = (critical_selected / critical_total * 100 if critical_total > 0 else 0.0)

        trr_by_conversation = 0.0
        if self._conversation_tokens > 0:
            trr_by_conversation = round((1 - self._context_tokens / self._conversation_tokens) * 100, 1)

        return {
            "turns": self._turn_index,
            "total_patches": len(self._memory),
            "critical_total": critical_total,
            "critical_selected": critical_selected,
            "crr": round(crr, 1),
            "trr_by_conversation": trr_by_conversation,
            "budget": self.budget,
            "conversation_tokens": self._conversation_tokens,
            "patch_tokens": self._patch_tokens,
            "context_tokens": self._context_tokens,
            "context_built": len(self._last_selected) > 0,
            "llm_calls": self._llm_calls,
            "llm_errors": self._llm_errors,
        }

    @property
    def critical_patches(self) -> List[SemanticPatch]:
        return [p for p in self._memory if p.patch_type in {"constraint", "decision"}]

    @property
    def all_patches(self) -> List[SemanticPatch]:
        return list(self._memory)

    @property
    def error_log(self) -> list:
        return self._error_log
