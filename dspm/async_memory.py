# dspm/async_memory.py — v0.6.0 AsyncDSPMMemory
"""
True async memory extraction using AsyncOpenAI.

2-pass design:
  Pass 1: Extract all turns concurrently (asyncio.gather)
  Pass 2: Merge sequentially (revision tracking needs order)

Usage:
    import asyncio
    from openai import AsyncOpenAI
    from dspm import AsyncDSPMMemory

    async def main():
        llm = AsyncOpenAI(api_key="...", base_url="https://api.groq.com/openai/v1")
        mem = AsyncDSPMMemory(budget=400, llm_client=llm, model="qwen/qwen3.8-27b")

        # Batch: extract all turns concurrently, then merge
        await mem.async_add_turns_batch([
            ("user", "Max fee is 0.5%. JWT expires in 30 minutes."),
            ("assistant", "Understood."),
            ("user", "Update JWT to 15 minutes."),
        ])

        ctx = await mem.async_get_context(query="What are the rules?")
        print(ctx)

    asyncio.run(main())
"""
import asyncio
import json
import time
import logging
from typing import List, Optional
from .patch import SemanticPatch, count_tokens
from .engine import DSPMEngine
from .extractor import extract_patches, merge_into_memory, ExtractionError
from .memory import DSPMMemory

# v0.6.0: type normalization map (same as extractor.py's local type_map)
TYPE_MAP = {
    "constraint": "constraint", "constrain": "constraint",
    "decision": "decision", "slot": "slot",
    "code": "code", "equation": "equation",
    "entity": "entity", "structure": "structure",
}

logger = logging.getLogger(__name__)


class AsyncDSPMMemory(DSPMMemory):
    """
    Async version of DSPMMemory.

    Uses AsyncOpenAI for concurrent extraction calls.
    Falls back to sync DSPMMemory methods for compression (pure Python, instant).

    Key difference from sync DSPMMemory:
    - async_add_turns_batch(): extracts ALL turns concurrently, then merges
      sequentially. This is the 2-pass design that enables true parallelism.
    - async_add_turn(): single async extraction (still sequential merge)
    - async_get_context(): same as sync (pure Python, but async-compatible)
    """

    def __init__(
        self,
        budget=400,
        llm_client=None,
        model: str = "gpt-4o-mini",
        max_extraction_tokens: int = 2000,
        ctx_window: int = 8192,
        max_concurrency: int = 10,
    ):
        super().__init__(
            budget=budget,
            llm_client=llm_client,
            model=model,
            max_extraction_tokens=max_extraction_tokens,
            ctx_window=ctx_window,
        )
        self._max_concurrency = max_concurrency
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def _async_extract_one_turn(
        self, turn_text: str, turn_index: int, recent_context: str
    ) -> List[SemanticPatch]:
        """Extract patches from one turn using async LLM call."""
        async with self._semaphore:
            try:
                # Build the chat messages
                from .extractor import EXTRACTION_SYSTEM_PROMPT, JSON_REMINDER
                messages = [
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": (
                        f"RECENT CONTEXT:\n{recent_context}\n\n"
                        f"CURRENT TURN INDEX: {turn_index} (use it in patch_id)\n"
                        f"CURRENT TURN:\n{turn_text}"
                    )},
                ]

                # Async LLM call
                response = await self._llm_client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=self._max_extraction_tokens,
                )

                raw = (response.choices[0].message.content or "").strip()

                # Parse JSON (reuse extractor's parsing)
                from .extractor import _try_parse_array
                arr = _try_parse_array(raw)

                if arr is None:
                    self._error_log.append({
                        "dialogue": "async_memory",
                        "turn": turn_index,
                        "error_type": "parse_failure",
                        "message": "JSON parse failed",
                    })
                    return []

                # Build patches (reuse extractor's logic)
                # type_map defined locally below (not importable from extractor)
                patches = []
                seen_fp = set()
                for d in arr:
                    if not isinstance(d, dict):
                        continue
                    if not all(k in d for k in ("type", "payload", "patch_id")):
                        continue
                    norm_type = str(d["type"]).lower().rstrip("s")

                    if norm_type not in TYPE_MAP:
                        continue
                    candidate = SemanticPatch(
                        patch_id=f"p{turn_index}_{len(patches)}",
                        turn_index=turn_index,
                        patch_type=tm[norm_type],
                        payload=str(d["payload"])[:200],
                        dependencies=[str(dep) for dep in d.get("dependencies", [])
                                      if isinstance(dep, (str, int))],
                    )
                    if candidate.fingerprint in seen_fp:
                        continue
                    seen_fp.add(candidate.fingerprint)
                    patches.append(candidate)

                return patches

            except Exception as e:
                self._error_log.append({
                    "dialogue": "async_memory",
                    "turn": turn_index,
                    "error_type": "api_error",
                    "message": str(e)[:300],
                })
                logger.warning(f"async extract turn {turn_index} failed: {type(e).__name__}: {str(e)[:200]}")
                return []

    async def async_add_turn(
        self, role: str, text: str
    ) -> List[SemanticPatch]:
        """
        Add one turn asynchronously.
        Extraction is async, merge is sync (sequential).
        """
        self._conversation_tokens += count_tokens(text)

        # Build recent context from memory
        recent_context = self._build_recent_context()

        # Async extraction
        new_patches = await self._async_extract_one_turn(
            text, self._turn_index, recent_context
        )

        # Sync merge (revision tracking needs order)
        merge_into_memory(self._memory, new_patches)
        self._patch_tokens = sum(p.token_cost for p in self._memory)
        self._turn_index += 1

        return new_patches

    async def async_add_turns_batch(
        self, turns: List[tuple], show_progress: bool = True
    ) -> dict:
        """
        Add multiple turns with CONCURRENT extraction.

        2-pass design:
          Pass 1: Extract all turns concurrently (asyncio.gather)
          Pass 2: Merge sequentially (preserves revision tracking)

        This is the key method for batch processing.
        Piotr verified: async (6.48 min) vs threads (7.44 min) on 96-turn dialogue.

        Args:
            turns: List of (role, text) tuples
            show_progress: Print progress

        Returns:
            dict with stats: {turns, patches, errors, time_s}
        """
        t0 = time.time()

        # Build recent context for each turn (cumulative)
        contexts = []
        accumulated = []
        for role, text in turns:
            self._conversation_tokens += count_tokens(text)
            contexts.append("\n".join(accumulated[-8:]))  # last 8 patches as context
            accumulated.append(f"[{role}] {text}")

        if show_progress:
            print(f"  Extracting {len(turns)} turns concurrently "
                  f"(max_concurrency={self._max_concurrency})...")

        # Pass 1: Extract ALL turns concurrently
        tasks = [
            self._async_extract_one_turn(text, idx, ctx)
            for idx, ((role, text), ctx) in enumerate(zip(turns, contexts))
        ]
        all_patches = await asyncio.gather(*tasks)

        if show_progress:
            total_patches = sum(len(p) for p in all_patches)
            print(f"  Extracted {total_patches} patches from {len(turns)} turns")

        # Pass 2: Merge sequentially (preserves revision tracking)
        total_merged = 0
        for idx, patches in enumerate(all_patches):
            merge_into_memory(self._memory, patches)
            total_merged += len(patches)
            self._turn_index = max(self._turn_index, idx + 1)

        self._patch_tokens = sum(p.token_cost for p in self._memory)

        elapsed = time.time() - t0
        stats = {
            "turns": len(turns),
            "patches_extracted": sum(len(p) for p in all_patches),
            "patches_merged": total_merged,
            "total_patches_in_memory": len(self._memory),
            "errors": len(self._error_log),
            "time_s": round(elapsed, 2),
        }

        if show_progress:
            print(f"  Merged {total_merged} patches in {elapsed:.1f}s")
            print(f"  Memory: {len(self._memory)} patches, "
                  f"{self._patch_tokens} tokens")

        return stats

    async def async_get_context(self, query: str = "") -> str:
        """Get compressed context. Same as sync (pure Python) but async-compatible."""
        return self.get_context(query)
