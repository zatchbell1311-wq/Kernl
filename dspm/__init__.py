"""
dspm — Dynamic Semantic Patch Memory
=====================================
Token-budget compression for long-context LLM conversations.
Achieves ~80%+ Token Reduction Rate with 100% Critical Retention Rate.

Research paper: https://doi.org/10.5281/zenodo.19438636
Author: Dhruv Dubey

Quick start
-----------
    import os
    os.environ["OPENROUTER_API_KEY"] = "YOUR_KEY_HERE"   # or GROQ_API_KEY

    from dspm import DSPMEngine, make_client, extract_patches, merge_into_memory

    client  = make_client()                  # auto-picks provider + model
    engine  = DSPMEngine(budget=400)         # token budget for compressed context
    memory  = []

    for turn_idx, (role, text) in enumerate(conversation):
        patches = extract_patches(text, turn_idx, client)
        merge_into_memory(memory, patches)

    compressed_context = engine.compress(memory, query="final architecture?", turn_index=len(conversation)-1)
    context_str        = engine.build_context(compressed_context[0])
"""

__version__ = "0.2.0"
__author__  = "Dhruv Dubey"

from .patch     import SemanticPatch, count_tokens
from .engine    import DSPMEngine
from .extractor import extract_patches, merge_into_memory, ERROR_LOG
from .client    import LLMClient, FatalBackendError, make_client
from .memory    import DSPMMemory, CorruptNotebookError

__all__ = [
    "DSPMMemory",
    "CorruptNotebookError",
    "DSPMEngine",
    "extract_patches",
    "merge_into_memory",
    "SemanticPatch",
    "count_tokens",
    "ERROR_LOG",
    "LLMClient",
    "FatalBackendError",
    "make_client",
]
