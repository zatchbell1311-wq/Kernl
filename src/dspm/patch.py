"""Patch data model and token counting utilities.

The package converts raw conversational turns into typed semantic patches.
SemanticPatch stores the payload, dependency links, scoring metadata, and
compression-oriented bookkeeping fields used in the engine.

v0.1.10: no functional changes — the data model already supports
auto-dense extraction and document ingestion. Only unused imports
(dataclasses.field, config.PATCH_TYPES) were removed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import List

import tiktoken

from dspm.config import MAX_PAYLOAD_CHARS, CRITICAL_TYPES, SHORT_TAGS


def count_tokens(text: str) -> int:
    """Return the approximate token count for text using the cl100k_base encoder."""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        # Deterministic fallback for environments without tiktoken.
        return max(1, len(re.findall(r"\w+|[^\w\s]", text)))


@dataclass
class SemanticPatch:
    """Represents one typed semantic memory patch extracted from a conversation turn.

    A SemanticPatch is the atomic unit that the DSPM engine scores,
    deduplicates, compresses, and writes back into a context string.
    """

    patch_id: str
    turn_index: int
    patch_type: str
    payload: str
    dependencies: List[str]
    utility: float = 0.0
    token_cost: int = 0
    fingerprint: str = ""
    slot_key: str = ""
    is_delta: bool = False
    delta_base: str = ""
    causal_depth: int = 0

    def __post_init__(self) -> None:
        """Normalize fields, clamp payload length, and compute bookkeeping values."""
        self.patch_type = self.patch_type.lower().strip()
        self.payload = re.sub(r"\s+", " ", self.payload or "").strip()
        if len(self.payload) > MAX_PAYLOAD_CHARS:
            self.payload = self.payload[:MAX_PAYLOAD_CHARS]

        words = sorted(re.findall(r"\w+", self.payload.lower()))
        if not words:
            self.fingerprint = hashlib.md5(b"").hexdigest()[:16]
        else:
            self.fingerprint = hashlib.md5(" ".join(words).encode("utf-8")).hexdigest()[:16]

        first_keyword = ""
        for word in re.findall(r"\w+", self.payload):
            if len(word) >= 3:
                first_keyword = word.lower()
                break
        if first_keyword:
            self.slot_key = f"{self.patch_type}::{first_keyword}"
        else:
            self.slot_key = f"{self.patch_type}::topic"

        self.token_cost = count_tokens(f"[{SHORT_TAGS.get(self.patch_type, self.patch_type.upper())}] {self.payload}")

    def recount(self) -> None:
        """Recompute the token cost for the current payload value."""
        self.token_cost = count_tokens(f"[{SHORT_TAGS.get(self.patch_type, self.patch_type.upper())}] {self.payload}")

    def to_prompt_str(self) -> str:
        """Round-trip the patch into the encoded prompt context string format."""
        tag = SHORT_TAGS.get(self.patch_type, self.patch_type.upper())
        return f"[{tag}] {self.payload}"

    @property
    def is_critical(self) -> bool:
        """Return True only for constraint and decision semantic patches."""
        return self.patch_type in CRITICAL_TYPES