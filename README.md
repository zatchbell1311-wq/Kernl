# DSPM Memory

**Compress multi-turn LLM conversations by 80%+ while guaranteeing every constraint and decision survives.**

[![PyPI version](https://img.shields.io/pypi/v/dspm-memory.svg)](https://pypi.org/project/dspm-memory/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

---

## The problem

Long conversations eat your context window. Naive truncation drops the constraint from turn 3 that the entire system depends on. DSPM fixes this.

## How it works

DSPM converts each conversation turn into typed **semantic patches** — constraint, decision, code, entity, structure — and compresses them under a fixed token budget. Critical patches (constraints and decisions) are structurally protected: they survive compression even when everything else is trimmed.


Raw conversation (452 tokens, 18 turns)
↓
Semantic extraction → 28 patches, 18 critical
↓
7-stage compression pipeline
↓
Compressed context (249 tokens) — 100% of criticals intact





The 7-stage pipeline: dedup → slot fusion → delta encoding → causal pruning → utility scoring → shadow selection → adaptive budgeting.

---

## Install

```bash
pip install dspm-memory
```

Requires Python 3.9+. Works with any OpenAI-compatible LLM provider.

---

## Quickstart

```python
from openai import OpenAI
from dspm import DSPMMemory

# Works with OpenAI, Groq, Together, Ollama, or any OpenAI-compatible endpoint
llm = OpenAI(
    api_key="sk-...",
    # base_url="https://api.groq.com/openai/v1"  # uncomment for Groq
)

memory = DSPMMemory(budget=250, llm_client=llm, model="gpt-4o-mini")

memory.add_turn("user", "Build a REST API. Must use PostgreSQL, JWT auth, deadline is Friday.")
memory.add_turn("assistant", "PostgreSQL with SQLAlchemy, JWT via python-jose. Access tokens 15 min.")
memory.add_turn("user", "All PII must be encrypted at rest with AES-256. No exceptions.")
memory.add_turn("assistant", "AES-256 at rest for all PII fields, keys in AWS KMS with quarterly rotation.")

context = memory.get_context(query="What are the hard requirements?")
print(context)
print(memory.stats)
```

**Output:**

[CON] Stack: PostgreSQL, JWT auth. Deadline Friday.
[CON] Access tokens 15 minutes.
[CON] All PII must be encrypted at rest with AES-256. No exceptions.
[CON] AES-256 at rest; keys in AWS KMS, quarterly rotation.


Every constraint is present. Every time.

---

## The guarantee

Constraint (`[CON]`) and decision (`[DEC]`) patches are **structurally protected**:

- Never dropped by deduplication, fusion, or pruning
- Under budget pressure, payloads are trimmed numbers-first — thresholds, versions, and units survive longest
- When a constraint is revised mid-conversation (e.g. TTL 60s → 300s), the new value supersedes the old
- A critical is only dropped as a last resort: every critical already at its 2-word floor and budget still cannot hold them

Ablation result: removing the shadow-selection mechanism collapses CRR from 100% to 37.9%, isolating the guarantee to a single identifiable component.

---

## Results

Tested across 7 domains × 40 turns each:

| Budget | Tokens used | TRR | CRR |
|--------|-------------|-----|-----|
| 150 | 150 | 66.8% | **100%** |
| 250 | 249 | 82.8% | **100%** |
| 400 | 395 | 72.4% | **100%** |

CRR = Critical Retention Rate. TRR = Token Reduction Ratio.

---

## API reference

### `DSPMMemory(budget, llm_client, model)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `budget` | `int` | `250` | Maximum tokens in the compressed output |
| `llm_client` | `OpenAI` | `None` | Any OpenAI-compatible client |
| `model` | `str` | `"gpt-4o-mini"` | Model used for patch extraction |

### Methods

| Method | Description |
|--------|-------------|
| `add_turn(role, text)` | Add a conversation turn. Returns extracted patches. |
| `get_context(query="")` | Returns compressed context string, ready for your prompt. |
| `reset()` | Clear all memory and start fresh. |

### Properties

| Property | Description |
|----------|-------------|
| `memory.stats` | Dict with token counts, patch counts, CRR |
| `memory.critical_patches` | List of all critical patches currently in memory |

---

## Supported providers

```python
# OpenAI
llm = OpenAI(api_key="sk-...")

# Groq (free tier available)
llm = OpenAI(base_url="https://api.groq.com/openai/v1", api_key="gsk-...")

# Together AI
llm = OpenAI(base_url="https://api.together.xyz/v1", api_key="...")

# Ollama (local, no key needed)
llm = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
```

---

## Paper

**DSPM: A Critical-Retention Approach to Long-Context Memory Compression for LLM Conversations**  
Dhruv Dubey, 2026  
[arXiv:2409.XXXXX](https://arxiv.org/abs/2409.XXXXX)

---

## Version history

| Version | Changes |
|---------|---------|
| 0.1.3 | Revision supersession fix: stale same-type criticals now removed when superseded (verified live — a webhook timeout revised 30s→10s collapsed from 4 contradictory entries to a single correct one). Robust normalized content-word matching. All stale matches removed. Same-value restatement collapse. |
| 0.1.2 | Fixed critical-patch ID collisions (CRR 36% → 100% in 18-turn live test). Budget enforced on joined context string. Cross-type revision supersession. |
| 0.1.1 | Fixed T4 dropping criticals with dependencies. Fixed T3 payload mangling. Fixed extractor schema mismatch. Corrected inverted trim sort. Deterministic patch IDs. |
| 0.1.0 | Initial release. |

---

## License

MIT © 2026 Dhruv Dubey
