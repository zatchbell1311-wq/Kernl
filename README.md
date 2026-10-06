# DSPM Memory

> Compress multi-turn LLM conversations by 80%+ while every constraint and decision survives.
>
> 100% critical retention across our benchmark suite — under impossibly small budgets, criticals are trimmed to their word floor and dropped only as a documented last resort.

[![PyPI version](https://img.shields.io/pypi/v/dspm-memory?label=dspm-memory%200.1.9&color=blue)](https://pypi.org/project/dspm-memory/0.1.9/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-30%20passed-brightgreen)](https://github.com/zatchbell1311-wq/dspm)

---

## Table of Contents

- [The Problem](#the-problem)
- [How It Works](#how-it-works)
- [Install](#install)
- [Quickstart](#quickstart)
- [Persistence](#persistence-memory-across-chats-and-sessions)
- [Dense Mode](#dense-mode-fact-heavy-content)
- [The Guarantee](#the-guarantee)
- [Results](#results)
- [Supported Providers](#supported-providers)
- [API Reference](#api-reference)
- [Paper](#paper)
- [Version History](#version-history)
- [License](#license)

---

## The Problem

Long conversations eat your context window. Naive truncation drops the constraint from turn 3 that the entire system depends on. DSPM fixes this.

---

## How It Works

DSPM converts each conversation turn into typed semantic patches — `constraint`, `decision`, `code`, `entity`, `structure` — and compresses them under a fixed token budget. Critical patches (constraints and decisions) are structurally protected: they survive compression even when everything else is trimmed.

```
Raw conversation (452 tokens, 18 turns)
        ↓
Semantic extraction → 28 patches, 18 critical
        ↓
7-stage compression pipeline
        ↓
Compressed context (249 tokens) — 100% of criticals intact
```

**The 7-stage pipeline:** dedup → slot fusion → delta encoding → causal pruning → utility scoring → shadow selection → adaptive budgeting.

---

## Install

```bash
pip install dspm-memory
```

Requires Python 3.9+. Works with any OpenAI-compatible LLM provider.

**Optional — semantic ranking:**

```bash
pip install "dspm-memory[semantic]"
```

Adds sentence-transformer-based query alignment for better non-critical patch ranking (downloads PyTorch and a small embedding model on first use). Without it, DSPM falls back to a neutral alignment score — criticals are fully protected either way.

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

# budget=60 so compression is visible even in this small example;
# use 200-300 for real conversations
memory = DSPMMemory(budget=60, llm_client=llm, model="gpt-4o-mini")

memory.add_turn("user", "Building a payment API. Hard rules: PCI-DSS compliant, max fee 0.5%, deadline Friday.")
memory.add_turn("assistant", "PCI-DSS needs tokenized card storage and quarterly ASV scans. Tracking the 0.5% fee cap.")
memory.add_turn("user", "Webhook timeout must be 30 seconds.")
memory.add_turn("assistant", "Webhook timeout set to 30 seconds with automatic retries.")
memory.add_turn("user", "Change the webhook timeout to 10 seconds instead — 30 is too slow.")
memory.add_turn("assistant", "Updated: webhook timeout is now 10 seconds; the 30s setting is superseded.")

context = memory.get_context(query="What are the hard requirements?")
print(context)
print(memory.stats)
```

**Output:**

```
[DEC] Track the 0.5% fee cap.
[CON] PCI-DSS requires tokenized card storage and ASV
[DEC] webhook timeout updated to 10 seconds
[CON] PCI-DSS compliant, max fee 0.5%, deadline Friday
{'turns': 6, 'total_patches': 4, 'critical_total': 4, 'critical_selected': 4, 'crr': 100, 'budget': 60, 'raw_tokens': 61, 'context_tokens': 56, 'trr': 9}
```

Note the revision: only the 10-second entry remains — the 30s setting is fully superseded. The fee cap and deadline survived compression, `trr: 9` shows real token reduction even at this small scale, and `crr: 100` throughout.

**Revision history at tight budgets:** even at `budget=34` (the same conversation, squeezed harder), the supersession bakes the replaced value directly into the surviving patch:

```
[DEC] webhook timeout now 10 seconds (was 30)
[CON] PCI-DSS compliant, max fee 0.5%, deadline Friday
```

The `(was X)` span is atomic — it survives trimming as a unit, so "what replaced what" is always answerable.

---

## Persistence: Memory Across Chats and Sessions

Save the notebook when a chat ends, load it when the next one starts — Chat 2 remembers Chat 1, and revisions supersede old values across sessions:

```python
# Chat 1 — Monday
memory = DSPMMemory(budget=250, llm_client=llm, model="gpt-4o-mini")
memory.add_turn("user", "Building a budgeting app. Hard rules: must work offline, deadline Oct 20.")
# ... chat ...
memory.save("user_dhruv.json")

# Chat 2 — Thursday, new process, fresh start
memory = DSPMMemory(budget=250, llm_client=llm, model="gpt-4o-mini")
memory.load("user_dhruv.json")
memory.get_context(query="What were my hard rules?")
# → [CON] must work offline
# → [CON] deadline Oct 20        ← recalled from Chat 1, zero API cost

# Revisions work across chats too:
memory.add_turn("user", "The deadline moved to November 5.")
# → Oct 20 is superseded. November 5 replaces it.
```

Save files are portable JSON, written atomically (a crash mid-save can't corrupt the notebook), and merge-on-load means saved revisions supersede stale values. One file per user = each person's long-term memory.

---

## Dense Mode: Fact-Heavy Content

Standard extraction caps at 5 patches per turn — right for conversations, lossy for documents. For fact-dense content (research papers, specs, agreements), enable dense mode:

```python
memory = DSPMMemory(budget=250, llm_client=llm, model="gpt-4o-mini", dense=True)
```

Dense mode raises the cap to 10 patches per turn, allows 2 constraints + 2 decisions (instead of 1 + 1), and prioritizes every distinct numeric value, threshold, and date. Costs more extraction tokens — leave it off for normal conversations.

---

## The Guarantee

`[CON]` and `[DEC]` patches are structurally protected:

- Never dropped by deduplication, fusion, or pruning
- Under budget pressure, payloads are trimmed numbers-first — units bind to their numbers (`30 seconds` stays `30 seconds`, never bare `30`), and temporal values bind to their nouns (`deadline Friday` keeps `Friday`)
- When a constraint is revised mid-conversation (e.g. TTL 60s → 300s), the new value supersedes the old — including cross-type revisions and terse patches. The superseded value is preserved in the surviving payload as `(was X)` — the full revision story survives even aggressive compression
- A critical is only dropped as a last resort: every critical already at its 2-word floor and budget still cannot hold them
- Transient API failures (rate limits, 5xx) are retried automatically with backoff — three attempts before the error surfaces

> **Ablation result:** Removing the shadow-selection mechanism collapses CRR from 100% to 37.9%, isolating the guarantee to a single identifiable component.

---

## Results

Tested across 7 domains × 40 turns each:

| Budget | Tokens Used | TRR   | CRR  |
|--------|-------------|-------|------|
| 150    | 150         | 66.8% | 100% |
| 250    | 249         | 82.8% | 100% |
| 400    | 395         | 72.4% | 100% |

`CRR` = Critical Retention Rate. `TRR` = Token Reduction Ratio.

**Provenance:** measured on 7 hand-authored 40-turn technical dialogues (API design, ML ops, IoT, security IR, supply chain, clinical workflow, project management); extraction and judging by GLM via OpenRouter. Numbers vary by extraction model and domain — check your own with `memory.stats`.

---

## Supported Providers

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

## API Reference

### `DSPMMemory(budget, llm_client, model, dense)`

| Parameter    | Type   | Default         | Description |
|--------------|--------|-----------------|-------------|
| `budget`     | int    | 250             | Maximum tokens in the compressed output |
| `llm_client` | OpenAI | None            | Any OpenAI-compatible client |
| `model`      | str    | `"gpt-4o-mini"` | Model used for patch extraction |
| `dense`      | bool   | False           | Fact-heavy mode: 10-patch cap, relaxed limits, numeric-value priority. For documents/specs; off for conversations. |

The budget can be changed at any time:

```python
memory.budget = 100       # takes effect on the next get_context() call
memory.set_budget(100)    # explicit alternative — same effect
```

### Methods

| Method                  | Description |
|-------------------------|-------------|
| `add_turn(role, text)`  | Add a conversation turn. Returns extracted patches. |
| `get_context(query="")` | Returns compressed context string, ready for your prompt. |
| `set_budget(budget)`    | Update the token budget. Takes effect on the next `get_context()` call. |
| `save(path)`            | Persist the memory notebook to JSON (atomic write). |
| `load(path)`            | Load a notebook into memory (merge semantics, supersession on load). |
| `reset()`               | Clear all memory and start fresh. |

### Properties

| Property                  | Description |
|---------------------------|-------------|
| `memory.stats`            | Dict with token counts, patch counts, active budget, dense flag, and CRR |
| `memory.critical_patches` | List of all critical patches currently in memory |
| `memory.all_patches`      | List of every patch in memory |

### Recommended Answer Prompt

Guards against numeric hallucination when a value is absent from the compressed context:

```python
SYSTEM = (
    "Answer ONLY from the provided memory. [CON]=constraint, [DEC]=decision. "
    "If the memory contains the topic but not the exact value, say "
    "'the exact value is not in memory' — NEVER estimate or invent numbers."
)
```

---

## Paper

**DSPM: A Critical-Retention Approach to Long-Context Memory Compression for LLM Conversations**  
Dhruv Dubey, 2026  
Zenodo: [10.5281/zenodo.19438636](https://doi.org/10.5281/zenodo.19438636)

---

## Version History

| Version | Changes |
|---------|---------|
| **0.1.9** | **Revision history retention:** superseded values are baked into surviving payloads as protected `(was X)` atomic spans — "what replaced what" is now answerable at any budget. Temporal values (`deadline Friday`) protected as constraint values. New opt-in **dense mode** (`dense=True`) widens extraction for fact-heavy content. 3 regression tests (30 total). |
| 0.1.8 | README wording corrected — 0.1.7's wheel shipped the previous wording, caught by post-build wheel verification check immediately after upload. No code changes. |
| 0.1.7 | README-first release. Compound units (`per minute`) and key constraint nouns (`timeout`, `limit`) survive trimming as atomic number-spans. `now`/`moved` recognized as revision verbs; extractor preserves revision verbs in payloads. Claims scoped to benchmarks; retry behavior accurately described. `semantic` extra documented. Python 3.13 classifier; status → Beta. 3 new tests (27 total). |
| 0.1.6 | **Cross-type revision supersession fixed:** revisions sharing only 3 content words previously survived as contradictions. Units now bind to their numbers during trimming. Automatic retry with backoff on rate limits and transient 5xx errors in `add_turn()`. 5 regression tests (24 total). |
| 0.1.5 | **Budget fix:** changing `memory.budget` was silently ignored — the engine kept an independent budget copy. Now synced on every `get_context()`. New `set_budget()` method. `stats` measures the actual joined context and reports the active budget. 2 regression tests (19 total). |
| 0.1.4 | Persistence: `memory.save()` / `memory.load()` — cross-session long-term memory as portable JSON. Atomic writes, merge-on-load, revisions supersede stale values. 8 new tests (17 total). |
| 0.1.3 | Revision supersession fix: stale same-type criticals now removed when superseded. Robust normalized content-word matching. |
| 0.1.2 | Fixed critical-patch ID collisions (CRR 36% → 100% in 18-turn live test). Budget enforced on joined context string. |
| 0.1.1 | Fixed T4 dropping criticals with dependencies. Fixed T3 payload mangling. Fixed extractor schema mismatch. |
| 0.1.0 | Initial release. |

---

## License

MIT © 2026 Dhruv Dubey
