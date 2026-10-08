# DSPM — Dynamic Semantic Patch Memory

**by Kernl.ai**

> Compress multi-turn LLM conversations while every constraint, decision and revision survives.
>
> Up to ~83% token reduction at a 250-token budget, with 100% critical retention across our benchmark suite. Under very small budgets, criticals are trimmed to their word floor and dropped only as a documented last resort.

[![PyPI version](https://img.shields.io/pypi/v/dspm-memory?label=pypi&cacheSeconds=0)](https://pypi.org/project/dspm-memory/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**Who it's for:** agents and chatbots whose long conversations must not lose a constraint, a decision, or a revised value (support bots, coding agents, assistants that remember a user's hard rules across sessions).

**Who it's not for:** summarizing or answering questions over documents. See [Known Limitations](#known-limitations).

---

## See It Work

A payment-API conversation where a value changes mid-chat. After compression, only the current value remains, and the hard rules are intact:

```python
from openai import OpenAI
from dspm import DSPMMemory

llm = OpenAI(api_key="sk-...")   # any OpenAI-compatible endpoint works

# budget=60 so compression is visible even in this small example;
# use 200-300 for real conversations
memory = DSPMMemory(budget=60, llm_client=llm, model="gpt-4o-mini")

memory.add_turn("user", "Building a payment API. Hard rules: PCI-DSS compliant, max fee 0.5%, deadline Friday.")
memory.add_turn("assistant", "PCI-DSS needs tokenized card storage and quarterly ASV scans. Tracking the 0.5% fee cap.")
memory.add_turn("user", "Webhook timeout must be 30 seconds.")
memory.add_turn("assistant", "Webhook timeout set to 30 seconds with automatic retries.")
memory.add_turn("user", "Change the webhook timeout to 10 seconds instead — 30 is too slow.")
memory.add_turn("assistant", "Updated: webhook timeout is now 10 seconds; the 30s setting is superseded.")

print(memory.get_context(query="What are the hard requirements?"))
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

Only the 10-second entry remains; the 30-second setting is superseded. The fee cap and deadline survived, and `crr: 100` throughout. At even tinier budgets the revision story is kept in minimal form, e.g. `10 (was 30)`.

**Try to break it:** give it a conversation where a value changes several times under a very small budget, and open an issue if a constraint or revision is lost.

---

## The Problem

Long conversations eat your context window. Naive truncation drops the constraint from turn 3 that the entire system depends on. DSPM is built so that does not happen.

---

## How It Works

DSPM converts each conversation turn into typed semantic patches — `constraint`, `decision`, `code`, `equation`, `entity`, `structure` — and compresses them under a fixed token budget. Critical patches (constraints and decisions) are structurally protected: they survive compression even when everything else is trimmed.

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

**Optional: semantic ranking**

```bash
pip install "dspm-memory[semantic]"
```

Adds sentence-transformer-based query alignment for better non-critical patch ranking (downloads PyTorch and a small embedding model on first use). Without it, DSPM falls back to a neutral alignment score — criticals are fully protected either way.

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

## The Guarantee

`[CON]` and `[DEC]` patches are structurally protected:

- Never dropped by deduplication, fusion, or pruning
- Under budget pressure, payloads are trimmed numbers-first — and units bind to their numbers (`30 seconds` stays `30 seconds`, never just `30`)
- When a constraint is revised mid-conversation (e.g. TTL 60s → 300s), the new value supersedes the old — including cross-type revisions (a constraint revised by a decision), terse patches, and full revision chains (60s → 300s → 120s keeps the whole lineage). Even when a starvation budget forces criticals to their word floor, the revision story survives in minimal form: `10 (was 30)`
- A critical is only dropped as a last resort: every critical already at its word floor and budget still cannot hold them
- Transient API failures (rate limits, 5xx) **and empty responses from reasoning models** are retried automatically with backoff — three attempts before the error surfaces; gpt-oss models run extraction at low reasoning effort by default

> **Ablation result:** Removing the shadow-selection mechanism collapses CRR from 100% to 37.9%, isolating the guarantee to a single identifiable component.

The guarantee covers what the extraction model captures as a constraint or decision. If the extractor misses or misclassifies something, DSPM cannot protect it (see Known Limitations).

---

## Results

Tested across 7 domains × 40 turns each:

| Budget | Tokens Used | TRR | CRR |
|--------|-------------|-----|-----|
| 150 | 150 | 66.8% | 100% |
| 250 | 249 | 82.8% | 100% |
| 400 | 395 | 72.4% | 100% |

`CRR` = Critical Retention Rate. `TRR` = Token Reduction Ratio.

**Provenance:** measured on 7 hand-authored 40-turn technical dialogues (API design, ML ops, IoT, security IR, supply chain, clinical workflow, project management); extraction and judging by GLM via OpenRouter (the since-renamed "ox-alpha" model). The same model family performed both extraction and judging, and the benchmark has not been independently replicated. Numbers vary by extraction model and domain — check your own with `memory.stats`.

---

## Known Limitations

- **Document QA:** DSPM is designed for conversations, where constraints and revisions matter. In head-to-head document question-answering tests it lost to a query-aware compressor (Compresr). Use it for dialogue memory, not for compressing papers or contracts.
- **Duplicate restatements:** when the same fact is restated by both user and assistant, duplicate criticals can compete for budget under very small budgets.
- **Extraction depends on the LLM:** quality varies by extraction model and domain, and a constraint the extractor misses or labels incorrectly is not protected.
- **Benchmark scope:** 7 hand-authored dialogues, extraction and judging by the same model family. Treat the numbers as a baseline, not a third-party evaluation.
- **Still early:** version 0.1.x, Beta. APIs may change.

---

## Document Mode: Dense Extraction

Fact-heavy content — research papers, specs, benchmark tables — needs a wider extraction funnel than conversation. DSPM detects it automatically: any turn whose numeric density (fraction of digit-bearing words) reaches 0.05 is extracted with the dense prompt — up to 10 patches per turn, with rules that keep each numeric result in its own patch with its unit and label verbatim, extract tables one patch per row, and stop experimental/config parameters from being misclassified as constraints.

```python
# automatic — this turn is fact-dense, so it routes through the dense funnel:
memory.add_turn("user", "p95 latency 200ms, fee 0.5%, timeout 30 seconds, budget 250 tokens")

# documents — sentence-aware chunking (never splits mid-sentence or mid-table-row),
# auto-dense per chunk:
n_chunks = memory.add_document(open("paper.txt", encoding="utf-8").read())

# or force it when you know the input is always fact-dense:
memory = DSPMMemory(budget=250, llm_client=llm, model="gpt-4o-mini", dense=True)
```

Live check (one results paragraph, `gpt-oss-20b` via Groq): **7/7 benchmark numbers captured** as `[EQ]` patches, zero misclassified constraints. Document mode improves extraction of numeric facts, but see Known Limitations for document QA.

---

## API Reference

### `DSPMMemory(budget, llm_client, model)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `budget` | int | 250 | Maximum tokens in the compressed output |
| `llm_client` | OpenAI | None | Any OpenAI-compatible client |
| `model` | str | `"gpt-4o-mini"` | Model used for patch extraction |
| `dense` | bool | `False` | Force the dense extraction funnel (up to 10 patches per turn) on every turn. Auto-dense also triggers per-turn on fact-heavy text regardless of this flag. |

The budget can be changed at any time:

```python
memory.budget = 100        # takes effect on the next get_context() call
memory.set_budget(100)     # explicit alternative — same effect
```

### Methods

| Method | Description |
|--------|-------------|
| `add_turn(role, text)` | Add a conversation turn. Returns extracted patches. |
| `add_document(text, chunk_tokens=400)` | Ingest a document: sentence-aware chunking (never splits mid-sentence or mid-table-row), auto-dense per chunk. Returns the chunk count. |
| `get_context(query="")` | Returns compressed context string, ready for your prompt. |
| `set_budget(budget)` | Update the token budget. Takes effect on the next `get_context()` call. |
| `save(path)` | Persist the memory notebook to JSON (atomic write). |
| `load(path)` | Load a notebook into memory (merge semantics, supersession on load). |
| `reset()` | Clear all memory and start fresh. |

### Properties

| Property | Description |
|----------|-------------|
| `memory.stats` | Dict with token counts, patch counts, active budget, CRR |
| `memory.critical_patches` | List of all critical patches currently in memory |
| `memory.all_patches` | List of every patch in memory |

---

## Paper

**DSPM: A Critical-Retention Approach to Long-Context Memory Compression for LLM Conversations**  
Dhruv Dubey, 2026  
Zenodo: [10.5281/zenodo.19438636](https://doi.org/10.5281/zenodo.19438636)

---

## Feedback

Found a case where a constraint or revision is lost, or a conversation DSPM handles badly? Please [open an issue](https://github.com/zatchbell1311-wq/Kernl/issues) with the conversation and the budget you used.

---

## Version History

| Version | Changes |
|---------|---------|
| 0.1.13 | **Value directives are decisions** (live conversation-mode finding): 'set the webhook timeout to 30 seconds' was classified as code — the v0.1.10 config-≠-constraint rule, written for document mode, over-applied to speaker directives, so the replaced value never entered the critical lineage and the revision history had nothing to bake. Both prompts now distinguish value DIRECTIVES ('set X to Y', 'change X to Y' — always decisions) from detached configuration DESCRIPTIONS (a document reporting its own settings — code/structure). New release rule: extraction-layer changes require BOTH live smoke gates (document + conversation) before shipping; adds `scripts/smoke_conversation.py`. 1 regression test (46 total). |
| 0.1.12 | **Revision stories survive supersession and starvation** (live webhook-rematch findings): the (was X) baking is history-aware — a stale patch that is itself a revision gives up its history when restated (`(was 30)`, not the wrong `(was 10)`) and chains full lineages (`(was 300 then 60)`); payloads carrying `(was X)` have a 3-word floor, and at that floor trimming returns the minimal story `10 (was 30)` — current value plus history — instead of a valueless `10 seconds`. 3 regression tests (45 total). |
| 0.1.11 | **Extractor hardening for reasoning models** (found in live document testing, 3 comparison rounds): extraction output budget raised 2000 → 4000 tokens and made overridable (`extract_turn(max_tokens=...)`); HTTP-200-but-empty responses — reasoning models burning the budget on chain-of-thought, which silently produced 0-patch chunks — are now retried with backoff; gpt-oss models automatically extract at low reasoning effort, with graceful fallback if the provider (400) or a custom client (TypeError) rejects the parameter. 6 regression tests (42 total). |
| 0.1.10 | **Auto-dense extraction:** fact-heavy turns (numeric density ≥ 0.05) automatically get the dense extraction funnel (up to 10 patches) — v0.1.9's density detector was never wired, so document runs silently used the 5-patch funnel and lost most numeric facts at extraction (live check: a results paragraph now captures 7/7 benchmark numbers, previously 2–3). Dense prompt hardened: experimental/config parameters are no longer misclassified as constraints; each numeric result gets its own patch; tables extract one patch per row. New `add_document()` — sentence-aware document ingestion. Recency in utility scoring now actually decays (was a constant). Corrupted notebook files return 0 on `load()` instead of raising. Fixed a vacuous budget test. 6 new tests (36 total). |
| 0.1.9 | Replaced values baked into superseded payloads as `(was X)` spans, atomic during trimming — the full revision story (current value + what it replaced) survives even at tiny budgets. Temporal values (weekdays/months/quarters) directly after key nouns survive trimming at priority 90. Dense-mode extraction for fact-heavy turns (manual `dense=True` flag). 3 new tests (30 total). |
| 0.1.8 | Retry wording corrected — 0.1.7's wheel shipped the previous wording, caught by post-build wheel verification check immediately after upload. No code changes. |
| 0.1.7 | README-first release (0.1.6's PyPI page was frozen pre-update — built after README finalization this time). Compound units ("per minute") and key constraint nouns ("timeout", "limit") survive trimming as atomic number-spans. "now"/"moved" recognized as revision verbs, and the extractor preserves revision verbs in payloads (real-model quickstart caught a stale-value contradiction). Claims scoped to benchmarks; retry behavior accurately described. Test scripts moved to `scripts/` with exposed API key revoked and stripped. `semantic` extra documented. Python 3.13 classifier; status → Beta. 3 new tests (27 total). |
| 0.1.6 | **Cross-type revision supersession fixed:** revisions sharing only 3 content words previously survived as contradictions (found in external review). Units now bind to their numbers during trimming (`30 seconds`, never bare `30`). Automatic retry with backoff on rate limits and transient 5xx errors in `add_turn()`. Homepage added to PyPI metadata. 5 regression tests (24 total). |
| 0.1.5 | **Budget fix:** changing `memory.budget` was silently ignored — the engine kept an independent budget copy. Now synced on every `get_context()`. New `set_budget()` method. `stats` now measures the actual joined context and reports the active budget. 2 regression tests (19 total). |
| 0.1.4 | Persistence: `memory.save()` / `memory.load()` — cross-session long-term memory as portable JSON. Atomic writes, merge-on-load, revisions supersede stale values. 8 new tests (17 total). |
| 0.1.3 | Revision supersession fix: stale same-type criticals now removed when superseded. Robust normalized content-word matching. |
| 0.1.2 | Fixed critical-patch ID collisions (CRR 36% → 100% in 18-turn live test). Budget enforced on joined context string. |
| 0.1.1 | Fixed T4 dropping criticals with dependencies. Fixed T3 payload mangling. Fixed extractor schema mismatch. |
| 0.1.0 | Initial release. |

---

## License

MIT © 2026 Dhruv Dubey
