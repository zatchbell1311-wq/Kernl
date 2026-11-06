# DSPM — memory for LLM conversations that keeps your constraints

Dynamic Semantic Patch Memory, by Dhruv Dubey · Python package: `dspm-memory` · Repo: [Kernl](https://github.com/zatchbell1311-wq/Kernl)

[![PyPI version](https://img.shields.io/pypi/v/dspm-memory)](https://pypi.org/project/dspm-memory/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![CI](https://github.com/zatchbell1311-wq/Kernl/actions/workflows/ci.yml/badge.svg)](https://github.com/zatchbell1311-wq/Kernl/actions/workflows/ci.yml)

```bash
pip install dspm-memory                  # lightweight (no torch)
pip install dspm-memory[embeddings]      # full semantic similarity (CPU-only)
```

Long chats drop the rule from turn 3 that everything else depends on. DSPM turns each turn into typed patches, protects constraints, decisions and revised values, and renders them under a fixed token budget. Training-free: no fine-tuning, just prompts and Python rules. Works with any OpenAI-compatible endpoint.

## Headline numbers (v0.2.4, verified on live data)

24-turn fintech conversation, 648 raw tokens, 3 revisions tracked (JWT 15→30min, S3 1h→15min, CPU 70%→60%), real Groq LLM:

| Budget | CPI | TRR | Context tokens |
|--------|-----|-----|----------------|
| 60 | **100%** | 92.3% | 50 |
| 100 | **100%** | 89.4% | 69 |
| 250 | **100%** | 59.0% | 266 |
| 400 | **100%** | 36.4% | 412 |
| 800 | **100%** | -2.0% | 661 |

CPI = Critical Patch Inclusion. TRR = Token Reduction Rate (conversation-based).
**CPI = 100% is a structural guarantee**, verified across budgets 59 to 5000 tokens.

## Real-world verification

**96-turn healthcare dialogue on local Gemma (27B)**

- Tested by **[Piotr Chlebek](https://github.com/pcbua)** (systems architect, patent holder in healthcare dialogue processing)
- Setup: vLLM serving Gemma (27B), 32 concurrent threads
- vLLM Throughput: 2,971 tok/s prompt, 80.5% prefix cache hit rate
- Result: **CPI = 100%**, ~6K raw tokens compressed to 1024 tokens
- All load-bearing medical facts preserved verbatim:
  - 1.7 kg weight gain (83.4 → 85.1 kg) flagged for nurse
  - Echocardiogram scheduled Wednesday 10:30, 40 min
  - Blood sugar 9.1 (Friday), 10.2 (Thursday)
  - Cataract surgery day procedure, requires accompaniment
  - Eye clinic requires cardiologist note
  - Diuretic causes nocturia, move to morning
- See `examples/gemma_healthcare/` for reproduction

**First external verification of CPI = 100% on a non-Groq model.**

Offline benchmark (v0.1.17, 7 hand-authored 40-turn dialogues, same model for extraction and judging):

| Budget | TRR | CPI |
|--------|-----|-----|
| 150 | 66.8% | 100% |
| 250 | 82.8% | 100% |
| 400 | 72.4% | 100% |

> [!NOTE]
> **Recognition:** Poster accepted at CAICS 2026 (Indo-US Workshop on AI & Cybersecurity), organized by IISc Bengaluru and Argonne National Laboratory, funded by the U.S. Department of State.

**Savings depend on conversation length.** Short conversations compress very little. Recommended budget: 250–400 for best readability + compression balance.

**Who it's for:** agents and chatbots whose long conversations must not lose a constraint, a decision, or a revised value (support bots, coding agents, assistants that remember a user's hard rules across sessions).

**Who it's not for:** summarizing or answering questions over documents. See [Known Limitations](#known-limitations).

## Quickstart (Sync)

```python
from openai import OpenAI
from dspm import DSPMMemory

llm = OpenAI(api_key="sk-...")   # any OpenAI-compatible endpoint
mem = DSPMMemory(budget=250, llm_client=llm, model="gpt-4o-mini")

mem.add_turn("user", "Always return JSON. Use POST /v1/items. Max fee 0.5%.")
mem.add_turn("assistant", "Webhook timeout set to 30 seconds.")

# Later: the webhook timeout is revised
mem.add_turn("user", "Webhook timeout updated to 10 seconds.")

print(mem.get_context(query="What must the contract remember?"))
print(mem.stats)
```

Illustrative output (exact wording varies by extraction model):

```text
[DEC] Webhook timeout updated to 10 seconds
[CON] Max fee 0.5%
[DEC] Use POST /v1/items
[CON] Always return JSON output.
```

## Quickstart (Async — v0.3.0)

Process multiple dialogues concurrently using `AsyncOpenAI`:

```python
import asyncio
from openai import AsyncOpenAI
from dspm import AsyncDSPMMemory

async def process_dialogue(llm, dialogue_text):
    mem = AsyncDSPMMemory(budget=400, llm_client=llm, model="gpt-4o-mini")
    for role, text in dialogue_text:
        await mem.async_add_turn(role, text)
    return await mem.async_get_context(query="Summarize key decisions and constraints")

async def main():
    llm = AsyncOpenAI(api_key="sk-...")
    # Run N dialogues in parallel!
    results = await asyncio.gather(
        *[process_dialogue(llm, d) for d in dialogues]
    )
```

### No paid key? Use a free Groq tier or a local Ollama model

```python
# Groq (free tier available at console.groq.com)
llm = OpenAI(base_url="https://api.groq.com/openai/v1", api_key="gsk_...")

# Ollama (local, no key needed)
llm = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# Together AI
llm = OpenAI(base_url="https://api.together.xyz/v1", api_key="...")

# Small-context local model (e.g., Gemma via vLLM with 2k-4k context)
mem = DSPMMemory(
    budget="auto",               # v0.2.5: automatically calculate budget
    ctx_window=2048,             # v0.2.5: specify your model's context window
    llm_client=llm,
    model="gemma3:27b",
    max_extraction_tokens=800,   # leave room for input prompt
)
```

## Why it's different

- **Constraints and decisions are protected.** `[CON]` and `[DEC]` patches are never dropped by dedup, fusion or pruning, and are selected first at render time.
- **Revisions supersede old values.** Change a value mid-chat and the current one survives; the old value is dropped from the active memory.
- **Memory persists across chats.** Save at the end of one chat, load at the start of the next, with no API cost for recall.
- **Fixed budget, predictable cost.** You set the token budget; the rendered context stays inside it.
- **Universal CPI guarantee.** CPI = 100% at any budget above the floor. Below the floor, raises `BudgetTooSmallError` instead of silently dropping criticals.
- **Fail-loud extraction.** If the LLM backend fails, `add_turn()` raises `ExtractionError` instead of silently losing the message (v0.2.3+).
- **True Async Architecture.** `AsyncDSPMMemory` provides non-blocking I/O for batch processing (v0.3.0+).
- **Semantic Recall (SR) Metric.** An honest evaluation metric that measures actual meaning retention, not just patch count (v0.3.0+).

## Universal CPI guarantee

```python
from dspm.engine import DSPMEngine, BudgetTooSmallError

# Find the minimum safe budget for N critical patches
floor = DSPMEngine.min_budget_for_crr(22)   # → 59 tokens for 22 criticals

# Below the floor, raises a clear error instead of silently dropping
try:
    mem.set_budget(10)
    mem.get_context()
except BudgetTooSmallError as e:
    print(e)  # Budget 10 is below the absolute floor (59 tokens for 22 criticals)
```

### 5-stage compression ladder

Escalates only as needed:

| Stage | When triggered | What it does |
|-------|---------------|--------------|
| 1. Numeric-trim | Always | Keeps numbers, units, acronyms; drops filler words |
| 2. Iterative-trim | Budget pressure | Trims the longest critical one word at a time |
| 3. Ultra-compact | Tight budget | Reduces each critical to key numbers only |
| 4. Dense-merge | Very tight | Merges all criticals into one line |
| 5. Ultra-dense | Extreme | Single line of space-separated tokens |

## Persistence: memory across chats and sessions

```python
# Chat 1 — Monday
mem = DSPMMemory(budget=250, llm_client=llm, model="gpt-4o-mini")
mem.add_turn("user", "Building a budgeting app. Hard rules: must work offline, deadline Oct 20.")
mem.save("user_42_memory.json")

# Chat 2 — Thursday, new process
mem2 = DSPMMemory(budget=250, llm_client=llm, model="gpt-4o-mini")
mem2.load("user_42_memory.json")
mem2.get_context(query="What were my hard rules?")
# → [CON] must work offline
# → [CON] deadline Oct 20   ← recalled from Chat 1, zero API cost

mem2.add_turn("user", "The deadline moved to November 5.")
# → Oct 20 is superseded. November 5 replaces it.
```

Persistence safety: writes are atomic, and `load(path, strict=True)` raises `CorruptNotebookError` on corrupt files.

## How it works

```text
Raw conversation (648 tokens, 24 turns)
        ↓
Semantic extraction → 39 patches, 20 critical
        ↓
7-stage compression pipeline
        ↓
Compressed context (264 tokens @ budget 250) — 100% of criticals selected
```

**Pipeline:** fingerprint dedup → slot fusion → delta encoding → causal pruning → utility scoring → shadow selection → adaptive budget allocation.

## Patch types

| Type | Tag | Critical? | Description |
|------|-----|-----------|-------------|
| constraint | `[CON]` | Yes | Non-negotiable requirements (deadlines, limits, compliance rules) |
| decision | `[DEC]` | Yes | Speaker value directives ("set timeout to 30 seconds") |
| code | `[CODE]` | No | Implementation detail |
| equation | `[EQ]` | No | Formulas and numeric results |
| entity | `[ENT]` | No | Named things |
| structure | `[STR]` | No | Schemas and workflows |

Regulatory and compliance requirements (PCI-DSS, HIPAA, GDPR, SOC 2, ISO 27001, FedRAMP) are classified as constraints by the extractor prompt.

## Stats and metrics

```python
print(mem.stats)
```

| Key | Description |
|-----|-------------|
| `turns` | Number of conversation turns added |
| `total_patches` | Active patches in the store |
| `critical_total` | Constraint + decision patches (protected) |
| `critical_selected` | Critical patches delivered in the last `get_context()` |
| `crr` | Critical Patch Inclusion: % of extracted criticals delivered |
| `trr_by_conversation` | Token Reduction Rate, conversation-based. **Use this for savings** |
| `budget` | Current token budget |
| `conversation_tokens` | Raw turn text fed to `add_turn()` |
| `patch_tokens` | Tokens in the extracted patches |
| `context_tokens` | Tokens in the delivered context |
| `llm_calls` | Total LLM calls made |
| `llm_errors` | Total LLM errors |

## Testing

### Offline tests (no API key needed)

```bash
python tests/test_universal_budgets.py
python tests/test_regression.py       # v0.2.4: fingerprint + slot key regression tests
```

### Live tests (API key required)

```bash
# Windows PowerShell
$env:GROQ_API_KEY = "gsk_..."

python tests/verify_crr.py                # quick 10-turn CPI check
python tests/small_live_test_budgets.py   # 10-turn REST API conversation
python tests/large_live_test_budgets.py   # 24-turn fintech + 3 revisions
python tests/test_async.py                # v0.3.0: 3 dialogues concurrently
python tests/test_batch.py                # v0.3.0: 2-pass parallel extraction
```

See `tests/README.md` for expected output and reproduction instructions.

## API reference

### `DSPMMemory(budget, llm_client, model, ...)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `budget` | int / "auto" | 400 | Maximum tokens in the compressed output. `budget="auto"` calculates automatically (v0.2.5+). |
| `llm_client` | OpenAI-compatible | None | Any sync OpenAI-compatible client |
| `model` | str | `"gpt-4o-mini"` | Model used for patch extraction |
| `max_extraction_tokens` | int | 2000 | Max output tokens for extractor LLM. **Reduce for small-context local models** (v0.2.2+). |
| `ctx_window` | int | 8192 | Context window for `budget="auto"` calculation (v0.2.5+). |
| `target_sr` | float | None | If set, enables SR-aware budget selection (v0.3.0+). |

### Methods

| Method | Description |
|--------|-------------|
| `add_turn(role, text)` | Add a conversation turn. Returns extracted patches. Raises `ExtractionError` on failure. |
| `get_context(query="", filter=None)` | Returns compressed context string. Supports structured `filter` dict (v0.3.0+). |
| `set_budget(budget)` | Update the token budget. |
| `save(path)` | Persist memory to JSON (atomic write). |
| `load(path, strict=False)` | Load a saved memory file. |
| `reset()` | Clear all memory. |

### Async Methods (v0.3.0+)

Use `AsyncDSPMMemory` with `AsyncOpenAI` for concurrent processing.

| Method | Description |
|--------|-------------|
| `await async_add_turn(role, text)` | Async version of `add_turn`. |
| `await async_add_turns_batch(turns)` | 2-pass parallel extraction for long dialogues. |
| `await async_get_context(query="")` | Async version of `get_context`. |

### Properties

| Property | Description |
|----------|-------------|
| `memory.stats` | Dict with token counts, patch counts, CPI, TRR and LLM telemetry |
| `memory.critical_patches` | List of critical patches in memory |
| `memory.all_patches` | List of every patch in memory |
| `memory.error_log` | Per-instance list of extraction errors (v0.2.3+). |
| `memory.last_sr` | Last Semantic Recall score (if SR-aware mode is active, v0.3.0+). |

### Advanced API

```python
from dspm import DSPMEngine, extract_patches, merge_into_memory, SemanticPatch, ExtractionError
from dspm.engine import BudgetTooSmallError
```

## Repository layout

```text
.
├── dspm/
│   ├── __init__.py        # public API
│   ├── patch.py           # SemanticPatch structure
│   ├── engine.py          # DSPMEngine — 7-stage pipeline + CPI guarantee
│   ├── extractor.py       # LLM-based patch extraction
│   ├── async_memory.py    # AsyncDSPMMemory (v0.3.0+)
│   ├── sr_budget.py       # SR-aware budget selection (v0.3.0+)
│   ├── client.py          # LLMClient + provider auto-detection
│   └── memory.py          # DSPMMemory high-level wrapper
├── eval/
│   └── semantic_recall.py # SR metric evaluation script (v0.3.0+)
├── tests/
│   ├── verify_crr.py
│   ├── test_universal_budgets.py
│   ├── test_regression.py
│   ├── small_live_test_budgets.py
│   ├── large_live_test_budgets.py
│   ├── test_async.py               # v0.3.0
│   ├── test_batch.py               # v0.3.0
│   └── README.md
├── examples/
│   └── gemma_healthcare/           # Piotr Chlebek (github.com/pcbua) — 96-turn test
├── pyproject.toml
├── README.md
└── LICENSE
```

## Known Limitations

- **Small budgets under-deliver readability.** At budget 60, context is trimmed to bare numbers. CPI = 100% but human readability suffers. Recommended minimum for readable output: budget 100.
- **Per-turn extraction cap.** The extraction prompt allows at most 1 constraint and 1 decision per turn. If a single message states several hard rules, split them across turns.
- **Revision matching is rule-based.** Numeric, day-name and acronym-bridged revisions are handled. Other free-text revisions may coexist with stale values.
- **Extraction depends on the LLM.** Quality varies by model and domain. A constraint the extractor misses is not protected.
- **CPI measures patch count, not meaning.** At very tight budgets, all critical patches are present but their payloads are compressed to bare numbers. The Semantic Recall (SR) metric (v0.3.0+) was added to measure actual fact recovery honestly.
- **Budget may exceed by a few tokens when all patches are criticals.** The v0.2.4 rendered-string trim only drops non-criticals. If criticals alone exceed the budget, `_fit_criticals` compresses payloads but may still be slightly over. Fix planned for v0.4.0.
- **Document QA.** DSPM is designed for conversations, not document compression.
- **Self-judged offline benchmark.** Same model family for extraction and judging; not independently replicated. Run live tests with your own data using `tests/large_live_test_budgets.py`.

## Paper

**DSPM: A Critical-Retention Approach to Long-Context Memory Compression for LLM Conversations**
Dhruv Dubey, 2026 · Zenodo: [10.5281/zenodo.19438636](https://doi.org/10.5281/zenodo.19438636)

> [!NOTE]
> **Recognition:** Poster accepted at CAICS 2026 (Indo-US Workshop on AI & Cybersecurity), jointly organized by IISc Bengaluru and Argonne National Laboratory, funded by the U.S. Department of State.

## Feedback

Found a case where a constraint or revision is lost? Open an issue with the conversation and the budget you used.

## License

MIT © 2026 Dhruv Dubey
