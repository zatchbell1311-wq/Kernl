# DSPM — memory for LLM conversations that keeps your constraints
**Dynamic Semantic Patch Memory**, by Dhruv Dubey · Python package: `dspm-memory` · Repo: Kernl

[![PyPI version](https://img.shields.io/pypi/v/dspm-memory)](https://pypi.org/project/dspm-memory/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)

```bash
pip install dspm-memory
```

Long chats drop the rule from turn 3 that everything else depends on. DSPM turns each turn into typed patches, protects constraints, decisions and revised values, and renders them under a fixed token budget. Training-free: no fine-tuning, just prompts and Python rules. Works with any OpenAI-compatible endpoint.

## Headline numbers (v0.2.2, verified on live data)

24-turn fintech conversation, 648 raw tokens, 3 revisions tracked (JWT 15→30min, S3 1h→15min, CPU 70%→60%), real Groq LLM:

| Budget | CRR | TRR | Context tokens |
|--------|-----|-----|----------------|
| 60 | 100% | 93.1% | 45 |
| 100 | 100% | 81.9% | 117 |
| 250 | 100% | 59.3% | 264 |
| 400 | 100% | 36.1% | 414 |
| 800 | 100% | 0.6% | 644 |

> **CRR = Critical Retention Rate. TRR = Token Reduction Rate (conversation-based).**
> CRR = 100% is a structural guarantee, verified across budgets 59 to 5000 tokens.

## Real-world verification

### 96-turn healthcare dialogue on local Gemma (27B)

> Tested by Piotr Chlebek (systems architect, patent holder in healthcare dialogue processing)
> Setup: vLLM serving Gemma (27B), 8k context window
> Result: CRR = 100%, ~6K raw tokens compressed to 1024 tokens

All load-bearing medical facts preserved verbatim:
- 1.7 kg weight gain (83.4 → 85.1 kg) flagged for nurse
- Echocardiogram scheduled Wednesday 10:30, 40 min
- Blood sugar 9.1 (Friday), 10.2 (Thursday)
- Cataract surgery day procedure, requires accompaniment
- Eye clinic requires cardiologist note
- Diuretic causes nocturia, move to morning

See `examples/gemma_healthcare/` for reproduction

> First external verification of CRR = 100% on a non-Groq model.

Offline benchmark (v0.1.17, 7 hand-authored 40-turn dialogues, same model for extraction and judging):

| Budget | TRR | CRR |
|--------|-----|-----|
| 150 | 66.8% | 100% |
| 250 | 82.8% | 100% |
| 400 | 72.4% | 100% |

> **Recognition:** Poster accepted at CAICS 2026 (Indo-US Workshop on AI & Cybersecurity), organized by IISc Bengaluru and Argonne National Laboratory, funded by the U.S. Department of State.

> Savings depend on conversation length. Short conversations compress very little. Recommended budget: 250–400 for best readability + compression balance.

> **Who it's for:** agents and chatbots whose long conversations must not lose a constraint, a decision, or a revised value (support bots, coding agents, assistants that remember a user's hard rules across sessions).
>
> **Who it's not for:** summarizing or answering questions over documents. See [Known Limitations](#known-limitations).

## Quickstart

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
[DEC] Webhook timeout updated to 10 seconds (was 30)
[CON] Max fee 0.5%
[DEC] Use POST /v1/items
[CON] Always return JSON output.
```

No paid key? Use a free Groq tier or a local Ollama model:

```python
# Groq (free tier available at console.groq.com)
llm = OpenAI(base_url="https://api.groq.com/openai/v1", api_key="gsk_...")

# Ollama (local, no key needed)
llm = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# Together AI
llm = OpenAI(base_url="https://api.together.xyz/v1", api_key="...")

# Small-context local model (e.g., Gemma via vLLM with 2k-4k context)
mem = DSPMMemory(
    budget=250,
    llm_client=llm,
    model="gemma3:27b",
    max_extraction_tokens=800,  # leave room for input prompt
)
```

## Why it's different

- **Constraints and decisions are protected.** `[CON]` and `[DEC]` patches are never dropped by dedup, fusion or pruning, and are selected first at render time.
- **Revisions supersede old values.** Change a value mid-chat and the current one survives, with the lineage baked in as `(was X)`.
- **Memory persists across chats.** Save at the end of one chat, load at the start of the next, with no API cost for recall.
- **Fixed budget, predictable cost.** You set the token budget; the rendered context stays inside it.
- **Universal CRR guarantee.** CRR = 100% at any budget above the floor. Below the floor, raises `BudgetTooSmallError` instead of silently dropping criticals.

## Universal CRR guarantee

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

5-stage compression ladder, which escalates only as needed:

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

> **Persistence safety:** writes are atomic, and `load(path, strict=True)` raises `CorruptNotebookError` on corrupt files.

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

Pipeline: fingerprint dedup → slot fusion → delta encoding → causal pruning → utility scoring → shadow selection → adaptive budget allocation.

## Patch types

| Type | Tag | Critical? | Description |
|------|-----|-----------|-------------|
| constraint | `[CON]` | Yes | Non-negotiable requirements (deadlines, limits, compliance rules) |
| decision | `[DEC]` | Yes | Speaker value directives ("set timeout to 30 seconds") |
| code | `[CODE]` | No | Implementation detail |
| equation | `[EQ]` | No | Formulas and numeric results |
| entity | `[ENT]` | No | Named things |
| structure | `[STR]` | No | Schemas and workflows |

> Regulatory and compliance requirements (PCI-DSS, HIPAA, GDPR, SOC 2, ISO 27001, FedRAMP) are classified as constraints by the extractor prompt.

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
| `crr` | Critical Retention Rate: % of extracted criticals delivered |
| `trr_by_conversation` | Token Reduction Rate, conversation-based. Use this for savings |
| `budget` | Current token budget |
| `conversation_tokens` | Raw turn text fed to `add_turn()` |
| `patch_tokens` | Tokens in the extracted patches |
| `context_tokens` | Tokens in the delivered context |
| `llm_calls` | Total LLM calls made |
| `llm_errors` | Total LLM errors |

## Testing

Run offline tests (no API key needed):

```bash
python tests/verify_crr.py
python tests/test_universal_budgets.py
```

Run live tests (API key required):

```bash
# Windows PowerShell
$env:GROQ_API_KEY = "gsk_..."

python tests/small_live_test_budgets.py   # 10-turn REST API conversation
python tests/large_live_test_budgets.py   # 24-turn fintech + 3 revisions
```

See `tests/README.md` for expected output and reproduction instructions.

## API reference

### `DSPMMemory(budget, llm_client, model, max_extraction_tokens)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `budget` | int | 400 | Maximum tokens in the compressed output |
| `llm_client` | OpenAI-compatible | None | Any sync OpenAI-compatible client |
| `model` | str | `"gpt-4o-mini"` | Model used for patch extraction |
| `max_extraction_tokens` | int | 2000 | Max output tokens for extractor LLM. Reduce for small-context local models (e.g., 800 for 2k-4k context models). Added in v0.2.2. |

### Methods

| Method | Description |
|--------|-------------|
| `add_turn(role, text)` | Add a conversation turn. Returns extracted patches. |
| `get_context(query="")` | Returns compressed context string. |
| `set_budget(budget)` | Update the token budget. |
| `save(path)` | Persist memory to JSON (atomic write). |
| `load(path, strict=False)` | Load a saved memory file. |
| `reset()` | Clear all memory. |

### Properties

| Property | Description |
|----------|-------------|
| `memory.stats` | Dict with token counts, patch counts, CRR, TRR and LLM telemetry |
| `memory.critical_patches` | List of critical patches in memory |
| `memory.all_patches` | List of every patch in memory |

## Advanced API

```python
from dspm import DSPMEngine, extract_patches, merge_into_memory, SemanticPatch
from dspm.engine import BudgetTooSmallError
```

## Repository layout

```text
.
├── dspm/
│   ├── __init__.py      # public API
│   ├── patch.py         # SemanticPatch structure
│   ├── engine.py        # DSPMEngine v4.5 — 7-stage pipeline + CRR guarantee
│   ├── extractor.py     # LLM-based patch extraction
│   ├── client.py        # LLMClient + provider auto-detection
│   └── memory.py        # DSPMMemory high-level wrapper
├── tests/
│   ├── verify_crr.py              # quick CRR check
│   ├── test_universal_budgets.py  # offline sweep, 17 budgets
│   ├── small_live_test_budgets.py # 10-turn live test
│   ├── large_live_test_budgets.py # 24-turn live test with 3 revisions
│   └── README.md                  # reproduction instructions
├── examples/
│   └── gemma_healthcare/          # Piotr Chlebek contribution (96-turn test)
├── pyproject.toml
├── README.md
└── LICENSE
```

## Known Limitations

- **Small budgets under-deliver readability.** At budget 60, context is trimmed to bare numbers. CRR = 100% but human readability suffers. Recommended minimum for readable output: budget 100.
- **Per-turn extraction cap.** The extraction prompt allows at most 1 constraint and 1 decision per turn. If a single message states several hard rules, split them across turns.
- **Revision matching is rule-based.** Numeric, day-name and acronym-bridged revisions are handled. Other free-text revisions may coexist with stale values.
- **Extraction depends on the LLM.** Quality varies by model and domain. A constraint the extractor misses is not protected.
- **No async support yet.** `add_turn()` is synchronous. Planned for v0.3.0 — driven by real-world feedback from Piotr Chlebek (vLLM batching use case).
- **Document QA.** DSPM is designed for conversations, not document compression.
- **Self-judged offline benchmark.** Same model family for extraction and judging; not independently replicated. Run live tests with your own data using `tests/large_live_test_budgets.py`.

## Paper

**DSPM: A Critical-Retention Approach to Long-Context Memory Compression for LLM Conversations**
Dhruv Dubey, 2026 · Zenodo: [10.5281/zenodo.19438636](https://doi.org/10.5281/zenodo.19438636)

> **Recognition:** Poster accepted at CAICS 2026 (Indo-US Workshop on AI & Cybersecurity), jointly organized by IISc Bengaluru and Argonne National Laboratory, funded by the U.S. Department of State.

## Feedback

Found a case where a constraint or revision is lost? Open an issue with the conversation and the budget you used.

## License

MIT © 2026 Dhruv Dubey
