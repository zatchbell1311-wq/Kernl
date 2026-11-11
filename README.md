<div align="center">

# DSPM — Dynamic Semantic Patch Memory

**Memory for LLM conversations that keeps your constraints**

*by Dhruv Dubey · Python package: `dspm-memory` · Repo: Kernl*

[![PyPI version](https://img.shields.io/pypi/v/dspm-memory)](https://pypi.org/project/dspm-memory/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)

</div>

```bash
pip install dspm-memory
```

Long chats drop the rule from turn 3 that everything else depends on. DSPM turns each turn into typed patches, protects constraints, decisions and revised values, and renders them under a fixed token budget. Training-free: no fine-tuning, just prompts and Python rules. Works with any OpenAI-compatible endpoint.

---

## What's new in v0.5.3

- `[SLOT]` patch type for slot-value pairs (`restaurant.area = centre`, `train.arriveBy = 21:15`)
- Two-tier extraction: `[CON]`/`[DEC]` are critical (max 1 per turn, merge aggressively), `[SLOT]` is non-critical but gets 30% budget allocation
- Budget fill fix: engine now uses the total budget instead of per-type caps
- Improved extraction prompt with domain-specific examples (attraction, restaurant, hotel, train, taxi slots)
- Lower `SHADOW_THRESHOLD` (0.05 → 0.01) to keep more patch candidates

---

## Benchmark results (v0.5.3)

### MultiWOZ 2.2 — Multi-turn task-oriented dialogue

10 dialogues, `budget=400`, model: `qwen/qwen3.8-27b` via Groq, compared at equal token budget:

| Method | SR | CPI | Tokens used | Compression |
|---|---|---|---|---|
| DSPM v0.5.3 (all_patches) | 89.5% | 0.73 | 116 | 1.6× |
| DSPM v0.5.3 (compressed) | 84.2% | 0.71 | 109 | 1.6× |
| last-k turns (baseline) | 89.5% | 0.74 | 161 | 1.0× |
| LLM summary (baseline) | 84.2% | 0.72 | 237 | 0.7× |

**DSPM matches last-k on recall (89.5% SR) while using 28% fewer tokens.**

> **SR** = Semantic Recall (judge LLM answers constraint-checking questions using compressed memory).  
> **CPI** = Critical Patch Inclusion (ground-truth constraints visible in compressed output).

---

### Fintech use case — Constraint-heavy conversation

24-turn fintech conversation, 446 raw tokens, 3 revisions (JWT 15→30min, S3 1h→15min, CPU 70%→60%):

| Budget | CRR | Readability | Context tokens |
|---|---|---|---|
| 60 | 100% | 19% (ultra-dense) | 64 |
| 100 | 100% | 31% | 107 |
| 250 | 100% | 63% | 245 |
| 400 | 100% | 100% | 402 |

> **CRR** = Critical Retention Rate (all critical patches selected — structural guarantee).  
> **Readability** = constraint values appear verbatim in output (depends on budget).

**CRR = 100% at every budget.** At budget 400, all 17 constraint values are fully readable.

---

### Real-world verification

**96-turn healthcare dialogue on local Gemma (27B)**

- Tested by **Piotr Chlebek** (systems architect, patent holder in healthcare dialogue processing)
- Setup: vLLM serving Gemma (27B), 8k context window
- Result: CRR = 100%, ~6K raw tokens compressed to 1024 tokens
- All load-bearing medical facts preserved verbatim
- First external verification of CRR = 100% on a non-Groq model

> 🏆 **Recognition:** Poster accepted at **CAICS 2026** (Indo-US Workshop on AI & Cybersecurity), organized by IISc Bengaluru and Argonne National Laboratory, funded by the U.S. Department of State.

---

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

```
[DEC] Webhook timeout updated to 10 seconds
[CON] Max fee 0.5%
[DEC] Use POST /v1/items
[CON] Always return JSON output.
```

### No paid key? Use a free Groq tier or a local Ollama model

```python
# Groq (free tier available at console.groq.com)
llm = OpenAI(base_url="https://api.groq.com/openai/v1", api_key="gsk_...")

# Ollama (local, no key needed)
llm = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# Small-context local model (e.g., Gemma via vLLM with 2k-4k context)
mem = DSPMMemory(
    budget=250,
    llm_client=llm,
    model="gemma3:27b",
    max_extraction_tokens=800,  # leave room for input prompt
)
```

---

## Why it's different

| Feature | Description |
|---|---|
| **Constraints and decisions are protected** | `[CON]` and `[DEC]` patches are never dropped by dedup, fusion or pruning, and are selected first at render time. |
| **Slot-value pairs captured** | `[SLOT]` patches extract every `service.slot = value` pair the user mentions — critical for task-oriented dialogues like MultiWOZ. |
| **Revisions supersede old values** | Change a value mid-chat and the current one survives; the old value is dropped from the active memory. |
| **Memory persists across chats** | Save at the end of one chat, load at the start of the next, with no API cost for recall. |
| **Fixed budget, predictable cost** | You set the token budget; the rendered context stays inside it. |
| **Universal CRR guarantee** | CRR = 100% at any budget above the floor. Below the floor, raises `BudgetTooSmallError` instead of silently dropping criticals. |
| **Fail-loud extraction** | If the LLM backend fails, `add_turn()` raises `ExtractionError` instead of silently losing the message. |

---

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

---

## 5-stage compression ladder

Escalates only as needed:

| Stage | When triggered | What it does |
|---|---|---|
| 1. Numeric-trim | Always | Keeps numbers, units, acronyms; drops filler words |
| 2. Iterative-trim | Budget pressure | Trims the longest critical one word at a time |
| 3. Ultra-compact | Tight budget | Reduces each critical to key numbers only |
| 4. Dense-merge | Very tight | Merges all criticals into one line |
| 5. Ultra-dense | Extreme | Single line of space-separated tokens |

---

## Two-tier extraction (v0.5.3)

The extraction prompt uses a two-tier system:

**Tier 1 — Critical** (max 1 per turn, merge aggressively):

- `[CON]` — the most important hard requirement from this turn
- `[DEC]` — the most important decision or revision from this turn

**Tier 2 — Non-critical** (extract ALL that apply):

- `[SLOT]` — every slot-value pair (`service.slot = value`)
- `[CODE]` — technology stack, libraries, APIs
- `[EQ]` — formulas, thresholds, numerical specs
- `[ENT]` — named systems, tools, organisations
- `[STR]` — schemas, data formats, workflow steps

This keeps the critical patch count low (~8–10 per conversation) while capturing all slot-value pairs for task-oriented dialogues.

---

## Patch types

| Type | Tag | Critical? | Description |
|---|---|---|---|
| constraint | `[CON]` | ✅ | Non-negotiable requirements (deadlines, limits, compliance rules) |
| decision | `[DEC]` | ✅ | Speaker value directives ("set timeout to 30 seconds") |
| slot | `[SLOT]` | ❌ | Slot-value pairs (`restaurant.area = centre`, `train.arriveBy = 21:15`) |
| code | `[CODE]` | ❌ | Implementation detail |
| equation | `[EQ]` | ❌ | Formulas and numeric results |
| entity | `[ENT]` | ❌ | Named things |
| structure | `[STR]` | ❌ | Schemas and workflows |

> Regulatory and compliance requirements (PCI-DSS, HIPAA, GDPR, SOC 2, ISO 27001, FedRAMP) are classified as constraints by the extractor prompt.

---

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

---

## How it works

```
Raw conversation (446 tokens, 24 turns)
        ↓
Two-tier semantic extraction → 29 patches, 9 critical
        ↓
7-stage compression pipeline (fingerprint dedup → slot fusion →
delta encoding → causal pruning → utility scoring → shadow
selection → adaptive budget allocation)
        ↓
Compressed context (402 tokens @ budget 400) — 100% of criticals selected
```

Pipeline: `fingerprint dedup → slot fusion → delta encoding → causal pruning → utility scoring → shadow selection → adaptive budget allocation`

---

## Stats and metrics

```python
print(mem.stats)
```

| Key | Description |
|---|---|
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

---

## Testing

### Offline tests (no API key needed)

```bash
python tests/test_universal_budgets.py
python tests/test_regression.py       # fingerprint + slot key regression tests
```

### Live tests (API key required)

```bash
# Windows PowerShell
$env:GROQ_API_KEY = "gsk_..."

python tests/verify_crr.py                # quick 10-turn CRR check
python tests/small_live_test_budgets.py   # 10-turn REST API conversation
python tests/large_live_test_budgets.py   # 24-turn fintech + 3 revisions
```

See [`tests/README.md`](tests/README.md) for expected output and reproduction instructions.

---

## API reference

### `DSPMMemory(budget, llm_client, model, max_extraction_tokens)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `budget` | `int` | `400` | Maximum tokens in the compressed output |
| `llm_client` | OpenAI-compatible | `None` | Any sync OpenAI-compatible client |
| `model` | `str` | `"gpt-4o-mini"` | Model used for patch extraction |
| `max_extraction_tokens` | `int` | `2000` | Max output tokens for extractor LLM. Reduce for small-context local models (e.g., `800` for 2k-context models). |

### Methods

| Method | Description |
|---|---|
| `add_turn(role, text)` | Add a conversation turn. Returns extracted patches. Raises `ExtractionError` on failure. |
| `get_context(query="")` | Returns compressed context string. |
| `set_budget(budget)` | Update the token budget. |
| `save(path)` | Persist memory to JSON (atomic write). |
| `load(path, strict=False)` | Load a saved memory file. |
| `reset()` | Clear all memory. |

### Properties

| Property | Description |
|---|---|
| `memory.stats` | Dict with token counts, patch counts, CRR, TRR and LLM telemetry |
| `memory.critical_patches` | List of critical patches in memory |
| `memory.all_patches` | List of every patch in memory |
| `memory.error_log` | Per-instance list of extraction errors |

### Advanced API

```python
from dspm import DSPMEngine, extract_patches, merge_into_memory, SemanticPatch, ExtractionError
from dspm.engine import BudgetTooSmallError
```

---

## Repository layout

```
.
├── dspm/
│   ├── __init__.py      # public API (v0.5.3)
│   ├── patch.py         # SemanticPatch structure + SHORT_TAGS (7 types)
│   ├── engine.py        # DSPMEngine — 7-stage pipeline + CRR guarantee
│   ├── extractor.py     # Two-tier LLM-based patch extraction
│   ├── client.py        # LLMClient + provider auto-detection
│   └── memory.py        # DSPMMemory high-level wrapper
├── tests/
│   ├── verify_crr.py              # quick CRR check
│   ├── test_universal_budgets.py  # offline sweep, 17 budgets
│   ├── test_regression.py         # fingerprint + slot key tests
│   ├── small_live_test_budgets.py # 10-turn live test
│   ├── large_live_test_budgets.py # 24-turn live test with 3 revisions
│   └── README.md                  # reproduction instructions
├── evals/
│   ├── dspm_multiwoz_v6_comparison.csv  # MultiWOZ benchmark results
│   ├── dspm_multiwoz_v6_results.csv     # per-dialogue results
│   └── dspm_v05_results.png             # comparison chart
├── pyproject.toml
├── README.md
└── LICENSE
```

---

## Known Limitations

- **Minimum recommended budget:** 250 tokens for constraint-heavy conversations with 8+ criticals. At budget 400, CRR=100% and readability=100% across all use cases.
- **Extreme low budgets (60–100 tokens)** may trigger ultra-dense compression mode when the conversation contains many criticals (8+). The critical patches are always selected (CRR = 100%), but their payloads are compressed to bare tokens (e.g., `"JWT"` instead of `"JWT expires in 30 minutes"`).
- **Two-tier extraction cap.** The extraction prompt allows at most 1 constraint and 1 decision per turn (critical patches). Slot-value pairs (`[SLOT]`) have no cap — all are extracted.
- **Revision matching is rule-based.** Numeric, day-name and acronym-bridged revisions are handled. Other free-text revisions may coexist with stale values.
- **Extraction depends on the LLM.** Quality varies by model and domain. A constraint the extractor misses is not protected.
- **CRR measures patch selection, not value readability.** At very tight budgets, all critical patches are present but their payloads are compressed. Use `budget ≥ 250` for readable output.
- **Document QA.** DSPM is designed for conversations, not document compression.

---

## Paper

**DSPM: A Critical-Retention Approach to Long-Context Memory Compression for LLM Conversations**  
Dhruv Dubey, 2026 · Zenodo: [10.5281/zenodo.19438636](https://doi.org/10.5281/zenodo.19438636)

> 🏆 **Recognition:** Poster accepted at **CAICS 2026** (Indo-US Workshop on AI & Cybersecurity), jointly organized by IISc Bengaluru and Argonne National Laboratory, funded by the U.S. Department of State.

---

## Feedback

Found a case where a constraint or revision is lost? Open an issue with the conversation and the budget you used.

---

## License

MIT © 2026 Dhruv Dubey