# DSPM — memory for LLM conversations that keeps your constraints

*Dynamic Semantic Patch Memory, by Dhruv Dubey · Python package: `dspm-memory` · Repo: Kernl*

[![PyPI version](https://img.shields.io/pypi/v/dspm-memory)](https://pypi.org/project/dspm-memory/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)

```bash
pip install dspm-memory
```

Long chats drop the rule from turn 3 that everything else depends on. DSPM turns each turn into typed patches, protects constraints, decisions and revised values, and compresses the rest under a fixed token budget. Training-free. Works with any OpenAI-compatible endpoint.

**Headline numbers**

- 18-turn example conversation: **452 → 249 tokens**, 100% of critical patches selected.
- 7 hand-authored 40-turn dialogues at a 250-token budget: **about 83% fewer tokens, 100% critical retention** (same model family extracted and judged; not independently replicated).
- Poster accepted at **CAICS 2026** (Indo-US Workshop on AI & Cybersecurity), organized by IISc Bengaluru and Argonne National Laboratory.

Savings depend on conversation length: short conversations compress very little. Under very small budgets, some facts are stored but not rendered. See [Known Limitations](#known-limitations).

**Who it's for:** agents and chatbots whose long conversations must not lose a constraint, a decision, or a revised value (support bots, coding agents, assistants that remember a user's hard rules across sessions).

**Who it's not for:** summarizing or answering questions over documents. See [Known Limitations](#known-limitations).

---

## Quickstart

```python
from openai import OpenAI
from dspm import DSPMMemory

llm = OpenAI(api_key="sk-...")   # any OpenAI-compatible endpoint
memory = DSPMMemory(budget=250, llm_client=llm, model="gpt-4o-mini")

memory.add_turn("user", "Hard rules: PCI-DSS compliant, max fee 0.5%, deadline Friday.")
# ... keep adding turns as the conversation goes ...
print(memory.get_context(query="What are the hard requirements?"))
print(memory.stats)   # token counts, CRR, TRR, LLM telemetry
```

DSPM uses an LLM to extract patches, so you need a client. No paid key? Use a free Groq tier or a local Ollama model:

```python
# Groq (free tier available)
llm = OpenAI(base_url="https://api.groq.com/openai/v1", api_key="gsk_...")

# Ollama (local, no key needed)
llm = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# Together AI
llm = OpenAI(base_url="https://api.together.xyz/v1", api_key="...")
```

**Optional: semantic ranking**

```bash
pip install "dspm-memory[semantic]"
```

Adds sentence-transformer query alignment for better ranking of non-critical patches (downloads PyTorch and a small embedding model on first use). Without it, DSPM uses a neutral alignment score.

---

## Why it's different

- **Constraints and decisions are protected.** `[CON]` and `[DEC]` patches are never dropped by dedup, fusion or pruning, and are selected first at render time.
- **Revisions supersede old values.** Change a value mid-chat and only the current one survives, even across sessions.
- **Memory persists across chats.** Save at the end of one chat, load at the start of the next, with no API cost for recall.
- **Fixed budget, predictable cost.** You set the token budget; the rendered context stays inside it.

---

## See a revision supersede an old value

A payment-API conversation where a value changes mid-chat. After compression only the current value remains and the hard rules are intact.

> This short example uses `budget=60` so the effect is visible, and it compresses only about 9% (61 → 56 tokens). The large savings show up on longer conversations, as in the benchmark below. Use 200–300 for real conversations.

```python
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

Example output (recorded with v0.1.17; exact wording varies by extraction model):

```
[DEC] Track the 0.5% fee cap.
[CON] PCI-DSS requires tokenized card storage and ASV
[DEC] webhook timeout updated to 10 seconds
[CON] PCI-DSS compliant, max fee 0.5%, deadline Friday
{'turns': 6, 'total_patches': 4, 'critical_total': 4, 'critical_selected': 4, 'crr': 100, 'budget': 60, 'raw_tokens': 61, 'context_tokens': 56, 'trr': 9, ...}
```

Only the 10-second entry remains; the 30-second setting is superseded. At even tinier budgets the revision story is kept in minimal form, e.g. `10 (was 30)`.

> **Try to break it:** give it a conversation where a value changes several times under a very small budget, and [open an issue](https://github.com/zatchbell1311-wq/Kernl/issues) if a constraint or revision is lost.

---

## Persistence: memory across chats and sessions

Save the notebook when a chat ends, load it when the next one starts. Chat 2 remembers Chat 1, and revisions supersede old values across sessions:

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

**Saved:** the patches, turn count and token counters. **Not saved:** the supersession/dedup journals from `dump_store(include_history=True)`, the last rendered context, and LLM telemetry. These start empty after `load()`.

---

## How it works

DSPM converts each conversation turn into typed semantic patches (constraint, decision, code, equation, entity, structure) and compresses them under a fixed token budget. Critical patches (constraints and decisions) are structurally protected in the store and are selected first when the context is rendered.

```
Raw conversation (452 tokens, 18 turns)
        ↓
Semantic extraction → 28 patches, 18 critical
        ↓
6-stage compression pipeline
        ↓
Compressed context (249 tokens) — 100% of criticals selected
```

The 6-stage pipeline: dedup → slot fusion → delta encoding → causal pruning → utility scoring → shadow selection. (A 7th stage, adaptive per-type budgeting, is designed but not wired in this release.)

---

## The guarantee

What DSPM is built to guarantee, and where it stops:

- **In the store:** `[CON]` and `[DEC]` patches are never dropped by deduplication, fusion, or pruning.
- **At render time:** criticals are selected first. Under budget pressure their payloads are trimmed numbers-first, with units, negations and comparators protected where the budget allows.
- **Revisions:** when a constraint is revised mid-conversation (e.g. TTL 60s → 300s), the new value supersedes the old, including cross-type revisions, terse patches, and full revision chains. Under a starvation budget the revision story survives in minimal form: `10 (was 30)`.
- **Last resort:** a critical is dropped from the rendered context only when every critical is already at its word floor and the budget still cannot hold them. It stays in the store.
- **Extraction failures:** transient API failures (rate limits, 5xx) and empty responses from reasoning models are retried automatically with backoff (three attempts). gpt-oss models run extraction at low reasoning effort by default. Use `strict_llm=True` and `memory.stats` to see failures.
- **Ablation result:** removing the shadow-selection mechanism collapses CRR from 100% to 37.9%, isolating the retention behavior to a single identifiable component.

**Important limits of this guarantee:**

- It covers what the extraction model captures as a constraint or decision. If the extractor misses or misclassifies something, DSPM cannot protect it.
- **"CRR" is the share of critical patches selected into the context, not the share of facts a reader can recover from the text.** At small budgets a selected line can be trimmed so far that it loses its unit, operator or subject (for example `must not exceed 90` or `120 (was 300)`).
- In the 0.1.18 diagnostic runs below, facts were present in the store while the rendered context under-delivered at small budgets.

---

## Results

### Benchmark suite (v0.1.17)

Tested across 7 domains × 40 turns each:

| Budget | Tokens Used | TRR | CRR |
|--------|-------------|-----|-----|
| 150 | 150 | 66.8% | 100% |
| 250 | 249 | 82.8% | 100% |
| 400 | 395 | 72.4% | 100% |

CRR = Critical Retention Rate (critical patches selected into the context). TRR = Token Reduction Ratio.

> **Provenance:** measured on 7 hand-authored 40-turn technical dialogues (API design, ML ops, IoT, security IR, supply chain, clinical workflow, project management); extraction and judging by GLM via OpenRouter. The same model family performed both extraction and judging, and the benchmark has not been independently replicated. Numbers vary by extraction model and domain; check your own with `memory.stats`.

### Diagnostic runs (v0.1.18, small sample)

One synthetic 48-message alerting-pipeline conversation (12 core messages, 36 filler messages, one revision, one bound), `openai/gpt-oss-20b` on Groq, two runs per setting. "Facts delivered" counts how many of 8 ground-truth facts a fixed substring scorer finds in the context rendered after message 48.

| Setting | Budget 120 | Budget 60 |
|---|---|---|
| `feedback_context=False` (default) | 7/8, 7/8 | 3/8, 3/8 |
| `feedback_context=True`, no guard | 7/8, 3/8 | 2/8, 2/8 |
| `feedback_context=True` + `echo_guard=True` (one run) | 3/8 | n/a |

What we observed:

- The memory store held all core facts in almost every run; the loss was mostly at render time under a small budget.
- With feedback on, results varied between identical runs (7/8 vs 3/8 at budget 120); with it off they did not.
- `echo_guard` blocked the fact-deleting case (and caught 5 echoes in its one live run), but it did not fix budget crowding: filler decisions such as a pod-memory limit are also extracted as critical and compete for slots.
- Across several hundred live extraction calls, none returned empty or failed.

This is one conversation, a few runs per setting, one model, and a substring scorer. Treat it as a diagnosis, not a benchmark.

### Compare it yourself

`experiments/benchmark_compare.py` measures DSPM against a same-budget rolling summary on fixed scenarios with shared gold checks. The offline mock (`--mode mock`) prints ground-truth retention per scenario; `--mode live` runs a real extractor, a true rolling summary, and an LLM judge over a dialogues file you supply (`--dialogues`), reporting mean ± std across dialogues.

```bash
python experiments/benchmark_compare.py --mode mock
python experiments/benchmark_compare.py --mode live --base-url https://api.groq.com/openai/v1 \
    --model openai/gpt-oss-20b --dialogues dialogues.json
```

Mock-mode reference: DSPM 100% mean retention vs rolling-window 70% at budget 250. These are pipeline-check numbers, not extraction quality; run `--mode live` for that.

`experiments/bench_kv_cache.py` answers whether a stable patch context keeps more of the prompt prefix byte-stable than a same-budget raw last-K window (reference: 85.5% vs 70.7% over 40 turns at budget 250). `experiments/bench_kv_live.py` measures cached-token reuse on a real endpoint. Neither harness yet accounts for extraction latency. See `experiments/README.md`.

```bash
python experiments/bench_kv_cache.py
python experiments/bench_kv_cache.py --sweep
python experiments/bench_kv_live.py
```

**Diagnostic scripts (v0.1.18):** `scripts/diag_48.py` runs the 48-message diagnostic at budgets 120 and 60 and saves the rendered context, store dump and stats at messages 12/24/36/48; `scripts/analyze_diag.py` classifies each ground-truth fact as in the store, in the context, both, or neither. Both need `GROQ_API_KEY`. Flags: `--no-feedback`, `--echo-guard`, `--fence`, `--out-dir`.

---

## Known Limitations

- **Small budgets under-deliver rendered facts.** At budget 60 in our diagnostic conversation, only 2–3 of 8 facts appeared in the rendered context even though the store held them. The renderer trims selected lines to a word floor, so units, operators or subjects can be lost (for example `must not exceed 90` or `90 days` instead of `must not exceed 90 days`).
- **Filler can crowd out core facts.** Routine status messages (such as "pod limit raised to 6Gi") can be extracted as decisions, which are critical, and then compete with older hard rules for slots. Selection has no notion of importance beyond recency and alignment.
- **Duplicate restatements:** when the same fact is stated as both a constraint and a decision, the pair is not merged and both consume budget (observed as `[CON] within 2` plus `[DEC] 2-second delivery`).
- **Non-numeric facts can be missed:** identity facts without numbers (for example the name of a plan) were extracted in some runs and not others, and were never rendered.
- **Feedback context (`feedback_context=True`)** can cause the extractor to re-extract trimmed context lines; see [CHANGELOG.md](CHANGELOG.md). `echo_guard` reduces the damage but is experimental: its thresholds were tuned on a single failure, and it can drop a genuine update that shares no wording with its turn.
- **Persistence:** `load()` restores patches but not the history journals, the last rendered context, or telemetry.
- **Numberless obligation echoes:** when a requirement is restated without any numeral (e.g. "all payments must be PCI-DSS compliant" restated as "PCI-DSS compliance"), no collapse rule can pin the pair, because the restatement rules are number-anchored. Structured slot identity is planned for a later release.
- **Polarity under starvation:** negations and comparators are protected down to the 2-word floor, but at the floor a comparator phrase can degrade to its bare value.
- **Extraction depends on the LLM:** quality varies by model and domain, and a constraint the extractor misses is not protected. Extraction is not fully deterministic even at temperature 0.
- **Document QA:** DSPM is designed for conversations. In head-to-head document QA tests it lost to a query-aware compressor. Use it for dialogue memory, not for compressing papers or contracts.
- **Benchmark scope:** 7 hand-authored dialogues plus one synthetic diagnostic conversation. Treat the numbers as a baseline, not a third-party evaluation.
- **Still early:** version 0.1.x, Beta. APIs may change.

---

## Document mode: dense extraction

```python
# automatic — fact-dense turns route through the dense funnel:
memory.add_turn("user", "p95 latency 200ms, fee 0.5%, timeout 30 seconds, budget 250 tokens")

# documents:
n_chunks = memory.add_document(open("paper.txt", encoding="utf-8").read())

# force dense mode:
memory = DSPMMemory(budget=250, llm_client=llm, model="gpt-4o-mini", dense=True)
```

---

## API reference

### `DSPMMemory(budget, llm_client, model, ...)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `budget` | int | 250 | Maximum tokens in the compressed output |
| `llm_client` | OpenAI | None | Any OpenAI-compatible client |
| `model` | str | `"gpt-4o-mini"` | Model used for patch extraction |
| `dense` | bool | False | Force dense extraction funnel on every turn |
| `stable_order` | bool | False | Emit patches in (turn, id) order for KV cache stability |
| `render_source_roles` | bool | False | Show the speaker role on rendered lines |
| `strict_llm` | bool | False | Raise on empty, truncated or unparseable extraction output instead of a silent 0-patch turn |
| `feedback_context` | bool | **False** (changed in 0.1.18) | Feed the last rendered context into the next extraction prompt |
| `echo_guard` | bool | False | Experimental. Drop patches that echo the fed-back context. Only active when `feedback_context=True` |
| `feedback_fence` | bool | False | Experimental. Wrap the fed-back context in do-not-extract markers |
| `protect_bound_wording` | bool | False | Experimental. Stop a plain restatement from replacing a stored bound |
| `reasoning_param_mode` | str | None | Where to send `reasoning_effort` for gpt-oss models: `"top"`, `"extra_body"`, `"off"` |
| `max_tokens_floor` | int | None | Minimum output budget for extraction calls |

The budget can be changed at any time:

```python
memory.budget = 100
memory.set_budget(100)   # explicit alternative — same effect
```

### Methods

| Method | Description |
|--------|-------------|
| `add_turn(role, text)` | Add a conversation turn. Returns extracted patches. |
| `add_document(text, chunk_tokens=400)` | Ingest a document with sentence-aware chunking. |
| `get_context(query="")` | Returns compressed context string. |
| `set_budget(budget)` | Update the token budget. |
| `dump_store(include_history=False)` | Inspect stored patches; with `include_history=True` also the supersession and dedup journals. |
| `save(path)` | Persist memory to JSON (atomic write). |
| `load(path)` | Load a notebook (merge semantics). |
| `reset()` | Clear all memory. |

### Properties

| Property | Description |
|----------|-------------|
| `memory.stats` | Dict with token counts, patch counts, CRR and `llm_*` telemetry |
| `memory.critical_patches` | List of critical patches in memory |
| `memory.all_patches` | List of every patch in memory |

---

## Repository layout

```
.
├── src/dspm/
│   ├── config.py
│   ├── patch.py
│   ├── extractor.py
│   ├── memory.py
│   ├── engine.py
│   └── persistence.py
├── tests/                        # 151 offline tests (pytest)
├── scripts/                      # live smoke gates and diagnostics (need GROQ_API_KEY)
├── experiments/
│   ├── benchmark_compare.py
│   ├── bench_kv_cache.py
│   ├── bench_kv_live.py
│   └── results/
├── examples/
│   └── basic_usage.py
├── pyproject.toml
├── CHANGELOG.md
├── LICENSE
└── README.md
```

Run the offline tests with `pytest tests -q`. Do not run a bare `pytest` from the repo root: some scripts under `scripts/` call a live API when collected.

---

## Paper

**DSPM: A Critical-Retention Approach to Long-Context Memory Compression for LLM Conversations**
Dhruv Dubey, 2026
Zenodo: [10.5281/zenodo.19438636](https://doi.org/10.5281/zenodo.19438636)

*Recognition:* Poster accepted at CAICS 2026 (Indo-US Workshop on AI & Cybersecurity), jointly organized by IISc Bengaluru and Argonne National Laboratory, funded by the U.S. Department of State.

---

## Feedback

Found a case where a constraint or revision is lost? Please [open an issue](https://github.com/zatchbell1311-wq/Kernl/issues) with the conversation and the budget you used.

## License

MIT © 2026 Dhruv Dubey
