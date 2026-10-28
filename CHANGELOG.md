# Changelog

## [0.1.21] - 2026-10-28

Major bug-fix release. Closes nine bugs across the memory, extractor, and persistence layers. Includes two behavior changes (CRR reporting and truncation handling) and adds `portalocker` as a dependency.

### Breaking Changes

- **CRR formula (Bug #4b):** `stats['crr']` no longer reports a vacuous 100 when `get_context()` has been called but the store has no critical patches. Before `get_context()`, CRR remains 100 (assumed perfect, for backward compatibility). After `get_context()`, it is 0 when there are no criticals; with criticals, the existing formula is unchanged.
- **Truncation handling (Bug #5):** `extract_turn()` no longer retries when `finish_reason="length"` and content is empty. It exits the retry loop immediately: `llm_calls=1`, `llm_length_finish=1`, `llm_empty=0`, and `llm_retries=0`.

### Added

- **File locking (Bug #7a):** added `portalocker>=2.7` for cross-platform locking. `save_memory()` and `load_memory()` lock the file during access. If `portalocker` is unavailable, they emit a `RuntimeWarning` and continue without locking.
- **Async turn ingestion (Bug #7b):** added `DSPMMemory.async_add_turn()` and `dspm.extractor.async_extract_turn()`. They await native async clients, support older SDKs with `acreate()`, and run sync clients through `asyncio.to_thread()`. The existing synchronous `add_turn()` is unchanged.
- **Conversation-based TRR (Bug #6):** added `trr_by_conversation` to stats to measure compression from raw conversation tokens to delivered context tokens. The existing `trr` key remains unchanged and measures patch compression.
- **Corrupt notebook handling (Bug #8):** added `CorruptNotebookError` and `load_memory(strict=False)`. With `strict=True`, corrupt notebooks are quarantined as `<path>.corrupt.<timestamp>` and the exception is raised. The default `strict=False` preserves the previous silent-skip behavior and returns 0.

### Fixed

- **Bug #1 — TTL acronym supersession:** `_head_anchored()` recognizes shared all-caps acronyms as topic anchors, so extractor rewrites such as `TTL` to `Time-to-live (TTL)` supersede the stale patch.
- **Bug #2 — duplicate deadline rendering:** added regression coverage confirming that `Deadline set to Friday` followed by `Deadline updated to Monday` renders as one line with `(was Friday)`. The existing `_day_revision()` rule handled the case.
- **Bug #3a — weekday capitalization:** added `_day_value_preserved()` to retain the original capitalization of weekday names in revision history.
- **Bug #3b — weekday revision chains:** `_replaced_value()` now detects weekday-only revisions and retains full lineage. For example, `Deadline updated to Monday (was Friday)` followed by `Deadline moved to Wednesday` records `(was Monday then Friday)`.
- **Bug #4a — compliance classification:** added `_COMPLIANCE_RULE` to both dense and non-dense extraction prompts, classifying regulatory, compliance, and security requirements as constraints rather than entities.
- **Bug #5 — empty truncated LLM responses:** see truncation handling above; truncation exits immediately with accurate telemetry.
- **Bug #6 — TRR denominator:** see conversation-based TRR above.
- **Bug #7a — concurrent save race:** `save_memory()` and `load_memory()` acquire an exclusive file lock during file access.
- **Bug #7b — blocking AI calls:** see async turn ingestion above.
- **Bug #8 — corrupt notebook data loss:** `save_memory()` backs up an existing file to `<path>.bak`; `load_memory()` quarantines corrupt files instead of leaving them to be overwritten by a later save; `strict=True` raises `CorruptNotebookError` for callers who need explicit failure handling.

### Tests

- Added 17 tests in `tests/test_memory.py` for Bugs #2, #3, #4, #5, and #6.
- Added 11 tests in `tests/test_backend_hardening.py` for Bugs #7a, #7b, and #8.
- Total: 187 tests passed, up from 162 in 0.1.20.

### Acknowledgments

Backend review for Bugs #7a, #7b, and #8 was inspired by a LinkedIn reviewer's feedback on concurrent saves, blocking calls, and silent file corruption.

## 0.1.20

- Fixes a packaging bug: 0.1.19 bumped the version in `pyproject.toml` but not in `src/dspm/__init__.py`, so `dspm.__version__` incorrectly reported `"0.1.18"` despite being installed as 0.1.19. No functional code changes beyond the version string fix.

## 0.1.19

Docs and repo hygiene only, no code changes.

- Fixed README repo links (Kernl, not dspm)
- Corrected an unconfirmed diagnostic table cell to "n/a"
- Updated CHANGELOG wording for deferred items
- Removed diagnostic scratch output and logs from version control

## [0.1.18] - 2026-10-26

- **`feedback_context` now defaults to `False`.** With it on, budget-trimmed context fed back into extraction was re-extracted as patches; in one run a fused fragment superseded two real facts and delivery at budget 120 swung between 7/8 and 3/8 across identical runs. Pass `feedback_context=True` to restore the old behavior.
- **LLM telemetry in `memory.stats`:** `llm_calls`, `llm_empty`, `llm_errors`, `llm_retries`, `llm_length_finish`, `llm_parse_failures`, `llm_fallbacks`, plus `llm_echo_drops` and `llm_unattributed_drops` when the echo guard is used. A turn whose extraction fails silently (empty or unparseable output) is now counted in `llm_fallbacks`.
- **`strict_llm=True`:** raises instead of silently producing a 0-patch turn when the extraction output is empty, truncated, or not parseable.
- **`dump_store(include_history=False)`:** returns every stored patch (text, type, criticality, turn, whether it was in the last rendered context), so "forgotten" can be told apart from "stored but not rendered". With `include_history=True` it also returns the supersession and dedup journals.
- **Opt-in, experimental:** `echo_guard=True` (experimental) drops patches that echo the fed-back context. It prevented the fact-deletion case in a live run but did not stop filler decisions from crowding small budgets. `feedback_fence=True` (experimental, no live evidence) wraps the fed-back context in do-not-extract markers. `protect_bound_wording=True` stops a plain restatement from replacing a stored bound.
- **Reasoning-model safety:** `reasoning_param_mode` (`"top"` | `"extra_body"` | `"off"`) controls where `reasoning_effort` is sent for gpt-oss models; `max_tokens_floor` enforces a minimum output budget for extraction calls.

## [0.1.17] - 2026-10-22

### Fixed
- **Cross-type restatements collapse (review finding S1)** - CON<->DEC restatements across types now collapse (rule b3); the live smoke CRR failure they caused is gone.
- **Thin cross-type echoes (review finding S2)** - the old frozen fallback rule is removed and replaced by the bound-head bridge (rule b4): echoes sharing the same bound head with two or more shared non-numeric words, equal numbers, equal units and compatible polarity now mark the old fact stale. The previously frozen fee-pair tests were unfrozen and rewritten to assert the (b4) collapse.
- **Docs honesty (review fixes)** - T7 adaptive budgeting is RESERVED, NOT WIRED: `ema_query` is only ever reset, never updated, and `ALPHA_EMA`, `SHADOW_THRESHOLD`, `DELTA_MIN_SAVING`, `BASE_BUDGET_FRACTIONS`, `CRITICAL_MAX_WORDS` and `REVISION_OVERLAP` are unwired config - do not quote T7 as active behavior. T2 slot fusion keeps the highest (utility, turn_index), which in practice is the newest turn (utility is 0.0 at that stage).

### Added
- **Patch source roles** - extracted facts remember whether they came from the user or the assistant; roles survive save/load, stats exposes `critical_by_role`, and rendered criticals are annotated, e.g. "fee cap set to 0.5% as tracked (user)".
- **Split token accounting** - stats reports `conversation_tokens` (raw turns), `patch_tokens` (extracted) and `context_tokens` (delivered) separately.
- **Evaluation harness** (`eval_harness.py`) - five scenarios, all green (AGGREGATE 5/5): `payments_echo`, `tier_trap`, `revision_chain`, `starvation_honesty`, `trim_quality`; `--live` mode adds latency / token / cost accounting.
- **14 new tests** (12 memory + 2 persistence); full suite 94/94 green.
- **Verified replays** - budget 59: `crr=100` with two spare capacity slots; variance probe delivers 7/7; starvation honesty reports `store=8, delivered=3, crr=37` instead of over-delivering.

### Known issues (repro locked, fix queued for a later release)
- **S3 trim quality** - at budget 59 the delivered revision line can lose the (was X) span
  form and the subject noun ("timeout 10 seconds, 30 seconds"); the current value always
  survives. Locked in as the trim_quality harness repro.
- **(b4) scope-noun limitation** - same-number scope-noun pairs (e.g. org-admins /
  max-users 500) merge by design; structured slots are the planned 0.1.18 fix. Documented by
  tier_trap's fifth split guard.
- Also queued for a later release: rule (a) multi-overlapping-slot sweep; rule (c) polarity-free pairs.
