# Changelog

## [0.1.18] - 2026-10-26

- feedback_context now defaults to False. With it on, budget-trimmed context
  fed back into extraction was re-extracted as patches; in one run a fused
  fragment superseded two real facts and delivery at budget 120 swung between
  7/8 and 3/8 across identical runs. Pass feedback_context=True to restore the
  old behavior.
- New opt-in echo_guard=True (experimental): drops patches that echo the
  fed-back context. It prevented the fact-deletion case in a live run but did
  not stop filler decisions from crowding small budgets.
- New opt-in feedback_fence=True (experimental, no live evidence).
- New llm_echo_drops / llm_unattributed_drops telemetry.

## [0.1.17] - 2026-10-22

### Fixed
- **Cross-type restatements collapse (review finding S1)** - CON<->DEC restatements across
  types now collapse (rule b3); the live smoke CRR failure they caused is gone.
- **Thin cross-type echoes (review finding S2)** - the old frozen fallback rule is removed and
  replaced by the bound-head bridge (rule b4): echoes sharing the same bound head with two or
  more shared non-numeric words, equal numbers, equal units and compatible polarity now mark
  the old fact stale. The previously frozen fee-pair tests were unfrozen and rewritten to
  assert the (b4) collapse.
- **Docs honesty (review fixes)** - **T7 adaptive budgeting is RESERVED, NOT WIRED**: ema_query
  is only ever reset, never updated, and ALPHA_EMA, SHADOW_THRESHOLD, DELTA_MIN_SAVING,
  BASE_BUDGET_FRACTIONS, CRITICAL_MAX_WORDS and REVISION_OVERLAP are unwired config - do not
  quote T7 as active behavior. T2 slot fusion keeps the highest (utility, turn_index), which
  in practice is the newest turn (utility is 0.0 at that stage).

### Added
- **Patch source roles** - extracted facts remember whether they came from the user or the
  assistant; roles survive save/load, stats exposes critical_by_role, and rendered criticals
  are annotated, e.g. "fee cap set to 0.5% as tracked (user)".
- **Split token accounting** - stats reports conversation_tokens (raw turns), patch_tokens
  (extracted) and context_tokens (delivered) separately.
- **Evaluation harness** (eval_harness.py) - five scenarios, all green (AGGREGATE 5/5):
  payments_echo, tier_trap, revision_chain, starvation_honesty, trim_quality; --live mode
  adds latency / token / cost accounting.
- **14 new tests** (12 memory + 2 persistence); full suite 94/94 green.
- **Verified replays** - budget 59: crr=100 with two spare capacity slots; variance probe
  delivers 7/7; starvation honesty reports store=8, delivered=3, crr=37 instead of
  over-delivering.

### Known issues (repro locked, fix queued for a later release)
- **S3 trim quality** - at budget 59 the delivered revision line can lose the (was X) span
  form and the subject noun ("timeout 10 seconds, 30 seconds"); the current value always
  survives. Locked in as the trim_quality harness repro.
- **(b4) scope-noun limitation** - same-number scope-noun pairs (e.g. org-admins /
  max-users 500) merge by design; structured slots are the planned 0.1.18 fix. Documented by
  tier_trap's fifth split guard.
- Also queued for a later release: rule (a) multi-overlapping-slot sweep; rule (c) polarity-free pairs.
