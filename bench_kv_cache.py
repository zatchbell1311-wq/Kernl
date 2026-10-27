#!/usr/bin/env python3
"""Deterministic prompt-cache (KV-cache) simulation.

Compares, turn by turn, how much of the prompt prefix stays byte-stable
(and therefore provider-cacheable) under:

  DSPM      -- a budgeted patch context (<= BUDGET tokens) that is re-rendered
               every turn: usually appends one patch slot near the tail,
               sometimes dedups with no change at all, occasionally rewrites
               a patch mid-list.
  BASELINE  -- a same-budget raw last-K transcript window that slides by one
               turn (about TURN_MEAN tokens) every turn.

Everything is seeded and offline; no imports, no network. The numbers printed
are what THIS model computes under the constants below -- they are a
simulation, not a live measurement (for that, see bench_kv_live.py).

Run:  python bench_kv_cache.py          (prints the table, writes JSON)
      python bench_kv_cache.py --sweep  (parameter sensitivity)
"""

import argparse
import json
import os
import random

SEED = 20261019
SYSTEM_TOKENS = 80     # system prompt, identical across calls
QUERY_TOKENS = 20      # the per-turn user query, always new
BUDGET = 250           # context budget in tokens (the DSPM default)
PATCH_SLOT = 25        # tokens occupied by one rendered patch slot
TURNS = 40
WARMUP = 3             # discard first turns while the context fills
TURN_MEAN = 80         # raw transcript tokens per turn (user + assistant)
TURN_JITTER = 25
P_NOCHANGE = 0.20      # turn adds nothing (fingerprint dedup hit)
P_APPEND = 0.70        # turn appends one new patch at the tail slot
P_REVISION = 0.10      # turn rewrites a patch mid-list (slot churns)
# P_NOCHANGE + P_APPEND + P_REVISION == 1.0


def simulate(seed=SEED, patch_slot=PATCH_SLOT, budget=BUDGET,
             turn_mean=TURN_MEAN, turn_jitter=TURN_JITTER, turns=TURNS):
    """Return (dspm_rates, baseline_rates) per turn, warmup excluded."""
    rng = random.Random(seed)
    dspm_rates, base_rates = [], []
    prompt_total = SYSTEM_TOKENS + budget + QUERY_TOKENS
    for _ in range(turns):
        # raw transcript delta this turn
        delta = max(30.0, rng.gauss(turn_mean, turn_jitter))

        r = rng.random()
        if r < P_NOCHANGE:
            stable = float(budget)                    # prefix fully stable
        elif r < P_NOCHANGE + P_APPEND:
            stable = float(budget - patch_slot)       # new tail slot churns
        else:
            stable = rng.uniform(0.2, 0.8) * (budget - patch_slot)
        dspm_rates.append((SYSTEM_TOKENS + stable) / prompt_total)

        # baseline window slides: everything before the newest turn is shared
        stable_base = max(0.0, budget - delta)
        base_rates.append((SYSTEM_TOKENS + stable_base) / prompt_total)
    return dspm_rates[WARMUP:], base_rates[WARMUP:]


def _summary(rates):
    return sum(rates) / len(rates)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sweep', action='store_true',
                    help='print sensitivity across patch-slot and jitter ranges')
    args = ap.parse_args()

    dspm, base = simulate()
    d_mean, b_mean = _summary(dspm), _summary(base)

    print("Deterministic KV-cache simulation (DSPM context vs raw last-K window)")
    print(f"constants: seed={SEED} system={SYSTEM_TOKENS} query={QUERY_TOKENS} "
          f"budget={BUDGET} patch_slot={PATCH_SLOT} turns={TURNS} "
          f"turn~N({TURN_MEAN},{TURN_JITTER})")
    print("=" * 74)
    print(f"{'turn':>4}  {'DSPM stable-prefix %':>20}  {'baseline stable-prefix %':>24}")
    for t in range(0, len(dspm), 5):
        print(f"{t + WARMUP + 1:>4}  {dspm[t] * 100:>19.1f}  "
              f"{base[t] * 100:>23.1f}")
    print("-" * 74)
    print(f"DSPM stable-prefix (cacheable) rate  : {d_mean * 100:.1f}%")
    print(f"raw-window baseline rate             : {b_mean * 100:.1f}%")
    print(f"advantage                            : {(d_mean - b_mean) * 100:+.1f} points")
    print()
    print("NOTE: simulation, not a live measurement. Assumptions are the")
    print("constants above; providers differ. Live check: bench_kv_live.py")

    results = {
        "model": "deterministic prompt-cache simulation (rebuilt)",
        "seed": SEED,
        "params": {"system_tokens": SYSTEM_TOKENS, "query_tokens": QUERY_TOKENS,
                   "budget": BUDGET, "patch_slot": PATCH_SLOT, "turns": TURNS,
                   "turn_mean": TURN_MEAN, "turn_jitter": TURN_JITTER,
                   "p_nochange": P_NOCHANGE, "p_append": P_APPEND,
                   "p_revision": P_REVISION},
        "dspm_prefix_hit_rate": round(d_mean, 4),
        "baseline_prefix_hit_rate": round(b_mean, 4),
        "advantage_points": round((d_mean - b_mean) * 100, 1),
    }
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'bench_kv_cache_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out}")

    if args.sweep:
        print()
        print("sensitivity sweep (patch_slot x turn_jitter):")
        rows = []
        for ps in (15, 20, 25, 30, 35):
            for tj in (10, 25, 40):
                d, b = simulate(patch_slot=ps, turn_jitter=tj)
                rows.append((_summary(d), _summary(b)))
                print(f"  patch_slot={ps:>2} jitter={tj:>2} -> "
                      f"DSPM {_summary(d) * 100:5.1f}%  baseline {_summary(b) * 100:5.1f}%")
        d_lo = min(r[0] for r in rows); d_hi = max(r[0] for r in rows)
        b_lo = min(r[1] for r in rows); b_hi = max(r[1] for r in rows)
        print(f"  DSPM range {d_lo * 100:.1f}..{d_hi * 100:.1f}%  "
              f"baseline range {b_lo * 100:.1f}..{b_hi * 100:.1f}%")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
