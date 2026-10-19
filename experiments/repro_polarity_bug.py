#!/usr/bin/env python3
"""Reconstructed regression harness for the v0.1.15 critical-payload
trimming ("polarity") bug -- the six documented failure modes.

PROVENANCE (read before trusting this file)
-------------------------------------------
The original experiments/repro_polarity_bug.py referenced by the README was
lost from disk before it was ever committed to git. This is a RECONSTRUCTION
built only from surviving evidence:

  * tests/test_memory.py -- the v0.1.13/v0.1.14 regression-test comments,
    which document the pre-fix greedy trimmer precisely:
      - numeric-first: numbers are selected before anything else
      - word priorities: number-adjacent = 40, key nouns = 30 (v0.1.13),
        raised to 50 in v0.1.14
      - tie-break: "priority desc, ties > earlier word"
    and the failures it caused in the field ("encryption AES-256 at",
    "rate limit 100 requests" losing the subject, "(was 20 then 20)").
  * experiments/polarity_after.txt -- the post-fix budget sweep produced by
    experiments/probe_polarity.py: omission bias, monotonic recovery, zero
    polarity inversions, quantity caps surviving from b=6-8, and the known
    b=8 edge case "[CON] Retries 3".

The pre-fix trimmer and the current contract are embedded here as FROZEN
MODELS so the six failure modes reproduce deterministically regardless of
which dspm version is installed. No imports, no network.

Run:  python experiments\\repro_polarity_bug.py
"""

import sys

KEY_NOUNS = {"retries", "retry", "limit", "rate", "session", "ttl", "cache",
             "encryption", "algorithm", "token", "timeout", "attempts",
             "requests", "expires", "expiry"}
NEGATIONS = {"no", "not", "never", "without", "zero"}

# Pre-fix (v0.1.13/v0.1.15) priorities -- the bug.
PRIORITY_NUM = 100
PRIORITY_ADJACENT = 40
PRIORITY_KEYNOUN_OLD = 30
PRIORITY_NEGATION_OLD = 10        # negations had no protection -> inversions

# Current contract: key nouns raised to 50 (v0.1.14), negations protected
# (v0.1.16 polarity fix, calibrated to the probe contract: never invert).
PRIORITY_KEYNOUN_NEW = 50
PRIORITY_NEGATION_NEW = 60


def _is_number(word):
    return any(ch.isdigit() for ch in word.strip(".,;:()"))


def frozen_trim(payload, budget, fixed=False):
    """Frozen model of the T6 critical-payload trimmer.

    Numeric-first greedy fill: numbers first, then priority-desc, ties
    resolved in favour of the earlier word. Selected words are re-joined in
    their original order. fixed=False reproduces the pre-fix behaviour;
    fixed=True reproduces the current contract.
    """
    words = payload.split()
    if budget <= 0 or not words:
        return ""
    if budget >= len(words):
        return " ".join(words)
    kn = PRIORITY_KEYNOUN_NEW if fixed else PRIORITY_KEYNOUN_OLD
    ng = PRIORITY_NEGATION_NEW if fixed else PRIORITY_NEGATION_OLD
    scored = []
    for i, w in enumerate(words):
        core = w.strip(".,;:()").lower()
        if _is_number(w):
            pri = PRIORITY_NUM
        elif i > 0 and _is_number(words[i - 1]):
            pri = PRIORITY_ADJACENT
        elif core in KEY_NOUNS:
            pri = kn
        elif core in NEGATIONS:
            pri = ng
        else:
            pri = 10
        scored.append((-pri, i, w))
    picked = sorted(scored)[:budget]
    chosen = {i for _, i, _ in picked}
    return " ".join(w for i, w in enumerate(words) if i in chosen)


def frozen_bake(current, replaced_values, fixed=False):
    """Frozen model of the (was X then Y) revision-chain baker."""
    if fixed:
        deduped = []
        for v in replaced_values:
            if not deduped or deduped[-1] != v:
                deduped.append(v)
        replaced_values = deduped
    chain = " then ".join(replaced_values)
    return f"{current} (was {chain})" if replaced_values else current


# The six documented failure modes -------------------------------------------

MODES = [
    ("F1 polarity inversion: negation dropped, verb kept",
     lambda: frozen_trim("no retries permitted", 1, fixed=False),
     lambda: frozen_trim("no retries permitted", 1, fixed=True),
     lambda old, new: old == "retries" and "no" in new.split()),

    ("F2 cap semantics lost: 'must not exceed' reduced to a plain statement",
     lambda: frozen_trim("cache ttl must not exceed 90 seconds", 4, fixed=False),
     lambda: frozen_trim("cache ttl must not exceed 90 seconds", 4, fixed=True),
     lambda old, new: "not" not in old.split() and "not" in new.split()),

    ("F3 subject lost to number-adjacent word (v0.1.14 test evidence)",
     lambda: frozen_trim("rate limit 100 requests per minute", 2, fixed=False),
     lambda: frozen_trim("rate limit 100 requests per minute", 2, fixed=True),
     lambda old, new: old == "100 requests" and new == "rate 100"),

    ("F4 orphan number+unit: the constraint's subject disappears",
     lambda: frozen_trim("session expires after 90 days grace period", 2, fixed=False),
     lambda: frozen_trim("session expires after 90 days grace period", 2, fixed=True),
     lambda old, new: old == "90 days" and new == "session 90"),

    ("F5 readability collapse: fragment keeps filler, loses the subject",
     lambda: frozen_trim("encryption uses AES-256 at rest", 2, fixed=False),
     lambda: frozen_trim("encryption uses AES-256 at rest", 2, fixed=True),
     lambda old, new: old == "AES-256 at" and new == "encryption AES-256"),

    ("F6 revision chain bakes consecutive duplicates '(was 20 then 20)'",
     lambda: frozen_bake("60", ["20", "20", "300"], fixed=False),
     lambda: frozen_bake("60", ["20", "20", "300"], fixed=True),
     lambda old, new: "20 then 20" in old and "20 then 20" not in new),
]


def main():
    print("v0.1.15 critical-payload trimming bug -- six documented failure modes")
    print("frozen-model reconstruction; see module docstring for provenance")
    print("=" * 74)
    reproduced = 0
    for title, old_fn, new_fn, check in MODES:
        old, new = old_fn(), new_fn()
        ok = check(old, new)
        reproduced += ok
        print(f"{title}")
        print(f"    pre-fix  : {old!r}   <-- failure{' (REPRODUCED)' if ok else ''}")
        print(f"    contract : {new!r}")
        print(f"    status   : {'PASS' if ok else 'FAIL -- model does not match evidence'}")
        print()
    print(f"reproduced {reproduced}/6 on the frozen pre-fix model")
    print("post-fix context-budget sweep: run experiments\\probe_polarity.py"
          " (evidence: experiments\\polarity_after.txt)")
    if reproduced != len(MODES):
        print("MODEL MISMATCH -- investigate before shipping", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
