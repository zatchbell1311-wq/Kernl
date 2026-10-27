#!/usr/bin/env python3
"""Live prompt-cache (KV-cache) measurement.

Sends two prompt sequences through a real OpenAI-compatible API and reads the
cached-token usage the provider reports:

  stable   -- the same context block every turn (the ideal DSPM regime: the
              rendered patch context barely moves, so the prefix stays
              byte-identical and cacheable).
  sliding  -- a same-size raw window whose start advances every turn (the
              naive last-K baseline).

bench_kv_cache.py models the DSPM churn BETWEEN these two poles; this script
measures the poles against a real provider. Requires an API key.

Run:  set ZAI_API_KEY=...   (or OPENAI_API_KEY)
      python bench_kv_live.py --model <name> [--turns 12] [--base-url URL]
"""

import argparse
import os
import statistics
import sys

SYSTEM = "You are a precise assistant. Answer from the rules in the context."
FILLER = ("Alpha and beta telemetry pipelines ship nightly dashboards with "
          "quota alerts, retry budgets, latency percentiles and capacity "
          "notes, plus weekly reviews of incident churn and postmortems. ")
WORDS = (FILLER * 8).split()
WINDOW = 180          # words in the context block
SLIDE = 15            # words the raw window slides per turn


def stable_context():
    return ("SERVICE RULES (stable block): "
            + " ".join(f"[R{i}] keep rule {i} enabled." for i in range(1, 40)))


def sliding_context(t):
    start = (t * SLIDE) % (len(WORDS) - WINDOW)
    return "RAW WINDOW: " + " ".join(WORDS[start:start + WINDOW])


def query(t):
    return f"Question {t}: restate rule {t % 39 + 1} only."


def cached_tokens(resp):
    u = getattr(resp, "usage", None)
    if u is None:
        return None, None
    pd = getattr(u, "prompt_tokens_details", None)
    if pd is not None and getattr(pd, "cached_tokens", None) is not None:
        return pd.cached_tokens, u.prompt_tokens
    for attr in ("prompt_tokens_cache_hit", "prompt_cache_hit_tokens",
                 "cached_tokens"):
        v = getattr(u, attr, None)
        if v is not None:
            return v, u.prompt_tokens
    return None, u.prompt_tokens


def run_mode(client, model, mode, turns):
    rates = []
    for t in range(turns):
        ctx = stable_context() if mode == "stable" else sliding_context(t)
        resp = client.chat.completions.create(
            model=model, max_tokens=1, temperature=0,
            messages=[{"role": "system", "content": SYSTEM + "\n" + ctx},
                      {"role": "user", "content": query(t)}])
        cached, prompt = cached_tokens(resp)
        if cached is None or prompt in (None, 0):
            return None
        if t > 0:                       # first call is always cold
            rates.append(cached / prompt)
        print(f"  {mode:>7} turn {t + 1:>2}: cached {cached}/{prompt} tokens")
    return statistics.mean(rates)


def main():
    ap = argparse.ArgumentParser(description="live KV-cache measurement")
    ap.add_argument("--model", required=True)
    ap.add_argument("--turns", type=int, default=12)
    ap.add_argument("--base-url", default=None)
    args = ap.parse_args()
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("live mode needs the openai package:  pip install openai")
    api_key = (os.environ.get("ZAI_API_KEY") or os.environ.get("DSPM_API_KEY")
               or os.environ.get("OPENAI_API_KEY"))
    if not api_key:
        sys.exit("set ZAI_API_KEY (or OPENAI_API_KEY / DSPM_API_KEY) first")
    base_url = (args.base_url or os.environ.get("ZAI_API_BASE")
                or ("https://api.z.ai/api/paas/v4"
                    if os.environ.get("ZAI_API_KEY") else None))
    client = OpenAI(api_key=api_key, base_url=base_url)

    out = {}
    for mode in ("stable", "sliding"):
        print(f"measuring {mode} prefix ({args.turns} calls)...")
        rate = run_mode(client, args.model, mode, args.turns)
        if rate is None:
            print(f"  provider did not report cached-token usage for {mode}; "
                  "cannot measure live cache hits on this endpoint.")
            return 2
        out[mode] = rate
        print(f"  {mode} mean cached fraction: {rate * 100:.1f}%")
    print()
    print(f"stable-prefix cached fraction : {out['stable'] * 100:.1f}%")
    print(f"sliding-window cached fraction : {out['sliding'] * 100:.1f}%")
    print("bench_kv_cache.py models the DSPM regime between these two poles.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
