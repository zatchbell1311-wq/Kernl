# Experiments

Reproducible evidence for the claims in the top-level README. Everything here
is deterministic and offline unless explicitly marked LIVE.

## Layout

| Path | What it is | How to run |
|---|---|---|
| `probe_polarity.py` | Post-fix polarity probe: sweeps the context budget 4-40 across four negative constraints and records what survives at each budget. | `python experiments\probe_polarity.py` |
| `polarity_after.txt` | Saved probe output: omission-only starvation, monotonic recovery, zero polarity inversions, quantity caps stable from b=6-8. | -- |
| `repro_polarity_bug.py` | Frozen-model reconstruction of the v0.1.15 critical-payload trimming bug; reproduces the six documented failure modes deterministically (see its provenance header). | `python experiments\repro_polarity_bug.py` |
| `2026-10-10-cross-family-pilot/` | The Oct 10 cross-family pilot: Colab notebook, its PDF export, and the result JSONs (`budget80_results.json`, `cross_model_results.json`, `self_judgment.json`). | open the notebook in Colab |

Companion scripts live at the repo root:

- `bench_kv_cache.py` -- deterministic prompt-cache simulation (writes
  `bench_kv_cache_results.json`; the README cites exactly these values).
- `bench_kv_live.py` -- LIVE cached-token measurement against a real
  OpenAI-compatible endpoint (needs `ZAI_API_KEY` or `OPENAI_API_KEY`).
- `benchmark_compare.py` -- DSPM vs rolling-summary retention harness;
  `--mode mock` is a deterministic offline pipeline check, `--mode live`
  produces the published mean +/- std figure.

## Provenance

The original `repro_polarity_bug.py` and the root bench scripts were lost
before ever being committed to git. The current versions were rebuilt from
the surviving evidence: the regression-test comments in
`tests/test_memory.py` (which document the pre-fix trimmer's exact word
priorities), `polarity_after.txt` (the post-fix contract), and the Oct 11
baseline-experiment notes. The six failure modes in the repro harness are
demonstrated on a frozen model of the pre-fix trimmer, so they reproduce
deterministically regardless of the installed dspm version.

## How the numbers connect

- Six failure modes -> `repro_polarity_bug.py` (6/6 reproduced on the frozen
  pre-fix model, 0/6 on the current contract).
- Post-fix polarity contract -> `probe_polarity.py` / `polarity_after.txt`.
- KV-cache stability -> `bench_kv_cache.py` (deterministic simulation under
  stated constants; sensitivity via `--sweep`; live poles via
  `bench_kv_live.py`).
- DSPM vs rolling summary -> `benchmark_compare.py --mode live` (mean +/- std
  across dialogues; `--mode mock` verifies the harness offline).
