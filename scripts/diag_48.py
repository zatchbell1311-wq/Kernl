"""UNTESTED -- 48-message DSPM diagnostic runner (budgets 120 and 60).

Written against the pasted dspm-memory 0.1.18 sources (config.py,
extractor.py, memory.py) and the notebook Cell 4 code only. persistence.py
and engine.py were NOT provided: everything below that touches save/load
is verify-then-trust, and no claim is made about engine internals. Not
executed by the author.

What it does
------------
Two arms, DSPMMemory(budget=120) and DSPMMemory(budget=60), fresh
instances, llm_client = Groq OpenAI-compatible endpoint, model
"openai/gpt-oss-20b", strict_llm=True, every other option at its default
(no protect_bound_wording, feedback_context on). One add_turn() call per
message, 48 messages. After messages 12/24/36/48 each arm writes
diag_out/b{budget}_cp{n}.json containing the rendered probe context
(get_context(query=FINAL_QUERY) -- engine-only, no LLM call),
dump_store(include_history=True), and stats.

Usage (PowerShell, repo root, venv active)
------------------------------------------
    $env:GROQ_API_KEY = "..."     # session-only; never hardcoded, never printed
    python scripts\diag_48.py [--pace 12.0] [--out-dir diag_out]

Resume semantics (from the API read: memory.save(path) writes the store
via dspm.persistence; memory.load(path) merges it back; the persistence
format itself was not provided)
------------------------------------------------------------------------
* Arm-level skip: if b{budget}_cp48.json exists and parses, the arm is
  already complete -- skipped, no LLM calls.
* Mid-arm resume: mem.save() runs after every checkpoint. On rerun the
  script calls mem.load() into a fresh instance and VERIFIES the
  round-trip against that checkpoint's dump_store (turns, budget,
  patches, superseded + deduped journals). Verified -> continue from the
  next message, no re-ingestion, no wasted quota. Any hard mismatch ->
  the arm aborts with the differences printed; delete the arm's
  state_b{budget}.json and b{budget}_cp*.json to rerun it from scratch.
* Post-load repairs for state persistence may not cover (all labeled in
  code, all attributes read in the pasted memory.py): llm_* telemetry,
  _last_context, _context_built, _conversation_tokens are restored to
  the values an uninterrupted run would hold at that boundary.
* CANNOT be verified from the provided source: the engine's internal
  EMA state. If persistence does not save it, post-resume compressions
  may differ slightly from an uninterrupted run; resumed arms are marked.

Pacing
------
--pace seconds slept between add_turn() calls (default 12.0 -> ~5
calls/min; a typical extraction call costs roughly 0.6-1.4k Groq tokens
including low-effort reasoning, staying under the 8000 tokens/min
free-tier cap). Transient 429s retry inside dspm.extractor (2s/4s
backoff) and appear as llm_errors/llm_retries in stats -- raise --pace
if those are nonzero. --pace 0 disables sleeping.

Nothing executes at import time (pytest-safe): all work lives in
functions called from main() under if __name__ == "__main__".
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

MODEL = "openai/gpt-oss-20b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
TOTAL_TURNS = 48
PROBE_TURNS = [12, 24, 36, 48]
SEED = 42
ARMS = [120, 60]
DEFAULT_PACE = 12.0
DEFAULT_OUT_DIR = "diag_out"

FINAL_QUERY = "What are the current alerting rules, limits, and routing for this workspace?"

# --- conversation (verbatim from notebook Cell 4) ---------------------------
# NOTE: entries 0, 1, 5 and 11 still carry the literal placeholder markers
# from the chat paste. Replace each "<<PASTE FULL LINE FROM NOTEBOOK>>"
# with the full line from your notebook BEFORE running; main() refuses to
# start while any marker remains (it prints the offending indices).
CORE = [
    ("user", "We're configuring the alerting pipeline for our Growth plan. Hard rules: 50 alerts per month are included, and alert delivery must complete within 2 seconds."),
    ("assistant", "Growth plan confirmed. 50 alerts per month included and a 2-second delivery target noted. Slack and PagerDuty are the supported destinations."),
    ("user", "Connect the Slack workspace acme-ops. Failed alert deliveries must retry automatically."),
    ("assistant", "Slack workspace acme-ops connected. Automatic retries on delivery failure are enabled."),
    ("user", "Set the retry cap to 3 attempts and the escalation timeout to 300 seconds."),
    ("assistant", "Done: retry cap is 3 attempts, escalation timeout 300 seconds. Old defaults were 5 attempts and 600 seconds."),
    ("user", "Change the escalation timeout to 120 seconds -- 300 is too slow for paging."),
    ("assistant", "Escalation timeout updated to 120 seconds. The 300-second setting is superseded."),
    ("user", "Audit retention must not exceed 90 days."),
    ("assistant", "Retention policy set: audit logs are kept at most 90 days."),
    ("user", "One more rule -- PagerDuty routing is only for severity-1 incidents. Everything else routes to Slack."),
    ("assistant", "Routing confirmed: severity-1 incidents go to PagerDuty, all other severities route to Slack. Setup complete."),
]


def build_long_conversation(core, total_turns, seed=SEED):
    rng = random.Random(seed)
    services = ["payments-api", "auth-service", "ingest-worker",
                "search-index", "report-gen", "cdn-edge"]
    conv = list(core)
    i = 0
    while len(conv) < total_turns:
        svc = services[i % len(services)]
        k = i % 6
        if k == 0: u = "[" + svc + "] CPU at " + str(rng.choice([55,61,72,80])) + "%, no action needed."
        elif k == 1: u = "[" + svc + "] deploy v" + str(rng.choice([41,42,43])) + " finished; all health checks green."
        elif k == 2: u = "[" + svc + "] dead-letter queue depth " + str(rng.choice([2,5,9,14])) + "; draining."
        elif k == 3: u = "[" + svc + "] GC pause " + str(rng.choice([40,85,110])) + "ms; within SLO."
        elif k == 4: u = "[" + svc + "] pod restarted after OOM; limit raised to " + str(rng.choice([2,4,6])) + "Gi."
        else: u = "[" + svc + "] disk usage " + str(rng.choice([61,68,74,82])) + "%; cleanup scheduled."
        conv.append(("user", u))
        conv.append(("assistant", "Acknowledged. Noted " + svc + " status; no SLA impact."))
        i += 1
    return conv[:total_turns]


# --- small helpers -----------------------------------------------------------

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _dspm_version() -> str:
    import dspm
    return getattr(dspm, "__version__", "unknown")


def make_client():
    """Groq OpenAI-compatible client. The key comes from the GROQ_API_KEY
    environment variable only -- never hardcoded, never logged, never printed."""
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("ERROR: the 'openai' package is not installed in this venv "
                 "(pip install openai).")
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        sys.exit("ERROR: GROQ_API_KEY is not set. For this session only:\n"
                 '  PowerShell: $env:GROQ_API_KEY = "your-key-here"')
    return OpenAI(base_url=GROQ_BASE_URL, api_key=api_key)


def checkpoint_path(out_dir: Path, budget: int, n: int) -> Path:
    return out_dir / f"b{budget}_cp{n}.json"


def state_path(out_dir: Path, budget: int) -> Path:
    return out_dir / f"state_b{budget}.json"


def failure_path(out_dir: Path, budget: int) -> Path:
    return out_dir / f"b{budget}_failure.json"


def read_json(path: Path):
    """Parsed JSON, or None if the file is missing or unreadable."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        print(f"WARNING: {path.name} exists but cannot be read ({exc!r}) -- treated as absent.")
        return None


def print_stats(stats: dict, budget: int, source: str) -> None:
    print(f"\n--- stats (budget={budget}, {source}) ---")
    for key in sorted(stats):
        marker = "   <-- llm_* telemetry" if key.startswith("llm_") else ""
        print(f"  {key}: {stats[key]}{marker}")
    print()


# --- resume verification -----------------------------------------------------

def _strip_volatile(dump: dict) -> dict:
    """Drop fields describing rendering history rather than memory content:
    dump['context_built'] and each patch's 'last_rendered'."""
    d = dict(dump)
    d.pop("context_built", None)
    d["patches"] = [{k: v for k, v in rec.items() if k != "last_rendered"}
                    for rec in d.get("patches", [])]
    return d


def _first_diff(items_a, items_b, label: str):
    if len(items_a) != len(items_b):
        return f"{label}: {len(items_a)} entries in checkpoint vs {len(items_b)} loaded"
    for idx, (a, b) in enumerate(zip(items_a, items_b)):
        if a != b:
            return f"{label}[{idx}]: checkpoint={a!r} loaded={b!r}"
    return None


def resume_discrepancies(cp_dump: dict, live_dump: dict):
    """Compare a checkpoint's dump_store against the dump of a freshly
    loaded memory. Returns (hard_discrepancies, telemetry_differs).
    Hard discrepancies (turns/budget/patches/journals) mean persistence did
    not round-trip the memory; llm_* telemetry is process-level diagnostics
    and is repairable by the caller."""
    cp = _strip_volatile(cp_dump)
    live = _strip_volatile(live_dump)
    hard = []
    if cp.get("turns") != live.get("turns"):
        hard.append(f"turns: checkpoint={cp.get('turns')} loaded={live.get('turns')}")
    if cp.get("budget") != live.get("budget"):
        hard.append(f"budget: checkpoint={cp.get('budget')} loaded={live.get('budget')}")
    for field in ("patches", "superseded", "deduped"):
        diff = _first_diff(cp.get(field, []), live.get(field, []), field)
        if diff:
            hard.append(diff)
    telemetry_differs = cp.get("llm_telemetry") != live.get("llm_telemetry")
    return hard, telemetry_differs


# --- checkpoint capture -------------------------------------------------------

def capture_checkpoint(mem, out_dir: Path, budget: int, n: int, pace: float) -> Path:
    """Render the probe context, snapshot the store, write the checkpoint
    JSON. get_context() is engine-only: no LLM call, no Groq tokens."""
    context = mem.get_context(query=FINAL_QUERY)
    dump = mem.dump_store(include_history=True)
    stats = mem.stats
    record = {
        "schema": "diag_48/checkpoint/1",
        "budget": budget,
        "checkpoint": n,
        "messages_ingested": n,
        "model": MODEL,
        "dspm_version": _dspm_version(),
        "final_query": FINAL_QUERY,
        "pace_seconds": pace,
        "created_utc": _utc_now(),
        "rendered_context": context,
        "dump_store": dump,
        "stats": stats,
    }
    path = checkpoint_path(out_dir, budget, n)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


def write_failure(out_dir: Path, budget: int, stage: str, i: int,
                  role: str, text: str, exc: Exception, mem) -> None:
    """Journal a strict_llm/engine failure for one arm; never silent."""
    record = {
        "schema": "diag_48/failure/1",
        "budget": budget,
        "stage": stage,                # "add_turn" or "checkpoint"
        "turn_index": i,               # 0-based index into CONVERSATION
        "message_count": i + 1,
        "role": role,
        "message": text,
        "exception_type": type(exc).__name__,
        "exception_text": str(exc),
        "stats_so_far": mem.stats if mem is not None else None,
        "active_payloads": [p.payload for p in mem.patches] if mem is not None else [],
        "created_utc": _utc_now(),
        "resume_hint": ("rerun this script with the same --out-dir; the arm "
                        "resumes from the last persisted checkpoint state file"),
    }
    path = failure_path(out_dir, budget)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"  FAILURE recorded -> {path.name}")
    print(f"  exception: {type(exc).__name__}: {exc}")
    if mem is not None:
        print_stats(mem.stats, budget, "at failure")


# --- one budget arm ------------------------------------------------------------

def run_arm(budget: int, conversation, out_dir: Path, pace: float, feedback: bool = True,
            echo_guard: bool = False, fence: bool = False) -> str:
    """Run one budget arm end-to-end. Returns 'complete', 'failed', or
    'aborted' (resume verification failed -- no quota spent)."""
    from dspm.memory import DSPMMemory

    print(f"\n=== budget={budget} ===")

    cp48_path = checkpoint_path(out_dir, budget, 48)
    finished = read_json(cp48_path)
    if finished is not None:
        print(f"[b{budget}] {cp48_path.name} present and valid -- arm already complete, skipping.")
        if isinstance(finished.get("stats"), dict):
            print_stats(finished["stats"], budget, f"from {cp48_path.name}")
        return "complete"

    client = make_client()
    # Constructed identically on fresh runs and resume (load() merges saved
    # state into this instance).
    mem = DSPMMemory(budget=budget, llm_client=client, model=MODEL, strict_llm=True,
                     feedback_context=feedback, echo_guard=echo_guard,
                     feedback_fence=fence)
    state_file = state_path(out_dir, budget)
    start_index = 0
    resumed = False

    if state_file.exists():
        print(f"[b{budget}] state file {state_file.name} found -- attempting load()-based resume ...")
        try:
            mem.load(str(state_file))
        except Exception as exc:
            print(f"[b{budget}] mem.load() raised {type(exc).__name__}: {exc}")
            print(f"[b{budget}] Cannot resume. To rerun this arm from scratch, delete "
                  f"{state_file.name} and this arm's b{budget}_cp*.json files. "
                  "No quota has been spent.")
            return "aborted"
        live = mem.dump_store(include_history=True)
        turns = live.get("turns")
        if turns not in PROBE_TURNS:
            print(f"[b{budget}] loaded state reports turns={turns!r}, not a checkpoint "
                  f"boundary in {PROBE_TURNS} -- refusing to resume.")
            print(f"[b{budget}] Delete {state_file.name} (and b{budget}_cp*.json) to rerun from scratch.")
            return "aborted"
        cp_record = read_json(checkpoint_path(out_dir, budget, turns))
        if cp_record is None or not isinstance(cp_record.get("dump_store"), dict):
            print(f"[b{budget}] checkpoint JSON at message {turns} is missing/unreadable -- "
                  "cannot verify the loaded state.")
            print(f"[b{budget}] Delete {state_file.name} (and b{budget}_cp*.json) to rerun from scratch.")
            return "aborted"
        hard, telemetry_differs = resume_discrepancies(cp_record["dump_store"], live)
        if hard:
            print(f"[b{budget}] mem.load() did NOT round-trip the memory -- refusing to resume. Differences:")
            for d in hard:
                print(f"    - {d}")
            print(f"[b{budget}] Delete {state_file.name} (and b{budget}_cp*.json) to rerun from scratch.")
            return "aborted"
        if telemetry_differs:
            # Process-level diagnostics, not memory state: adopt the values the
            # checkpoint recorded so resumed stats stay cumulative.
            saved_tel = dict(cp_record["dump_store"].get("llm_telemetry", {}))
            mem._llm_telemetry.clear()
            mem._llm_telemetry.update(saved_tel)
            print(f"[b{budget}] llm_* telemetry adopted from checkpoint JSON "
                  "(persistence did not round-trip it).")
        # Restore the exact post-get_context state an uninterrupted run holds
        # at this boundary (plain attributes read in memory.py):
        mem._last_context = cp_record.get("rendered_context", "")
        mem._context_built = True
        # add_turn() accumulated count_tokens(text) per ingested message:
        from dspm.patch import count_tokens
        mem._conversation_tokens = sum(count_tokens(t) for _, t in conversation[:turns])
        start_index = int(turns)
        resumed = True
        stale = [n for n in PROBE_TURNS
                 if n > start_index and checkpoint_path(out_dir, budget, n).exists()]
        if stale:
            print(f"[b{budget}] stale checkpoint JSON(s) {stale} from an interrupted run "
                  "will be overwritten as this run passes them.")
        print(f"[b{budget}] resume VERIFIED against b{budget}_cp{turns}.json -- continuing "
              f"from message {start_index + 1}/{TOTAL_TURNS}.")
        print(f"[b{budget}] caveat (engine.py not provided, unverifiable): engine EMA state "
              "may not survive save/load; later compressions could differ slightly "
              "from an uninterrupted run.")
    else:
        existing = [n for n in PROBE_TURNS if checkpoint_path(out_dir, budget, n).exists()]
        if existing:
            print(f"[b{budget}] no state file; stale checkpoint JSON(s) {existing} will be "
                  "overwritten as the fresh run passes them.")

    print(f"[b{budget}] ingesting messages {start_index + 1}..{TOTAL_TURNS} at "
          f"pace={pace}s {'(resumed)' if resumed else '(fresh run)'}")

    i = start_index
    while i < TOTAL_TURNS:
        role, text = conversation[i]
        preview = text if len(text) <= 72 else text[:69] + "..."
        print(f"[b{budget}] msg {i + 1:>2}/{TOTAL_TURNS} ({role}): {preview}")
        try:
            mem.add_turn(role, text)
        except Exception as exc:
            write_failure(out_dir, budget, "add_turn", i, role, text, exc, mem)
            return "failed"
        if pace > 0 and i + 1 < TOTAL_TURNS:
            time.sleep(pace)
        if (i + 1) in PROBE_TURNS:
            try:
                path = capture_checkpoint(mem, out_dir, budget, i + 1, pace)
                print(f"[b{budget}] checkpoint {i + 1} saved -> {path.name}")
            except Exception as exc:
                write_failure(out_dir, budget, "checkpoint", i, role, text, exc, mem)
                return "failed"
            try:
                mem.save(str(state_file))
            except Exception as exc:
                print(f"[b{budget}] WARNING: mem.save() failed ({type(exc).__name__}: {exc}) "
                      "-- mid-arm resume unavailable for this arm; the run continues.")
        i += 1

    stale_failure = failure_path(out_dir, budget)
    if stale_failure.exists():
        stale_failure.unlink()
        print(f"[b{budget}] removed stale failure record {stale_failure.name} (arm completed).")
    print_stats(mem.stats, budget, "final")
    return "complete"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="48-message DSPM diagnostic (budgets 120/60): checkpointed "
                    "context/store/stats JSONs under diag_out/.")
    parser.add_argument("--pace", type=float, default=DEFAULT_PACE,
                        help=f"seconds slept between add_turn LLM calls "
                             f"(default {DEFAULT_PACE}). ~5 calls/min keeps a typical "
                             "0.6-1.4k-token extraction call under Groq's 8000 "
                             "tokens/min free-tier cap; raise it if llm_errors/"
                             "llm_retries in stats are nonzero (429s); 0 disables.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                        help=f"output directory (default {DEFAULT_OUT_DIR})")
    parser.add_argument("--no-feedback", action="store_true",
                        help="diagnostic: construct DSPMMemory with feedback_context=False")
    parser.add_argument("--echo-guard", action="store_true",
                        help="diagnostic: DSPMMemory(echo_guard=True)")
    parser.add_argument("--fence", action="store_true",
                        help="diagnostic: DSPMMemory(feedback_fence=True)")
    args = parser.parse_args(argv)

    # Windows console safety: a unicode payload must never crash a print.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    conversation = build_long_conversation(CORE, TOTAL_TURNS)
    if len(conversation) != TOTAL_TURNS:
        print(f"ERROR: build_long_conversation produced {len(conversation)} messages, "
              f"expected {TOTAL_TURNS}.")
        return 2

    placeholders = [idx for idx, (_role, text) in enumerate(CORE)
                    if "<<" in text or ">>" in text]
    if placeholders:
        print(f"ERROR: CORE still contains chat-paste placeholders at indices {placeholders}.")
        print("       Edit CORE in scripts/diag_48.py and replace each")
        print("       <<PASTE FULL LINE FROM NOTEBOOK>> marker with the full line from")
        print("       your notebook, then rerun. Refusing to spend Groq quota on a")
        print("       corrupted conversation.")
        return 2

    if not os.environ.get("GROQ_API_KEY"):
        sys.exit("ERROR: GROQ_API_KEY is not set. For this session only:\n"
                 '  PowerShell: $env:GROQ_API_KEY = "your-key-here"\n'
                 "The key is never hardcoded, logged, or printed by this script.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"diag_48 | dspm={_dspm_version()} | model={MODEL} | budgets={ARMS} | "
          f"pace={args.pace}s | out_dir={out_dir} | started={_utc_now()}")
    print(f"conversation: {len(conversation)} messages (12 CORE + "
          f"{len(conversation) - 12} filler), checkpoints at {PROBE_TURNS}")
    if args.pace > 0:
        rough = int(((args.pace + 4.0) * 2 * TOTAL_TURNS) / 60) + 1
        print(f"rough wall-time if both arms run from scratch: ~{rough} min, plus a "
              "one-time local engine warmup at the first get_context (no Groq tokens)")

    statuses = {}
    for budget in ARMS:
        statuses[budget] = run_arm(budget, conversation, out_dir, args.pace,
                                   not args.no_feedback, args.echo_guard, args.fence)

    print("\n=== run summary ===")
    for budget in ARMS:
        print(f"  budget={budget}: {statuses[budget]}")
    if all(s == "complete" for s in statuses.values()):
        print(f"\nAll checkpoint JSONs are under {out_dir}.")
        print("Next: python scripts\\analyze_diag.py --out-dir " + str(out_dir))
        return 0
    print("\nSome arms did not complete this run. Rerun this script to resume each "
          "arm from its last verified checkpoint.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
