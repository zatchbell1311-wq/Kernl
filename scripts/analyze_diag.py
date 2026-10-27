"""UNTESTED -- offline analyzer for the JSON files written by scripts/diag_48.py.

No network, no dspm import: reads only diag_out/b{budget}_cp{n}.json
(budgets 120/60, checkpoints 12/24/36/48) plus any b{budget}_failure.json
records, and prints:

* per budget, per checkpoint, per ground-truth fact, exactly one of
      both          -- matched an active payload AND a rendered-context line
      store only    -- matched an active payload but NOT the rendered
                       context ("didn't surface")
      context only  -- matched the rendered context but no active payload
      neither       -- matched neither ("forgot"; superseded-journal matches
                       are annotated when found)
  using the notebook's scorer verbatim (_norm/_tok_match/line_matches with
  each fact's "must" tokens) applied to active dump_store payloads and
  rendered-context lines;
* the active payload text for 'retention_bound' and 'escalation_rev' at
  every checkpoint;
* superseded and deduped journal entries (counts and kinds);
* the llm_* telemetry counters;
* a closing one-paragraph plain summary per budget, labeled as an
  observation, not a conclusion.

Known limitation, stated up front: deduped journal entries record ids and
kinds only (the shape of dspm.memory._dedup_log) -- no payload text -- so a
fact lost to a merge rule cannot be confirmed from the journal alone.
'neither' therefore means "not in the active store", which is compatible
with both "never extracted" and "collapsed by a merge rule"; a superseded-
journal annotation narrows it further when present.

Usage:  python scripts\analyze_diag.py [--out-dir diag_out]
Nothing executes at import time.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from collections import Counter
from pathlib import Path

PROBE_TURNS = [12, 24, 36, 48]
BUDGETS = [120, 60]

GROUND_TRUTH = [
    dict(id="alert_cap", cls="numeric", must=["50", "alert"]),
    dict(id="latency_bound", cls="bound", must=["2", "second"]),
    dict(id="retry_obligation", cls="obligation", must=["retr", "fail"]),
    dict(id="retry_cap", cls="numeric", must=["3", "attempt"]),
    dict(id="escalation_rev", cls="revision", must=["120", "second"]),
    dict(id="retention_bound", cls="negation", must=["90", "day"]),
    dict(id="routing", cls="scope", must=["severity", "1", "pagerduty"]),
    dict(id="plan", cls="plain", must=["growth", "plan"]),
]

LLM_KEYS = ["llm_calls", "llm_empty", "llm_errors", "llm_retries",
            "llm_length_finish", "llm_parse_failures", "llm_fallbacks"]

SPOTLIGHT_FACTS = ["retention_bound", "escalation_rev"]


# --- notebook scorer, verbatim (Cell 4) --------------------------------------

def _norm(s):
    return re.sub(r"[^a-z0-9\s]", " ", s.lower())


def _tok_match(must, tokens):
    if must.isdigit():
        return any(t == must for t in tokens)
    return any(len(t) >= 3 and (t.startswith(must) or must.startswith(t)) for t in tokens)


def line_matches(line, must_tokens):
    toks = _norm(line).split()
    return all(_tok_match(m, toks) for m in must_tokens)


def bound_wording_kept(ctx):
    for line in ctx.splitlines():
        if line_matches(line, ["90", "day"]):
            w = _norm(line)
            if any(k in w for k in ("not exceed", "at most", "no more", "max",
                                    "maximum", "under", "within", "up to")):
                return True
    return False


# --- helpers -------------------------------------------------------------------

def _clip(text, limit=72):
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _first_match(lines, must):
    for line in lines:
        if line_matches(line, must):
            return line
    return None


def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  [corrupt file: {path.name} ({exc!r}) -- skipped]")
        return None


# --- per-checkpoint report -------------------------------------------------------

def report_checkpoint(budget: int, n: int, record: dict) -> dict:
    ctx = record.get("rendered_context", "") or ""
    dump = record.get("dump_store", {}) or {}
    stats = record.get("stats", {}) or {}
    active = [r.get("payload", "") for r in dump.get("patches", [])]
    ctx_lines = [ln for ln in ctx.splitlines() if ln.strip()]
    superseded = dump.get("superseded", []) or []

    print(f"\n--- budget={budget} | checkpoint {n} "
          f"(messages ingested: {record.get('messages_ingested', n)}) ---")

    rows = []
    for gt in GROUND_TRUTH:
        store_hit = _first_match(active, gt["must"])
        ctx_hit = _first_match(ctx_lines, gt["must"])
        if store_hit and ctx_hit:
            cls = "both"
        elif store_hit:
            cls = "store only (didn't surface)"
        elif ctx_hit:
            cls = "context only"
        else:
            cls = "neither (forgot)"
        evidence = store_hit or ctx_hit
        print(f"  {gt['id']:<18}{cls:<30}{_clip(evidence) if evidence else '-'}")
        if cls == "neither (forgot)":
            sup_hit = _first_match([e.get("payload", "") for e in superseded], gt["must"])
            if sup_hit:
                print(f"  {'':<18}{'':<30}(superseded payload matched: {_clip(sup_hit)})")
        rows.append((gt["id"], cls))

    for fid in SPOTLIGHT_FACTS:
        gt = next(g for g in GROUND_TRUTH if g["id"] == fid)
        hits = [p for p in active if line_matches(p, gt["must"])]
        shown = "; ".join(_clip(h) for h in hits) if hits else "(no active payload matched)"
        print(f"  {fid} active payload(s): {shown}")

    print(f"  retention bound wording kept in context: {bound_wording_kept(ctx)}")

    print(f"  superseded journal: {len(superseded)} "
          f"entr{'y' if len(superseded) == 1 else 'ies'}")
    for e in superseded:
        print(f"    - {e.get('patch_id')} ({e.get('patch_type')}) "
              f"superseded_by={e.get('superseded_by')} "
              f"at_turn={e.get('at_turn')}: {_clip(e.get('payload', ''))}")
    deduped = dump.get("deduped", []) or []
    kinds = Counter(e.get("kind", "unknown") for e in deduped)
    print(f"  deduped journal: {len(deduped)} entries; kinds: {dict(kinds)}")
    for e in deduped:
        print(f"    - kind={e.get('kind')} kept={e.get('kept')} "
              f"dropped={e.get('dropped')} at_turn={e.get('at_turn')}")

    tel = {k: stats.get(k) for k in LLM_KEYS}
    print(f"  llm_* counters: {tel}")

    print("  rendered context (from JSON):")
    if ctx_lines:
        for ln in ctx_lines:
            print(f"    | {ln}")
    else:
        print("    | (empty)")

    return {
        "n": n,
        "cls": {fid: cls for fid, cls in rows},
        "wording": bound_wording_kept(ctx),
        "stats": stats,
        "active": len(active),
        "sup": len(superseded),
        "ded": len(deduped),
    }


def report_failure(out_dir: Path, budget: int) -> None:
    path = out_dir / f"b{budget}_failure.json"
    if not path.exists():
        return
    record = _load_json(path)
    if record is None:
        return
    completed = (out_dir / f"b{budget}_cp48.json").exists()
    suffix = ("  (checkpoint 48 now exists -- the arm completed later; "
              "this record is history)" if completed else "")
    print(f"\n!!! failure record -- budget={budget}{suffix}")
    print(f"    stage={record.get('stage')}  turn_index={record.get('turn_index')}  "
          f"message_count={record.get('message_count')}")
    print(f"    exception: {record.get('exception_type')}: "
          f"{_clip(record.get('exception_text', ''), 120)}")
    stats = record.get("stats_so_far") or {}
    tel = {k: stats.get(k) for k in LLM_KEYS}
    print(f"    llm_* at failure: {tel}")


def summarize_budget(budget: int, checkpoints) -> None:
    print(f"\n=== summary -- budget={budget} ===")
    if not checkpoints:
        print("  (no checkpoint records loaded for this budget -- nothing to summarize)")
        return
    parts = []
    for gt in GROUND_TRUTH:
        fid = gt["id"]
        issues = [(cp["n"], cp["cls"][fid])
                  for cp in checkpoints if cp["cls"].get(fid) != "both"]
        if not issues:
            parts.append(f"'{fid}' matched both the store and the rendered context "
                         "at every loaded checkpoint")
        else:
            detail = ", ".join(f"cp{n}: {cls}" for n, cls in issues)
            parts.append(f"'{fid}' deviated ({detail})")
    last = checkpoints[-1]
    wording = ", ".join(f"cp{cp['n']}: {'kept' if cp['wording'] else 'NOT kept'}"
                        for cp in checkpoints)
    tel = ", ".join(f"{k}={last['stats'].get(k)}" for k in LLM_KEYS)
    paragraph = (
        "Observation (not a conclusion): "
        + ". ".join(parts)
        + f". Retention bound wording in the rendered context: {wording}."
        + f" At the last loaded checkpoint ({last['n']}) the store held "
        f"{last['active']} active patches, with {last['sup']} superseded and "
        f"{last['ded']} deduped journal entries; llm telemetry there: {tel}."
    )
    print(textwrap.fill(paragraph, width=100, initial_indent="  ", subsequent_indent="  "))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline analysis of diag_48.py checkpoint JSONs: fact "
                    "retention/surfacing, journals, llm telemetry.")
    parser.add_argument("--out-dir", default="diag_out",
                        help="directory holding b{budget}_cp{n}.json (default: diag_out)")
    args = parser.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    out_dir = Path(args.out_dir)
    if not out_dir.is_dir():
        print(f"ERROR: {out_dir} does not exist -- run scripts/diag_48.py first.")
        return 2

    print("classification: both | store only (didn't surface) | context only | "
          "neither (forgot)")
    print("matching: notebook scorer line_matches() on each fact's 'must' tokens, "
          "applied to active dump_store payloads and rendered-context lines")
    print("note: deduped journal entries carry ids/kinds only (no payload text), so a "
          "merge-away cannot be confirmed from the journal alone")

    for budget in BUDGETS:
        print(f"\n================ budget={budget} ================")
        report_failure(out_dir, budget)
        checkpoints = []
        for n in PROBE_TURNS:
            path = out_dir / f"b{budget}_cp{n}.json"
            if not path.exists():
                print(f"\n  [missing checkpoint: {path.name} -- skipped]")
                continue
            record = _load_json(path)
            if record is None:
                continue
            checkpoints.append(report_checkpoint(budget, n, record))
        summarize_budget(budget, checkpoints)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())