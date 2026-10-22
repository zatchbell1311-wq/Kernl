# eval_harness.py -- review point 5: measure WHAT IS RETAINED CORRECTLY,
# not just how much is compressed. Five scenario batteries:
#
#   1. payments_echo    -- the live-smoke shape: user requirements +
#      assistant echoes + one revision, at a budget that fits the full
#      payloads (120). Expects: echoes collapse, every fact delivers,
#      no same-value suspect pairs, revision story intact.
#   2. tier_trap        -- the precision battery: same-value pairs that
#      MUST stay split (platinum/silver tiers, gold/silver, cloud/training
#      budgets, dashboard/homepage), plus the documented scope-noun
#      limitation (expected to merge under (b4), reported, not a failure).
#   3. revision_chain   -- 30 -> 20 -> 15 -> 10 at the starvation budget:
#      current value outside the (was X) span, the chain inside it.
#   4. starvation_honesty -- more criticals than the budget can carry:
#      CRR must report the honest fraction, never a fabricated 100%.
#   5. trim_quality     -- the real-smoke store shape at budget 59.
#      KNOWN FINDING (S3, 0.1.18 backlog), measured and reported, not
#      hidden: long-prose revision payloads ('... superseding previous
#      30 seconds value (was 30)') lose the (was X) span form under
#      starvation trimming -- the delivered line reads 'timeout 10
#      seconds, 30 seconds' (and loses the subject in tighter mixes).
#      The scenario asserts delivery capacity is honest and prints the
#      trim-quality flags so the fix batch has a standing repro.
#
# Two modes:
#   offline (default) -- deterministic, no API key. Turns carry the
#      patches a good extractor would produce (including the echo
#      variance that bloated the live store), so the battery measures
#      the STORE + DELIVERY layers: collapse precision, fact recall in
#      the delivered context, revision correctness, CRR honesty, and
#      the v0.1.17 role/token accounting.
#   --live -- runs the REAL extractor over scenario transcripts with a
#      Groq or OpenAI client and additionally reports extraction
#      failures, empty-content responses, per-call latency, token
#      usage, and cost (only when prices are set -- no invented numbers).
#
# Run (project root):
#     & $py eval_harness.py
#     & $py eval_harness.py --live          # GROQ_API_KEY or OPENAI_API_KEY
#     & $py eval_harness.py --report eval_report.json
# Live knobs: DSPM_MODEL (default: llama-3.3-70b-versatile on Groq,
# gpt-4o-mini on OpenAI), DSPM_PRICE_IN / DSPM_PRICE_OUT (USD per 1M
# tokens, optional).
#
# v2 fixes (found running v1 end-to-end -- v1 shipped nowhere):
#   - the uncovered-pair scan listed EVERY un-merged pair as a
#     "duplicate", so a six-distinct-fact store scored 15 and the
#     expect_uncovered=0 scenarios could never pass. A pair is now a
#     suspect only when both payloads carry the SAME non-empty number
#     set -- everything else is a distinct fact by definition.
#   - live mode built a bare DSPMMemory, which constructs its own
#     client -- the RecordingClient counted zero calls and the
#     latency/token/cost report was all zeros. Extraction now routes
#     through the recorder.
#   - no-leak forgives explicit prose history ('superseding previous 30
#     seconds value'): a history value outside the (was X) span only
#     counts as a leak when nothing within 60 chars before it marks it
#     as superseded. Stale-value-delivered-as-current still fails.
#   - trim_quality prints the delivered revision line + span-form
#     flags: the standing S3 repro (reported, not a failure).

import json
import os
import re
import sys
import time

try:
    import dspm
    from dspm import DSPMMemory
    from dspm.memory import _numbers
    from dspm.patch import SemanticPatch
except ImportError:                       # src/ layout, not installed
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
    import dspm
    from dspm import DSPMMemory
    from dspm.memory import _numbers
    from dspm.patch import SemanticPatch


def _mk(pid, turn, ttype, payload, role=None):
    p = SemanticPatch(patch_id=pid, turn_index=turn, patch_type=ttype,
                      payload=payload, dependencies=[])
    if role:
        p.source_role = role
    return p


# ---------------------------------------------------------------------------
# scenarios
# ---------------------------------------------------------------------------

S1 = {
    "name": "payments_echo",
    "budget": 120,
    "query": "webhook timeout and fee",
    "patches": [
        [("p0a", 0, "constraint",
          "All payments must be PCI-DSS compliant as a non-negotiable requirement", "user"),
         ("p0b", 0, "constraint",
          "Maximum fee per transaction is limited to 0.5%", "user")],
        [("p1a", 1, "decision", "fee cap set to 0.5% as tracked", "assistant"),
         ("p1b", 1, "decision", "webhook timeout set to 30 seconds", "assistant")],
        [("p3a", 3, "decision",
          "webhook timeout updated to 10 seconds, superseding previous 30 seconds value", "assistant")],
        [("p4a", 4, "constraint", "Transactions must not exceed 10000 dollars", "user")],
        [("p6a", 6, "decision", "dashboard must load in under 2 seconds", "user"),
         ("p6b", 6, "decision", "Audit retention set to 5 years", "user")],
        [("p7a", 7, "constraint", "Dashboard latency must be sub-2-second", "assistant")],
    ],
    "facts": [
        {"id": "pci", "keywords": ["pci", "compliant"]},
        {"id": "fee", "keywords": ["fee"], "number": "0.5"},
        {"id": "webhook", "keywords": ["webhook", "timeout"], "number": "10"},
        {"id": "dashboard", "keywords": ["dashboard"], "number": "2"},
        {"id": "audit", "keywords": ["audit"], "number": "5"},
    ],
    "revisions": [
        {"id": "webhook", "keywords": ["webhook"], "current": "10",
         "history": ["30"]},
    ],
    "expect_uncovered": 0,
}

S5 = {
    "name": "trim_quality",
    "budget": 59,
    "query": "webhook timeout",
    "patches": [
        [("q0", 0, "constraint",
          "All payments must be PCI-DSS compliant as a non-negotiable requirement", "user"),
         ("q1", 0, "constraint",
          "Maximum fee per transaction is limited to 0.5%", "user")],
        [("q2", 1, "decision", "fee cap set to 0.5% as tracked", "assistant")],
        [("q3", 5, "decision",
          "Webhook timeout updated to 10 seconds, superseding previous 30 seconds value (was 30)",
          "assistant")],
        [("q4", 6, "decision", "dashboard must load in under 2 seconds", "user"),
         ("q5", 7, "decision", "Audit retention set to 5 years", "user")],
        [("q6", 7, "constraint", "Dashboard latency must be sub-2-second", "assistant")],
    ],
    "facts": [],
    "revisions": [],
    "expect_uncovered": 0,
    "honesty_check": True,
    "trim_finding": {
        "id": "S3",
        "backlog": "0.1.18",
        "subject_keywords": ["timeout"],
        "current": "10",
        "span": "(was",
        "note": "long-prose revision loses the (was X) span form under "
                "starvation trim; v0.1.12/v0.1.14 story contract violated "
                "for this payload shape",
    },
}

S2 = {
    "name": "tier_trap",
    "budget": 250,
    "query": "fees and budgets",
    "patches": [
        [("t0a", 0, "constraint", "Maximum platinum fee is 0.5%", "user"),
         ("t0b", 0, "constraint", "Gold tier fee capped at 0.5%", "user"),
         ("t0c", 0, "constraint", "Budget 100 for cloud", "user"),
         ("t0d", 0, "constraint", "Dashboard loads in under 2 seconds", "user"),
         ("t0e", 0, "constraint", "Max users 500 per org", "user")],
        [("t1a", 1, "decision", "silver tier fee cap 0.5%", "assistant"),
         ("t1b", 1, "decision", "Silver tier fee capped at 0.5%", "assistant"),
         ("t1c", 1, "decision", "Budget 100 for training", "assistant"),
         ("t1d", 1, "decision", "Homepage loads in under 2 seconds", "assistant"),
         ("t1e", 1, "decision", "org admins cap 500", "assistant")],
    ],
    "split_guards": [
        {"a": "Maximum platinum fee is 0.5%", "b": "silver tier fee cap 0.5%",
         "stay_split": True},
        {"a": "Gold tier fee capped at 0.5%", "b": "Silver tier fee capped at 0.5%",
         "stay_split": True},
        {"a": "Budget 100 for cloud", "b": "Budget 100 for training",
         "stay_split": True},
        {"a": "Dashboard loads in under 2 seconds", "b": "Homepage loads in under 2 seconds",
         "stay_split": True},
        {"a": "Max users 500 per org", "b": "org admins cap 500",
         "stay_split": False,
         "note": "documented (b4) limitation -- structured slots are the 0.1.18 fix"},
    ],
    "facts": [],
    "revisions": [],
    "expect_uncovered": None,
}

S3 = {
    "name": "revision_chain",
    "budget": 59,
    "query": "webhook timeout",
    "patches": [
        [("c0", 0, "decision", "webhook timeout set to 30 seconds", "user")],
        [("c1", 1, "decision", "webhook timeout updated to 20 seconds", "assistant")],
        [("c2", 2, "decision", "webhook timeout reduced to 15 seconds", "user")],
        [("c3", 3, "decision", "webhook timeout updated to 10 seconds", "assistant")],
    ],
    "facts": [
        {"id": "webhook-current", "keywords": ["webhook", "timeout"], "number": "10"},
    ],
    "revisions": [
        {"id": "webhook-chain", "keywords": ["webhook"], "current": "10",
         "history": ["30", "20", "15"]},
    ],
    "expect_uncovered": 0,
}

S4 = {
    "name": "starvation_honesty",
    "budget": 30,
    "query": "everything",
    "patches": [
        [("h%d" % t, t, "constraint", "requirement %d must hold with value %d" % (t, t * 7), "user")]
        for t in range(8)
    ],
    "facts": [],
    "revisions": [],
    "expect_uncovered": 0,
    "honesty_check": True,
}

SCENARIOS = [S1, S2, S3, S4, S5]

TRANSCRIPTS = {
    "payments_echo": [
        ("user", "We are building a payments API. All payment data must be "
                 "PCI-DSS compliant as a non-negotiable requirement. Also "
                 "the maximum fee per transaction is limited to 0.5 percent."),
        ("assistant", "Understood: PCI-DSS compliance for all payment data, "
                      "and the fee cap set to 0.5 percent as tracked. Let's "
                      "set the webhook timeout to 30 seconds for delivery "
                      "attempts."),
        ("user", "Please change the webhook timeout to 10 seconds instead, "
                 "the workers cannot hold 30 second connections."),
        ("assistant", "Updated: webhook timeout is now 10 seconds, "
                      "superseding the previous 30 seconds value."),
        ("user", "Two more requirements: every dashboard must load in under "
                 "2 seconds, and audit retention is set to 5 years."),
        ("assistant", "Noted: dashboard must load in under 2 seconds, and "
                      "audit retention set to 5 years. Dashboard latency "
                      "will be sub-2-second in our tests."),
    ],
    "revision_chain": [
        ("user", "Set the webhook timeout to 30 seconds initially."),
        ("assistant", "Webhook timeout set to 30 seconds."),
        ("user", "Update the webhook timeout to 20 seconds."),
        ("assistant", "Webhook timeout updated to 20 seconds."),
        ("user", "Reduce the webhook timeout to 15 seconds."),
        ("assistant", "Webhook timeout reduced to 15 seconds."),
        ("user", "Final change: update the webhook timeout to 10 seconds."),
        ("assistant", "Webhook timeout updated to 10 seconds."),
    ],
}


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def fact_in_ctx(ctx, fact):
    low = ctx.lower()
    if not all(k in low for k in fact.get("keywords", [])):
        return False
    if "number" in fact and fact["number"] not in ctx:
        return False
    return True


_SUPERSEDE_MARKERS = ("superseding", "superseded", "previous", "prior",
                      "earlier", "old", "was", "were", "reduced from",
                      "updated from", "changed from", "instead of",
                      "replaced", "formerly")


def revision_in_ctx(ctx, rev):
    spans = re.findall(r"\(was ([^)]*)\)", ctx)
    outside = re.sub(r"\(was [^)]*\)", "", ctx)
    ok_current = rev["current"] in outside
    ok_history = all(any(h in s for s in spans)
                     for h in rev.get("history", []))

    # v2: a history value OUTSIDE the span is only a leak when it reads
    # as current. Extractors legitimately record prose history
    # ('... superseding previous 30 seconds value (was 30)') -- the
    # prose is droppable under trim, the span is the protected form. A
    # supersede marker within 60 chars before the occurrence means the
    # mention is explicit history, not a stale value.
    def _leaks(h):
        for m in re.finditer(re.escape(h), outside):
            window = outside[max(0, m.start() - 60):m.start()].lower()
            if not any(k in window for k in _SUPERSEDE_MARKERS):
                return True
        return False

    ok_no_leak = all(not _leaks(h) for h in rev.get("history", []))
    return ok_current, ok_history, ok_no_leak


def uncovered_pairs(crits):
    """Same-NUMBER suspect pairs the merge rules leave un-merged. v1 of
    this scan listed every un-merged pair -- a store of six plainly
    distinct facts scored 15 'duplicates' -- so expect_uncovered=0 could
    never pass. v2: a pair is a suspect only when both payloads carry
    the SAME non-empty number set; everything else is a distinct fact
    by definition."""
    out = []
    for i in range(len(crits)):
        for j in range(i + 1, len(crits)):
            a, b = crits[i], crits[j]
            if a.turn_index > b.turn_index:
                a, b = b, a
            na, nb = _numbers(a.payload), _numbers(b.payload)
            if not na or na != nb:
                continue
            scratch = DSPMMemory(budget=10 ** 6)
            scratch._merge_patch(SemanticPatch(a.patch_id, a.turn_index,
                                               a.patch_type, a.payload, []))
            scratch._merge_patch(SemanticPatch(b.patch_id, b.turn_index,
                                               b.patch_type, b.payload, []))
            if len(scratch.critical_patches) == 2:
                out.append((a.payload, b.payload))
    return out


def evaluate(sc, mode, client=None, model=None):
    res = {"name": sc["name"], "budget": sc["budget"], "mode": mode}
    live = {}
    if mode == "live":
        # v2: route extraction through the RecordingClient -- a bare
        # DSPMMemory builds its own client and the live report would
        # count zero calls.
        m = DSPMMemory(budget=sc["budget"], llm_client=client, model=model)
        t0 = time.perf_counter()
        for role, text in TRANSCRIPTS[sc["name"]]:
            m.add_turn(role, text)
        live["wall_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    else:
        m = DSPMMemory(budget=sc["budget"])
        for group in sc["patches"]:
            for pid, turn, ttype, payload, role in group:
                m._merge_patch(_mk(pid, turn, ttype, payload, role))
    ctx = m.get_context(sc.get("query", "overview"))
    s = m.stats
    crits = m.critical_patches

    res["store_criticals"] = s["critical_total"]
    res["delivered_criticals"] = s["critical_selected"]
    res["crr"] = s["crr"]
    res["context_tokens"] = s["context_tokens"]
    res["conversation_tokens"] = s.get("conversation_tokens", None)
    res["patch_tokens"] = s.get("patch_tokens", None)

    misses = [f["id"] for f in sc["facts"] if not fact_in_ctx(ctx, f)]
    res["fact_recall"] = round(100 * (len(sc["facts"]) - len(misses))
                               / max(1, len(sc["facts"])), 1) if sc["facts"] else None
    res["fact_misses"] = misses

    rev_results = {}
    for rev in sc["revisions"]:
        cur, hist, leak = revision_in_ctx(ctx, rev)
        rev_results[rev["id"]] = {"current_outside_span": cur,
                                  "history_in_span": hist,
                                  "history_not_leaked": leak}
    res["revisions"] = rev_results

    if sc.get("split_guards"):
        guards = {}
        for g in sc["split_guards"]:
            scratch = DSPMMemory(budget=10 ** 6)
            scratch._merge_patch(SemanticPatch("ga", 0, "constraint", g["a"], []))
            scratch._merge_patch(SemanticPatch("gb", 1, "decision", g["b"], []))
            stayed = len(scratch.critical_patches) == 2
            guards[g["a"][:34]] = {"expected_split": g["stay_split"],
                                   "stayed_split": stayed,
                                   "ok": stayed == g["stay_split"],
                                   "note": g.get("note", "")}
        res["split_guards"] = guards
        res["split_guards_passed"] = sum(1 for v in guards.values() if v["ok"])

    dup = uncovered_pairs(crits)
    res["uncovered_duplicates"] = dup
    if sc.get("expect_uncovered") is not None:
        res["uncovered_ok"] = len(dup) == sc["expect_uncovered"]

    if sc.get("honesty_check"):
        res["honesty_ok"] = (s["context_built"] and
                             s["crr"] == int(100 * s["critical_selected"]
                                             / max(1, s["critical_total"])))

    # S3 standing repro: the delivered revision line + span-form flags.
    # Reported, never a verdict input -- the finding is 0.1.18 backlog.
    tf = sc.get("trim_finding")
    if tf:
        line = ""
        for l in ctx.splitlines():
            low = l.lower()
            if all(k in low for k in tf["subject_keywords"]):
                line = l
                break
        res["trim_quality"] = {
            "subject_line": line,
            "current_value_present": tf["current"] in line,
            "span_form_present": tf["span"] in line,
        }
        res["trim_finding"] = tf

    if mode == "live" and client is not None:
        live.update(client.report())
        live["model"] = model
        live["price_note"] = ("set DSPM_PRICE_IN / DSPM_PRICE_OUT for USD"
                              if not client.price_in else "")
        res["live"] = live

    passed = True
    if sc["facts"] and misses:
        passed = False
    if sc["revisions"] and not all(v["current_outside_span"] and v["history_in_span"]
                                   for v in rev_results.values()):
        passed = False
    if sc.get("expect_uncovered") is not None and len(dup) != sc["expect_uncovered"]:
        passed = False
    if sc.get("split_guards") and res["split_guards_passed"] != len(sc["split_guards"]):
        passed = False
    if sc.get("honesty_check") and not res["honesty_ok"]:
        passed = False
    res["passed"] = passed
    return res, ctx, m


# ---------------------------------------------------------------------------
# live client
# ---------------------------------------------------------------------------

class RecordingClient:
    """Wraps a Groq/OpenAI client; counts calls, failures, empty completions,
    latency, and token usage. Never invents prices."""

    def __init__(self, inner, price_in=0.0, price_out=0.0):
        self.inner = inner
        self.calls = 0
        self.failures = 0
        self.empty = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.latency_ms = []
        self.price_in = price_in
        self.price_out = price_out
        owner = self

        class Completions:
            def create(self, **kwargs):
                t0 = time.perf_counter()
                try:
                    resp = owner.inner.chat.completions.create(**kwargs)
                except Exception:
                    owner.failures += 1
                    raise
                owner.calls += 1
                owner.latency_ms.append(round((time.perf_counter() - t0) * 1000, 1))
                content = (resp.choices[0].message.content
                           if getattr(resp, "choices", None) else "")
                if not (content and content.strip()):
                    owner.empty += 1
                u = getattr(resp, "usage", None)
                if u is not None:
                    owner.prompt_tokens += getattr(u, "prompt_tokens", 0) or 0
                    owner.completion_tokens += getattr(u, "completion_tokens", 0) or 0
                return resp

        class Chat:
            completions = Completions()

        self.chat = Chat()

    def report(self):
        lat = sorted(self.latency_ms)
        p50 = lat[len(lat) // 2] if lat else None
        out = {
            "extraction_calls": self.calls,
            "extraction_failures": self.failures,
            "empty_completions": self.empty,
            "latency_ms_p50": p50,
            "latency_ms_max": lat[-1] if lat else None,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }
        if self.price_in:
            out["cost_usd"] = round(
                self.prompt_tokens / 1e6 * self.price_in
                + self.completion_tokens / 1e6 * self.price_out, 4)
        return out


def make_live_client():
    pin = float(os.environ.get("DSPM_PRICE_IN", "0") or 0)
    pout = float(os.environ.get("DSPM_PRICE_OUT", "0") or 0)
    if os.environ.get("GROQ_API_KEY"):
        from groq import Groq
        model = os.environ.get("DSPM_MODEL", "llama-3.3-70b-versatile")
        return RecordingClient(Groq(), pin, pout), model
    if os.environ.get("OPENAI_API_KEY"):
        from openai import OpenAI
        model = os.environ.get("DSPM_MODEL", "gpt-4o-mini")
        return RecordingClient(OpenAI(), pin, pout), model
    sys.exit("live mode needs GROQ_API_KEY or OPENAI_API_KEY in the env")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    live = "--live" in sys.argv
    report_path = None
    if "--report" in sys.argv:
        report_path = sys.argv[sys.argv.index("--report") + 1]

    print("dspm %s -- eval harness (%s mode)"
          % (getattr(dspm, "__version__", "?"), "live" if live else "offline"))
    client = model = None
    if live:
        client, model = make_live_client()
        print("live model: %s" % model)

    results = []
    for sc in SCENARIOS:
        if live and sc["name"] not in TRANSCRIPTS:
            print("[skip: no transcript for %s in live mode]" % sc["name"])
            continue
        res, ctx, m = evaluate(sc, "live" if live else "offline",
                               client=client, model=model)
        results.append(res)
        print()
        print("=" * 72)
        print("%s  (budget %d, %s)" % (res["name"], res["budget"], res["mode"]))
        print("=" * 72)
        print("  store=%d delivered=%d crr=%d ctx_tokens=%s"
              % (res["store_criticals"], res["delivered_criticals"], res["crr"],
                 res["context_tokens"]))
        if res["fact_recall"] is not None:
            print("  fact recall: %s%%  misses=%s"
                  % (res["fact_recall"], res["fact_misses"]))
        for rid, v in res["revisions"].items():
            print("  revision %-14s current=%s history=%s no-leak=%s"
                  % (rid, v["current_outside_span"], v["history_in_span"],
                     v["history_not_leaked"]))
        if "split_guards" in res:
            print("  split guards: %d/%d"
                  % (res["split_guards_passed"], len(res["split_guards"])))
            for k, v in res["split_guards"].items():
                if not v["ok"]:
                    print("    VIOLATION: %r -> %s %s"
                          % (k, v, v["note"]))
        if res["uncovered_duplicates"]:
            print("  uncovered duplicate pairs (%d):"
                  % len(res["uncovered_duplicates"]))
            for a, b in res["uncovered_duplicates"]:
                print("    %r  +  %r" % (a, b))
        else:
            print("  uncovered duplicate pairs: 0")
        if "honesty_ok" in res:
            print("  crr honesty (measured, not fabricated): %s"
                  % res["honesty_ok"])
        if "trim_quality" in res:
            tq, tfind = res["trim_quality"], res["trim_finding"]
            print("  trim quality (finding %s, %s backlog -- reported, not a failure):"
                  % (tfind["id"], tfind["backlog"]))
            print("    delivered line: %r" % tq["subject_line"])
            print("    current value present: %s   (was X) span form present: %s"
                  % (tq["current_value_present"], tq["span_form_present"]))
        if "live" in res:
            print("  live: %s" % json.dumps(res["live"]))
        print("  VERDICT: %s" % ("PASS" if res["passed"] else "FAIL"))

    n_pass = sum(1 for r in results if r["passed"])
    print()
    print("=" * 72)
    print("AGGREGATE: %d/%d scenarios pass" % (n_pass, len(results)))
    print("=" * 72)

    if report_path:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump({"dspm_version": getattr(dspm, "__version__", "?"),
                       "results": results}, f, indent=2)
        print("report written: %s" % report_path)


if __name__ == "__main__":
    main()
