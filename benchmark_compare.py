#!/usr/bin/env python3
"""DSPM vs rolling-summary retention harness.

MOCK mode (default) is fully deterministic and offline: a canned extractor
(the same MockLLMClient shape used by tests/test_memory.py) and a
string-match judge. It verifies the harness end to end and demonstrates the
guarantee, but the published figure comes from LIVE runs.

LIVE mode uses the real extractor, a true rolling-summary baseline (an LLM
re-summarizes the whole transcript every turn) and an LLM judge, then prints
per-dialogue retention plus mean +/- std across dialogues.

Run:  python benchmark_compare.py --mode mock
      python benchmark_compare.py --mode live --model <name> \
              [--dialogues dialogues.json] [--budget 250]
"""

import argparse
import json
import os
import re
import statistics
import sys


def normalize(s):
    return re.sub(r"\s+", " ", s.lower()).strip()


def judge_contains(text, ground_truth):
    hit = sum(1 for g in ground_truth if normalize(g) in normalize(text))
    return hit / len(ground_truth)


def rolling_window(turns, budget):
    transcript = " ".join(f"{role}: {text}" for role, text in turns)
    tokens = transcript.split()
    keep = max(10, budget - 12)          # 12 tokens reserved for framing
    return " ".join(tokens[-keep:])


# --- canned extractor client (same shape as tests/test_memory.py) ----------

class MockResponse:
    def __init__(self, content):
        self.choices = [type('Choice', (), {'message': type('Message', (), {'content': content})()})]


class Completions:
    def __init__(self, owner):
        self.owner = owner

    def create(self, **kwargs):
        return MockResponse(self.owner.content)


class Chat:
    def __init__(self, owner):
        self.owner = owner
        self.completions = Completions(owner)


class MockLLMClient:
    def __init__(self, content):
        self.content = content
        self.chat = Chat(self)


DIALOGUE_1 = {
    "name": "api-client-config",
    "turns": [
        ("user", "Configure the payments client. Two hard rules: all "
                 "responses must be JSON, and use POST /v1/items for "
                 "creation."),
        ("assistant", "Done. Both are noted in the runbook now, and I will "
                      "surface them in the weekly ops review so the rest of "
                      "the team sees them too. The rest of the shift stayed "
                      "quiet on alerts, the dashboards held green apart from "
                      "a brief blip in the ingest queue that cleared itself "
                      "before anyone had to page anyone, and the deploy "
                      "checklist template got two new checkboxes added under "
                      "compliance so nothing depends on memory alone."),
        ("user", "New rate limit: at most 100 requests per minute."),
        ("assistant", "Noted and applied exactly as described. The change "
                      "ticket links back to this thread for provenance, the "
                      "on-call rotation has been pinged with the summary, "
                      "and the capacity chart picked up the new ceiling "
                      "automatically, so over the next few days we can "
                      "watch how the live traffic settles against it and "
                      "decide whether the alert thresholds need any further "
                      "tuning or whether the defaults keep working."),
        ("user", "Also: never log raw request bodies, they contain customer "
                 "data. And requests must not exceed 5 seconds."),
        ("assistant", "Both are wired into the guard config now, each with a "
                      "small test attached so they cannot silently regress "
                      "later. I also filed the two follow-ups we discussed, "
                      "tagged the release-notes draft accordingly, and left "
                      "the audit trail pointing back to this conversation, "
                      "which should make the compliance review "
                      "straightforward when it comes around next quarter."),
    ],
    "patches": [
        {"patch_id": "p1", "turn_index": 0, "patch_type": "constraint",
         "payload": "All responses must be JSON", "dependencies": [],
         "utility": 0.95, "fingerprint": "", "slot_key": ""},
        {"patch_id": "p2", "turn_index": 0, "patch_type": "constraint",
         "payload": "Use POST /v1/items for creation", "dependencies": [],
         "utility": 0.93, "fingerprint": "", "slot_key": ""},
        {"patch_id": "p3", "turn_index": 2, "patch_type": "constraint",
         "payload": "Rate limit: at most 100 requests per minute",
         "dependencies": [], "utility": 0.94, "fingerprint": "", "slot_key": ""},
        {"patch_id": "p4", "turn_index": 4, "patch_type": "constraint",
         "payload": "Never log raw request bodies", "dependencies": [],
         "utility": 0.96, "fingerprint": "", "slot_key": ""},
        {"patch_id": "p5", "turn_index": 4, "patch_type": "decision",
         "payload": "Request timeout: 5 seconds", "dependencies": [],
         "utility": 0.92, "fingerprint": "", "slot_key": ""},
    ],
    "ground_truth": ["all responses must be json", "post /v1/items",
                     "100 requests per minute",
                     "never log raw request bodies", "5 seconds"],
}

DIALOGUE_2 = {
    "name": "deployment-rules",
    "turns": [
        ("user", "Two deploy rules for this repo: never deploy on Fridays, "
                 "and keep staging in sync with production after every "
                 "release."),
        ("assistant", "Both are registered in the pipeline guard now. The "
                      "freeze blocks the deploy job outright, the sync hook "
                      "runs right after the release step finishes, and I "
                      "put both rules into the onboarding doc as well, "
                      "since new joiners keep asking where the release "
                      "conventions live and this saves repeating the same "
                      "explanation at every single sprint retrospective "
                      "when somebody reopens the question about windows."),
        ("user", "Log retention is 90 days now, not the old 30. And every "
                 "deploy needs a ticket number in the commit message."),
        ("assistant", "Retention updated as requested and the commit "
                      "template now requires a ticket reference before "
                      "anything can land. The audit script can link deploys "
                      "back to their change records again, the dashboard "
                      "widgets reading the retention buckets got their "
                      "default window bumped to match, and I rescheduled "
                      "the archive export so it no longer races the new "
                      "cutoff at month end, which was the last open "
                      "question from the review."),
        ("user", "One more: secret rotation is 30 days, not 90."),
        ("assistant", "Rotation window set to 30 days across the board, "
                      "with a renewal reminder firing a week ahead so "
                      "nothing lapses quietly. The security review doc got "
                      "the same update, the automation that rotates service "
                      "credentials picked up the schedule automatically, and "
                      "I flagged the one exception we discussed as a "
                      "tracked follow-up so it does not get lost between "
                      "now and the next quarterly compliance pass."),
    ],
    "patches": [
        {"patch_id": "d1", "turn_index": 0, "patch_type": "constraint",
         "payload": "Never deploy on Fridays", "dependencies": [],
         "utility": 0.95, "fingerprint": "", "slot_key": ""},
        {"patch_id": "d2", "turn_index": 0, "patch_type": "constraint",
         "payload": "Keep staging in sync with production after every release",
         "dependencies": [], "utility": 0.90, "fingerprint": "", "slot_key": ""},
        {"patch_id": "d3", "turn_index": 2, "patch_type": "decision",
         "payload": "Log retention: 90 days", "dependencies": [],
         "utility": 0.92, "fingerprint": "", "slot_key": ""},
        {"patch_id": "d4", "turn_index": 2, "patch_type": "constraint",
         "payload": "Every deploy needs a ticket number in the commit message",
         "dependencies": [], "utility": 0.88, "fingerprint": "", "slot_key": ""},
        {"patch_id": "d5", "turn_index": 4, "patch_type": "decision",
         "payload": "Secret rotation: 30 days", "dependencies": [],
         "utility": 0.91, "fingerprint": "", "slot_key": ""},
    ],
    "ground_truth": ["never deploy on fridays",
                     "staging in sync with production", "90 days",
                     "ticket number", "30 days"],
}


def mock(args):
    from dspm import DSPMMemory
    rows = []
    for d in (DIALOGUE_1, DIALOGUE_2):
        mem = DSPMMemory(budget=args.budget,
                         llm_client=MockLLMClient(json.dumps(d["patches"])),
                         model="test-model")
        for role, text in d["turns"]:
            mem.add_turn(role, text)
        ctx = mem.get_context(query="overview")
        crr_dspm = judge_contains(ctx, d["ground_truth"])
        crr_roll = judge_contains(rolling_window(d["turns"], args.budget),
                                   d["ground_truth"])
        rows.append((d["name"], crr_dspm, crr_roll))
        print(f"{d['name']:<18} DSPM {crr_dspm * 100:5.1f}%   "
              f"rolling-window {crr_roll * 100:5.1f}%   (budget {args.budget})")
    print(f"mean               DSPM {statistics.mean(r[1] for r in rows) * 100:5.1f}%   "
          f"rolling-window {statistics.mean(r[2] for r in rows) * 100:5.1f}%")
    print()
    print("NOTE: --mode mock is a deterministic pipeline check with a canned")
    print("extractor and a string-match judge. The published retention figure")
    print("comes from --mode live (real extractor, true rolling summary, LLM judge).")


def _client(args):
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("live mode needs the openai package:  pip install openai")
    api_key = (os.environ.get("ZAI_API_KEY") or os.environ.get("DSPM_API_KEY")
               or os.environ.get("OPENAI_API_KEY"))
    if not api_key:
        sys.exit("set ZAI_API_KEY (or OPENAI_API_KEY / DSPM_API_KEY) for live mode")
    base_url = (args.base_url or os.environ.get("ZAI_API_BASE")
                or ("https://api.z.ai/api/paas/v4"
                    if os.environ.get("ZAI_API_KEY") else None))
    return OpenAI(api_key=api_key, base_url=base_url)


def _chat(client, model, prompt, max_tokens=400):
    resp = client.chat.completions.create(
        model=model, max_tokens=max_tokens, temperature=0,
        messages=[{"role": "user", "content": prompt}])
    return resp.choices[0].message.content


def llm_summarize(client, model, rolling, role, text):
    return _chat(client, model,
                 "Summarize the conversation so far in at most 120 tokens. "
                 "Preserve every rule, constraint, decision and number "
                 "verbatim.\n\nConversation so far:\n"
                 + (rolling or "(none)")
                 + f"\n\nNew turn ({role}): {text}")


def llm_judge(client, model, context, ground_truth):
    resp = _chat(client, model,
                 "For each item below, answer whether the CONTEXT explicitly "
                 "states it (present, including as a rule/decision). Reply "
                 "with ONLY a JSON array of true/false in the same order.\n\n"
                 "ITEMS: " + json.dumps(ground_truth)
                 + "\n\nCONTEXT:\n" + context)
    m = re.search(r"\[.*?\]", resp, re.S)
    if not m:
        raise RuntimeError(f"judge returned unparseable output: {resp[:200]}")
    found = json.loads(m.group(0))
    return sum(1 for v in found if v is True) / len(ground_truth)


def live(args):
    from dspm import DSPMMemory
    if not args.model:
        sys.exit("live mode needs --model (your extractor/judge model name)")
    client = _client(args)
    if args.dialogues:
        with open(args.dialogues, encoding="utf-8") as f:
            dialogues = json.load(f)
    else:
        dialogues = [DIALOGUE_1, DIALOGUE_2]
    d_scores, r_scores = [], []
    for d in dialogues:
        mem = DSPMMemory(budget=args.budget, llm_client=client, model=args.model)
        rolling = ""
        for role, text in d["turns"]:
            mem.add_turn(role, text)
            rolling = llm_summarize(client, args.model, rolling, role, text)
        ctx = mem.get_context(query="overview")
        s_d = llm_judge(client, args.model, ctx, d["ground_truth"])
        s_r = llm_judge(client, args.model, rolling, d["ground_truth"])
        d_scores.append(s_d)
        r_scores.append(s_r)
        print(f"{d.get('name', 'dialogue'):<18} DSPM {s_d * 100:5.1f}%   "
              f"rolling-summary {s_r * 100:5.1f}%")
    fmt = lambda xs: (f"{statistics.mean(xs) * 100:.1f}%"
                      if len(xs) > 1
                      else f"{xs[0] * 100:.1f}%")
    extra = (f" +/- {statistics.stdev(d_scores) * 100:.1f}"
             if len(d_scores) > 1 else "")
    extra_r = (f" +/- {statistics.stdev(r_scores) * 100:.1f}"
              if len(r_scores) > 1 else "")
    print(f"mean               DSPM {fmt(d_scores)}{extra}   "
          f"rolling-summary {fmt(r_scores)}{extra_r}   "
          f"({len(d_scores)} dialogues, budget {args.budget})")


def main():
    ap = argparse.ArgumentParser(description="DSPM vs rolling-summary retention")
    ap.add_argument("--mode", choices=["mock", "live"], default="mock")
    ap.add_argument("--budget", type=int, default=250)
    ap.add_argument("--model", default=None,
                    help="extractor/judge model name (live mode)")
    ap.add_argument("--base-url", default=None, help="OpenAI-compatible base URL")
    ap.add_argument("--dialogues", default=None,
                    help="JSON file: list of {turns, ground_truth} (live mode)")
    args = ap.parse_args()
    if args.mode == "mock":
        mock(args)
    else:
        live(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
