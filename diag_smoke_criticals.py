# diag_smoke_criticals.py -- READ-ONLY diagnostic wrapper for the smoke
# CRR failure (87% = 7/8 delivered). Monkey-patches DSPMMemory.get_context
# to dump EVERY stored critical (type, payload, turn) plus the stats right
# after the smoke builds its context, then runs scripts/smoke_conversation.py
# UNCHANGED. Writes nothing; costs one extra API pass over the transcript.
#
# Run (with your GROQ key already set, as before):
#     & $py diag_smoke_criticals.py
# Then paste EVERYTHING from "DIAG: ALL STORED CRITICALS" onward.

import runpy
import sys
from pathlib import Path

import dspm.memory as mm

ORIG = mm.DSPMMemory.get_context


def patched(self, query=""):
    ctx = ORIG(self, query)
    print()
    print("=" * 72)
    print("DIAG: ALL STORED CRITICALS (%d)" % len(self.critical_patches))
    print("=" * 72)
    for p in self.critical_patches:
        print("  [%s] %r  (turn %d)"
              % (p.patch_type.upper(), p.payload, p.turn_index))
    s = self.stats
    print("-" * 72)
    print("DIAG stats: critical_total=%d critical_selected=%d crr=%d "
          "budget=%d context_built=%s"
          % (s["critical_total"], s["critical_selected"], s["crr"],
             s["budget"], s["context_built"]))
    print("=" * 72)
    return ctx


mm.DSPMMemory.get_context = patched

smoke = Path(__file__).resolve().parent / "scripts" / "smoke_conversation.py"
if not smoke.exists():
    sys.exit("smoke script not found at " + str(smoke))
runpy.run_path(str(smoke), run_name="__main__")
