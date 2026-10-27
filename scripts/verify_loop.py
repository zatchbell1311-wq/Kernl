# scripts/verify_loop.py -- UNTESTED
import json
import re
from pathlib import Path

TAG = re.compile(r"^\[[A-Z]+\]\s*")

for budget in (120, 60):
    prev_lines = set()
    for n in (12, 24, 36, 48):
        path = Path(f"diag_out/b{budget}_cp{n}.json")
        record = json.loads(path.read_text(encoding="utf-8"))
        stats = record["stats"]
        dump = record["dump_store"]
        echoes = [p["payload"] for p in dump["patches"]
                  if TAG.sub("", p["payload"].strip()) in prev_lines]
        print(f"b{budget} cp{n:>2}: patches={stats['total_patches']:>2} "
              f"crit={stats['critical_total']:>2} sel={stats['critical_selected']:>2} "
              f"context-echo actives={len(echoes)}")
        for e in echoes:
            print(f"          echo: {e}")
        prev_lines = {TAG.sub("", ln.strip())
                      for ln in record["rendered_context"].splitlines() if ln.strip()}