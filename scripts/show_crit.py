import json
import sys

folder = sys.argv[1] if len(sys.argv) > 1 else "diag_out_guard2"
budget = sys.argv[2] if len(sys.argv) > 2 else "120"
d = json.load(open(f"{folder}/b{budget}_cp48.json"))
print("RENDERED CONTEXT:")
print(d["rendered_context"])
print("--- CRITICAL PATCHES IN STORE:")
for p in d["dump_store"]["patches"]:
    if p["is_critical"]:
        print(p["patch_type"], p["turn_index"], p["payload"])
print("--- NON-CRITICAL COUNT:",
      sum(1 for p in d["dump_store"]["patches"] if not p["is_critical"]))