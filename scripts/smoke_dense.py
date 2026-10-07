"""v0.1.10 pre-ship smoke test: auto-dense + numeric capture with a real model."""
import os
from openai import OpenAI
from dspm import DSPMMemory

KEY = os.environ.get("GROQ_API_KEY", "")
assert KEY, "Set key first: $env:GROQ_API_KEY = 'gsk_...'"

llm = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=KEY)
memory = DSPMMemory(budget=250, llm_client=llm, model="openai/gpt-oss-20b")

text = (
    "At the 250-token budget DSPM achieved a mean token reduction rate of 82.84 percent "
    "while retaining 100 percent of constraint and decision patches. Mean semantic "
    "consistency was 2.14 with Wilcoxon p = 0.031. At the 400-token budget consistency "
    "was 3.00 with p = 1.000. Removing shadow selection collapsed CRR to 37.9 percent."
)

from dspm.extractor import numeric_density
from dspm.config import DENSE_AUTO_THRESHOLD
d = numeric_density(text)
mode = "DENSE (up to 10 patches)" if d >= DENSE_AUTO_THRESHOLD else "standard (max 5)"
print(f"numeric density: {d:.3f} vs threshold {DENSE_AUTO_THRESHOLD} -> {mode}")

patches = memory.add_turn("user", text)
print(f"\nextracted {len(patches)} patches")
for p in patches:
    print(f"  [{p.patch_type[:4].upper()}] {p.payload}")

nums = {"82.84", "100", "2.14", "0.031", "3.00", "1.000", "37.9"}
found = {n for p in patches for n in nums if n in p.payload}
missing = sorted(nums - found)
print(f"\nnumbers captured ({len(found)}/{len(nums)}): {sorted(found)}")
if missing:
    print(f"numbers missed: {missing}")

print("\nVERDICT")
print(f"[{'PASS' if len(found) >= 5 else 'WEAK'}] numeric capture: {len(found)}/7 "
      "(v0.1.9 behavior typically captured 2-3)")
crits = [p for p in patches if p.is_critical]
print(f"[INFO] criticals extracted: {len(crits)} (result stats should be equation patches, not constraints)")
for p in crits:
    print(f"       -> [{p.patch_type[:4].upper()}] {p.payload}")
