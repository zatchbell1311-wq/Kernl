#!/usr/bin/env python3
"""
DSPM v0.5.3 MultiWOZ Benchmark — Results Graph

Reads the ACTUAL results from dspm_multiwoz_v6_results_sr.jsonl
and creates a professional bar chart with the REAL numbers.

Run:  python make_multiwoz_chart.py
Output:  multiwoz_v053_results.png
"""

import json
from pathlib import Path
from statistics import mean
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# LOAD REAL RESULTS
# ============================================================
results_path = Path("dspm_multiwoz_v6_results_sr.jsonl")

if not results_path.exists():
    print(f"ERROR: {results_path} not found.")
    print("Run the benchmark first: python run_dspm_multiwoz_v6.py")
    exit(1)

results = [json.loads(line) for line in open(results_path) if line.strip()]

# Group by method
methods_data = {}
for r in results:
    m = r.get("method")
    if m not in methods_data:
        methods_data[m] = []
    methods_data[m].append(r)

# Calculate real metrics per method
method_names = ["dspm_compressed", "dspm_all_patches", "last_k", "llm_summary"]
display_names = {
    "dspm_compressed": "DSPM v0.5.3\n(compressed)",
    "dspm_all_patches": "DSPM v0.5.3\n(all_patches)",
    "last_k": "last-k turns\n(baseline)",
    "llm_summary": "LLM summary\n(baseline)",
}

metrics = {}
for method in method_names:
    if method not in methods_data:
        continue
    items = [r for r in methods_data[method] if "error" not in r and r.get("compressed_tokens", 0) > 0]
    if not items:
        continue

    # SR: count sr_q_ fields that are True
    sr_correct = 0
    sr_total = 0
    for r in items:
        for k, v in r.items():
            if k.startswith("sr_q_") and isinstance(v, bool):
                sr_total += 1
                sr_correct += int(v)

    metrics[method] = {
        "sr": sr_correct / sr_total * 100 if sr_total > 0 else 0,
        "sr_correct": sr_correct,
        "sr_total": sr_total,
        "comp_tokens": mean(r["compressed_tokens"] for r in items),
        "raw_tokens": mean(r["raw_tokens"] for r in items),
        "cpi": mean(r["cpi"] for r in items if r.get("cpi") is not None),
        "n": len(items),
    }

# Print the real numbers
print("=" * 70)
print("REAL MultiWOZ v0.5.3 Results (from dspm_multiwoz_v6_results_sr.jsonl)")
print("=" * 70)
print()
print(f"{'Method':<22} {'N':>4} {'SR':>8} {'CPI':>6} {'Comp tok':>9} {'Raw tok':>8}")
print("-" * 60)
for method in method_names:
    if method in metrics:
        m = metrics[method]
        print(f"{display_names[method]:<22} {m['n']:>4} {m['sr']:>6.1f}%  {m['cpi']:>5.2f} "
              f"{m['comp_tokens']:>8.0f}  {m['raw_tokens']:>7.0f}")
print()

# ============================================================
# CHART
# ============================================================
COLOR_DSPM = "#4C6EF5"
COLOR_BASELINE = "#94A3B8"
COLOR_BG = "#F8FAFC"
COLOR_TEXT = "#243447"
COLOR_GRID = "#E9EEF3"
COLOR_MUTED = "#6B7280"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.edgecolor": COLOR_MUTED,
    "axes.linewidth": 0.8,
    "text.color": COLOR_TEXT,
    "axes.labelcolor": COLOR_TEXT,
    "xtick.color": COLOR_TEXT,
    "ytick.color": COLOR_TEXT,
})

# 3 subplots: SR, Tokens, CPI
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6), constrained_layout=True)
fig.patch.set_facecolor(COLOR_BG)

x = np.arange(len(method_names))
bar_width = 0.55

# Determine which methods exist
present_methods = [m for m in method_names if m in metrics]
x = np.arange(len(present_methods))

# Bar colors: DSPM = blue, baselines = gray
bar_colors = [COLOR_DSPM if "dspm" in m else COLOR_BASELINE for m in present_methods]
labels = [display_names[m] for m in present_methods]

# --- LEFT: SR (%) ---
sr_values = [metrics[m]["sr"] for m in present_methods]
bars1 = ax1.bar(x, sr_values, bar_width, color=bar_colors, edgecolor="white", linewidth=1.5, zorder=3)
ax1.set_facecolor(COLOR_BG)
ax1.set_ylabel("Semantic Recall (%)", fontsize=13, fontweight="bold")
ax1.set_title("Recall", fontsize=14, fontweight="bold", pad=12)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=9)
ax1.set_ylim(0, 105)
for bar, val in zip(bars1, sr_values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             f"{val:.1f}%", ha="center", va="bottom",
             fontsize=11, fontweight="bold", color=COLOR_TEXT)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.yaxis.grid(True, color=COLOR_GRID, linewidth=0.8, alpha=0.7, zorder=1)
ax1.set_axisbelow(True)
ax1.tick_params(axis="x", length=0)
ax1.tick_params(axis="y", length=0)

# --- MIDDLE: Tokens used ---
tok_values = [metrics[m]["comp_tokens"] for m in present_methods]
bars2 = ax2.bar(x, tok_values, bar_width, color=bar_colors, edgecolor="white", linewidth=1.5, zorder=3)
ax2.set_facecolor(COLOR_BG)
ax2.set_ylabel("Tokens Used (out of 400 budget)", fontsize=13, fontweight="bold")
ax2.set_title("Cost (lower = better)", fontsize=14, fontweight="bold", pad=12)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=9)
ax2.set_ylim(0, max(tok_values) * 1.25)
ax2.invert_yaxis()
for bar, val in zip(bars2, tok_values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
             f"{val:.0f}t", ha="center", va="top",
             fontsize=11, fontweight="bold", color=COLOR_TEXT)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.yaxis.grid(True, color=COLOR_GRID, linewidth=0.8, alpha=0.7, zorder=1)
ax2.set_axisbelow(True)
ax2.tick_params(axis="x", length=0)
ax2.tick_params(axis="y", length=0)

# --- RIGHT: CPI ---
cpi_values = [metrics[m]["cpi"] * 100 for m in present_methods]
bars3 = ax3.bar(x, cpi_values, bar_width, color=bar_colors, edgecolor="white", linewidth=1.5, zorder=3)
ax3.set_facecolor(COLOR_BG)
ax3.set_ylabel("CPI (%)", fontsize=13, fontweight="bold")
ax3.set_title("Critical Patch Inclusion", fontsize=14, fontweight="bold", pad=12)
ax3.set_xticks(x)
ax3.set_xticklabels(labels, fontsize=9)
ax3.set_ylim(0, 105)
for bar, val in zip(bars3, cpi_values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             f"{val:.1f}%", ha="center", va="bottom",
             fontsize=11, fontweight="bold", color=COLOR_TEXT)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3.yaxis.grid(True, color=COLOR_GRID, linewidth=0.8, alpha=0.7, zorder=1)
ax3.set_axisbelow(True)
ax3.tick_params(axis="x", length=0)
ax3.tick_params(axis="y", length=0)

# Footer
fig.text(0.5, -0.01,
         f"DSPM v0.5.3 on MultiWOZ 2.2  |  N={metrics[present_methods[0]]['n']} dialogues  |  "
         f"Budget=400 tokens  |  Model: qwen/qwen3.8-27b via Groq\n"
         f"Data from: dspm_multiwoz_v6_results_sr.jsonl (real benchmark results)",
         ha="center", fontsize=9, color=COLOR_MUTED, style="italic")

# Save
output_path = Path("multiwoz_v053_results.png")
fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=COLOR_BG, edgecolor="none")
plt.close()

print(f"\nChart saved to: {output_path.resolve()}")
print(f"\nChart shows 3 panels:")
print(f"  LEFT  — SR (Semantic Recall): DSPM vs last_k vs llm_summary")
print(f"  MIDDLE — Tokens used (lower = better)")
print(f"  RIGHT — CPI (Critical Patch Inclusion)")
print(f"\nAll numbers are from the REAL benchmark results file.")