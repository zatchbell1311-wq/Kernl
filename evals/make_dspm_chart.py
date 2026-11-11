#!/usr/bin/env python3
"""
DSPM v0.5.0 MultiWOZ Results — Bar Chart Generator

Creates a professional bar chart comparing 4 methods:
  - DSPM v0.5.0 (compressed)
  - DSPM v0.5.0 (all_patches)
  - last-k turns (baseline)
  - LLM summary (baseline)

Shows SR (%) and tokens used side by side.

Run:  python make_dspm_chart.py
Output:  dspm_v05_results.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ============================================================
# DATA — from v0.5.0 MultiWOZ run
# ============================================================
methods = [
    "DSPM v0.5.0\n(compressed)",
    "DSPM v0.5.0\n(all_patches)",
    "last-k turns\n(baseline)",
    "LLM summary\n(baseline)",
]

sr_values = [89.47, 89.47, 89.47, 84.21]          # Semantic Recall %
tokens_used = [104, 107, 161, 236]                # compressed tokens
cpi_values = [0.71, 0.71, 0.74, 0.72]             # CPI (0-1)

# ============================================================
# CHART SETUP
# ============================================================
# Color palette — Business Cool from charts skill
COLOR_DSPM = "#4C6EF5"        # accent blue for DSPM
COLOR_BASELINE = "#E9EEF3"    # neutral gray for baselines
COLOR_DSPM_DARK = "#243447"   # dark text
COLOR_BG = "#F8FAFC"          # background
COLOR_GRID = "#E9EEF3"        # grid lines
COLOR_TEXT_MUTED = "#6B7280"  # muted text

# Bar colors: DSPM = accent, baselines = neutral
bar_colors_sr = [COLOR_DSPM, COLOR_DSPM, "#94A3B8", "#94A3B8"]
bar_colors_tok = [COLOR_DSPM, COLOR_DSPM, "#94A3B8", "#94A3B8"]

# Font setup
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.edgecolor": COLOR_TEXT_MUTED,
    "axes.linewidth": 0.8,
    "text.color": COLOR_DSPM_DARK,
    "axes.labelcolor": COLOR_DSPM_DARK,
    "xtick.color": COLOR_DSPM_DARK,
    "ytick.color": COLOR_DSPM_DARK,
})

# ============================================================
# CREATE FIGURE — 2 subplots side by side
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5), constrained_layout=True)
fig.patch.set_facecolor(COLOR_BG)

x = np.arange(len(methods))
bar_width = 0.55

# ── LEFT: Semantic Recall (%) ──
bars1 = ax1.bar(x, sr_values, bar_width, color=bar_colors_sr,
                edgecolor="white", linewidth=1.5, zorder=3)

ax1.set_facecolor(COLOR_BG)
ax1.set_ylabel("Semantic Recall (%)", fontsize=13, fontweight="bold", color=COLOR_DSPM_DARK)
ax1.set_title("Recall: DSPM matches last-k baseline",
              fontsize=14, fontweight="bold", color=COLOR_DSPM_DARK, pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(methods, fontsize=10)
ax1.set_ylim(0, 105)
ax1.axhline(y=89.47, color=COLOR_DSPM, linestyle="--", linewidth=1, alpha=0.4, zorder=2)
ax1.text(3.4, 91, "89.5% tie", fontsize=9, color=COLOR_DSPM, alpha=0.7,
         ha="right", style="italic")

# Add value labels on bars
for bar, val in zip(bars1, sr_values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             f"{val:.1f}%", ha="center", va="bottom",
             fontsize=12, fontweight="bold", color=COLOR_DSPM_DARK)

# Style: remove top/right spines, light grid
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.yaxis.grid(True, color=COLOR_GRID, linewidth=0.8, alpha=0.7, zorder=1)
ax1.set_axisbelow(True)
ax1.tick_params(axis="x", length=0)
ax1.tick_params(axis="y", length=0)

# ── RIGHT: Tokens Used ──
bars2 = ax2.bar(x, tokens_used, bar_width, color=bar_colors_tok,
                edgecolor="white", linewidth=1.5, zorder=3)

ax2.set_facecolor(COLOR_BG)
ax2.set_ylabel("Tokens Used (out of 400 budget)",
               fontsize=13, fontweight="bold", color=COLOR_DSPM_DARK)
ax2.set_title("Cost: DSPM uses 35% fewer tokens than last-k",
              fontsize=14, fontweight="bold", color=COLOR_DSPM_DARK, pad=15)
ax2.set_xticks(x)
ax2.set_xticklabels(methods, fontsize=10)
ax2.set_ylim(0, 280)
ax2.invert_yaxis()  # Lower = better, so invert

# Add value labels on bars
for bar, val in zip(bars2, tokens_used):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             f"{val}t", ha="center", va="top",
             fontsize=12, fontweight="bold", color=COLOR_DSPM_DARK)

# Style
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.yaxis.grid(True, color=COLOR_GRID, linewidth=0.8, alpha=0.7, zorder=1)
ax2.set_axisbelow(True)
ax2.tick_params(axis="x", length=0)
ax2.tick_params(axis="y", length=0)

# ============================================================
# ADD ANNOTATIONS
# ============================================================
# Left chart: arrow showing the tie
ax1.annotate("DSPM = last-k\n(89.5% SR)",
             xy=(0, 89.47), xytext=(0.5, 75),
             fontsize=10, color=COLOR_DSPM, fontweight="bold",
             ha="center",
             arrowprops=dict(arrowstyle="->", color=COLOR_DSPM, lw=1.5))

# Right chart: highlight DSPM's advantage
ax2.annotate("35% fewer\ntokens",
             xy=(0, 104), xytext=(0.5, 145),
             fontsize=10, color=COLOR_DSPM, fontweight="bold",
             ha="center",
             arrowprops=dict(arrowstyle="->", color=COLOR_DSPM, lw=1.5))

# ============================================================
# FOOTER
# ============================================================
fig.text(0.5, -0.02,
         "DSPM v0.5.0 on MultiWOZ 2.2  |  N=10 dialogues  |  Budget=400 tokens  |  Model: qwen/qwen3.8-27b via Groq\n"
         "From v0.2.2 (42% SR) to v0.5.0 (89% SR): +47 points in 2 iterations",
         ha="center", fontsize=9, color=COLOR_TEXT_MUTED, style="italic")

# ============================================================
# SAVE
# ============================================================
output_path = Path("dspm_v05_results.png")
fig.savefig(output_path, dpi=200, bbox_inches="tight",
            facecolor=COLOR_BG, edgecolor="none")
plt.close()

print(f"Chart saved to: {output_path.resolve()}")
print(f"\nChart shows:")
print(f"  LEFT  — Semantic Recall: DSPM (89.5%) = last-k (89.5%) > LLM summary (84.2%)")
print(f"  RIGHT — Tokens used:      DSPM (104) < last-k (161) < LLM summary (236)")
print(f"\nKey insight: DSPM matches last-k on recall while using 35% fewer tokens.")
