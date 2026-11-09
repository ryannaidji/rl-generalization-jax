"""Figures from results/part_a/study.json. Usage: python -m experiments.plot [results_dir] [fig_dir]"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"1": "#2a78d6", "8": "#eb6834", "64": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
LABELS = {"length": "Half pole length (m)", "force_mag": "Force magnitude (N)"}


def main(res="results/part_a", figs="report/figures"):
    study = json.loads((Path(res) / "study.json").read_text())
    Path(figs).mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, sweep in zip(axes, ("length", "force_mag")):
        for n, row in study["conditions"].items():
            s = row["sweeps"][sweep]
            lo = [c["ci_low"] for c in s["ci"]]
            hi = [c["ci_high"] for c in s["ci"]]
            ax.fill_between(s["values"], lo, hi, color=COLORS[n], alpha=0.15, linewidth=0)
            ax.plot(s["values"], s["mean"], color=COLORS[n], linewidth=2, marker="o", markersize=4,
                    label=f"{n} training context" + ("" if n == "1" else "s"))
        ax.set_xlabel(LABELS[sweep], color=MUTED)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(colors=MUTED, length=0)
    axes[0].set_ylabel("Greedy return (max 500)", color=MUTED)
    axes[0].set_ylim(0, 520)
    axes[0].legend(frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(Path(figs) / "sweeps.png", dpi=160)


if __name__ == "__main__":
    main(*sys.argv[1:])
