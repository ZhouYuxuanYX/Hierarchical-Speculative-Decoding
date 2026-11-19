"""Throughput vs concurrency for stock EAGLE-3 draft shapes (oracle study, runs/oracle/).

python plot_throughput.py  ->  runs/oracle/throughput_vs_concurrency.png
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CONC = [1, 2, 4, 8, 16, 32, 64]
# (run name, legend label, color, marker, linestyle); reference palette slots 1-6 in fixed order
SERIES = [
    ("chain3", "chain, depth 3 (N=4)", "#2a78d6", "o", "-"),
    ("chain6", "chain, depth 6 (N=7)", "#eb6834", "s", "-"),
    ("tree4x6n7", "tree k=4 d=6, N=7", "#1baf7a", "^", "-"),
    ("tree4x6n16", "tree k=4 d=6, N=16", "#eda100", "D", "-"),
    ("tree8x5n32", "tree k=8 d=5, N=32", "#e87ba4", "v", "-"),
    ("tree10x5n60", "tree k=10 d=5, N=60", "#008300", "P", "-"),
    ("target", "no speculation", "#8a8984", None, "--"),
]
PANELS = [(0, "short context (~115 prompt tokens)"), (8, "long context (~1.85k prompt tokens)")]
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a66", "#e6e5e1", "#fcfcfb"


def load(run, shots):
    return [json.load(open(f"runs/oracle/{run}/s{shots}_c{c}.json"))["summary"]["total"] for c in CONC]


def main():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True, facecolor=SURFACE)
    for ax, (shots, title) in zip(axes, PANELS):
        ax.set_facecolor(SURFACE)
        for run, label, color, marker, ls in SERIES:
            ax.plot(CONC, load(run, shots), color=color, lw=2, ls=ls, marker=marker, ms=8,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=label, zorder=3)
        ax.set_xscale("log", base=2)
        ax.set_xticks(CONC, [str(c) for c in CONC])
        ax.set_title(title, color=INK, fontsize=12, loc="left")
        ax.set_xlabel("concurrency (requests in flight)", color=MUTED)
        ax.grid(True, color=GRID, lw=1, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED)
        ax.set_ylim(0, 2000)
    axes[0].set_ylabel("total throughput (tokens/s)", color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=10,
               labelcolor=INK, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Stock EAGLE-3 on Llama-3.1-8B (RTX 6000 Ada, GSM8K, T=0.6): "
                 "big trees win at low concurrency, short chains at high",
                 color=INK, fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    fig.savefig("runs/oracle/throughput_vs_concurrency.png", dpi=150, facecolor=SURFACE)


if __name__ == "__main__":
    main()
