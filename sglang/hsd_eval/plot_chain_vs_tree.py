"""Chain vs tree, fixed (best per point) and adaptive, total tok/s vs concurrency (runs/chain_vs_tree.json)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
rows = json.load(open("runs/chain_vs_tree.json"))
CONC = [1, 2, 4, 8, 16, 32, 64]
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a66", "#e6e5e1", "#fcfcfb"
# chain = slot 1 (blue), tree = slot 2 (orange); fixed dashed, adaptive solid
SERIES = [("best fixed chain", "#2a78d6", "--", "o"), ("adaptive chain (SGLang)", "#2a78d6", "-", "s"),
          ("best fixed tree", "#eb6834", "--", "^"), ("adaptive tree (ours)", "#eb6834", "-", "D")]
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True, facecolor=SURFACE)
for ax, ctx in zip(axes, ("short context", "long context")):
    ax.set_facecolor(SURFACE)
    for label, color, ls, mk in SERIES:
        y = [next(r["tok_s"] for r in rows if r["context"] == ctx and r["concurrency"] == c and r["structure"] == label) for c in CONC]
        ax.plot(CONC, y, color=color, ls=ls, lw=2, marker=mk, ms=8, markeredgecolor=SURFACE, markeredgewidth=1.5, label=label, zorder=3)
    ax.set_xscale("log", base=2); ax.set_xticks(CONC, [str(c) for c in CONC])
    ax.set_title(f"{ctx} ({'~115' if ctx.startswith('short') else '~1.85k'} prompt tokens)", color=INK, fontsize=12, loc="left")
    ax.set_xlabel("concurrency (requests in flight)", color=MUTED); ax.grid(True, color=GRID, lw=1, zorder=0)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED); ax.set_ylim(0, 2100)
axes[0].set_ylabel("total throughput (tokens/s), mean of 3 runs", color=MUTED)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=10, labelcolor=INK, bbox_to_anchor=(0.5, -0.01))
fig.suptitle("Chain vs tree drafts, EAGLE-3 on Llama-3.1-8B (RTX 6000 Ada, GSM8K): fixed trees win only at low load; "
             "adaptive trees win almost everywhere", color=INK, fontsize=12, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0.07, 1, 0.95))
fig.savefig("runs/chain_vs_tree.png", dpi=150, facecolor=SURFACE)
