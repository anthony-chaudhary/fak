#!/usr/bin/env python3
"""
fanout_plot.py — render the fanbench one-master-goal -> N-subagent sweep as PNGs.

Reads the cmd/fanbench CSVs under fak/experiments/fanout/ and renders:

  * fanout-dashboard.png — the 2x2 headline over the fan-out width N (1..1024):
      (a) MEASURED cross-agent tool-result dedup (shared vs isolated, uplift shaded);
      (b) MODELED token multiplier (naive vs prefix-cache-reuse) + tax clawed back;
      (c) MODELED parallel speedup (rises then saturates as the fold cost grows);
      (d) MODELED net $ saved per fan-out run (N=1 is a small net LOSS, surfaced).
  * fanout-model-scaling.png — the "bigger models help MORE" panel: at fixed N=256,
      sweep the shared-prefix length (proxy for a larger model's longer goal context)
      and show tax-clawed-back climbing toward the 90% prompt-cache ceiling while the
      absolute $ saved grows ~linearly. Reads experiments/fanout/pscale/p*.csv.

White background (the repo's visuals convention). MEASURED vs MODELED is labelled on
every panel — the two halves are never blended. Run: python tools/fanout_plot.py
"""
import csv
import glob
import os
import re

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FAN = os.path.join(HERE, "..", "fak", "experiments", "fanout")

MEAS = "#1f6feb"   # measured (blue)
MODEL = "#d29922"  # modeled (amber)
NAIVE = "#cf222e"  # naive baseline (red)
SHADE = "#2da44e"  # uplift / gap fill (green)


def load_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0].keys()}


def dashboard_slice(cols):
    if "prefix_tokens" not in cols:
        return cols, None
    prefixes = np.unique(cols["prefix_tokens"])
    if len(prefixes) <= 1:
        return cols, int(prefixes[0]) if len(prefixes) else None
    target = 2048 if np.any(prefixes == 2048) else prefixes[0]
    mask = cols["prefix_tokens"] == target
    return {k: v[mask] for k, v in cols.items()}, int(target)


def _logx(ax, N):
    ax.set_xscale("log", base=2)
    ax.set_xticks(N)
    ax.set_xticklabels([f"{int(n)}" for n in N], fontsize=7)
    ax.set_xlabel("fan-out width  N  (sub-agents per goal)")
    ax.grid(True, which="both", alpha=0.25)


def tag(ax, text, color):
    ax.text(0.015, 0.97, text, transform=ax.transAxes, fontsize=7.5, va="top",
            ha="left", color="white", bbox=dict(boxstyle="round,pad=0.25", fc=color, ec="none", alpha=0.9))


def dashboard(cols, out, prefix=None):
    N = cols["agents"]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9))
    pnote = f", P={prefix:,}" if prefix else ""
    fig.suptitle(f"fanbench — one master goal → N sub-agents, swept N=1…1024 (research-goal profile, 16 trials{pnote})",
                 fontsize=13, fontweight="bold")

    # (a) MEASURED cross-agent dedup
    ax = axes[0, 0]
    ax.plot(N, cols["shared_saved_p50"], "-o", color=MEAS, ms=4, label="SHARED (fan-out, one world)")
    ax.plot(N, cols["isolated_saved_p50"], "--s", color="#8250df", ms=4, label="ISOLATED (sub-agents solo)")
    ax.fill_between(N, cols["isolated_saved_p50"], cols["shared_saved_p50"], color=SHADE, alpha=0.25,
                    label="cross_uplift (fan-out-only dedup)")
    _logx(ax, N)
    ax.set_ylabel("model turns deleted (p50)")
    ax.set_title("(a) Cross-agent tool-result dedup")
    ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(0.0, 0.92))
    tag(ax, "MEASURED — real k.Syscall tier-2 events", MEAS)
    ax.annotate(f"+{int(cols['cross_uplift_p50'][-1])} turns\nat N={int(N[-1])}",
                xy=(N[-1], cols["shared_saved_p50"][-1]), xytext=(-95, -10),
                textcoords="offset points", fontsize=8, color=SHADE, fontweight="bold")

    # (b) MODELED token multiplier + tax clawed back
    ax = axes[0, 1]
    ax.plot(N, cols["token_mult_naive"], "-o", color=NAIVE, ms=4, label="naive (re-send prefix per sub-agent)")
    ax.plot(N, cols["token_mult_reuse"], "-o", color=MODEL, ms=4, label="prefix-cache reuse")
    ax.fill_between(N, cols["token_mult_reuse"], cols["token_mult_naive"], color=MODEL, alpha=0.15)
    ax.set_yscale("log")
    _logx(ax, N)
    ax.set_ylabel("input+output token multiplier  vs 1 agent")
    ax.set_title("(b) Token tax — and how much the prefix-cache lever claws back")
    ax.legend(fontsize=7.5, loc="upper left")
    tag(ax, "MODELED — transparent cost model", MODEL)
    ax.annotate(f"{cols['tax_clawed_back'][-1]*100:.0f}% of the tax clawed back\n(plateau by N≈256)",
                xy=(N[-1], cols["token_mult_reuse"][-1]), xytext=(-150, 18),
                textcoords="offset points", fontsize=8, color=MODEL, fontweight="bold")

    # (c) MODELED parallel speedup (saturation)
    ax = axes[1, 0]
    ax.plot(N, cols["parallel_speedup"], "-o", color=MODEL, ms=4)
    _logx(ax, N)
    ax.set_ylabel("parallel speedup  (total work ÷ critical path)")
    ax.set_title("(c) Latency saturation — the fold's coordination tax grows with N")
    tag(ax, "MODELED — critical-path vs total-work", MODEL)
    ax.axhline(cols["parallel_speedup"][-1], color="#999", ls=":", lw=1)
    ax.annotate(f"saturates ≈ {cols['parallel_speedup'][-1]:.0f}×\n(fold-bound past N≈256)",
                xy=(N[-1], cols["parallel_speedup"][-1]), xytext=(-130, -28),
                textcoords="offset points", fontsize=8, color=MODEL, fontweight="bold")

    # (d) MODELED net $ saved per run
    ax = axes[1, 1]
    net = cols["net_dollars_saved"]
    ax.plot(N, net, "-o", color=SHADE, ms=4)
    ax.axhline(0, color="#444", lw=1)
    ax.scatter([N[0]], [net[0]], color=NAIVE, zorder=5)
    _logx(ax, N)
    ax.set_ylabel("net $ saved per fan-out run  (default cost model)")
    ax.set_title("(d) Net savings — and the honest N=1 loss")
    tag(ax, "MODELED — prefix-cache + measured dedup", MODEL)
    ax.annotate(f"N=1: ${net[0]:.3f}\n(fan-out to 1 = a LOSS:\norchestration + cache-write)",
                xy=(N[0], net[0]), xytext=(20, 30), textcoords="offset points",
                fontsize=7.5, color=NAIVE, arrowprops=dict(arrowstyle="->", color=NAIVE, lw=1))
    ax.annotate(f"N={int(N[-1])}: ${net[-1]:.2f}", xy=(N[-1], net[-1]), xytext=(-70, -16),
                textcoords="offset points", fontsize=8, color=SHADE, fontweight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print("wrote", out)


def model_scaling(out):
    files = glob.glob(os.path.join(FAN, "pscale", "p*.csv"))
    pts = []
    for fp in files:
        m = re.search(r"p(\d+)\.csv$", os.path.basename(fp))
        if not m:
            continue
        c = load_csv(fp)
        if not c:
            continue
        pts.append((int(m.group(1)), c["tax_clawed_back"][-1], c["net_dollars_saved"][-1],
                    c["prefix_tokens_saved"][-1]))
    if not pts:
        print("no pscale CSVs found; skipping model-scaling figure")
        return
    pts.sort()
    P = np.array([p[0] for p in pts])
    tax = np.array([p[1] for p in pts]) * 100
    dollars = np.array([p[2] for p in pts])

    fig, ax = plt.subplots(figsize=(9, 5.6))
    fig.suptitle("fanbench — the fan-out lever scales UP with model size (fixed N=256)",
                 fontsize=12.5, fontweight="bold")
    ax.plot(P, tax, "-o", color=MODEL, ms=6, label="token tax clawed back (%)")
    ax.axhline(90, color="#999", ls=":", lw=1)
    ax.text(P[0], 90.6, "90% prompt-cache ceiling", fontsize=7.5, color="#666")
    ax.set_xscale("log", base=2)
    ax.set_xticks(P)
    ax.set_xticklabels([f"{int(p//1024)}K" for p in P])
    ax.set_xlabel("shared master-goal prefix length  (tokens) — proxy for a bigger model's longer goal context")
    ax.set_ylabel("tax clawed back  (%)", color=MODEL)
    ax.set_ylim(40, 100)
    ax.tick_params(axis="y", labelcolor=MODEL)
    ax.grid(True, which="both", alpha=0.25)

    ax2 = ax.twinx()
    ax2.plot(P, dollars, "--s", color=SHADE, ms=6, label="net $ saved per run")
    ax2.set_yscale("log")
    ax2.set_ylabel("net $ saved per fan-out run  (log)", color=SHADE)
    ax2.tick_params(axis="y", labelcolor=SHADE)

    for x, y in zip(P, tax):
        ax.annotate(f"{y:.0f}%", (x, y), textcoords="offset points", xytext=(0, 8),
                    fontsize=8, color=MODEL, ha="center", fontweight="bold")
    ax.annotate("longer shared context ⇒ prefix dominates ⇒\nthe lever claws back more (toward 90%),\nand absolute $ grows ~linearly",
                xy=(P[-2], tax[-2]), xytext=(-10, -64), textcoords="offset points",
                fontsize=8.5, color="#333", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", fc="#f6f8fa", ec="#ddd"))
    tag(ax, "MODELED — cost model; dedup half is model-independent", MODEL)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print("wrote", out)


def main():
    cols = load_csv(os.path.join(FAN, "fanbench-research.csv"))
    if cols is None:
        raise SystemExit("missing fak/experiments/fanout/fanbench-research.csv — run cmd/fanbench first")
    dash_cols, prefix = dashboard_slice(cols)
    dashboard(dash_cols, os.path.join(FAN, "fanout-dashboard.png"), prefix)
    model_scaling(os.path.join(FAN, "fanout-model-scaling.png"))


if __name__ == "__main__":
    main()
