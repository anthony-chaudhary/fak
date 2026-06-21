#!/usr/bin/env python3
"""
fleet_compare.py — the "fleet vs baseline, in absolute terms" comparison figure.

The heatmaps (fleet_heatmap.py) show the whole surface; this is the plain-language
companion the operator asked for: at a representative slice, plot the SHARED-cache
fleet against its ISOLATED-agents baseline as two lines, with the gap (the
cross-agent uplift) shaded — so "how much did sharing buy, in absolute turns
deleted, vs running the same agents apart" is one glance.

Two panels:
  * Panel A — fix agents A=50, sweep turns T: the T-axis (savings saturate).
  * Panel B — fix turns T=30, sweep agents A: the A-axis (baseline ~linear, the
    shared fleet pulls away — the cross gap widens with fleet size).

White background (repo convention). Run: python tools/fleet_compare.py
"""
import csv
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FLEET = os.path.join(HERE, "..", "fak", "experiments", "fleet")


def load_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0].keys()}


def slice_fixed(cols, key, val):
    """Return (xs, shared, isolated, cross) for the cells where cols[key]==val,
    sorted by the OTHER axis."""
    m = cols[key] == val
    other = "agents" if key == "turns" else "turns"
    order = np.argsort(cols[other][m])
    xs = cols[other][m][order]
    sh = cols["shared_saved_mean"][m][order]
    cr = cols["cross_uplift_mean"][m][order]
    iso = sh - cr  # isolated_saved_mean (CSV carries shared+cross means; isolated = shared − cross)
    return xs, sh, iso, cr


def panel(ax, xs, sh, iso, cr, xlabel, title):
    ax.plot(xs, iso, "o-", ms=3, color="#7A8A99",
            label="baseline: agents run ALONE (isolated)")
    ax.plot(xs, sh, "o-", ms=3, color="#1f77b4",
            label="fleet: agents SHARE the cache")
    ax.fill_between(xs, iso, sh, color="#3FA45F", alpha=0.25,
                    label="cross-agent uplift (what sharing bought)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("tool round-trips DELETED (absolute)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    # annotate the endpoint gap in absolute + relative terms
    gap = sh[-1] - iso[-1]
    pct = (gap / iso[-1] * 100) if iso[-1] > 0 else 0
    ax.annotate(f"+{gap:.0f} turns\n(+{pct:.0f}% over baseline)",
                xy=(xs[-1], sh[-1]), xytext=(xs[-1] * 0.62, sh[-1] * 0.92),
                fontsize=9, color="#1C5230",
                arrowprops=dict(arrowstyle="->", color="#3FA45F"))


def main():
    head = os.path.join(FLEET, "readfleet-50x50.csv")
    if not os.path.exists(head):
        print(f"need {head}", file=sys.stderr)
        sys.exit(1)
    cols = load_csv(head)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), facecolor="white")
    for ax in axes:
        ax.set_facecolor("white")

    # Panel A: fix the biggest fleet, sweep session length.
    Amax = int(cols["agents"].max())
    xs, sh, iso, cr = slice_fixed(cols, "agents", Amax)
    panel(axes[0], xs, sh, iso, cr, "turns per agent  (T)",
          f"Fix fleet = {Amax} agents · sweep session length\n(savings SATURATE in turns)")

    # Panel B: fix a mid session length, sweep fleet size.
    Tfix = 30.0 if np.any(cols["turns"] == 30) else cols["turns"].max()
    xs, sh, iso, cr = slice_fixed(cols, "turns", Tfix)
    panel(axes[1], xs, sh, iso, cr, "agents in fleet  (A)",
          f"Fix session = {int(Tfix)} turns · sweep fleet size\n(baseline ~flat per-agent; fleet pulls AWAY)")

    fig.suptitle("Fleet vs baseline — absolute tool round-trips deleted (read-only fleet, shared_pool=8)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(FLEET, "fleet-compare.png")
    fig.savefig(out, dpi=120, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
