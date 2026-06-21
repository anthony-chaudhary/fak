#!/usr/bin/env python3
"""
fleet_eraser.py — the "finer eraser fixes the write-crossover" comparison figure.

The 2-D sweep found that cross-agent sharing flips to a NET LOSS once ~1% of fleet
actions are writes, because the v0.1 vDSO bumps the GLOBAL world-version on any
write (one booking strands every agent's warmed reads). The finer erasers
(`internal/vdso/scope.go`: namespace = per-resource-class, resource = per-entity)
evict only what actually changed. This plots cross_uplift vs write rate for all
three erasers over the SAME generated work — so the fix is one glance: the global
line dives through zero; the resource line stays flat and positive.

Reads `experiments/fleet/eraser/eraser-summary.csv`. White background (repo
convention). Run: python tools/fleet_eraser.py
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


def rows_for(path, eraser, axis):
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["eraser"] == eraser and r["axis"] == axis:
                out.append((float(r["x"]), float(r["cross_uplift_mean"])))
    out.sort()
    return np.array([x for x, _ in out]), np.array([y for _, y in out])


def main():
    summ = os.path.join(FLEET, "eraser", "eraser-summary.csv")
    if not os.path.exists(summ):
        print(f"need {summ}", file=sys.stderr)
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(9, 5.6), facecolor="white")
    ax.set_facecolor("white")
    styles = {
        "global":    ("#C9453F", "o-", "global flush (v0.1 baseline) — any write clears ALL"),
        "namespace": ("#C9A227", "s-", "namespace eraser — clears one resource CLASS"),
        "resource":  ("#3FA45F", "D-", "resource eraser — clears one ENTITY only"),
    }
    for eraser, (color, fmt, label) in styles.items():
        xs, ys = rows_for(summ, eraser, "write_rate")
        if len(xs):
            ax.plot(xs * 100, ys, fmt, ms=4, color=color, label=label)

    ax.axhline(0, color="#444", lw=1, ls="--")
    ax.set_xlabel("fleet write rate  (% of turns that change the world)")
    ax.set_ylabel("cross-agent uplift  (turns deleted by sharing, A=50 T=30)")
    ax.set_title("The finer eraser fixes the write-crossover\n"
                 "(global flush loses sharing at ~1% writes; the resource eraser keeps it)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    # mark the global crossover
    ax.annotate("v0.1 sharing turns into a LOSS here\n(~1% writes)",
                xy=(1.0, 0), xytext=(2.2, 140), fontsize=9, color="#7A1F1B",
                arrowprops=dict(arrowstyle="->", color="#C9453F"))

    fig.tight_layout()
    out = os.path.join(FLEET, "fleet-eraser.png")
    fig.savefig(out, dpi=120, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
