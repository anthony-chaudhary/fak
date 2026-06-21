#!/usr/bin/env python3
"""
fleet_heatmap.py — render the 2-D fleet turn-tax sweep surfaces as PNG heatmaps.

Reads the cmd/fleetbench CSVs under fak/experiments/fleet/ and renders, on the
turns (T) × agents (A) grid:

  * cross_uplift  — the turns the SHARED-cache fleet deletes that the same agents
    isolated cannot (the cross-agent dedup payoff). The headline surface.
  * shared_saved  — total turns the fleet deletes.
  * a 1-D write-rate crossover curve and the shared-pool slope law, from the axis
    scans.

White background (the repo's visuals convention). Run: python tools/fleet_heatmap.py
"""
import csv
import glob
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
    if not rows:
        return None
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0].keys()}


def grid(cols, zcol):
    Ts = np.array(sorted(set(cols["turns"].tolist())))
    As = np.array(sorted(set(cols["agents"].tolist())))
    Z = np.full((len(As), len(Ts)), np.nan)
    ti = {t: i for i, t in enumerate(Ts)}
    ai = {a: i for i, a in enumerate(As)}
    for k in range(len(cols["turns"])):
        Z[ai[cols["agents"][k]], ti[cols["turns"][k]]] = cols[zcol][k]
    return Ts, As, Z


def heatmap(ax, Ts, As, Z, title, cmap, cbar_label):
    im = ax.imshow(Z, origin="lower", aspect="auto", cmap=cmap,
                   extent=[Ts.min(), Ts.max(), As.min(), As.max()])
    ax.set_xlabel("turns per agent  (T)")
    ax.set_ylabel("agents in fleet  (A)")
    ax.set_title(title)
    cb = plt.colorbar(im, ax=ax)
    cb.set_label(cbar_label)


def main():
    head_path = os.path.join(FLEET, "readfleet-50x50.csv")
    if not os.path.exists(head_path):
        print(f"headline CSV not present at {head_path}", file=sys.stderr)
        sys.exit(1)
    head = load_csv(head_path)

    # --- figure 1: the two headline heatmaps side by side ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor="white")
    for ax in axes:
        ax.set_facecolor("white")
    Ts, As, Zc = grid(head, "cross_uplift_mean")
    heatmap(axes[0], Ts, As, Zc, "Cross-agent uplift (turns)\nshared-cache fleet − isolated agents",
            "viridis", "turns deleted by sharing")
    Ts, As, Zs = grid(head, "shared_saved_mean")
    heatmap(axes[1], Ts, As, Zs, "Total fleet turns deleted (shared)",
            "magma", "turns saved")
    fig.suptitle("fak fleet turn-tax: 2-D sweep (read-only fleet, shared_pool=8)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p1 = os.path.join(FLEET, "fleet-heatmap.png")
    fig.savefig(p1, dpi=120, facecolor="white")
    print(f"wrote {p1}")

    # --- figure 2: companion axes (write-rate crossover + pool slope law) ---
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")
    for ax in axes2:
        ax.set_facecolor("white")

    # write-rate crossover at A=50
    rows = []
    for p in glob.glob(os.path.join(FLEET, "writeaxis-w*.csv")):
        w = float(os.path.basename(p)[len("writeaxis-w"):-len(".csv")])
        c = load_csv(p)
        if c is None:
            continue
        m = c["agents"] == 50
        if np.any(m):
            rows.append((w, float(c["cross_uplift_mean"][m][0])))
    rows.sort()
    if rows:
        ws = [r[0] for r in rows]
        cu = [r[1] for r in rows]
        ax = axes2[0]
        ax.plot(ws, cu, "o-", color="#1f77b4")
        ax.axhline(0, color="#888", lw=1, ls="--")
        ax.set_xscale("symlog", linthresh=0.005)
        ax.set_xlabel("fleet write rate  (P(write) per turn)")
        ax.set_ylabel("cross-agent uplift  (A=50, T=30)")
        ax.set_title("Write-rate crossover\n(global world-bump invalidates the shared cache)")
        ax.grid(True, alpha=0.3)

    # shared-pool slope law
    prows = []
    for p in glob.glob(os.path.join(FLEET, "poolaxis-p*.csv")):
        pool = int(os.path.basename(p)[len("poolaxis-p"):-len(".csv")])
        c = load_csv(p)
        if c is None:
            continue
        A = c["agents"]
        y = c["cross_uplift_mean"]
        x = A - 1
        denom = float(np.sum(x * x))
        slope = float(np.sum(x * y) / denom) if denom else float("nan")
        prows.append((pool, slope))
    prows.sort()
    if prows:
        pools = [r[0] for r in prows]
        slopes = [r[1] for r in prows]
        ax = axes2[1]
        ax.plot(pools, slopes, "s-", color="#d62728", label="measured per-agent slope")
        ax.plot(pools, pools, ":", color="#888", label="slope = pool (no-saturation ideal)")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("shared catalog size  (distinct shared reads)")
        ax.set_ylabel("per-agent uplift slope  (T=30)")
        ax.set_title("Shared-pool slope law\n(per-agent uplift = min(pool, routes covered in T))")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig2.tight_layout()
    p2 = os.path.join(FLEET, "fleet-axes.png")
    fig2.savefig(p2, dpi=120, facecolor="white")
    print(f"wrote {p2}")


if __name__ == "__main__":
    main()
