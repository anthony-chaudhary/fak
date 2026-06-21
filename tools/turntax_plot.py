#!/usr/bin/env python3
"""
turntax_plot.py — render the turn-tax break-even curve as a PNG.

Reads fak/experiments/turn-tax/turntax-breakeven.json (produced by
`fak turntax --breakeven`, harness internal/turnbench/stochastic.go) and renders:

  * turntax-breakeven.png — the 2-panel headline over the addressable hit-rate h:
      (a) MEASURED expected model turns deleted per session vs h (real-kernel
          replay), with the §3.1 real-world ~0.7% point and the airline demo
          slice (9/14 ≈ 64%) marked — the answer to "is 9 cherry-picked";
      (b) MODELED self-host amortization: sessions to recover the §3.1 ~$2.8M
          fork on the turn-saving alone, vs h. At the real rate it never pays;
          the provider-ships regime ($0 marginal) breaks even on session one.

White background (the repo's visuals convention, matching tools/fanout_plot.py).
MEASURED vs MODELED is labelled on each panel — the two halves are never blended.
Run: python tools/turntax_plot.py
"""
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
TT = os.path.join(HERE, "..", "fak", "experiments", "turn-tax")

MEAS = "#1f6feb"   # measured (blue)
MODEL = "#d29922"  # modeled (amber)
REAL = "#cf222e"   # the real-world rate marker (red)
SHADE = "#2da44e"  # fill / positive (green)

# The airline demo slice is 9 saved turns over 14 calls — its addressable share,
# plotted as the "this is the high end of the curve" reference, not a swept point.
AIRLINE_H = 9.0 / 14.0


def tag(ax, text, color, loc="top"):
    y, va = (0.97, "top") if loc == "top" else (0.03, "bottom")
    ax.text(0.015, y, text, transform=ax.transAxes, fontsize=7.5, va=va,
            ha="left", color="white",
            bbox=dict(boxstyle="round,pad=0.25", fc=color, ec="none", alpha=0.9))


def load(path):
    with open(path) as f:
        return json.load(f)


def self_host_sessions(point):
    for rg in point["regimes"]:
        if rg["name"] == "self-host-fork":
            return rg["sessions_to_break_even"]
    return None


def plot(rep, out):
    pts = sorted(rep["points"], key=lambda p: p["hit_rate"])
    # drop the h=0 control from the log-x sweep (it is the floor, annotated in text)
    swept = [p for p in pts if p["hit_rate"] > 0]
    h = np.array([p["hit_rate"] for p in swept])
    turns = np.array([p["mean_turns_saved"] for p in swept])
    sessions = np.array([self_host_sessions(p) for p in swept], dtype=float)
    real = rep["real_world_hit_rate"]

    realpt = next(p for p in swept if p["hit_rate"] == real)
    real_turns = realpt["mean_turns_saved"]
    real_sess = self_host_sessions(realpt)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    fig.suptitle(
        "turn-tax break-even — turns saved per session vs the addressable hit-rate  h\n"
        f"(clean {rep['base_calls']}-call base, {rep['trials']} real-kernel trials/point; "
        "the §3.1 real-world rate is ~0.7%)",
        fontsize=12.5, fontweight="bold")

    # (a) MEASURED expected turns deleted per session
    ax = axes[0]
    ax.plot(h * 100, turns, "-o", color=MEAS, ms=5, label="expected turns saved / session")
    ax.axhline(1.0, color="#888", ls=":", lw=1)
    ax.text(h[0] * 100, 1.06, "one turn / session", fontsize=7.5, color="#666")
    # real-world ~0.7% marker
    ax.scatter([real * 100], [real_turns], color=REAL, zorder=6, s=55)
    ax.annotate(f"real-world ~{real*100:.1f}%:\n{real_turns:.2f} turns/session\n(below 1 — §3.1 marginal)",
                xy=(real * 100, real_turns), xytext=(14, 30), textcoords="offset points",
                fontsize=8.5, color=REAL, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=REAL, lw=1.2))
    # airline demo slice reference
    ax.axvline(AIRLINE_H * 100, color=SHADE, ls="--", lw=1.2, alpha=0.8)
    ax.annotate(f"airline demo slice\n(9/14 ≈ {AIRLINE_H*100:.0f}%) —\nthe far HIGH end",
                xy=(AIRLINE_H * 100, turns[-1] * 0.5), xytext=(-120, 0),
                textcoords="offset points", fontsize=8, color=SHADE, fontweight="bold",
                va="center")
    ax.set_xscale("log")
    ax.set_xlabel("addressable hit-rate  h  (% of eligible calls)")
    ax.set_ylabel("model turns deleted per session  (mean)")
    ax.set_title("(a) The win is error-rate-dependent — and small at the real rate")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.9))
    tag(ax, "MEASURED — real k.Syscall replay, per point", MEAS)

    # (b) MODELED self-host amortization
    ax = axes[1]
    ax.plot(h * 100, sessions, "-o", color=MODEL, ms=5)
    ax.scatter([real * 100], [real_sess], color=REAL, zorder=6, s=55)
    ax.annotate(f"real-world ~{real*100:.1f}%:\n~{real_sess/1e9:.1f} BILLION sessions\n(never pays on efficiency)",
                xy=(real * 100, real_sess), xytext=(22, -52), textcoords="offset points",
                fontsize=8.5, color=REAL, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=REAL, lw=1.2))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("addressable hit-rate  h  (% of eligible calls)")
    ax.set_ylabel("sessions to recover the ~$2.8M self-host fork  (log)")
    ax.set_title("(b) Self-host amortization — turn-saving alone never justifies the fork")
    ax.grid(True, which="both", alpha=0.25)
    # provider-ships reference line: $0 marginal => break-even on session one
    ax.axhline(1, color=SHADE, ls="--", lw=1.2)
    ax.text(h[-1] * 100, 1.4, "provider-ships ($0 marginal): break-even on session one",
            fontsize=7.5, color=SHADE, ha="right")
    tag(ax, "MODELED — §3.1 build-cost amortization", MODEL, loc="bottom")

    fig.text(0.5, 0.005,
             "The deterministic SAFETY FLOOR (1 injection quarantined, 1 destructive op denied) holds at EVERY h, including 0 — "
             "it is the engine-agnostic reason to run the kernel; the turn-saving above is self-host-regime upside.",
             ha="center", fontsize=8.5, color="#444", style="italic")

    fig.tight_layout(rect=[0, 0.035, 0.995, 0.93])
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print("wrote", out)


def main():
    path = os.path.join(TT, "turntax-breakeven.json")
    if not os.path.exists(path):
        raise SystemExit(
            "missing fak/experiments/turn-tax/turntax-breakeven.json — "
            "run `fak turntax --breakeven` first")
    rep = load(path)
    plot(rep, os.path.join(TT, "turntax-breakeven.png"))


if __name__ == "__main__":
    main()
