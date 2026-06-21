#!/usr/bin/env python3
"""Plot node-macos-a Mac turn-agent sweep against the bounded-long baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAC = (
    ROOT
    / "fak"
    / "experiments"
    / "fleet-nodes"
    / "private-bench-node"
    / "turn-agent-mac-longer"
    / "turn-agent-fak-q8.json"
)
DEFAULT_BASELINE = (
    ROOT
    / "fak"
    / "experiments"
    / "model-baseline"
    / "turn-agent-realistic"
    / "bounded-long-compare.json"
)
DEFAULT_OUT = DEFAULT_MAC.with_name("mac-longer-vs-baseline.png")

MAC = "#1f6feb"
MAC_DARK = "#0a3069"
FAK_BASE = "#2da44e"
LLAMA = "#8250df"
GRID = "#d0d7de"
TEXT = "#24292f"
MUTED = "#57606a"


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def mac_series(mac: dict) -> dict[int, list[tuple[int, float]]]:
    out: dict[int, list[tuple[int, float]]] = {}
    for point in mac["points"]:
        out.setdefault(int(point["turns"]), []).append(
            (
                int(point["concurrency"]),
                float(point["reuse_agent_turns_per_sec"]),
            )
        )
    for points in out.values():
        points.sort()
    return dict(sorted(out.items()))


def baseline_series(baseline: dict, field: str) -> dict[int, list[tuple[int, float]]]:
    out: dict[int, list[tuple[int, float]]] = {}
    for cell in baseline["cells"]:
        out.setdefault(int(cell["turns"]), []).append(
            (
                int(cell["agents"]),
                float(cell[field]),
            )
        )
    for points in out.values():
        points.sort()
    return dict(sorted(out.items()))


def point_lookup(points: list[dict], turns: int, agents: int, field: str) -> float | None:
    for point in points:
        a = int(point.get("concurrency", point.get("agents", -1)))
        if int(point["turns"]) == turns and a == agents:
            return float(point[field])
    return None


def baseline_lookup(cells: list[dict], turns: int, agents: int, field: str) -> float | None:
    for cell in cells:
        if int(cell["turns"]) == turns and int(cell["agents"]) == agents:
            return float(cell[field])
    return None


def annotate_points(ax, xs, ys, color, yoffset=8):
    for x, y in zip(xs, ys):
        ax.annotate(
            f"{y:.2f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, yoffset),
            ha="center",
            fontsize=8,
            color=color,
            fontweight="bold",
        )


def plot(mac_path: Path, baseline_path: Path, out_path: Path) -> None:
    mac = load_json(mac_path)
    baseline = load_json(baseline_path)
    mac_by_t = mac_series(mac)
    base_fak_by_t = baseline_series(baseline, "fak_reuse_agent_turns_per_sec")
    base_llama_by_t = baseline_series(baseline, "llama_agent_turns_per_sec")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(13.5, 8.2), facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.45, 1.0], hspace=0.34, wspace=0.28)
    ax = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    fig.suptitle(
        "Mac longer turn-agent sweep vs bounded-long baseline",
        fontsize=15,
        fontweight="bold",
        color=TEXT,
        y=0.98,
    )
    fig.text(
        0.5,
        0.943,
        "Metric: agent-turns/sec, higher is better. Mac = FAK Q8 on Apple M3 Pro; baseline = Zen5/WSL FAK + llama.cpp endpoint sweep.",
        ha="center",
        fontsize=9,
        color=MUTED,
    )

    # Main throughput shape.
    markers = {80: "o", 120: "^"}
    colors = {80: MAC, 120: MAC_DARK}
    for turns, points in mac_by_t.items():
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(
            xs,
            ys,
            marker=markers.get(turns, "o"),
            lw=2.7,
            ms=7,
            color=colors.get(turns, MAC),
            label=f"Mac FAK, T={turns} (P=2048, R=16)",
        )
        annotate_points(ax, xs, ys, colors.get(turns, MAC), yoffset=(7 if turns == 80 else -15))

    for turns, points in base_fak_by_t.items():
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        alpha = 0.9 if turns == 80 else 0.35
        ax.plot(
            xs,
            ys,
            "--s",
            lw=2.1,
            ms=6,
            color=FAK_BASE,
            alpha=alpha,
            label=f"Baseline FAK, T={turns} (P=1024, R=48)",
        )
    for turns, points in base_llama_by_t.items():
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        alpha = 0.9 if turns == 80 else 0.35
        ax.plot(
            xs,
            ys,
            ":D",
            lw=2.0,
            ms=5.5,
            color=LLAMA,
            alpha=alpha,
            label=f"Baseline llama.cpp, T={turns} (P=1024, R=48)",
        )

    ax.set_title("Throughput shape across agent count", loc="left", fontsize=11, fontweight="bold", color=TEXT)
    ax.set_xlabel("live agents")
    ax.set_ylabel("agent-turns/sec")
    ax.set_xticks([8, 12, 16, 20])
    ax.set_xlim(7, 21)
    ax.set_ylim(0, max(3.7, ax.get_ylim()[1]))
    ax.grid(True, axis="both", color=GRID, alpha=0.7, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=2, fontsize=8, frameon=False, loc="upper right")
    ax.text(
        0.01,
        0.04,
        "Only T=80/A=8 and T=80/A=20 overlap exactly by turns+agents; profile context differs.",
        transform=ax.transAxes,
        fontsize=8.5,
        color=MUTED,
    )

    # Overlap absolute comparison.
    overlap = [(80, 8), (80, 20)]
    labels = [f"T={t}\nA={a}" for t, a in overlap]
    mac_vals = [
        point_lookup(mac["points"], t, a, "reuse_agent_turns_per_sec") for t, a in overlap
    ]
    fak_vals = [
        baseline_lookup(baseline["cells"], t, a, "fak_reuse_agent_turns_per_sec") for t, a in overlap
    ]
    llama_vals = [
        baseline_lookup(baseline["cells"], t, a, "llama_agent_turns_per_sec") for t, a in overlap
    ]
    if any(v is None for v in mac_vals + fak_vals + llama_vals):
        raise ValueError("missing overlap cells for T=80/A=8 or T=80/A=20")

    x = np.arange(len(overlap))
    width = 0.24
    b1 = ax2.bar(x - width, fak_vals, width, color=FAK_BASE, label="Baseline FAK")
    b2 = ax2.bar(x, llama_vals, width, color=LLAMA, label="Baseline llama.cpp")
    b3 = ax2.bar(x + width, mac_vals, width, color=MAC, label="Mac FAK")
    ax2.set_title("Overlapping cells: absolute throughput", loc="left", fontsize=11, fontweight="bold", color=TEXT)
    ax2.set_ylabel("agent-turns/sec")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylim(0, max(fak_vals + llama_vals + mac_vals) * 1.28)
    ax2.grid(True, axis="y", color=GRID, alpha=0.7, lw=0.8)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.legend(fontsize=8, frameon=False)
    for bars in (b1, b2, b3):
        for bar in bars:
            ax2.annotate(
                f"{bar.get_height():.2f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
                fontsize=8,
                color=TEXT,
            )

    # Overlap normalized comparison.
    llama_ratio = [ll / fak for ll, fak in zip(llama_vals, fak_vals)]
    mac_ratio = [mv / fak for mv, fak in zip(mac_vals, fak_vals)]
    b4 = ax3.bar(x - width / 2, llama_ratio, width, color=LLAMA, label="Baseline llama / baseline FAK")
    b5 = ax3.bar(x + width / 2, mac_ratio, width, color=MAC, label="Mac FAK / baseline FAK")
    ax3.axhline(1.0, color=FAK_BASE, lw=1.5, ls="--")
    ax3.text(0.02, 1.02, "baseline FAK = 1.00x", transform=ax3.get_yaxis_transform(), fontsize=8, color=FAK_BASE)
    ax3.set_title("Overlapping cells: normalized to baseline FAK", loc="left", fontsize=11, fontweight="bold", color=TEXT)
    ax3.set_ylabel("relative throughput")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.set_ylim(0, 1.25)
    ax3.grid(True, axis="y", color=GRID, alpha=0.7, lw=0.8)
    ax3.spines[["top", "right"]].set_visible(False)
    ax3.legend(fontsize=8, frameon=False, loc="upper right")
    for bars in (b4, b5):
        for bar in bars:
            ax3.annotate(
                f"{bar.get_height():.2f}x",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
                fontsize=8,
                color=TEXT,
            )

    footer = (
        f"Mac artifact: {mac_path.relative_to(ROOT)}  |  "
        f"Baseline artifact: {baseline_path.relative_to(ROOT)}"
    )
    fig.text(0.5, 0.013, footer, ha="center", fontsize=7.5, color=MUTED)
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.09, top=0.88, hspace=0.38, wspace=0.28)
    fig.savefig(out_path, dpi=140)
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)
    print(f"wrote {out_path}")
    print(f"wrote {out_path.with_suffix('.svg')}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mac", type=Path, default=DEFAULT_MAC)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    plot(args.mac.resolve(), args.baseline.resolve(), args.out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
