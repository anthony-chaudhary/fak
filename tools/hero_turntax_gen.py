#!/usr/bin/env python3
"""Generate the fak TURN-TAX efficiency curves — visuals/60-hero-turntax-curves.svg.

fak's analog of the MiniMax "GQA vs MSA" 3-panel attention-efficiency chart: in
each panel a baseline curve rises ~linearly while the fak curve stays ~flat, the
gap between them is shaded, and one big multiplier sits in the field. The shape is
not cosmetic — re-prefill cost IS linear in context / fleet / turns, and resident,
addressable KV is not — so the visual grammar of "a steep line and a flat line"
is the honest picture, not a stylisation.

Each panel names its OWN baseline so the framing stays honest:
  1. per-turn prefill cost vs context     — baseline = re-prefill on a cold KV miss
  2. WebVoyager fleet prefill (MODELED)    — baseline = naive re-prefill floor, 643 tasks
  3. 50-turn x 5-agent fleet               — baseline = a tuned warm-cache SOTA stack

  Source of truth : tools/hero_turntax.data.json
  Emits           : visuals/60-hero-turntax-curves.svg   (meta.out_svg)

Usage:
  python tools/hero_turntax_gen.py            # (re)write the SVG
  python tools/hero_turntax_gen.py --check     # don't write; exit 1 if it drifted

Pure stdlib — runs in the `python -m pytest tools` gate and the CI drift step.
"""
from __future__ import annotations

import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA = os.path.join(ROOT, "tools", "hero_turntax.data.json")
DEFAULT_OUT = os.path.join("visuals", "60-hero-turntax-curves.svg")

# House palette (matches the hero statcard + visuals/ set; reference uses navy vs teal).
BASE, BASE_FILL = "#3b4a7a", "#e7ebf4"
FAK, FAK_FILL = "#2e8b57", "#e9f5ee"
INK, BODY, MUTED, MICRO = "#14202b", "#415266", "#5b6a7a", "#8090a0"
GRID, AXIS = "#e3e9ee", "#b9c4cf"

# geometry
W = 1320
MARGIN_L, MARGIN_R = 64, 64
PANEL_GAP = 40
N_PANELS = 3
PANEL_W = (W - MARGIN_L - MARGIN_R - (N_PANELS - 1) * PANEL_GAP) // N_PANELS  # 370
PLOT_PAD_L, PLOT_PAD_R = 40, 10
PANELS_TOP = 196
PLOT_H = 250
TITLE_GAP = 56          # space above the plot for the per-panel title + subtitle
H = PANELS_TOP + TITLE_GAP + PLOT_H + 120  # axis labels + legend + footer

STYLE = '''  <defs>
    <linearGradient id="tt-bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#fbfcfd"/>
      <stop offset="60%" stop-color="#f4f7f9"/>
      <stop offset="100%" stop-color="#f6f4f0"/>
    </linearGradient>
    <style>
      .k    { font: 800 14px "Segoe UI", Arial, sans-serif; fill: #2e8b57; letter-spacing: 1.6px; }
      .ti   { font: 900 33px "Segoe UI", Arial, sans-serif; fill: #14202b; }
      .sub  { font: 400 16px "Segoe UI", Arial, sans-serif; fill: #415266; }
      .pt   { font: 800 18px "Segoe UI", Arial, sans-serif; fill: #14202b; }
      .pst  { font: 400 12.5px "Segoe UI", Arial, sans-serif; fill: #5b6a7a; }
      .axt  { font: 600 12px "Segoe UI", Arial, sans-serif; fill: #6a7685; }
      .axl  { font: 700 12.5px "Segoe UI", Arial, sans-serif; fill: #415266; }
      .ytk  { font: 400 11px "Segoe UI", Arial, sans-serif; fill: #8090a0; }
      .mult { font: 900 50px "Segoe UI", Arial, sans-serif; fill: #14202b; }
      .msub { font: 700 13px "Segoe UI", Arial, sans-serif; fill: #415266; }
      .mfoot{ font: 400 11.5px "Segoe UI", Arial, sans-serif; fill: #6a7685; }
      .leg  { font: 600 14px "Segoe UI", Arial, sans-serif; fill: #1b2733; }
      .foot { font: 400 12px "Segoe UI", Arial, sans-serif; fill: #6a7685; }
    </style>
  </defs>'''


def esc(s: object) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def load_data(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def read_text(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def panel_x(i: int) -> int:
    return MARGIN_L + i * (PANEL_W + PANEL_GAP)


def render_panel(out: list, i: int, p: dict) -> None:
    px = panel_x(i)
    plot_x0 = px + PLOT_PAD_L
    plot_x1 = px + PANEL_W - PLOT_PAD_R
    plot_y0 = PANELS_TOP + TITLE_GAP
    plot_y1 = plot_y0 + PLOT_H
    plot_w = plot_x1 - plot_x0

    xs = p["x_vals"]
    x_lo, x_hi = xs[0], xs[-1]
    y_max = float(p["y_max"])

    def X(v):
        return plot_x0 + (v - x_lo) / (x_hi - x_lo) * plot_w

    def Y(v):
        return plot_y1 - (v / y_max) * PLOT_H

    # per-panel title + subtitle
    out.append(f'  <text x="{px}" y="{PANELS_TOP + 14}" class="pt">{esc(p["title"])}</text>')
    out.append(f'  <text x="{px}" y="{PANELS_TOP + 34}" class="pst">{esc(p["subtitle"])}</text>')

    # faint horizontal gridlines + y tick labels
    for tk in p.get("y_ticks", []):
        gy = Y(float(tk["v"]))
        out.append(f'  <line x1="{plot_x0}" y1="{gy:.1f}" x2="{plot_x1}" y2="{gy:.1f}" stroke="{GRID}" stroke-width="1"/>')
        out.append(f'  <text x="{plot_x0 - 6}" y="{gy + 4:.1f}" class="ytk" text-anchor="end">{esc(tk["t"])}</text>')

    # axes (left + bottom)
    out.append(f'  <line x1="{plot_x0}" y1="{plot_y0}" x2="{plot_x0}" y2="{plot_y1}" stroke="{AXIS}" stroke-width="1.5"/>')
    out.append(f'  <line x1="{plot_x0}" y1="{plot_y1}" x2="{plot_x1}" y2="{plot_y1}" stroke="{AXIS}" stroke-width="1.5"/>')

    base_pts = [(X(xs[j]), Y(float(p["baseline"][j]))) for j in range(len(xs))]
    fak_pts = [(X(xs[j]), Y(float(p["fak"][j]))) for j in range(len(xs))]

    # shaded gap between the two curves (baseline forward, fak back)
    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in base_pts)
    poly += " " + " ".join(f"{x:.1f},{y:.1f}" for x, y in reversed(fak_pts))
    out.append(f'  <polygon points="{poly}" fill="{BASE_FILL}" opacity="0.65"/>')

    # fak curve (flat, on top of its own faint fill to the axis)
    fak_area = " ".join(f"{x:.1f},{y:.1f}" for x, y in fak_pts)
    fak_area += f" {fak_pts[-1][0]:.1f},{plot_y1:.1f} {fak_pts[0][0]:.1f},{plot_y1:.1f}"
    out.append(f'  <polygon points="{fak_area}" fill="{FAK_FILL}" opacity="0.7"/>')

    def polyline(pts, color, wdt):
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        out.append(f'  <polyline points="{d}" fill="none" stroke="{color}" stroke-width="{wdt}" stroke-linejoin="round" stroke-linecap="round"/>')

    polyline(base_pts, BASE, 3)
    polyline(fak_pts, FAK, 3)

    # endpoint dots at the anchor index
    ai = int(p.get("anchor_idx", len(xs) - 1))
    for pts, color in ((base_pts, BASE), (fak_pts, FAK)):
        cx, cy = pts[ai]
        out.append(f'  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="{color}"/>')

    # x ticks
    for j, t in enumerate(p["x_ticks"]):
        tx = X(xs[j])
        out.append(f'  <text x="{tx:.1f}" y="{plot_y1 + 18}" class="axt" text-anchor="middle">{esc(t)}</text>')
    # x axis title
    out.append(f'  <text x="{(plot_x0 + plot_x1) / 2:.1f}" y="{plot_y1 + 38}" class="axl" text-anchor="middle">{esc(p["x_label"])}</text>')
    # y axis title (rotated)
    ylx = px + 12
    yly = (plot_y0 + plot_y1) / 2
    out.append(f'  <text x="{ylx}" y="{yly:.1f}" class="axl" text-anchor="middle" transform="rotate(-90 {ylx} {yly:.1f})">{esc(p["y_label"])}</text>')

    # big multiplier callout, parked in the open wedge BELOW the rising curve
    mx = plot_x0 + plot_w * 0.55
    my = plot_y0 + PLOT_H * 0.60
    out.append(f'  <text x="{mx:.1f}" y="{my:.1f}" class="mult" text-anchor="middle">{esc(p["mult"])}</text>')
    out.append(f'  <text x="{mx:.1f}" y="{my + 20:.1f}" class="msub" text-anchor="middle">{esc(p["mult_sub"])}</text>')
    # foot note wraps to two lines if long
    foot = p.get("mult_foot", "")
    out.append(f'  <text x="{mx:.1f}" y="{my + 40:.1f}" class="mfoot" text-anchor="middle">{esc(foot)}</text>')


def svg_turntax(d: dict) -> str:
    out = []
    out.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" role="img" aria-labelledby="tt-title tt-desc">'
    )
    out.append(f'  <title id="tt-title">{esc(d["meta"]["title"])}</title>')
    mults = " / ".join(p["mult"] for p in d["panels"])
    out.append(f'  <desc id="tt-desc">Three panels — per-turn prefill cost vs context, WebVoyager fleet prefill vs workers, and 50-turn fleet serving work vs turns. In each, a baseline re-prefill curve rises ~linearly while fak\'s resident-KV curve stays flat; multipliers {esc(mults)}. Panel 2 is modeled geometry vs the naive floor (643 WebVoyager tasks); panel 3 is the conservative number vs a tuned SOTA cache.</desc>')
    out.append(STYLE)

    out.append(f'  <rect width="{W}" height="{H}" fill="url(#tt-bg)"/>')

    # masthead
    out.append(f'  <text x="{MARGIN_L}" y="58" class="k">{esc(d["kicker"])}</text>')
    out.append(f'  <text x="{MARGIN_L}" y="96" class="ti">{esc(d["headline"])}</text>')
    out.append(f'  <text x="{MARGIN_L}" y="124" class="sub">{esc(d["subhead"])}</text>')

    for i, p in enumerate(d["panels"]):
        render_panel(out, i, p)

    # legend (centered, below the panels)
    leg_y = PANELS_TOP + TITLE_GAP + PLOT_H + 64
    items = d["legend"]
    # measure-ish: lay out around center
    seg = 430
    total = seg * len(items)
    start = (W - total) / 2 + 40
    for j, it in enumerate(items):
        lx = start + j * seg
        out.append(f'  <line x1="{lx:.0f}" y1="{leg_y - 5}" x2="{lx + 34:.0f}" y2="{leg_y - 5}" stroke="{it["color"]}" stroke-width="4" stroke-linecap="round"/>')
        out.append(f'  <circle cx="{lx + 17:.0f}" cy="{leg_y - 5}" r="4.5" fill="{it["color"]}"/>')
        out.append(f'  <text x="{lx + 46:.0f}" y="{leg_y}" class="leg">{esc(it["label"])}</text>')

    # footer
    out.append(f'  <text x="{MARGIN_L}" y="{H - 20}" class="foot">{esc(d["footer"])}</text>')

    out.append('</svg>\n')
    return "\n".join(out)


def out_path(d: dict) -> str:
    return os.path.join(ROOT, d.get("meta", {}).get("out_svg", DEFAULT_OUT))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generate the fak turn-tax efficiency curves SVG.")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--check", action="store_true", help="don't write; exit 1 if the on-disk SVG drifted from the data")
    args = ap.parse_args(argv)

    d = load_data(args.data)
    svg = svg_turntax(d)
    path = out_path(d)

    if args.check:
        if read_text(path) != svg:
            print(f"DRIFT — {os.path.relpath(path, ROOT)} is stale (run: python tools/hero_turntax_gen.py)")
            return 1
        print(f"check: {os.path.relpath(path, ROOT)} is up to date with {os.path.relpath(args.data, ROOT)}.")
        return 0

    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(svg)
    print(f"wrote {os.path.relpath(path, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
