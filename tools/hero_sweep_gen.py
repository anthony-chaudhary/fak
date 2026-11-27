#!/usr/bin/env python3
"""Generate the fak BENCHMARK SWEEP grid — visuals/61-hero-benchmark-sweep.svg.

fak's analog of the GLM-5.2 "LLM Performance Evaluation" 2x4 bar-grid: one small
grouped-bar panel per benchmark, a value label on top of each bar, a monogram
badge on the bar, the benchmark name in a pill below, the subject bar in the
bright accent. Eight axes: Inkling-2-Mini inference first, then the kernel's
serving and security axes, then an honest fence where fak still trails.

Honesty is structural:
  * every fak figure is the bright green bar and traces to a commit;
  * each panel compares fak against ONE named honest reference (a tuned-SOTA
    baseline, a deterministic ceiling, the optimal, a spawned hook, or — for the
    two capability panels — a field that ships no such gate), never a fabricated
    competitor number;
  * the final panel is the fence and INVERTS the accent: the engine that leads
    owns the tall navy bar, fak shows its honest lower number in grey.

  Source of truth : tools/hero_sweep.data.json
  Emits           : visuals/61-hero-benchmark-sweep.svg   (meta.out_svg)

Usage:
  python tools/hero_sweep_gen.py            # (re)write the SVG
  python tools/hero_sweep_gen.py --check     # don't write; exit 1 if it drifted

Pure stdlib — runs in the `python -m pytest tools` gate and the CI drift step.
"""
from __future__ import annotations

import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA = os.path.join(ROOT, "tools", "hero_sweep.data.json")
DEFAULT_OUT = os.path.join("visuals", "61-hero-benchmark-sweep.svg")

# House palette
GREEN, GREEN_DK = "#2e8b57", "#1f6b46"
NAVY = "#3b4a7a"
GREY_BAR, GREY_STROKE = "#cdd5dd", "#aeb9c4"
INK, BODY, MUTED, MICRO = "#14202b", "#415266", "#5b6a7a", "#8090a0"
PILLAR = {"INFERENCE": "#2e6fb0", "SERVING": "#3a7d9c", "CORRECTNESS": "#7a5cb0", "SECURITY": "#1c7c92", "FENCE": "#6a7685"}

# geometry
W = 1320
MARGIN = 56
N_COLS, N_ROWS = 4, 2
COL_GAP, ROW_GAP = 24, 26
PC_W = (W - 2 * MARGIN - (N_COLS - 1) * COL_GAP) // N_COLS  # 284
PLOT_H = 150
BAR_W = 58
BAR_GAP = 30
GRID_TOP = 206
PC_H = PLOT_H + 118        # plot + name pill + foot + commit
ROW_PITCH = PC_H + ROW_GAP
H = GRID_TOP + N_ROWS * PC_H + ROW_GAP + 92

STYLE = '''  <defs>
    <linearGradient id="sw-bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#fbfcfd"/>
      <stop offset="60%" stop-color="#f4f7f9"/>
      <stop offset="100%" stop-color="#f7f4f0"/>
    </linearGradient>
    <style>
      .k    { font: 800 14px "Segoe UI", Arial, sans-serif; fill: #2e8b57; letter-spacing: 1.6px; }
      .ti   { font: 900 32px "Segoe UI", Arial, sans-serif; fill: #14202b; }
      .sub  { font: 400 15px "Segoe UI", Arial, sans-serif; fill: #415266; }
      .val  { font: 900 17px "Segoe UI", Arial, sans-serif; fill: #14202b; }
      .valm { font: 700 14px "Segoe UI", Arial, sans-serif; fill: #6a7685; }
      .bdg  { font: 800 12px "Segoe UI", Arial, sans-serif; }
      .who  { font: 600 11px "Segoe UI", Arial, sans-serif; fill: #6a7685; }
      .pill { font: 800 13px "Segoe UI", Arial, sans-serif; fill: #1b2733; }
      .foot { font: 400 11.5px "Segoe UI", Arial, sans-serif; fill: #6a7685; }
      .cm   { font: 400 10.5px "Segoe UI", Arial, sans-serif; fill: #9aa6b2; }
      .lmul { font: 900 15px "Segoe UI", Arial, sans-serif; fill: #1c7c92; }
      .leg  { font: 600 13.5px "Segoe UI", Arial, sans-serif; fill: #1b2733; }
      .ftr  { font: 400 12px "Segoe UI", Arial, sans-serif; fill: #6a7685; }
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


def wrap(text: str, max_chars: int):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > max_chars:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return lines


def render_panel(out: list, p: dict, px: int, py: int) -> None:
    baseline = py + PLOT_H
    fence = bool(p.get("fence"))
    pillar = p.get("pillar", "SERVING")
    accent = PILLAR.get(pillar, "#3a7d9c")

    bars = p["bars"]
    y_max = float(p["y_max"])
    bar_w = 50 if len(bars) >= 3 else BAR_W
    bar_gap = 18 if len(bars) >= 3 else BAR_GAP
    total_bw = len(bars) * bar_w + (len(bars) - 1) * bar_gap
    bx0 = px + (PC_W - total_bw) // 2

    # baseline rule
    out.append(f'  <line x1="{px + 12}" y1="{baseline}" x2="{px + PC_W - 12}" y2="{baseline}" stroke="#dfe5ea" stroke-width="1.5"/>')

    for j, b in enumerate(bars):
        subj = bool(b.get("subject"))
        bx = bx0 + j * (bar_w + bar_gap)
        v = float(b["v"])
        bh = max(0.0, (v / y_max) * PLOT_H)
        bt = baseline - bh
        if subj:
            fill = NAVY if fence else GREEN
            stroke = NAVY if fence else GREEN_DK
        else:
            fill, stroke = GREY_BAR, GREY_STROKE

        if bh >= 2:
            out.append(f'  <rect x="{bx}" y="{bt:.1f}" width="{bar_w}" height="{bh:.1f}" rx="5" fill="{fill}" stroke="{stroke}" stroke-width="1"/>')
        else:
            # zero/near-zero capability bar — a faint stub on the baseline
            out.append(f'  <rect x="{bx}" y="{baseline - 3}" width="{bar_w}" height="3" rx="1.5" fill="{GREY_BAR}"/>')

        # value label above the bar
        vy = (bt - 10) if bh >= 2 else (baseline - 12)
        vcls = "val" if subj else "valm"
        out.append(f'  <text x="{bx + bar_w // 2}" y="{vy:.1f}" class="{vcls}" text-anchor="middle">{esc(b["label"])}</text>')

        # monogram badge — on the bar if tall enough, else at the baseline
        by = (bt + 16) if bh >= 26 else (baseline - 16)
        btxt = ("#ffffff" if (subj and bh >= 26) else MUTED)
        bg = (fill if (subj and bh >= 26) else "#ffffff")
        bstroke = stroke if (subj and bh >= 26) else "#c2ccd6"
        cxb = bx + bar_w // 2
        out.append(f'  <rect x="{cxb - 9}" y="{by - 12:.1f}" width="18" height="18" rx="5" fill="{bg}" stroke="{bstroke}" stroke-width="1.1"/>')
        out.append(f'  <text x="{cxb}" y="{by + 1:.1f}" class="bdg" text-anchor="middle" style="fill:{btxt};">{esc(b["badge"])}</text>')
        # who label under the baseline
        out.append(f'  <text x="{cxb}" y="{baseline + 15}" class="who" text-anchor="middle">{esc(b["who"])}</text>')

    # optional multiplier / callout tag floating in the panel
    if p.get("callout"):
        callout_x = (px + 18) if p.get("lower_is_better") else (px + PC_W - 16)
        callout_anchor = "start" if p.get("lower_is_better") else "end"
        out.append(f'  <text x="{callout_x}" y="{py + 20}" class="lmul" text-anchor="{callout_anchor}">{esc(p["callout"])}</text>')
    elif p.get("log_multiplier"):
        out.append(f'  <text x="{px + PC_W - 16}" y="{py + 20}" class="lmul" text-anchor="end">{esc(p["log_multiplier"])} faster</text>')

    # name pill (pillar-coloured outline)
    pill_y = baseline + 30
    name = p["name"]
    pill_w = min(PC_W - 24, 13 + len(name) * 8)
    pill_cx = px + PC_W // 2
    out.append(f'  <rect x="{pill_cx - pill_w // 2}" y="{pill_y}" width="{pill_w}" height="26" rx="13" fill="#ffffff" stroke="{accent}" stroke-width="1.5"/>')
    out.append(f'  <text x="{pill_cx}" y="{pill_y + 18}" class="pill" text-anchor="middle">{esc(name)}</text>')
    # foot note + commit
    out.append(f'  <text x="{pill_cx}" y="{pill_y + 44}" class="foot" text-anchor="middle">{esc(p.get("foot", ""))}</text>')
    out.append(f'  <text x="{pill_cx}" y="{pill_y + 60}" class="cm" text-anchor="middle">{esc(p.get("commit", ""))}</text>')


def svg_sweep(d: dict) -> str:
    panels = d["panels"]
    out = []
    out.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" role="img" aria-labelledby="sw-title sw-desc">'
    )
    out.append(f'  <title id="sw-title">{esc(d["meta"]["title"])}</title>')
    out.append(f'  <desc id="sw-desc">{esc(d.get("desc", d["subtitle"]))}</desc>')
    out.append(STYLE)
    out.append(f'  <rect width="{W}" height="{H}" fill="url(#sw-bg)"/>')

    # masthead
    out.append(f'  <text x="{MARGIN}" y="58" class="k">{esc(d["kicker"])}</text>')
    out.append(f'  <text x="{MARGIN}" y="96" class="ti">{esc(d["title"])}</text>')
    for j, line in enumerate(wrap(d["subtitle"], 142)):
        out.append(f'  <text x="{MARGIN}" y="{124 + j * 22}" class="sub">{esc(line)}</text>')

    for idx, p in enumerate(panels):
        r, c = divmod(idx, N_COLS)
        px = MARGIN + c * (PC_W + COL_GAP)
        py = GRID_TOP + r * ROW_PITCH
        render_panel(out, p, px, py)

    # legend
    leg_y = GRID_TOP + N_ROWS * PC_H + ROW_GAP + 30
    items = d["legend"]
    seg = [220, 560]
    start = MARGIN + 120
    x = start
    for j, it in enumerate(items):
        out.append(f'  <rect x="{x}" y="{leg_y - 13}" width="22" height="14" rx="3" fill="{it["color"]}"/>')
        out.append(f'  <text x="{x + 30}" y="{leg_y}" class="leg">{esc(it["label"])}</text>')
        x += seg[j] if j < len(seg) else 260

    # footer
    for j, line in enumerate(wrap(d["footer"], 168)):
        out.append(f'  <text x="{MARGIN}" y="{H - 32 + j * 15}" class="ftr">{esc(line)}</text>')

    out.append('</svg>\n')
    return "\n".join(out)


def out_path(d: dict) -> str:
    return os.path.join(ROOT, d.get("meta", {}).get("out_svg", DEFAULT_OUT))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generate the fak benchmark sweep grid SVG.")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--check", action="store_true", help="don't write; exit 1 if the on-disk SVG drifted from the data")
    args = ap.parse_args(argv)

    d = load_data(args.data)
    svg = svg_sweep(d)
    path = out_path(d)

    if args.check:
        if read_text(path) != svg:
            print(f"DRIFT — {os.path.relpath(path, ROOT)} is stale (run: python tools/hero_sweep_gen.py)")
            return 1
        print(f"check: {os.path.relpath(path, ROOT)} is up to date with {os.path.relpath(args.data, ROOT)}.")
        return 0

    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(svg)
    print(f"wrote {os.path.relpath(path, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
