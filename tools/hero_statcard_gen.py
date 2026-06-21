#!/usr/bin/env python3
"""Generate the fak BENCHMARK BREADTH card — a capabilities sweep, not one number.

A frontier-lab "by the numbers" card for `fak`: a side-by-side three-pillar
capability matrix (Serving efficiency · Correctness · Security kernel) over an
honest single-stream fence, with a left boundary-map rail that makes the
industry-impact thesis — "fak owns the whole agent-infrastructure boundary" —
literal rather than a slogan.

It is the opposite of a single-hero-number card on purpose (operator direction:
"be higher class, not a single number — show breadth and industry impact with
the variety of benchmarks"). Every pillar's lead metric shares the same weight,
so the visual claim IS breadth. Honesty is carried structurally:

  - the 4.1× row is [SOTA-style] teal (NOT green [SOTA]): its baseline is fak's
    OWN kernel held constant in warm-KV mode, so it isolates reuse, not kernel
    speed (HERO-BENCHMARK doc) — never a measured SGLang/vLLM process.
  - the 139.3× row is [NAIVE] amber with a neutral-gray number, so the biggest
    figure on the card can never be misread as a SOTA win.
  - the single-stream fence is INVERTED: llama.cpp OWNS the bold number, fak
    sits in a loss-amber cell with the explicit sub-1× ratio.

Design = a judge-panel synthesis over Anthropic model-card / NVIDIA MLPerf /
DeepMind matrix / system-card references.

  Source of truth : tools/hero_statcard.data.json
  Emits           : visuals/55-hero-statcard.svg   (meta.statcard_svg)

Usage:
  python tools/hero_statcard_gen.py            # (re)write the card
  python tools/hero_statcard_gen.py --check    # don't write; exit 1 if it drifted

Pure stdlib, so it runs in the pytest gate and the CI drift step.
"""
from __future__ import annotations

import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA = os.path.join(ROOT, "tools", "hero_statcard.data.json")
DEFAULT_OUT = os.path.join("visuals", "55-hero-statcard.svg")

INK, INK2, BODY, MUTED, MICRO = "#14202b", "#1b2733", "#415266", "#5b6a7a", "#8090a0"
GRAY, SLAB, CARD_STROKE = "#6a7685", "#2e3a47", "#d9e1e8"
GREEN, GREEN_FILL, GREEN_STROKE = "#2e8b57", "#edf7f0", "#85bc98"
AMBER_FILL, AMBER_STROKE, AMBER_TEXT = "#fff5e8", "#dfbd78", "#b06a2a"

# Honesty chip palette by class. (fill, stroke, text, label)
CHIP = {
    "SOTA-style":  ("#eef4f8", "#3a7d9c", "#155a6b", "SOTA-style"),
    "SOTA-parity": ("#eef4f8", "#3a7d9c", "#155a6b", "SOTA-parity"),
    "reuse":       ("#eef4f8", "#3a7d9c", "#155a6b", "reuse"),
    "NAIVE":       (AMBER_FILL, AMBER_STROKE, AMBER_TEXT, "NAIVE baseline"),
    "exact":       ("#f1edf8", "#7a5cb0", "#54458a", "exact"),
    "parity":      ("#f1edf8", "#7a5cb0", "#54458a", "parity"),
    "absolute":    ("#e9f3f6", "#1c7c92", "#155a6b", "absolute"),
    "LOSS":        (AMBER_FILL, AMBER_STROKE, AMBER_TEXT, "LOSS"),
}

# geometry
W = 1440
RAIL_X, RAIL_W = 40, 320
COLS = [392, 732, 1072]
COL_W = 320
TILE_H, TILE_GAP = 132, 14
BAND_Y, BAND_H = 196, 46
TILES_TOP = 286

STYLE = '''  <defs>
    <linearGradient id="sc-bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#f7f9fb"/>
      <stop offset="55%" stop-color="#eef4f1"/>
      <stop offset="100%" stop-color="#f8f3ee"/>
    </linearGradient>
    <style>
      .kick { font: 800 13px "Segoe UI", Arial, sans-serif; fill: #2e8b57; letter-spacing: 2px; }
      .ti   { font: 900 40px "Segoe UI", Arial, sans-serif; fill: #14202b; }
      .sub  { font: 400 19px "Segoe UI", Arial, sans-serif; fill: #415266; }
      .pchip { font: 700 12px "Segoe UI", Arial, sans-serif; fill: #5b6a7a; }
      .rlab { font: 800 12px "Segoe UI", Arial, sans-serif; fill: #5b6a7a; letter-spacing: 1.5px; }
      .rbody { font: 400 14px "Segoe UI", Arial, sans-serif; fill: #415266; }
      .rchipT { font: 700 13px "Segoe UI", Arial, sans-serif; fill: #1b2733; }
      .rchipN { font: 400 11px "Segoe UI", Arial, sans-serif; fill: #6a7a89; }
      .brkt { font: 800 11px "Segoe UI", Arial, sans-serif; fill: #14202b; }
      .brktG { font: 700 10px "Segoe UI", Arial, sans-serif; fill: #6a7685; }
      .imlab { font: 800 11px "Segoe UI", Arial, sans-serif; fill: #2e8b57; letter-spacing: 1px; }
      .imbody { font: 400 13px "Segoe UI", Arial, sans-serif; fill: #1b2733; }
      .pull { font: 800 17px "Segoe UI", Arial, sans-serif; fill: #ffffff; }
      .bandT { font: 800 15px "Segoe UI", Arial, sans-serif; fill: #ffffff; }
      .bandR { font: 700 11px "Segoe UI", Arial, sans-serif; fill: #ffffff; fill-opacity: 0.85; }
      .shead { font: 400 12px "Segoe UI", Arial, sans-serif; fill: #5b6a7a; }
      .tlab { font: 700 14px "Segoe UI", Arial, sans-serif; fill: #1b2733; }
      .tbase { font: 400 10.5px "Segoe UI", Arial, sans-serif; fill: #6a7a89; }
      .big  { font: 900 33px "Segoe UI", Arial, sans-serif; }
      .bigV { font: 900 25px "Segoe UI", Arial, sans-serif; }
      .tsub { font: 700 13px "Segoe UI", Arial, sans-serif; fill: #415266; }
      .note { font: 400 11px "Segoe UI", Arial, sans-serif; fill: #5b6a7a; }
      .barcap { font: 400 9.5px "Segoe UI", Arial, sans-serif; fill: #8090a0; }
      .chipt { font: 700 11px "Segoe UI", Arial, sans-serif; }
      .commit { font: 400 12px "Segoe UI", Arial, sans-serif; fill: #8090a0; }
      .fhead { font: 800 14px "Segoe UI", Arial, sans-serif; fill: #b06a2a; }
      .flab { font: 700 13px "Segoe UI", Arial, sans-serif; fill: #1b2733; }
      .fown { font: 800 14px "Segoe UI", Arial, sans-serif; fill: #1b2733; }
      .ffak { font: 800 15px "Segoe UI", Arial, sans-serif; fill: #b06a2a; }
      .fnote { font: 400 12px "Segoe UI", Arial, sans-serif; fill: #5b6a7a; }
      .ft1 { font: 800 16px "Segoe UI", Arial, sans-serif; fill: #14202b; }
      .ft2 { font: 700 14px "Segoe UI", Arial, sans-serif; fill: #3a7d9c; }
      .ftr { font: 400 12px "Segoe UI", Arial, sans-serif; fill: #5b6a7a; }
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


def wrap(text: str, maxchars: int, maxlines: int = 2) -> list:
    """Greedy word-wrap to at most maxlines; the last line gets an ellipsis if cut."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        cand = (cur + " " + w).strip()
        if len(cand) <= maxchars or not cur:
            cur = cand
        else:
            lines.append(cur)
            cur = w
            if len(lines) == maxlines:
                cur = ""
                break
    if cur and len(lines) < maxlines:
        lines.append(cur)
    consumed = sum(len(ln.split()) for ln in lines)        # count via real word splits
    if consumed < len(words) and lines:
        lines[-1] = lines[-1].rstrip(".") + "…"
    return lines[:maxlines]


# --------------------------------------------------------------------------- #
def _masthead(d: dict, out: list) -> None:
    m = d["masthead"]
    out.append(f'  <text x="56" y="70" class="kick">{esc(m["kicker"])}</text>')
    out.append(f'  <text x="56" y="116" class="ti">{esc(m["title"])}</text>')
    out.append(f'  <text x="56" y="150" class="sub">{esc(m["subtitle"])}</text>')
    out.append('  <line x1="56" y1="172" x2="1384" y2="172" stroke="#d9e1e8" stroke-width="1"/>')


def _rail(d: dict, out: list) -> None:
    r = d["rail"]
    out.append(f'  <rect x="{RAIL_X}" y="196" width="{RAIL_W}" height="660" rx="10" fill="#f4f8f6" stroke="#d9e1e8"/>')
    out.append(f'  <text x="60" y="230" class="rlab">{esc(r["thesis_label"])}</text>')
    for i, ln in enumerate(wrap(r["thesis"], 38, 4)):
        out.append(f'  <text x="60" y="{258 + i * 20}" class="rbody">{esc(ln)}</text>')

    # boundary map: 3 role chips + fak bracket (all 3) vs SOTA bracket (chip 1)
    chips = r["boundary_map"]
    cy0, ch, cgap = 364, 52, 12
    for i, c in enumerate(chips):
        y = cy0 + i * (ch + cgap)
        out.append(f'  <rect x="96" y="{y}" width="236" height="{ch}" rx="8" fill="#ffffff" stroke="#e3e9ee"/>')
        out.append(f'  <rect x="96" y="{y}" width="5" height="{ch}" rx="2.5" fill="{c["bar"]}"/>')
        out.append(f'  <text x="112" y="{y + 22}" class="rchipT">{esc(c["name"])}</text>')
        out.append(f'  <text x="112" y="{y + 40}" class="rchipN">{esc(c["note"])}</text>')
    top = cy0
    bot = cy0 + 2 * (ch + cgap) + ch
    # fak bracket spans all three
    out.append(f'  <path d="M84 {top} h-8 v{bot - top} h8" fill="none" stroke="#14202b" stroke-width="2"/>')
    out.append(f'  <text x="74" y="{(top + bot) // 2}" class="brkt" text-anchor="middle" transform="rotate(-90 74 {(top + bot) // 2})">fak spans all 3</text>')
    # SOTA bracket: chip 1 only
    out.append(f'  <path d="M340 {top} h8 v{ch} h-8" fill="none" stroke="#6a7685" stroke-width="1.5"/>')
    out.append(f'  <text x="352" y="{top + ch + 14}" class="brktG">SOTA serving</text>')
    out.append(f'  <text x="352" y="{top + ch + 26}" class="brktG">= serving only</text>')

    # industry-impact callout
    iy = bot + 24
    out.append(f'  <rect x="60" y="{iy}" width="288" height="120" rx="8" fill="#edf7f0" stroke="#85bc98"/>')
    out.append(f'  <text x="76" y="{iy + 24}" class="imlab">{esc(r["impact_label"])}</text>')
    for i, ln in enumerate(wrap(r["impact"], 40, 5)):
        out.append(f'  <text x="76" y="{iy + 46 + i * 17}" class="imbody">{esc(ln)}</text>')

    # pull-quote slab
    py = iy + 136
    out.append(f'  <rect x="60" y="{py}" width="288" height="64" rx="8" fill="#2e3a47"/>')
    for i, ln in enumerate(r["pullquote"][:2]):
        out.append(f'  <text x="78" y="{py + 28 + i * 24}" class="pull">{esc(ln)}</text>')
    # rail provenance footer
    out.append(f'  <text x="60" y="{py + 92}" class="rchipN">Every number → fak/BENCHMARK-AUTHORITY.md</text>')
    out.append(f'  <text x="60" y="{py + 108}" class="rchipN">(commit + JSON artifact, per row).</text>')


def _tile(col_x: int, y: int, row: dict, accent: str, out: list) -> None:
    x = col_x
    out.append(f'  <rect x="{x}" y="{y}" width="{COL_W}" height="{TILE_H}" rx="8" fill="#ffffff" stroke="#d9e1e8"/>')
    # label (1-2 lines)
    labs = wrap(row["label"], 40, 2)
    for i, ln in enumerate(labs):
        out.append(f'  <text x="{x + 16}" y="{y + 22 + i * 16}" class="tlab">{esc(ln)}</text>')
    by = y + 22 + len(labs) * 16 + 30        # big-number baseline
    num_fill = GRAY if row.get("naive") else accent
    if row.get("verdict"):
        out.append(f'  <text x="{x + 16}" y="{by - 2}" class="bigV" style="fill:{num_fill};">✓ {esc(row["metric"])}</text>')
    else:
        out.append(f'  <text x="{x + 16}" y="{by}" class="big" style="fill:{num_fill};">{esc(row["metric"])}</text>')
        # sub to the right of the number
        out.append(f'  <text x="{x + COL_W - 16}" y="{by}" class="tsub" text-anchor="end">{esc(row.get("sub", ""))}</text>')
    # optional ceiling bar
    if row.get("bar_frac") is not None:
        bw = 200
        out.append(f'  <rect x="{x + 16}" y="{by + 10}" width="{bw}" height="7" rx="3" fill="{accent}" fill-opacity="0.15"/>')
        out.append(f'  <rect x="{x + 16}" y="{by + 10}" width="{round(bw * float(row["bar_frac"]))}" height="7" rx="3" fill="{accent}"/>')
        out.append(f'  <text x="{x + 16}" y="{by + 31}" class="barcap">{esc(row.get("bar_cap", ""))}</text>')
    elif row.get("note"):
        out.append(f'  <text x="{x + 16}" y="{by + 16}" class="note">{esc(row["note"])}</text>')
    else:
        # short baseline line
        bl = wrap(row.get("baseline", ""), 50, 1)
        if bl:
            out.append(f'  <text x="{x + 16}" y="{by + 16}" class="tbase">{esc(bl[0])}</text>')
    # honesty chip (bottom-left) + commit stub (bottom-right)
    fill, stroke, tcol, label = CHIP.get(row.get("chip", ""), ("#eef1f4", "#d9e1e8", MUTED, row.get("chip", "")))
    cw = 11 + len(label) * 6.6
    out.append(f'  <rect x="{x + 16}" y="{y + TILE_H - 28}" width="{round(cw)}" height="18" rx="9" fill="{fill}" stroke="{stroke}"/>')
    out.append(f'  <text x="{x + 16 + round(cw) / 2}" y="{y + TILE_H - 15}" class="chipt" text-anchor="middle" style="fill:{tcol};">{esc(label)}</text>')
    if row.get("naive"):
        out.append(f'  <text x="{x + COL_W - 16}" y="{y + TILE_H - 15}" class="commit" text-anchor="end" style="fill:{AMBER_TEXT};">not a SOTA win</text>')
    else:
        out.append(f'  <text x="{x + COL_W - 16}" y="{y + TILE_H - 15}" class="commit" text-anchor="end">{esc(row.get("commit", ""))}</text>')


def _pillars(d: dict, out: list) -> None:
    for ci, p in enumerate(d["pillars"]):
        x = COLS[ci]
        out.append(f'  <rect x="{x}" y="{BAND_Y}" width="{COL_W}" height="{BAND_H}" rx="8" fill="{p["band"]}"/>')
        out.append(f'  <text x="{x + 16}" y="{BAND_Y + 29}" class="bandT">{esc(p["name"])}</text>')
        out.append(f'  <text x="{x + COL_W - 14}" y="{BAND_Y + 29}" class="bandR" text-anchor="end">{esc(p["tag"])}</text>')
        out.append(f'  <text x="{x + 2}" y="{BAND_Y + BAND_H + 22}" class="shead">{esc(p["subhead"])}</text>')
        for ti, row in enumerate(p["rows"]):
            _tile(x, TILES_TOP + ti * (TILE_H + TILE_GAP), row, p["accent"], out)


def _fence(d: dict, out: list, y: int) -> None:
    f = d["fence"]
    out.append(f'  <rect x="40" y="{y}" width="1360" height="116" rx="10" fill="{AMBER_FILL}" stroke="{AMBER_STROKE}"/>')
    out.append(f'  <text x="60" y="{y + 28}" class="fhead">{esc(f["name"])} — {esc(f["tag"])}</text>')
    tiles = f["rows"]
    tw, gap = 656, 28
    for i, r in enumerate(tiles):
        tx = 60 + i * (tw + gap)
        ty = y + 42
        out.append(f'  <rect x="{tx}" y="{ty}" width="{tw}" height="56" rx="8" fill="#ffffff" stroke="{AMBER_STROKE}"/>')
        out.append(f'  <text x="{tx + 16}" y="{ty + 24}" class="flab">{esc(r["label"])}</text>')
        out.append(f'  <text x="{tx + 16}" y="{ty + 44}" class="fown">{esc(r["owned"])}</text>')
        out.append(f'  <text x="{tx + tw - 88}" y="{ty + 35}" class="ffak" text-anchor="end">{esc(r["fak"])}</text>')
        # [LOSS] tag + em-dash gutter + commit
        out.append(f'  <rect x="{tx + tw - 78}" y="{ty + 20}" width="44" height="18" rx="9" fill="{AMBER_FILL}" stroke="{AMBER_STROKE}"/>')
        out.append(f'  <text x="{tx + tw - 56}" y="{ty + 33}" class="chipt" text-anchor="middle" style="fill:{AMBER_TEXT};">LOSS</text>')
        out.append(f'  <text x="{tx + tw - 18}" y="{ty + 35}" font-size="20" text-anchor="middle" style="fill:{AMBER_TEXT};font-weight:800;">—</text>')
        out.append(f'  <text x="{tx + 16}" y="{ty + 44}" class="commit" text-anchor="start" opacity="0"> </text>')
        out.append(f'  <text x="{tx + tw - 16}" y="{ty + 52}" class="commit" text-anchor="end">{esc(r["commit"])}</text>')


def svg_statcard(d: dict) -> str:
    H = 1080
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="sc-title sc-desc">',
        '  <title id="sc-title">fak benchmark breadth — a three-pillar capability sweep (serving, correctness, security kernel) with an honest single-stream fence</title>',
        f'  <desc id="sc-desc">{esc(d["masthead"]["title"])}. {esc(d["masthead"]["subtitle"])} Every number traces to fak/BENCHMARK-AUTHORITY.md.</desc>',
        STYLE,
        f'  <rect width="{W}" height="{H}" fill="url(#sc-bg)"/>',
        f'  <rect x="24" y="20" width="{W - 48}" height="{H - 40}" rx="12" fill="#ffffff" opacity="0.62" stroke="#d9e1e8"/>',
    ]
    _masthead(d, out)
    _rail(d, out)
    _pillars(d, out)

    fence_y = 884
    _fence(d, out, fence_y)

    f = d["footer"]
    out.append(f'  <text x="40" y="1034" class="ft1">{esc(f["line1"])}</text>')
    out.append(f'  <text x="40" y="1058" class="ft2">{esc(f["line2"])}</text>')
    out.append(f'  <text x="1400" y="1058" class="ftr" text-anchor="end">{esc(f["right"])}</text>')
    out.append('</svg>\n')
    return "\n".join(out)


# --------------------------------------------------------------------------- #
def out_path(d: dict) -> str:
    return os.path.join(ROOT, d.get("meta", {}).get("statcard_svg", DEFAULT_OUT))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generate the fak benchmark-breadth card SVG.")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--check", action="store_true", help="don't write; exit 1 if the on-disk card drifted")
    args = ap.parse_args(argv)

    d = load_data(args.data)
    svg = svg_statcard(d)
    path = out_path(d)

    if args.check:
        if read_text(path) != svg:
            print(f"DRIFT - {os.path.relpath(path, ROOT)} is stale vs the data file (run: python tools/hero_statcard_gen.py)")
            return 1
        print(f"check: {os.path.relpath(path, ROOT)} is up to date with {os.path.relpath(args.data, ROOT)}.")
        return 0

    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(svg)
    print(f"wrote {os.path.relpath(path, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
