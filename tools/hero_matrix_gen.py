#!/usr/bin/env python3
"""Generate the fak CAPABILITY MATRIX — visuals/59-hero-capability-matrix.svg.

fak's analog of the MiniMax-M3 model-comparison table: capabilities grouped by
category down the left, the serving stacks across the top, the subject (fak)
column washed and highlighted, and an "Evaluation methodology" footnote block
below — the visual grammar a frontier lab ships a model card in.

The thesis is structural: fak carries a value down the WHOLE matrix, while the
SOTA serving stacks light up only the SERVING REUSE and SINGLE-STREAM bands and
carry an honest em-dash everywhere else. The em-dash means "not a capability that
stack ships," NOT a measured zero — so competitor cells are coverage marks, never
fabricated numbers; only fak's column carries measured figures (each tracing to a
commit). The SINGLE-STREAM fence inverts the highlight: the SOTA stack owns the
bold cell, fak shows its honest lower number.

  Source of truth : tools/hero_matrix.data.json
  Emits           : visuals/59-hero-capability-matrix.svg   (meta.out_svg)

Usage:
  python tools/hero_matrix_gen.py            # (re)write the SVG
  python tools/hero_matrix_gen.py --check     # don't write; exit 1 if it drifted

Pure stdlib — runs in the `python -m pytest tools` gate and the CI drift step.
"""
from __future__ import annotations

import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA = os.path.join(ROOT, "tools", "hero_matrix.data.json")
DEFAULT_OUT = os.path.join("visuals", "59-hero-capability-matrix.svg")

# House palette
GREEN, GREEN_DK, GREEN_WASH, GREEN_STROKE = "#2e8b57", "#1f6b46", "#edf7f0", "#85bc98"
INK, BODY, MUTED, MICRO = "#14202b", "#415266", "#5b6a7a", "#8090a0"
ABSENT, NA, SLAB, ROWLINE = "#b8c2cc", "#cdd5dd", "#2e3a47", "#e7ecf0"
WEAK = "#6a7685"

# geometry
W = 1320
LABEL_X = 72
COLS_X0, COL_W, N_STACK = 560, 130, 5
COMMIT_X = 1256
MAST_BOTTOM = 196
HEADER_H = 62
CAT_H = 28
ROW_H = 34
# inline coverage pip-strip (one pip per stack, in the gutter left of the fak column)
PIP_X0, PIP_GAP, PIP_R = 488, 14, 4
PIP_CX = PIP_X0 + 6 + 2 * PIP_GAP   # strip centre, for the COVERAGE header

STYLE = '''  <defs>
    <linearGradient id="mx-bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#fbfcfd"/>
      <stop offset="60%" stop-color="#f4f7f9"/>
      <stop offset="100%" stop-color="#f7f4f0"/>
    </linearGradient>
    <style>
      .k    { font: 800 14px "Segoe UI", Arial, sans-serif; fill: #2e8b57; letter-spacing: 1.6px; }
      .ti   { font: 900 31px "Segoe UI", Arial, sans-serif; fill: #14202b; }
      .sub  { font: 400 15px "Segoe UI", Arial, sans-serif; fill: #415266; }
      .hcol { font: 800 14.5px "Segoe UI", Arial, sans-serif; fill: #1b2733; }
      .hsub { font: 800 14.5px "Segoe UI", Arial, sans-serif; fill: #2e8b57; }
      .htype { font: 700 11px "Segoe UI", Arial, sans-serif; fill: #6a7685; }
      .htypeF { font: 800 11px "Segoe UI", Arial, sans-serif; fill: #ffffff; letter-spacing: 0.3px; }
      .covh { font: 800 10px "Segoe UI", Arial, sans-serif; fill: #8a96a3; letter-spacing: 0.5px; }
      .bdg  { font: 800 13px "Segoe UI", Arial, sans-serif; }
      .cat  { font: 800 13.5px "Segoe UI", Arial, sans-serif; fill: #ffffff; letter-spacing: 0.6px; }
      .catn { font: 400 12px "Segoe UI", Arial, sans-serif; fill: #ffffff; opacity: 0.92; }
      .cap  { font: 700 14px "Segoe UI", Arial, sans-serif; fill: #1b2733; }
      .csub { font: 400 11.5px "Segoe UI", Arial, sans-serif; fill: #6a7a89; }
      .cm   { font: 400 10.5px "Segoe UI", Arial, sans-serif; fill: #9aa6b2; }
      .num  { font: 800 14.5px "Segoe UI", Arial, sans-serif; fill: #1f6b46; }
      .numc { font: 700 13px "Segoe UI", Arial, sans-serif; fill: #51616f; }
      .yes  { font: 800 16px "Segoe UI", Arial, sans-serif; fill: #2e8b57; }
      .weak { font: 400 12px "Segoe UI", Arial, sans-serif; fill: #6a7685; }
      .base { font: 700 12.5px "Segoe UI", Arial, sans-serif; fill: #8a96a3; }
      .lead { font: 800 13.5px "Segoe UI", Arial, sans-serif; fill: #14202b; }
      .lose { font: 400 12.5px "Segoe UI", Arial, sans-serif; fill: #51616f; }
      .mh   { font: 800 16px "Segoe UI", Arial, sans-serif; fill: #14202b; letter-spacing: 0.3px; }
      .mn   { font: 400 12.5px "Segoe UI", Arial, sans-serif; fill: #51616f; }
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


def col_center(i: int) -> int:
    return COLS_X0 + COL_W // 2 + i * COL_W


def cell_svg(cx: int, y: int, kind: str, text: str, subject: bool) -> str:
    t = esc(text)
    if kind == "num":
        cls = "num" if subject else "numc"
        return f'  <text x="{cx}" y="{y}" class="{cls}" text-anchor="middle">{t}</text>'
    if kind == "yes":
        return f'  <text x="{cx}" y="{y}" class="yes" text-anchor="middle">{t}</text>'
    if kind in ("no",):
        return f'  <text x="{cx}" y="{y}" font-size="18" text-anchor="middle" style="fill:{ABSENT};">—</text>'
    if kind == "weak":
        return f'  <text x="{cx}" y="{y}" class="weak" text-anchor="middle">{t}</text>'
    if kind == "base":
        return f'  <text x="{cx}" y="{y}" class="base" text-anchor="middle">{t}</text>'
    if kind == "na":
        return f'  <text x="{cx}" y="{y}" text-anchor="middle" style="fill:{NA};font:400 11.5px Segoe UI,Arial;">{t}</text>'
    if kind == "lead":
        return f'  <text x="{cx}" y="{y}" class="lead" text-anchor="middle">{t}</text>'
    if kind == "lose":
        return f'  <text x="{cx}" y="{y}" class="lose" text-anchor="middle">{t}</text>'
    return f'  <text x="{cx}" y="{y}" class="weak" text-anchor="middle">{t}</text>'


def svg_matrix(d: dict) -> str:
    cols = d["columns"]
    cats = d["categories"]
    n_rows = sum(len(c["rows"]) for c in cats)
    n_cats = len(cats)

    table_top = MAST_BOTTOM
    table_h = HEADER_H + n_cats * CAT_H + n_rows * ROW_H
    table_bottom = table_top + table_h

    meth = d["methodology"]
    # pre-wrap methodology notes to compute height
    note_lines = [wrap(n, 168) for n in meth["notes"]]
    meth_top = table_bottom + 30
    meth_h = 30 + sum((len(ls) * 16 + 8) for ls in note_lines)
    H = meth_top + meth_h + 36

    wash_x = COLS_X0 + 2
    wash_w = COL_W - 4

    out = []
    out.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" role="img" aria-labelledby="mx-title mx-desc">'
    )
    out.append(f'  <title id="mx-title">{esc(d["meta"]["title"])}</title>')
    out.append('  <desc id="mx-desc">A capability matrix: capabilities grouped by SERVING REUSE, CORRECTNESS, SECURITY KERNEL and a SINGLE-STREAM fence down the left; fak, vLLM, SGLang, llama.cpp and an API prompt cache across the top, fak highlighted. fak carries a value in every row; the serving stacks carry an em-dash outside the serving and single-stream bands. Only fak shows measured numbers; the fence inverts the highlight to the SOTA leader.</desc>')
    out.append(STYLE)
    out.append(f'  <rect width="{W}" height="{H}" fill="url(#mx-bg)"/>')

    # masthead
    out.append(f'  <text x="{LABEL_X}" y="58" class="k">{esc(d["kicker"])}</text>')
    out.append(f'  <text x="{LABEL_X}" y="96" class="ti">{esc(d["title"])}</text>')
    for j, line in enumerate(wrap(d["subtitle"], 138)):
        out.append(f'  <text x="{LABEL_X}" y="{124 + j * 22}" class="sub">{esc(line)}</text>')

    # fak-column highlight wash (behind everything, full table height)
    out.append(f'  <rect x="{wash_x}" y="{table_top}" width="{wash_w}" height="{table_h}" rx="7" fill="{GREEN_WASH}" opacity="0.55"/>')
    out.append(f'  <rect x="{wash_x}" y="{table_top}" width="{wash_w}" height="{table_h}" rx="7" fill="none" stroke="{GREEN_STROKE}" stroke-width="1.5" opacity="0.8"/>')

    # header row: column badges + names + a type sub-label (fak = "addressable",
    # the differentiator; the serving stacks = "front-prefix" / "front-only") +
    # a COVERAGE header above the inline pip-strip gutter.
    name_y = table_top + 36
    type_y = table_top + 54
    out.append(f'  <text x="{LABEL_X}" y="{name_y}" class="hcol">CAPABILITY</text>')
    out.append(f'  <text x="{PIP_CX}" y="{type_y}" class="covh" text-anchor="middle">{esc(d.get("coverage_label", "COVERAGE"))}</text>')
    for i, c in enumerate(cols):
        cx = col_center(i)
        subj = bool(c.get("subject"))
        # badge chip
        bx = cx - 58
        bfill = GREEN if subj else "#ffffff"
        bstroke = GREEN if subj else "#c2ccd6"
        btext = "#ffffff" if subj else "#5b6a7a"
        out.append(f'  <rect x="{bx}" y="{table_top + 6}" width="20" height="20" rx="5" fill="{bfill}" stroke="{bstroke}" stroke-width="1.2"/>')
        out.append(f'  <text x="{bx + 10}" y="{table_top + 20}" class="bdg" text-anchor="middle" style="fill:{btext};">{esc(c["badge"])}</text>')
        out.append(f'  <text x="{cx + 8}" y="{name_y}" class="{"hsub" if subj else "hcol"}" text-anchor="middle">{esc(c["name"])}</text>')
        # type sub-label: fak in a green pill, the serving stacks plain grey
        typ = c.get("type", "")
        if typ and subj:
            pill_w = 12 + len(typ) * 6
            out.append(f'  <rect x="{cx + 8 - pill_w // 2}" y="{type_y - 12}" width="{pill_w}" height="16" rx="8" fill="{GREEN}"/>')
            out.append(f'  <text x="{cx + 8}" y="{type_y}" class="htypeF" text-anchor="middle">{esc(typ)}</text>')
        elif typ:
            out.append(f'  <text x="{cx + 8}" y="{type_y}" class="htype" text-anchor="middle">{esc(typ)}</text>')
    out.append(f'  <line x1="{LABEL_X}" y1="{table_top + HEADER_H}" x2="{COMMIT_X}" y2="{table_top + HEADER_H}" stroke="{SLAB}" stroke-width="2"/>')

    # body
    y = table_top + HEADER_H
    for cat in cats:
        # category band
        out.append(f'  <rect x="{LABEL_X}" y="{y}" width="{COMMIT_X - LABEL_X}" height="{CAT_H}" rx="5" fill="{cat["accent"]}"/>')
        out.append(f'  <text x="{LABEL_X + 14}" y="{y + 19}" class="cat">{esc(cat["name"])}</text>')
        out.append(f'  <text x="{COMMIT_X - 12}" y="{y + 19}" class="catn" text-anchor="end">{esc(cat.get("note", ""))}</text>')
        y += CAT_H
        for r in cat["rows"]:
            l1, l2 = y + 16, y + 29
            out.append(f'  <text x="{LABEL_X}" y="{l1}" class="cap">{esc(r["cap"])}</text>')
            out.append(f'  <text x="{LABEL_X}" y="{l2}" class="csub">{esc(r.get("sub", ""))}</text>')
            out.append(f'  <text x="{COMMIT_X}" y="{l1}" class="cm" text-anchor="end">{esc(r.get("commit", ""))}</text>')
            cy = y + 22
            # inline coverage pip-strip (one pip per stack, derived from the cells):
            # a filled pip = that stack ships the capability; fak's pip is green, the
            # serving stacks grey; a hollow ring = not shipped. The security/correctness
            # rows visibly carry one green pip — the moat, at a glance.
            for i, c in enumerate(cols):
                kind = r["cells"][c["key"]][0]
                pcx = PIP_X0 + 6 + i * PIP_GAP
                if kind not in ("no", "na"):
                    pf = GREEN if c.get("subject") else "#9aa6b2"
                    out.append(f'  <circle cx="{pcx}" cy="{cy - 4}" r="{PIP_R}" fill="{pf}"/>')
                else:
                    out.append(f'  <circle cx="{pcx}" cy="{cy - 4}" r="{PIP_R}" fill="#ffffff" stroke="#cdd5dd" stroke-width="1.2"/>')
            for i, c in enumerate(cols):
                kind, text = r["cells"][c["key"]]
                out.append(cell_svg(col_center(i), cy, kind, text, bool(c.get("subject"))))
            out.append(f'  <line x1="{LABEL_X}" y1="{y + ROW_H}" x2="{COMMIT_X}" y2="{y + ROW_H}" stroke="{ROWLINE}" stroke-width="1"/>')
            y += ROW_H

    # methodology block
    my = meth_top
    out.append(f'  <text x="{LABEL_X}" y="{my}" class="mh">{esc(meth["title"])}</text>')
    my += 22
    for ls in note_lines:
        out.append(f'  <circle cx="{LABEL_X + 4}" cy="{my - 4}" r="2.5" fill="{GREEN}"/>')
        for j, line in enumerate(ls):
            out.append(f'  <text x="{LABEL_X + 16}" y="{my + j * 16}" class="mn">{esc(line)}</text>')
        my += len(ls) * 16 + 8

    out.append('</svg>\n')
    return "\n".join(out)


def out_path(d: dict) -> str:
    return os.path.join(ROOT, d.get("meta", {}).get("out_svg", DEFAULT_OUT))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generate the fak capability matrix SVG.")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--check", action="store_true", help="don't write; exit 1 if the on-disk SVG drifted from the data")
    args = ap.parse_args(argv)

    d = load_data(args.data)
    svg = svg_matrix(d)
    path = out_path(d)

    if args.check:
        if read_text(path) != svg:
            print(f"DRIFT — {os.path.relpath(path, ROOT)} is stale (run: python tools/hero_matrix_gen.py)")
            return 1
        print(f"check: {os.path.relpath(path, ROOT)} is up to date with {os.path.relpath(args.data, ROOT)}.")
        return 0

    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(svg)
    print(f"wrote {os.path.relpath(path, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
