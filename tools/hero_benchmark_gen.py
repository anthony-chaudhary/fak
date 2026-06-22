#!/usr/bin/env python3
"""Rebuild the fak HERO benchmark comparison from one source-of-truth data file.

The hero comparison (frontier-lab style: headline + top-3 SOTA chart + top-10
leaderboard with fak bolded where it wins) used to be three hand-authored files
that had to be edited by hand every time a benchmark changed. This makes it
DATA-DRIVEN: edit `tools/hero_benchmark.data.json`, run this, and the doc + both
SVGs (+ an optional standalone HTML widget) regenerate.

  Source of truth : tools/hero_benchmark.data.json
  Emits           : fak/HERO-BENCHMARK-*.md            (meta.doc_path)
                    visuals/53-hero-benchmark-comparison.svg   (meta.hero_svg)
                    visuals/54-hero-benchmark-leaderboard.svg  (meta.leaderboard_svg)
                    visuals/hero-benchmark.html         (meta.html_path, with --html)

Usage:
  python tools/hero_benchmark_gen.py                 # rebuild doc + both SVGs
  python tools/hero_benchmark_gen.py --html          # also rebuild the HTML widget
  python tools/hero_benchmark_gen.py --check         # don't write; fail if on-disk drifted
  python tools/hero_benchmark_gen.py --verify-sources # re-assert numbers vs committed artifacts
  python tools/hero_benchmark_gen.py --data path.json # use a different data file

No third-party deps (SVG/MD/HTML are emitted as text). Pure stdlib so it runs in
the `python -m pytest tools` gate and on any bench node.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA = os.path.join(ROOT, "tools", "hero_benchmark.data.json")

# Role -> bar fill for the cost-collapse hero chart (white-bg house palette).
ROLE_FILL = {"naive": "#6a7685", "sota": "#1c7c92", "fak": "#2e8b57"}
ROLE_SHORT = {"naive": "Naive re-prefill", "sota": "Tuned SOTA cache", "fak": "fak fused"}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def esc(s: object) -> str:
    """XML-escape text for SVG (× → ≈ etc. are valid UTF-8, only &<> need it)."""
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def load_data(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def validate_data(d: dict) -> None:
    """Fail loud, with a helpful message, when the data file can't be rendered.

    Catches the two ways "add a new result" silently breaks: a top10 regime with
    no palette colour (would render a default-gray pill) and a cost-collapse arm
    with an unknown role (would render a default-gray bar)."""
    problems = []
    pal = d.get("palette", {})
    for r in d.get("top10", []):
        if r.get("regime") not in pal:
            problems.append(f"top10 row {r.get('rank')}: regime {r.get('regime')!r} has no palette colour")
    for a in d.get("cost_collapse", {}).get("arms", []):
        if a.get("role") not in ROLE_FILL:
            problems.append(f"cost_collapse arm {a.get('name')!r}: unknown role {a.get('role')!r} "
                            f"(known: {sorted(ROLE_FILL)})")
    if problems:
        raise ValueError("hero_benchmark.data.json validation failed:\n  " + "\n  ".join(problems))


def resolve_field(obj: object, path: str):
    """Resolve a dotted path with array indices, e.g. 'cells[0].net_value_add'."""
    cur = obj
    for tok in path.split("."):
        m = re.match(r"^([^\[\]]+)((\[\d+\])*)$", tok)
        if not m:
            raise KeyError(f"bad path token: {tok!r} in {path!r}")
        key = m.group(1)
        cur = cur[key]
        for idx in re.findall(r"\[(\d+)\]", m.group(2) or ""):
            cur = cur[int(idx)]
    return cur


def read_text(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


# --------------------------------------------------------------------------- #
# SVG 1 — the hero: headline card + cost-collapse bars + top-3 SOTA strip
# --------------------------------------------------------------------------- #
def svg_hero(d: dict) -> str:
    h = d["headline"]
    cc = d["cost_collapse"]
    t3 = d["top3_sota"]

    # cost-collapse bar chart geometry
    cx0, cx1 = 588, 1500            # axis x span
    axis_w = cx1 - cx0
    cmax = float(cc["chart_max_min"])
    scale = axis_w / cmax
    baseline_y = 540
    bar_top, pitch, barh = 298, 70, 44

    bars = []
    for i, arm in enumerate(cc["arms"]):
        y = bar_top + i * pitch
        fill = ROLE_FILL.get(arm["role"], "#6a7685")
        name = esc(arm["name"])
        disp = esc(arm["display"])
        if arm.get("offchart"):
            bars.append(
                f'    <rect x="{cx0}" y="{y}" width="884" height="{barh}" rx="6" fill="{fill}"/>\n'
                f'    <rect x="1472" y="{y}" width="28" height="{barh}" fill="{fill}"/>\n'
                f'    <path d="M1472 {y} l14 22 l-14 22" fill="none" stroke="#ffffff" stroke-width="2.5"/>\n'
                f'    <text x="606" y="{y + 28}" class="barLabel">{name}</text>\n'
                f'    <text x="1490" y="{y - 8}" class="barVal" text-anchor="end" style="fill:{fill};">{disp}</text>'
            )
        else:
            w = round(arm["minutes"] * scale)
            stroke = (f'\n    <rect x="{cx0}" y="{y}" width="{w}" height="{barh}" rx="6" '
                      f'fill="none" stroke="#1f6e43" stroke-width="2"/>' if arm["role"] == "fak" else "")
            if w >= 500:
                val = (f'    <text x="{cx0 + w - 10}" y="{y + 28}" class="barVal" '
                       f'text-anchor="end" style="fill:#ffffff;">{disp}</text>')
            else:
                val = (f'    <text x="{cx0 + w + 12}" y="{y + 28}" class="barVal">{disp}</text>')
            bars.append(
                f'    <rect x="{cx0}" y="{y}" width="{w}" height="{barh}" rx="6" fill="{fill}"/>{stroke}\n'
                f'    <text x="606" y="{y + 28}" class="barLabel">{name}</text>\n'
                f'{val}'
            )
    ticks = []
    for tk in cc["axis_ticks_min"]:
        x = round(cx0 + tk * scale)
        anchor = "start" if tk == cc["axis_ticks_min"][0] else ("end" if tk == cc["axis_ticks_min"][-1] else "middle")
        label = "0" if tk == 0 else f"{tk} min"
        ticks.append(f'    <text x="{x}" y="562" class="axis" text-anchor="{anchor}">{esc(label)}</text>')

    # top-3 SOTA cards
    cards = []
    cw, gap = 466, 29
    for i, it in enumerate(t3["items"]):
        x = 72 + i * (cw + gap)
        head_fill = "#1c7c92" if it["header_role"] == "sota" else "#6a7685"
        pos_fill = "#2e8b57" if it["position_role"] == "win" else "#b06a2a"
        gives = "".join(
            f'\n    <text x="{x + 20}" y="{800 + j * 24}" class="body">{esc(g)}</text>' for j, g in enumerate(it["gives"]))
        posn = "".join(
            f'\n    <text x="{x + 20}" y="{896 + j * 22}" class="body" style="fill:{pos_fill};">{esc(p)}</text>'
            for j, p in enumerate(it["position"]))
        cards.append(
            f'    <rect x="{x}" y="690" width="{cw}" height="240" rx="8" class="cardT"/>\n'
            f'    <rect x="{x}" y="690" width="{cw}" height="6" rx="3" fill="{head_fill}"/>\n'
            f'    <text x="{x + 20}" y="726" class="tagD" style="fill:{head_fill};">{esc(it["tag"])}</text>\n'
            f'    <text x="{x + 20}" y="772" class="label">What it gives</text>{gives}\n'
            f'    <text x="{x + 20}" y="868" class="label">fak’s position</text>{posn}'
        )

    sec = h["secondary"]
    # Split the secondary note into the headline takeaway ("4.1× less work") and a
    # quiet qualifier, so the hero card leads with the two numbers that matter
    # (the serving time and the multiplier) and demotes everything else to context.
    note = sec.get("note", "")
    if "—" in note:
        delta_big, delta_small = (p.strip() for p in note.split("—", 1))
    else:
        delta_big, delta_small = note.strip(), ""
    # Keep the context line short so it never runs past the card edge: strip the
    # vendor parenthetical off the baseline and fold it into the tiny qualifier.
    base = sec["baseline"]
    m_par = re.search(r"\((.*?)\)", base)
    vendor = m_par.group(1) if m_par else ""
    base_short = re.sub(r"\s*\(.*?\)", "", base).strip()
    ctx_line = f'vs {sec["value"]} · {base_short}'
    small_line = " · ".join(x for x in (vendor, delta_small) if x)
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1060" viewBox="0 0 1600 1060" role="img" aria-labelledby="title desc">
  <title id="title">fak hero benchmark — agent-fleet serving efficiency vs the SOTA serving floor</title>
  <desc id="desc">{esc(h["statement"].replace("**", "").replace("`", "").replace("*", ""))}</desc>
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#f7faf9"/>
      <stop offset="50%" stop-color="#eef5f6"/>
      <stop offset="100%" stop-color="#faf4ee"/>
    </linearGradient>
    <filter id="shadow" x="-10%" y="-10%" width="120%" height="130%">
      <feDropShadow dx="0" dy="8" stdDeviation="10" flood-color="#14202b" flood-opacity="0.13"/>
    </filter>
    <style>
      .title {{ font: 900 46px "Segoe UI", Arial, sans-serif; fill: #14202b; }}
      .subtitle {{ font: 400 21px "Segoe UI", Arial, sans-serif; fill: #405063; }}
      .kicker {{ font: 800 17px "Segoe UI", Arial, sans-serif; fill: #2e8b57; letter-spacing: 2px; }}
      .cardTitle {{ font: 800 22px "Segoe UI", Arial, sans-serif; fill: #1b2733; }}
      .hero {{ font: 900 122px "Segoe UI", Arial, sans-serif; fill: #2e8b57; }}
      .heroUnit {{ font: 700 24px "Segoe UI", Arial, sans-serif; fill: #5b6a7a; }}
      .heroDelta {{ font: 900 36px "Segoe UI", Arial, sans-serif; fill: #2e8b57; }}
      .heroCtx {{ font: 600 19px "Segoe UI", Arial, sans-serif; fill: #314153; }}
      .big {{ font: 900 40px "Segoe UI", Arial, sans-serif; fill: #17202a; }}
      .cardT {{ fill: #ffffff; stroke: #e3e9ee; stroke-width: 1; }}
      .tagD {{ font: 800 13px "Segoe UI", Arial, sans-serif; letter-spacing: 1px; }}
      .label {{ font: 700 17px "Segoe UI", Arial, sans-serif; fill: #314153; }}
      .barLabel {{ font: 800 20px "Segoe UI", Arial, sans-serif; fill: #ffffff; }}
      .barVal {{ font: 900 22px "Segoe UI", Arial, sans-serif; fill: #17202a; }}
      .body {{ font: 400 16px "Segoe UI", Arial, sans-serif; fill: #415266; }}
      .small {{ font: 400 14px "Segoe UI", Arial, sans-serif; fill: #5b6a7a; }}
      .axis {{ font: 700 14px "Segoe UI", Arial, sans-serif; fill: #8090a0; }}
      .tag {{ font: 800 13px "Segoe UI", Arial, sans-serif; fill: #ffffff; letter-spacing: 1px; }}
      .card {{ fill: #ffffff; stroke: #d9e1e8; stroke-width: 1.5; }}
      .highlight {{ fill: #edf7f0; stroke: #85bc98; stroke-width: 2; }}
      .warn {{ fill: #fff5e8; stroke: #dfbd78; }}
    </style>
  </defs>

  <rect width="1600" height="1060" fill="url(#bg)"/>
  <rect x="38" y="28" width="1524" height="1004" rx="10" fill="#ffffff" opacity="0.62"/>

  <text x="72" y="86" class="kicker">{esc(h["kicker"])}</text>
  <text x="72" y="138" class="title">{esc(h["title"])}</text>
  <text x="72" y="174" class="subtitle">{esc(h["subtitle"])}</text>

  <g filter="url(#shadow)">
    <rect x="72" y="214" width="430" height="360" rx="10" class="highlight"/>
    <text x="100" y="258" class="label">{esc(h["primary"]["baseline"])}</text>
    <text x="92" y="368" class="hero">{esc(h["primary"]["value"])}</text>
    <text x="100" y="404" class="heroUnit">{esc(h["primary"]["label"])}</text>
    <line x1="100" y1="430" x2="474" y2="430" stroke="#cfe3d6" stroke-width="2"/>
    <text x="100" y="476" class="heroDelta">{esc(delta_big)}</text>
    <text x="100" y="510" class="heroCtx">{esc(ctx_line)}</text>
    <text x="100" y="540" class="small">{esc(small_line)}</text>
  </g>

  <g>
    <rect x="540" y="214" width="988" height="360" rx="10" class="card"/>
    <text x="568" y="258" class="cardTitle">{esc(cc["title"])}</text>
    <line x1="{cx0}" y1="{baseline_y}" x2="{cx1}" y2="{baseline_y}" stroke="#c4cfd8" stroke-width="2"/>
{chr(10).join(ticks)}
{chr(10).join(bars)}
    <text x="568" y="528" class="small">{esc(cc["footnote"])}</text>
  </g>

  <text x="72" y="636" class="cardTitle">{esc(t3["heading"])}</text>
  <text x="72" y="664" class="body">{esc(t3["intro"])}</text>

  <g>
{chr(10).join(cards)}
  </g>

  <rect x="72" y="952" width="1456" height="62" rx="8" class="warn"/>
  <text x="92" y="982" class="label">Two regimes, one honest answer.</text>
  <text x="92" y="1006" class="small">fak is not a faster model server — on a single stream it currently trails llama.cpp (closing) and tops out at 7B on this 36 GB box. The win is the agent fleet: cross-agent prefix sharing. Every number traces to {esc(d["meta"]["authority"])} (commit {esc(h["provenance"]["commit"])}, {esc(os.path.basename(h["provenance"]["artifact"]))}).</text>
</svg>
'''


# --------------------------------------------------------------------------- #
# SVG 2 — the top-10 leaderboard (dynamic row count, fak bolded on wins)
# --------------------------------------------------------------------------- #
def svg_leaderboard(d: dict) -> str:
    lb = d["leaderboard"]
    rows = d["top10"]
    palette = d["palette"]
    n = len(rows)
    rows_top, pitch, rowh = 226, 66, 62
    height = rows_top + n * pitch + 78

    body = []
    for i, r in enumerate(rows):
        y = rows_top + i * pitch
        win = r["win"]
        if win:
            rowfill, rowstroke = "#edf7f0", "#cfe7d8"
        else:
            rowfill, rowstroke = "#fff5e8", "#e7d3a8"
        ty = y + 38           # rank/value baseline
        reg_fill = palette.get(r["regime"], "#6a7685")
        # benchmark column
        bname = esc(r["benchmark"])
        bsub = esc(r["sub"])
        baseline = esc(r["baseline"])
        base_attr = ' style="font-weight:800;fill:#1b2733;"' if not win else ""
        fakval = esc(r["fak"])
        if win:
            fak_cls, verdict, vsize = "fakWin", "✅", 24
        else:
            fak_cls, verdict, vsize = "fakLose", "—", 22
        body.append(
            f'  <rect x="64" y="{y}" width="1472" height="{rowh}" rx="4" fill="{rowfill}" stroke="{rowstroke}"/>\n'
            f'  <text x="80" y="{ty}" class="rank">{r["rank"]}</text>\n'
            f'  <text x="132" y="{y + 32}" class="bname">{bname}</text>\n'
            f'  <text x="132" y="{y + 52}" class="bsub">{bsub}</text>\n'
            f'  <rect x="600" y="{y + 18}" width="160" height="26" rx="13" fill="{reg_fill}"/>'
            f'<text x="680" y="{y + 36}" class="regTag" text-anchor="middle">{esc(r["regime"])}</text>\n'
            f'  <text x="800" y="{ty - 1}" class="base"{base_attr}>{baseline}</text>\n'
            f'  <text x="1168" y="{ty + 1}" class="{fak_cls}">{fakval}</text>\n'
            f'  <text x="1452" y="{ty + 2}" font-size="{vsize}" text-anchor="middle">{verdict}</text>'
        )

    legend_y = rows_top + n * pitch + 6
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="{height}" viewBox="0 0 1600 {height}" role="img" aria-labelledby="title desc">
  <title id="title">fak SOTA benchmark leaderboard — fak's number bolded where it leads the SOTA baseline</title>
  <desc id="desc">{esc(lb["subtitle"])}</desc>
  <defs>
    <linearGradient id="bg2" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#f7faf9"/>
      <stop offset="50%" stop-color="#eef5f6"/>
      <stop offset="100%" stop-color="#faf4ee"/>
    </linearGradient>
    <style>
      .title {{ font: 900 42px "Segoe UI", Arial, sans-serif; fill: #14202b; }}
      .subtitle {{ font: 400 19px "Segoe UI", Arial, sans-serif; fill: #405063; }}
      .kicker {{ font: 800 16px "Segoe UI", Arial, sans-serif; fill: #2e8b57; letter-spacing: 2px; }}
      .hcol {{ font: 800 15px "Segoe UI", Arial, sans-serif; fill: #ffffff; letter-spacing: 1px; }}
      .rank {{ font: 900 22px "Segoe UI", Arial, sans-serif; fill: #9aa7b3; }}
      .bname {{ font: 700 18px "Segoe UI", Arial, sans-serif; fill: #1b2733; }}
      .bsub {{ font: 400 13px "Segoe UI", Arial, sans-serif; fill: #6a7a89; }}
      .base {{ font: 400 15px "Segoe UI", Arial, sans-serif; fill: #51616f; }}
      .fakWin {{ font: 900 22px "Segoe UI", Arial, sans-serif; fill: #2e8b57; }}
      .fakLose {{ font: 400 17px "Segoe UI", Arial, sans-serif; fill: #51616f; }}
      .regTag {{ font: 800 11px "Segoe UI", Arial, sans-serif; fill: #ffffff; letter-spacing: 0.5px; }}
      .legend {{ font: 600 14px "Segoe UI", Arial, sans-serif; fill: #51616f; }}
      .small {{ font: 400 13px "Segoe UI", Arial, sans-serif; fill: #5b6a7a; }}
    </style>
  </defs>

  <rect width="1600" height="{height}" fill="url(#bg2)"/>
  <rect x="32" y="24" width="1536" height="{height - 48}" rx="10" fill="#ffffff" opacity="0.62"/>

  <text x="64" y="76" class="kicker">SOTA BENCHMARK LEADERBOARD</text>
  <text x="64" y="120" class="title">{esc(lb["heading"])}</text>
  <text x="64" y="150" class="subtitle">{esc(lb["subtitle"])}</text>

  <rect x="64" y="178" width="1472" height="44" rx="6" fill="#2e3a47"/>
  <text x="80" y="206" class="hcol">#</text>
  <text x="132" y="206" class="hcol">BENCHMARK</text>
  <text x="600" y="206" class="hcol">REGIME</text>
  <text x="800" y="206" class="hcol">SOTA BASELINE</text>
  <text x="1168" y="206" class="hcol">fak RESULT</text>
  <text x="1436" y="206" class="hcol">LEAD?</text>

{chr(10).join(body)}

  <rect x="64" y="{legend_y}" width="22" height="22" rx="4" fill="#edf7f0" stroke="#85bc98"/>
  <text x="96" y="{legend_y + 17}" class="legend"><tspan style="font-weight:900;fill:#2e8b57;">Bold green</tspan> = fak leads the SOTA baseline.</text>
  <rect x="520" y="{legend_y}" width="22" height="22" rx="4" fill="#fff5e8" stroke="#dfbd78"/>
  <text x="552" y="{legend_y + 17}" class="legend">Amber = fak trails (single-stream raw throughput) — shown honestly.</text>
  <text x="1180" y="{legend_y + 17}" class="small">All numbers trace to {esc(d["meta"]["authority"])}.</text>
</svg>
'''


# --------------------------------------------------------------------------- #
# Markdown doc
# --------------------------------------------------------------------------- #
def render_doc(d: dict) -> str:
    m, h, cc, t3 = d["meta"], d["headline"], d["cost_collapse"], d["top3_sota"]
    # the doc lives at the public repo root, so visuals/ is a sibling — no "../"
    hero_rel = m["hero_svg"]
    lb_rel = m["leaderboard_svg"]

    # optional "what the 4.1x is measured against" section (data-driven; rendered
    # right after the headline when present)
    be = d.get("baseline_explainer")
    be_section = (f'## {be["heading"]}\n\n{be["body_md"]}\n\n{be["bound_md"]}\n\n---\n\n'
                  if be else "")

    # mermaid hero bars
    labels = ", ".join(f'"{ROLE_SHORT.get(a["role"], a["role"])}"' for a in cc["arms"])
    mins = [a["minutes"] for a in cc["arms"]]
    ymax = (int(max(mins) / 200) + 1) * 200
    bardata = ", ".join(str(x) for x in mins)

    # top-3 table
    t3_rows = "\n".join(
        f'| **{it["rank"]}** | **{it["system"]}** | {it["ships_md"]} | {it["position_md"]} |'
        for it in t3["items"])

    # top-10 table
    t10_rows = []
    for r in d["top10"]:
        rid = "✅" if r["win"] else "—"
        bench = f'**{r["benchmark"].strip()}**' if r["win"] else r["benchmark"].strip()
        base = r["baseline"] if r["win"] else f'**{r["baseline"]}**'
        fak = f'**{r["fak_md"]}**' if r["win"] else r["fak_md"]
        t10_rows.append(f'| {r["rank"]} | {bench} | {r["regime"].lower()} | {base} | {fak} | {rid} |')
    t10 = "\n".join(t10_rows)

    # traceability (derived from top10)
    tr_rows = []
    for r in d["top10"]:
        src = r.get("artifact") or r.get("provenance_doc", "—")
        com = f'`{r["commit"]}`' if r.get("commit") else "—"
        tr_rows.append(f'| {r["benchmark"].strip()} | {r["fak_md"]} | {com} | {esc(src)} |')
    tr = "\n".join(tr_rows)

    lineage = "\n".join(f'| **{row["field"]}** | {row["value"]} |' for row in d["lineage"])
    nl = d["not_leading"]
    nl_bullets = "\n".join(f"- {b}" for b in nl["bullets"])
    repro = "\n".join(d["reproduce"])
    # A "private" companion is named but NOT linked (it does not exist in the public
    # tree); a normal entry renders as a relative link.
    see = "\n".join(
        (f'- `{s["href"]}` (private companion — not published) — {s["note"]}'
         if s.get("private")
         else f'- [`{s["text"]}`]({s["href"]}) — {s["note"]}')
        for s in d["see_also"])

    return f'''# {m["title"]}

**Date:** {m["as_of"]} · **Status:** ✅ {m["version"]} shipped · **Authority:** every number here traces to
[`{m["authority"]}`]({os.path.basename(m["authority"])}) (commit + JSON artifact).

> **Generated, not hand-maintained.** This file is rebuilt from
> [`tools/hero_benchmark.data.json`](tools/hero_benchmark.data.json) by
> [`tools/hero_benchmark_gen.py`](tools/hero_benchmark_gen.py). Edit the data file and rerun the
> generator when a benchmark changes — do not hand-edit this doc or the two SVGs.

> **The frontier-lab move, done honestly.** When a frontier lab ships a model it leads with one
> headline number, a hero chart against its top-3 competitors, and a table of benchmarks with
> its own number **bolded where it wins** — each cell the **absolute measured score** (e.g.
> "77.2%"), the competitor's score beside it, *not* a "5× better than a weak baseline" claim —
> and the good ones also show the benchmarks they *lose*
> (Opus 4.5 led SWE-bench / Terminal-Bench / ARC-AGI-2 but trailed Gemini 3 Pro on GPQA and
> Humanity's Last Exam). This is that comparison for `fak` — with the same honesty: we bold where we
> win and we show, un-bolded, the two benchmarks where we lose.

---

## What this benchmark actually measures (read this first)

`fak` is **not a frontier model** and **not a faster model server**. So the hero number is *not* a
SWE-bench resolve rate and *not* raw tokens/sec — comparing it on those would be the dishonest
apples-to-oranges con. `fak` is an **agent-serving kernel**, and the axis it is built to win is
**agent-fleet serving efficiency**: how much of a multi-agent, multi-turn workload's *shared* prefill
work it can do **once** instead of every turn, for every agent.

So the "frontier" here is the set of **SOTA serving systems** — SGLang, vLLM, llama.cpp — and the
hero result is measured against the best already-shipped one, **not a strawman**. One honest caveat
up front: the fleet baseline reproduces that warm-KV reuse *discipline* on `fak`'s **own** kernel
(both arms, kernel held constant), so the headline 4.1× isolates **reuse**, not kernel speed — the
next section spells that out.

---

## The headline

![Hero: fak does the fleet's shared work once, not every turn]({hero_rel})

> {h["statement"]}

{h["bitexact"]}

```mermaid
---
config:
  theme: base
---
xychart-beta
    title "Total time to serve the 50×5 fleet (minutes — lower is better)"
    x-axis [{labels}]
    y-axis "Minutes" 0 --> {ymax}
    bar [{bardata}]
```

---

{be_section}## The chart: top-3 SOTA comparison

{t3["intro_md"]}

| # | SOTA system | What it ships (the floor) | Where `fak` sits |
|---|---|---|---|
{t3_rows}

**Read:** on the cross-agent reuse axis, #1 and #2 define the floor `fak` clears by 4.1×; #3 is the
honest place `fak` is behind. That is the whole comparison in one line.

---

## The SOTA benchmark leaderboard

![SOTA leaderboard: fak's number bolded where it leads]({lb_rel})

{d["leaderboard"]["note_md"]}

| # | Benchmark | Regime | SOTA baseline | `fak` result | Lead? |
|---|---|---|---|---:|:---:|
{t10}

> {d["leaderboard_takeaway_md"]}

---

## Where `fak` does **not** lead (the honest fence)

{nl["intro"]}

{nl_bullets}

{nl["synthesis"]}

---

## Lineage

| field | value |
|---|---|
{lineage}

> {d["regime_rule_md"]}

---

## Traceability — every hero number to its source

| Benchmark | `fak` | Commit | Artifact / doc |
|---|---|---|---|
{tr}

---

## Reproduce

```bash
{repro}
```

---

## See also

{see}
'''


# --------------------------------------------------------------------------- #
# Optional standalone HTML widget (Chart.js cost-collapse)
# --------------------------------------------------------------------------- #
def render_html(d: dict) -> str:
    h, cc = d["headline"], d["cost_collapse"]
    cards = "\n".join(
        f'    <div class="card"><div class="lbl">{esc(c["label"])}</div>'
        f'<div class="val">{esc(c["value"])}</div><div class="sub">{esc(c["sub"])}</div></div>'
        for c in h["cards"])
    labels = ", ".join(json.dumps(f'{ROLE_SHORT.get(a["role"], a["role"])}  {a["display"].split("·")[0].strip()}')
                       for a in cc["arms"])
    mins = ", ".join(str(a["minutes"]) for a in cc["arms"])
    colors = ", ".join(json.dumps(ROLE_FILL.get(a["role"], "#6a7685")) for a in cc["arms"])
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{esc(d["meta"]["title"])}</title>
<style>
  body {{ font-family: "Segoe UI", system-ui, Arial, sans-serif; color: #1b2733; max-width: 760px; margin: 1.5rem auto; padding: 0 1rem; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 1rem 0; }}
  .card {{ background: #eef5f6; border-radius: 8px; padding: 1rem; }}
  .lbl {{ font-size: 13px; color: #5b6a7a; }}
  .val {{ font-size: 26px; font-weight: 800; color: #2e8b57; }}
  .sub {{ font-size: 12px; color: #8090a0; }}
  .note {{ font-size: 12px; color: #8090a0; line-height: 1.6; }}
  h1 {{ font-size: 22px; }}
</style></head>
<body>
  <h1>{esc(d["meta"]["title"])}</h1>
  <p class="note">Generated from <code>tools/hero_benchmark.data.json</code>. {esc(h["primary"]["baseline"])}: <b>{esc(h["primary"]["value"])}</b> {esc(h["primary"]["label"])}.</p>
  <div class="cards">
{cards}
  </div>
  <div style="position: relative; height: 220px;">
    <canvas id="hero" role="img" aria-label="Total time to serve the 50 by 5 agent fleet, log scale minutes."></canvas>
  </div>
  <p class="note">Log scale (minutes) so all three bars are visible — the gap is the point. {esc(cc["footnote"])}</p>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
  <script>
    new Chart(document.getElementById('hero'), {{
      type: 'bar',
      data: {{ labels: [{labels}], datasets: [{{ data: [{mins}], backgroundColor: [{colors}], borderRadius: 4, barPercentage: 0.7 }}] }},
      options: {{ indexAxis: 'y', responsive: true, maintainAspectRatio: false,
        plugins: {{ legend: {{ display: false }}, tooltip: {{ callbacks: {{ label: (c) => c.parsed.x + ' min' }} }} }},
        scales: {{ x: {{ type: 'logarithmic', title: {{ display: true, text: 'minutes (log scale) — lower is better' }},
          ticks: {{ callback: (v) => ([10,100,1000].includes(v) ? v : '') }} }} }} }}
    }});
  </script>
</body></html>
'''


# --------------------------------------------------------------------------- #
# verify-sources: re-assert each top10 number against its committed artifact
# --------------------------------------------------------------------------- #
def verify_sources(d: dict) -> int:
    art_root = os.path.join(ROOT, d["meta"].get("artifact_root", ""))
    checked = failed = 0
    for r in d["top10"]:
        v = r.get("verify")
        if not v:
            continue
        checked += 1
        art = os.path.join(art_root, r["artifact"])
        try:
            obj = load_data(art)
            got = float(resolve_field(obj, v["field"]))
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  row {r['rank']}: {r['artifact']} :: {v['field']} -> {exc}")
            continue
        exp, tol = float(v["expect"]), float(v.get("tol", 1e-6))
        ok = abs(got - exp) <= tol * max(1.0, abs(exp))
        flag = "ok  " if ok else "DRIFT"
        if not ok:
            failed += 1
        print(f"  {flag}  row {r['rank']:>2}: {v['field']} = {got!r} (expect {exp!r}) [{os.path.basename(r['artifact'])}]")
    print(f"verify-sources: {checked - failed}/{checked} artifact fields match the data file.")
    return 1 if failed else 0


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def build_outputs(d: dict, want_html: bool) -> dict:
    m = d["meta"]
    out = {
        os.path.join(ROOT, m["doc_path"]): render_doc(d),
        os.path.join(ROOT, m["hero_svg"]): svg_hero(d),
        os.path.join(ROOT, m["leaderboard_svg"]): svg_leaderboard(d),
    }
    if want_html:
        out[os.path.join(ROOT, m["html_path"])] = render_html(d)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Rebuild the fak hero benchmark comparison from data.")
    ap.add_argument("--data", default=DEFAULT_DATA, help="source-of-truth data file")
    ap.add_argument("--check", action="store_true", help="don't write; exit 1 if any output drifted")
    ap.add_argument("--verify-sources", action="store_true", help="re-assert numbers vs committed artifacts")
    ap.add_argument("--html", action="store_true", help="also (re)generate the standalone HTML widget")
    args = ap.parse_args(argv)

    d = load_data(args.data)
    validate_data(d)

    if args.verify_sources:
        return verify_sources(d)

    # --check considers the html only if it already exists (don't force its creation)
    want_html = args.html or (args.check and os.path.exists(os.path.join(ROOT, d["meta"]["html_path"])))
    outputs = build_outputs(d, want_html)

    if args.check:
        drift = [p for p, content in outputs.items() if read_text(p) != content]
        if drift:
            print("DRIFT — these outputs are stale vs the data file (run: python tools/hero_benchmark_gen.py):")
            for p in drift:
                print(f"  {os.path.relpath(p, ROOT)}")
            return 1
        print(f"check: all {len(outputs)} generated outputs are up to date with {os.path.relpath(args.data, ROOT)}.")
        return 0

    for path, content in outputs.items():
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(content)
        print(f"wrote {os.path.relpath(path, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
