#!/usr/bin/env python3
"""context_tape.py — render the "how context assembles" tape for ANY data source.

The picture this draws is the live demo's most-shared panel (`cmd/ctxdemo` panel ②,
"how each agent's context assembles"): a row per agent (or per turn, per RSI cycle…),
each row a horizontal bar whose coloured segments are drawn TO SCALE — the shared
prefix, the model's decode, and every tool result a different size. You can see the
context grow unevenly, and you can see how little of it is new each step.

That panel is the best one-glance explanation fak has of why a multi-agent fleet stresses
the KV cache, so this tool does three things the live demo cannot:

  1. LIVE FOR ANY TRAJECTORY. Point it at a real Claude Code session transcript
     (the `.jsonl` `session_audit.py` already reads) and it renders that session's
     real per-turn context tape — the reused prefix (an exact `cache_read` band) plus
     the small fresh tip each turn, coloured by the tool that turn called. The cache
     story, on YOUR session, not a synthetic one.

  2. MODULAR FOR RSI LOOPS AND SIMILAR. The interchange format is a tiny, source-
     agnostic `Tape` JSON: `{title, legend, rows:[{label,total,segments:[{key,value}]}]}`.
     ANY producer can emit a Tape and get the same picture — so the RSI loop's
     keep/revert journal (`internal/rsiloop`) renders as a verdict ladder with the SAME
     renderer, and so can the next producer nobody has written yet. One renderer, many
     sources: that is the "more modular" ask made structural.

  3. EDUCATION / COURSE MATERIAL. It emits a standalone SVG (no server, no JS) that
     drops straight into a README, a docs page, or a `LEARNING-PATH.md` course — so the
     same explanatory visual can live anywhere a human reads, not only on a running
     demo. `--ascii` gives the terminal/markdown-fence twin for Jekyll-served docs.

Subcommands (every adapter BUILDS a Tape, then the ONE renderer draws it):
  scenario   <id>            the synthetic catalog shape (reproduces the demo panel)
  trajectory <session.jsonl> a real Claude Code session — live for any trajectory
  rsi        <journal.jsonl> an rsiloop keep/revert journal — the RSI verdict ladder
  render     <tape.json>     render a hand-authored / piped Tape JSON
Every subcommand takes one of: --svg OUT.svg | --html OUT.html | --ascii | (default) Tape JSON.

Honest fences:
  * The trajectory bands (reused prefix / fresh input / decode) are EXACT provider token
    counts from `message.usage`; the COLOUR of the fresh band is the turn's dominant tool
    name (an attribution of an exact-size band, not a re-measurement of per-tool tokens).
  * The scenario shape is a faithful Python mirror of the Go catalog in
    `cmd/ctxdemo/scenario.go` (same seeded LCG). For the byte-authoritative numbers pass
    `--from-json <(go run ./cmd/ctxdemo -print -json)`; the mirror is for a zero-dependency
    default and is determinism-tested.
  * Drawn to scale means widths are proportional to token counts within ONE tape; two
    different tapes are not necessarily on the same scale.

Companion: docs/explainers/context-tape-visuals.md (the program + the three applications).
Stdlib-only, deterministic. Read-only except the --svg/--html file it is asked to write.
"""
from __future__ import annotations

import argparse
import collections
import html
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

SCHEMA = "fak-context-tape/1"

# ---------------------------------------------------------------------------
# Palette — mirrors cmd/ctxdemo/page.html :root so a rendered SVG matches the
# live demo exactly. Tool segments cycle through TOOL_COLORS (the t0..t3 ramp).
# ---------------------------------------------------------------------------
COL_BG = "#ffffff"
COL_PANEL2 = "#f0f3f6"
COL_LINE = "#d0d7de"
COL_TXT = "#1f2328"
COL_DIM = "#6e7781"
COL_DIM2 = "#57606a"
COL_PREFIX = "#6366f1"   # shared prefix (system + tool schema)
COL_DECODE = "#9aa5b1"   # decode (model output)
COL_REUSED = "#c7d2fe"   # reused prefix (KV cache hit) — a faded prefix
COL_FAK = "#1a7f37"      # keep / win green
COL_AMBER = "#bf8700"    # revert / warn
COL_RED = "#bf3989"      # escalate / fail
COL_SLATE = "#8c96a0"    # neutral / track
TOOL_COLORS = ["#0a85d1", "#8250df", "#bf3989", "#bf8700", "#1a7f37", "#0d9488", "#cf222e", "#6e7781"]

# Stable colours for the well-known segment kinds so every tape agrees.
KIND_COLORS = {
    "prefix": COL_PREFIX,
    "reused": COL_REUSED,
    "decode": COL_DECODE,
    "keep": COL_FAK,
    "revert": COL_AMBER,
    "escalate": COL_RED,
    "track": COL_SLATE,
    "baseline": COL_DIM,
    "none": COL_SLATE,
}


# ---------------------------------------------------------------------------
# The Tape model — the source-agnostic interchange format every adapter emits
# and the one renderer consumes. Plain dataclasses so it round-trips to JSON.
# ---------------------------------------------------------------------------
@dataclass
class Seg:
    key: str                 # segment kind: "prefix" | "reused" | "decode" | a tool name | …
    value: float             # width basis (tokens, or any positive magnitude)
    label: str = ""          # optional inline label drawn if the segment is wide enough


@dataclass
class Row:
    label: str               # left-hand row label, e.g. "agent 0" / "turn 12" / "cycle 3"
    segments: list[Seg] = field(default_factory=list)
    total_label: str = ""    # right-hand value, e.g. "9,569 tok" / "0.83 ✓ kept"

    def total(self) -> float:
        return sum(s.value for s in self.segments)


@dataclass
class Tape:
    title: str
    tag: str = ""                                   # the bordered pill next to the title
    legend: list[dict[str, str]] = field(default_factory=list)   # [{key,label,color}]
    rows: list[Row] = field(default_factory=list)
    caption: str = ""
    scale: str = "shared"                           # "shared" (one scale) | "row" (each row full-width)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA, "title": self.title, "tag": self.tag,
            "legend": self.legend, "scale": self.scale, "caption": self.caption,
            "rows": [{"label": r.label, "total_label": r.total_label,
                      "segments": [asdict(s) for s in r.segments]} for r in self.rows],
        }

    @staticmethod
    def from_json(d: dict[str, Any]) -> "Tape":
        return Tape(
            title=d.get("title", ""), tag=d.get("tag", ""),
            legend=d.get("legend", []), caption=d.get("caption", ""),
            scale=d.get("scale", "shared"),
            rows=[Row(label=r.get("label", ""), total_label=r.get("total_label", ""),
                      segments=[Seg(**s) for s in r.get("segments", [])])
                  for r in d.get("rows", [])],
        )


def color_for(key: str, tool_order: list[str]) -> str:
    """Resolve a segment kind to a colour: well-known kinds are fixed; tool names cycle
    through the tool ramp in first-seen order so a tape's legend and bars always agree."""
    if key in KIND_COLORS:
        return KIND_COLORS[key]
    if key in tool_order:
        return TOOL_COLORS[tool_order.index(key) % len(TOOL_COLORS)]
    return COL_SLATE


def _tool_order(tape: Tape) -> list[str]:
    """First-seen order of the non-builtin segment kinds (the tools), for stable colours."""
    order: list[str] = []
    for r in tape.rows:
        for s in r.segments:
            if s.key not in KIND_COLORS and s.key not in order:
                order.append(s.key)
    return order


# ---------------------------------------------------------------------------
# SVG renderer — one function, the whole picture. Self-contained: no CSS file,
# no JS, no fonts beyond the system stack, so the output drops into any doc.
# ---------------------------------------------------------------------------
def _esc(s: str) -> str:
    return html.escape(str(s), quote=True)


def _wrap(text: str, width_chars: int) -> list[str]:
    """Greedy word-wrap for the caption (monospace-ish metrics, ~width_chars per line)."""
    out, line = [], ""
    for word in text.split():
        if line and len(line) + 1 + len(word) > width_chars:
            out.append(line)
            line = word
        else:
            line = word if not line else line + " " + word
    if line:
        out.append(line)
    return out


def render_svg(tape: Tape, width: int = 920) -> str:
    pad = 20
    font = ("font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;")
    label_w, gap, total_w = 78, 10, 104
    bar_x = pad + label_w + gap
    bar_w = width - bar_x - gap - total_w - pad
    bar_h, row_gap = 22, 9
    row_pitch = bar_h + row_gap

    tools = _tool_order(tape)

    # vertical layout
    y = pad
    title_h = 26
    legend_rows = max(1, (len(tape.legend) + 2) // 3) if tape.legend else 0
    legend_h = legend_rows * 18 + (8 if tape.legend else 0)
    rows_top = y + title_h + legend_h + 6
    rows_h = len(tape.rows) * row_pitch
    cap_lines = _wrap(tape.caption, max(40, (width - 2 * pad) // 7)) if tape.caption else []
    cap_h = (len(cap_lines) * 16 + 12) if cap_lines else 0
    height = rows_top + rows_h + cap_h + pad

    # shared scale: widest row total across the tape
    max_total = max((r.total() for r in tape.rows), default=1.0) or 1.0

    P: list[str] = []
    P.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{int(height)}" '
             f'viewBox="0 0 {width} {int(height)}" font-size="12" style="{font}">')
    P.append(f'<rect width="{width}" height="{int(height)}" fill="{COL_BG}"/>')

    # title + tag pill
    tx = pad
    P.append(f'<text x="{tx}" y="{y+14}" fill="{COL_DIM2}" font-size="13" font-weight="700" '
             f'letter-spacing="0.06em">{_esc(tape.title.upper())}</text>')
    if tape.tag:
        tw = 8 * len(tape.tag) + 16
        tgx = tx + 9 * len(tape.title) + 14
        tgx = min(tgx, width - pad - tw)
        P.append(f'<rect x="{tgx}" y="{y+2}" width="{tw}" height="17" rx="4" fill="none" '
                 f'stroke="{COL_LINE}"/>')
        P.append(f'<text x="{tgx+tw/2:.0f}" y="{y+14}" fill="{COL_DIM}" font-size="10" '
                 f'text-anchor="middle" letter-spacing="0.04em">{_esc(tape.tag.upper())}</text>')

    # legend
    if tape.legend:
        ly = y + title_h + 4
        col_w = (width - 2 * pad) // 3
        for i, item in enumerate(tape.legend):
            col = i % 3
            rowi = i // 3
            lx = pad + col * col_w
            yy = ly + rowi * 18
            c = item.get("color") or color_for(item.get("key", ""), tools)
            P.append(f'<rect x="{lx}" y="{yy}" width="11" height="11" rx="2" fill="{c}"/>')
            P.append(f'<text x="{lx+16}" y="{yy+10}" fill="{COL_DIM2}" font-size="10">'
                     f'{_esc(item.get("label",""))}</text>')

    # rows
    for ri, r in enumerate(tape.rows):
        ry = rows_top + ri * row_pitch
        # left label (right-aligned to the bar)
        P.append(f'<text x="{bar_x-gap}" y="{ry+bar_h-7}" fill="{COL_DIM2}" font-size="11" '
                 f'text-anchor="end">{_esc(r.label)}</text>')
        # track (rounded, faint) + clip so segments keep the rounded corners
        clip = f"clip{ri}"
        P.append(f'<clipPath id="{clip}"><rect x="{bar_x}" y="{ry}" width="{bar_w}" '
                 f'height="{bar_h}" rx="6"/></clipPath>')
        P.append(f'<rect x="{bar_x}" y="{ry}" width="{bar_w}" height="{bar_h}" rx="6" '
                 f'fill="{COL_PANEL2}" stroke="{COL_LINE}"/>')
        P.append(f'<g clip-path="url(#{clip})">')
        rtot = r.total() if tape.scale == "shared" else (r.total() or 1.0)
        denom = max_total if tape.scale == "shared" else (rtot or 1.0)
        x = float(bar_x)
        for s in r.segments:
            w = (s.value / denom) * bar_w if denom else 0.0
            if w <= 0:
                continue
            c = color_for(s.key, tools)
            txtcol = "#ffffff" if c not in (COL_DECODE, COL_REUSED) else COL_TXT
            P.append(f'<rect x="{x:.2f}" y="{ry}" width="{w:.2f}" height="{bar_h}" fill="{c}" '
                     f'stroke="#0f1117" stroke-opacity="0.16" stroke-width="0.6"/>')
            if s.label and w > 22:
                P.append(f'<text x="{x+w/2:.2f}" y="{ry+bar_h-7}" fill="{txtcol}" font-size="9" '
                         f'text-anchor="middle">{_esc(s.label)}</text>')
            x += w
        P.append('</g>')
        # right total
        if r.total_label:
            P.append(f'<text x="{width-pad}" y="{ry+bar_h-7}" fill="{COL_DIM}" font-size="10" '
                     f'text-anchor="end">{_esc(r.total_label)}</text>')

    # caption
    if cap_lines:
        cy = rows_top + rows_h + 14
        for i, line in enumerate(cap_lines):
            P.append(f'<text x="{pad}" y="{cy+i*16}" fill="{COL_DIM2}" font-size="11">'
                     f'{_esc(line)}</text>')

    P.append('</svg>')
    return "\n".join(P)


def render_html(tape: Tape, width: int = 920) -> str:
    """Wrap the SVG in a minimal standalone page (for opening directly in a browser)."""
    svg = render_svg(tape, width)
    return ("<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>{_esc(tape.title)}</title>"
            "<style>body{background:#f6f8fa;margin:0;padding:28px;"
            "font-family:ui-monospace,Menlo,Consolas,monospace}"
            ".card{background:#fff;border:1px solid #d0d7de;border-radius:10px;"
            "padding:8px;max-width:980px;margin:0 auto}</style></head>"
            f"<body><div class='card'>{svg}</div></body></html>")


def render_ascii(tape: Tape, width: int = 92) -> str:
    """Terminal / markdown-fence twin (the Jekyll-served-docs visual). Same data, drawn
    with block glyphs to scale within each row's share of the width."""
    tools = _tool_order(tape)
    glyph_for: dict[str, str] = {
        "prefix": "█", "reused": "▒", "decode": "░",
        "baseline": "░", "keep": "█", "revert": "▓", "escalate": "▚", "track": "▒", "none": "▓",
    }
    L: list[str] = []
    L.append(tape.title.upper() + (f"   [{tape.tag}]" if tape.tag else ""))
    # legend
    leg = []
    tletters: dict[str, str] = {}
    for i, t in enumerate(tools):
        tletters[t] = chr(ord("a") + i)
    for item in tape.legend:
        k = item.get("key", "")
        g = glyph_for.get(k, tletters.get(k, "#"))
        leg.append(f"{g} {item.get('label','')}")
    if leg:
        L.append("legend: " + "  ".join(leg))
    L.append("")
    max_total = max((r.total() for r in tape.rows), default=1.0) or 1.0
    label_w = max((len(r.label) for r in tape.rows), default=6)
    barw = max(20, width - label_w - 18)
    for r in tape.rows:
        denom = max_total if tape.scale == "shared" else (r.total() or 1.0)
        cells = ""
        for s in r.segments:
            n = int(round((s.value / denom) * barw)) if denom else 0
            g = glyph_for.get(s.key, tletters.get(s.key, "#"))
            cells += g * max(0, n)
        cells = cells[:barw].ljust(barw)
        L.append(f"{r.label:>{label_w}} │{cells}│ {r.total_label}")
    if tape.caption:
        L.append("")
        L.extend(_wrap(tape.caption, width))
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Adapter: scenario — the synthetic catalog shape. A faithful Python mirror of
# cmd/ctxdemo/scenario.go (the same seeded LCG), so `scenario fleet-5x50`
# reproduces the live demo's headline panel with no Go build.
# ---------------------------------------------------------------------------
U64 = (1 << 64) - 1

# Mirror of catalog() in cmd/ctxdemo/scenario.go. (id, label, prefix, agents,
# turns, decode, seed, tools[(name,min,max)]). Kept faithful; the Go catalog is
# authoritative — use --from-json for the byte-exact numbers.
CATALOG = {
    "support-bot": dict(label="support bot — many short agents", prefix=256, agents=4,
                        turns=3, decode=16, seed=0xA1,
                        tools=[("kb_lookup", 16, 40), ("order_status", 20, 64)]),
    "coding-agent": dict(label="coding agent — few agents, many turns", prefix=768, agents=3,
                         turns=6, decode=24, seed=0xC0DE,
                         tools=[("read_file", 48, 256), ("grep", 32, 128), ("run_tests", 64, 200)]),
    "deep-research": dict(label="deep research — long context", prefix=1536, agents=4,
                          turns=5, decode=32, seed=0x4EE7,
                          tools=[("web_search", 64, 160), ("web_fetch", 160, 512), ("summarize", 48, 128)]),
    "mixed-fleet": dict(label="mixed fleet — heterogeneous agents, one org prefix", prefix=512,
                        agents=6, turns=4, decode=20, seed=0xF1EE7,
                        tools=[("kb_lookup", 16, 40), ("read_file", 48, 256),
                               ("web_fetch", 128, 448), ("run_tests", 64, 200)]),
    "fleet-5x50": dict(label="FLEET — 5 agents × 50 turns (the headline fleet shape)", prefix=1024,
                       agents=5, turns=50, decode=24, seed=0xF1EE75050,
                       tools=[("kb_lookup", 16, 40), ("read_file", 48, 256),
                              ("web_fetch", 128, 448), ("run_tests", 64, 200)]),
}


def scenario_workload(sid: str) -> dict[str, Any]:
    """Expand a scenario into its deterministic workload — the exact mirror of
    Scenario.Build() in scenario.go (per-agent seeded LCG; T-1 results per agent)."""
    s = CATALOG[sid]
    C, T = s["agents"], s["turns"]
    rt = max(0, T - 1)
    results: list[list[int]] = []
    names: list[list[str]] = []
    tools = s["tools"]
    for c in range(C):
        st = ((0x9E3779B97F4A7C15 * (c + 1)) & U64) ^ s["seed"]
        st &= U64
        row_r, row_n = [], []
        for _ in range(rt):
            st = (st * 6364136223846793005 + 1442695040888963407) & U64
            tool = tools[(st >> 33) % len(tools)]
            st = (st * 6364136223846793005 + 1442695040888963407) & U64
            span = max(1, tool[2] - tool[1] + 1)
            row_r.append(tool[1] + ((st >> 33) % span))
            row_n.append(tool[0])
        results.append(row_r)
        names.append(row_n)
    return {"scenario": dict(id=sid, **s), "results": results, "tool_names": names}


def scenario_tape(sid: str, from_json: str | None = None, max_rows: int = 0) -> Tape:
    if from_json is not None:
        views = json.loads(Path(from_json).read_text(encoding="utf-8"))
        view = next((v for v in views if v["scenario"]["id"] == sid), None)
        if view is None:
            raise SystemExit(f"scenario {sid!r} not in {from_json}")
        scn = view["scenario"]
        wl = view["workload"]
        results, names = wl["results"], wl["tool_names"]
        prefix, decode = scn["prefix"], scn["decode"]
        tools = [(t["name"], t["min_tok"], t["max_tok"]) for t in scn["tools"]]
        label = scn["label"]
    else:
        wl = scenario_workload(sid)
        scn = wl["scenario"]
        results, names = wl["results"], wl["tool_names"]
        prefix, decode = scn["prefix"], scn["decode"]
        tools = scn["tools"]
        label = scn["label"]

    tool_names = [t[0] for t in tools]
    rows: list[Row] = []
    for a, (res, tn) in enumerate(zip(results, names)):
        segs = [Seg("prefix", prefix, "prefix")]
        total = prefix
        for t in range(len(res)):
            segs.append(Seg("decode", decode))
            total += decode
            segs.append(Seg(tn[t], res[t], str(res[t])))
            total += res[t]
        segs.append(Seg("decode", decode))   # the final turn's decode (no tool)
        total += decode
        rows.append(Row(label=f"agent {a}", segments=segs, total_label=f"{total:,} tok"))

    if max_rows and len(rows) > max_rows:
        rows = rows[:max_rows]

    legend = [{"key": "prefix", "label": "shared prefix (system + tool schema)"},
              {"key": "decode", "label": "decode (model output)"}]
    for name, mn, mx in tools:
        legend.append({"key": name, "label": f"{name} ({mn}–{mx} tok)",
                       "color": color_for(name, tool_names)})
    return Tape(
        title="How each agent's context assembles",
        tag="tool results drawn to scale",
        legend=legend, rows=rows,
        caption=("Each agent's context is the same shared prefix plus its own uneven trail of "
                 "decode + tool results — every result a different size (that is the context "
                 "changing per turn). A warm per-agent KV cache reads that shared prefix once per "
                 "agent; fak reads it once for the whole fleet and only ingests the coloured tips. "
                 f"Scenario: {label}."),
    )


# ---------------------------------------------------------------------------
# Adapter: trajectory — a REAL Claude Code session transcript. The cache story
# on your own session: each row is one billed turn; the bands are EXACT provider
# token counts (reused prefix = cache_read, fresh tip = input + cache_creation,
# decode = output), coloured by the turn's dominant tool.
# ---------------------------------------------------------------------------
def trajectory_turns(path: str) -> list[dict[str, Any]]:
    """Per-billed-turn usage + tools, de-duped on message.id (the same identity
    session_audit.py uses; Claude Code writes multiple transcript lines per billed turn)."""
    seen: set[str] = set()
    turns: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("type") != "assistant":
            continue
        msg = r.get("message", {}) or {}
        mid = msg.get("id")
        if mid is not None:
            if mid in seen:
                continue
            seen.add(mid)
        u = msg.get("usage", {}) or {}
        tools = [b.get("name", "?") for b in (msg.get("content") or [])
                 if isinstance(b, dict) and b.get("type") == "tool_use"]
        turns.append({
            "input": int(u.get("input_tokens", 0) or 0),
            "output": int(u.get("output_tokens", 0) or 0),
            "cache_read": int(u.get("cache_read_input_tokens", 0) or 0),
            "cache_create": int(u.get("cache_creation_input_tokens", 0) or 0),
            "tools": tools,
            "model": msg.get("model", "?"),
        })
    return turns


def trajectory_tape(path: str, max_rows: int = 48) -> Tape:
    turns = trajectory_turns(path)
    if not turns:
        raise SystemExit(f"no billed assistant turns found in {path} "
                         "(is it a Claude Code session .jsonl?)")
    note = ""
    if max_rows and len(turns) > max_rows:
        step = len(turns) / max_rows
        sampled = [turns[int(i * step)] for i in range(max_rows)]
        note = (f" (showing {max_rows} of {len(turns)} turns, evenly sampled — "
                "the band sizes shown are still each turn's exact usage)")
        turns = sampled

    tool_universe: list[str] = []
    for t in turns:
        for nm in t["tools"]:
            if nm not in tool_universe:
                tool_universe.append(nm)

    rows: list[Row] = []
    for i, t in enumerate(turns):
        reused = t["cache_read"]
        fresh = t["input"] + t["cache_create"]
        decode = t["output"]
        ingested = reused + fresh
        # colour the fresh tip by the turn's dominant tool (an attribution of an exact band)
        dom = collections.Counter(t["tools"]).most_common(1)
        fresh_key = dom[0][0] if dom else "none"
        segs = [Seg("reused", reused)]
        if fresh:
            segs.append(Seg(fresh_key, fresh, str(fresh) if fresh > 200 else ""))
        if decode:
            segs.append(Seg("decode", decode))
        rows.append(Row(label=f"turn {i+1}", segments=segs,
                        total_label=f"{ingested:,} in"))

    legend = [{"key": "reused", "label": "reused prefix (KV cache hit · cache_read)"},
              {"key": "decode", "label": "decode (model output)"}]
    seen_tools = set()
    for nm in tool_universe:
        if nm == "?" or nm in seen_tools:
            continue
        seen_tools.add(nm)
        legend.append({"key": nm, "label": f"fresh input · {nm}",
                       "color": color_for(nm, tool_universe)})
    if any(not t["tools"] for t in turns):
        legend.append({"key": "none", "label": "fresh input · (no tool this turn)"})

    model = collections.Counter(t["model"] for t in turns).most_common(1)
    model_name = model[0][0] if model else "?"
    return Tape(
        title="How this session's context assembles",
        tag="real trajectory · drawn to scale",
        legend=legend, rows=rows,
        caption=("Each row is one billed turn of a real Claude Code session. The big faded band "
                 "is the prefix the model REUSED from cache that turn (an exact cache_read count) — "
                 "it grows as the transcript grows; the small coloured tip is the fresh input that "
                 "turn (input + cache_creation), and the grey tail is decode. The reused band "
                 "dwarfing the tip IS the frozen-trajectory cache win, made visible on your own "
                 f"session. Model: {model_name}.{note}"),
    )


# ---------------------------------------------------------------------------
# Adapter: rsi — an rsiloop keep/revert journal (internal/rsiloop Row JSONL).
# The SAME renderer, a different shape: a verdict ladder. Each cycle is a row;
# the bar is the candidate metric, the baseline is a faint marker, the colour is
# the decision (keep / revert / escalate / track).
# ---------------------------------------------------------------------------
DECISION_KEY = {"KEEP": "keep", "REVERT": "revert", "ESCALATE": "escalate", "TRACK": "track"}
DECISION_MARK = {"KEEP": "kept", "REVERT": "reverted", "ESCALATE": "escalated", "TRACK": "track"}


def rsi_rows(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def rsi_tape(path: str, max_rows: int = 0) -> Tape:
    jr = rsi_rows(path)
    if not jr:
        raise SystemExit(f"no journal rows in {path} (is it an rsiloop journal .jsonl?)")
    if max_rows and len(jr) > max_rows:
        jr = jr[-max_rows:]   # most recent cycles
    metric_name = next((r.get("metric_name") for r in jr if r.get("metric_name")), "metric")
    lower_better = bool(jr[0].get("lower_better", False))
    # scale to the largest metric magnitude seen (baseline or candidate)
    max((max(abs(r.get("candidate_metric", 0.0)), abs(r.get("baseline", 0.0))) for r in jr),
               default=1.0) or 1.0

    rows: list[Row] = []
    for r in jr:
        dec = r.get("decision", "?")
        key = DECISION_KEY.get(dec, "none")
        cand = float(r.get("candidate_metric", 0.0))
        base = float(r.get("baseline", 0.0))
        measured = r.get("measured", True)
        # the bar: the candidate metric, coloured by the verdict; a faint baseline stub
        # behind it shows the bar it had to beat.
        lo, hi = (min(cand, base), max(cand, base))
        segs = [Seg("baseline", lo, "")]
        delta = hi - lo
        if delta > 0:
            # the part above/below the shared floor is the candidate's gain (keep) or the
            # baseline's lead (revert) — colour that contested band by the verdict.
            segs.append(Seg(key, delta, ""))
        cyc = r.get("cycle", "")
        cand_txt = f"{cand:.3f}" if measured else "n/a"
        lab = f"cycle {cyc}: {r.get('candidate','')}".strip()
        rows.append(Row(label=lab[:28], segments=segs,
                        total_label=f"{cand_txt} {DECISION_MARK.get(dec, dec)}"))

    legend = [
        {"key": "baseline", "label": f"baseline ({metric_name}, the bar to beat)"},
        {"key": "keep", "label": "KEEP — measured gain, suite green, truth clean"},
        {"key": "revert", "label": "REVERT — no non-forgeable gain"},
        {"key": "escalate", "label": "ESCALATE — breaker tripped"},
    ]
    if any(r.get("decision") == "TRACK" for r in jr):
        legend.append({"key": "track", "label": "TRACK — baseline-vs-main point"})
    direction = "lower is better" if lower_better else "higher is better"
    return Tape(
        title="RSI loop — keep or revert, cycle by cycle",
        tag="non-forgeable keep-bit",
        legend=legend, rows=rows, scale="shared",
        caption=("Each row is one recursive-self-improvement cycle: a candidate is measured in an "
                 "isolated worktree, and KEPT only when a witness the author did not write confirms a "
                 f"strictly-measured gain ({metric_name}, {direction}) with the suite green and the "
                 "truth syscall clean — else REVERTed; a streak of non-keeps trips the breaker to "
                 "ESCALATE. The coloured band over the shared baseline floor is the contested delta, "
                 "tinted by the verdict. Same renderer as the context tape — a different producer."),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _emit(tape: Tape, args) -> None:
    if args.svg:
        Path(args.svg).write_text(render_svg(tape, args.width), encoding="utf-8")
        print(f"wrote {args.svg}", file=sys.stderr)
    elif args.html:
        Path(args.html).write_text(render_html(tape, args.width), encoding="utf-8")
        print(f"wrote {args.html}", file=sys.stderr)
    elif args.ascii:
        print(render_ascii(tape))
    else:
        print(json.dumps(tape.to_json(), indent=2))


def _add_out_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--svg", help="write a standalone SVG to this path")
    p.add_argument("--html", help="write a standalone HTML page to this path")
    p.add_argument("--ascii", action="store_true", help="print the terminal/markdown ASCII twin")
    p.add_argument("--width", type=int, default=920, help="SVG width in px (default 920)")
    p.add_argument("--max-rows", type=int, default=0, help="cap rows (0 = no cap; adapters set a default)")


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Render the context-assembly tape for any data source.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sc = sub.add_parser("scenario", help="synthetic catalog shape (reproduces the demo panel)")
    sc.add_argument("id", nargs="?", default="fleet-5x50",
                    help="scenario id (default fleet-5x50 = the headline panel); "
                         "one of: " + ", ".join(CATALOG))
    sc.add_argument("--from-json", help="read the workload from `ctxdemo -print -json` (byte-authoritative)")
    _add_out_flags(sc)

    tr = sub.add_parser("trajectory", help="a real Claude Code session .jsonl — live for any trajectory")
    tr.add_argument("session")
    _add_out_flags(tr)

    rs = sub.add_parser("rsi", help="an rsiloop keep/revert journal .jsonl — the RSI verdict ladder")
    rs.add_argument("journal")
    _add_out_flags(rs)

    rn = sub.add_parser("render", help="render a hand-authored / piped Tape JSON ('-' = stdin)")
    rn.add_argument("tape")
    _add_out_flags(rn)

    sub.add_parser("list", help="list the scenario ids")
    args = ap.parse_args(argv)

    if args.cmd == "list":
        for k, v in CATALOG.items():
            print(f"{k:14} {v['label']}")
        return 0

    if args.cmd == "scenario":
        tape = scenario_tape(args.id, args.from_json, args.max_rows)
    elif args.cmd == "trajectory":
        tape = trajectory_tape(args.session, args.max_rows or 48)
    elif args.cmd == "rsi":
        tape = rsi_tape(args.journal, args.max_rows)
    elif args.cmd == "render":
        raw = sys.stdin.read() if args.tape == "-" else Path(args.tape).read_text(encoding="utf-8")
        tape = Tape.from_json(json.loads(raw))
    else:
        ap.error("unknown command")
        return 2

    _emit(tape, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
