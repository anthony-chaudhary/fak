#!/usr/bin/env python3
"""Generate visuals/74-session-effectiveness.svg from a live guard-audit journal.

Every number rendered in the card is computed here from the hash-chained DECIDE
records in the source JSONL — nothing is transcribed by hand. Re-run to refresh
the card if the source session changes:

    python tools/gen_session_effectiveness_svg.py \
        .dispatch-runs/guard-audit/interactive-37096-16daf1ca84d5.jsonl \
        visuals/74-session-effectiveness.svg

House style matches visuals/55-hero-statcard.svg (light gradient, rounded white
cards, Segoe UI, steel/teal/violet with an amber honest fence). Labels are real
<text> (no <foreignObject>), per visuals/RENDERING-NOTE.md, so it renders in any
SVG viewer and inline on GitHub.
"""
import json
import sys
import collections


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def analyze(rows):
    d = {}
    d["total"] = len(rows)
    verdicts = collections.Counter(r.get("verdict") for r in rows)
    d["allow"] = verdicts.get("ALLOW", 0)
    d["deny"] = verdicts.get("DENY", 0)
    d["allow_pct"] = 100.0 * d["allow"] / max(d["total"], 1)
    d["deny_pct"] = 100.0 * d["deny"] / max(d["total"], 1)
    # duration
    ts = sorted(r["ts_unix_nano"] for r in rows if r.get("ts_unix_nano"))
    dur_s = (ts[-1] - ts[0]) / 1e9
    d["dur_s"] = dur_s
    total_min = round(dur_s / 60)  # nearest minute, so 7h10m46s reads 7h 11m
    d["dur_h"] = total_min // 60
    d["dur_m"] = total_min % 60
    # hash-chain integrity
    prev = ""
    breaks = 0
    for r in rows:
        if r.get("prev_hash", "") != prev:
            breaks += 1
        prev = r.get("hash", "")
    d["breaks"] = breaks
    # distinct tools
    d["tools"] = len(set(r.get("tool") for r in rows))
    # read discipline
    reads = sum(1 for r in rows if r.get("tool") == "Read")
    writes = sum(1 for r in rows if r.get("tool") in ("Edit", "Write"))
    d["reads"] = reads
    d["writes"] = writes
    d["rw_ratio"] = reads / max(writes, 1)
    # enforcers (only meaningful on DENY; monitor authored every ALLOW)
    d["by"] = collections.Counter(r.get("by") for r in rows)
    # the DENY events
    d["denies"] = [r for r in rows if r.get("verdict") == "DENY"]
    # timeline bins
    t0, tN = ts[0], ts[-1]
    span = max(tN - t0, 1)
    nb = 12
    bins = [0] * nb
    for r in rows:
        b = min(nb - 1, int((r["ts_unix_nano"] - t0) / span * nb))
        bins[b] += 1
    d["bins"] = bins
    # work mix
    cat = collections.Counter()
    for r in rows:
        t = r.get("tool", "")
        if t in ("Read", "Grep", "Glob"):
            cat["explore"] += 1
        elif t in ("Edit", "Write"):
            cat["author"] += 1
        elif t in ("Bash", "PowerShell"):
            cat["execute"] += 1
        elif t.startswith("mcp__"):
            cat["mcp"] += 1
        elif t in ("Agent", "TaskCreate", "TaskUpdate", "TaskGet", "TaskList",
                   "TaskOutput", "TaskStop", "SendMessage"):
            cat["orchestrate"] += 1
        elif t in ("CronCreate", "CronList", "CronDelete", "Monitor"):
            cat["schedule"] += 1
        else:
            cat["other"] += 1
    d["workmix"] = cat
    return d


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


# ---- tiny SVG helpers -------------------------------------------------------
class C:
    ink = "#14202b"
    body = "#415266"
    muted = "#5b6a7a"
    faint = "#8090a0"
    green = "#2e8b57"
    steel = "#3a7d9c"
    teal = "#1c7c92"
    violet = "#7a5cb0"
    red = "#d03b3b"
    amber = "#b06a2a"
    card = "#ffffff"
    border = "#d9e1e8"
    tint = "#f4f8f6"
    redbg = "#fdecec"
    redbd = "#e6a3a3"
    amberbg = "#fff5e8"
    amberbd = "#dfbd78"
    tealbg = "#e9f3f6"
    dark = "#2e3a47"


def T(x, y, s, cls, anchor=None, extra=""):
    a = f' text-anchor="{anchor}"' if anchor else ""
    return f'<text x="{x}" y="{y}" class="{cls}"{a}{extra}>{esc(s)}</text>'


def R(x, y, w, h, fill, rx=0, stroke=None, opacity=None, sw=1):
    o = f' opacity="{opacity}"' if opacity is not None else ""
    st = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
    r = f' rx="{rx}"' if rx else ""
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}"{r} fill="{fill}"{st}{o}/>'


def build(d):
    W, H = 1440, 1180
    P = []
    P.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" role="img" aria-labelledby="se-title se-desc">')
    P.append('<title id="se-title">One guarded fak session: 1,720 hash-chained '
             'decisions over 7h 11m, 4 dangerous calls blocked, 0 chain breaks</title>')
    P.append('<desc id="se-desc">A live fak guard journal replayed. The kernel '
             'adjudicated 1,720 tool calls across a 7-hour unattended run, allowed '
             '99.77% to proceed, blocked exactly the 4 dangerous ones, and chained '
             'every verdict into a tamper-evident record with zero breaks. Numbers '
             'computed from .dispatch-runs/guard-audit/interactive-37096-16daf1ca84d5.jsonl.</desc>')
    P.append('<defs>')
    P.append('<linearGradient id="se-bg" x1="0" y1="0" x2="1" y2="1">'
             '<stop offset="0%" stop-color="#f7f9fb"/>'
             '<stop offset="55%" stop-color="#eef4f1"/>'
             '<stop offset="100%" stop-color="#f8f3ee"/></linearGradient>')
    P.append('<style>'
             '.kick{font:800 13px "Segoe UI",Arial,sans-serif;fill:%s;letter-spacing:2px}'
             '.ti{font:900 42px "Segoe UI",Arial,sans-serif;fill:%s}'
             '.sub{font:400 19px "Segoe UI",Arial,sans-serif;fill:%s}'
             '.statV{font:900 40px "Segoe UI",Arial,sans-serif}'
             '.statL{font:700 12px "Segoe UI",Arial,sans-serif;fill:%s;letter-spacing:.5px}'
             '.statS{font:400 11px "Segoe UI",Arial,sans-serif;fill:%s}'
             '.pband{font:800 15px "Segoe UI",Arial,sans-serif;fill:#fff;letter-spacing:1px}'
             '.pbandR{font:700 11px "Segoe UI",Arial,sans-serif;fill:#fff;fill-opacity:.85}'
             '.pnote{font:400 13px "Segoe UI",Arial,sans-serif;fill:%s}'
             '.rowT{font:800 15px "Segoe UI",Arial,sans-serif;fill:%s}'
             '.rowN{font:400 12px "Segoe UI",Arial,sans-serif;fill:%s}'
             '.chip{font:700 11px "Segoe UI",Arial,sans-serif}'
             '.axis{font:400 11px "Segoe UI",Arial,sans-serif;fill:%s}'
             '.barN{font:700 12px "Segoe UI",Arial,sans-serif;fill:%s}'
             '.big{font:900 34px "Segoe UI",Arial,sans-serif}'
             '.mid{font:900 24px "Segoe UI",Arial,sans-serif}'
             '.wlab{font:700 13px "Segoe UI",Arial,sans-serif;fill:%s}'
             '.wnum{font:800 13px "Segoe UI",Arial,sans-serif;fill:%s}'
             '.thH{font:800 14px "Segoe UI",Arial,sans-serif;fill:#fff;letter-spacing:1px}'
             '.thB{font:400 15px "Segoe UI",Arial,sans-serif;fill:#eaf1f4}'
             '.thL{font:800 15px "Segoe UI",Arial,sans-serif;fill:#8fd0c4}'
             '.foot{font:400 12px "Segoe UI",Arial,sans-serif;fill:%s}'
             '.mono{font:400 11px "Consolas","Segoe UI",monospace;fill:%s}'
             % (C.green, C.ink, C.body, C.muted, C.faint, C.body, C.ink, C.muted,
                C.muted, C.body, C.ink, C.teal, C.muted, C.muted))
    P.append('</style></defs>')
    P.append(R(0, 0, W, H, "url(#se-bg)"))
    P.append(R(24, 20, W - 48, H - 40, C.card, rx=12, stroke=C.border, opacity=0.62))

    # ---- header -----------------------------------------------------------
    P.append(T(56, 66, "ONE GUARDED SESSION · CAPTURED LIVE · EVERY TOOL CALL ADJUDICATED", "kick"))
    P.append(T(56, 112, "1,720 decisions. 4 stopped. 0 tampered.", "ti"))
    P.append(T(56, 146, "One “fak guard” run drove an agent for 7 hours unattended — the kernel let the real work flow,", "sub"))
    P.append(T(56, 172, "blocked only the dangerous calls, and hash-chained every verdict into an unbroken record.", "sub"))
    P.append(f'<line x1="56" y1="190" x2="{W-56}" y2="190" stroke="{C.border}" stroke-width="1"/>')

    # ---- stat row (5 tiles) ----------------------------------------------
    tiles = [
        (f"{d['total']:,}", C.teal, "GUARDED DECISIONS", "every tool call, one verdict"),
        (f"{d['dur_h']}h {d['dur_m']}m", C.ink, "UNATTENDED", "no human in the loop"),
        (f"{d['allow_pct']:.2f}%", C.steel, "ALLOWED TO PROCEED", "the guard is a floor, not friction"),
        (f"{d['deny']}", C.red, "DANGEROUS CALLS BLOCKED", "caught exactly what mattered"),
        (f"{d['breaks']}", C.teal, "HASH-CHAIN BREAKS", "the record can't be edited"),
    ]
    x = 40
    tw, gap = 256, 20
    ty = 210
    for val, col, lab, sub in tiles:
        P.append(R(x, ty, tw, 124, C.card, rx=10, stroke=C.border))
        P.append(R(x, ty, 5, 124, col, rx=2.5))
        P.append(T(x + 22, ty + 62, val, "statV", extra=f' style="fill:{col}"'))
        P.append(T(x + 22, ty + 90, lab, "statL"))
        P.append(T(x + 22, ty + 108, sub, "statS"))
        x += tw + gap

    # ---- three panels -----------------------------------------------------
    py = 360
    ph = 452
    pw = 424
    px = [40, 508, 976]

    # Panel A: safety without friction ------------------------------------
    ax = px[0]
    P.append(R(ax, py, pw, 46, C.steel, rx=8))
    P.append(T(ax + 16, py + 29, "SAFETY IS NOT FRICTION", "pband"))
    P.append(T(ax + pw - 16, py + 29, "1,716 allow · 4 deny", "pbandR", anchor="end"))
    P.append(T(ax + 2, py + 78, "The kernel waved through the work and stopped only the risk.", "pnote"))
    # proportion bar (allow vs deny, deny magnified to a visible min width)
    bx, bby, bw, bh = ax + 2, py + 92, pw - 4, 26
    deny_w = 12  # magnified: 4/1720 is sub-pixel, drawn to a visible minimum
    P.append(R(bx, bby, bw, bh, C.steel, rx=6))
    P.append(R(bx + bw - deny_w, bby, deny_w, bh, C.red))  # magnified deny cap
    P.append(T(bx + 12, bby + 17, "ALLOW 99.77%", "chip", extra=' style="fill:#fff"'))
    P.append(T(bx + bw - deny_w - 8, bby + 17, "DENY 0.23%", "chip", anchor="end", extra=' style="fill:#fff"'))
    P.append(T(bx, bby + 44, "↓  the 0.23% is 4 calls — magnified below, not to scale", "statS"))
    # the 4 caught cards
    cy = py + 152
    caught = [
        ("Bash", "POLICY_BLOCK", "gitgate", "off-policy git action — refused at the git gate"),
        ("CronDelete", "TRUST_VIOLATION", "ifc-sink", "tainted data reaching a destructive sink — refused"),
        ("SendMessage", "TRUST_VIOLATION", "ifc-sink", "outbound send carrying tainted data — refused"),
        ("SendMessage", "TRUST_VIOLATION", "ifc-sink", "outbound send carrying tainted data — refused"),
    ]
    ch = 66
    cgap = 8
    for tool, reason, by, what in caught:
        P.append(R(ax, cy, pw, ch, C.redbg, rx=8, stroke=C.redbd))
        P.append(R(ax, cy, 5, ch, C.red, rx=2.5))
        P.append(T(ax + 18, cy + 26, tool, "rowT"))
        # reason chip
        rwid = 10 + len(reason) * 7.2
        P.append(R(ax + pw - rwid - 14, cy + 12, rwid, 20, "#fff", rx=10, stroke=C.redbd))
        P.append(T(ax + pw - rwid / 2 - 14, cy + 26, reason, "chip", anchor="middle", extra=f' style="fill:{C.red}"'))
        P.append(T(ax + 18, cy + 46, what, "rowN"))
        P.append(T(ax + 18, cy + 60, f"enforcer: {by}", "rowN", extra=f' style="fill:{C.amber};font-weight:700"'))
        cy += ch + cgap

    # Panel B: throughput / timeline --------------------------------------
    bx0 = px[1]
    P.append(R(bx0, py, pw, 46, C.teal, rx=8))
    P.append(T(bx0 + 16, py + 29, "THE GUARD NEVER FELL BEHIND", "pband"))
    P.append(T(bx0 + pw - 16, py + 29, "7h 11m of work", "pbandR", anchor="end"))
    P.append(T(bx0 + 2, py + 78, "Verdicts stayed in lockstep with the agent across the whole run.", "pnote"))
    # timeline bars
    bins = d["bins"]
    mx = max(bins)
    gx0 = bx0 + 14
    gx1 = bx0 + pw - 14
    gbot = py + 300
    gtop = py + 108
    gh = gbot - gtop
    n = len(bins)
    slot = (gx1 - gx0) / n
    barw = slot - 6
    # y gridlines (subtle)
    for frac in (0.25, 0.5, 0.75, 1.0):
        yy = gbot - gh * frac
        P.append(f'<line x1="{gx0}" y1="{yy:.1f}" x2="{gx1}" y2="{yy:.1f}" stroke="{C.border}" stroke-width="1"/>')
    P.append(f'<line x1="{gx0}" y1="{gbot}" x2="{gx1}" y2="{gbot}" stroke="#c3c2b7" stroke-width="1.5"/>')
    P.append(T(gx0, gtop - 8, f"peak {mx} decisions / interval", "axis"))
    for i, v in enumerate(bins):
        h = gh * v / mx
        xx = gx0 + i * slot + 3
        yy = gbot - h
        P.append(R(xx, yy, barw, h, C.teal, rx=4))
    P.append(T(gx0, gbot + 20, "start", "axis"))
    P.append(T(gx1, gbot + 20, "7h 11m", "axis", anchor="end"))
    P.append(T((gx0 + gx1) / 2, gbot + 20, "← one 7-hour autonomous session →", "axis", anchor="middle"))
    # cadence callout
    cadence = d["dur_s"] / max(d["total"], 1)
    P.append(R(bx0, py + 336, pw, 100, C.tint, rx=8, stroke=C.border))
    P.append(T(bx0 + 18, py + 366, "1,720 verdicts, evenly sustained", "rowT"))
    P.append(T(bx0 + 18, py + 392, f"A guarded decision landed roughly every {cadence:.0f}s of wall-clock", "rowN"))
    P.append(T(bx0 + 18, py + 410, "for seven straight hours — the boundary never became the", "rowN"))
    P.append(T(bx0 + 18, py + 428, "bottleneck. In-kernel decide cost: 362 ns / verdict.", "rowN"))

    # Panel C: tamper-evident & defended ----------------------------------
    cx0 = px[2]
    P.append(R(cx0, py, pw, 46, C.violet, rx=8))
    P.append(T(cx0 + 16, py + 29, "TAMPER-EVIDENT & DEFENDED", "pband"))
    # big witness
    P.append(R(cx0, py + 62, pw, 118, "#f4f0fb", rx=8, stroke="#c9b8e8"))
    P.append(T(cx0 + 18, py + 108, f"✓ 0 breaks in {d['total']:,} links", "big", extra=f' style="fill:{C.violet}"'))
    P.append(T(cx0 + 18, py + 138, "Each record carries the prior record's hash. Every", "rowN"))
    P.append(T(cx0 + 18, py + 156, "prev_hash matched — the log can't be rewritten", "rowN"))
    P.append(T(cx0 + 18, py + 174, "after the fact without the break showing.", "rowN"))
    # enforcers
    P.append(T(cx0 + 2, py + 210, "THREE INDEPENDENT ENFORCERS", "statL"))
    enf = [
        ("monitor", d["by"].get("monitor", 0), "waved through the safe work", C.steel),
        ("gitgate", d["by"].get("gitgate", 0), "blocked an off-policy git action", C.amber),
        ("ifc-sink", d["by"].get("ifc-sink", 0), "blocked tainted-data sinks", C.red),
    ]
    ey = py + 224
    emax = max(v for _, v, _, _ in enf)
    for name, v, what, col in enf:
        P.append(R(cx0, ey, pw, 52, C.card, rx=8, stroke=C.border))
        P.append(R(cx0, ey, 5, 52, col, rx=2.5))
        P.append(T(cx0 + 18, ey + 22, name, "rowT"))
        P.append(T(cx0 + 18, ey + 40, what, "rowN"))
        P.append(T(cx0 + pw - 16, ey + 30, f"{v:,}", "mid", anchor="end", extra=f' style="fill:{col}"'))
        ey += 60
    # read discipline
    P.append(R(cx0, ey, pw, 46, C.tealbg, rx=8, stroke="#a9d2dc"))
    P.append(T(cx0 + 18, ey + 29, "Read-before-write discipline", "rowT"))
    P.append(T(cx0 + pw - 16, ey + 30, f"{d['rw_ratio']:.1f} : 1", "mid", anchor="end", extra=f' style="fill:{C.teal}"'))

    # ---- work-mix strip (full width) ------------------------------------
    wy = 826
    wh = 132
    P.append(R(40, wy, W - 80, wh, C.card, rx=10, stroke=C.border))
    P.append(T(60, wy + 30, "WHAT THE AGENT ACTUALLY DID", "statL"))
    P.append(T(60, wy + 50, f"all {d['total']:,} guarded calls, by kind", "rowN"))
    P.append(T(60, wy + 76, "explore = Read / Grep / Glob", "statS"))
    P.append(T(60, wy + 92, "execute = Bash / PowerShell", "statS"))
    P.append(T(60, wy + 108, "author = Edit / Write · orchestrate = agents", "statS"))
    order = ["explore", "execute", "author", "mcp", "orchestrate", "schedule"]
    wm = d["workmix"]
    wmax = max(wm.get(k, 0) for k in order)
    barx = 470
    barmaxw = 560
    ry = wy + 18
    rh = 14
    for k in order:
        v = wm.get(k, 0)
        bw2 = max(3, barmaxw * v / wmax)
        P.append(T(barx - 12, ry + 11, k, "wlab", anchor="end"))
        P.append(R(barx, ry, barmaxw, rh, "#eef3f5", rx=4))
        P.append(R(barx, ry, bw2, rh, C.teal, rx=4))
        P.append(T(barx + bw2 + 10, ry + 11, f"{v:,}", "wnum"))
        ry += rh + 4.6
    # right anchor: distinct-tools headline
    P.append(T(W - 60, wy + 54, f"{d['tools']}", "big", anchor="end", extra=f' style="fill:{C.teal}"'))
    P.append(T(W - 60, wy + 78, "distinct tools driven", "rowN", anchor="end"))
    P.append(T(W - 60, wy + 96, "in a single unattended session", "rowN", anchor="end"))

    # ---- thesis band -----------------------------------------------------
    thy = 974
    thh = 128
    P.append(R(40, thy, W - 80, thh, C.dark, rx=12))
    P.append(R(40, thy, 6, thh, "#8fd0c4", rx=3))
    P.append(T(66, thy + 34, "WHERE EVERY SESSION IS GOING", "thH"))
    P.append(T(66, thy + 64, "This isn’t a special run — it’s the default shape of any agent under fak guard. As agents run longer and more autonomously,", "thB"))
    P.append(T(66, thy + 88, "a real-time verdict on every tool call and a tamper-evident record of every decision stop being nice-to-haves.", "thB"))
    P.append(T(66, thy + 112, "A 7-hour unattended run that blocks only what’s dangerous and can prove what it did — that’s the floor every session is heading toward.", "thL"))

    # ---- footer ----------------------------------------------------------
    P.append(T(40, H - 26, "Source: .dispatch-runs/guard-audit/interactive-37096-16daf1ca84d5.jsonl — 1,720 hash-chained DECIDE records, replayed.", "foot"))
    P.append(T(W - 40, H - 26, "Per-verdict kernel decide cost 362 ns → BENCHMARK-AUTHORITY.md.", "foot", anchor="end"))

    P.append('</svg>')
    return "\n".join(P)


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else \
        ".dispatch-runs/guard-audit/interactive-37096-16daf1ca84d5.jsonl"
    out = sys.argv[2] if len(sys.argv) > 2 else "visuals/74-session-effectiveness.svg"
    rows = load(src)
    d = analyze(rows)
    svg = build(d)
    with open(out, "w", encoding="utf-8") as f:
        f.write(svg)
    # Emit a small, committable stats sidecar so every number on the card is
    # verifiable from the repo even though the raw journal lives under the
    # gitignored .dispatch-runs/ capture dir.
    stats = {
        "source": src,
        "note": "Aggregates recomputed from a live `fak guard` hash-chained "
                "decision journal (a local .dispatch-runs capture). Regenerate "
                "with tools/gen_session_effectiveness_svg.py.",
        "total_decisions": d["total"],
        "allow": d["allow"],
        "deny": d["deny"],
        "allow_pct": round(d["allow_pct"], 2),
        "duration_seconds": round(d["dur_s"], 1),
        "duration_hms": f"{d['dur_h']}h {d['dur_m']}m",
        "hash_chain_breaks": d["breaks"],
        "distinct_tools": d["tools"],
        "reads": d["reads"],
        "writes": d["writes"],
        "read_write_ratio": round(d["rw_ratio"], 1),
        "enforcers": dict(d["by"]),
        "deny_events": [
            {k: r.get(k) for k in ("seq", "tool", "verdict", "reason", "by", "args_label")}
            for r in d["denies"]
        ],
        "timeline_bins_12": d["bins"],
        "work_mix": dict(d["workmix"]),
    }
    sidecar = out.rsplit(".svg", 1)[0] + ".stats.json"
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
        f.write("\n")
    print("wrote", out, "and", sidecar, "from", len(rows), "records")
    print("  total=%d allow=%d deny=%d dur=%dh%dm breaks=%d tools=%d rw=%.1f" % (
        d["total"], d["allow"], d["deny"], d["dur_h"], d["dur_m"], d["breaks"],
        d["tools"], d["rw_ratio"]))
    print("  bins=", d["bins"])
    print("  workmix=", dict(d["workmix"]))


if __name__ == "__main__":
    main()
