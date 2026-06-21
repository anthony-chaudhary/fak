#!/usr/bin/env python3
"""
agent_walltime.py - where do the HOURS go? Wall-clock attribution for Claude
Code agent sessions, from the transcript timestamps (not token counts).

`session_audit.py` answers the *token*-weighted questions (input:output, cache
reuse, $). It cannot answer "why did this agent take three hours of wall-clock?"
because tokens are not seconds. This tool does: it folds each session's record
timestamps into a wall-clock budget and splits it into the parts an operator can
actually act on:

  * MODEL   - time the agent sat waiting for the model to produce the next turn
              (think + generate + provider queue). The gap from the last tool
              result (or the user prompt) to the assistant record that answers it.
  * TOOLS   - time the agent sat waiting for tools to run. The gap from an
              assistant record that emitted tool_use(s) to the tool_result that
              answers them. Parallel tool batches share one wall-clock segment
              (they overlap), so they are counted ONCE - no double counting.
  * IDLE    - a machine gap so large it is almost certainly a human-away pause or
              a stalled/hung call, not real model/tool work. Bucketed separately
              (default cutoff --idle-cutoff seconds) so it cannot inflate MODEL.

The headline it prints: of the machine wall-clock, what fraction is MODEL vs
TOOLS, and which tools dominate the TOOLS fraction (so you know whether to chase
faster tools - and which - or accept that it is model latency and attack the
turn COUNT instead).

Timestamp model (Claude Code JSONL): the assistant record is written when a turn
arrives; the tool_result (user) record is written when the tool returns. So
  tool_result.ts - assistant.ts  ~= tool execution wall
  assistant.ts   - prev_input.ts ~= model latency for that turn
Both are exact to transcript-write time. Per-tool granularity inside a PARALLEL
batch is not recoverable from record timestamps (the results share one record /
timestamp); such segments are tagged as a batch and attributed to each tool's
"appeared-in-slow-segment" tally, while the wall total counts the segment once.

Usage:
  python agent_walltime.py [--root DIR ...] [--since-hours N] [--ns-prefix STR]
                           [--idle-cutoff SEC] [--top N] [--json OUT] [--md OUT]
                           [--session FILE]   # attribute ONE transcript in detail
"""
import sys, os, json, glob, argparse, collections, datetime

DEFAULT_ROOTS = [
    os.path.join(os.environ.get("CLAUDE_CONFIG_DIR", os.path.expanduser("~/.claude")), "projects"),
]
NS_INCLUDE_PREFIX = "C--work"   # real project namespaces by default

READ_ONLY_TOOLS = {"Read", "Glob", "Grep", "LS", "NotebookRead", "WebFetch",
                   "WebSearch", "TodoRead", "ToolSearch"}

# duration histogram edges (seconds). A tool's wall lands in the first bin whose
# upper edge it is < ; the last bin is the >= tail. Tells fixed-per-call tax (mass
# in the low bins x many calls) apart from a few long build/test runs (high bins).
BIN_EDGES = [2, 5, 15, 60, 300, float("inf")]
BIN_LABELS = ["<2s", "2-5s", "5-15s", "15-60s", "1-5m", ">5m"]


def bin_index(d):
    for i, edge in enumerate(BIN_EDGES):
        if d < edge:
            return i
    return len(BIN_EDGES) - 1


def parse_ts(s):
    """Parse an ISO-8601 transcript timestamp to epoch seconds (float), or None."""
    if not s or not isinstance(s, str):
        return None
    t = s.strip()
    if t.endswith("Z"):
        t = t[:-1] + "+00:00"
    try:
        return datetime.datetime.fromisoformat(t).timestamp()
    except ValueError:
        # tolerate a trailing 'Z' that fromisoformat on older pythons rejects, or
        # fractional-second weirdness; fall back to a coarse parse.
        try:
            base = t.split("+")[0].split(".")[0]
            return datetime.datetime.fromisoformat(base).replace(
                tzinfo=datetime.timezone.utc).timestamp()
        except ValueError:
            return None


def iter_records(path):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def content_blocks(rec):
    msg = rec.get("message")
    if not isinstance(msg, dict):
        return []
    c = msg.get("content")
    if isinstance(c, list):
        return c
    return []


def model_of(rec):
    msg = rec.get("message")
    if isinstance(msg, dict):
        return msg.get("model") or ""
    return ""


def attribute_session(path):
    """Return a dict of wall-clock buckets for one transcript, plus segment lists."""
    # Pass 1: a time-ordered list of timestamped events we care about.
    #   ("assistant", ts, [tool_use names], msg_id)
    #   ("tool_result", ts, [tool names answered])  (names resolved via the id map)
    #   ("user_prompt", ts)   human typed a new instruction (content is a str)
    tool_name_by_id = {}
    events = []
    for rec in iter_records(path):
        typ = rec.get("type")
        ts = parse_ts(rec.get("timestamp"))
        if typ == "assistant":
            uses = []
            for b in content_blocks(rec):
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    nm = b.get("name") or "?"
                    tid = b.get("id")
                    if tid:
                        tool_name_by_id[tid] = nm
                    uses.append(nm)
            if ts is not None:
                events.append(["assistant", ts, uses, model_of(rec)])
        elif typ == "user":
            msg = rec.get("message")
            c = msg.get("content") if isinstance(msg, dict) else None
            if isinstance(c, list):
                answered = []
                for b in c:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        tid = b.get("tool_use_id")
                        answered.append(tool_name_by_id.get(tid, "?"))
                if answered and ts is not None:
                    events.append(["tool_result", ts, answered, ""])
                elif ts is not None and not rec.get("isMeta"):
                    # a real human prompt (string content) - a turn boundary / idle source
                    events.append(["user_prompt", ts, [], ""])
            elif isinstance(c, str) and ts is not None and not rec.get("isMeta"):
                events.append(["user_prompt", ts, [], ""])

    events.sort(key=lambda e: e[1])

    buckets = {"model": 0.0, "tools": 0.0, "idle": 0.0}
    tool_wall = collections.Counter()      # wall attributed to segments a tool appeared in
    tool_calls = collections.Counter()
    tool_bins = collections.defaultdict(lambda: [[0, 0.0] for _ in BIN_EDGES])
    model_turns = 0
    slow_tool_segs = []                    # (dur, [tools])
    slow_model_segs = []                   # (dur, model)
    idle_segs = []

    return (events, buckets, tool_wall, tool_calls, tool_bins,
            slow_tool_segs, slow_model_segs, idle_segs, model_turns)


def fold(path, idle_cutoff):
    (events, buckets, tool_wall, tool_calls, tool_bins, slow_tool, slow_model,
     idle_segs, model_turns) = attribute_session(path)
    span = 0.0
    if events:
        span = events[-1][1] - events[0][1]
    for i in range(1, len(events)):
        prev, cur = events[i - 1], events[i]
        dur = cur[1] - prev[1]
        if dur < 0:
            continue
        kind_prev, kind_cur = prev[0], cur[0]
        # classify the GAP by what bookends it.
        if kind_cur == "assistant":
            # model produced this turn after the previous event.
            if dur >= idle_cutoff:
                buckets["idle"] += dur
                idle_segs.append((dur, "before-assistant"))
            else:
                buckets["model"] += dur
                model_turns += 1
                slow_model.append((dur, cur[3]))
        elif kind_cur == "tool_result":
            # tools ran between the prior assistant and this result.
            for nm in cur[2]:
                tool_calls[nm] += 1
            if dur >= idle_cutoff:
                buckets["idle"] += dur
                idle_segs.append((dur, "tools:" + ",".join(sorted(set(cur[2])))))
            else:
                buckets["tools"] += dur
                names = sorted(set(cur[2]))
                bi = bin_index(dur)
                for nm in names:
                    tool_wall[nm] += dur
                    tool_bins[nm][bi][0] += 1
                    tool_bins[nm][bi][1] += dur
                slow_tool.append((dur, names))
        elif kind_cur == "user_prompt":
            # human typed; this gap is operator wall, not the agent's fault.
            buckets["idle"] += dur
            idle_segs.append((dur, "human-prompt"))
    slow_tool.sort(reverse=True)
    slow_model.sort(reverse=True)
    idle_segs.sort(reverse=True)
    return {
        "path": path,
        "session": os.path.basename(path)[:8],
        "span_s": span,
        "buckets": buckets,
        "tool_wall": dict(tool_wall),
        "tool_calls": dict(tool_calls),
        "tool_bins": {nm: [[c, w] for c, w in bins] for nm, bins in tool_bins.items()},
        "model_turns": model_turns,
        "slow_tool": slow_tool[:8],
        "slow_model": slow_model[:8],
        "idle_segs": idle_segs[:8],
        "n_events": len(events),
    }


def discover(roots, ns_prefix, since_hours):
    cutoff = None
    if since_hours:
        cutoff = datetime.datetime.now().timestamp() - since_hours * 3600
    found = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for ns in os.listdir(root):
            if ns_prefix and not ns.startswith(ns_prefix):
                continue
            for p in glob.glob(os.path.join(root, ns, "*.jsonl")):
                try:
                    mt = os.path.getmtime(p)
                except OSError:
                    continue
                if cutoff and mt < cutoff:
                    continue
                if os.path.getsize(p) == 0:
                    continue
                found.append(p)
    return sorted(found, key=os.path.getmtime, reverse=True)


def hms(s):
    s = int(round(s))
    h, r = divmod(s, 3600)
    m, sec = divmod(r, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"


def pct(x, total):
    return (100.0 * x / total) if total else 0.0


def render(sessions, top, idle_cutoff):
    g_model = sum(s["buckets"]["model"] for s in sessions)
    g_tools = sum(s["buckets"]["tools"] for s in sessions)
    g_idle = sum(s["buckets"]["idle"] for s in sessions)
    g_active = g_model + g_tools
    g_tool_wall = collections.Counter()
    g_tool_calls = collections.Counter()
    g_tool_bins = collections.defaultdict(lambda: [[0, 0.0] for _ in BIN_EDGES])
    for s in sessions:
        g_tool_wall.update(s["tool_wall"])
        g_tool_calls.update(s["tool_calls"])
        for nm, bins in s.get("tool_bins", {}).items():
            for i, (c, w) in enumerate(bins):
                g_tool_bins[nm][i][0] += c
                g_tool_bins[nm][i][1] += w

    L = []
    L.append("# Agent wall-clock attribution")
    L.append("")
    L.append(f"Sessions analyzed: **{len(sessions)}**  -  idle/stall cutoff: "
             f"{idle_cutoff}s  -  machine-active wall = MODEL + TOOLS (idle excluded)")
    L.append("")
    L.append(f"- **MODEL latency**: {hms(g_model)}  ({pct(g_model, g_active):.1f}% of active)")
    L.append(f"- **TOOL execution**: {hms(g_tools)}  ({pct(g_tools, g_active):.1f}% of active)")
    L.append(f"- **IDLE/stall** (excluded from active): {hms(g_idle)}")
    L.append(f"- **ACTIVE machine wall**: {hms(g_active)}")
    L.append("")
    L.append("## Tool execution wall, by tool (the fixable bucket)")
    L.append("")
    L.append("| tool | calls | wall (segments tool appeared in) | % of TOOL wall | read-only |")
    L.append("|---|---:|---:|---:|:--:|")
    for nm, w in g_tool_wall.most_common(top):
        ro = "ro" if nm in READ_ONLY_TOOLS else ""
        L.append(f"| {nm} | {g_tool_calls.get(nm,0)} | {hms(w)} | "
                 f"{pct(w, g_tools):.1f}% | {ro} |")
    L.append("")
    L.append("> Note: parallel tool batches share one wall segment (counted once in "
             "ACTIVE); per-tool 'wall' sums every segment a tool appeared in, so a "
             "batch's seconds are credited to each tool in it. Use it to rank which "
             "tools sit on the critical path, not as a strict partition.")
    L.append("")
    L.append("## Duration shape of the dominant tools (fixed per-call tax vs long runs)")
    L.append("")
    L.append("For each tool: wall-clock that fell in each duration bin (and call count). "
             "Mass in the low bins x a big call count = a fixed per-call tax worth cutting "
             "once for every call; mass in the high bins = a few long runs worth cutting "
             "individually.")
    L.append("")
    header = "| tool | " + " | ".join(BIN_LABELS) + " |"
    L.append(header)
    L.append("|---|" + "|".join(["---:"] * len(BIN_LABELS)) + "|")
    for nm, _w in g_tool_wall.most_common(6):
        bins = g_tool_bins.get(nm)
        if not bins:
            continue
        cells = []
        for c, w in bins:
            cells.append(f"{hms(w)} / {c}" if c else "-")
        L.append(f"| {nm} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("_cell = wall / #calls in that bin._")
    L.append("")
    L.append("## Heaviest sessions (by active machine wall)")
    L.append("")
    L.append("| session | active | model | tools | idle | turns | top tool waits |")
    L.append("|---|---:|---:|---:|---:|---:|---|")
    ranked = sorted(sessions, key=lambda s: s["buckets"]["model"] + s["buckets"]["tools"],
                    reverse=True)
    for s in ranked[:top]:
        act = s["buckets"]["model"] + s["buckets"]["tools"]
        top_waits = "; ".join(f"{hms(d)} {'+'.join(t)}" for d, t in s["slow_tool"][:2])
        L.append(f"| {s['session']} | {hms(act)} | {hms(s['buckets']['model'])} | "
                 f"{hms(s['buckets']['tools'])} | {hms(s['buckets']['idle'])} | "
                 f"{s['model_turns']} | {top_waits} |")
    L.append("")
    L.append("## Single longest tool waits across all sessions")
    L.append("")
    all_slow = []
    for s in sessions:
        for d, t in s["slow_tool"]:
            all_slow.append((d, t, s["session"]))
    all_slow.sort(reverse=True)
    L.append("| wall | tool(s) | session |")
    L.append("|---:|---|---|")
    for d, t, sess in all_slow[:top]:
        L.append(f"| {hms(d)} | {'+'.join(t)} | {sess} |")
    L.append("")
    return "\n".join(L), {
        "model_s": g_model, "tools_s": g_tools, "idle_s": g_idle,
        "active_s": g_active, "sessions": len(sessions),
        "tool_wall_s": dict(g_tool_wall), "tool_calls": dict(g_tool_calls),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", default=None)
    ap.add_argument("--ns-prefix", default=NS_INCLUDE_PREFIX)
    ap.add_argument("--since-hours", type=float, default=None)
    ap.add_argument("--idle-cutoff", type=float, default=900.0,
                    help="a machine gap >= this many seconds is idle/stall, not work")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--session", default=None, help="attribute ONE transcript by path")
    ap.add_argument("--json", default=None)
    ap.add_argument("--md", default=None)
    args = ap.parse_args()

    roots = args.root or DEFAULT_ROOTS
    if args.session:
        if not os.path.isfile(args.session):
            print(f"agent_walltime: session transcript not found: {args.session}")
            return 1
        paths = [args.session]
    else:
        paths = discover(roots, args.ns_prefix, args.since_hours)
    if not paths:
        print("agent_walltime: no transcripts found "
              f"(roots={roots}, ns_prefix={args.ns_prefix!r}, since_hours={args.since_hours})")
        return 1

    sessions = [fold(p, args.idle_cutoff) for p in paths]
    md, summary = render(sessions, args.top, args.idle_cutoff)
    sys.stdout.write(md + "\n")
    if args.md:
        with open(args.md, "w", encoding="utf-8") as fh:
            fh.write(md + "\n")
        print(f"\n[wrote {args.md}]")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"summary": summary,
                       "sessions": [{k: v for k, v in s.items() if k != "path"}
                                    for s in sessions]}, fh, indent=2)
        print(f"[wrote {args.json}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
