#!/usr/bin/env python3
"""
transcript_workload.py — turn real Claude Code session transcripts into a *replayable*
benchmark workload profile (schema `fak.workload.v1`).

The fak serving benchmarks (`fleetserve`, `bench_llamacpp_turn_agents.py`) drive a
T x C cross-agent shared-prefix workload parameterised by four numbers:

    P  shared prefix tokens (system prompt + tool schemas, prefilled once, cloned to C agents)
    D  assistant tokens decoded per turn
    R  private tool/result tokens ingested between turns
    T  turns per agent

Until now those were *synthetic constants* (prefix=1024, decode=32, result=128). Real
agent sessions look nothing like that: the transcripts on this machine show I:O ~100:1,
93% cache-read (KV reuse), 17-200 tool calls over up to ~450 turns. This tool reads the
EXACT per-turn token accounting out of the transcript JSONL and emits:

  1. marginal DISTRIBUTIONS of P, D, R, T, tool-call fraction, tools/turn, tool mix; and
  2. concrete REPLAY TRACKS — a handful of real sessions reproduced turn-by-turn as
     [{decode, tool, result, name}, ...] so the benchmark can PLAY ONE BACK rather than
     only sampling from the marginals.

The replay tracks are the point: a real session is a *sequence* (a Read, then an Edit,
then a Bash that emits 4 KB, then three text-only turns, ...), and the tool-call fraction
+ result-size pattern is what stresses the kernel's KV reuse. `fleetserve -workload` and
the tuning sweep consume this file.

What is EXACT vs ESTIMATED (be honest — this drives benchmark numbers):
  * output_tokens (D)            EXACT  — billed token count from message.usage
  * cache_read / creation / input EXACT
  * prefix P                     EXACT  — first non-sidechain turn's cache_read (the
                                          preamble already cached before the convo) or,
                                          if cold, its cache_creation
  * result tokens R              ESTIMATE — tool_result has no per-block token count in the
                                          transcript, so R ~= ceil(chars / CHARS_PER_TOK).
                                          We also record the measured cache_creation delta
                                          as an independent cross-check.

Usage:
  python tools/transcript_workload.py profile [--since-days N] [--ns-prefix C--work]
      [--all] [--root DIR ...] [--max N] [--replay-tracks K]
      [--out profile.json] [--md summary.md]
  python tools/transcript_workload.py show profile.json     # human summary of a profile
"""
import sys, os, json, argparse, math, statistics, collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fleet_version
from session_audit import discover, DEFAULT_ROOTS, NS_INCLUDE_PREFIX, READ_ONLY_TOOLS

SCHEMA = "fak.workload.v1"
CHARS_PER_TOK = 4.0   # rough bytes->tokens for tool_result text (flagged ESTIMATE)


def _txt_len(content):
    """char length of a tool_result content that may be str or list of blocks."""
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        n = 0
        for b in content:
            if isinstance(b, dict):
                n += _txt_len(b.get("content", b.get("text", "")))
            elif isinstance(b, str):
                n += len(b)
        return n
    if isinstance(content, dict):
        return _txt_len(content.get("content", content.get("text", "")))
    return 0


def _pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = max(0, min(len(xs) - 1, int(round((p / 100) * (len(xs) - 1)))))
    return xs[k]


def _stats(xs, ints=True):
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"n": 0}
    f = (lambda v: int(round(v))) if ints else (lambda v: round(v, 3))
    return {
        "n": len(xs),
        "min": f(min(xs)),
        "p10": f(_pct(xs, 10)),
        "median": f(statistics.median(xs)),
        "mean": f(statistics.mean(xs)),
        "p90": f(_pct(xs, 90)),
        "max": f(max(xs)),
    }


def analyze_turns(path):
    """Walk ONE transcript in order, collapsing the per-block split records of each API
    response into ONE logical turn, and return per-session + per-turn workload features.

    Claude Code writes one assistant API response as SEVERAL JSONL records — one per content
    block (thinking / text / each tool_use) — and stamps the SAME message.usage on every one
    (verified: 45 records ↔ 16 message ids, out_tokens repeated across a response's splits).
    So a turn MUST be keyed by message.id (fall back to requestId), merging the split records
    and counting usage ONCE; otherwise turns ~3× over-count, output_tokens triple-counts, and
    the tool-call fraction is diluted by the thinking/text-only split records. The trailing
    tool_result(s) are attributed to the turn that issued the tool_use, as R."""
    try:
        lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    except OSError:
        return None

    turns = []           # one dict per LOGICAL turn (one message.id), in order
    cur = None           # the open turn, accumulating its split content-block records

    def flush():
        nonlocal cur
        if cur is not None:
            turns.append(cur)
            cur = None

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("isSidechain"):
            continue     # subagent/workflow turns live in their own track; not the session
        t = r.get("type")
        if t == "assistant":
            msg = r.get("message", {}) or {}
            u = msg.get("usage", {}) or {}
            mid = msg.get("id") or r.get("requestId")
            n_tool, names = 0, []
            for b in (msg.get("content") or []):
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    n_tool += 1
                    names.append(b.get("name", "?"))
            out = u.get("output_tokens", 0) or 0
            cr = u.get("cache_read_input_tokens", 0) or 0
            cc = u.get("cache_creation_input_tokens", 0) or 0
            inp = u.get("input_tokens", 0) or 0
            if cur is not None and mid is not None and cur["mid"] == mid:
                # another content-block record of the SAME response: merge tool_use blocks,
                # keep usage once (max defends against a partial-stream split with 0 usage).
                cur["n_tool"] += n_tool
                cur["names"].extend(names)
                cur["out"] = max(cur["out"], out)
                cur["cache_read"] = max(cur["cache_read"], cr)
                cur["cache_create"] = max(cur["cache_create"], cc)
                cur["in"] = max(cur["in"], inp)
            else:
                flush()
                cur = {"mid": mid, "out": out, "in": inp, "cache_read": cr,
                       "cache_create": cc, "n_tool": n_tool, "names": list(names),
                       "result_chars": 0}
        elif t == "user":
            msg = r.get("message", {}) or {}
            content = msg.get("content")
            if isinstance(content, list):
                chars = 0
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        chars += _txt_len(b.get("content", ""))
                if chars:
                    # tool_result follows its tool_use response: close it and attribute R
                    flush()
                    if turns and turns[-1]["n_tool"] > 0:
                        turns[-1]["result_chars"] += chars
    flush()
    if not turns:
        return None

    # prefix P: the preamble cached before the conversation produced its first answer.
    first = turns[0]
    prefix = first["cache_read"] or first["cache_create"] or first["in"]
    tool_turns = [t for t in turns if t["n_tool"] > 0]
    n_ro = sum(1 for t in turns for nm in t["names"] if nm in READ_ONLY_TOOLS)
    n_tool_calls = sum(t["n_tool"] for t in turns)

    # concrete replay track: per-turn shape the benchmark plays back
    track = []
    for t in turns:
        track.append({
            "decode": t["out"],
            "tool": t["n_tool"] > 0,
            "result": int(math.ceil(t["result_chars"] / CHARS_PER_TOK)),
            "name": (t["names"][0] if t["names"] else None),
        })

    return {
        "session": os.path.splitext(os.path.basename(path))[0],
        "ns": os.path.basename(os.path.dirname(path)),
        "prefix_tokens": prefix,
        "n_turns": len(turns),
        "n_tool_turns": len(tool_turns),
        "n_tool_calls": n_tool_calls,
        "tool_call_fraction": len(tool_turns) / len(turns),
        "read_only_fraction": (n_ro / n_tool_calls) if n_tool_calls else None,
        "decode_per_turn": [t["out"] for t in turns],
        "result_per_tool_turn": [int(math.ceil(t["result_chars"] / CHARS_PER_TOK))
                                 for t in tool_turns],
        "tools_per_tool_turn": [t["n_tool"] for t in tool_turns],
        "tool_names": [nm for t in turns for nm in t["names"]],
        # per-turn cache-hit fraction (KV reuse the harness already captures)
        "cache_hit_per_turn": [
            (t["cache_read"] / max(t["cache_read"] + t["cache_create"] + t["in"], 1))
            for t in turns if (t["cache_read"] + t["cache_create"] + t["in"]) > 0
        ],
        "track": track,
    }


def build_profile(sessions, replay_tracks=4):
    app_ver = fleet_version.app_version()
    S = [s for s in sessions if s]
    # Keep sessions that are meaningful agent trajectories: >=3 turns AND a real shared
    # preamble. A prefix of ~0 means the first record carried no usage (a resumed/compacted
    # session that began mid-stream) — useless for a prefix-REUSE benchmark, so drop it.
    S = [s for s in S if s["n_turns"] >= 3 and s["prefix_tokens"] >= 256]
    if not S:
        raise SystemExit("no usable sessions (need >=3 turns and a >=256-tok shared prefix)")

    prefix = [s["prefix_tokens"] for s in S]
    turns = [s["n_turns"] for s in S]
    decode = [d for s in S for d in s["decode_per_turn"]]
    result = [r for s in S for r in s["result_per_tool_turn"]]
    tools_per = [n for s in S for n in s["tools_per_tool_turn"]]
    cache_hit = [c for s in S for c in s["cache_hit_per_turn"]]
    tool_frac = [s["tool_call_fraction"] for s in S]
    ro_frac = [s["read_only_fraction"] for s in S if s["read_only_fraction"] is not None]

    tool_mix = collections.Counter()
    for s in S:
        tool_mix.update(s["tool_names"])

    total_turns = sum(s["n_turns"] for s in S)
    total_tool_turns = sum(s["n_tool_turns"] for s in S)
    total_tool_calls = sum(s["n_tool_calls"] for s in S)

    # replay tracks: representative sessions across the turn-count distribution
    by_turns = sorted(S, key=lambda s: s["n_turns"])
    picks = []
    if by_turns:
        for p in (50, 75, 90, 99):
            idx = max(0, min(len(by_turns) - 1, int(round((p / 100) * (len(by_turns) - 1)))))
            picks.append((p, by_turns[idx]))
    seen, replay = set(), []
    for p, s in picks:
        if s["session"] in seen:
            continue
        seen.add(s["session"])
        replay.append({
            "version": app_ver,
            "session": s["session"][:8],
            "percentile_by_turns": p,
            "prefix_tokens": s["prefix_tokens"],
            "n_turns": s["n_turns"],
            "tool_call_fraction": round(s["tool_call_fraction"], 3),
            "track": fleet_version.versioned_rows(s["track"], app_ver),
        })
        if len(replay) >= replay_tracks:
            break

    return {
        "schema": SCHEMA,
        "app_version": app_ver,
        "source": {
            "n_sessions": len(S),
            "n_turns": total_turns,
            "namespaces": sorted({s["ns"] for s in S}),
            "generated_by": "tools/transcript_workload.py",
            "chars_per_tok_estimate": CHARS_PER_TOK,
        },
        # headline knobs the benchmark + tuning sweep consume
        "tool_call_fraction": round(total_tool_turns / total_turns, 4),
        "tool_calls_per_turn": round(total_tool_calls / total_turns, 4),
        "read_only_fraction": round(statistics.mean(ro_frac), 4) if ro_frac else None,
        # distributions (P, D, R, T, ...)
        "prefix_tokens": _stats(prefix),
        "decode_tokens_per_turn": _stats(decode),
        "result_tokens_per_tool_turn": _stats(result),
        "turns_per_session": _stats(turns),
        "tools_per_tool_turn": _stats(tools_per),
        "tool_call_fraction_per_session": _stats(tool_frac, ints=False),
        "cache_hit_fraction_per_turn": _stats(cache_hit, ints=False),
        "tool_mix": dict(tool_mix.most_common()),
        "tool_mix_read_only": {k: (k in READ_ONLY_TOOLS) for k in tool_mix},
        # concrete tracks for playback
        "replay": replay,
    }


def summarize(prof):
    L = []
    src = prof["source"]
    L.append(f"# Realistic workload profile ({prof['schema']})\n")
    L.append(f"- sessions: {src['n_sessions']}  ·  turns: {src['n_turns']:,}  "
             f"·  namespaces: {', '.join(src['namespaces'])}")
    L.append(f"- **tool-call fraction (turns with >=1 tool_use): {prof['tool_call_fraction']:.1%}**  "
             f"·  tool calls/turn: {prof['tool_calls_per_turn']}  "
             f"·  read-only: {prof['read_only_fraction']:.1%}")
    def row(label, d, unit="tok"):
        if d.get("n"):
            L.append(f"- {label}: median {d['median']:,} {unit}  "
                     f"(p10 {d['p10']:,} · mean {d['mean']:,} · p90 {d['p90']:,} · max {d['max']:,})")
    row("prefix P (shared preamble)", prof["prefix_tokens"])
    row("decode D /turn (EXACT output_tokens)", prof["decode_tokens_per_turn"])
    row("result R /tool-turn (ESTIMATE chars/4)", prof["result_tokens_per_tool_turn"])
    row("turns T /session", prof["turns_per_session"], unit="turns")
    ch = prof["cache_hit_fraction_per_turn"]
    if ch.get("n"):
        L.append(f"- cache-hit /turn (KV reuse): median {ch['median']}  (p10 {ch['p10']} · p90 {ch['p90']})")
    L.append("\n## Tool mix (top 15)\n")
    L.append("| Tool | Calls | Read-only |")
    L.append("|---|---:|:--:|")
    for name, n in list(prof["tool_mix"].items())[:15]:
        L.append(f"| {name} | {n:,} | {'✓' if prof['tool_mix_read_only'].get(name) else ''} |")
    L.append("\n## Replay tracks\n")
    for t in prof["replay"]:
        tools = sum(1 for x in t["track"] if x["tool"])
        L.append(f"- `{t['session']}` (p{t['percentile_by_turns']} by turns): "
                 f"{t['n_turns']} turns, prefix {t['prefix_tokens']:,} tok, "
                 f"tool-frac {t['tool_call_fraction']}, {tools} tool turns")
    return "\n".join(L)


def cmd_profile(a):
    roots = a.root or DEFAULT_ROOTS
    files = discover(roots, a.since_days, "" if a.all else a.ns_prefix)
    if a.max:
        files = files[:a.max]
    print(f"profiling {len(files)} transcripts ...", file=sys.stderr)
    sessions = []
    for f in files:
        s = analyze_turns(f["path"])
        if s:
            sessions.append(s)
    prof = build_profile(sessions, replay_tracks=a.replay_tracks)
    md = summarize(prof)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(prof, open(a.out, "w", encoding="utf-8"), indent=2)
        print(f"wrote {a.out}", file=sys.stderr)
    if a.md:
        os.makedirs(os.path.dirname(a.md) or ".", exist_ok=True)
        open(a.md, "w", encoding="utf-8").write(md + "\n")
        print(f"wrote {a.md}", file=sys.stderr)
    print(md)


def cmd_show(a):
    prof = json.load(open(a.profile, encoding="utf-8"))
    print(summarize(prof))


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("profile")
    q.add_argument("--root", action="append")
    q.add_argument("--since-days", type=float, default=None)
    q.add_argument("--ns-prefix", default=NS_INCLUDE_PREFIX)
    q.add_argument("--all", action="store_true", help="all namespaces (no prefix filter)")
    q.add_argument("--max", type=int, default=None)
    q.add_argument("--replay-tracks", type=int, default=4)
    q.add_argument("--out", default=None)
    q.add_argument("--md", default=None)
    s = sub.add_parser("show")
    s.add_argument("profile")
    a = p.parse_args()
    {"profile": cmd_profile, "show": cmd_show}[a.cmd](a)


if __name__ == "__main__":
    main()
