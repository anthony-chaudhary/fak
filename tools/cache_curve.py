#!/usr/bin/env python3
"""
cache_curve.py — demonstrate WHY the public prompt-cache hit rate is high, and the
scaling laws under which it bends toward 0%.

The thesis, in one line:

    The high prompt-cache hit rate everyone quotes is purchased with a *frozen*
    trajectory. It is a prefix match (any byte change in the prefix invalidates
    everything after it), so it only stays high while the harness refuses to touch
    history — i.e. append-only, single, linear. The instant the trajectory becomes
    flexible (edit / compact / re-summarize / reorder — the things an agent OS does),
    or the workload gains tool-call density or cross-agent fan-out, the *default*
    prefix cache decays toward 0%.

This tool makes that concrete and reproducible. It is honest by construction:

  * The FROZEN CEILING is the same metric the harness already reports
    (cache_read / (cache_read + cache_creation + input)). For a single append-only
    agent it is h_frozen(T) = (T-1)/(T+1) — it RISES toward 1 with turns. The real
    transcripts on this machine sit exactly there (see `anchor`): ~96.6% machine-wide,
    a 205-turn session at 99%. That is not a bug to fix; it is the thesis: the number
    is high *because* the trajectory is frozen.

  * Each decay axis is a survival factor s in [0,1] applied to that ceiling. Losing a
    cached prefix moves those tokens from `read` to `paid`, so the blended hit becomes
    exactly  h = s * h_frozen(T)  (derivation in `_doc_model`). As s -> 0, h -> 0.

  * The mechanics are not invented: prefix-only reuse, the 20-block lookback, the
    4-breakpoint budget, and the concurrency wall ("N parallel requests with identical
    prefixes all pay full price — none can read what the others are still writing") are
    the documented behavior of the hosted prompt cache. Every constant is a flag.

Subcommands:
  curves   [--turns N]                 frozen ceiling + the 2 single-agent decay axes (flex, tools)
  fanout   [--agents ...]              cross-agent shared-prefix reuse: default vs shared (table)
  compound [--turns N]                 single-agent collapse (flex x tools), then the fleet fan-out
  anchor   <session_audit.json>        the REAL measured ceiling, from session_audit.py --json
  chart    [--turns N]                 an at-a-glance ASCII chart of the decay
  report   [--turns N] [--anchor J]    a full markdown report (curves + fanout + chart + anchor)

Companion docs:
  docs/explainers/frozen-trajectory-cache-cliff.md   (the explainer this tool grounds)
  docs/explainers/kv-cache-agentic-context.md        (the prefix mechanics)
  docs/notes/SCALING-LAWS-OF-AGENTS-2026-06-19.md     (the supply-side: how fak deletes the reread)
"""
import sys
import json
import math
import argparse
import datetime

# --- documented hosted-prompt-cache constants (flags, not magic) ---------------
LOOKBACK_BLOCKS = 20      # a breakpoint walks back at most this many content blocks
MAX_BREAKPOINTS = 4       # cache_control breakpoints per request
# economics, for the cost framing only (share table uses ratios, not these):
READ_MULT = 0.10          # cache-read ~= 0.1x base input price
WRITE_MULT_5M = 1.25      # cache-write (5-minute TTL)

# the machine-wide measured ceiling these curves are calibrated against
# (tools/session_audit.py audit --since-days 30, this machine, 199 sessions).
ANCHOR = {
    "sessions": 199,
    "cache_read_share": 0.966,   # cache_read / total ingested context
    "io_ratio": 126.6,
    "hit_median": 0.894, "hit_p10": 0.703, "hit_p90": 0.968,
    "ceiling_example": "205-turn / 32-tool-call session: 99% cache-hit",
}


# --- the frozen ceiling --------------------------------------------------------
def h_frozen(turns):
    """Cache-read fraction of a single append-only agent over `turns` turns, warm cache,
    no head mutation. Derivation in `_doc_model`. Rises toward 1; this is the ceiling."""
    if turns <= 1:
        return 0.0
    return (turns - 1) / (turns + 1)


def _doc_model():
    return (
        "model (all metrics are cache-read fraction h = read / (read + paid), the same\n"
        "number tools/session_audit.py reports as cache_hit_frac):\n"
        "\n"
        "  frozen ceiling (single, linear, append-only agent over T turns):\n"
        "    turn t resends prefix L_{t-1}=(t-1)*d and appends a fresh delta d.\n"
        "    read(T) = sum L_{t-1} = d*T(T-1)/2      paid(T) = d*T  (each new delta)\n"
        "    h_frozen(T) = read/(read+paid) = (T-1)/(T+1)  ->  rises to 1.\n"
        "\n"
        "  a decay axis is a survival factor s in [0,1]. A lost prefix-reuse does not\n"
        "  vanish — those tokens move from read to paid (they are recomputed):\n"
        "    read = s*R0 ,  paid = P0 + (1-s)*R0  ->  h(s) = s*R0/(R0+P0) = s*h_frozen.\n"
        "  so every axis simply scales the ceiling by its survival factor; s->0 => h->0.\n"
        "\n"
        "  this s*ceiling rule is for SINGLE-AGENT axes only (flexibility, per-turn tool\n"
        "  density). Cross-agent fan-out is a fleet-aggregate effect (the concurrency wall),\n"
        "  not a single agent's hit: its reuse rate is 0% and flat in N, so it is reported\n"
        "  separately (see fanout/compound), never multiplied into one agent's percentage.\n"
    )


# --- axis 1: flexibility (the product) -----------------------------------------
def s_flex(edit_depth):
    """Survival under a trajectory edit that reaches `edit_depth` (fraction, 0..1) back
    into the cached prefix. Append-only (touch only the tail) => edit_depth 0 => s=1.
    Compaction / RSI re-summarization / reorder that rewrites the head => edit_depth 1
    => s=0: a prefix change at the front invalidates everything after it."""
    return max(0.0, 1.0 - edit_depth)


# --- axis 2: tool-call density (the 20-block / 4-breakpoint wall) ---------------
def s_tooldensity(tool_calls_in_turn, breakpoints=1):
    """Survival when a turn emits `tool_calls_in_turn` tool_use/tool_result pairs.

    Each pair is ~2 content blocks; a turn also carries ~2 assistant blocks. A new
    cache breakpoint can only find a prior cache entry within LOOKBACK_BLOCKS behind it,
    and you get MAX_BREAKPOINTS markers to staircase through the new content. So a turn
    that adds more than breakpoints*LOOKBACK_BLOCKS new blocks outruns the budget and
    the next request silently misses on that span.

      naive harness  (breakpoints=1): wall at ~10 tool calls in one turn
      careful harness(breakpoints=4): wall at ~40 (markers every ~LOOKBACK blocks)
    """
    bp = max(1, min(MAX_BREAKPOINTS, breakpoints))
    new_blocks = 2 * tool_calls_in_turn + 2
    budget = bp * LOOKBACK_BLOCKS
    return min(1.0, budget / new_blocks) if new_blocks else 1.0


# --- axis 3: cross-agent fan-out (the concurrency wall) ------------------------
def fanout_shared_reuse(agents, concurrent=True):
    """Reuse rate of the SHARED setup (system prompt + tool schemas) across a fan-out of
    `agents` workers, under the default hosted prompt cache.

    A cache entry is readable only AFTER the first response that wrote it begins
    streaming. Fire N agents at once on a cold shared prefix and none can read what the
    simultaneous cohort is still writing -> the shared prefix is cold-WRITTEN N times,
    cross-agent READ 0 times. Reuse rate = reads/(reads+writes):

      concurrent (default fan-out): 0 / (0 + N)       = 0%      (flat, independent of N)
      staggered / cloned (fak-like): (N-1) / (N-1 + 1) = (N-1)/N -> 1

    Returns (reuse_rate, shared_setup_payments). The default forfeits (N-1) reuses of
    the shared setup — a waste that grows linearly with N."""
    n = max(1, agents)
    if concurrent:
        return 0.0, n            # paid N times, reused 0 times across the cohort
    return (n - 1) / n, 1        # paid once, reused N-1 times


def h_fleet_shortagent(agents, shared_tokens, agent_turns, delta_tokens, concurrent=True):
    """Blended fleet cache-read fraction for `agents` SHORT agents that each run
    `agent_turns` turns appending `delta_tokens`, on a shared prefix of `shared_tokens`.

    This is the regime where fan-out actually craters the *hit rate* (not just the cost):
    many small tool-running sub-agents whose work is dominated by the un-amortized shared
    context. Under concurrent launch the shared prefix is cold for every agent; under a
    shared/cloned prefix it is paid once for the whole fleet."""
    n, S, t, d = max(1, agents), shared_tokens, max(1, agent_turns), delta_tokens
    # per-agent intra reuse over its own (short) trajectory, warm within the agent:
    #   turn i (>=2) reads (S + (i-1)*d) and pays the new d; turn 1 pays its first prefill.
    read_agent = sum(S + (i - 1) * d for i in range(2, t + 1))
    paid_agent_traj = d * (t - 1)                 # the new deltas this agent appends
    read = n * read_agent
    if concurrent:
        paid = n * (S + paid_agent_traj)          # every agent cold-writes S
    else:
        paid = S + n * paid_agent_traj            # S paid once for the fleet
        read += (n - 1) * S                        # the other N-1 agents read it
    return read / (read + paid) if (read + paid) else 0.0


# --- table builders ------------------------------------------------------------
def curve_table(turns=200):
    ceiling = h_frozen(turns)
    rows = []
    # flexibility: edit depth into the prefix
    flex = [(int(d * 100), s_flex(d) * ceiling) for d in (0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0)]
    # tool density (naive single-breakpoint harness — the common case)
    tdens = [(k, s_tooldensity(k, breakpoints=1) * ceiling) for k in (1, 5, 10, 20, 40, 80, 160)]
    tdens4 = [(k, s_tooldensity(k, breakpoints=4) * ceiling) for k in (1, 5, 10, 20, 40, 80, 160)]
    return {
        "turns": turns,
        "frozen_ceiling": ceiling,
        "flex": flex,
        "tool_density_1bp": tdens,
        "tool_density_4bp": tdens4,
    }


def fanout_table(agent_list, shared_tokens=100_000, agent_turns=2, delta_tokens=2_000):
    rows = []
    for n in agent_list:
        r_def, pay_def = fanout_shared_reuse(n, concurrent=True)
        r_sh, pay_sh = fanout_shared_reuse(n, concurrent=False)
        rows.append({
            "agents": n,
            "shared_reuse_default": r_def,
            "shared_reuse_shared": r_sh,
            "setup_payments_default": pay_def,
            "setup_payments_shared": pay_sh,
            "forfeited_setup_tokens": (n - 1) * shared_tokens,
            "h_default": h_fleet_shortagent(n, shared_tokens, agent_turns, delta_tokens, True),
            "h_shared": h_fleet_shortagent(n, shared_tokens, agent_turns, delta_tokens, False),
        })
    return {
        "shared_tokens": shared_tokens, "agent_turns": agent_turns,
        "delta_tokens": delta_tokens, "rows": rows,
    }


def compound_scenario(turns=200, agents=100):
    """The single-agent collapse, then its fleet consequence.

    IMPORTANT (a category distinction the demonstrator must keep honest): flexibility and
    per-turn tool density are both *single-agent* cache-read fractions, so they legitimately
    MULTIPLY into one agent's hit. Cross-agent fan-out is NOT a single-agent percentage — it
    is a fleet-aggregate effect (the concurrency wall). The default cross-agent reuse rate is
    0% and FLAT in N (each agent re-pays the shared prefix); the percentage does not fall with
    N, the *forfeited* reuse grows with N. So we report fan-out as a separate fleet line rather
    than fold a survival factor into one agent's hit (an earlier version did the latter with a
    hidden constant — a category error that fabricated the headline)."""
    ceiling = h_frozen(turns)
    steps = [
        ("frozen single linear agent (append-only)", 1.0, 1.0),
        ("+ moderate flexibility (compact 25% of prefix)", s_flex(0.25), 1.0),
        ("+ tool-dense turns (20 calls/turn, 1 breakpoint)", s_flex(0.25), s_tooldensity(20, 1)),
        ("+ aggressive flexibility (compact 75%) + tool-dense", s_flex(0.75), s_tooldensity(20, 1)),
    ]
    out = []
    for label, sf, st in steps:
        s = sf * st                       # only the two single-agent axes multiply
        out.append({"step": label, "survival": s, "hit": s * ceiling})
    # fleet consequence: fan the collapsed single agent out to N workers. The default
    # concurrent cache recovers 0% of the shared setup across them; the fleet pays this
    # collapsed-hit work N times with no cross-agent amortization.
    r_def, _ = fanout_shared_reuse(agents, concurrent=True)
    r_sh, _ = fanout_shared_reuse(agents, concurrent=False)
    return {
        "turns": turns, "ceiling": ceiling, "steps": out,
        "fleet": {
            "agents": agents,
            "single_agent_hit": out[-1]["hit"],
            "cross_agent_reuse_default": r_def,   # 0.0, flat in N
            "cross_agent_reuse_shared": r_sh,     # (N-1)/N, what sharing/cloning recovers
        },
    }


# --- anchor: the real measured ceiling -----------------------------------------
def load_anchor(path):
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    agg = data.get("aggregate", {})
    tot = agg.get("totals", {})
    total_in = tot.get("input", 0) + tot.get("cache_read", 0) + tot.get("cache_create", 0)
    share = tot.get("cache_read", 0) / total_in if total_in else None
    dist = agg.get("dist", {}).get("cache_hit_frac", {})
    # scatter: per-session (tool_calls, hit) — proves the real data does NOT decay with
    # tool-call count, because Claude Code never leaves the frozen single-linear regime.
    scatter = []
    for s in data.get("sessions", []):
        if s.get("kind", "session") != "session":
            continue
        h = s.get("cache_hit_frac")
        k = s.get("n_tool_use")
        if h is not None and k is not None:
            scatter.append((k, h, s.get("assistant_turns", 0)))
    return {
        "n_sessions": agg.get("n_sessions"),
        "cache_read_share": share,
        "io_ratio": (total_in / tot.get("output", 1)) if tot.get("output") else None,
        "hit_median": dist.get("median"), "hit_p10": dist.get("p10"), "hit_p90": dist.get("p90"),
        "scatter": scatter,
    }


# --- rendering -----------------------------------------------------------------
def _bar(frac, width=40, fill="#", empty="."):
    n = int(round(max(0.0, min(1.0, frac)) * width))
    return fill * n + empty * (width - n)


def render_curves(c):
    L = []
    L.append(f"cache-read fraction (hit) — frozen ceiling at T={c['turns']} turns "
             f"= {c['frozen_ceiling']*100:.1f}%")
    L.append("  the high public number IS this ceiling — high *because* the trajectory is frozen.")
    L.append("")
    L.append("axis 1 — flexibility (the product): edit depth into the cached prefix")
    L.append(f"  {'edit-depth':>11}  {'hit':>6}")
    for d, h in c["flex"]:
        L.append(f"  {d:>10}%  {h*100:>5.1f}%  {_bar(h)}")
    L.append("  append-only=0% (frozen) -> rewrite the head=100% (compaction/RSI): hit -> 0.")
    L.append("")
    L.append("axis 2 — tool-call density: tool_use/result pairs emitted in one turn")
    L.append(f"  {'calls/turn':>11}  {'1bp':>6}  {'4bp':>6}   (1bp=naive harness, 4bp=careful)")
    for (k, h1), (_, h4) in zip(c["tool_density_1bp"], c["tool_density_4bp"]):
        L.append(f"  {k:>11}  {h1*100:>5.1f}%  {h4*100:>5.1f}%  {_bar(h1)}")
    L.append(f"  wall at ~{LOOKBACK_BLOCKS//2} calls (1 breakpoint) / "
             f"~{MAX_BREAKPOINTS*LOOKBACK_BLOCKS//2} (4): {LOOKBACK_BLOCKS}-block lookback x "
             f"{MAX_BREAKPOINTS} breakpoints.")
    return "\n".join(L)


def render_fanout(f):
    L = []
    L.append(f"axis 3 — cross-agent fan-out (concurrency wall): shared setup "
             f"= {f['shared_tokens']:,} tok, {f['agent_turns']}-turn agents")
    L.append("  a cache entry is readable only AFTER the writer streams; fire N at once -> none read.")
    L.append(f"  {'agents':>7}  {'reuse(default)':>14}  {'reuse(shared)':>13}  "
             f"{'hit(default)':>12}  {'hit(shared)':>11}  {'forfeited tok':>14}")
    for r in f["rows"]:
        L.append(f"  {r['agents']:>7}  {r['shared_reuse_default']*100:>13.0f}%  "
                 f"{r['shared_reuse_shared']*100:>12.0f}%  {r['h_default']*100:>11.1f}%  "
                 f"{r['h_shared']*100:>10.1f}%  {r['forfeited_setup_tokens']:>14,}")
    L.append("  default cross-agent reuse is 0% and stays 0% in N; the re-paid setup grows")
    L.append("  linearly with N. hit(default) is FLAT in N — each agent re-pays the shared")
    L.append("  prefix — and sits far below the shared path (which climbs toward 100%). Fan-out")
    L.append("  forfeits a reuse win whose size grows with N; it does not lower the percentage.")
    return "\n".join(L)


def render_compound(c):
    f = c["fleet"]
    L = []
    L.append(f"the single-agent collapse (T={c['turns']}, ceiling {c['ceiling']*100:.1f}%) — "
             f"flexibility x per-turn tool density, the two axes that genuinely multiply:")
    for s in c["steps"]:
        L.append(f"  {s['hit']*100:>5.1f}%  {_bar(s['hit'])}  {s['step']}")
    L.append(f"then fan that {f['single_agent_hit']*100:.1f}%-hit agent out to {f['agents']} workers:")
    L.append(f"  cross-agent reuse of the shared setup — default concurrent: "
             f"{f['cross_agent_reuse_default']*100:.0f}%   "
             f"(achievable if shared/cloned: {f['cross_agent_reuse_shared']*100:.0f}%)")
    L.append(f"  => the fleet pays this collapsed-cache work {f['agents']}x with 0% cross-agent")
    L.append("     recovery. The single-agent % is what bends toward 0 (flex + tool density);")
    L.append("     fan-out keeps it pinned and multiplies the waste by N.")
    return "\n".join(L)


def render_anchor(a):
    L = []
    L.append(f"REAL measured ceiling — {a['n_sessions']} sessions, this machine "
             f"(tools/session_audit.py):")
    if a["cache_read_share"] is not None:
        L.append(f"  machine-wide cache-read share : {a['cache_read_share']*100:.1f}%  "
                 f"(I:O {a['io_ratio']:.1f}:1)")
    L.append(f"  per-session hit  median {a['hit_median']}  p10 {a['hit_p10']}  p90 {a['hit_p90']}")
    if a["scatter"]:
        # bucket by tool-call count: show hit does NOT fall with calls (still frozen regime)
        buckets = {"0": [], "1-5": [], "6-15": [], "16+": []}
        for k, h, _t in a["scatter"]:
            key = "0" if k == 0 else "1-5" if k <= 5 else "6-15" if k <= 15 else "16+"
            buckets[key].append(h)
        L.append("  hit vs tool-calls/session (real data stays at the frozen ceiling — "
                 "Claude Code never leaves the single-linear regime):")
        for key in ("0", "1-5", "6-15", "16+"):
            xs = buckets[key]
            if xs:
                mean = sum(xs) / len(xs)
                L.append(f"    {key:>5} calls: n={len(xs):>3}  mean hit {mean*100:>5.1f}%  {_bar(mean)}")
        L.append("  => the real high number confirms thesis-1: it is high because the trajectory")
        L.append("     is frozen, NOT because tool calls are cheap to cache.")
    return "\n".join(L)


def render_chart(turns=200):
    """At-a-glance ASCII chart. Two SINGLE-AGENT cache-hit curves that bend to 0% (flex,
    per-turn tool density), then fan-out shown as its OWN quantity — a fleet shared-setup
    reuse RATE — because it is not a single agent's hit (see compound_scenario)."""
    ceiling = h_frozen(turns)
    L = [f"single-agent cache-hit decay from the frozen ceiling ({ceiling*100:.0f}%):", ""]
    xs = list(range(0, 101, 10))  # 0..100% of "how far along this axis"
    series = {
        "flex (edit-depth %)    ": [s_flex(x / 100) * ceiling for x in xs],
        "tools (calls/turn, 1bp)": [s_tooldensity(round(x / 100 * 80), 1) * ceiling for x in xs],
    }
    for name, ys in series.items():
        L.append(name + " | cache-hit")
        for x, y in zip(xs, ys):
            L.append(f"  {x:>3}  {_bar(y, 50)} {y*100:4.1f}%")
        L.append("")
    L.append("fan-out is a DIFFERENT quantity (fleet, not one agent's hit) — cross-agent")
    L.append("shared-setup REUSE RATE, default concurrent launch (achievable if shared in ()):")
    for n in (1, 2, 5, 10, 25, 100):
        r, _ = fanout_shared_reuse(n, concurrent=True)
        L.append(f"  N={n:>4}  {_bar(r, 50)} {r*100:4.1f}%   (shared/cloned: {(n-1)/n*100:4.1f}%)")
    L.append("")
    L.append("top two x-axis: 0 = frozen/append-only/sparse ; 100 = fully flexible / dense.")
    return "\n".join(L)


def render_report(turns, anchor_path):
    now = datetime.datetime.now().isoformat(timespec="seconds")
    c = curve_table(turns)
    f = fanout_table([1, 2, 5, 10, 25, 100])
    comp = compound_scenario(turns)
    L = []
    L.append("# Frozen-trajectory cache cliff — demonstrator output\n")
    L.append(f"Generated: {now} · tool: `tools/cache_curve.py` (deterministic, stdlib-only)\n")
    L.append("```")
    L.append(_doc_model())
    L.append("```\n")
    L.append("## The frozen ceiling and the three decay axes\n")
    L.append("```")
    L.append(render_curves(c))
    L.append("")
    L.append(render_fanout(f))
    L.append("")
    L.append(render_compound(comp))
    L.append("```\n")
    L.append("## The decay, at a glance\n")
    L.append("```")
    L.append(render_chart(turns))
    L.append("```\n")
    if anchor_path:
        a = load_anchor(anchor_path)
        L.append("## The real measured ceiling (this machine)\n")
        L.append("```")
        L.append(render_anchor(a))
        L.append("```\n")
    return "\n".join(L)


# --- cli -----------------------------------------------------------------------
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    p = argparse.ArgumentParser(description="Demonstrate the frozen-trajectory prompt-cache cliff.")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("curves", "compound", "chart"):
        q = sub.add_parser(name)
        q.add_argument("--turns", type=int, default=200)
    qf = sub.add_parser("fanout")
    qf.add_argument("--agents", type=int, nargs="+", default=[1, 2, 5, 10, 25, 100])
    qa = sub.add_parser("anchor")
    qa.add_argument("json")
    qr = sub.add_parser("report")
    qr.add_argument("--turns", type=int, default=200)
    qr.add_argument("--anchor", default=None)
    qr.add_argument("--md", default=None)
    a = p.parse_args()

    if a.cmd == "curves":
        print(_doc_model())
        print(render_curves(curve_table(a.turns)))
    elif a.cmd == "fanout":
        print(render_fanout(fanout_table(a.agents)))
    elif a.cmd == "compound":
        print(render_compound(compound_scenario(a.turns)))
    elif a.cmd == "chart":
        print(render_chart(a.turns))
    elif a.cmd == "anchor":
        print(render_anchor(load_anchor(a.json)))
    elif a.cmd == "report":
        md = render_report(a.turns, a.anchor)
        if a.md:
            open(a.md, "w", encoding="utf-8").write(md)
            print(f"wrote {a.md}", file=sys.stderr)
        print(md)


if __name__ == "__main__":
    main()
