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
  curves    [--turns N]                frozen ceiling + the 2 single-agent decay axes (flex, tools)
  fanout    [--agents ...]             cross-agent shared-prefix reuse: default vs shared (table)
  compound  [--turns N]                single-agent collapse (flex x tools), then the fleet fan-out
  inversion [--turns N] [--delta D]    the VANITY inversion: hit rate vs the per-turn bill it hides,
            [--baton B]                plus the cut-to-a-fresh-leg break-even (a couple of turns)
  anchor   <session_audit.json>        the REAL measured ceiling, from session_audit.py --json
  validate <measured_decay.json>       compare modeled survival factors to measured anchors
  chart    [--turns N]                 an at-a-glance ASCII chart of the decay
  costcurve [--turns N] [--delta D]    the COST CURVE SVG: billed tokens/turn over a session,
            [--out PATH]               naive re-send vs fak prefix-preserving (cache held).
                                       Generated from this tool's own model fns + measured ANCHOR,
                                       stdlib-only, deterministic. Default out:
                                       docs/adoption/diagrams/cost-curve.svg
  report   [--turns N] [--anchor J]    a full markdown report (curves + fanout + chart + anchors)

Companion docs:
  docs/explainers/frozen-trajectory-cache-cliff.md   (the explainer this tool grounds)
  docs/explainers/kv-cache-agentic-context.md        (the prefix mechanics)
  docs/notes/SCALING-LAWS-OF-AGENTS-2026-06-19.md     (the supply-side: how fak deletes the reread)
"""
import sys
import json
import argparse
import datetime
from pathlib import Path

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


# --- the cost inversion: the vanity metric vs the bill it hides ----------------
# The frozen ceiling above is a *hit RATE*. This block prices the SAME frozen,
# append-only trajectory in base-input-price units, so the rate can be put next to
# the per-turn BILL it conceals. The one fact that makes cache-hit a vanity metric:
# the rate and the bill rise together, and in the wrong direction. The reported hit
# rises toward 1 *because* the cached prefix each turn grows — and that same growing
# prefix makes every turn cost strictly more (a cache read is cheap, not free). A
# number that improves precisely because the thing you pay for got bigger is not
# measuring efficiency; it is measuring length. See
# docs/notes/CACHE-HIT-VANITY-METRIC-SELF-FULFILLING-2026-07-01.md.
def turn_cost(t, delta, read_mult=READ_MULT):
    """Billed cost of turn t of a frozen append-only agent, in base-input-price units.

    Turn t re-sends a prefix of (t-1)*delta cached tokens (billed at read_mult) plus a
    fresh delta of `delta` new tokens (billed at 1x): c(t) = delta*(1 + read_mult*(t-1)).
    It RISES linearly with t, without bound, even as the hit rate rises toward 1."""
    if t < 1:
        return 0.0
    return delta * (1.0 + read_mult * (t - 1))


def naive_turn_cost(t, delta):
    """Billed cost of turn t with NO usable cache (the prefix match is broken -- a
    summarize that rewrote the body, or a cold provider cache): the whole prefix
    (t-1)*delta plus the fresh delta, re-sent at full (1x) base price. Grows linearly
    with t at slope `delta` -- steeper than the cache-held line, which is the whole
    point of the cost curve."""
    if t < 1:
        return 0.0
    return delta * t


def cum_cost(turns, delta, read_mult=READ_MULT):
    """Cumulative billed cost over `turns` turns = sum of turn_cost. Grows ~quadratically
    in T (delta*T + read_mult*delta*T(T-1)/2), so the running bill AND the 'saved-token'
    headline that tracks the read both scale with T^2 — celebrating the saving is
    celebrating length (docs/notes/SESSION-CACHE-SAVINGS-ABLATION-2026-06-29.md)."""
    return sum(turn_cost(t, delta, read_mult) for t in range(1, max(0, turns) + 1))


def saved_headline(turns, delta, read_mult=READ_MULT):
    """The 'saved-token-equiv' headline for the same run: the read rebate (1-read_mult)*read,
    with read(T)=delta*T(T-1)/2. Quadratic in T, so the number that gets quoted grows with
    trajectory length by construction — the longer the session, the bigger the 'saving'."""
    read = delta * turns * (turns - 1) / 2 if turns > 1 else 0.0
    return (1.0 - read_mult) * read


def cut_break_even_turns(prefix_now, baton_prefix, read_mult=READ_MULT, write_mult=WRITE_MULT_5M):
    """Turns after which cutting to a fresh, flat-prefix leg (a relay leg: system + an O(1)
    baton of size `baton_prefix`) has already repaid its one-time cold-prefix write.

    Cutting costs write_mult*baton_prefix ONCE (the cold prefill of the small new prefix), then
    buys back read_mult*(prefix_now - baton_prefix) every subsequent turn (the monolith re-reads
    its whole grown prefix; the fresh leg re-reads only the small baton). So it pays for itself
    after  n* = write_mult*baton_prefix / (read_mult*(prefix_now - baton_prefix))  turns.
    Returns +inf when the fresh prefix is not smaller (nothing to buy back). This is the number
    the cache-hit metric hides: the 'regression' of a cut is recouped in a couple of turns once
    the session is long."""
    gap = prefix_now - baton_prefix
    if gap <= 0:
        return float("inf")
    return (write_mult * baton_prefix) / (read_mult * gap)


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


# --- the cost inversion table + render -----------------------------------------
def inversion_table(turns=200, delta=2_000, baton_prefix=20_000):
    """The vanity metric next to the bill it hides, over a frozen append-only trajectory.

    For a scan of turn counts: the reported cumulative hit (rising toward 1), the per-turn
    bill at that length (rising without bound), the cumulative bill (~T^2), and the
    'saved-token' headline (~T^2). Plus the cut break-even at each length — if the run were
    cut now to a flat `baton_prefix` leg, how few turns until that cut has repaid itself."""
    marks = [t for t in (2, 5, 10, 25, 50, 100, 200, 400) if t <= turns] or [turns]
    if marks[-1] != turns:
        marks.append(turns)
    rows = []
    for t in marks:
        prefix_now = (t - 1) * delta
        rows.append({
            "turn": t,
            "hit_cum": h_frozen(t),
            "turn_cost": turn_cost(t, delta),
            "cum_cost": cum_cost(t, delta),
            "saved_headline": saved_headline(t, delta),
            "prefix_now": prefix_now,
            "cut_break_even_turns": cut_break_even_turns(prefix_now, baton_prefix),
        })
    return {"turns": turns, "delta": delta, "baton_prefix": baton_prefix, "rows": rows}


def render_inversion(inv):
    d, B = inv["delta"], inv["baton_prefix"]
    saved_lbl = '"saved" hdln'   # a quoted label, hoisted out to keep the f-string escape-free
    L = []
    L.append(f"the cache-hit VANITY inversion — frozen append-only agent, delta={d:,} tok/turn")
    L.append("  the reported hit RATE and the per-turn BILL rise together: the rate is high")
    L.append("  *because* the cached prefix grew, and that same prefix makes each turn cost more.")
    L.append(f"  {'turn':>5}  {'hit(cum)':>8}  {'turn bill':>10}  {'cum bill':>13}  "
             f"{saved_lbl:>13}  {'cut pays back in':>16}")
    for r in inv["rows"]:
        be = r["cut_break_even_turns"]
        be_s = "n/a" if be == float("inf") else f"{be:.2f} turns"
        L.append(f"  {r['turn']:>5}  {r['hit_cum']*100:>7.1f}%  {r['turn_cost']:>10,.0f}  "
                 f"{r['cum_cost']:>13,.0f}  {r['saved_headline']:>13,.0f}  {be_s:>16}")
    L.append(f"  bill/headline in base-input-price units (read={READ_MULT}x, write={WRITE_MULT_5M}x).")
    L.append(f"  'cut pays back in' = turns to repay one cold {B:,}-tok baton prefix vs re-reading")
    L.append("  the grown prefix — a couple of turns once the session is long. The vanity metric")
    L.append("  frames that cut as a REGRESSION (a hit dip); the bill says it is a fast win.")
    return "\n".join(L)


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


# --- validation: measured anchors for the modeled decay axes --------------------
def _samples_from_validation_doc(data):
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        raise ValueError("validation JSON must be an object or list of sample objects")
    for key in ("samples", "validations", "rows"):
        rows = data.get(key)
        if isinstance(rows, list):
            return rows
    return [data]


def _sample_float(sample, keys, required=True, default=None):
    for key in keys:
        if key in sample and sample[key] is not None:
            return float(sample[key])
    if required:
        raise ValueError(f"sample {sample.get('name') or sample.get('id') or '<unnamed>'} missing one of {keys}")
    return default


def _sample_int(sample, keys, required=True, default=None):
    val = _sample_float(sample, keys, required, default)
    return int(val) if val is not None else None


def _sample_bool(sample, key, default):
    if key not in sample:
        return default
    val = sample[key]
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() not in ("0", "false", "no", "off")
    return bool(val)


def _measured_survival(sample):
    return _sample_float(sample, (
        "measured_survival", "measured_reuse", "measured_reuse_rate",
        "observed_survival", "observed_reuse", "observed_reuse_rate",
        "actual_survival", "actual_reuse", "actual_reuse_rate",
    ))


def _modeled_survival(sample):
    axis = str(sample.get("axis", "")).lower().replace("-", "_")
    params = {}
    if axis in ("fanout", "cross_agent_fanout", "cross_agent"):
        agents = _sample_int(sample, ("agents", "n", "workers"))
        concurrent = _sample_bool(sample, "concurrent", True)
        modeled, _payments = fanout_shared_reuse(agents, concurrent=concurrent)
        params = {"agents": agents, "concurrent": concurrent}
        return "cross_agent_reuse", modeled, params
    if axis in ("flex", "flexibility"):
        edit_depth = _sample_float(sample, ("edit_depth", "prefix_edit_depth"))
        params = {"edit_depth": edit_depth}
        return "flex_survival", s_flex(edit_depth), params
    if axis in ("tools", "tool_density", "tooldensity"):
        calls = _sample_int(sample, ("tool_calls_in_turn", "tool_calls", "calls_per_turn"))
        breakpoints = _sample_int(sample, ("breakpoints",), required=False, default=1)
        params = {"tool_calls_in_turn": calls, "breakpoints": breakpoints}
        return "tool_density_survival", s_tooldensity(calls, breakpoints), params
    if axis in ("compound", "single_agent_compound"):
        edit_depth = _sample_float(sample, ("edit_depth", "prefix_edit_depth"))
        calls = _sample_int(sample, ("tool_calls_in_turn", "tool_calls", "calls_per_turn"))
        breakpoints = _sample_int(sample, ("breakpoints",), required=False, default=1)
        params = {"edit_depth": edit_depth, "tool_calls_in_turn": calls, "breakpoints": breakpoints}
        return "single_agent_compound_survival", s_flex(edit_depth) * s_tooldensity(calls, breakpoints), params
    raise ValueError(f"sample {sample.get('name') or sample.get('id') or '<unnamed>'} has unknown axis {axis!r}")


def validate_measurements(data, tolerance=0.05, source=None):
    """Compare modeled survival factors against measured anchors.

    The input is intentionally small and explicit. Each sample must carry an `axis`
    plus a measured survival/reuse field such as `measured_survival` or
    `measured_reuse`. This keeps the modeled fan-out/flex constants falsifiable
    without pretending a generic hit-rate export is the same quantity.
    """
    rows = []
    for i, sample in enumerate(_samples_from_validation_doc(data), start=1):
        if not isinstance(sample, dict):
            raise ValueError(f"validation sample #{i} is not an object")
        metric, modeled, params = _modeled_survival(sample)
        measured = _measured_survival(sample)
        residual = measured - modeled
        abs_residual = abs(residual)
        rows.append({
            "name": sample.get("name") or sample.get("id") or f"sample-{i}",
            "axis": sample.get("axis"),
            "metric": metric,
            "parameters": params,
            "measured_survival": measured,
            "modeled_survival": modeled,
            "residual": residual,
            "abs_residual": abs_residual,
            "tolerance": tolerance,
            "status": "PASS" if abs_residual <= tolerance else "FAIL",
            "source": sample.get("source"),
        })
    max_abs = max((r["abs_residual"] for r in rows), default=0.0)
    failures = sum(1 for r in rows if r["status"] != "PASS")
    return {
        "schema": "fak.cache_curve.validation.v1",
        "source": source,
        "tolerance": tolerance,
        "summary": {
            "samples": len(rows),
            "failures": failures,
            "max_abs_residual": max_abs,
            "status": "PASS" if failures == 0 else "FAIL",
        },
        "rows": rows,
    }


def load_validation(path, tolerance=0.05):
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return validate_measurements(data, tolerance=tolerance, source=path)


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


def render_validation(v):
    L = []
    s = v["summary"]
    L.append(f"measured decay validation — {s['samples']} sample(s), tolerance ±{v['tolerance']:.3f}: "
             f"{s['status']} (max |residual| {s['max_abs_residual']:.3f})")
    L.append(f"  {'sample':<22} {'axis':<18} {'metric':<30} {'measured':>9} {'modeled':>9} {'residual':>9} {'status':>6}")
    for r in v["rows"]:
        L.append(f"  {r['name']:<22} {str(r['axis']):<18} {r['metric']:<30} "
                 f"{r['measured_survival']:>9.3f} {r['modeled_survival']:>9.3f} "
                 f"{r['residual']:>+9.3f} {r['status']:>6}")
    L.append("  measured_survival is an explicit anchor for the same s/reuse quantity the model uses;")
    L.append("  generic cache hit-rate exports are not silently treated as fan-out or flex survival.")
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


# --- the cost curve SVG: billed tokens/turn, naive vs fak prefix-preserving ---------
def _xml_esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def render_costcurve_svg(turns=50, delta=2_000, read_mult=READ_MULT):
    """A standalone, deterministic SVG of the long-session cost curve: billed prompt
    tokens PER TURN over a `turns`-turn session, naive full-price re-send vs the fak
    prefix-preserving (cache-held) line. Every plotted number is computed from
    cache_curve's own model (turn_cost / naive_turn_cost / h_frozen) and the measured
    ANCHOR, so the chart is generated from real cache_curve.py data, not hand-drawn.

    Witnessed-vs-modeled is drawn explicitly: the two curves are the modeled per-turn
    bill (illustrative shape at cache-read 0.1x base); the witnessed anchors (the 96.6%
    machine-wide cache-read ceiling, the ~4.1x vs tuned warm-cache fleet result) sit in
    a separate annotation box, never folded into the modeled lines. Axes and units are
    labeled; the SVG carries real <text> (renders in any viewer) and explicit width/height."""
    ts = list(range(1, turns + 1))
    fak_line = [turn_cost(t, delta, read_mult) for t in ts]
    naive_line = [naive_turn_cost(t, delta) for t in ts]
    cum_fak = cum_cost(turns, delta, read_mult)
    cum_naive = sum(naive_turn_cost(t, delta) for t in ts)
    modeled_ceiling = h_frozen(turns)
    witnessed_ceiling = ANCHOR["cache_read_share"]

    # y-axis: round top tick with headroom above the tallest point (integer math).
    step = 20000
    ymax = step
    top_needed = max(naive_line) * 1.08
    while ymax < top_needed:
        ymax += step
    yticks = list(range(0, int(ymax) + 1, step))

    # layout
    W, H = 920, 580
    ML, MR, MT, MB = 84, 40, 104, 84
    px0, py1 = ML, H - MB
    pw, ph = W - ML - MR, H - MT - MB

    def xof(t):
        return px0 + (t - 1) / max(1, turns - 1) * pw

    def yof(v):
        return py1 - v / ymax * ph

    NAIVE_C = "#cf222e"
    FAK_C = "#1a7f37"
    GRID_C = "#d0d7de"
    TEXT_C = "#24292f"
    MUTED_C = "#57606a"
    WIT_C = "#0969da"

    pts_naive = " ".join(f"{xof(t):.1f},{yof(v):.1f}" for t, v in zip(ts, naive_line))
    pts_fak = " ".join(f"{xof(t):.1f},{yof(v):.1f}" for t, v in zip(ts, fak_line))
    area = (f"M {xof(1):.1f},{yof(fak_line[0]):.1f} "
            + " ".join(f"L {xof(t):.1f},{yof(v):.1f}" for t, v in zip(ts, fak_line))
            + " " + " ".join(f"L {xof(t):.1f},{yof(v):.1f}"
                             for t, v in zip(reversed(ts), reversed(naive_line)))
            + " Z")

    L = []
    L.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" '
             f'font-family="-apple-system,Segoe UI,Helvetica,Arial,sans-serif" '
             f'role="img" aria-labelledby="cc-title cc-desc">')
    L.append(f'<title id="cc-title">{_xml_esc("Long-session cost curve: naive re-send vs fak prefix-preserving")}</title>')
    L.append(f'<desc id="cc-desc">{_xml_esc("Billed prompt tokens per turn over a " + str(turns) + "-turn session. The naive full-price re-send line rises steeply; the fak prefix-preserving line (provider prompt-cache prefix kept byte-identical so the discount holds) stays shallow. Curves are the cache_curve.py modeled per-turn bill; witnessed anchors (96.6% machine-wide cache-read share; ~4.1x vs tuned warm-cache fleet) are annotated separately.")}</desc>')
    L.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>')

    # title + subtitle
    L.append(f'<text x="{W // 2}" y="34" text-anchor="middle" font-size="20" '
             f'font-weight="700" fill="{TEXT_C}">Long-session cost: naive re-send vs fak prefix-preserving</text>')
    L.append(f'<text x="{W // 2}" y="58" text-anchor="middle" font-size="12.5" fill="{MUTED_C}">'
             f'Billed prompt tokens per turn over a {turns}-turn session '
             f'(delta={delta:,} tok/turn, cache-read {read_mult:g}&#215; base). '
             f'Curves = modeled; box = witnessed.</text>')

    # legend (horizontal, below subtitle)
    ly = 80
    L.append(f'<rect x="{ML}" y="{ly - 9}" width="22" height="10" fill="{NAIVE_C}"/>')
    L.append(f'<text x="{ML + 28}" y="{ly}" font-size="12" fill="{TEXT_C}">naive re-send (no cache match) &#8212; modeled</text>')
    nlabel_w = 300
    L.append(f'<rect x="{ML + nlabel_w}" y="{ly - 9}" width="22" height="10" fill="{FAK_C}"/>')
    L.append(f'<text x="{ML + nlabel_w + 28}" y="{ly}" font-size="12" fill="{TEXT_C}">fak prefix-preserving (cache held) &#8212; modeled</text>')
    L.append(f'<rect x="{ML + 620}" y="{ly - 9}" width="22" height="10" fill="{WIT_C}" fill-opacity="0.25"/>')
    L.append(f'<rect x="{ML + 620}" y="{ly - 9}" width="22" height="10" fill="none" stroke="{WIT_C}"/>')
    L.append(f'<text x="{ML + 648}" y="{ly}" font-size="12" fill="{TEXT_C}">witnessed anchor</text>')

    # gridlines + y ticks + value labels
    for tk in yticks:
        yy = yof(tk)
        L.append(f'<line x1="{px0}" y1="{yy:.1f}" x2="{px0 + pw:.1f}" y2="{yy:.1f}" '
                 f'stroke="{GRID_C}" stroke-width="1"/>')
        L.append(f'<text x="{px0 - 10}" y="{yy + 4:.1f}" text-anchor="end" font-size="11" '
                 f'fill="{MUTED_C}">{tk:,}</text>')
    # x ticks (1, every 10, endpoint)
    for t in range(1, turns + 1):
        if t == 1 or t == turns or t % 10 == 0:
            xx = xof(t)
            L.append(f'<line x1="{xx:.1f}" y1="{py1}" x2="{xx:.1f}" y2="{py1 + 5}" '
                     f'stroke="{MUTED_C}" stroke-width="1"/>')
            L.append(f'<text x="{xx:.1f}" y="{py1 + 22}" text-anchor="middle" font-size="11" '
                     f'fill="{MUTED_C}">{t}</text>')
    # axes
    L.append(f'<line x1="{px0}" y1="{MT}" x2="{px0}" y2="{py1}" stroke="{TEXT_C}" stroke-width="1.2"/>')
    L.append(f'<line x1="{px0}" y1="{py1}" x2="{px0 + pw}" y2="{py1}" stroke="{TEXT_C}" stroke-width="1.2"/>')
    # axis titles
    L.append(f'<text x="{px0 + pw // 2}" y="{H - 22}" text-anchor="middle" font-size="12.5" '
             f'font-weight="600" fill="{TEXT_C}">turn</text>')
    L.append(f'<text transform="translate(24,{MT + ph // 2}) rotate(-90)" text-anchor="middle" '
             f'font-size="12.5" font-weight="600" fill="{TEXT_C}">billed tokens / turn '
             f'(&#215; base input price)</text>')

    # area between curves (the saving) + the two curves
    L.append(f'<path d="{area}" fill="{FAK_C}" fill-opacity="0.08" stroke="none"/>')
    L.append(f'<polyline points="{pts_naive}" fill="none" stroke="{NAIVE_C}" stroke-width="2.6"/>')
    L.append(f'<polyline points="{pts_fak}" fill="none" stroke="{FAK_C}" stroke-width="2.6"/>')

    # endpoint dots + last-turn value annotations
    ex = xof(turns)
    n_lx, n_ly = ex, yof(naive_line[-1])
    f_lx, f_ly = ex, yof(fak_line[-1])
    L.append(f'<circle cx="{n_lx:.1f}" cy="{n_ly:.1f}" r="4" fill="{NAIVE_C}"/>')
    L.append(f'<circle cx="{f_lx:.1f}" cy="{f_ly:.1f}" r="4" fill="{FAK_C}"/>')
    L.append(f'<text x="{n_lx - 8:.1f}" y="{n_ly - 10:.1f}" text-anchor="end" font-size="12" '
             f'font-weight="700" fill="{NAIVE_C}">{naive_line[-1]:,.0f}</text>')
    L.append(f'<text x="{f_lx - 8:.1f}" y="{f_ly + 18:.1f}" text-anchor="end" font-size="12" '
             f'font-weight="700" fill="{FAK_C}">{fak_line[-1]:,.0f}</text>')
    L.append(f'<text x="{xof(1) + 8:.1f}" y="{yof(fak_line[0]) - 8:.1f}" font-size="11" '
             f'fill="{MUTED_C}">turn 1: {fak_line[0]:,.0f} (both equal)</text>')
    ratio_last = naive_line[-1] / fak_line[-1] if fak_line[-1] else 0.0
    L.append(f'<text x="{ex - 8:.1f}" y="{(n_ly + f_ly) / 2:.0f}" text-anchor="end" '
             f'font-size="12.5" font-weight="700" fill="{TEXT_C}">{ratio_last:.1f}&#215; at turn {turns}</text>')

    # witness annotation box (top-left of plot, clear of both low-start lines)
    bx0, by0, bw, bh = px0 + 14, MT + 14, 330, 132
    L.append(f'<rect x="{bx0}" y="{by0}" width="{bw}" height="{bh}" rx="8" '
             f'fill="{WIT_C}" fill-opacity="0.06" stroke="{WIT_C}" stroke-width="1"/>')
    L.append(f'<text x="{bx0 + 12}" y="{by0 + 22}" font-size="11.5" font-weight="700" '
             f'fill="{WIT_C}">WITNESSED (measured, not modeled)</text>')
    wl = [
        f'cache-read share: {witnessed_ceiling * 100:.1f}% machine-wide',
        f'(session_audit.py, {ANCHOR["sessions"]} sessions; I:O {ANCHOR["io_ratio"]:.1f}:1)',
        f'fleet reuse: ~4.1&#215; vs tuned warm-cache',
        f'(50-turn &#215; 5-agent, M3 Pro); ~60&#215; vs naive*',
    ]
    for i, line in enumerate(wl):
        L.append(f'<text x="{bx0 + 12}" y="{by0 + 46 + i * 20}" font-size="11" '
                 f'fill="{TEXT_C}">{line}</text>')

    # footer / fence notes
    fy = H - MB + 40
    L.append(f'<text x="{px0}" y="{fy}" font-size="10.5" fill="{MUTED_C}">'
             f'modeled: per-turn bill from cache_curve.py (read {read_mult:g}&#215;, full price 1&#215;); '
             f'illustrative shape, not a price quote. cumulative {turns} turns &#8594; '
             f'naive {cum_naive:,.0f} vs fak {cum_fak:,.0f} billed tok.</text>')
    L.append(f'<text x="{px0}" y="{fy + 15}" font-size="10.5" fill="{MUTED_C}">'
             f'calibration: modeled frozen ceiling at T={turns} = {modeled_ceiling * 100:.1f}% '
             f'&#8776; witnessed {witnessed_ceiling * 100:.1f}%. *naive-arm wall-clock modeled from '
             f'prefill curve; token ratios exact. fak guarantees prefix byte-identity, not a cache hit.</text>')

    L.append('</svg>')
    return "\n".join(L)


def render_report(turns, anchor_path, validate_path=None, tolerance=0.05):
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
    L.append("## The vanity inversion: the hit rate vs the bill it hides\n")
    L.append("```")
    L.append(render_inversion(inversion_table(turns)))
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
    if validate_path:
        v = load_validation(validate_path, tolerance=tolerance)
        L.append("## Measured decay validation\n")
        L.append("```")
        L.append(render_validation(v))
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
    qi = sub.add_parser("inversion")
    qi.add_argument("--turns", type=int, default=200)
    qi.add_argument("--delta", type=int, default=2_000, help="new tokens appended per turn")
    qi.add_argument("--baton", type=int, default=20_000, help="fresh-leg prefix (system + O(1) baton)")
    qa = sub.add_parser("anchor")
    qa.add_argument("json")
    qv = sub.add_parser("validate")
    qv.add_argument("json")
    qv.add_argument("--tolerance", type=float, default=0.05)
    qv.add_argument("--json", dest="json_output", action="store_true")
    qc = sub.add_parser("costcurve", help="emit the cost-curve SVG (billed tokens/turn, naive vs fak)")
    qc.add_argument("--turns", type=int, default=50, help="session length in turns")
    qc.add_argument("--delta", type=int, default=2_000, help="new tokens appended per turn")
    qc.add_argument("--out", default=None,
                    help="output SVG path (default: docs/adoption/diagrams/cost-curve.svg)")
    qr = sub.add_parser("report")
    qr.add_argument("--turns", type=int, default=200)
    qr.add_argument("--anchor", default=None)
    qr.add_argument("--validate", default=None)
    qr.add_argument("--tolerance", type=float, default=0.05)
    qr.add_argument("--md", default=None)
    a = p.parse_args()

    if a.cmd == "curves":
        print(_doc_model())
        print(render_curves(curve_table(a.turns)))
        return 0
    elif a.cmd == "fanout":
        print(render_fanout(fanout_table(a.agents)))
        return 0
    elif a.cmd == "compound":
        print(render_compound(compound_scenario(a.turns)))
        return 0
    elif a.cmd == "inversion":
        print(render_inversion(inversion_table(a.turns, a.delta, a.baton)))
        return 0
    elif a.cmd == "chart":
        print(render_chart(a.turns))
        return 0
    elif a.cmd == "costcurve":
        out = a.out or str(Path(__file__).resolve().parents[1] / "docs" / "adoption"
                           / "diagrams" / "cost-curve.svg")
        svg = render_costcurve_svg(a.turns, a.delta)
        out_p = Path(out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(svg, encoding="utf-8")
        print(f"wrote {out_p}")
        return 0
    elif a.cmd == "anchor":
        print(render_anchor(load_anchor(a.json)))
        return 0
    elif a.cmd == "validate":
        try:
            v = load_validation(a.json, tolerance=a.tolerance)
        except Exception as e:
            print(f"cache_curve.py validate: {e}", file=sys.stderr)
            return 2
        if a.json_output:
            print(json.dumps(v, indent=2, sort_keys=True))
        else:
            print(render_validation(v))
        return 0 if v["summary"]["failures"] == 0 else 1
    elif a.cmd == "report":
        md = render_report(a.turns, a.anchor, validate_path=a.validate, tolerance=a.tolerance)
        if a.md:
            open(a.md, "w", encoding="utf-8").write(md)
            print(f"wrote {a.md}", file=sys.stderr)
        print(md)
        return 0


if __name__ == "__main__":
    sys.exit(main())
