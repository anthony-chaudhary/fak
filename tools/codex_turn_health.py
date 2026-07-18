#!/usr/bin/env python3
"""Codex session turn/compaction health monitor.

Rolls up the local Codex rollout store (``~/.codex/sessions/**/*.jsonl``) into a
small, privacy-preserving health report that surfaces the three turn/token
pathologies found in the 2026-07-15 fleet audit:

  1. turn inflation      - turns that call NO tool (talk-not-act), the model
                           yielding with prose instead of continuing to work;
  2. guard-refusal loops - turns whose every proposed tool call was refused by
                           the fak kernel (REQUIRE_WITNESS / POLICY_BLOCK), the
                           model re-proposing blocked calls turn after turn;
  3. compaction posture  - the occupancy at which auto-compaction fired, so a
                           healthy near-budget compaction is distinguishable
                           from a premature one on a stuck loop.

It does NOT copy prompts, tool arguments, tool results, diffs, or model text.
Only structural counts, rates, session ids (already opaque UUIDs), and the
coarse *category* of a turn's trailing message survive into the report.

Companion to ``codex_dos_recent_audit.py`` (DOS-hook evidence) and the runtime
alignment note ``docs/notes/CODEX-TURN-COMPACTION-ALIGNMENT-2026-07-15.md``.
Background on why the compaction budget sits at 96K (not the 258K model
window): ``cmd/fak/codex_launcher.go`` (``model_auto_compact_token_limit``).
"""
from __future__ import annotations

import argparse
from collections import Counter
import glob
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable

SCHEMA = "fak-codex-turn-health/1"

# --- classification thresholds (named, not magic) ---------------------------
# fak sets codex `model_auto_compact_token_limit` to this for guarded launches
# (cmd/fak/codex_launcher.go, cmd/dispatchworker/guard.go). A compaction firing
# near this value is the intended #4253 budget, NOT premature.
COMPACT_BUDGET = 96_000
# A compaction below this occupancy is genuinely premature -- almost always the
# symptom of a stuck no-op loop re-accreting tiny context, not a full window.
PREMATURE_FILL = 40_000
# A session is a "guard-refusal loop" once this many of its turns end with every
# proposed tool call refused.
REFUSAL_LOOP_MIN = 3
# A session flagged for turn inflation: enough turns, and most call no tool.
INFLATION_MIN_TURNS = 5
INFLATION_ZERO_RATIO = 0.5

# Markers used to categorise the trailing message of a zero-tool turn. These
# match the fak kernel's refusal envelope and the model's skill-announce
# preamble; substring checks only, never stored.
_REFUSAL_MARKERS = ("refused by the fak kernel", "require_witness", "policy_block",
                    "were refused", "proposed tool call")
_PREAMBLE_MARKERS = ("i'll use the", "i’ll use the", "using `", "using the `")


def _payload_type(obj: dict) -> str | None:
    p = obj.get("payload")
    if isinstance(p, dict):
        return p.get("type")
    return None


def classify_zero_tool(msg: str) -> str:
    """Category of a zero-tool turn from its trailing agent message (lowered)."""
    low = (msg or "").lower()
    if any(m in low for m in _REFUSAL_MARKERS):
        return "guard_refused"
    if any(m in low for m in _PREAMBLE_MARKERS) and "skill" in low:
        return "preamble_noop"
    if low.strip():
        return "talk_only"
    return "silent"


def fold_rows(rows: Iterable[dict]) -> dict[str, Any]:
    """Pure fold of one session's rollout rows into a structural stat dict.

    Kept pure (no IO) so the test can drive it with synthetic rows.
    """
    model = None
    turns = 0
    tool_calls = 0
    zero = Counter()          # category -> count of zero-tool turns
    compactions: list[int] = []  # occupancy at each real compaction

    in_turn = False
    cur_tools = 0
    cur_msg = ""
    last_input = 0

    for obj in rows:
        t = obj.get("type")
        p = obj.get("payload") if isinstance(obj.get("payload"), dict) else {}
        pt = p.get("type")

        if t == "turn_context" and isinstance(p, dict):
            model = p.get("model") or model
        elif t == "response_item" and pt == "function_call":
            tool_calls += 1
            cur_tools += 1
        elif pt == "token_count":
            info = p.get("info") or {}
            li = (info.get("last_token_usage") or {}).get("input_tokens")
            if li:
                last_input = li
        elif pt == "task_started":
            in_turn = True
            turns += 1
            cur_tools = 0
            cur_msg = ""
        elif pt == "agent_message":
            cur_msg = p.get("message") or ""
        elif pt == "task_complete":
            if in_turn and cur_tools == 0:
                zero[classify_zero_tool(cur_msg)] += 1
            in_turn = False
        elif t == "compacted":
            # top-level `compacted` is the real compaction marker; the paired
            # event_msg/context_compacted is deliberately ignored to avoid 2x.
            if last_input:
                compactions.append(last_input)

    return {
        "model": model,
        "turns": turns,
        "tool_calls": tool_calls,
        "zero_tool": dict(zero),
        "zero_tool_total": sum(zero.values()),
        "compactions": compactions,
    }


def scan_file(path: str) -> dict[str, Any] | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            stat = fold_rows(_parse_lines(fh))
    except OSError:
        return None
    stat["session"] = Path(path).stem
    return stat


def _parse_lines(lines: Iterable[str]) -> Iterable[dict]:
    """Yield parsed dict rows, silently skipping blank or malformed jsonl lines.

    Rollout files are append-only and can carry a torn final line (a crashed
    writer) or a stray non-JSON line; a monitor must survive them, not abort the
    whole scan on one bad byte.
    """
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            obj = json.loads(ln)
        except (ValueError, TypeError):
            continue
        if isinstance(obj, dict):
            yield obj


def _pct(vals: list[int], p: float) -> int:
    if not vals:
        return 0
    v = sorted(vals)
    k = (len(v) - 1) * p / 100.0
    f = int(k)
    if f + 1 < len(v):
        return int(v[f] + (v[f + 1] - v[f]) * (k - f))
    return int(v[f])


def roll_up(stats: list[dict[str, Any]], top: int = 15) -> dict[str, Any]:
    """Pure aggregation of per-session stats into the health report."""
    stats = [s for s in stats if s and s.get("turns")]
    turns = sum(s["turns"] for s in stats)
    tool_calls = sum(s["tool_calls"] for s in stats)
    zero_break = Counter()
    all_fills: list[int] = []
    refusal_loops = []
    inflation = []

    for s in stats:
        for k, v in s.get("zero_tool", {}).items():
            zero_break[k] += v
        all_fills.extend(s.get("compactions", []))
        refused = s.get("zero_tool", {}).get("guard_refused", 0)
        if refused >= REFUSAL_LOOP_MIN:
            refusal_loops.append({"session": s["session"], "model": s.get("model"),
                                  "refused_turns": refused, "turns": s["turns"]})
        z = s.get("zero_tool_total", 0)
        if s["turns"] >= INFLATION_MIN_TURNS and z / s["turns"] >= INFLATION_ZERO_RATIO:
            inflation.append({"session": s["session"], "model": s.get("model"),
                              "turns": s["turns"], "tool_calls": s["tool_calls"],
                              "zero_tool_turns": z})

    zero_total = sum(zero_break.values())
    premature = [f for f in all_fills if f < PREMATURE_FILL]
    near_budget = [f for f in all_fills if 0.85 * COMPACT_BUDGET <= f <= 1.10 * COMPACT_BUDGET]
    near_window = [f for f in all_fills if f >= 200_000]

    refusal_loops.sort(key=lambda x: x["refused_turns"], reverse=True)
    inflation.sort(key=lambda x: x["zero_tool_turns"], reverse=True)

    flags = []
    if turns and zero_total / turns > 0.20:
        flags.append(f"HIGH_ZERO_TOOL_RATE: {zero_total}/{turns} turns ({100*zero_total/turns:.0f}%) call no tool")
    if refusal_loops:
        flags.append(f"GUARD_REFUSAL_LOOPS: {len(refusal_loops)} session(s) re-propose refused tool calls")
    if premature:
        flags.append(f"PREMATURE_COMPACTION: {len(premature)} compaction(s) fired under {PREMATURE_FILL//1000}K occupancy (stuck-loop symptom)")

    return {
        "schema": SCHEMA,
        "sessions_with_turns": len(stats),
        "totals": {
            "turns": turns,
            "tool_calls": tool_calls,
            "tool_calls_per_turn": round(tool_calls / turns, 1) if turns else 0,
            "zero_tool_turns": zero_total,
            "zero_tool_rate": round(zero_total / turns, 3) if turns else 0,
        },
        "zero_tool_breakdown": dict(zero_break),
        "compaction": {
            "events": len(all_fills),
            "budget": COMPACT_BUDGET,
            "occupancy_p50": _pct(all_fills, 50),
            "occupancy_p90": _pct(all_fills, 90),
            "near_budget_96k": len(near_budget),
            "near_window_200k_plus": len(near_window),
            "premature_lt40k": len(premature),
        },
        "guard_refusal_loops": refusal_loops[:top],
        "turn_inflation": inflation[:top],
        "flags": flags,
    }


def default_session_glob() -> str:
    root = os.environ.get("CODEX_HOME") or os.path.join(os.path.expanduser("~"), ".codex")
    return os.path.join(root, "sessions", "**", "*.jsonl")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Codex turn/compaction health monitor")
    ap.add_argument("--glob", default=default_session_glob(),
                    help="glob for rollout jsonl files (default: ~/.codex/sessions/**/*.jsonl)")
    ap.add_argument("--top", type=int, default=15, help="worst-offender rows to list")
    ap.add_argument("--limit", type=int, default=0, help="cap files scanned (0 = all)")
    args = ap.parse_args(argv)

    files = sorted(glob.glob(args.glob, recursive=True))
    if args.limit:
        files = files[-args.limit:]
    stats = [scan_file(f) for f in files]
    report = roll_up([s for s in stats if s], top=args.top)
    report["scanned_files"] = len(files)
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    # Non-zero exit if any red flag fired, so CI/nightrun can gate on it.
    return 1 if report["flags"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
