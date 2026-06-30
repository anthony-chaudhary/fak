#!/usr/bin/env python3
"""Hook-boundary microbench: turn the per-tool-call DOS hook spawn tax into a watched number.

docs/perf-dos-hook-cost.md argues the dominant felt cost of wiring the DOS kernel
as Claude Code hooks is *process-per-decide*: every tool call spawns
``dos hook pretool`` + ``dos hook posttool``. The doc names a microbench "like
``fak bench`` but at the hook boundary" as its acceptance criterion -- but none
existed, so the felt cost was *estimated* (the doc said 6-50ms/spawn), never
measured on the host. This is that bench, the latency-side twin of the process-side
``tools/proc_resource_guard.py`` (docs/perf-runaway-guard.md).

It feeds a frozen synthetic PreToolUse / PostToolUse event over STDIN to the real
``dos hook pretool|posttool``, times N warm spawns of each (wall-clock), and
reports p10/p50/p90/max. It also folds the *passthrough fraction* and the
in-process internal latency from ``.dos/metrics/observations.jsonl`` (the share of
spawns that changed nothing, and how much of the wall-clock is kernel work vs cold
process startup) so the spawn-startup overhead is isolated.

Read-only w.r.t. the git tree -- it writes nothing tracked. It DOES invoke the
real hooks, which append to the workspace's gitignored ``.dos/metrics`` observation
log (the same log normal operation writes); run it for an occasional measurement,
not in a tight loop. If ``dos`` is not on PATH it prints a clean SKIP and exits 0,
so a CI runner without the dos-kernel package never hard-fails.

    python tools/dos_hook_bench.py                 # the card
    python tools/dos_hook_bench.py --json          # machine-readable
    python tools/dos_hook_bench.py --iters 30 --tool Read
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from dispatch_worker import install_no_window_subprocess_defaults
import sys
import time
from pathlib import Path
from typing import Any, Callable
install_no_window_subprocess_defaults(subprocess)

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except (AttributeError, ValueError):
    pass

SCHEMA = "fleet-dos-hook-bench/1"
DEFAULT_ITERS = 20
DEFAULT_TOOL = "Read"
_OBS_REL = (".dos", "metrics", "observations.jsonl")


def repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    return here.parent.parent


# ---------------------------------------------------------------------------
# Pure helpers (the test pins these; no subprocess / wall-clock)
# ---------------------------------------------------------------------------

def percentile(samples: list[float], pct: float) -> float | None:
    """Nearest-rank percentile (deterministic, no interpolation)."""
    if not samples:
        return None
    s = sorted(samples)
    if pct <= 0:
        return s[0]
    if pct >= 100:
        return s[-1]
    k = math.ceil(pct / 100.0 * len(s)) - 1
    k = max(0, min(k, len(s) - 1))
    return s[k]


def summarize(samples: list[float]) -> dict[str, Any]:
    """p10/p50/p90/min/max/mean over a millisecond sample list (rounded 1dp)."""
    if not samples:
        return {"n": 0, "p10": None, "p50": None, "p90": None,
                "min": None, "max": None, "mean": None}
    return {
        "n": len(samples),
        "p10": round(percentile(samples, 10), 1),
        "p50": round(percentile(samples, 50), 1),
        "p90": round(percentile(samples, 90), 1),
        "min": round(min(samples), 1),
        "max": round(max(samples), 1),
        "mean": round(sum(samples) / len(samples), 1),
    }


def synthetic_event(tool: str, idx: int, *, kind: str, workspace: str) -> dict[str, Any]:
    """A Claude Code hook stdin payload. ``kind`` is 'pre' or 'post'.

    Each call varies ``tool_input`` by ``idx`` so the bench measures the clean
    passthrough spawn cost rather than tripping the posttool repeat-detector
    (which is a separate, already-established behavior, not the spawn tax).
    """
    event: dict[str, Any] = {
        "session_id": "dos-hook-bench",
        "transcript_path": "dos-hook-bench",
        "cwd": workspace,
        "tool_name": tool,
        "tool_input": {"file_path": f"README.md#bench-{idx}"},
    }
    if kind == "post":
        event["hook_event_name"] = "PostToolUse"
        event["tool_response"] = {"type": "text", "text": f"bench-{idx}"}
    else:
        event["hook_event_name"] = "PreToolUse"
    return event


def observation_stats(obs_path: Path, *, tail: int = 20000) -> dict[str, Any]:
    """Passthrough fraction + internal latency p50 from the observation log.

    Best-effort: returns {} if the log is absent/unreadable. Counts only the
    per-tool-call hooks (pretool/posttool); a 'passthrough' outcome is a spawn
    that changed nothing (paid the process tax, bought no verdict)."""
    if not obs_path.exists():
        return {}
    try:
        lines = obs_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return {}
    lines = lines[-tail:]
    hook_total = 0
    passthrough = 0
    latencies: list[float] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if not isinstance(rec, dict):
            continue
        # The live observation schema keys the hook on "verb" (pretool/posttool);
        # accept the alt names too so the fold survives a schema rename.
        hook = str(rec.get("verb") or rec.get("hook") or rec.get("event") or "")
        if hook not in ("pretool", "posttool", "PreToolUse", "PostToolUse"):
            continue
        hook_total += 1
        outcome = str(rec.get("outcome") or rec.get("decision") or "")
        if outcome in ("passthrough", "allow", "", "pass"):
            passthrough += 1
        lat = rec.get("latency_ms")
        if isinstance(lat, (int, float)):
            latencies.append(float(lat))
    if not hook_total:
        return {}
    return {
        "hook_observations": hook_total,
        "passthrough": passthrough,
        "passthrough_fraction": round(passthrough / hook_total, 4),
        "internal_latency_ms_p50": (round(percentile(latencies, 50), 2)
                                    if latencies else None),
    }


# ---------------------------------------------------------------------------
# Timing seam (real subprocess; tests inject a fake)
# ---------------------------------------------------------------------------

Spawner = Callable[[str, dict[str, Any], Path], float]


def time_one_spawn(hook: str, event: dict[str, Any], workspace: Path) -> float:
    """Wall-clock milliseconds for one ``dos hook <hook>`` spawn fed ``event``."""
    t0 = time.perf_counter()
    try:
        subprocess.run(
            ["dos", "hook", hook, "--workspace", str(workspace)],
            input=json.dumps(event), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return float("nan")
    return (time.perf_counter() - t0) * 1000.0


def time_spawns(hook: str, tool: str, iters: int, workspace: Path, *,
                spawner: Spawner | None = None) -> list[float]:
    """Warm once, then time ``iters`` spawns; drop NaN (failed) samples."""
    spawn = spawner or time_one_spawn
    kind = "post" if hook == "posttool" else "pre"
    spawn(hook, synthetic_event(tool, -1, kind=kind, workspace=str(workspace)), workspace)  # warm
    out: list[float] = []
    for i in range(iters):
        ms = spawn(hook, synthetic_event(tool, i, kind=kind, workspace=str(workspace)), workspace)
        if ms == ms:  # drop NaN
            out.append(ms)
    return out


# ---------------------------------------------------------------------------
# Wiring + render
# ---------------------------------------------------------------------------

def build_payload(*, workspace: str, pre: list[float], post: list[float],
                  obs: dict[str, Any]) -> dict[str, Any]:
    pre_s = summarize(pre)
    post_s = summarize(post)
    per_call_p50 = None
    if pre_s["p50"] is not None and post_s["p50"] is not None:
        per_call_p50 = round(pre_s["p50"] + post_s["p50"], 1)
    return {
        "schema": SCHEMA,
        "ok": True,
        "workspace": workspace,
        "pretool_ms": pre_s,
        "posttool_ms": post_s,
        "per_tool_call_p50_ms": per_call_p50,
        "observations": obs,
    }


def collect(workspace: Path, *, iters: int, tool: str,
            spawner: Spawner | None = None) -> dict[str, Any]:
    pre = time_spawns("pretool", tool, iters, workspace, spawner=spawner)
    post = time_spawns("posttool", tool, iters, workspace, spawner=spawner)
    obs = observation_stats(workspace.joinpath(*_OBS_REL))
    return build_payload(workspace=str(workspace), pre=pre, post=post, obs=obs)


def render(p: dict[str, Any]) -> str:
    pre = p.get("pretool_ms") or {}
    post = p.get("posttool_ms") or {}
    obs = p.get("observations") or {}
    lines = [
        "dos hook-boundary microbench (wall-clock per spawn)",
        f"  pretool  n={pre.get('n')}  p10={pre.get('p10')}  p50={pre.get('p50')}  "
        f"p90={pre.get('p90')}  max={pre.get('max')} ms",
        f"  posttool n={post.get('n')}  p10={post.get('p10')}  p50={post.get('p50')}  "
        f"p90={post.get('p90')}  max={post.get('max')} ms",
        f"  -> per-tool-call p50 floor (pre+post): {p.get('per_tool_call_p50_ms')} ms",
    ]
    if obs:
        lines.append(
            f"  observations: passthrough {obs.get('passthrough_fraction')} of "
            f"{obs.get('hook_observations')} hook spawns; internal latency p50 "
            f"{obs.get('internal_latency_ms_p50')} ms (the rest is process startup)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Hook-boundary microbench for the per-tool-call DOS hook spawn tax (read-only).")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--iters", type=int, default=DEFAULT_ITERS, help=f"spawns per hook (default {DEFAULT_ITERS})")
    ap.add_argument("--tool", default=DEFAULT_TOOL, help=f"tool_name in the synthetic event (default {DEFAULT_TOOL})")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args(argv)

    if shutil.which("dos") is None:
        msg = {"schema": SCHEMA, "ok": True, "skipped": "dos not on PATH"}
        print(json.dumps(msg, indent=2) if args.json
              else "dos hook-boundary microbench: SKIP (dos not on PATH)")
        return 0

    workspace = Path(args.workspace).resolve() if args.workspace else repo_root()
    payload = collect(workspace, iters=max(1, args.iters), tool=args.tool)
    print(json.dumps(payload, indent=2) if args.json else render(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
