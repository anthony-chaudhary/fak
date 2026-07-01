#!/usr/bin/env python3
r"""One guarded, switcher-routed, bounded dispatch TICK — the keystone that turns
the existing pieces into a safe always-on issue dispatcher.

The historical spawn path (``dos loop --enact`` -> ``dispatch_worker.py --lane
{lane}`` -> ``claude -p /dos-kernel:dos-dispatch-loop``) had two holes this tick
closes, in order, before a single worker is launched:

  1. PREFLIGHT (DoS safety)   tools/dispatch_preflight.py must return SPAWN_OK:
                              host guard clean ∧ an account is free ∧ live workers
                              < cap. If not, this tick REFUSES and exits non-zero
                              — that refusal IS the no-DoS guarantee (the live
                              population can never exceed the cap, so per-session
                              hook pressure stays bounded).
  2. SWITCHER PIN (routing)   the worker is launched with CLAUDE_CONFIG_DIR pinned
                              to the switcher's chosen account — never the ambient
                              default or a sibling token that historically ate the
                              dispatch when it was throttled/auth-blocked.

It then picks the lane with the most open issues (the issue_lane_router fold), or
an explicit ``--lane``, and launches ONE detached worker on it. DRY-RUN BY
DEFAULT: it prints exactly what it would do (account, lane, command, witness).
``--live`` spawns. The witness the worker should use to keep/revert its own work
is the BENCHMARK (tools/bench_witness.py --lane <lane>), not the test suite — the
operator's rule for this loop — and is named in the tick record for the worker.

    python tools/issue_dispatch.py                 # plan one safe tick (dry-run)
    python tools/issue_dispatch.py --max-workers 2 --live   # spawn one worker

``--wave`` (#1335) fans this out: in ONE tick it spawns up to ``--max-workers``
workers across pairwise TREE-DISJOINT lanes, each on its own seat. The partition is
PRICED — every lane is arbitrated (``dos arbitrate``) against the wave's already
admitted leases, so a colliding set is caught BEFORE any agent launches, not when a
lease is refused mid-wave — and the same preflight gate is re-checked per spawn, so
the live population still never exceeds the cap.

    python tools/issue_dispatch.py --wave --max-workers 4         # plan a wave (dry)
    python tools/issue_dispatch.py --wave --max-workers 4 --live  # spawn the wave
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except (AttributeError, ValueError):
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dispatch_worker  # noqa: E402  (sibling tool: build_command/child_env)
import fleet_accounts  # noqa: E402  (the switcher: optional setup-token read)

SCHEMA = "fleet-issue-dispatch/1"
WAVE_SCHEMA = "fleet-issue-dispatch-wave/1"
RUNS_DIRNAME = ".dispatch-runs"
USE_SETUP_TOKEN_ENV = "FLEET_CLAUDE_USE_OAUTH_TOKEN"

# An inflight marker is one {lane, pid} record written next to a worker's log the
# instant it is spawned. It is the missing CROSS-TICK de-confliction: the wave path
# already prices lanes pairwise-disjoint WITHIN one tick, but nothing stopped a
# re-ticking cron from picking the same richest lane again and stacking a second
# worker on a lane the first is still working (the witnessed "tick re-pick churn").
# busy_lanes() folds these markers (pruning dead/stale ones) so the next tick can
# prefer a lane NOT already in flight. Markers live in the gitignored .dispatch-runs/
# scratch dir; they never touch the DoS cap (that stays sourced from resolve-*.pid).
INFLIGHT_PREFIX = "inflight-"
# Far longer than any worker run: a backstop that garbage-collects a marker leaked
# by a tick that crashed before it could prune, even if the pid was since reused.
INFLIGHT_TTL_SECONDS = 12 * 3600

# Mirrors internal/dispatchtick/selfmodify.go's SelfSourceTreePrefixes on the native
# Go dispatch path. A lane whose tree touches fak's own source risks poisoning
# ``go build ./...`` for every OTHER concurrently-running agent on the shared trunk
# (#1397; #1338 cost two runs, ~52 turns, 0 commits) -- a build-poisoning risk, not
# just a tool-call-denial risk, so this is deliberately broader than the runtime
# adjudicator's own (narrower) SelfModifyGlobs deny-list.
SELF_SOURCE_TREE_PREFIXES = ("cmd/", "internal/")


def is_self_source_tree(tree: Any) -> bool:
    """True iff any glob in ``tree`` names fak's own source tree (cmd/**, internal/**)."""
    if not tree:
        return False
    for glob in tree:
        if str(glob).strip().startswith(SELF_SOURCE_TREE_PREFIXES):
            return True
    return False


def repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    return here.parent.parent


def _py() -> str:
    return sys.executable or "python"


def run_json(cmd: list[str], cwd: Path, timeout: int,
             ok_codes: set[int] | None = None) -> dict[str, Any]:
    ok_codes = ok_codes if ok_codes is not None else set(range(0, 16))
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout,
                              creationflags=dispatch_worker.no_window_creationflags())
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"_error": str(exc), "_cmd": cmd}
    out = (proc.stdout or "").strip()
    try:
        doc = json.loads(out) if out else {}
    except ValueError:
        doc = {}
        for line in reversed(out.splitlines()):
            try:
                doc = json.loads(line.strip())
                break
            except ValueError:
                continue
    if not isinstance(doc, dict):
        doc = {}
    doc.setdefault("_returncode", proc.returncode)
    return doc


def refresh_registry(root: Path) -> dict[str, Any]:
    """Re-derive the account registry from live sessions BEFORE routing.

    The switcher (``fleet_accounts route``, called inside the preflight) reads the
    cached ``tools/_registry/sessions.json`` snapshot. On an always-on tick that
    snapshot goes stale between launches, so an account that just hit a weekly
    limit — or whose org disabled Claude-Code subscription access — would still be
    handed out, the worker would spawn and instantly die, and the loop would make
    no progress (the exact failure that left ``.dispatch-runs/`` empty: the tick
    routed to a dead default account every time). Regenerating the registry
    each tick folds the current session evidence (throttle/auth/org-disabled) into
    the roster the switcher reads, so a freshly-blocked account is skipped
    automatically. A no-probe scan — cheap, read-only, no model call. Best-effort:
    a refresh failure is recorded, never fatal (the tick proceeds on the prior
    snapshot rather than refusing)."""
    doc = run_json([_py(), str(root / "tools" / "fleet_sessions.py"), "registry"],
                   root, timeout=120, ok_codes=set(range(0, 16)))
    return {"ok": "_error" not in doc, "error": doc.get("_error")}


def preflight(root: Path, *, max_workers: int, work_kind: str,
              product: str = "claude") -> dict[str, Any]:
    return run_json([_py(), str(root / "tools" / "dispatch_preflight.py"), "--json",
                     "--max-workers", str(max_workers), "--work-kind", work_kind,
                     "--product", product],
                    root, timeout=120)


def pick_lane(root: Path, explicit: str | None,
              busy: set[str] | None = None,
              guarded: bool | None = None) -> dict[str, Any]:
    """The lane with the most open issues, or an explicit override.

    When ``busy`` names lanes that already have a live dispatched worker (from
    ``busy_lanes``), the richest lane NOT already in flight is preferred, so a
    re-ticking cron spreads across the backlog instead of stacking a second worker
    on the same lane (the "tick re-pick churn" failure). If EVERY lane with open
    issues is busy, it falls back to the richest overall and sets ``stacked`` so the
    stack is surfaced, never silent — the DoS cap, not this hint, is what bounds the
    live population, so falling back keeps throughput when no free lane is left.
    An explicit lane is honored verbatim (operator intent overrides the spread).

    ``guarded`` (default: ``dispatch_worker.guard_enabled()``) additionally EXCLUDES
    a self-source lane (``is_self_source_tree``, cmd/**, internal/**) from the
    automatic pool -- proactively, before any worker is spawned. This mirrors the
    native Go dispatch path's ``SELF_MODIFY_HOLD``; this legacy Python path
    previously had only REACTIVE, post-hoc detection (``NO_COMMIT_SELF_MODIFY`` in
    issue_resolve_dispatch.py, discovered from a worker's session log tail AFTER it
    already burned turns). Unlike the busy-lane fallback, self-source lanes are a
    HARD exclude even when every other lane is busy — falling back to one would spawn
    the exact build-poisoning risk the guard exists to prevent — so if every lane
    with open issues is self-source-held, ``lane`` comes back ``None`` and the held
    lanes are named in ``self_modify_held``. An explicit lane is still honored
    verbatim regardless of guard state (operator intent overrides the guard, same as
    the Go path)."""
    busy = busy or set()
    guarded = dispatch_worker.guard_enabled() if guarded is None else guarded
    router = run_json([_py(), str(root / "tools" / "issue_lane_router.py"), "--json"],
                      root, timeout=130)
    lanes = router.get("lanes") or {}
    counts = {}
    trees = {}
    for ln, info in lanes.items():
        iss = info.get("issues") if isinstance(info, dict) else info
        counts[ln] = len(iss) if hasattr(iss, "__len__") else 0
        trees[ln] = info.get("tree") if isinstance(info, dict) else None
    if explicit:
        return {"lane": explicit, "issues": counts.get(explicit, 0), "by_lane": counts,
                "explicit": True, "busy": sorted(busy),
                "router_error": router.get("_error")}
    if not counts:
        return {"lane": None, "issues": 0, "by_lane": {}, "busy": sorted(busy),
                "router_error": router.get("_error")}
    self_source = ({ln for ln in counts if is_self_source_tree(trees.get(ln))}
                   if guarded else set())
    held = sorted(ln for ln in counts if ln in self_source and counts[ln] > 0)
    dispatchable = {ln: n for ln, n in counts.items() if ln not in self_source}
    if not dispatchable:
        return {"lane": None, "issues": 0, "by_lane": counts, "busy": sorted(busy),
                "self_modify_held": held, "router_error": router.get("_error")}
    free = {ln: n for ln, n in dispatchable.items() if ln not in busy}
    pool = free or dispatchable
    stacked = not free   # every dispatchable lane with open issues is already being worked
    # Highest open-issue count first; lane name as a stable lexicographic tiebreak so
    # the pick is deterministic across ticks rather than dependent on router order.
    lane = sorted(pool, key=lambda k: (-pool[k], k))[0]
    return {"lane": lane, "issues": counts[lane], "by_lane": counts,
            "busy": sorted(busy), "stacked": stacked,
            "self_modify_held": held, "router_error": router.get("_error")}


def _pid_is_alive(pid: int) -> bool:
    """Cross-platform live-PID check, mirrored from dispatch_preflight._pid_is_alive.

    ``os.kill(pid, 0)`` *terminates* a process on Windows, so the nt branch shells to
    Get-Process instead. Mirrored rather than imported: pulling dispatch_preflight in
    for one helper would drag its whole shell-out surface into the tick."""
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            proc = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                 f"Get-Process -Id {int(pid)} -ErrorAction SilentlyContinue | "
                 "Select-Object -First 1 -ExpandProperty Id"],
                capture_output=True, text=True, timeout=5,
                creationflags=dispatch_worker.no_window_creationflags())
            return proc.returncode == 0 and bool((proc.stdout or "").strip())
        except (OSError, subprocess.TimeoutExpired):
            return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def _write_inflight_marker(runs_dir: Path, lane: str, pid: int) -> str | None:
    """Record {lane, pid} so a LATER tick sees this lane is in flight and spreads to
    a different one. Best-effort: a write failure never blocks the spawn."""
    try:
        runs_dir.mkdir(parents=True, exist_ok=True)
        path = runs_dir / f"{INFLIGHT_PREFIX}{lane}-{int(pid)}.json"
        path.write_text(
            json.dumps({"lane": lane, "pid": int(pid),
                        "stamp": dt.datetime.now(dt.timezone.utc).isoformat()}),
            encoding="utf-8")
        return str(path)
    except (OSError, ValueError):
        return None


def busy_lanes(runs_dir: Path, *, is_alive: Callable[[int], bool] | None = None,
               now: float | None = None,
               ttl_seconds: int = INFLIGHT_TTL_SECONDS) -> set[str]:
    """The set of lanes with a currently-live dispatched worker, folded from the
    inflight markers ``spawn_detached`` writes. Self-healing: a marker whose pid is
    dead, whose record is unreadable, or whose file is older than ``ttl_seconds`` is
    pruned in the same pass, so the marker set stays bounded without a separate
    sweeper. ``is_alive`` / ``now`` are injectable so the fold is hermetically
    testable without real processes or a real clock."""
    alive = is_alive or _pid_is_alive
    if not runs_dir.is_dir():
        return set()
    when = now if now is not None else time.time()
    lanes: set[str] = set()
    for marker in sorted(runs_dir.glob(f"{INFLIGHT_PREFIX}*.json")):
        try:
            rec = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _safe_unlink(marker)
            continue
        lane = str((rec or {}).get("lane") or "").strip()
        raw_pid = (rec or {}).get("pid")
        pid = int(raw_pid) if isinstance(raw_pid, int) else (
            int(raw_pid) if isinstance(raw_pid, str) and raw_pid.strip().isdigit() else 0)
        try:
            stale = (when - marker.stat().st_mtime) > ttl_seconds
        except OSError:
            stale = True
        if not lane or pid <= 0 or stale or not alive(pid):
            _safe_unlink(marker)
            continue
        lanes.add(lane)
    return lanes


def _truthy_env(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def worker_env(account_dir: str | None, lane: str, workspace: Path) -> dict[str, str]:
    """Child env: the switcher account pinned, self-describing dispatch vars,
    and the benchmark-witness hint."""
    env = dispatch_worker.child_env(lane, "claude", workspace)
    if account_dir:
        env["CLAUDE_CONFIG_DIR"] = account_dir
        # Match account_probe.py: validate and launch against the account directory.
        # A stale ambient/setup token can belong to another account or org and turn a
        # healthy config dir into an immediate ACCESS wall, so clear it by default.
        env.pop("CLAUDE_CODE_OAUTH_TOKEN", None)
        if _truthy_env(env.get(USE_SETUP_TOKEN_ENV)):
            tok = fleet_accounts.read_oauth_token(account_dir)
            if tok:
                env["CLAUDE_CODE_OAUTH_TOKEN"] = tok
    # The witness for this loop is the benchmark, not the unit-test suite.
    env["FLEET_DISPATCH_WITNESS"] = "benchmark"
    env["FLEET_BENCH_WITNESS_CMD"] = f"python tools/bench_witness.py --lane {lane}"
    # Arm the DOS verdict-journal auto-emit (#465) on the *dispatch* surface, NOT the
    # session surface. The kernel's verdict-journal is append-only and NOT auto-rotated
    # (dos verdict_journal.py: "grows unbounded on a long-lived fleet"; the
    # [retention] audits_keep_last cap does not fold over it). So arming it via
    # settings.json `env` — which fires on every idle Claude Code session — would
    # violate this issue's own "retention bounded" acceptance. Arming it here instead
    # bounds growth to actual dispatched issue-resolution runs: a worker's verify/recall
    # adjudications land in .dos/verdict-journal.jsonl while it works, an idle session
    # writes nothing, and the journal rides the existing .dos/ backup story.
    env["DISPATCH_OBSERVE"] = "1"
    return env


def spawn_detached(command: list[str], env: dict[str, str], cwd: Path,
                   log_dir: Path, lane: str) -> dict[str, Any]:
    """Launch the worker DETACHED so it outlives this tick; log to a dated file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_log = log_dir / f"dispatch-{lane}-{stamp}.log"
    exe = shutil.which(command[0]) or command[0]
    argv = [exe, *command[1:]]
    kwargs: dict[str, Any] = {}
    if os.name == "nt":
        # CREATE_NO_WINDOW, not DETACHED_PROCESS: a worker with NO console
        # (DETACHED_PROCESS) makes every console tool it spawns — git, gh, fak, the
        # shell — pop its OWN visible window. CREATE_NO_WINDOW gives the worker one
        # HIDDEN console the whole subtree inherits: it still outlives this tick, but
        # no popup ever flashes. (Matches claude_agent_chat.detached_creationflags.)
        kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
    else:
        kwargs["start_new_session"] = True
    fh = open(out_log, "w", encoding="utf-8")
    proc = subprocess.Popen(argv, cwd=str(cwd), env=env, stdin=subprocess.DEVNULL,
                            stdout=fh, stderr=subprocess.STDOUT, **kwargs)
    # Stamp this lane as in flight so a later tick spreads off it (cross-tick
    # de-confliction). Best-effort and pid-keyed; busy_lanes prunes it on death.
    marker = _write_inflight_marker(log_dir, lane, proc.pid)
    return {"pid": proc.pid, "log": str(out_log), "inflight": marker}


def _build_launch(root: Path, lane: str | None) -> tuple[list[str], list[str], bool]:
    """The raw agent argv + the (kernel-fronted) launch argv for one lane.

    Dogfood: front the worker with the kernel (``fak guard``) so every tool call it
    proposes crosses the capability floor and lands in a durable, hash-chained
    decision journal. ``command`` stays the raw agent argv (the logical worker
    command); ``launch_command`` is what actually spawns (kernel-fronted when a fak
    binary resolves and FLEET_DOGFOOD_GUARD!=0; unchanged otherwise -- fail open).
    Shared by the single-tick and wave spawn paths so both front the same kernel."""
    command = dispatch_worker.build_command(lane, "claude") if lane else []
    launch_command, guarded = (
        dispatch_worker.guarded_launch_command(command, lane, "claude", root)
        if command else ([], False)
    )
    return command, launch_command, guarded


def evaluate(root: Path, *, max_workers: int, work_kind: str, lane: str | None,
             live: bool, refresh: bool = True) -> dict[str, Any]:
    # Refresh the account registry from live sessions FIRST, so the switcher routes
    # off current evidence (a freshly weekly-limited / org-disabled account is
    # skipped, not handed out to a worker that would instantly die). Skippable for
    # tests/inspection via refresh=False.
    reg = refresh_registry(root) if refresh else {"ok": None, "skipped": True}
    pre = preflight(root, max_workers=max_workers, work_kind=work_kind)
    pre_ok = pre.get("verdict") == "SPAWN_OK"
    acct = pre.get("account") or {}
    # Lanes already in flight from a prior tick: prefer a different one so the cron
    # spreads across the backlog rather than re-picking the richest lane every tick.
    busy = busy_lanes(root / RUNS_DIRNAME)
    lane_pick = pick_lane(root, lane, busy=busy)
    chosen = lane_pick.get("lane")
    command, launch_command, guarded = _build_launch(root, chosen)

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "workspace": str(root),
        "live": live,
        "max_workers": max_workers,
        "registry_refresh": reg,
        "preflight": {"verdict": pre.get("verdict"), "reason": pre.get("reason"),
                      "cap": pre.get("cap"), "live": pre.get("live"),
                      # host_cap is the host-derived ADAPTIVE ceiling (#1337): when it
                      # is the binding term the cap tracks live host headroom (cores,
                      # free RAM, OS-thread total) rather than a static number, so a
                      # loaded box auto-throttles and recovers as load clears. Surface
                      # it in the dispatcher's own telemetry so "live population tracks
                      # host_cap" is observable here, not only inside the preflight
                      # reason string.
                      "host_cap": pre.get("host_cap")},
        "account": {k: acct.get(k) for k in ("tag", "tier", "model", "dir")},
        "lane": chosen,
        "lane_issue_count": lane_pick.get("issues"),
        "busy_lanes": sorted(busy),
        "lane_stacked": bool(lane_pick.get("stacked")),
        "self_modify_held": lane_pick.get("self_modify_held") or [],
        "command": command,
        "guarded": guarded,
        "launch_command": launch_command,
        "witness": {"kind": "benchmark",
                    "cmd": f"python tools/bench_witness.py --lane {chosen}" if chosen else None},
    }

    if not pre_ok:
        payload.update({"ok": False, "action": "refused",
                        "verdict": pre.get("verdict") or "REFUSE",
                        "reason": f"preflight refused: {pre.get('reason')}"})
        return payload
    if not chosen:
        held = lane_pick.get("self_modify_held") or []
        if held:
            payload.update({"ok": False, "action": "no_lane", "verdict": "SELF_MODIFY_HOLD",
                            "reason": (f"every lane with open issues is self-modify held "
                                       f"under guard ({', '.join(held)}) -- worktree "
                                       f"isolation (#1334) is needed before these can be "
                                       f"safely auto-dispatched")})
        else:
            payload.update({"ok": False, "action": "no_lane", "verdict": "NO_LANE",
                            "reason": "no lane has open issues (router empty/error)"})
        return payload
    if not live:
        payload.update({"ok": True, "action": "would_spawn", "verdict": "WOULD_SPAWN",
                        "reason": (f"safe to spawn 1 worker on lane '{chosen}' "
                                   f"({lane_pick.get('issues')} issues) under account "
                                   f"'{acct.get('tag')}' (t{acct.get('tier')})")})
        return payload

    env = worker_env(acct.get("dir"), chosen, root)
    if guarded:
        dispatch_worker.guard_env_augment(env)
    spawned = spawn_detached(launch_command, env, root, root / RUNS_DIRNAME, chosen)
    payload.update({"ok": True, "action": "spawned", "verdict": "SPAWNED",
                    "spawned": spawned,
                    "reason": (f"spawned worker pid {spawned['pid']} on lane '{chosen}' "
                               f"under account '{acct.get('tag')}'")})
    _record(root / RUNS_DIRNAME, payload)
    return payload


def _record(runs_dir: Path, payload: dict[str, Any]) -> None:
    try:
        runs_dir.mkdir(parents=True, exist_ok=True)
        (runs_dir / "last-tick.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        pass


def render(p: dict[str, Any]) -> str:
    a = p.get("account") or {}
    pf = p.get("preflight") or {}
    lines = [
        f"issue-dispatch: {p.get('verdict')} ({'ok' if p.get('ok') else 'refuse'})  live={p.get('live')}",
        f"  preflight : {pf.get('verdict')} ({pf.get('live')}/{pf.get('cap')} live"
        + (f", host_cap {pf.get('host_cap')}" if pf.get('host_cap') is not None else "")
        + ")",
        f"  account   : {a.get('tag') or '-'} (t{a.get('tier')})  {a.get('model') or ''}",
        f"  lane      : {p.get('lane') or '-'}  ({p.get('lane_issue_count')} issues)"
        + (f"  [busy: {', '.join(p.get('busy_lanes'))}]" if p.get('busy_lanes') else "")
        + ("  STACKED (all lanes in flight)" if p.get('lane_stacked') else ""),
        f"  witness   : {(p.get('witness') or {}).get('cmd') or '-'}",
        f"  command   : {' '.join(p.get('command') or []) or '-'}",
        f"  -> {p.get('reason')}",
    ]
    if p.get("spawned"):
        lines.append(f"  spawned pid={p['spawned'].get('pid')} log={p['spawned'].get('log')}")
    return "\n".join(lines)


# --- WAVE: spawn K disjoint-lane workers in one tick (#1335) ----------------
# The single tick above picks ONE lane and spawns ONE worker. A wave fans that out
# to K workers per tick across pairwise TREE-DISJOINT lanes so they neither collide
# nor serialize on a shared lane lease — the "best effort up to K" shape (#1333). It
# adds two guarantees the single tick does not: the partition is PRICED (each lane
# arbitrated against the wave's already-admitted leases, so a colliding set is caught
# BEFORE any agent launches, not when a lease is refused mid-wave) and seated (each
# worker draws its own distinct account pool). The DoS bound is unchanged: the same
# dispatch_preflight gate is re-checked per spawn, so the live population still never
# exceeds the cap.


def _dos_cmd() -> list[str]:
    """The ``dos`` kernel CLI as an argv prefix: the installed console script when on
    PATH, else the module form so a venv without the script still arbitrates."""
    exe = shutil.which("dos")
    return [exe] if exe else [_py(), "-m", "dos.cli"]


def lane_candidates(root: Path, guarded: bool | None = None) -> dict[str, Any]:
    """Candidate lanes for a wave: every lane the router reports with at least one
    open issue, ordered by open-issue count (richest first, lane name as tiebreak),
    each carrying its canonical file ``tree`` so the partition can be priced for
    disjointness.

    ``guarded`` (default: ``dispatch_worker.guard_enabled()``) proactively drops a
    self-source lane (``is_self_source_tree``, cmd/**, internal/**) from the
    candidate list before it ever reaches ``dos arbitrate`` or a seat — same rule and
    same reasoning as ``pick_lane``'s single-tick hold. Held lanes are named in
    ``self_modify_held`` for the wave payload's audit trail."""
    guarded = dispatch_worker.guard_enabled() if guarded is None else guarded
    router = run_json([_py(), str(root / "tools" / "issue_lane_router.py"), "--json"],
                      root, timeout=130)
    lanes = router.get("lanes") or {}
    cands: list[dict[str, Any]] = []
    held: list[str] = []
    for ln, info in lanes.items():
        if isinstance(info, dict):
            iss, tree = info.get("issues"), info.get("tree") or []
        else:
            iss, tree = info, []
        n = len(iss) if hasattr(iss, "__len__") else 0
        if n <= 0:
            continue
        if guarded and is_self_source_tree(tree):
            held.append(ln)
            continue
        cands.append({"lane": ln, "issues": n, "tree": list(tree)})
    cands.sort(key=lambda c: (-c["issues"], c["lane"]))
    return {"candidates": cands, "self_modify_held": sorted(held),
            "router_error": router.get("_error")}


def arbitrate_lane(root: Path, lane: str, tree: list[str],
                   leases: list[dict[str, Any]]) -> dict[str, Any]:
    """Price ONE lane against the wave's already-admitted leases via ``dos
    arbitrate``. The kernel ACQUIRES a lane iff its file tree is disjoint from every
    lease in ``leases``; when the requested tree collides it AUTO-PICKS a different
    free lane instead of refusing, so a redirect (``auto_picked`` set, or a returned
    lane != the requested one) is the collision signal. Admitted == the kernel
    granted the REQUESTED lane. Read-only: a decision, not a held lease — the spawned
    worker materializes its own lane lease on start (the dos-dispatch-loop skill),
    while this priced partition guarantees the K trees are pairwise-disjoint."""
    cmd = [*_dos_cmd(), "arbitrate", "--workspace", str(root), "--lane", lane,
           "--kind", "cluster", "--leases", json.dumps(leases), "--output", "json"]
    if tree:
        cmd += ["--tree", *tree]
    doc = run_json(cmd, root, timeout=90)
    got = doc.get("lane")
    admitted = (doc.get("outcome") == "acquire" and not doc.get("auto_picked")
                and got == lane)
    return {"admitted": admitted, "outcome": doc.get("outcome"), "got": got,
            "auto_picked": bool(doc.get("auto_picked")),
            "tree": doc.get("tree") or list(tree),
            "reason": doc.get("reason"), "error": doc.get("_error")}


def allocate_seats(root: Path, max_workers: int, work_kind: str) -> dict[str, Any]:
    """The seat budget for a wave: up to ``max_workers`` DISTINCT account pools, so
    each worker draws on its own rate-limit pool instead of re-serializing on one.
    Delegates to fleet_accounts.allocate_wave (the shipped wave-allocation
    primitive); a granted lane carries config_dir/tag/pool + a rank-stamped
    membership. Fail-open: a seat-allocation failure resolves to zero seats (the wave
    refuses), never an exception that wedges the tick."""
    try:
        return fleet_accounts.allocate_wave(max_workers, work_kind=work_kind,
                                            product="claude")
    except Exception as exc:  # noqa: BLE001 — fail-open boundary: no seats, never fatal
        return {"ok": False, "granted": 0, "lanes": [], "wave_id": None,
                "error": str(exc)}


def _wave_env(rank: int, wave_id: str, size: int, shortfall: int) -> dict[str, str]:
    """Stamp a worker's place in its wave (rank/id/size/shortfall) — the same
    ``FLEET_WAVE_*`` convention issue_resolve_dispatch writes, so an auditor reads one
    grammar across both spawn paths. Labels an independent detached worker; grants no
    collective (no barrier/gather) — a wave stays N lanes whose only shared fabric is
    git + the dos arbitrate lease."""
    return {"FLEET_WAVE_ID": str(wave_id), "FLEET_WAVE_RANK": str(int(rank)),
            "FLEET_WAVE_SIZE": str(int(size)),
            "FLEET_WAVE_SHORTFALL": str(int(shortfall))}


def _spawn_wave_member(root: Path, lane: str, seat: dict[str, Any], wave_id: str,
                       rank: int, size: int, shortfall: int) -> dict[str, Any]:
    """Launch one wave worker on its seat, stamped with its wave membership. Writes a
    per-worker ``.wave`` sidecar next to the log so the whole wave is enumerable from
    disk, never from a worker's self-report."""
    command, launch_command, guarded = _build_launch(root, lane)
    if not launch_command:
        return {"error": f"no command for lane '{lane}'"}
    env = worker_env(seat.get("config_dir"), lane, root)
    env.update(_wave_env(rank, wave_id, size, shortfall))
    if guarded:
        dispatch_worker.guard_env_augment(env)
    spawned = spawn_detached(launch_command, env, root, root / RUNS_DIRNAME, lane)
    spawned["guarded"] = guarded
    try:
        Path(spawned["log"]).with_suffix(".wave").write_text(
            json.dumps({"wave_id": wave_id, "rank": rank, "size": size,
                        "shortfall": shortfall}, sort_keys=True), encoding="utf-8")
    except OSError:
        pass
    return spawned


def _write_wave_artifacts(runs_dir: Path, payload: dict[str, Any]) -> None:
    """Write the wave-level sidecar the done-condition names — ``{wave_id, size,
    lanes, seats}`` — plus a full ``last-wave.json`` for inspection. The sidecar is
    the contract artifact: one record per tick enumerable from disk alongside the
    per-worker ``.wave`` membership stamps."""
    try:
        runs_dir.mkdir(parents=True, exist_ok=True)
        sidecar = {"wave_id": payload.get("wave_id"), "size": payload.get("size"),
                   "lanes": payload.get("lanes"), "seats": payload.get("seats_used")}
        (runs_dir / f"dispatch-wave-{payload.get('wave_id')}.json").write_text(
            json.dumps(sidecar, indent=2), encoding="utf-8")
        (runs_dir / "last-wave.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        pass


def evaluate_wave(root: Path, *, max_workers: int, work_kind: str, live: bool,
                  refresh: bool = True) -> dict[str, Any]:
    """One WAVE tick: spawn up to ``max_workers`` workers across pairwise
    tree-disjoint lanes in a single tick, each on its own seat, never exceeding the
    dispatch_preflight cap.

    The wave size is ``min(free_seats, free_lanes, preflight_headroom)`` discovered
    ONLINE: candidate lanes are taken richest-first; each is PRICED against the wave's
    already-admitted leases via ``dos arbitrate`` (a colliding lane is skipped BEFORE
    any agent launches); each admitted lane draws the next distinct seat; and
    ``dispatch_preflight`` is re-checked per spawn so the live population provably
    never exceeds the cap. A wave sidecar records ``{wave_id, size, lanes, seats}``."""
    reg = refresh_registry(root) if refresh else {"ok": None, "skipped": True}
    seats = allocate_seats(root, max_workers, work_kind)
    seat_lanes = seats.get("lanes") or []
    free_seats = len(seat_lanes)
    wave_id = seats.get("wave_id") or "wave-unallocated"

    cand = lane_candidates(root)
    candidates = cand.get("candidates") or []
    # Lanes a prior tick's worker still holds: skipped here so a wave never re-stacks
    # a lane already in flight (the within-tick arbiter only de-conflicts THIS wave's
    # own picks; busy_lanes carries the de-confliction ACROSS ticks).
    busy = busy_lanes(root / RUNS_DIRNAME)

    leases: list[dict[str, Any]] = []   # accumulating disjoint-tree leases (priced)
    members: list[dict[str, Any]] = []
    skipped_busy: list[str] = []
    baseline_live: int | None = None
    cap_seen: int | None = None
    refusal: str | None = None

    payload: dict[str, Any] = {
        "schema": WAVE_SCHEMA,
        "workspace": str(root),
        "live": live,
        "max_workers": max_workers,
        "wave_id": wave_id,
        "registry_refresh": reg,
        "free_seats": free_seats,
        "seats": {"granted": seats.get("granted"), "requested": seats.get("requested"),
                  "shortfall": seats.get("shortfall"), "wave_id": seats.get("wave_id"),
                  "tags": [s.get("tag") for s in seat_lanes], "error": seats.get("error")},
        "candidate_lanes": [c["lane"] for c in candidates],
        "busy_lanes": sorted(busy),
        "self_modify_held": cand.get("self_modify_held") or [],
        "router_error": cand.get("router_error"),
    }

    for c in candidates:
        if len(members) >= max_workers:
            break
        if len(members) >= free_seats:
            refusal = refusal or "SEATS_EXHAUSTED"
            break
        # Cross-tick de-confliction: a lane a prior tick's worker still holds is
        # skipped before it costs a preflight re-check or a seat.
        if c["lane"] in busy:
            skipped_busy.append(c["lane"])
            continue
        # Per-spawn preflight re-check: the live population must never exceed the cap.
        pre = preflight(root, max_workers=max_workers, work_kind=work_kind)
        if pre.get("verdict") != "SPAWN_OK":
            refusal = pre.get("verdict") or "REFUSE"
            break
        cap = pre.get("cap")
        cap_seen = cap if isinstance(cap, int) else cap_seen
        live_now = pre.get("live") if isinstance(pre.get("live"), int) else 0
        if baseline_live is None:
            baseline_live = live_now
        # Defeat OS-scan lag: count our own in-tick spawns even before the scan sees
        # them, so the bound holds WITHIN the tick. effective = max(scan, base+spawned).
        effective_live = max(live_now, baseline_live + len(members))
        if isinstance(cap, int) and effective_live >= cap:
            refusal = "REFUSE_AT_CAP"
            break
        # Price this lane against the wave's already-admitted leases.
        dec = arbitrate_lane(root, c["lane"], c["tree"], leases)
        if not dec.get("admitted"):
            continue   # collides with an admitted lane (or kernel error) -> skip
        rank = len(members)
        seat = seat_lanes[rank]
        member: dict[str, Any] = {
            "lane": c["lane"], "tree": dec["tree"], "issues": c["issues"], "rank": rank,
            "account": {"tag": seat.get("tag"), "tier": seat.get("model_tier"),
                        "pool": seat.get("pool"), "dir": seat.get("config_dir")},
            "arbitrate": dec.get("reason"),
        }
        if live:
            member["spawned"] = _spawn_wave_member(
                root, c["lane"], seat, wave_id, rank, free_seats,
                int(seats.get("shortfall") or 0))
        members.append(member)
        leases.append({"lane": c["lane"], "lane_kind": "cluster", "tree": dec["tree"]})

    size = len(members)
    lanes_used = [m["lane"] for m in members]
    seats_used = [(m["account"] or {}).get("tag") for m in members]
    payload.update({"size": size, "lanes": lanes_used, "members": members,
                    "seats_used": seats_used, "cap": cap_seen, "refusal": refusal,
                    "skipped_busy": skipped_busy})

    if size > 0:
        payload.update({
            "ok": True,
            "verdict": "WAVED" if live else "WOULD_WAVE",
            "action": "waved" if live else "would_wave",
            "reason": (f"{'spawned' if live else 'would spawn'} {size} worker(s) across "
                       f"pairwise-disjoint lanes {lanes_used} (wave {wave_id})"
                       + (f"; stopped on {refusal}" if refusal else ""))})
        if live:
            _write_wave_artifacts(root / RUNS_DIRNAME, payload)
    elif not candidates:
        held = cand.get("self_modify_held") or []
        if held:
            payload.update({"ok": False, "verdict": "SELF_MODIFY_HOLD", "action": "no_lane",
                            "reason": (f"every lane with open issues is self-modify held "
                                       f"under guard ({', '.join(held)}) -- worktree "
                                       f"isolation (#1334) is needed before these can be "
                                       f"safely auto-dispatched")})
        else:
            payload.update({"ok": False, "verdict": "WAVE_NO_LANE", "action": "no_lane",
                            "reason": "no lane has open issues (router empty/error)"})
    elif free_seats == 0:
        payload.update({"ok": False, "verdict": "WAVE_NO_SEATS", "action": "no_seats",
                        "reason": (f"no free seats for a wave "
                                   f"({seats.get('reason') or seats.get('error')})")})
    else:
        payload.update({"ok": False, "verdict": refusal or "WAVE_EMPTY",
                        "action": "refused",
                        "reason": (f"no worker spawned (stopped on {refusal})" if refusal
                                   else "no candidate lane was admissible (all collided)")})
    return payload


def render_wave(p: dict[str, Any]) -> str:
    seats = (p.get("seats") or {}).get("tags") or []
    lines = [
        f"issue-dispatch WAVE: {p.get('verdict')} ({'ok' if p.get('ok') else 'refuse'})  live={p.get('live')}",
        f"  wave_id   : {p.get('wave_id')}  size={p.get('size')}  cap={p.get('cap')}",
        f"  seats     : {p.get('free_seats')} free  ({', '.join(t for t in seats if t) or '-'})",
        f"  candidates: {len(p.get('candidate_lanes') or [])} lane(s) with open issues",
    ]
    if p.get("busy_lanes"):
        lines.append(f"  busy      : {', '.join(p.get('busy_lanes'))} (in flight; skipped)")
    for m in p.get("members") or []:
        sp = m.get("spawned") or {}
        tag = (m.get("account") or {}).get("tag") or "-"
        pid = f" pid={sp.get('pid')}" if sp.get("pid") else ""
        lines.append(f"    [{m.get('rank')}] {str(m.get('lane') or '-'):<12} "
                     f"{m.get('issues')} issues  seat={tag}{pid}")
    if p.get("refusal"):
        lines.append(f"  stopped   : {p.get('refusal')}")
    lines.append(f"  -> {p.get('reason')}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="One guarded, switcher-routed, bounded dispatch tick (dry-run by default).")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--max-workers", type=int, default=2,
                    help="hard cap on live workers, enforced by the preflight (default: 2)")
    ap.add_argument("--work-kind", default="engineering",
                    help="switcher work kind (engineering->t1, gardening->t2)")
    ap.add_argument("--lane", default=None,
                    help="explicit lane (default: the lane with the most open issues)")
    ap.add_argument("--wave", action="store_true",
                    help="WAVE mode (#1335): spawn up to --max-workers workers in ONE "
                         "tick across pairwise tree-disjoint lanes (priced + arbitrated), "
                         "each on its own seat; preflight re-checked per spawn")
    ap.add_argument("--live", action="store_true",
                    help="actually spawn the worker (default: dry-run / plan only)")
    ap.add_argument("--no-refresh", action="store_true",
                    help="skip the per-tick account-registry refresh (route off the "
                         "cached snapshot; for inspection / when a fresh scan just ran)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args(argv)

    root = Path(args.workspace).resolve() if args.workspace else repo_root()
    if args.wave:
        payload = evaluate_wave(root, max_workers=args.max_workers,
                                work_kind=args.work_kind, live=args.live,
                                refresh=not args.no_refresh)
        print(json.dumps(payload, indent=2) if args.json else render_wave(payload))
        return 0 if payload.get("ok") else 1
    payload = evaluate(root, max_workers=args.max_workers, work_kind=args.work_kind,
                       lane=args.lane, live=args.live, refresh=not args.no_refresh)
    print(json.dumps(payload, indent=2) if args.json else render(payload))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
