#!/usr/bin/env python3
"""Dry-run-by-default watchdog for fleet's generic DOS supervisor.

This is the operator gate between the read-only readiness card and a real
worker dispatch tick. By default it only reports the exact bounded command
it would run. Passing ``--live`` is the only path that can launch workers.

For a canary the watchdog invokes the worker backend DIRECTLY for a single bounded
tick -- the compiled Go launcher tools/.bin/dispatchworker when built (no interpreter
spawn), else tools/dispatch_worker.py via this process's own sys.executable (portable
to python3-only nodes) -- instead of routing through ``dos loop --enact``. The
--enact/--max-ticks flags still exist on ``dos loop`` (that continuous population
supervisor is kept alive separately by fleet_dos_dispatch_watchdog.{py,ps1}); this
is a launch-shape choice, not a claim that the flags were removed.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import dos_supervisor_status as status

SCHEMA = "fleet-dos-supervisor-watchdog/1"

ENACTABLE_VERDICTS = {"READY_TO_CANARY"}
NOOP_VERDICTS = {"AT_TARGET", "READY"}


Runner = Callable[[list[str], Path, int], dict[str, Any]]


def go_worker_binary(workspace: Path) -> Path | None:
    """The compiled Go dispatch-worker (tools/.bin/dispatchworker[.exe]) if built.

    Preferred over the Python launcher so the canary spawns NO interpreter -- and,
    being interpreter-free, it can't ENOENT on a python3-only node the way a bare
    `python` token does (#22). `make build` drops it there. Returns None when not
    built, so the caller falls back to dispatch_worker.py. The launched worker's
    `claude -p ... /dos-kernel:dos-dispatch-loop --lane X` command line is identical
    either way, so dispatch_preflight's `dos-dispatch-loop` worker count is unchanged
    (worker detection is launcher-agnostic; the cutover does not blind it)."""
    base = Path(workspace) / "tools" / ".bin"
    for name in ("dispatchworker.exe", "dispatchworker"):
        p = base / name
        if p.exists():
            return p
    return None


def enact_command(workspace: Path, target: int, max_ticks: int, lane: str | None = None) -> list[str]:
    """Return the worker dispatch command for a canary tick.

    The supervisor launches the worker backend directly for a single bounded canary,
    rather than driving `dos loop --enact` -- a launch-shape choice (NOT because
    --enact/--max-ticks were removed; they still exist on `dos loop`). The continuous
    `dos loop --enact` supervisor is kept alive by fleet_dos_dispatch_watchdog.

    Backend launcher: prefer the compiled Go binary (tools/.bin/dispatchworker), which
    spawns no interpreter; fall back to dispatch_worker.py via THIS process's own
    interpreter (sys.executable, never a bare "python" -- that ENOENTs on python3-only
    nodes) when the binary is not built.

    Args:
        workspace: The workspace root
        target: Target worker count (unused for single canary, kept for interface)
        max_ticks: Max iterations (unused, kept for interface)
        lane: The specific lane to dispatch on (from spawn list)

    Returns:
        The command list to execute.
    """
    go_bin = go_worker_binary(workspace)
    if go_bin is not None:
        cmd = [str(go_bin), "--workspace", str(workspace)]
    else:
        cmd = [sys.executable, str(workspace / "tools" / "dispatch_worker.py")]

    # Add lane if specified
    if lane:
        cmd.extend(["--lane", lane])

    return cmd


def build_plan(
    readiness: dict[str, Any],
    *,
    workspace: Path,
    target: int,
    max_ticks: int,
    live: bool,
    safety: dict[str, Any] | None = None,
) -> dict[str, Any]:
    verdict = str(readiness.get("verdict") or "")
    supervise = readiness.get("supervise") or {}
    spawn = [str(lane) for lane in supervise.get("spawn") or []]

    # Build the command with the first available spawn lane
    lane = spawn[0] if spawn else None
    command = enact_command(workspace, target, max_ticks, lane=lane)

    if not readiness.get("ok"):
        return _plan(
            readiness=readiness,
            workspace=workspace,
            target=target,
            max_ticks=max_ticks,
            live=live,
            action="refuse",
            ok=False,
            reason=f"readiness verdict blocks launch: {verdict or 'unknown'}",
            command=command,
            safety=safety,
        )
    if verdict in NOOP_VERDICTS or not spawn:
        return _plan(
            readiness=readiness,
            workspace=workspace,
            target=target,
            max_ticks=max_ticks,
            live=live,
            action="noop",
            ok=True,
            reason=f"no supervisor tick needed for readiness verdict: {verdict or 'unknown'}",
            command=command,
            safety=safety,
        )
    if verdict not in ENACTABLE_VERDICTS:
        return _plan(
            readiness=readiness,
            workspace=workspace,
            target=target,
            max_ticks=max_ticks,
            live=live,
            action="refuse",
            ok=False,
            reason=f"readiness verdict is not an enactable canary state: {verdict or 'unknown'}",
            command=command,
            safety=safety,
        )

    return _plan(
        readiness=readiness,
        workspace=workspace,
        target=target,
        max_ticks=max_ticks,
        live=live,
        action="enact" if live else "would_enact",
        ok=True,
        reason=f"bounded canary available for {len(spawn)} spawn lane(s)",
        command=command,
        safety=safety,
    )


def _plan(
    *,
    readiness: dict[str, Any],
    workspace: Path,
    target: int,
    max_ticks: int,
    live: bool,
    action: str,
    ok: bool,
    reason: str,
    command: list[str],
    safety: dict[str, Any] | None = None,
) -> dict[str, Any]:
    supervise = readiness.get("supervise") or {}
    return {
        "schema": SCHEMA,
        "ok": ok,
        "action": action,
        "reason": reason,
        "workspace": str(workspace),
        "live": live,
        "target": target,
        "max_ticks": max_ticks,
        "command": command,
        "safety": safety or {"ok": True, "blockers": []},
        "readiness": {
            "schema": readiness.get("schema"),
            "ok": readiness.get("ok"),
            "verdict": readiness.get("verdict"),
            "why": readiness.get("why"),
            "next_action": readiness.get("next_action"),
            "spawn": supervise.get("spawn") or [],
            "alive": supervise.get("alive"),
            "target": supervise.get("target"),
        },
    }


def run_watchdog(
    *,
    workspace: Path,
    target: int | None,
    max_ticks: int,
    live: bool,
    timeout_s: int,
    readiness: dict[str, Any] | None = None,
    runner: Runner | None = None,
    safety: dict[str, Any] | None = None,
    allow_dirty: bool = False,
    allow_stale: bool = False,
) -> dict[str, Any]:
    root = workspace.resolve()
    payload = readiness if readiness is not None else status.collect(root)
    safety_payload = safety if safety is not None else workspace_safety(root)
    resolved_target = resolve_target(payload, target)
    plan = build_plan(
        payload,
        workspace=root,
        target=resolved_target,
        max_ticks=max_ticks,
        live=live,
        safety=safety_payload,
    )
    if plan["action"] != "enact":
        return plan

    safety_refusal = live_safety_refusal(safety_payload, allow_dirty=allow_dirty, allow_stale=allow_stale)
    if safety_refusal:
        plan["ok"] = False
        plan["action"] = "refuse"
        plan["reason"] = safety_refusal
        return plan

    runner = runner or run_command
    result = runner(plan["command"], root, timeout_s)
    plan["result"] = result
    plan["ok"] = result.get("returncode") == 0
    plan["action"] = "enacted" if plan["ok"] else "enact_failed"
    if not plan["ok"]:
        plan["reason"] = "worker dispatch returned non-zero"
    return plan


def live_safety_refusal(safety: dict[str, Any], *, allow_dirty: bool, allow_stale: bool) -> str:
    blockers = [
        blocker for blocker in safety.get("blockers") or []
        if not _safety_blocker_allowed(str(blocker.get("kind") or ""), allow_dirty=allow_dirty, allow_stale=allow_stale)
    ]
    if not blockers:
        return ""
    first = blockers[0]
    return str(first.get("detail") or first.get("kind") or "workspace safety preflight blocked live enact")


def _safety_blocker_allowed(kind: str, *, allow_dirty: bool, allow_stale: bool) -> bool:
    if kind == "dirty":
        return allow_dirty
    if kind in {"behind", "diverged"}:
        return allow_stale
    return False


def workspace_safety(workspace: Path) -> dict[str, Any]:
    root = workspace.resolve()
    git_probe = _git(root, ["rev-parse", "--is-inside-work-tree"])
    if git_probe.get("returncode") != 0 or str(git_probe.get("stdout") or "").strip().lower() != "true":
        return {
            "ok": False,
            "blockers": [{
                "kind": "git",
                "detail": "workspace is not inside a git worktree",
                "next_action": "run the supervisor from a git workspace",
            }],
            "git": {"available": False, "error": git_probe.get("stderr") or git_probe.get("error")},
        }

    blockers: list[dict[str, Any]] = []
    status_probe = _git(root, ["status", "--porcelain=v1", "--untracked-files=normal"])
    dirty_lines = [
        line for line in str(status_probe.get("stdout") or "").splitlines()
        if line.strip()
    ]
    if status_probe.get("returncode") != 0:
        blockers.append({
            "kind": "git",
            "detail": "could not read git worktree status",
            "next_action": "inspect git status before running a live supervisor canary",
        })
    elif dirty_lines:
        blockers.append({
            "kind": "dirty",
            "detail": f"worktree has {len(dirty_lines)} dirty path(s)",
            "next_action": "commit, stash, or pass --allow-dirty for an operator-approved live canary",
        })

    branch = str((_git(root, ["branch", "--show-current"]).get("stdout") or "")).strip()
    upstream_probe = _git(root, ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"])
    upstream = str(upstream_probe.get("stdout") or "").strip() if upstream_probe.get("returncode") == 0 else ""
    relation = "unknown"
    if upstream:
        head_ancestor = _git(root, ["merge-base", "--is-ancestor", "HEAD", upstream]).get("returncode") == 0
        upstream_ancestor = _git(root, ["merge-base", "--is-ancestor", upstream, "HEAD"]).get("returncode") == 0
        if head_ancestor and upstream_ancestor:
            relation = "in_sync"
        elif head_ancestor:
            relation = "behind"
            blockers.append({
                "kind": "behind",
                "detail": f"workspace is behind upstream {upstream}",
                "next_action": "run `python tools/safe_ff_sync.py apply --fetch` to fast-forward safely, or pass --allow-stale for an operator-approved live canary",
            })
        elif upstream_ancestor:
            relation = "ahead"
        else:
            relation = "diverged"
            blockers.append({
                "kind": "diverged",
                "detail": f"workspace has diverged from upstream {upstream}",
                "next_action": "resolve branch divergence before running a live supervisor canary",
            })

    return {
        "ok": not blockers,
        "blockers": blockers,
        "git": {
            "available": True,
            "dirty": bool(dirty_lines),
            "dirty_count": len(dirty_lines),
            "dirty_sample": dirty_lines[:20],
            "branch": branch,
            "upstream": upstream,
            "relation": relation,
        },
    }


def _git(cwd: Path, args: list[str]) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"returncode": None, "stdout": "", "stderr": str(exc), "error": str(exc)}
    return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


def resolve_target(readiness: dict[str, Any], explicit_target: int | None) -> int:
    if explicit_target is not None:
        return max(1, int(explicit_target))
    supervise = readiness.get("supervise") or {}
    configured_target = _int(supervise.get("target"), 1)
    alive = _int(supervise.get("alive"), 0)
    if readiness.get("verdict") == "READY_TO_CANARY":
        return max(1, min(configured_target, alive + 1))
    return max(1, configured_target)


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def run_command(cmd: list[str], cwd: Path, timeout_s: int) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": None,
            "timeout": True,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or str(exc),
        }
    except OSError as exc:
        return {"returncode": None, "error": str(exc), "stdout": "", "stderr": str(exc)}
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def route_defect_stop(workspace: Path, plan: dict[str, Any]) -> dict[str, Any] | None:
    """Self-route a live enact failure into the findings queue. Fail-open — never raises.

    The deterministic-loop-spine closure rung (issue #381, tools/findings_route.py): a
    defect STOP self-routes one pickable findings-queue row at the moment it dies, not by
    a later audit — so a supervisor loop that keeps hitting the same wall leaves a trail a
    future dispatch picks up instead of dying silently. ``cause_key`` is the lane-stable
    identity (no run ts embedded) so concurrent re-fires damp onto one row and a re-fire
    after a "fixed" close escalates. Returns the route verdict, or None if routing was
    unavailable. Any failure here is swallowed: routing must never kill the loop.
    """
    try:
        import findings_route
    except Exception:
        return None
    try:
        readiness = plan.get("readiness") or {}
        spawn = readiness.get("spawn") or []
        lane = str(spawn[0]) if spawn else "default"
        result = plan.get("result") or {}
        rc = result.get("returncode")
        signature = "timeout" if result.get("timeout") else f"rc{rc}"
        cause = f"supervisor-enact-failed:{lane}"
        item = (
            f"DOS supervisor canary enact failed on lane '{lane}' ({signature}): "
            f"{plan.get('reason') or 'worker dispatch returned non-zero'}"
        )
        return findings_route.route_finding(
            key=f"{cause}:{findings_route._utcnow()}",
            sev="P2",
            pattern="supervisor-enact-failed",
            item=item,
            source=f"dos-supervisor-watchdog:{lane}",
            owning_plan="RSI",
            cause_key=cause,
            root=workspace,
        )
    except Exception:
        return None  # fail-open: routing must never kill the supervisor loop


def render(payload: dict[str, Any]) -> str:
    command = " ".join(payload.get("command") or [])
    lines = [
        f"DOS supervisor watchdog: {payload.get('action')} ({'ok' if payload.get('ok') else 'blocked'})",
        f"reason: {payload.get('reason')}",
    ]
    if command:
        lines.append(f"command: {command}")
    result = payload.get("result")
    if isinstance(result, dict):
        lines.append(f"returncode: {result.get('returncode')}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Dry-run or run one bounded DOS supervisor canary.")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument(
        "--target",
        type=int,
        default=None,
        help="bounded canary target (default: one above the current live count)",
    )
    ap.add_argument("--max-ticks", type=int, default=1, help="max iterations (default: 1)")
    ap.add_argument("--timeout-s", type=int, default=120, help="live dispatch timeout in seconds")
    ap.add_argument("--live", action="store_true", help="actually run the worker dispatch")
    ap.add_argument("--allow-dirty", action="store_true", help="permit --live from a dirty worktree")
    ap.add_argument("--allow-stale", action="store_true", help="permit --live when the workspace is behind or diverged from upstream")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args(argv)

    workspace = Path(args.workspace).resolve() if args.workspace else status.repo_root()
    payload = run_watchdog(
        workspace=workspace,
        target=args.target,
        max_ticks=args.max_ticks,
        live=args.live,
        timeout_s=args.timeout_s,
        allow_dirty=args.allow_dirty,
        allow_stale=args.allow_stale,
    )
    if args.live and payload.get("action") == "enact_failed":
        # Defect STOP: self-route a pickable findings-queue row before we exit (#381).
        route_defect_stop(workspace, payload)
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(render(payload))
    if payload.get("action") == "enact_failed":
        result = payload.get("result") or {}
        return int(result.get("returncode") or 1)
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
