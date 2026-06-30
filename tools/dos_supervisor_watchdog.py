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

## Cross-node lane serialization (issue #21)

This is the live fleet-code seam where the GitRefStore cross-node lease is ACTIVATED.
Before a ``--live`` canary launches a worker on one of the 3 global exclusive lanes
(abi/release/global), ``fleet_lease_gate`` publishes this node's held set and
materializes the fleet-wide union through the configured fleet store, then REFUSES the
launch if a live FOREIGN peer already holds the lane. It is a NON-mutating cross-node
check, not a kernel pre-acquire: the launched worker still does its own ``dos`` lane
acquire, so the gate never holds a kernel lease that would wedge the worker. The store
backend follows ``dos_fleet_lease``: global lanes default to the GitRefStore CAS, while
an explicit ``FAK_FLEET_LEASE_STORE`` still overrides for tests/dev. The git floor fails
CLOSED (a store it cannot consult refuses, never silently double-grants).

``FLEET_LANE_ALLOWLIST`` is DECIDED here as ADVISORY steering, NOT a serialization
floor: the dos kernel's bare-autopick ladder (``arbiter.py`` reads ``cfg.lanes.autopick``)
exposes no per-node override, so a fleet allowlist can only filter what the supervisor
*proposes* (``build_plan``), never bind what the kernel *picks*. The floor for the global
lanes is the GitRefStore gate above; the allowlist only steers/refuses the canary lane
this layer chooses, and a real per-node allowlist waits on a dos-kernel autopick-override
seam.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from dispatch_worker import install_no_window_subprocess_defaults
install_no_window_subprocess_defaults(subprocess)
import sys
from pathlib import Path
from typing import Any, Callable

import dos_supervisor_status as status

try:  # The cross-node lease transport (issue #21).
    import dos_fleet_lease
except Exception:  # noqa: BLE001 - import is best-effort; global lanes then fail closed
    dos_fleet_lease = None  # type: ignore[assignment]

SCHEMA = "fleet-dos-supervisor-watchdog/1"

ENACTABLE_VERDICTS = {"READY_TO_CANARY"}
NOOP_VERDICTS = {"AT_TARGET", "READY"}
PLAN_EMPTY_VERDICT = "PLAN_SURFACE_EMPTY"

# The 3 exclusive lanes that need fleet-wide consensus (mirrors dos_fleet_lease.GLOBAL_LANES).
GLOBAL_LANES = ("abi", "release", "global")


Runner = Callable[[list[str], Path, int], dict[str, Any]]

# A lease gate returns None to PROCEED or a refusal dict to BLOCK a live launch.
LeaseGate = Callable[..., "dict[str, Any] | None"]


def lane_allowlist(env: dict[str, str] | None = None) -> set[str] | None:
    """Parse ``FLEET_LANE_ALLOWLIST`` (comma/space separated) into a set, or None.

    ADVISORY ONLY (issue #21): this filters the lanes the supervisor PROPOSES for a
    canary; it is NOT a serialization floor. The dos kernel's bare-autopick ladder reads
    ``cfg.lanes.autopick`` with no fleet override, so a worker that bare-autopicks can
    still land off-allowlist — the real cross-node floor for global lanes is the
    GitRefStore lease, not this list. None = no allowlist (all lanes pass).
    """
    raw = (env if env is not None else os.environ).get("FLEET_LANE_ALLOWLIST")
    if not raw or not raw.strip():
        return None
    return {tok for tok in raw.replace(",", " ").split() if tok}


def _command_lane(command: list[str]) -> str | None:
    """The ``--lane`` value in an enact command (the lane that will actually launch)."""
    for i, tok in enumerate(command):
        if tok == "--lane" and i + 1 < len(command):
            return command[i + 1]
    return None


def fleet_lease_store(workspace: Path, env: dict[str, str], *, lane: str = "") -> Any:
    """Select the cross-node lease store for the live gate (issue #21).

    ``FAK_FLEET_LEASE_STORE=git`` -> the GitRefStore cross-node floor; a path -> a
    LocalDirStore there. With no override, the 3 global lanes default to GitRefStore
    and shard lanes stay on the same-box LocalDirStore fast path. Kept independent of
    ``dos_fleet_lease._store_for`` so this leaf never couples to that CLI helper's
    signature.
    """
    backend = (env.get("FAK_FLEET_LEASE_STORE") or "").strip()
    if backend == "git":
        return dos_fleet_lease.GitRefStore(workspace)
    if backend:
        return dos_fleet_lease.LocalDirStore(Path(backend))
    if lane and dos_fleet_lease.is_global_lane(lane):
        return dos_fleet_lease.GitRefStore(workspace)
    return dos_fleet_lease.LocalDirStore(Path(workspace) / "tools" / "_registry" / "fleet-leases")


def fleet_lease_gate(
    lane: str | None,
    *,
    workspace: Path,
    env: dict[str, str] | None = None,
    store: Any | None = None,
    live: Callable[[Path], list[dict[str, Any]]] | None = None,
    now: float | None = None,
) -> dict[str, Any] | None:
    """Serialize a GLOBAL lane across nodes BEFORE a live launch (issue #21 activation).

    Returns None to PROCEED (the lane is not global or no live fleet peer holds the
    lane), or a refusal dict to BLOCK the launch.

    For a global lane it (1) publishes this node's held lease set so a peer's tick can
    observe it, (2) materializes the global union through the store, and (3) refuses if a
    live FOREIGN peer already holds the lane. NON-mutating w.r.t. the kernel WAL: the
    launched worker still does its own ``dos`` lane acquire. The global default is the
    GitRefStore floor and fails CLOSED (could not publish/consult -> refuse).

    Honest residual (per dos_fleet_lease): publish and a peer's materialize are one tick
    apart, so a freshly-acquired peer lane is visible only on the NEXT peer tick — the
    overlay NARROWS, it does not close, the cross-node double-grant.
    """
    if not lane:
        return None
    if dos_fleet_lease is None:
        if lane in GLOBAL_LANES:
            return {
                "reason": "SCHEMA_UNREADABLE",
                "detail": f"cross-node lease helper unavailable for global lane {lane!r}; refusing launch",
            }
        return None
    if not dos_fleet_lease.is_global_lane(lane):
        return None
    e = env if env is not None else os.environ
    host = dos_fleet_lease.host_id()
    now = dos_fleet_lease.now_s() if now is None else now
    live = live or dos_fleet_lease.kernel_live
    try:
        st = store if store is not None else fleet_lease_store(workspace, e, lane=lane)
        pub, _ = dos_fleet_lease.do_publish(workspace, st, host=host, now=now, live=live)
        if not pub.get("ok"):
            # Could not stake this node's claim in the cross-node store -> do not launch.
            return {
                "reason": dos_fleet_lease.REASON_STALE_RECORD,
                "detail": f"could not publish lease set to the cross-node store for {lane!r}; refusing launch",
            }
        mat, _ = dos_fleet_lease.do_materialize(workspace, st, scope="global", host=host, now=now, live=live)
        foreign = [
            r for r in mat["leases"]
            if r.get("lane") == lane and r.get("host_id") and r.get("host_id") != host
        ]
    except Exception as exc:  # noqa: BLE001 - report; never crash the supervisor tick
        return {
            "reason": dos_fleet_lease.REASON_STALE_RECORD,
            "detail": f"cross-node lease store unavailable for global lane {lane!r}: {exc!r}",
        }
    hit = dos_fleet_lease.would_collide(lane, [f"{lane}/**"], foreign)
    if hit.get("collides"):
        return {
            "reason": hit.get("reason") or dos_fleet_lease.REASON_HELD_REMOTE,
            "detail": f"global lane {lane!r} held by live fleet peer {hit.get('host_id')!r}",
            "holder": hit.get("holder"),
            "host_id": hit.get("host_id"),
        }
    return None


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


def issue_surface_fallback(workspace: Path, *, timeout_s: int = 130) -> dict[str, Any]:
    """Read the issue-lane router as the fallback work surface.

    The router may exit non-zero when unrouted work is still high; that is not a
    fallback failure if it also returns routed lanes. We consume the JSON payload
    and let _issue_fallback_ready decide whether there is dispatchable issue work.
    """
    cmd = [sys.executable, str(workspace / "tools" / "issue_lane_router.py"), "--json"]
    try:
        proc = subprocess.run(
            cmd, cwd=workspace, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout_s)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"_error": str(exc), "_cmd": cmd}
    doc = status.read_json_from_text(proc.stdout)
    if not isinstance(doc, dict):
        doc = {}
    doc["_returncode"] = proc.returncode
    doc["_cmd"] = cmd
    if proc.returncode not in range(0, 16) and "_error" not in doc:
        doc["_error"] = (proc.stderr or proc.stdout or "").strip()[-500:]
    return doc


def _issue_fallback_ready(issue_fallback: dict[str, Any] | None) -> bool:
    if not issue_fallback or issue_fallback.get("_error"):
        return False
    counts = issue_fallback.get("counts") or {}
    lanes = issue_fallback.get("lanes") or {}
    return _int(counts.get("routed"), 0) > 0 and bool(lanes)


def issue_fallback_command(workspace: Path, target: int, *, live: bool) -> list[str]:
    cmd = [
        sys.executable,
        str(workspace / "tools" / "issue_resolve_dispatch.py"),
        "--workspace", str(workspace),
        "--max-workers", str(max(1, target)),
        "--json",
    ]
    if live:
        cmd.append("--live")
    return cmd


def build_plan(
    readiness: dict[str, Any],
    *,
    workspace: Path,
    target: int,
    max_ticks: int,
    live: bool,
    safety: dict[str, Any] | None = None,
    env: dict[str, str] | None = None,
    issue_fallback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    verdict = str(readiness.get("verdict") or "")
    supervise = readiness.get("supervise") or {}
    spawn = [str(lane) for lane in supervise.get("spawn") or []]

    # FLEET_LANE_ALLOWLIST advisory steer (issue #21): narrow the proposed spawn lanes to
    # the allowlist before picking one. This is best-effort steering, NOT a floor — see the
    # module docstring. When EVERY proposed lane is off-allowlist and the state would
    # otherwise enact, refuse with the steer reason; non-enactable states fall through.
    allow = lane_allowlist(env)
    if allow is not None and spawn:
        allowed = [ln for ln in spawn if ln in allow]
        if not allowed:
            if readiness.get("ok") and verdict in ENACTABLE_VERDICTS:
                return _plan(
                    readiness=readiness,
                    workspace=workspace,
                    target=target,
                    max_ticks=max_ticks,
                    live=live,
                    action="refuse",
                    ok=False,
                    reason=(
                        f"FLEET_LANE_ALLOWLIST advisory steer refused launch: proposed lane(s) "
                        f"{spawn} not in allowlist {sorted(allow)}"
                    ),
                    command=enact_command(workspace, target, max_ticks, lane=spawn[0]),
                    safety=safety,
                )
        else:
            spawn = allowed

    # Build the command with the first available (allowlist-steered) spawn lane
    lane = spawn[0] if spawn else None
    command = enact_command(workspace, target, max_ticks, lane=lane)

    if verdict == PLAN_EMPTY_VERDICT and _issue_fallback_ready(issue_fallback):
        counts = issue_fallback.get("counts") or {}
        fallback_command = issue_fallback_command(workspace, target, live=live)
        return _plan(
            readiness=readiness,
            workspace=workspace,
            target=target,
            max_ticks=max_ticks,
            live=live,
            action="enact" if live else "would_enact",
            ok=True,
            reason=(
                "plan surface empty; falling back to issue surface with "
                f"{counts.get('routed')} routed open issue(s)"
            ),
            command=fallback_command,
            safety=safety,
            issue_fallback={
                "schema": issue_fallback.get("schema"),
                "routed": counts.get("routed"),
                "unrouted": counts.get("unrouted"),
                "open": counts.get("open"),
                "lanes": sorted((issue_fallback.get("lanes") or {}).keys()),
            },
        )

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
    issue_fallback: dict[str, Any] | None = None,
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
        **({"issue_fallback": issue_fallback} if issue_fallback else {}),
    }


def run_watchdog(
    *,
    workspace: Path,
    target: int | None,
    max_ticks: int,
    live: bool,
    timeout_s: int,
    readiness: dict[str, Any] | None = None,
    issue_fallback: dict[str, Any] | None = None,
    runner: Runner | None = None,
    safety: dict[str, Any] | None = None,
    allow_dirty: bool = False,
    allow_stale: bool = False,
    env: dict[str, str] | None = None,
    lease_gate: LeaseGate | None = None,
) -> dict[str, Any]:
    root = workspace.resolve()
    payload = readiness if readiness is not None else status.collect(root)
    if issue_fallback is None and payload.get("verdict") == PLAN_EMPTY_VERDICT:
        issue_fallback = issue_surface_fallback(root)
    safety_payload = safety if safety is not None else workspace_safety(root)
    resolved_target = resolve_target(payload, target)
    plan = build_plan(
        payload,
        workspace=root,
        target=resolved_target,
        max_ticks=max_ticks,
        live=live,
        safety=safety_payload,
        env=env,
        issue_fallback=issue_fallback,
    )
    if plan["action"] != "enact":
        return plan

    safety_refusal = live_safety_refusal(safety_payload, allow_dirty=allow_dirty, allow_stale=allow_stale)
    if safety_refusal:
        plan["ok"] = False
        plan["action"] = "refuse"
        plan["reason"] = safety_refusal
        return plan

    # Cross-node lane serialization (issue #21): for a global lane, refuse the launch if a
    # live fleet peer already holds it. A no-op for non-global lanes.
    gate = lease_gate if lease_gate is not None else fleet_lease_gate
    gate_lane = _command_lane(plan.get("command") or [])
    refusal = gate(gate_lane, workspace=root, env=env)
    if refusal:
        plan["ok"] = False
        plan["action"] = "refuse"
        plan["reason"] = refusal.get("detail") or "cross-node fleet lease refused the launch"
        plan["fleet_lease"] = refusal
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


def slack_event(payload: dict[str, Any]) -> dict[str, Any] | None:
    """The actionable Slack event for one canary tick, or None when nothing happened.

    Pure: only a launched canary (``enacted``), a failed launch (``enact_failed``), or a
    refused launch (``refuse``) is worth a channel post — a ``noop`` / ``would_enact``
    dry tick is the quiet steady state and stays silent so the gate never spams. Returns
    {title, detail, level} for slack_post.event."""
    action = payload.get("action")
    reason = payload.get("reason") or ""
    if action == "enacted":
        return {"title": "DOS supervisor canary launched",
                "detail": reason or "bounded canary worker dispatched", "level": "info"}
    if action == "enact_failed":
        rc = (payload.get("result") or {}).get("returncode")
        return {"title": "DOS supervisor canary FAILED",
                "detail": f"{reason} (rc={rc})", "level": "crit"}
    if action == "refuse":
        return {"title": "DOS supervisor canary refused",
                "detail": reason or "a safety / cross-node lease gate blocked the launch",
                "level": "warn"}
    return None  # noop / would_enact -> quiet steady state


def _slack_enabled(args_slack: bool) -> bool:
    """Post events iff --slack OR the cron opt-in env FAK_DISPATCH_SLACK=1."""
    return bool(args_slack) or os.environ.get("FAK_DISPATCH_SLACK") == "1"


def maybe_post_slack(payload: dict[str, Any], *, enabled: bool, dry_run: bool = False,
                     channel: str = "", transport: Any | None = None) -> dict[str, Any] | None:
    """Post the canary tick's actionable event to Slack when enabled and the tick is
    actionable. Never raises — a missing poster or Slack failure becomes a logged
    verdict. Returns the slack_post verdict, or None when nothing to post / disabled."""
    if not enabled:
        return None
    ev = slack_event(payload)
    if ev is None:
        return None
    try:
        import slack_post  # sibling module in tools/
    except Exception as exc:  # noqa: BLE001
        return {"posted": False, "error": f"slack_post unavailable: {exc}"}
    return slack_post.event(ev["title"], ev["detail"], level=ev["level"],
                            channel=channel, dry_run=dry_run, transport=transport)


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
    ap.add_argument("--slack", action="store_true",
                    help="post a Slack event on a launched/failed/refused canary "
                         "(FAK_DISPATCH_SLACK=1 also enables); a noop tick stays silent")
    ap.add_argument("--slack-dry-run", action="store_true",
                    help="with --slack: report what WOULD be posted without sending")
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
    slack_verdict = maybe_post_slack(payload, enabled=_slack_enabled(args.slack),
                                     dry_run=args.slack_dry_run)
    if slack_verdict is not None:
        payload["slack"] = slack_verdict
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(render(payload))
        if slack_verdict is not None:
            if slack_verdict.get("posted"):
                print(f"slack: posted event to {slack_verdict.get('channel')}")
            elif slack_verdict.get("dry_run"):
                print(f"slack (dry-run): would post to {slack_verdict.get('channel') or '(unset)'}")
            elif slack_verdict.get("skipped"):
                print(f"slack: skipped — {slack_verdict.get('skipped')}")
            else:
                print(f"slack: FAILED — {slack_verdict.get('error')}")
    if payload.get("action") == "enact_failed":
        result = payload.get("result") or {}
        return int(result.get("returncode") or 1)
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
