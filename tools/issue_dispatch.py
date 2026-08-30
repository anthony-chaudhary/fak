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

``--warm-floor`` / ``--stagger-s`` (#3610) prime the shared ~35.8k startup floor once
per wave and space the launches so members 2..N re-enter that prefix as a cache READ.
``--floor-cache-meter`` is the read-only counterpart: it joins the durable wave
sidecars to the gateway-usage ledger by guard pid and prints the warm-vs-cold A/B over
wave FOLLOWERS, refusing to grade an arm it has no evidence for.

    python tools/issue_dispatch.py --floor-cache-meter             # the warm/cold A/B
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
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
import worker_worktree  # noqa: E402  (#3181: per-worker worktree isolation, shared with the Go spine)

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

# --- #6492: the wave's spawn READ-BACK gate ---------------------------------------
# A wave used to report WAVED off `subprocess.Popen` returning a pid, and a pid only
# proves the OS accepted the exec -- not that a worker survived long enough to do any
# work. Witnessed 2026-08-11: a 4-lane live wave printed `WAVED (ok)` with four pids
# while all four children were gone within seconds, their logs 0 bytes, and the
# spawn-failure telemetry still reporting zero failures. The single-issue path already
# had the answer (issue_resolve_dispatch.probe_spawned_worker): wait a bounded moment
# and read the child back before claiming it. These knobs are that gate for the wave
# path. It only ever DEMOTES: a child already dead when we look is typed SPAWN_FAILED
# with its returncode + log tail instead of being counted as a live member.
WAVE_SPAWN_PROBE_ENV = "FLEET_WAVE_SPAWN_PROBE_S"
# Mirrors issue_resolve_dispatch.DEFAULT_SPAWN_PROBE_S -- the same bounded window the
# per-issue path already pays. 0 disables the gate (back to Popen-success reporting).
DEFAULT_WAVE_SPAWN_PROBE_S = 5.0
# The terminal record written next to the log of a child that did not survive the
# gate. It is what stops an empty `dispatch-<lane>-<stamp>.log` from reading as an
# apparent run: the runs dir carries an EXPLICIT failure record instead of silence,
# and dispatch_status.spawn_failed_cause_breakdown folds it as a real failure.
SPAWN_FAILED_SIDECAR_SUFFIX = ".spawn-failed.json"
# Mirrors issue_resolve_dispatch.EARLY_EXIT_TAIL_CHARS (bounded evidence, not a dump).
WAVE_EARLY_EXIT_TAIL_CHARS = 8192

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
    cached ``sessions.json`` snapshot in the host registry -- $FLEET_REG_DIR when the
    fleet names it, else whatever ``fleet_regdir`` resolves. This tick runs with the
    variable UNSET, and until #5390 that meant the clone-root ``tools/_registry``: this
    refresh maintained a SECOND, ledger-less registry beside the prober's, so the
    switcher routed off a snapshot that could not derive a single block. On an always-on tick that
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


def _preflight_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def preflight_public(doc: dict[str, Any]) -> dict[str, Any]:
    out = {"verdict": doc.get("verdict"), "reason": doc.get("reason"),
           "cap": doc.get("cap"), "live": doc.get("live")}
    for key in ("headroom", "max_workers", "host_cap"):
        if doc.get(key) is not None:
            out[key] = doc.get(key)
    limiter = doc.get("capacity_limiter")
    if isinstance(limiter, dict):
        out["capacity_limiter"] = limiter
    seat = doc.get("seat")
    if isinstance(seat, dict):
        out["seat"] = {k: seat.get(k) for k in (
            "total", "free", "leased", "depleted", "unattributed_live")
            if seat.get(k) is not None}
    return out


def preflight_refusal_hint(doc: dict[str, Any]) -> dict[str, Any] | None:
    if doc.get("verdict") != "REFUSE_AT_CAP":
        return None
    limiter = doc.get("capacity_limiter") if isinstance(doc.get("capacity_limiter"), dict) else {}
    raw = limiter.get("raw") if isinstance(limiter.get("raw"), dict) else {}
    term = str(limiter.get("term") or "")
    maxw = _preflight_int(raw.get("max_workers"))
    if maxw is None:
        maxw = _preflight_int(doc.get("max_workers"))
    host_cap = _preflight_int(raw.get("host_cap"))
    if host_cap is None:
        host_cap = _preflight_int(doc.get("host_cap"))
    live_count = _preflight_int(doc.get("live"))
    cap_count = _preflight_int(doc.get("cap"))
    needed = live_count + 1 if live_count is not None else None
    configured_cap = term == "max_workers" or (
        maxw is not None and cap_count == maxw
        and host_cap is not None and host_cap > maxw)
    host_headroom_available = (
        host_cap is None or live_count is None or live_count < host_cap)
    needed_within_host = needed is None or host_cap is None or needed <= host_cap
    if configured_cap and host_headroom_available and needed_within_host:
        return {
            "kind": "configured_max_workers",
            "message": (
                f"configured --max-workers={maxw} is the binding cap; "
                + (f"rerun with --max-workers >= {needed} " if needed else "raise --max-workers ")
                + (f"(still bounded by host_cap={host_cap}) "
                   if host_cap is not None else "")
                + "only if the operator intends to use available host headroom"),
            "max_workers": maxw,
            "required_min": needed,
            "host_cap": host_cap,
        }
    if limiter:
        return {
            "kind": term or str(limiter.get("primary") or "capacity"),
            "message": (f"capacity limiter {limiter.get('primary') or '?'}/"
                        f"{term or '?'} is binding; do not route around "
                        "the preflight refusal"),
        }
    return None


def lane_priority_scores(root: Path,
                         issue_nums: dict[str, list[int]]) -> dict[str, int]:
    """Map each lane to the highest issue_triage score among ITS open issues.

    ``issue_nums`` is ``{lane: [open issue numbers]}`` from the router fold. Shells to
    ``issue_triage.py --json`` once, joins its per-issue ``score`` (P0=1000 / P1=400 /
    P2=150 plus the orphan/stale bonuses) to each lane's issue set, and returns
    ``{lane: max(score)}`` for every lane with at least one scored issue. A lane whose
    issues carry no triage row is simply absent (the caller reads it as priority 0).

    FAIL-OPEN by construction: any triage read/parse error yields ``{}`` so the single
    caller (``pick_lane``) falls straight back to raw-count ranking — a gh/triage
    hiccup never wedges or reshapes the tick beyond its historical behavior."""
    if not issue_nums:
        return {}
    doc = run_json([_py(), str(root / "tools" / "issue_triage.py"), "--json"],
                   root, timeout=130)
    rows = doc.get("rows") if isinstance(doc, dict) else None
    if not isinstance(rows, list):
        return {}
    score_by_num: dict[int, int] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        num, score = r.get("number"), r.get("score")
        if isinstance(num, int) and isinstance(score, int):
            score_by_num[num] = score
    out: dict[str, int] = {}
    for lane, nums in issue_nums.items():
        scored = [score_by_num[n] for n in nums if n in score_by_num]
        if scored:
            out[lane] = max(scored)
    return out


def held_self_source_lanes(root: Path) -> dict[str, Any]:
    """The self-modify-HELD lanes carrying their file tree AND their open issue numbers.

    This is the ONE primitive neither ``pick_lane`` nor ``lane_candidates`` exposes:
    ``pick_lane`` reports held lane NAMES only (``self_modify_held``) and
    ``lane_candidates`` DROPS self-source lanes entirely. The unguarded escape
    (``escape_self_source``) needs both halves the guarded path throws away — the
    ``tree`` (to price a lane via ``dos arbitrate``) and the open issue NUMBERS (to
    rank the held lanes by ``issue_triage`` score). So it re-reads the router ONCE
    (the same ``run_json`` call ``pick_lane`` makes) and keeps precisely the lanes the
    guard would hold: ``is_self_source_tree`` trees with at least one open issue.

    Read-only and FAIL-OPEN: any router hiccup yields ``{'held': [], 'router_error': …}``
    so the escape simply finds nothing to do rather than wedging — it never fabricates a
    lane the guarded pick would not itself have held."""
    router = run_json([_py(), str(root / "tools" / "issue_lane_router.py"), "--json"],
                      root, timeout=130)
    lanes = router.get("lanes") or {}
    held: list[dict[str, Any]] = []
    for ln, info in lanes.items():
        iss = info.get("issues") if isinstance(info, dict) else info
        tree = info.get("tree") if isinstance(info, dict) else None
        nums = [n for n in (iss or []) if isinstance(n, int)]
        if nums and is_self_source_tree(tree):
            held.append({"lane": ln, "issue_nums": nums, "tree": list(tree or [])})
    return {"held": held, "router_error": router.get("_error")}


def rank_held_by_triage(root: Path, held: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order held self-source lanes highest-VALUE first: by ``issue_triage`` score, then
    open-issue count, then lane name.

    Reuses ``lane_priority_scores`` (the #2064 per-lane triage-score join) so the escape
    ranks by the SAME score the guarded pick uses — a P1 self-source lane (#3091 accounts
    740, #2354 security 704) is drained before a P2 one. FAIL-OPEN identically: a triage
    read error collapses every score to 0, leaving the deterministic count-then-name
    order (never a crash, never an empty ranking). Each returned row is the input row
    plus a ``score`` field."""
    priority = lane_priority_scores(
        root, {h["lane"]: h.get("issue_nums", []) for h in held})
    scored = [{**h, "score": priority.get(h["lane"], 0)} for h in held]
    scored.sort(key=lambda h: (-h["score"], -len(h.get("issue_nums", [])), h["lane"]))
    return scored


def escape_candidates(root: Path,
                      safe_lane_priority: dict[str, int] | None) -> dict[str, Any]:
    """The Arm-1 SIGNAL: the held self-source P1s a guarded tick legally cannot spawn,
    surfaced LOUDLY so the silent docs fall-through becomes an actionable field.

    A guarded tick holds every ``cmd/**``/``internal/**`` lane (#1397), so its highest
    triage score can be a *docs* leaf while the real top P1 (#3091 accounts) sits
    unreachable — the "loop grinds docs" symptom. This ranks the held lanes and marks the
    top one ``preferred=True`` iff it OUT-SCORES the best safe (dispatchable) lane, i.e.
    the exact condition under which the operator/super-loop should run
    ``--escape-self-source`` instead of draining another low-value safe lane. It NEVER
    changes ``chosen``/``members`` — the guarded spawn cannot target a held lane; this is
    purely a payload signal. ``safe_lane_priority`` is the guarded pool's
    ``{lane: score}`` (``priority_by_lane`` from ``pick_lane``, or issue-count as the
    wave's proxy)."""
    ranked = rank_held_by_triage(root, held_self_source_lanes(root)["held"])
    best_safe = max(safe_lane_priority.values(), default=0) if safe_lane_priority else 0
    top = ranked[0]["score"] if ranked else None
    for i, r in enumerate(ranked):
        r["preferred"] = bool(i == 0 and top is not None and top > best_safe)
    return {"escape_candidates": ranked, "top_held_score": top,
            "best_safe_score": best_safe}


HELD_TICKS_BASENAME = "self-modify-held-ticks.json"


def fold_held_ticks(runs_dir: Path, held: list[dict[str, Any]]) -> dict[str, int]:
    """Persist the consecutive-ticks-held counter for self-modify-held lanes (#3125).

    ``held`` rows are ``{lane, issue_nums}``. A lane held THIS tick increments its
    counter from the previous tick's file; a lane that left the held set is dropped
    (consecutive means uninterrupted), so "held for N consecutive ticks" is a
    checkable field, not a narration. Reads and writes FAIL-OPEN like ``_record`` —
    a counter hiccup can only under-count a hold, never wedge a tick."""
    path = runs_dir / HELD_TICKS_BASENAME
    try:
        prev = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        prev = {}
    if not isinstance(prev, dict):
        prev = {}
    now: dict[str, Any] = {}
    for h in held:
        ln = h.get("lane")
        if not ln:
            continue
        row = prev.get(ln)
        n = row.get("consecutive_ticks", 0) if isinstance(row, dict) else 0
        now[ln] = {"consecutive_ticks": int(n) + 1,
                   "issue_nums": [x for x in (h.get("issue_nums") or [])
                                  if isinstance(x, int)]}
    try:
        runs_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(now, indent=2), encoding="utf-8")
    except OSError:
        pass
    return {ln: v["consecutive_ticks"] for ln, v in now.items()}


def clear_held_ticks(runs_dir: Path) -> None:
    """A tick with NO held lanes resets the #3125 counter — consecutive, not lifetime."""
    try:
        (runs_dir / HELD_TICKS_BASENAME).unlink(missing_ok=True)
    except OSError:
        pass


def held_report(held: list[dict[str, Any]], ticks: dict[str, int]) -> list[dict[str, Any]]:
    """Per-ISSUE "held, needs unguarded/worktree lane" rows for the tick surface (#3125).

    One row per open issue in a held self-source lane — the explicit acceptance surface:
    a held issue is either routed to the escape path or REPORTED here, never silently
    dropped from the tick. A lane whose issue numbers are unknown (router hiccup) still
    gets one lane-level row (``issue: None``) so the hold is never invisible."""
    rows: list[dict[str, Any]] = []
    for h in held:
        ln = h.get("lane")
        nums = [n for n in (h.get("issue_nums") or []) if isinstance(n, int)]
        base = {"lane": ln, "status": "held, needs unguarded/worktree lane",
                "consecutive_ticks": ticks.get(ln, 1),
                "escape": "python tools/issue_dispatch.py --escape-self-source"}
        if nums:
            rows.extend({**base, "issue": n} for n in nums)
        else:
            rows.append({**base, "issue": None})
    return rows


def _attach_held_surface(payload: dict[str, Any], runs_dir: Path,
                         held_lanes: list[str]) -> None:
    """Fold the #3125 held-forever surface into a tick/wave payload IN PLACE.

    Joins the lane names the picker held (the payload spine) with the issue numbers
    ``escape_candidates`` already fetched (zero extra shell-outs), folds the persisted
    consecutive-ticks counter, and attaches the per-issue report. Rides INSIDE the
    existing ``self_modify_held`` gate, so the default (unheld/hermetic) tick payload
    stays byte-for-byte unchanged."""
    by_lane = {r.get("lane"): r.get("issue_nums") or []
               for r in payload.get("escape_candidates") or []}
    rows = [{"lane": ln, "issue_nums": by_lane.get(ln, [])} for ln in held_lanes]
    ticks = fold_held_ticks(runs_dir, rows)
    payload["self_modify_held_ticks"] = ticks
    payload["self_modify_held_report"] = held_report(rows, ticks)


def pick_lane(root: Path, explicit: str | None,
              busy: set[str] | None = None,
              guarded: bool | None = None) -> dict[str, Any]:
    """The highest-priority lane by issue_triage score (open-issue count as the
    tiebreak), or an explicit override.

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
    issue_nums: dict[str, list[int]] = {}
    for ln, info in lanes.items():
        iss = info.get("issues") if isinstance(info, dict) else info
        counts[ln] = len(iss) if hasattr(iss, "__len__") else 0
        issue_nums[ln] = list(iss) if isinstance(iss, (list, tuple)) else []
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
    # Priority-weight the pick (#2064): rank the pool by the highest issue_triage score
    # in each lane FIRST, raw open-issue count only as the tiebreak, so a single
    # sequential seat resolves the highest-VALUE leaf across lanes instead of always
    # draining the lane with the biggest (often low-priority) backlog. `priority` is
    # empty on any triage read error, collapsing the key to the historical count order.
    priority = lane_priority_scores(root, {ln: issue_nums.get(ln, []) for ln in pool})
    # Highest triage priority first, then highest open-issue count, then lane name as a
    # stable lexicographic tiebreak so the pick is deterministic across ticks.
    lane = sorted(pool, key=lambda k: (-priority.get(k, 0), -pool[k], k))[0]
    return {"lane": lane, "issues": counts[lane], "by_lane": counts,
            "busy": sorted(busy), "stacked": stacked,
            "lane_priority": priority.get(lane, 0), "priority_by_lane": priority,
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


def _write_inflight_marker(runs_dir: Path, lane: str, pid: int,
                           account: str | None = None,
                           guarded: bool = True) -> str | None:
    """Record {lane, pid, account, guarded} so a LATER tick sees this lane AND this
    account are in flight and spreads to a different one. ``account`` (the worker's
    pinned CLAUDE_CONFIG_DIR) lets the next wave avoid double-loading an account that
    already has a live worker — the cross-tick ACCOUNT de-confliction (#2060), the twin
    of the cross-tick LANE de-confliction. ``guarded`` self-describes whether this
    worker is fronted by ``fak guard`` (the default) or is an unguarded escape worker,
    so ``guarded_worker_in_flight`` can read the build-integrity gate off the marker set
    alone — the escape must not land an unguarded self-source commit while a guarded
    peer's ``go build`` could be poisoned. Best-effort: a write failure never blocks the
    spawn."""
    try:
        runs_dir.mkdir(parents=True, exist_ok=True)
        path = runs_dir / f"{INFLIGHT_PREFIX}{lane}-{int(pid)}.json"
        path.write_text(
            json.dumps({"lane": lane, "pid": int(pid),
                        "account": account or None,
                        "guarded": bool(guarded),
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


def busy_accounts(runs_dir: Path, *, is_alive: Callable[[int], bool] | None = None,
                  now: float | None = None,
                  ttl_seconds: int = INFLIGHT_TTL_SECONDS) -> set[str]:
    """The set of ACCOUNTS (pinned CLAUDE_CONFIG_DIRs) with a currently-live dispatched
    worker, folded from the same inflight markers ``busy_lanes`` reads. This is the
    cross-tick ACCOUNT de-confliction (#2060): a wave excludes an account already
    running a worker so a single free seat is placed on the IDLE account instead of
    double-loading one (which the prior seat allocation, blind to a prior tick's live
    worker, did — risking a single account's rate/usage cap). Same self-heal as
    ``busy_lanes``; a marker with no recorded account contributes nothing (older
    markers predate the field)."""
    alive = is_alive or _pid_is_alive
    if not runs_dir.is_dir():
        return set()
    when = now if now is not None else time.time()
    accounts: set[str] = set()
    for marker in sorted(runs_dir.glob(f"{INFLIGHT_PREFIX}*.json")):
        try:
            rec = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _safe_unlink(marker)
            continue
        raw_pid = (rec or {}).get("pid")
        pid = int(raw_pid) if isinstance(raw_pid, int) else (
            int(raw_pid) if isinstance(raw_pid, str) and raw_pid.strip().isdigit() else 0)
        try:
            stale = (when - marker.stat().st_mtime) > ttl_seconds
        except OSError:
            stale = True
        if pid <= 0 or stale or not alive(pid):
            _safe_unlink(marker)
            continue
        account = str((rec or {}).get("account") or "").strip()
        if account:
            accounts.add(account)
    return accounts


def guarded_worker_in_flight(runs_dir: Path, *,
                             is_alive: Callable[[int], bool] | None = None,
                             now: float | None = None,
                             ttl_seconds: int = INFLIGHT_TTL_SECONDS) -> bool:
    """True iff a currently-live dispatched worker is GUARDED — the build-integrity gate
    an unguarded escape must clear before it lands a self-source commit.

    Folds the SAME inflight markers ``busy_lanes``/``busy_accounts`` read, counting a
    marker only when its pid is alive and its file is within ``ttl_seconds``. A guarded
    peer may be mid-edit on the shared Go source, so its ``go build`` can be transiently
    red; an unguarded escape worker that commits self-source in that window would ship on
    top of a poisoned build. Reading the gate off the marker set (not a live process
    probe) keeps it hermetic and injectable. A marker with NO ``guarded`` field predates
    the escape work, when every worker was guarded, so it is treated as guarded — an old
    marker never opens the gate by omission. Prunes the dead/unreadable/stale markers it
    scans, the same self-heal as its siblings; a first live guarded marker short-circuits
    to ``True`` (the remaining markers are pruned by the next ``busy_lanes`` pass)."""
    alive = is_alive or _pid_is_alive
    if not runs_dir.is_dir():
        return False
    when = now if now is not None else time.time()
    for marker in sorted(runs_dir.glob(f"{INFLIGHT_PREFIX}*.json")):
        try:
            rec = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _safe_unlink(marker)
            continue
        raw_pid = (rec or {}).get("pid")
        pid = int(raw_pid) if isinstance(raw_pid, int) else (
            int(raw_pid) if isinstance(raw_pid, str) and raw_pid.strip().isdigit() else 0)
        try:
            stale = (when - marker.stat().st_mtime) > ttl_seconds
        except OSError:
            stale = True
        if pid <= 0 or stale or not alive(pid):
            _safe_unlink(marker)
            continue
        if bool((rec or {}).get("guarded", True)):
            return True
    return False


def _truthy_env(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _sanitize_worker_id(s: str) -> str:
    """A commit-trailer-safe worker token: keep ``[A-Za-z0-9._-]``, fold the rest to
    ``-``. Mirrors dispatch_status._dispatch_lease_token's alphabet so the id survives
    round-tripping through a ``(fak-worker <id>)`` trailer and the read-only fold."""
    out = "".join(c if (c.isalnum() or c in "-_.") else "-" for c in str(s or "").strip())
    return out.strip("-.") or "unknown"


def worker_id(account_dir: str | None, lane: str,
              base_env: dict[str, str] | None = None) -> str:
    """A best-effort, human-legible id for the worker this env launches (#2065).

    Stamped into ``FLEET_WORKER_ID`` so the dispatched agent can carry it as a
    ``(fak-worker <id>)`` commit trailer, making a worker's witnessed ships
    attributable on a shared trunk. Precedence: an explicit ambient
    ``FLEET_WORKER_ID`` (operator / wave override) > the pinned account-seat directory
    basename (the seat that ran the work) > the lane. The trailer is an attribution
    AID, never a witness."""
    env = base_env if base_env is not None else os.environ
    explicit = (env.get("FLEET_WORKER_ID") or "").strip()
    if explicit:
        return _sanitize_worker_id(explicit)
    seat = Path(account_dir).name if account_dir else ""
    return _sanitize_worker_id(seat or lane or "unknown")


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
    # A best-effort worker identity the dispatched agent can carry as a
    # (fak-worker <id>) commit trailer, so its witnessed ships are attributable on the
    # shared trunk (#2065). Set from the pinned account-seat basename (or the lane), and
    # read back read-only by dispatch_status.ships_per_worker. Aid, not a witness — it
    # never replaces the (fak <leaf>) ship stamp and nothing is gated on it.
    env["FLEET_WORKER_ID"] = worker_id(account_dir, lane, env)
    return env


def wave_spawn_probe_s(env: "dict[str, str] | None" = None) -> float:
    """The bounded read-back window one wave spawn is checked over (#6492).

    Env-overridable (``FLEET_WAVE_SPAWN_PROBE_S``), defaulting to the same window the
    per-issue path already pays. ``0`` disables the gate, which restores the old
    Popen-success reporting exactly — kept as an explicit operator escape, never a
    silent default. An unparseable value falls back to the default rather than
    disabling a safety gate by typo."""
    raw = ((env if env is not None else os.environ).get(WAVE_SPAWN_PROBE_ENV) or "").strip()
    if not raw:
        return DEFAULT_WAVE_SPAWN_PROBE_S
    try:
        return max(0.0, float(raw))
    except ValueError:
        return DEFAULT_WAVE_SPAWN_PROBE_S


def probe_spawned_child(proc: Any, out_log: Path, wait_s: float) -> dict[str, Any]:
    """Briefly check whether a just-spawned child died before it could log (#6492).

    Mirrors ``issue_resolve_dispatch.probe_spawned_worker`` — deliberately duplicated
    rather than imported, because that module imports THIS one (importing it back at
    spawn time would be an import cycle on the hot path).

    A healthy worker can be silent for many seconds while the agent starts, so a LIVE
    process with a 0-byte log is not a failure and is never reported as one. The
    failure witnessable here is narrower and unambiguous: the process has ALREADY
    exited by the time the bounded wait elapses. FAIL-OPEN: a probe that cannot run
    (a stub proc, an OS error) reports ``checked=False`` and the spawn stands."""
    if wait_s <= 0:
        return {"checked": False}
    try:
        returncode = proc.wait(timeout=wait_s)
    except subprocess.TimeoutExpired:
        return {"checked": True, "alive": True, "wait_s": wait_s}
    except (AttributeError, OSError, ValueError):
        return {"checked": False}
    try:
        log_bytes = out_log.stat().st_size
    except OSError:
        log_bytes = 0
    rec: dict[str, Any] = {
        "checked": True,
        "alive": False,
        "wait_s": wait_s,
        "returncode": returncode,
        "log_bytes": log_bytes,
        "silent": log_bytes == 0,
    }
    if log_bytes:
        try:
            rec["tail"] = out_log.read_text(
                encoding="utf-8", errors="replace")[-WAVE_EARLY_EXIT_TAIL_CHARS:]
        except OSError:
            pass
    return rec


def classify_spawn_failure(early: dict[str, Any]) -> str:
    """Attribute one early-exit to a cause bucket, reusing the #2635 classifier.

    Lazily imported (``issue_resolve_dispatch`` imports this module) and FAIL-OPEN: if
    the classifier is unavailable the event is still counted, just as ``unknown`` — a
    missing attribution must never turn a witnessed failure back into a success."""
    try:
        import issue_resolve_dispatch as ird  # noqa: PLC0415  (lazy: import cycle)

        return str(ird.classify_spawn_failed_cause(early))
    except Exception:   # noqa: BLE001  -- attribution is advisory; the count is not
        return "unknown"


def spawn_failure_record(spawned: dict[str, Any], lane: str,
                         seat: str | None = None) -> dict[str, Any] | None:
    """The typed SPAWN_FAILED record for a child that did not survive the read-back
    gate, or ``None`` when it is still alive (or was never probed).

    ``None`` is the honest answer for an unprobed spawn: absence of a probe is not
    evidence of death, and this gate only ever demotes on POSITIVE evidence."""
    early = spawned.get("early_exit") or {}
    if not early.get("checked") or early.get("alive"):
        return None
    return {
        "verdict": "SPAWN_FAILED",
        "cause": classify_spawn_failure(early),
        "lane": lane,
        "seat": seat,
        "pid": spawned.get("pid"),
        "log": spawned.get("log"),
        "returncode": early.get("returncode"),
        "log_bytes": early.get("log_bytes"),
        "silent": bool(early.get("silent")),
        "tail": early.get("tail") or "",
        "stamp": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


def retire_failed_spawn(spawned: dict[str, Any], failure: dict[str, Any]) -> str | None:
    """Close out a dead spawn on disk: write the terminal failure record next to its
    log and drop the in-flight marker it wrote at exec.

    Both halves matter. The record is what stops an empty ``dispatch-<lane>-<stamp>``
    log from reading as an apparent run — the run is explicitly terminal, with the
    returncode and tail that killed it. Dropping the marker stops a dead child from
    holding its lane busy against the NEXT tick. Best-effort: a disk fault never
    escalates a spawn failure into a tick crash."""
    written: str | None = None
    log = spawned.get("log")
    if log:
        try:
            path = Path(str(log)).with_suffix(SPAWN_FAILED_SIDECAR_SUFFIX)
            path.write_text(json.dumps(failure, indent=2, sort_keys=True),
                            encoding="utf-8")
            written = str(path)
        except (OSError, ValueError):
            written = None
    marker = spawned.get("inflight")
    if marker:
        try:
            Path(str(marker)).unlink()
        except OSError:
            pass
    return written


def spawn_detached(command: list[str], env: dict[str, str], cwd: Path,
                   log_dir: Path, lane: str, guarded: bool = True,
                   worktree_git: "Callable[..., Any] | None" = None,
                   probe_s: float = 0.0) -> dict[str, Any]:
    """Launch the worker DETACHED so it outlives this tick; log to a dated file.

    ``probe_s`` (#6492) is the bounded read-back window: with a positive value the
    spawn is checked back before it is reported, and a child that has already exited
    comes back stamped ``early_exit``. It defaults to 0 (no read-back) so every
    existing caller keeps today's behaviour; the wave path passes the real window.

    ``guarded`` records whether THIS worker is fronted by ``fak guard`` so the in-flight
    marker self-describes its guard status; a later tick reads it via
    ``guarded_worker_in_flight`` as the build-integrity gate before an unguarded escape
    lands a self-source commit.

    #3181: when ``FLEET_WORKER_WORKTREE`` is on, isolate the worker in its own detached
    worktree pinned at trunk HEAD (``worktree_git`` is the injectable git runner) —
    ``cwd`` becomes that worktree and GOCACHE/GOTMPDIR are redirected inside it, the same
    gate + fail-open contract as the Go spine (#3168). A ``.worktree`` sidecar records the
    path + base SHA so the witness sweep can land the diff under the lane lease and reap.
    FAIL-OPEN: any worktree fault leaves ``cwd``/``env`` untouched (shared-trunk spawn as
    before), and flag-off is byte-identical to today."""
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_log = log_dir / f"dispatch-{lane}-{stamp}.log"
    # #3181: opt-in per-worker worktree isolation, shared helper/flag with the Go spine.
    # Record the worktree as a PLAIN path in `<log>.worktree` — the same sidecar shape
    # the shared witness sweep consumes (cmd/fak/dispatch_tick_witness). The land+reap
    # sweep scans `resolve-*.log`, so this tick site's build-isolation lands via its
    # per-issue resolve children; the breadcrumb keeps the contract identical.
    spawn_cwd, env, wt_info = worker_worktree.isolate_spawn(
        Path(cwd), lane, lane, cwd, env, git=worktree_git)
    if wt_info.get("worktree"):
        try:
            out_log.with_suffix(".worktree").write_text(
                str(wt_info["worktree"]), encoding="utf-8")
        except OSError:
            pass
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
    proc = subprocess.Popen(argv, cwd=str(spawn_cwd), env=env, stdin=subprocess.DEVNULL,
                            stdout=fh, stderr=subprocess.STDOUT, **kwargs)
    # Stamp this lane AND its pinned account as in flight so a later tick spreads off
    # both (cross-tick lane + account de-confliction). Best-effort and pid-keyed;
    # busy_lanes/busy_accounts prune it on death.
    marker = _write_inflight_marker(log_dir, lane, proc.pid,
                                    account=env.get("CLAUDE_CONFIG_DIR"),
                                    guarded=guarded)
    result = {"pid": proc.pid, "log": str(out_log), "inflight": marker}
    if wt_info.get("worktree"):
        result["worktree"] = wt_info["worktree"]
    # #6492: read the child back before this spawn is reported as a worker.
    early = probe_spawned_child(proc, out_log, probe_s)
    if early.get("checked"):
        result["early_exit"] = early
    return result


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


# --- #3610: cross-worker floor cache-warm -----------------------------------------
# The startup floor (system prompt + tool schemas, ~35.8k — internal/gateway/
# ctxfootprint.go) is byte-identical across every claude worker of a wave, but the
# launcher fires N Popens back-to-back, so each member pays a full provider
# cache-WRITE. Priming that prefix ONCE and spacing the members inside the cache TTL
# turns N writes into 1 write + (N-1) reads (~10% cost).
#
# BOTH knobs default OFF (warm) / ZERO (stagger): an unset config reproduces today's
# launch behaviour byte-for-byte, because this is the shared wave path every fleet
# lane dispatches through.
#
# TTL sizing: do NOT size the stagger against Anthropic's default 5-min TTL. fak
# forces the stable-prefix 1h-TTL upgrade by default (an UNSET FAK_MANAGED_CACHE
# resolves to `on` — cmd/fak/guard_cache_posture.go:14-19), so the live budget is
# plausibly 1h, which makes a modest stagger nearly free. The posture can still
# degrade to PASSIVE on a subscription-OAuth seat, so it must be READ, not assumed —
# which is why the default here is 0.0 (opt-in) rather than a guessed interval.
WARM_FLOOR_ENV = "FLEET_WARM_FLOOR"
LAUNCH_STAGGER_ENV = "FLEET_LAUNCH_STAGGER_S"
WARM_FLOOR_TIMEOUT_S = 120.0

# Injectable so a test can observe the stagger without actually sleeping.
_sleep = time.sleep


def _env_flag(env: dict[str, str], name: str) -> bool:
    return (env.get(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def _env_float(env: dict[str, str], name: str, default: float) -> float:
    try:
        return max(0.0, float((env.get(name) or "").strip()))
    except ValueError:
        return default


def warm_floor_prefix(root: Path, *, live: bool,
                      timeout_s: float = WARM_FLOOR_TIMEOUT_S,
                      run: "Callable[..., Any] | None" = None) -> dict[str, Any]:
    """Issue exactly ONE floor-warm pre-request for the whole wave (#3610).

    Runs the same guarded argv a member runs, with only the trailing user turn
    swapped (``dispatch_worker.build_warm_floor_command``), so the prefix it primes is
    the prefix the members re-enter on. Blocking with a bounded timeout: the write must
    LAND before member 1 launches, or there is nothing warm to read.

    FAIL-OPEN by construction — a warm fault (no binary, timeout, non-zero exit) is
    recorded and the wave proceeds unstaggered-but-correct. Warming is a cost
    optimization; it must never be able to block a dispatch.

    Never spends tokens on a dry run: ``live=False`` returns a ``would_warm`` record.
    """
    command = dispatch_worker.build_warm_floor_command("claude")
    launch_command, guarded = dispatch_worker.guarded_launch_command(
        command, dispatch_worker.WARM_FLOOR_LANE, "claude", root)
    rec: dict[str, Any] = {"guarded": guarded,
                           "command": launch_command or command}
    if not live:
        rec["would_warm"] = True
        return rec
    if not launch_command:
        rec["error"] = "no warm-floor command resolved"
        return rec
    runner = run or subprocess.run
    exe = shutil.which(launch_command[0]) or launch_command[0]
    started = time.monotonic()
    try:
        proc = runner([exe, *launch_command[1:]], cwd=str(root),
                      stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                      stderr=subprocess.STDOUT, timeout=timeout_s)
        rec["returncode"] = getattr(proc, "returncode", None)
        rec["warmed"] = rec["returncode"] == 0
    except subprocess.TimeoutExpired:
        rec["error"] = f"warm-floor pre-request exceeded {timeout_s}s"
    except OSError as exc:
        rec["error"] = f"warm-floor pre-request failed: {exc}"
    rec["elapsed_s"] = round(time.monotonic() - started, 3)
    return rec


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
    marker_busy = busy_lanes(root / RUNS_DIRNAME)
    lease_busy = lease_ref_busy_lanes(root)
    lease_busy_set = set(lease_busy.get("lanes") or set())
    busy = marker_busy | lease_busy_set
    lane_pick = pick_lane(root, lane, busy=busy)
    chosen = lane_pick.get("lane")
    command, launch_command, guarded = _build_launch(root, chosen)

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "workspace": str(root),
        "live": live,
        "max_workers": max_workers,
        "registry_refresh": reg,
        "preflight": preflight_public(pre),
        "account": {k: acct.get(k) for k in ("tag", "tier", "model", "dir")},
        "lane": chosen,
        "lane_issue_count": lane_pick.get("issues"),
        "busy_lanes": sorted(busy),
        "busy_lane_sources": {
            "inflight_markers": sorted(marker_busy),
            "lease_refs": sorted(lease_busy_set),
        },
        "lane_stacked": bool(lane_pick.get("stacked")),
        "self_modify_held": lane_pick.get("self_modify_held") or [],
        "command": command,
        "guarded": guarded,
        "launch_command": launch_command,
        "witness": {"kind": "benchmark",
                    "cmd": f"python tools/bench_witness.py --lane {chosen}" if chosen else None},
    }
    if lease_busy.get("error"):
        payload["lease_busy_error"] = lease_busy.get("error")
    # Build-integrity gate (UNGATED): reads the same inflight markers busy_lanes already
    # folded above — a pure local file read, no shell-out — so it is cheap and hermetic
    # enough to surface every tick. An unguarded escape worker consults it before landing
    # a self-source commit while a guarded peer's go build could be transiently red.
    payload["guarded_worker_in_flight"] = guarded_worker_in_flight(root / RUNS_DIRNAME)
    # Arm-1 signal (GATED): surface the held self-source P1s the guarded pick legally
    # cannot spawn, but ONLY when a lane is actually held. The gate is load-bearing for
    # both cost and hermeticity: escape_candidates() re-shells to issue_lane_router +
    # issue_triage(gh), so firing it unconditionally would add two live shell-outs to
    # every default tick (measured ~450x slower under the hermetic pinned suite, which
    # stubs pick_lane but not run_json). Under every test stub self_modify_held is
    # empty, so this never fires there — the default path stays byte-for-byte unchanged.
    if lane_pick.get("self_modify_held"):
        payload.update(escape_candidates(root, lane_pick.get("priority_by_lane")))
        # #3125: a held issue is either routed to the escape path or explicitly
        # reported per-issue with its consecutive-ticks-held count — never silently
        # dropped from the tick surface.
        _attach_held_surface(payload, root / RUNS_DIRNAME,
                             lane_pick.get("self_modify_held") or [])
    else:
        clear_held_ticks(root / RUNS_DIRNAME)

    if not pre_ok:
        hint = preflight_refusal_hint(pre)
        if hint:
            payload["preflight_hint"] = hint
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
    spawned = spawn_detached(launch_command, env, root, root / RUNS_DIRNAME, chosen,
                             guarded=guarded)
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


def _escape_render_lines(p: dict[str, Any]) -> list[str]:
    """Operator lines for the Arm-1 escape SIGNAL, emitted ONLY when a self-source lane
    is held (``escape_candidates`` present in the payload). Makes the otherwise
    payload-only signal visible in the human render: the top held P1, whether it
    out-scores the best dispatchable lane, and — when it does — the build-integrity gate
    an unguarded escape must clear. On a normal tick the key is absent, so this returns
    nothing and the render is byte-for-byte unchanged."""
    cands = p.get("escape_candidates")
    if not cands:
        return []
    top = cands[0]
    nums = ", ".join(f"#{n}" for n in (top.get("issue_nums") or [])) or "-"
    verb = ("PREFER --escape-self-source" if top.get("preferred")
            else "held (best safe lane wins)")
    more = f" (+{len(cands) - 1} more held)" if len(cands) > 1 else ""
    lines = [f"  escape    : {top.get('lane')} {nums} score={top.get('score')} "
             f"vs safe {p.get('best_safe_score')} -> {verb}{more}"]
    if top.get("preferred"):
        gate = ("guarded build in flight — hold the unguarded escape"
                if p.get("guarded_worker_in_flight")
                else "clear (no guarded build in flight)")
        lines.append(f"              gate: {gate}")
    # #3125: list each held-forever issue explicitly (capped), so a held issue is
    # visibly triaged on the operator surface, never silently skipped tick after tick.
    report = p.get("self_modify_held_report") or []
    for r in report[:6]:
        iss = f"#{r['issue']}" if r.get("issue") is not None else "(issues unknown)"
        lines.append(f"  held      : {iss} [{r.get('lane')}] {r.get('status')} "
                     f"x{r.get('consecutive_ticks')} ticks")
    if len(report) > 6:
        lines.append(f"  held      : (+{len(report) - 6} more held issues)")
    return lines


def render(p: dict[str, Any]) -> str:
    a = p.get("account") or {}
    pf = p.get("preflight") or {}
    hint = p.get("preflight_hint") or {}
    busy_sources = p.get("busy_lane_sources") or {}
    busy_detail = []
    if busy_sources.get("inflight_markers"):
        busy_detail.append(f"markers: {', '.join(busy_sources.get('inflight_markers'))}")
    if busy_sources.get("lease_refs"):
        busy_detail.append(f"lease refs: {', '.join(busy_sources.get('lease_refs'))}")
    lines = [
        f"issue-dispatch: {p.get('verdict')} ({'ok' if p.get('ok') else 'refuse'})  live={p.get('live')}",
        f"  preflight : {pf.get('verdict')} ({pf.get('live')}/{pf.get('cap')} live"
        + (f", host_cap {pf.get('host_cap')}" if pf.get('host_cap') is not None else "")
        + ")",
        f"  account   : {a.get('tag') or '-'} (t{a.get('tier')})  {a.get('model') or ''}",
        f"  lane      : {p.get('lane') or '-'}  ({p.get('lane_issue_count')} issues)"
        + (f"  [busy: {'; '.join(busy_detail) or ', '.join(p.get('busy_lanes'))}]"
           if p.get('busy_lanes') else "")
        + ("  STACKED (all lanes in flight)" if p.get('lane_stacked') else ""),
        f"  witness   : {(p.get('witness') or {}).get('cmd') or '-'}",
        f"  command   : {' '.join(p.get('command') or []) or '-'}",
    ]
    if isinstance(hint, dict) and hint.get("message"):
        lines.append(f"  hint      : {hint.get('message')}")
    lines.extend(_escape_render_lines(p))
    lines.append(f"  -> {p.get('reason')}")
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


def _fak_cmd() -> list[str] | None:
    """The installed ``fak`` CLI for read-only lease inspection.

    Do not fall back to ``go run`` here: a dirty shared cmd/ tree can fail to build,
    and a capacity planner should fail open rather than wedge on a compile.
    """
    configured = (os.environ.get("FAK_BIN") or "").strip()
    if configured:
        return shlex.split(configured, posix=(os.name != "nt"))
    exe = shutil.which("fak")
    return [exe] if exe else None


def _lease_lane(lease_id: Any) -> str | None:
    raw = str(lease_id or "").strip()
    if not raw:
        return None
    lane = raw[len("resolve-"):] if raw.startswith("resolve-") else raw
    base, sep, suffix = lane.rpartition("-")
    if sep and suffix.isdigit():
        lane = base
    return lane or None


def lease_ref_busy_lanes(root: Path) -> dict[str, Any]:
    """Fold refs/fak live lane leases into the same busy-lane set as local markers.

    Inflight markers only cover workers launched by this checkout. Cross-session and
    cross-machine leases are visible through ``fak leaseref liveness``; a wave must
    not plan onto those lanes just because it lacks a local pid sidecar.
    """
    cmd_prefix = _fak_cmd()
    if not cmd_prefix:
        return {"lanes": set(), "error": "no fak binary"}
    cmd = [*cmd_prefix, "leaseref", "liveness"]
    try:
        proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=30,
                              creationflags=dispatch_worker.no_window_creationflags())
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"lanes": set(), "error": str(exc)}
    out = (proc.stdout or "").strip()
    try:
        records = json.loads(out) if out else []
    except ValueError:
        records = []
    if not isinstance(records, list):
        return {"lanes": set(), "error": "unexpected leaseref liveness shape",
                "returncode": proc.returncode}
    lanes: set[str] = set()
    rows: list[dict[str, Any]] = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        if rec.get("reclaimable") is True:
            continue
        lane = _lease_lane(rec.get("id"))
        if not lane:
            continue
        lanes.add(lane)
        rows.append({k: rec.get(k) for k in (
            "id", "holder", "session_id", "liveness", "reclaimable")
            if rec.get(k) is not None} | {"lane": lane})
    payload: dict[str, Any] = {"lanes": lanes, "records": rows,
                               "returncode": proc.returncode}
    if proc.returncode != 0:
        payload["error"] = (proc.stderr or "").strip() or f"exit {proc.returncode}"
    return payload


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
        leases = fleet_accounts.live_seat_leases(str(root / RUNS_DIRNAME))
        return fleet_accounts.allocate_wave(max_workers, work_kind=work_kind,
                                            product="claude", leases=leases)
    except Exception as exc:  # noqa: BLE001 — fail-open boundary: no seats, never fatal
        return {"ok": False, "granted": 0, "lanes": [], "wave_id": None,
                "error": str(exc)}


def wave_admission_budget(pre: dict[str, Any], max_workers: int) -> dict[str, Any]:
    """Seat request bound from the same preflight snapshot that will gate spawning."""
    requested = max(0, int(max_workers))
    headroom = _preflight_int(pre.get("headroom"))
    seat = pre.get("seat") if isinstance(pre.get("seat"), dict) else {}
    seat_free = _preflight_int(seat.get("free"))
    if pre.get("verdict") != "SPAWN_OK":
        return {
            "max_workers": requested,
            "preflight_headroom": headroom,
            "seat_free": seat_free,
            "requested_seats": 0,
        }
    budget = requested
    if headroom is not None:
        budget = min(budget, max(0, headroom))
    if seat_free is not None:
        budget = min(budget, max(0, seat_free))
    return {
        "max_workers": requested,
        "preflight_headroom": headroom,
        "seat_free": seat_free,
        "requested_seats": budget,
    }


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
                       rank: int, size: int, shortfall: int,
                       probe_s: float | None = None) -> dict[str, Any]:
    """Launch one wave worker on its seat, stamped with its wave membership. Writes a
    per-worker ``.wave`` sidecar next to the log so the whole wave is enumerable from
    disk, never from a worker's self-report.

    #6492: the spawn is READ BACK over a bounded window before it is reported. A child
    already dead when we look comes back as a ``spawn_failed`` record (typed
    SPAWN_FAILED, with cause/returncode/tail), its log is closed out with a terminal
    failure record, and its in-flight marker is dropped — so a dead child is never
    counted as a live member and never holds its lane against the next tick."""
    command, launch_command, guarded = _build_launch(root, lane)
    if not launch_command:
        return {"error": f"no command for lane '{lane}'"}
    env = worker_env(seat.get("config_dir"), lane, root)
    env.update(_wave_env(rank, wave_id, size, shortfall))
    if guarded:
        dispatch_worker.guard_env_augment(env)
    spawned = spawn_detached(launch_command, env, root, root / RUNS_DIRNAME, lane,
                             guarded=guarded,
                             probe_s=(wave_spawn_probe_s() if probe_s is None
                                      else probe_s))
    spawned["guarded"] = guarded
    try:
        Path(spawned["log"]).with_suffix(".wave").write_text(
            json.dumps({"wave_id": wave_id, "rank": rank, "size": size,
                        "shortfall": shortfall}, sort_keys=True), encoding="utf-8")
    except OSError:
        pass
    failure = spawn_failure_record(spawned, lane, seat=seat.get("tag"))
    if failure:
        failure["wave_id"] = wave_id
        failure["rank"] = rank
        spawned["spawn_failed"] = failure
        record = retire_failed_spawn(spawned, failure)
        if record:
            spawned["spawn_failed_record"] = record
    return spawned


def _write_wave_artifacts(runs_dir: Path, payload: dict[str, Any]) -> None:
    """Write the wave-level sidecar the done-condition names — ``{wave_id, size,
    lanes, seats}`` — plus a full ``last-wave.json`` for inspection. The sidecar is
    the contract artifact: one record per tick enumerable from disk alongside the
    per-worker ``.wave`` membership stamps.

    #3610: the sidecar also carries the launch-cache POSTURE and the per-member
    ``{rank, pid}`` join key, because ``last-wave.json`` is overwritten every tick and
    an A/B of warm-vs-cold waves needs two DURABLE records to compare. Without these
    two fields the wave leaves no trace that can be joined to provider cache counters,
    which is exactly what blocked the meter this issue's third box asks for.
    """
    try:
        runs_dir.mkdir(parents=True, exist_ok=True)
        sidecar = {"wave_id": payload.get("wave_id"), "size": payload.get("size"),
                   "lanes": payload.get("lanes"), "seats": payload.get("seats_used"),
                   "launched_at_ms": int(time.time() * 1000),
                   "launch_cache": payload.get("launch_cache"),
                   "warm_floor": payload.get("warm_floor"),
                   "members": _wave_join_members(payload.get("members") or [])}
        (runs_dir / f"dispatch-wave-{payload.get('wave_id')}.json").write_text(
            json.dumps(sidecar, indent=2), encoding="utf-8")
        (runs_dir / "last-wave.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        pass


def _wave_join_members(members: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project the wave's members down to the floor-cache JOIN KEY (#3610).

    The pid is the spawned ``fak guard`` process, and guard stamps ``os.Getpid()`` into
    every gateway-usage row it writes (internal/gatewayusageledger.NewRow), so the pid
    is what resolves a member to its OWN provider cache counters. Members that never
    spawned (dry run) carry a null pid and simply do not join.
    """
    out: list[dict[str, Any]] = []
    for m in members:
        spawned = m.get("spawned") or {}
        out.append({"lane": m.get("lane"), "rank": m.get("rank"),
                    "pid": spawned.get("pid"),
                    "seat": (m.get("account") or {}).get("tag")})
    return out


def _wave_launch_knobs(warm_floor: bool | None,
                       stagger_s: float | None) -> tuple[bool, float]:
    """Resolve the #3610 launch-cache knobs from their env defaults.

    Both stay off/zero when neither the argument nor the env knob is set, so an
    unconfigured wave launches exactly as it did before #3610.
    """
    env = os.environ
    if warm_floor is None:
        warm_floor = _env_flag(env, WARM_FLOOR_ENV)
    if stagger_s is None:
        stagger_s = _env_float(env, LAUNCH_STAGGER_ENV, 0.0)
    return bool(warm_floor), max(0.0, float(stagger_s))


def _wave_seat_lanes(root: Path, requested: int,
                     work_kind: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Allocate the wave's seats, minus any seat whose ACCOUNT is already busy."""
    seats = allocate_seats(root, requested, work_kind)
    seat_lanes = seats.get("lanes") or []
    # Cross-tick ACCOUNT de-confliction (#2060): the seat allocation is re-derived each
    # tick and is blind to a PRIOR tick's still-live worker, so with one seat free it
    # re-picked the SAME account (double-loading it, risking that account's usage cap)
    # while the peer account idled. Drop any seat whose account already runs an
    # in-flight worker so the free seat lands on the IDLE account. Never spawns onto a
    # busy account; if all accounts are busy the wave simply finds no free seat.
    busy_acct = busy_accounts(root / RUNS_DIRNAME)
    if busy_acct:
        seat_lanes = [s for s in seat_lanes if s.get("config_dir") not in busy_acct]
    return seats, seat_lanes


def _wave_busy_lanes(root: Path) -> tuple[set[str], dict[str, Any], set[str]]:
    """Lanes a PRIOR tick still holds, from both sources: markers and lease refs."""
    # Lanes a prior tick's worker still holds: skipped here so a wave never re-stacks
    # a lane already in flight (the within-tick arbiter only de-conflicts THIS wave's
    # own picks; busy_lanes carries the de-confliction ACROSS ticks).
    marker_busy = busy_lanes(root / RUNS_DIRNAME)
    lease_busy = lease_ref_busy_lanes(root)
    lease_busy_set = set(lease_busy.get("lanes") or set())
    return marker_busy, lease_busy, lease_busy_set


def _attach_wave_guard_surface(payload: dict[str, Any], root: Path,
                               cand: dict[str, Any],
                               candidates: list[dict[str, Any]]) -> None:
    """Attach the build-integrity and self-modify-hold surfaces to the tick record."""
    # Build-integrity gate (UNGATED, same rationale as evaluate): a local marker read the
    # wave already did via busy_lanes/busy_accounts, no shell-out, so it rides every wave.
    payload["guarded_worker_in_flight"] = guarded_worker_in_flight(root / RUNS_DIRNAME)
    # Arm-1 signal (GATED, same rationale as evaluate): only when a self-source lane is
    # actually held. The wave has no per-lane triage score computed, so it passes the
    # candidate issue-COUNT as the safe-priority proxy — enough to decide whether a held
    # P1 out-ranks the best dispatchable lane and warrants an escape. Under WaveTest's
    # stub lane_candidates returns no self_modify_held, so this never fires hermetically.
    if cand.get("self_modify_held"):
        payload.update(escape_candidates(
            root, {c["lane"]: c["issues"] for c in candidates}))
        # #3125: same held-forever surface as evaluate — the wave is a tick too.
        _attach_held_surface(payload, root / RUNS_DIRNAME,
                             cand.get("self_modify_held") or [])
    else:
        clear_held_ticks(root / RUNS_DIRNAME)


def _admit_wave_members(root: Path, *, candidates: list[dict[str, Any]],
                        busy: set[str], seat_lanes: list[dict[str, Any]],
                        free_seats: int, seats: dict[str, Any], wave_id: str,
                        max_workers: int, work_kind: str, live: bool,
                        first_preflight: dict[str, Any],
                        warm_floor: bool, stagger_s: float) -> dict[str, Any]:
    """Admit candidate lanes into the wave, richest-first, one distinct seat each.

    Each lane is re-checked against ``dispatch_preflight``, held to the in-tick cap
    accounting, and PRICED against the leases this wave already admitted before any
    agent launches; the walk stops at the first refusal. Returns the tick's admission
    record: the members (spawned when ``live``), the lanes skipped as cross-tick busy,
    the refusal that stopped the walk, and the launch-cache posture.

    #6492: a live spawn only becomes a MEMBER once it survives the read-back gate. A
    child that is already dead when we look is moved to ``spawn_failed`` instead: it
    holds no lease, takes no rank, and never counts toward the wave's size — but it
    DOES consume its attempt and its seat, so one tick cannot re-burn the whole roster
    retrying lanes behind a seat that is killing children.
    """
    leases: list[dict[str, Any]] = []   # accumulating disjoint-tree leases (priced)
    warm_record: dict[str, Any] | None = None   # #3610 once-per-wave warm latch
    staggered = 0                               # #3610 count of delayed launches
    members: list[dict[str, Any]] = []
    spawn_failed: list[dict[str, Any]] = []     # #6492 children that died on launch
    skipped_busy: list[str] = []
    baseline_live: int | None = None
    cap_seen: int | None = None
    refusal: str | None = None
    last_preflight: dict[str, Any] | None = preflight_public(first_preflight)
    preflight_hint: dict[str, Any] | None = None
    if first_preflight.get("verdict") != "SPAWN_OK":
        refusal = first_preflight.get("verdict") or "REFUSE"
        preflight_hint = preflight_refusal_hint(first_preflight)
    if not refusal:
        preflight_seed = first_preflight
        for c in candidates:
            # Attempts, not survivors, bound the walk: a seat whose child died is spent
            # for this tick, so a failing seat cannot be silently re-drawn.
            attempts = len(members) + len(spawn_failed)
            if attempts >= max_workers:
                break
            if attempts >= free_seats:
                if free_seats > 0:
                    refusal = refusal or "SEATS_EXHAUSTED"
                break
            # Cross-tick de-confliction: a lane a prior tick's worker still holds is
            # skipped before it costs a preflight re-check or a seat.
            if c["lane"] in busy:
                skipped_busy.append(c["lane"])
                continue
            # Per-spawn preflight re-check: the live population must never exceed the cap.
            pre = preflight_seed or preflight(
                root, max_workers=max_workers, work_kind=work_kind)
            preflight_seed = None
            last_preflight = preflight_public(pre)
            if pre.get("verdict") != "SPAWN_OK":
                refusal = pre.get("verdict") or "REFUSE"
                preflight_hint = preflight_refusal_hint(pre)
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
                last_preflight["effective_live"] = effective_live
                preflight_hint = {
                    "kind": "in_tick_cap",
                    "message": (
                        f"wave in-tick accounting reached cap {cap}: effective_live "
                        f"{effective_live} >= cap; do not add another member until "
                        "a worker exits or the next preflight admits headroom"),
                }
                break
            # Price this lane against the wave's already-admitted leases.
            dec = arbitrate_lane(root, c["lane"], c["tree"], leases)
            if not dec.get("admitted"):
                continue   # collides with an admitted lane (or kernel error) -> skip
            rank = len(members) + len(spawn_failed)
            seat = seat_lanes[rank]
            member: dict[str, Any] = {
                "lane": c["lane"], "tree": dec["tree"], "issues": c["issues"], "rank": rank,
                "account": {"tag": seat.get("tag"), "tier": seat.get("model_tier"),
                            "pool": seat.get("pool"), "dir": seat.get("config_dir")},
                "arbitrate": dec.get("reason"),
            }
            if live:
                # #3610: prime the shared floor prefix ONCE, lazily — here rather than
                # before the loop, so a wave that ends up spawning nothing (every lane
                # busy, cap reached, arbiter refused) never pays for a warm it cannot
                # use. `warm_record is None` is the once-per-wave latch.
                if warm_floor and warm_record is None:
                    warm_record = warm_floor_prefix(root, live=True)
                # Space consecutive launches inside the cache TTL. Both lists are
                # appended AFTER the spawn, so a truthy `attempts` means at least one
                # launch is already out: the delay lands BETWEEN spawns, never before
                # the first (a failed spawn is still a launch that hit the provider).
                if stagger_s > 0 and attempts:
                    _sleep(stagger_s)
                    staggered += 1
                member["spawned"] = _spawn_wave_member(
                    root, c["lane"], seat, wave_id, rank, free_seats,
                    int(seats.get("shortfall") or 0))
                failure = (member["spawned"] or {}).get("spawn_failed")
                if failure:
                    # Witnessed dead on read-back: not a member, no lease, no seat
                    # reuse. The wave reports it as a typed failure instead of a pid.
                    member["spawn_failed"] = failure
                    spawn_failed.append(member)
                    continue
            members.append(member)
            leases.append({"lane": c["lane"], "lane_kind": "cluster", "tree": dec["tree"]})
    return {"members": members, "spawn_failed": spawn_failed,
            "skipped_busy": skipped_busy, "refusal": refusal,
            "preflight_hint": preflight_hint, "last_preflight": last_preflight,
            "cap": cap_seen, "warm_record": warm_record, "staggered": staggered}


def _spawn_failure_causes(failures: list[dict[str, Any]]) -> str:
    """``lane=cause`` for each dead child, in launch order — the one-line evidence
    summary a verdict carries so an operator reads WHICH lane died of WHAT without
    opening the per-log terminal records."""
    return ", ".join(
        f"{m.get('lane')}={(m.get('spawn_failed') or {}).get('cause') or 'unknown'}"
        for m in failures)


def _apply_wave_verdict(payload: dict[str, Any], root: Path, cand: dict[str, Any], *,
                        size: int, lanes_used: list[str], wave_id: str, live: bool,
                        refusal: str | None, candidates: list[dict[str, Any]],
                        free_seats: int, seats: dict[str, Any],
                        spawn_failed: list[dict[str, Any]] | None = None) -> None:
    """Stamp the tick's terminal verdict: what the wave did, or why it did nothing.

    #6492: ``size`` counts only children that SURVIVED the read-back gate, so WAVED is
    a claim about live workers rather than about accepted execs. Children that died on
    launch are named in the verdict either way — as a caveat on a partly-live wave, or
    as the wave's own SPAWN_FAILED verdict when none of them survived."""
    failures = list(spawn_failed or [])
    failed_note = ""
    if failures:
        causes = _spawn_failure_causes(failures)
        failed_note = (f"; {len(failures)} child(ren) died on launch "
                       f"[{causes}] and were NOT counted")
    if size > 0:
        payload.update({
            "ok": True,
            "verdict": "WAVED" if live else "WOULD_WAVE",
            "action": "waved" if live else "would_wave",
            "reason": (f"{'spawned' if live else 'would spawn'} {size} worker(s) across "
                       f"pairwise-disjoint lanes {lanes_used} (wave {wave_id})"
                       + (f"; stopped on {refusal}" if refusal else "")
                       + failed_note)})
        if live:
            _write_wave_artifacts(root / RUNS_DIRNAME, payload)
    elif failures:
        # Every child this wave launched was already dead when we read it back. That is
        # a SPAWN_FAILED tick, not a WAVED one and not a bare refusal: the tick DID act,
        # and the evidence (returncode + log tail per lane) is on disk.
        payload.update({
            "ok": False, "verdict": "SPAWN_FAILED", "action": "spawn_failed",
            "reason": (f"no worker survived launch: {len(failures)} child(ren) exited "
                       f"inside the read-back window [{_spawn_failure_causes(failures)}]"
                       + (f"; stopped on {refusal}" if refusal else ""))})
    elif refusal:
        payload.update({"ok": False, "verdict": refusal,
                        "action": "refused",
                        "reason": f"no worker spawned (stopped on {refusal})"})
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


def evaluate_wave(root: Path, *, max_workers: int, work_kind: str, live: bool,
                  refresh: bool = True, warm_floor: bool | None = None,
                  stagger_s: float | None = None) -> dict[str, Any]:
    """One WAVE tick: spawn up to ``max_workers`` workers across pairwise
    tree-disjoint lanes in a single tick, each on its own seat, never exceeding the
    dispatch_preflight cap.

    The wave size is ``min(free_seats, free_lanes, preflight_headroom)`` discovered
    ONLINE: candidate lanes are taken richest-first; each is PRICED against the wave's
    already-admitted leases via ``dos arbitrate`` (a colliding lane is skipped BEFORE
    any agent launches); each admitted lane draws the next distinct seat; and
    ``dispatch_preflight`` is re-checked per spawn so the live population provably
    never exceeds the cap. A wave sidecar records ``{wave_id, size, lanes, seats}``.

    #3610: ``warm_floor`` primes the shared ~35.8k floor prefix with ONE pre-request
    before any member spawns, and ``stagger_s`` spaces consecutive member launches so
    workers 2..N re-enter that warm prefix as a cache READ. Both default to their env
    knobs (``FLEET_WARM_FLOOR`` / ``FLEET_LAUNCH_STAGGER_S``) and are off/zero unset,
    so an unconfigured wave launches exactly as it does today."""
    warm_floor, stagger_s = _wave_launch_knobs(warm_floor, stagger_s)
    reg = refresh_registry(root) if refresh else {"ok": None, "skipped": True}
    first_preflight: dict[str, Any] | None = preflight(
        root, max_workers=max_workers, work_kind=work_kind)
    budget = wave_admission_budget(first_preflight, max_workers)
    seats, seat_lanes = _wave_seat_lanes(root, int(budget["requested_seats"]),
                                         work_kind)
    free_seats = len(seat_lanes)
    wave_id = seats.get("wave_id") or "wave-unallocated"

    cand = lane_candidates(root)
    candidates = cand.get("candidates") or []
    marker_busy, lease_busy, lease_busy_set = _wave_busy_lanes(root)
    busy = marker_busy | lease_busy_set

    payload: dict[str, Any] = {
        "schema": WAVE_SCHEMA,
        "workspace": str(root),
        "live": live,
        "max_workers": max_workers,
        "wave_id": wave_id,
        "registry_refresh": reg,
        "admission_budget": budget,
        "free_seats": free_seats,
        "seats": {"granted": seats.get("granted"), "requested": seats.get("requested"),
                  "shortfall": seats.get("shortfall"), "wave_id": seats.get("wave_id"),
                  "tags": [s.get("tag") for s in seat_lanes], "error": seats.get("error")},
        "candidate_lanes": [c["lane"] for c in candidates],
        "busy_lanes": sorted(busy),
        "busy_lane_sources": {
            "inflight_markers": sorted(marker_busy),
            "lease_refs": sorted(lease_busy_set),
        },
        "self_modify_held": cand.get("self_modify_held") or [],
        "router_error": cand.get("router_error"),
    }
    if lease_busy.get("error"):
        payload["lease_busy_error"] = lease_busy.get("error")
    _attach_wave_guard_surface(payload, root, cand, candidates)

    admitted = _admit_wave_members(
        root, candidates=candidates, busy=busy, seat_lanes=seat_lanes,
        free_seats=free_seats, seats=seats, wave_id=wave_id,
        max_workers=max_workers, work_kind=work_kind, live=live,
        first_preflight=first_preflight, warm_floor=warm_floor,
        stagger_s=stagger_s)
    members = admitted["members"]
    spawn_failed = admitted["spawn_failed"]
    refusal = admitted["refusal"]
    if admitted["warm_record"] is not None:
        payload["warm_floor"] = admitted["warm_record"]

    size = len(members)
    lanes_used = [m["lane"] for m in members]
    seats_used = [(m["account"] or {}).get("tag") for m in members]
    payload.update({"size": size, "lanes": lanes_used, "members": members,
                    "seats_used": seats_used, "cap": admitted["cap"],
                    "refusal": refusal, "skipped_busy": admitted["skipped_busy"]})
    # #6492: the launched-but-dead population is always reported (0 when clean) so a
    # reader never has to infer "no failures" from an absent key. `size` above excludes
    # them by construction — this is the record of what the wave lost on launch.
    payload["spawn_failed"] = [m.get("spawn_failed") for m in spawn_failed]
    payload["spawn_failed_count"] = len(spawn_failed)
    # #3610: the launch-cache posture is always reported (even off/zero) so an auditor
    # reads the knob's live value from the tick record instead of inferring it.
    payload["launch_cache"] = {"warm_floor": bool(warm_floor),
                               "stagger_s": stagger_s,
                               "staggered_spawns": admitted["staggered"]}
    if admitted["last_preflight"]:
        payload["last_preflight"] = admitted["last_preflight"]
    if admitted["preflight_hint"]:
        payload["preflight_hint"] = admitted["preflight_hint"]

    _apply_wave_verdict(payload, root, cand, size=size, lanes_used=lanes_used,
                        wave_id=wave_id, live=live, refusal=refusal,
                        candidates=candidates, free_seats=free_seats, seats=seats,
                        spawn_failed=spawn_failed)
    return payload


def render_wave(p: dict[str, Any]) -> str:
    seats = (p.get("seats") or {}).get("tags") or []
    busy_sources = p.get("busy_lane_sources") or {}
    busy_detail = []
    if busy_sources.get("inflight_markers"):
        busy_detail.append(f"markers: {', '.join(busy_sources.get('inflight_markers'))}")
    if busy_sources.get("lease_refs"):
        busy_detail.append(f"lease refs: {', '.join(busy_sources.get('lease_refs'))}")
    lines = [
        f"issue-dispatch WAVE: {p.get('verdict')} ({'ok' if p.get('ok') else 'refuse'})  live={p.get('live')}",
        f"  wave_id   : {p.get('wave_id')}  size={p.get('size')}  cap={p.get('cap')}",
        f"  seats     : {p.get('free_seats')} free  ({', '.join(t for t in seats if t) or '-'})",
        f"  candidates: {len(p.get('candidate_lanes') or [])} lane(s) with open issues",
    ]
    if p.get("busy_lanes"):
        detail = "; ".join(busy_detail) or ", ".join(p.get("busy_lanes"))
        lines.append(f"  busy      : {detail} (skipped)")
    for m in p.get("members") or []:
        sp = m.get("spawned") or {}
        tag = (m.get("account") or {}).get("tag") or "-"
        pid = f" pid={sp.get('pid')}" if sp.get("pid") else ""
        lines.append(f"    [{m.get('rank')}] {str(m.get('lane') or '-'):<12} "
                     f"{m.get('issues')} issues  seat={tag}{pid}")
    # #6492: dead-on-launch children are printed as loudly as live ones. A wave that
    # reports only its survivors is how "WAVED (ok)" survived four dead pids.
    for f in p.get("spawn_failed") or []:
        f = f or {}
        lines.append(f"    [x] {str(f.get('lane') or '-'):<12} SPAWN_FAILED "
                     f"cause={f.get('cause')} rc={f.get('returncode')} "
                     f"log_bytes={f.get('log_bytes')} seat={f.get('seat') or '-'}")
    # #3610: the launch-cache posture, so an operator reads whether the wave paid N
    # floor cache-writes or 1 write + (N-1) reads without opening the JSON.
    lc = p.get("launch_cache") or {}
    if lc.get("warm_floor") or lc.get("stagger_s"):
        warm = p.get("warm_floor") or {}
        state = ("warmed" if warm.get("warmed") else
                 "would-warm" if warm.get("would_warm") else
                 f"warm-failed ({warm.get('error')})" if warm.get("error") else "off")
        lines.append(f"  floor     : {state}; stagger={lc.get('stagger_s')}s "
                     f"x{lc.get('staggered_spawns')} launch(es)")
    if p.get("refusal"):
        lines.append(f"  stopped   : {p.get('refusal')}")
    hint = p.get("preflight_hint")
    if isinstance(hint, dict) and hint.get("message"):
        lines.append(f"  hint      : {hint.get('message')}")
    lines.extend(_escape_render_lines(p))
    lines.append(f"  -> {p.get('reason')}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# #3610 box 3: the per-wave floor cache meter (the warm-vs-cold A/B readout).
#
# The wave now leaves a joinable sidecar (_write_wave_artifacts), and `fak guard`
# already appends ONE gateway-usage row per session carrying the provider's own
# cache counters keyed by the guard process pid. Joining those two planes on pid is
# what turns "we shipped a warm knob" into "here is what the warm knob measured".
#
# HONEST BASIS — read this before quoting a number from it. The gateway-usage row is
# a WHOLE-SESSION aggregate: it cannot separate the ~35.8k launch floor from the
# cache writes a worker accrues later in its own turns. So this meter does NOT
# isolate the floor slice. What it does measure is the follower population's cache
# READ SHARE and cache-WRITE volume under warm-vs-cold launches, which is the
# observable the floor effect moves. A warm cohort whose followers read materially
# more than the cold cohort's is the promotion evidence; equal cohorts are the
# demotion evidence. Isolating the floor itself needs a FIRST-TURN usage row, which
# the ledger does not emit today (see the final report's next step).
GATEWAY_USAGE_REL = ".fak/nightrun/gateway-usage.jsonl"
FLOOR_CACHE_SCHEMA = "fleet-wave-floor-cache/1"
# A follower whose session read share clears this is counted as having re-entered a
# warm prefix rather than paid for a fresh one. Deliberately a coarse majority test:
# the aggregate basis above cannot support a tighter threshold honestly.
FLOOR_READER_SHARE = 0.5
FLOOR_CACHE_BASIS = (
    "session-aggregate provider counters joined to wave members by guard pid; "
    "the launch floor is NOT separated from in-session cache writes")


def read_usage_rows(path: Path) -> list[dict[str, Any]]:
    """Parse the gateway-usage JSONL ledger. A missing/unreadable ledger is a clean
    first-run state (empty), never an error — matching the Go reader's fall-open
    posture. A corrupt line is skipped rather than aborting the whole read."""
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("schema") and row.get("pid") is not None:
            rows.append(row)
    return rows


def _usage_for_pid(rows: list[dict[str, Any]], pid: int,
                   since_ms: int | None) -> dict[str, Any] | None:
    """The richest usage row a member's pid resolves to.

    ``since_ms`` is the PID-REUSE GUARD: an OS recycles pids, and the ledger spans
    days, so a row written BEFORE the wave launched cannot belong to this member. The
    last surviving row wins because guard's exit row carries the full session totals.
    """
    best: dict[str, Any] | None = None
    for row in rows:
        if row.get("pid") != pid:
            continue
        stamp = row.get("unix_millis")
        if since_ms is not None and isinstance(stamp, int) and stamp < since_ms:
            continue
        if best is None or (row.get("unix_millis") or 0) >= (best.get("unix_millis") or 0):
            best = row
    return best


def _cache_axes(row: dict[str, Any]) -> dict[str, Any]:
    """Project a usage row onto the three prompt-token axes the floor question needs.

    ``read_share`` uses the SAME denominator as the Go per-session view
    (cmd/fak/dispatch_sessions.go: dispatchSessionCacheReadShare) so the two planes
    never disagree about what "cache read share" means.
    """
    counters = row.get("counters") or {}

    def _n(key: str) -> int:
        val = counters.get(key)
        return int(val) if isinstance(val, (int, float)) else 0

    read, write, uncached = (_n("cached_prompt_tokens"),
                             _n("cache_creation_tokens"), _n("input_tokens"))
    prompt = read + write + uncached
    return {"cache_read": read, "cache_write": write, "uncached_input": uncached,
            "read_share": round(read / prompt, 4) if prompt else None,
            "observed_turns": _n("observed_turns")}


def read_wave_sidecars(runs_dir: Path) -> tuple[list[dict[str, Any]], int]:
    """Every durable per-wave sidecar, oldest launch first, split from the LEGACY ones.

    ``last-wave.json`` is deliberately NOT read: it is overwritten each tick, so it can
    never supply the second arm of an A/B.

    A sidecar written before this issue carries neither the launch-cache posture nor
    the member pids, so it is UNJOINABLE — and, critically, it is not a cold-arm data
    point either: it records no posture at all. Counting those as "cold" would inflate
    the cold arm with dozens of waves that contribute zero followers, which reads as
    evidence when it is silence. They are returned as a COUNT so the readout can say
    how much history is invisible instead of quietly absorbing it.
    """
    try:
        paths = sorted(runs_dir.glob("dispatch-wave-*.json"))
    except OSError:
        return [], 0
    waves: list[dict[str, Any]] = []
    legacy = 0
    for path in paths:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(doc, dict) or not doc.get("wave_id"):
            continue
        if not isinstance(doc.get("members"), list) or not isinstance(
                doc.get("launch_cache"), dict):
            legacy += 1
            continue
        waves.append(doc)
    waves.sort(key=lambda w: w.get("launched_at_ms") or 0)
    return waves, legacy


def _meter_wave(wave: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Join ONE wave's members to their provider cache counters and classify each."""
    launched = wave.get("launched_at_ms")
    since = launched if isinstance(launched, int) else None
    posture = wave.get("launch_cache") or {}
    members: list[dict[str, Any]] = []
    for m in wave.get("members") or []:
        rank, pid = m.get("rank"), m.get("pid")
        rec: dict[str, Any] = {"lane": m.get("lane"), "rank": rank, "pid": pid,
                               "role": "leader" if rank == 0 else "follower"}
        row = _usage_for_pid(rows, pid, since) if isinstance(pid, int) else None
        if row is None:
            rec["matched"] = False
            # Named, not silent: an unmatched member is missing evidence, and a meter
            # that quietly drops it would read as a cleaner result than it earned.
            rec["reason"] = ("no gateway-usage row for this pid at/after launch "
                             "(worker still live, or guard wrote no exit row)")
        else:
            rec["matched"] = True
            rec.update(_cache_axes(row))
            share = rec.get("read_share")
            rec["floor"] = ("unknown" if share is None else
                            "read" if share >= FLOOR_READER_SHARE else "write")
        members.append(rec)
    followers = [m for m in members if m["role"] == "follower" and m["matched"]]
    return {
        "wave_id": wave.get("wave_id"),
        "launched_at_ms": launched,
        "warm_floor": bool(posture.get("warm_floor")),
        "stagger_s": posture.get("stagger_s"),
        "staggered_spawns": posture.get("staggered_spawns"),
        "size": wave.get("size"),
        "members": members,
        "followers_matched": len(followers),
        "followers_reading": sum(1 for m in followers if m.get("floor") == "read"),
    }


def _cohort(waves: list[dict[str, Any]]) -> dict[str, Any]:
    """Fold one A/B arm over its FOLLOWERS only (rank>=1).

    Rank 0 is excluded by construction: someone has to pay the first write, so the
    leader is never evidence either way. The claim under test is strictly about
    workers 2..N.
    """
    followers = [m for w in waves for m in w["members"]
                 if m["role"] == "follower" and m["matched"]]
    shares = [m["read_share"] for m in followers if m.get("read_share") is not None]
    writes = [m["cache_write"] for m in followers if m.get("cache_write") is not None]
    return {
        "waves": len(waves),
        "followers": len(followers),
        "reading": sum(1 for m in followers if m.get("floor") == "read"),
        "mean_read_share": round(sum(shares) / len(shares), 4) if shares else None,
        "mean_cache_write": round(sum(writes) / len(writes), 1) if writes else None,
    }


def wave_floor_cache(root: Path, *, runs_dir: Path | None = None,
                     usage_path: Path | None = None) -> dict[str, Any]:
    """#3610 box 3: the per-wave cache read/write meter and its warm-vs-cold A/B.

    A PURE fold over two on-disk planes (wave sidecars + the gateway-usage ledger):
    same inputs -> same readout, launches nothing, writes nothing, so a test drives it
    hermetically by planting both. Read ``FLOOR_CACHE_BASIS`` before quoting it — the
    counters are whole-session aggregates, so this measures the follower population's
    cache posture, not the floor slice in isolation.
    """
    runs = runs_dir if runs_dir is not None else root / RUNS_DIRNAME
    usage = usage_path if usage_path is not None else root / GATEWAY_USAGE_REL
    rows = read_usage_rows(usage)
    sidecars, legacy = read_wave_sidecars(runs)
    metered = [_meter_wave(w, rows) for w in sidecars]
    warm = [w for w in metered if w["warm_floor"]]
    cold = [w for w in metered if not w["warm_floor"]]
    ab = {"warm": _cohort(warm), "cold": _cohort(cold)}
    doc: dict[str, Any] = {
        "schema": FLOOR_CACHE_SCHEMA,
        "workspace": str(root),
        "usage_ledger": str(usage),
        "usage_rows": len(rows),
        "basis": FLOOR_CACHE_BASIS,
        "reader_share_threshold": FLOOR_READER_SHARE,
        "waves": metered,
        "legacy_waves": legacy,
        "ab": ab,
    }
    doc.update(_floor_cache_verdict(ab))
    return doc


def _floor_cache_verdict(ab: dict[str, Any]) -> dict[str, Any]:
    """Grade the A/B, and REFUSE to grade it when an arm has no evidence.

    An empty arm is the common state (the knob is opt-in and defaults off), so the
    honest verdict there names the missing arm instead of reporting a comparison it
    cannot make. ``ok`` is False for that case: an operator running this to decide
    promotion should get a non-zero exit, not a green-looking blank.
    """
    warm, cold = ab["warm"], ab["cold"]
    if not warm["followers"] or not cold["followers"]:
        missing = "warm" if not warm["followers"] else "cold"
        other = "cold" if missing == "warm" else "warm"
        return {"ok": False, "verdict": "INSUFFICIENT_EVIDENCE",
                "reason": (f"no matched followers in the {missing} arm "
                           f"({other} arm has {ab[other]['followers']}); run a wave "
                           f"with {'--warm-floor' if missing == 'warm' else 'the knob off'} "
                           "and let its workers exit, then re-read")}
    w_share, c_share = warm["mean_read_share"], cold["mean_read_share"]
    if w_share is None or c_share is None:
        return {"ok": False, "verdict": "INSUFFICIENT_EVIDENCE",
                "reason": "matched followers carried no prompt-token counters to compare"}
    if w_share > c_share:
        return {"ok": True, "verdict": "WARM_READS_MORE",
                "reason": (f"warm followers read {w_share:.1%} of their prompt tokens "
                           f"vs {c_share:.1%} cold ({warm['reading']}/{warm['followers']} "
                           f"vs {cold['reading']}/{cold['followers']} over the "
                           f"{FLOOR_READER_SHARE:.0%} bar)")}
    return {"ok": False, "verdict": "NO_MEASURED_EFFECT",
            "reason": (f"warm followers read {w_share:.1%} vs {c_share:.1%} cold — the "
                       "warm pre-request did not move the follower cache posture")}


def render_floor_cache(doc: dict[str, Any]) -> str:
    """Operator readout for the meter."""
    ab = doc.get("ab") or {}
    lines = [f"floor cache meter: {doc.get('verdict')}",
             f"  ledger    : {doc.get('usage_rows')} usage row(s) @ {doc.get('usage_ledger')}",
             f"  basis     : {doc.get('basis')}"]
    if doc.get("legacy_waves"):
        lines.append(f"  legacy    : {doc.get('legacy_waves')} pre-#3610 sidecar(s) "
                     "carry no posture/pids — invisible to the A/B, not cold evidence")
    for arm in ("warm", "cold"):
        c = ab.get(arm) or {}
        share = c.get("mean_read_share")
        lines.append(
            f"  {arm:<9} : {c.get('waves')} wave(s), {c.get('followers')} matched "
            f"follower(s), {c.get('reading')} reading; "
            f"mean read share {'n/a' if share is None else f'{share:.1%}'}, "
            f"mean cache write {c.get('mean_cache_write')}")
    for w in doc.get("waves") or []:
        posture = "warm" if w.get("warm_floor") else "cold"
        floors = ",".join(f"r{m.get('rank')}={m.get('floor') or 'unmatched'}"
                          for m in w.get("members") or [])
        lines.append(f"  {w.get('wave_id')} [{posture}, "
                     f"stagger={w.get('stagger_s') or 0}s] {floors}")
    lines.append(f"  -> {doc.get('reason')}")
    return "\n".join(lines)


def escape_plan(root: Path, safe_priority: dict[str, int] | None) -> dict[str, Any]:
    """The read-only ACTION plan behind ``--escape-self-source``: turn the Arm-1 escape
    SIGNAL into the exact operator command that drains the top held self-source P1 the
    guarded pool cannot reach.

    Composes the three checks an escape must pass, IN ORDER, stopping at the first that
    fails so ``reason`` names the blocker:
      1. a held self-source lane must OUT-SCORE the best dispatchable lane
         (``escape_candidates`` marked it ``preferred``) — else draining a safe lane is
         the better use of the seat;
      2. the build-integrity gate must be CLEAR (no guarded peer's ``go build`` in
         flight, via ``guarded_worker_in_flight``) — else an unguarded self-source commit
         could land on a poisoned build;
      3. the lane must be FREE (no worker already in flight on it, via
         ``busy_lanes``/``lease_ref_busy_lanes``) — the one-worker-per-self-source-lane
         cap the design requires.
    When all three pass it emits ``recommend=True`` plus the unguarded operator command
    (``FLEET_DOGFOOD_GUARD=0`` disables the guard so the worker may edit self-source ON
    MAIN — the supported operator escape, NOT a worktree) and the exclusive lease the
    worker materializes on start. PURE PLAN: never spawns, never mutates — the live
    unguarded spawn is a separate opt-in step."""
    ec = escape_candidates(root, safe_priority)
    ranked = ec["escape_candidates"]
    top = ranked[0] if ranked else None
    gate_clear = not guarded_worker_in_flight(root / RUNS_DIRNAME)
    busy = busy_lanes(root / RUNS_DIRNAME) | set(
        lease_ref_busy_lanes(root).get("lanes") or [])
    plan: dict[str, Any] = {
        "target_lane": None,
        "recommend": False,
        "reason": "",
        "gate": {"clear": gate_clear, "guarded_worker_in_flight": not gate_clear},
        "plan_only": True,
        **ec,
    }
    if not top:
        plan["reason"] = "no held self-source lane with open issues"
        return {"escape_plan": plan}
    if not top.get("preferred"):
        plan["reason"] = (
            f"top held lane {top['lane']} (score {top['score']}) does not out-score the "
            f"best safe lane (score {ec['best_safe_score']}); drain a safe lane instead")
        return {"escape_plan": plan}
    lane = top["lane"]
    lane_free = lane not in busy
    nums = top.get("issue_nums") or []
    plan.update({
        "target_lane": lane,
        "issue_nums": nums,
        "score": top.get("score"),
        "lane_free": lane_free,
        "lease": {"lane": lane, "mode": "exclusive", "tree": top.get("tree") or []},
        "command": dispatch_worker.build_command(lane, "claude"),
        "env_overrides": {"FLEET_DOGFOOD_GUARD": "0"},
    })
    if not gate_clear:
        plan["reason"] = ("a guarded worker's build may be in flight; hold the unguarded "
                          "escape until it clears")
    elif not lane_free:
        plan["reason"] = f"lane {lane} already has a worker in flight; nothing to escape"
    else:
        plan["recommend"] = True
        issues = ", ".join(f"#{n}" for n in nums) or "-"
        plan["reason"] = (
            f"escape to {lane} ({issues}): held score {top['score']} > best safe "
            f"{ec['best_safe_score']}, gate clear, lane free")
    return {"escape_plan": plan}


def render_escape(doc: dict[str, Any]) -> str:
    plan = doc.get("escape_plan") or {}
    lines = [f"escape-self-source: {'RECOMMEND' if plan.get('recommend') else 'HOLD'}"]
    lane = plan.get("target_lane")
    if lane:
        nums = ", ".join(f"#{n}" for n in (plan.get("issue_nums") or [])) or "-"
        gate = plan.get("gate") or {}
        lease = plan.get("lease") or {}
        lines += [
            f"  lane    : {lane} {nums}  score={plan.get('score')} "
            f"vs best safe {plan.get('best_safe_score')}",
            f"  gate    : {'clear' if gate.get('clear') else 'HOLD (guarded build in flight)'}",
            f"  free    : {'yes' if plan.get('lane_free') else 'NO (worker in flight)'}",
            f"  lease   : {lease.get('lane')} [{lease.get('mode')}] "
            f"tree={', '.join(lease.get('tree') or []) or '-'}",
        ]
        if plan.get("recommend"):
            env = " ".join(f"{k}={v}" for k, v in (plan.get("env_overrides") or {}).items())
            cmd = " ".join(plan.get("command") or [])
            lines.append(f"  run     : {env} {cmd}".rstrip())
    lines.append(f"  -> {plan.get('reason')}")
    lines.append("  (plan only — the live unguarded spawn is a follow-on; run the above "
                 "command by hand as the operator escape)")
    return "\n".join(lines)


def _default_max_workers(platform_name: str = sys.platform) -> int:
    """Return the gate-mirrored worker ceiling, with Darwin defaulting to 30.

    FAK_MAX_WORKERS remains the explicit override. The preflight's lower host, RAM,
    thread, seat, account, and target bounds still constrain every spawn.
    """
    raw = os.environ.get("FAK_MAX_WORKERS", "").strip()
    try:
        if raw and int(raw) > 0:
            return int(raw)
    except ValueError:
        pass
    return 30 if platform_name == "darwin" else 20


def main(argv: list[str] | None = None) -> int:
    default_workers = _default_max_workers()
    ap = argparse.ArgumentParser(
        description="One guarded, switcher-routed, bounded dispatch tick (dry-run by default).")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--max-workers", type=int, default=default_workers,
                    help="hard cap on live workers, enforced by the preflight "
                         f"(default: {default_workers}, the preflight ceiling; "
                         "FAK_MAX_WORKERS retunes it)")
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
    ap.add_argument("--escape-self-source", action="store_true",
                    help="PLAN the unguarded escape for the top held self-source P1 the "
                         "guarded pool cannot reach (read-only; emits the FLEET_DOGFOOD_"
                         "GUARD=0 operator command to run by hand). The live spawn is a "
                         "follow-on.")
    ap.add_argument("--no-refresh", action="store_true",
                    help="skip the per-tick account-registry refresh (route off the "
                         "cached snapshot; for inspection / when a fresh scan just ran)")
    ap.add_argument("--warm-floor", action="store_true", default=None,
                    help="#3610: prime the shared ~35.8k floor prefix with ONE "
                         "pre-request before the wave spawns, so members 2..N read "
                         "that cache instead of each paying a full cache-write "
                         f"(default: off; {WARM_FLOOR_ENV} sets it fleet-wide)")
    ap.add_argument("--stagger-s", type=float, default=None,
                    help="#3610: seconds between consecutive wave launches, to space "
                         "members inside the cache TTL (default: 0.0 = today's "
                         f"back-to-back spawn; {LAUNCH_STAGGER_ENV} retunes it). Note "
                         "fak forces the 1h stable-prefix TTL by default, so the "
                         "budget is usually far wider than the provider's 5-min floor")
    ap.add_argument("--floor-cache-meter", action="store_true",
                    help="#3610: READ-ONLY per-wave cache read/write meter — join the "
                         "durable wave sidecars to the gateway-usage ledger by guard "
                         "pid and print the warm-vs-cold A/B over wave FOLLOWERS "
                         "(rank>=1). Launches nothing. Exits non-zero when an arm has "
                         "no matched followers, so it cannot report a blank as a pass")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args(argv)

    root = Path(args.workspace).resolve() if args.workspace else repo_root()
    if args.floor_cache_meter:
        doc = wave_floor_cache(root)
        print(json.dumps(doc, indent=2) if args.json else render_floor_cache(doc))
        return 0 if doc.get("ok") else 1
    if args.escape_self_source:
        pick = pick_lane(root, None)
        doc = escape_plan(root, pick.get("priority_by_lane"))
        print(json.dumps(doc, indent=2) if args.json else render_escape(doc))
        return 0 if doc["escape_plan"].get("recommend") else 1
    if args.wave:
        payload = evaluate_wave(root, max_workers=args.max_workers,
                                work_kind=args.work_kind, live=args.live,
                                refresh=not args.no_refresh,
                                warm_floor=args.warm_floor,
                                stagger_s=args.stagger_s)
        print(json.dumps(payload, indent=2) if args.json else render_wave(payload))
        return 0 if payload.get("ok") else 1
    payload = evaluate(root, max_workers=args.max_workers, work_kind=args.work_kind,
                       lane=args.lane, live=args.live, refresh=not args.no_refresh)
    print(json.dumps(payload, indent=2) if args.json else render(payload))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
