#!/usr/bin/env python3
"""Auto-push trunk when — and ONLY when — push-lag has clearly stalled.

The witnessed issue closer (tools/issue_resolve_witnessed.py, cron
FleetResolveProgress) only closes an issue whose resolving commit is an ANCESTOR
of origin/main (correct: don't close an issue whose fix isn't durably on GitHub).
So when push-to-origin silently stalls, workers keep committing locally,
`witnessed_open` climbs, but `would_close` stays 0 and closure goes flat — the
loop looks dead when only the push step died. On 2026-07-07 a ~7h push gap froze
closure for hours; draining it once pushes resumed booked 24 closures at once.

fresh_status.py already DETECTS this (`panes.git.push_lag_stale` trips past 45
min), but nothing remediated it. This tick is that backstop: on push-lag stale it
runs the repo's existing safe-push primitive (`fak sync push`) — which
fast-forwards origin/main, retries only transient races, and HALTS (never
merges/forces) on a genuine behind/diverged trunk or a pre-push gate rejection.
It never bypasses that safety, never fires on a merely-dirty tree, and stays
dormant in steady state (a working pusher keeps lag under threshold).

EDGE-TRIGGERED, NOT BLIND POLLING (#6511). A 15-minute level-triggered cadence
measured 232 ticks at 85.3% no-op, 10.8% rejected and 3.9% pushed — and a rejected
push re-fired every 15 minutes for ~5.5 hours while Task Scheduler still reported
result 0. Three things fix that, all in the pure seams below:

  - admit() fires on the ahead-of-origin EVENT (the unpushed tip changed since the
    last live attempt), so re-observing the same unpushed tip is NO_EVENT, not a
    fresh push attempt;
  - a rejected push parks the backstop for backoff_seconds(streak) — 30m, 1h, 2h,
    ... capped at 6h — and records the TYPED guard reason from `fak sync push`
    (BEHIND, PUSH_ERROR, RETRIES_EXHAUSTED, ...) so the divergence that blocked
    delivery is in the ledger instead of a generic lag;
  - every tick carries a typed health (ok / degraded / failed) whose exit code
    (0 / 2 / 1) becomes LastTaskResult, so a failed or parked push can never read
    as a green scheduler run.

`--metrics` folds the JSONL ledger into pushes/tick, no-op rate, failure streaks,
lag reduced and cost. THAT is the evidence the tombstone registry requires before
this task is re-enabled on a cadence: re-enable only if event-driven operation
cannot hold the push-lag SLO. Each finishing worker pushing its own lane is still
the primary path; this stays the on-demand backstop.

SAFE BY DEFAULT: without --live it only reports what it WOULD push.

  python tools/auto_push_on_lag.py --workspace . --json            # dry-run report
  python tools/auto_push_on_lag.py --workspace . --json --live     # arm the backstop
  python tools/auto_push_on_lag.py --workspace . --push-lag-mins 0 --json  # force the trigger path (still dry)
  python tools/auto_push_on_lag.py --workspace . --metrics --json   # yield/cost of the ledger so far
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dispatch_worker import install_no_window_subprocess_defaults  # noqa: E402
from fresh_status import git_pane, repo_root, DEFAULT_PUSH_LAG_ACTION_SECONDS  # noqa: E402

install_no_window_subprocess_defaults(subprocess)

SCHEMA = "fak-auto-push-on-lag/1"
STATE_SCHEMA = "fak-auto-push-on-lag-state/1"
METRICS_SCHEMA = "fak-auto-push-on-lag-metrics/1"
LOG_REL = ".dispatch-runs/auto-push.jsonl"
# The tick-to-tick memory that turns a level-triggered poll into an EDGE-triggered
# backstop: the unpushed tip we last actually attempted, the consecutive-rejection
# streak with its parking deadline, and the cumulative yield/cost counters. It lives
# beside the ledger (untracked .dispatch-runs/), so losing it is harmless — an absent
# state simply means the next stale tip reads as a fresh event.
STATE_REL = ".dispatch-runs/auto-push-state.json"
# 600, not 300: the pre-push build gate (`fak hooks pre-push`) materializes the full
# committed tip via `git archive HEAD | tar -x` (~130MB in this repo) BEFORE it builds,
# and under shared-trunk I/O contention that materialize alone was observed to exceed
# 300s — timing out an otherwise-clean fast-forward as a false PUSH_FAILED (push-timeout,
# rc124). 600s clears the contended-materialize window while staying well under the
# 15-min cron period, so a slow tick still never overlaps the next.
PUSH_TIMEOUT_SECONDS = 600

# Backoff after a REJECTED push. The observed pathology was a rejection re-fired
# every 15 minutes for ~5.5 hours (22 identical failures): the trunk was genuinely
# behind/diverged or a pre-push gate was refusing, and no amount of re-pushing was
# going to change that. Doubling from 30m to a 6h cap turns that streak into ~4
# attempts while still draining the lag on its own once the blocker clears.
BACKOFF_BASE_SECONDS = 30 * 60
BACKOFF_MAX_SECONDS = 6 * 60 * 60

# Typed tick health. `degraded` is the state the old boolean `ok` could not say:
# the lag is still there and the backstop is deliberately NOT acting (parked in
# backoff, or fak is unavailable). Task Scheduler stores the exit code as
# LastTaskResult, so each of these is independently visible to an operator.
HEALTH_OK = "ok"
HEALTH_DEGRADED = "degraded"
HEALTH_FAILED = "failed"

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_DEGRADED = 2

_HEALTH_BY_VERDICT = {
    "NO_LAG": HEALTH_OK,        # nothing waiting — the steady state
    "NOT_ON_MAIN": HEALTH_OK,   # not this loop's business
    "NO_EVENT": HEALTH_OK,      # same unpushed tip we already attempted; no new event
    "WOULD_PUSH": HEALTH_OK,    # dry-run report
    "PUSHED": HEALTH_OK,        # the lag was drained
    "BACKOFF": HEALTH_DEGRADED,  # lag persists and we are parked after a rejection
    "SKIPPED": HEALTH_DEGRADED,  # lag persists and we could not even try safely
    "PUSH_FAILED": HEALTH_FAILED,
}

_EXIT_BY_HEALTH = {
    HEALTH_OK: EXIT_OK,
    HEALTH_DEGRADED: EXIT_DEGRADED,
    HEALTH_FAILED: EXIT_FAILED,
}

# Verdicts the event/backoff gate suppressed. Counted separately from the plain
# NO_LAG steady state so `--metrics` can show what edge-triggering actually saved.
_SUPPRESSED_VERDICTS = {"BACKOFF", "NO_EVENT"}


def health_of(verdict: str) -> str:
    """Typed health for a verdict. An unknown verdict is degraded, never ok — a
    verdict nobody classified is exactly the false-green case this replaces."""
    return _HEALTH_BY_VERDICT.get(verdict, HEALTH_DEGRADED)


def exit_code(health: str) -> int:
    """Process exit code for a typed health, i.e. the task's LastTaskResult."""
    return _EXIT_BY_HEALTH.get(health, EXIT_DEGRADED)


def backoff_seconds(streak: int) -> int:
    """How long to park the backstop after `streak` consecutive rejected pushes.

    Doubles from BACKOFF_BASE_SECONDS and saturates at BACKOFF_MAX_SECONDS. The
    shift is clamped before exponentiation so a long-running stall cannot build a
    thousand-bit integer just to be min()'d back down to the cap.
    """
    if streak <= 0:
        return 0
    shift = min(streak - 1, 16)
    return min(BACKOFF_BASE_SECONDS << shift, BACKOFF_MAX_SECONDS)


def push_guard_reason(pr: dict[str, Any] | None) -> str:
    """The TYPED cause a push did not land, for the ledger and the backoff record.

    Prefers the safe-push primitive's own closed vocabulary (`fak sync push --json`
    emits PushResult.reason: BEHIND / PUSH_ERROR / RETRIES_EXHAUSTED /
    REMOTE_UNREACHABLE / GIT_UNAVAILABLE / ...), appending its classified
    divergence when there is one, and falls back to our own transport-level cause
    (push-timeout / push-oserror) when the child produced no parsable JSON. Without
    this the ledger only ever said "push-rejected", so a diverged trunk and a
    refusing pre-push gate — which need completely different human actions — looked
    identical for hours.
    """
    if not pr or pr.get("pushed"):
        return ""
    parsed = pr.get("result")
    if isinstance(parsed, dict):
        typed = str(parsed.get("reason") or "").strip()
        if typed:
            divergence = str(parsed.get("divergence") or "").strip()
            return f"{typed}:{divergence}" if divergence else typed
    return str(pr.get("reason") or "").strip() or "unknown"


def resolve_fak(root: Path) -> list[str] | None:
    """The fak-binary ladder used across the Fleet PS1 registrars, in Python.

    $FAK_BIN -> <root>/fak(.exe) -> `fak` on PATH -> `go run ./cmd/fak` (only if
    `go` is present). Returns the command prefix, or None if nothing resolves —
    in which case we SKIP rather than fall back to a raw `git push` that would
    bypass the safe-push primitive and its pre-push gate.
    """
    env = os.environ.get("FAK_BIN", "").strip()
    if env and Path(env).exists():
        return [env]
    for name in ("fak.exe", "fak"):
        cand = root / name
        if cand.exists():
            return [str(cand)]
    which = shutil.which("fak")
    if which:
        return [which]
    if shutil.which("go"):
        return ["go", "run", "./cmd/fak"]
    return None


def decide(pane: dict[str, Any]) -> tuple[bool, str]:
    """Pure admission: should we push, and why/why-not. The unit-test seam.

    Gate STRICTLY on push_lag_stale (never dirty_lag_stale) so a dirty working
    tree — the far more common ACTION cause — can never trigger a push.
    """
    if pane.get("branch") != "main":
        return (False, "not-on-main")
    if not pane.get("push_lag_stale"):
        return (False, "no-push-lag")
    if not pane.get("ahead"):
        return (False, "nothing-ahead")
    mins = (pane.get("push_lag_seconds") or 0) // 60
    return (True, f"push-lag-{mins}m")


class Admission(NamedTuple):
    """The full tick decision: act or not, the typed verdict, and why."""

    should: bool
    verdict: str
    reason: str


def admit(pane: dict[str, Any], state: dict[str, Any], now_ts: int) -> Admission:
    """Event-driven admission: decide() plus the backoff and edge gates.

    decide() answers the level question ("is there stale unpushed work?"). That
    alone is what made the cadence blind: the same stale tip re-admitted every 15
    minutes forever. admit() adds the two memory-dependent gates, in order:

      - BACKOFF: a previous push was REJECTED and its parking window has not
        elapsed. Carries the streak and the typed guard reason so the operator sees
        WHAT is blocking delivery, not just that it is blocked.
      - NO_EVENT: the unpushed tip is byte-identical to the one we last actually
        attempted, and nothing has failed since. There has been no ahead-of-origin
        transition, so there is nothing new to deliver.

    The streak==0 guard on NO_EVENT is what makes backoff a RETRY rather than a
    permanent park: once the window elapses the same tip is admitted again, so a
    stall that clears on its own still drains without a new commit to trigger it.
    """
    should, reason = decide(pane)
    if not should:
        return Admission(False, "NOT_ON_MAIN" if reason == "not-on-main" else "NO_LAG", reason)
    streak = _as_int(state.get("fail_streak"))
    until = _as_int(state.get("backoff_until"))
    if streak > 0 and now_ts < until:
        mins = max(1, (until - now_ts + 59) // 60)
        guard = str(state.get("guard_reason") or "unknown")
        return Admission(False, "BACKOFF", f"backoff-{mins}m-after-{streak}-rejects:{guard}")
    tip = str(pane.get("sha") or "")
    if streak == 0 and tip and tip == str(state.get("tip") or ""):
        return Admission(False, "NO_EVENT", f"no-new-commit-since-{tip}")
    return Admission(True, "WOULD_PUSH", reason)


def _as_int(value: Any) -> int:
    """Coerce a state field to int, treating anything unusable as 0. The state file
    is untracked scratch that a killed tick can leave half-written, so a corrupt
    field must degrade to "no memory", never crash the backstop."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def load_state(root: Path) -> dict[str, Any]:
    """Read the tick memory. A missing, unreadable or non-object state is {} — the
    backstop then behaves exactly as it did before it had a memory."""
    try:
        raw = (root / STATE_REL).read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        state = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return state if isinstance(state, dict) else {}


def save_state(root: Path, state: dict[str, Any]) -> None:
    """Best-effort persist of the tick memory, same posture as the JSONL ledger: a
    write failure must never turn a healthy no-op into a failed tick."""
    try:
        path = root / STATE_REL
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    except OSError:
        pass


def push_child_env(env: dict[str, str] | None = None) -> dict[str, str]:
    """The child environment for `fak sync push` — with the WARN-ONLY build gate
    forced OFF so its tip-materialize can never time out delivery (#3300).

    The pre-push build gate (`fak hooks pre-push`, env `FLEET_BUILD_GUARD`)
    materializes the full committed tip (~130MB in this repo) via
    `git archive HEAD | tar -x` BEFORE it builds. In its DEFAULT `warn` mode that
    materialize is pure cost: it burns the whole push budget (observed >300s under
    shared-trunk I/O contention → rc124 push-timeout) only to emit an ADVISORY that
    never blocks the push anyway. During this unattended backstop we therefore force
    `FLEET_BUILD_GUARD=off` whenever the gate is NOT hard-enforcing, dropping the
    materialize entirely; warn already lets a compile break reach origin, so nothing
    the gate was catching is lost, and `make ci` stays the authoritative build oracle.

    The one mode we DON'T touch is `block`: there an operator has made the gate
    enforcing, so a real `TRUNK_WOULD_NOT_COMPILE` must still refuse the push. We
    respect that and pay the materialize (the 600s timeout covers it).
    """
    src = os.environ if env is None else env
    child = dict(src)
    mode = child.get("FLEET_BUILD_GUARD", "warn").strip().lower()
    if mode != "block":
        child["FLEET_BUILD_GUARD"] = "off"
    return child


def push_main(root: Path, fak: list[str]) -> dict[str, Any]:
    """Delegate to the repo's safe-push primitive. Never a raw `git push`.

    `fak sync push` fast-forwards origin/main, retries only transient non-ff
    races / network blips, halts on genuine behind/diverged (Reason=BEHIND), and
    surfaces a pre-push hook rejection as PUSH_ERROR — all of which correctly
    LEAVE the lag in place for a human. We treat the process exit code as the
    authoritative success signal and attach the parsed JSON verbatim.

    Runs with `push_child_env()` so a warn-only build gate is skipped (#3300): its
    slow tip-materialize would otherwise burn the push timeout to produce an
    advisory that can't block delivery anyway.
    """
    cmd = [*fak, "sync", "push", "--remote", "origin", "--branch", "main",
           "--repo", str(root), "--json"]
    try:
        p = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True,
                           timeout=PUSH_TIMEOUT_SECONDS, env=push_child_env())
    except subprocess.TimeoutExpired:
        return {"ok": False, "pushed": False, "reason": "push-timeout",
                "returncode": 124}
    except OSError as exc:
        return {"ok": False, "pushed": False, "reason": f"push-oserror: {exc}",
                "returncode": 127}
    parsed: Any = None
    out = (p.stdout or "").strip()
    if out:
        try:
            parsed = json.loads(out)
        except json.JSONDecodeError:
            parsed = None
    ok = p.returncode == 0
    return {
        "ok": ok,
        "pushed": ok,
        "returncode": p.returncode,
        "reason": "pushed" if ok else "push-rejected",
        "result": parsed,
        "stderr": (p.stderr or "").strip()[-800:] if not ok else "",
    }


def _append_log(root: Path, record: dict[str, Any]) -> None:
    """Best-effort JSONL breadcrumb so a future stall is visible in a log rather
    than silent (the same observability pattern as issue_resolve_progress.py)."""
    try:
        log = root / LOG_REL
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, separators=(",", ":")) + "\n")
    except OSError:
        pass


def run(root: Path, *, live: bool, push_lag_mins: int,
        now: datetime | None = None) -> dict[str, Any]:
    """Read the git pane, admit, and (only in --live) push. Returns the report.

    Also folds this tick into the persisted memory: the tip that was attempted, the
    rejection streak and its parking deadline, and the cumulative yield/cost
    counters `--metrics` reports against.
    """
    started = time.monotonic()
    pane = git_pane(root, now=now, push_lag_action_seconds=push_lag_mins * 60)
    moment = now or datetime.now(timezone.utc)
    now_ts = int(moment.timestamp())
    generated_at = moment.strftime("%Y-%m-%dT%H:%M:%SZ")
    state = load_state(root)
    adm = admit(pane, state, now_ts)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "ok": True,
        "verdict": adm.verdict,
        "health": HEALTH_OK,
        "live": live,
        "branch": pane.get("branch"),
        "sha": pane.get("sha"),
        "ahead": pane.get("ahead"),
        "behind": pane.get("behind"),
        "push_lag_seconds": pane.get("push_lag_seconds"),
        "push_lag_stale": pane.get("push_lag_stale"),
        "push_result": None,
        "guard_reason": "",
        "fail_streak": _as_int(state.get("fail_streak")),
        "backoff_seconds": 0,
        "lag_reduced_seconds": 0,
        "reason": adm.reason,
        "generated_at": generated_at,
    }

    if adm.should and not live:
        # Dry run: report, and deliberately DON'T record the tip. A dry tick that
        # claimed the event would suppress the next live one and hide a real stall.
        result["verdict"] = "WOULD_PUSH"
    elif adm.should:
        fak = resolve_fak(root)
        if fak is None:
            result["verdict"] = "SKIPPED"
            result["reason"] = "fak-unavailable"
        else:
            pr = push_main(root, fak)
            result["push_result"] = pr
            pushed = bool(pr.get("pushed"))
            result["verdict"] = "PUSHED" if pushed else "PUSH_FAILED"
            if pushed:
                # The lag this push actually drained — the loop's only real yield.
                result["lag_reduced_seconds"] = _as_int(pane.get("push_lag_seconds"))
                result["fail_streak"] = 0
                state.update(fail_streak=0, backoff_until=0, guard_reason="")
            else:
                if pr.get("reason"):
                    # Surface the REAL push-failure cause (push-rejected / push-timeout /
                    # push-oserror) as the reason, NOT the stale decide reason ("push-lag-NNm"):
                    # the JSONL breadcrumb below logs result["reason"], so without this a
                    # multi-hour stall reads as a generic lag and hides WHICH step died — a
                    # rejected pre-push gate vs a timed-out tip-materialize vs a credential hang.
                    result["reason"] = pr["reason"]
                guard = push_guard_reason(pr)
                streak = _as_int(state.get("fail_streak")) + 1
                parked = backoff_seconds(streak)
                state.update(fail_streak=streak, backoff_until=now_ts + parked,
                             guard_reason=guard)
                result["guard_reason"] = guard
                result["fail_streak"] = streak
                result["backoff_seconds"] = parked
            # Only a LIVE attempt records the tip, so the next identical tip is a
            # genuine no-event rather than an untried one.
            state["tip"] = str(pane.get("sha") or "")
            state["attempted_at"] = now_ts
    elif adm.verdict == "BACKOFF":
        result["guard_reason"] = str(state.get("guard_reason") or "")
        result["backoff_seconds"] = max(0, _as_int(state.get("backoff_until")) - now_ts)

    if not pane.get("ahead"):
        # Nothing unpushed at all: whatever blocked the last push is moot, so a
        # parked backoff must not outlive it and stall the NEXT real event.
        state.update(fail_streak=0, backoff_until=0, guard_reason="")
        result["fail_streak"] = 0

    health = health_of(result["verdict"])
    result["health"] = health
    result["ok"] = health == HEALTH_OK
    result["duration_ms"] = int((time.monotonic() - started) * 1000)
    result["counters"] = _fold_counters(state, result, generated_at)
    save_state(root, state)

    _append_log(root, {
        "ts": generated_at, "verdict": result["verdict"], "live": live,
        "health": health, "branch": result["branch"], "sha": result["sha"],
        "ahead": result["ahead"],
        "push_lag_seconds": result["push_lag_seconds"], "reason": result["reason"],
        "guard_reason": result["guard_reason"], "fail_streak": result["fail_streak"],
        "backoff_seconds": result["backoff_seconds"],
        "lag_reduced_seconds": result["lag_reduced_seconds"],
        "duration_ms": result["duration_ms"],
        "pushed": bool(result["push_result"] and result["push_result"].get("pushed")),
    })
    return result


def _fold_counters(state: dict[str, Any], result: dict[str, Any],
                   generated_at: str) -> dict[str, Any]:
    """Accumulate this tick into the lifetime counters, in place on `state`.

    These are the four numbers the loop was never able to answer for itself:
    pushes per tick (yield), the rejection streak, the lag actually drained, and
    the wall-clock cost of running at all.
    """
    counters = dict(state.get("counters") or {})
    verdict = result["verdict"]
    counters["ticks"] = _as_int(counters.get("ticks")) + 1
    counters["pushes"] = _as_int(counters.get("pushes")) + (1 if verdict == "PUSHED" else 0)
    counters["failures"] = _as_int(counters.get("failures")) + (1 if verdict == "PUSH_FAILED" else 0)
    counters["suppressed"] = _as_int(counters.get("suppressed")) + (
        1 if verdict in _SUPPRESSED_VERDICTS else 0)
    counters["lag_reduced_seconds"] = (_as_int(counters.get("lag_reduced_seconds"))
                                       + result["lag_reduced_seconds"])
    counters["cost_ms"] = _as_int(counters.get("cost_ms")) + result["duration_ms"]
    counters["max_fail_streak"] = max(_as_int(counters.get("max_fail_streak")),
                                      result["fail_streak"])
    counters["pushes_per_tick"] = round(counters["pushes"] / counters["ticks"], 4)
    state["counters"] = counters
    state["schema"] = STATE_SCHEMA
    state["updated_at"] = generated_at
    return counters


def read_ledger(root: Path) -> list[dict[str, Any]]:
    """Every well-formed JSONL record the backstop has written. Malformed lines are
    skipped rather than fatal — a truncated tail from a killed tick must not make
    the whole yield unreadable."""
    try:
        text = (root / LOG_REL).read_text(encoding="utf-8")
    except OSError:
        return []
    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            records.append(rec)
    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Fold the ledger into the yield/cost measurement (#6511 done condition 4).

    `failure_rate` is deliberately over ACTING ticks (pushes + failures), not over
    all ticks: a backstop that correctly sits idle must not be able to dilute its
    own failure rate toward zero simply by ticking more often. `no_op_rate` covers
    that separately, and is the number that condemned the 15-minute cadence.
    """
    ticks = len(records)
    pushes = failures = suppressed = 0
    lag_reduced = cost_ms = 0
    max_streak = current_streak = 0
    for rec in records:
        verdict = str(rec.get("verdict") or "")
        if verdict == "PUSHED":
            pushes += 1
        if verdict == "PUSH_FAILED":
            failures += 1
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
        if verdict in _SUPPRESSED_VERDICTS:
            suppressed += 1
        lag_reduced += _as_int(rec.get("lag_reduced_seconds"))
        cost_ms += _as_int(rec.get("duration_ms"))
    acting = pushes + failures
    return {
        "schema": METRICS_SCHEMA,
        "ticks": ticks,
        "pushes": pushes,
        "failures": failures,
        "suppressed": suppressed,
        "no_op_ticks": ticks - acting,
        "pushes_per_tick": round(pushes / ticks, 4) if ticks else 0.0,
        "no_op_rate": round((ticks - acting) / ticks, 4) if ticks else 0.0,
        "failure_rate": round(failures / acting, 4) if acting else 0.0,
        "max_fail_streak": max_streak,
        "current_fail_streak": current_streak,
        "lag_reduced_seconds": lag_reduced,
        "cost_ms": cost_ms,
        "cost_ms_per_push": round(cost_ms / pushes, 1) if pushes else None,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace", default="", help="repo root (default: fak repo root)")
    ap.add_argument("--push-lag-mins", type=int,
                    default=DEFAULT_PUSH_LAG_ACTION_SECONDS // 60,
                    help="trip the backstop past this many minutes of push lag (default 45)")
    ap.add_argument("--live", action="store_true",
                    help="actually push (default: report what it WOULD push)")
    ap.add_argument("--metrics", action="store_true",
                    help="report yield/cost from the ledger instead of ticking")
    ap.add_argument("--json", action="store_true", help="emit the machine-readable report")
    args = ap.parse_args(argv)

    root = Path(args.workspace).resolve() if args.workspace else repo_root()

    if args.metrics:
        # Read-only: never ticks, never pushes, so it is safe to run at any time.
        summary = summarize(read_ledger(root))
        if args.json:
            print(json.dumps(summary, indent=2))
        else:
            print(f"[auto-push metrics] ticks={summary['ticks']} "
                  f"pushes={summary['pushes']} ({summary['pushes_per_tick']}/tick) "
                  f"no-op={summary['no_op_rate']} failures={summary['failures']} "
                  f"max-streak={summary['max_fail_streak']} "
                  f"lag-reduced={summary['lag_reduced_seconds'] // 60}m "
                  f"cost={summary['cost_ms'] // 1000}s")
        return EXIT_OK

    result = run(root, live=args.live, push_lag_mins=args.push_lag_mins)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        pl = result["push_lag_seconds"]
        lag = f"{pl // 60}m" if isinstance(pl, int) else "n/a"
        mode = "LIVE" if result["live"] else "dry-run"
        print(f"[auto-push {mode}] {result['verdict']}/{result['health']} — {result['reason']} "
              f"(branch={result['branch']}, ahead={result['ahead']}, lag={lag})")
    # Typed exit: 0 healthy, 2 degraded (lag persists, deliberately not acting),
    # 1 failed. Task Scheduler stores this as LastTaskResult, so a rejected push
    # can no longer report a green run.
    return exit_code(result["health"])


if __name__ == "__main__":
    raise SystemExit(main())
