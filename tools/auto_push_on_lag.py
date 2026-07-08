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

SAFE BY DEFAULT: without --live it only reports what it WOULD push.

  python tools/auto_push_on_lag.py --workspace . --json            # dry-run report
  python tools/auto_push_on_lag.py --workspace . --json --live     # arm the backstop
  python tools/auto_push_on_lag.py --workspace . --push-lag-mins 0 --json  # force the trigger path (still dry)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dispatch_worker import install_no_window_subprocess_defaults  # noqa: E402
from fresh_status import git_pane, repo_root, DEFAULT_PUSH_LAG_ACTION_SECONDS  # noqa: E402

install_no_window_subprocess_defaults(subprocess)

SCHEMA = "fak-auto-push-on-lag/1"
LOG_REL = ".dispatch-runs/auto-push.jsonl"
# 600, not 300: the pre-push build gate (`fak hooks pre-push`) materializes the full
# committed tip via `git archive HEAD | tar -x` (~130MB in this repo) BEFORE it builds,
# and under shared-trunk I/O contention that materialize alone was observed to exceed
# 300s — timing out an otherwise-clean fast-forward as a false PUSH_FAILED (push-timeout,
# rc124). 600s clears the contended-materialize window while staying well under the
# 15-min cron period, so a slow tick still never overlaps the next.
PUSH_TIMEOUT_SECONDS = 600

# Verdicts that mean "the backstop needs a human's eye" (lag persists). Everything
# else is a healthy no-op or a completed push. Exit code is 0 iff ok.
_NOT_OK_VERDICTS = {"PUSH_FAILED", "SKIPPED"}


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


def push_main(root: Path, fak: list[str]) -> dict[str, Any]:
    """Delegate to the repo's safe-push primitive. Never a raw `git push`.

    `fak sync push` fast-forwards origin/main, retries only transient non-ff
    races / network blips, halts on genuine behind/diverged (Reason=BEHIND), and
    surfaces a pre-push hook rejection as PUSH_ERROR — all of which correctly
    LEAVE the lag in place for a human. We treat the process exit code as the
    authoritative success signal and attach the parsed JSON verbatim.
    """
    cmd = [*fak, "sync", "push", "--remote", "origin", "--branch", "main",
           "--repo", str(root), "--json"]
    try:
        p = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True,
                           timeout=PUSH_TIMEOUT_SECONDS)
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
    """Read the git pane, decide, and (only in --live) push. Returns the report."""
    pane = git_pane(root, now=now, push_lag_action_seconds=push_lag_mins * 60)
    should, reason = decide(pane)
    generated_at = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "ok": True,
        "verdict": "NO_LAG",
        "live": live,
        "branch": pane.get("branch"),
        "ahead": pane.get("ahead"),
        "behind": pane.get("behind"),
        "push_lag_seconds": pane.get("push_lag_seconds"),
        "push_lag_stale": pane.get("push_lag_stale"),
        "push_result": None,
        "reason": reason,
        "generated_at": generated_at,
    }

    if not should:
        result["verdict"] = "NOT_ON_MAIN" if reason == "not-on-main" else "NO_LAG"
    elif not live:
        result["verdict"] = "WOULD_PUSH"
    else:
        fak = resolve_fak(root)
        if fak is None:
            result["verdict"] = "SKIPPED"
            result["reason"] = "fak-unavailable"
        else:
            pr = push_main(root, fak)
            result["push_result"] = pr
            result["verdict"] = "PUSHED" if pr.get("pushed") else "PUSH_FAILED"
            if not pr.get("pushed") and pr.get("reason"):
                # Surface the REAL push-failure cause (push-rejected / push-timeout /
                # push-oserror) as the reason, NOT the stale decide reason ("push-lag-NNm"):
                # the JSONL breadcrumb below logs result["reason"], so without this a
                # multi-hour stall reads as a generic lag and hides WHICH step died — a
                # rejected pre-push gate vs a timed-out tip-materialize vs a credential hang.
                result["reason"] = pr["reason"]

    result["ok"] = result["verdict"] not in _NOT_OK_VERDICTS
    _append_log(root, {
        "ts": generated_at, "verdict": result["verdict"], "live": live,
        "branch": result["branch"], "ahead": result["ahead"],
        "push_lag_seconds": result["push_lag_seconds"], "reason": result["reason"],
        "pushed": bool(result["push_result"] and result["push_result"].get("pushed")),
    })
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace", default="", help="repo root (default: fak repo root)")
    ap.add_argument("--push-lag-mins", type=int,
                    default=DEFAULT_PUSH_LAG_ACTION_SECONDS // 60,
                    help="trip the backstop past this many minutes of push lag (default 45)")
    ap.add_argument("--live", action="store_true",
                    help="actually push (default: report what it WOULD push)")
    ap.add_argument("--json", action="store_true", help="emit the machine-readable report")
    args = ap.parse_args(argv)

    root = Path(args.workspace).resolve() if args.workspace else repo_root()
    result = run(root, live=args.live, push_lag_mins=args.push_lag_mins)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        pl = result["push_lag_seconds"]
        lag = f"{pl // 60}m" if isinstance(pl, int) else "n/a"
        mode = "LIVE" if result["live"] else "dry-run"
        print(f"[auto-push {mode}] {result['verdict']} — {result['reason']} "
              f"(branch={result['branch']}, ahead={result['ahead']}, lag={lag})")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
