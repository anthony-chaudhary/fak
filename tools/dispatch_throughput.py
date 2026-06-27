#!/usr/bin/env python3
r"""Closed-issues-per-hour throughput meter — the observable the loop was missing.

The dispatch loop's operating GOAL is a sustained close RATE (issues/hour), but
every existing surface measures a STOCK (the open backlog) or HONESTY
(``closure_rate`` = resolved/claimed) — never the RATE. Without it the question
"are we actually closing ~10 issues/hour?" is unanswerable from evidence, so a
wedged loop (see the #517 cap=0 / opencode-tier regressions) can sit dead for
hours while every other card still reads "healthy backlog, honest closures".

This meter folds two read-only surfaces and grades them against a target:

  * gh ``closedAt`` timestamps — the GROUND TRUTH of when issues actually closed,
    bucketed into trailing windows (1h/3h/6h/12h/24h) and divided by the window
    to a closes/hour rate. It counts EVERY close (loop, peer, human), because
    that is the real backlog-drain an operator sees; ``stateReason=COMPLETED``
    is tracked separately as the *productive* (resolved, not wontfix/dup) rate.
  * the loop's own ``.dispatch-runs/progress.jsonl`` close-arm records — each
    tick's ``closed_now`` summed over the same windows into a LOOP-ATTRIBUTED
    rate. This separates the dispatcher's productive throughput from foreign
    closes: if the gh-rate runs far above the loop-rate, something other than
    the loop is draining the backlog; if the loop-rate collapses while gh-rate
    holds, the loop is wedged and humans are carrying it.

Verdict grades the PRIMARY window (default 6h — long enough to smooth the cron's
bursty 10/15-min cadence, recent enough to reflect current state) against
``--target-per-hour`` (default 10): ON_TRACK when met, BELOW_TARGET with the
gap, WARMING_UP when the loop was recently unwedged and the short window is
already climbing back. Read-only — launches nothing, writes nothing (except an
optional ``--md`` block).

A rate needs a clock, so ``now`` is INJECTED for testability: every pure grader
takes ``now_ts``; only the I/O seam reads the wall clock.

    python tools/dispatch_throughput.py                 # the card
    python tools/dispatch_throughput.py --json          # machine-readable
    python tools/dispatch_throughput.py --target-per-hour 10 --primary-window 6
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except (AttributeError, ValueError):
    pass

SCHEMA = "fleet-dispatch-throughput/1"
RUNS_DIRNAME = ".dispatch-runs"
PROGRESS_LOG = "progress.jsonl"
DEFAULT_WINDOWS = (1, 3, 6, 12, 24)
DEFAULT_TARGET_PER_HOUR = 10.0
DEFAULT_PRIMARY_WINDOW = 6
# Below this much wall-clock since the loop's last attributed close, a short
# primary window that already meets target reads as WARMING_UP rather than a
# settled ON_TRACK — the rate is recovering from a stall, not yet sustained.
_WARMUP_RECENT_MIN = 30


def repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    return here.parent.parent


def _py() -> str:
    return sys.executable or "python"


# ---------------------------------------------------------------------------
# I/O seams (real implementations; tests pass fakes / synthetic data)
# ---------------------------------------------------------------------------

ClosedFetcher = Callable[[Path, int], list[dict[str, Any]]]


def fetch_closed_issues(root: Path, limit: int = 400) -> list[dict[str, Any]]:
    """Closed issues with their close timestamp + reason — the ground truth."""
    try:
        proc = subprocess.run(
            # sort:updated-desc is load-bearing: `gh issue list` defaults to
            # creation order, so a bare --limit returns the most-recently-CREATED
            # closed issues and MISSES recently-closed OLD-numbered ones (e.g. the
            # docs backlog the glm pool drains: #135, #150, #161…). Closing updates
            # an issue, so sorting by updated-desc surfaces the actual recent closes
            # — without it the windowed rate undercounts the fleet by 2-4x.
            ["gh", "issue", "list", "--state", "closed", "--limit", str(limit),
             "--search", "sort:updated-desc",
             "--json", "number,closedAt,stateReason,title"],
            cwd=str(root), capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [{"_error": str(exc)}]
    text = (proc.stdout or "").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except ValueError:
        return []
    return data if isinstance(data, list) else []


def read_progress_records(runs_dir: Path) -> list[dict[str, Any]]:
    """The close-arm's per-tick records (utc + closed_now) from progress.jsonl."""
    log = runs_dir / PROGRESS_LOG
    if not log.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        for line in log.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if isinstance(rec, dict):
                out.append(rec)
    except OSError:
        return []
    return out


# ---------------------------------------------------------------------------
# Pure graders (clock injected via now_ts)
# ---------------------------------------------------------------------------

def _parse_iso(ts: str | None) -> dt.datetime | None:
    if not ts:
        return None
    try:
        return dt.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def windowed_closes(closed_issues: list[dict[str, Any]], *, now_ts: float,
                    windows: tuple[int, ...] = DEFAULT_WINDOWS) -> dict[str, Any]:
    """Closes (total and COMPLETED) bucketed into trailing windows → per-hour rates.

    ``now_ts`` is epoch seconds (UTC). A close with no parseable ``closedAt`` is
    skipped (it can't be placed in a window) but counted in ``undated`` so the
    coverage is honest rather than silently dropping closes.
    """
    now = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(seconds=now_ts)
    undated = 0
    ages_total: list[float] = []
    ages_completed: list[float] = []
    for issue in closed_issues:
        t = _parse_iso(issue.get("closedAt"))
        if t is None:
            undated += 1
            continue
        if t.tzinfo is None:
            t = t.replace(tzinfo=dt.timezone.utc)
        age_h = (now - t).total_seconds() / 3600.0
        if age_h < 0:
            age_h = 0.0
        ages_total.append(age_h)
        if str(issue.get("stateReason") or "").upper() == "COMPLETED":
            ages_completed.append(age_h)

    per_window: dict[str, Any] = {}
    for w in windows:
        n_total = sum(1 for a in ages_total if a <= w)
        n_completed = sum(1 for a in ages_completed if a <= w)
        per_window[f"{w}h"] = {
            "window_hours": w,
            "closed": n_total,
            "completed": n_completed,
            "rate_per_hour": round(n_total / w, 2),
            "completed_rate_per_hour": round(n_completed / w, 2),
        }
    return {"per_window": per_window, "undated": undated,
            "closed_total_scanned": len(ages_total)}


def windowed_loop_closes(progress_records: list[dict[str, Any]], *, now_ts: float,
                         windows: tuple[int, ...] = DEFAULT_WINDOWS) -> dict[str, Any]:
    """The loop's own attributed closes (sum of ``closed_now``) per window → rate,
    plus minutes since the loop last drove ANY close (its liveness pulse)."""
    now = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(seconds=now_ts)
    events: list[tuple[float, int]] = []  # (age_hours, closed_now)
    last_close_age_min: float | None = None
    for rec in progress_records:
        t = _parse_iso(rec.get("utc"))
        if t is None:
            continue
        if t.tzinfo is None:
            t = t.replace(tzinfo=dt.timezone.utc)
        age_h = max(0.0, (now - t).total_seconds() / 3600.0)
        try:
            closed_now = int(rec.get("closed_now") or 0)
        except (TypeError, ValueError):
            closed_now = 0
        if closed_now > 0:
            events.append((age_h, closed_now))
            age_min = age_h * 60.0
            if last_close_age_min is None or age_min < last_close_age_min:
                last_close_age_min = age_min

    per_window: dict[str, Any] = {}
    for w in windows:
        n = sum(c for age, c in events if age <= w)
        per_window[f"{w}h"] = {"window_hours": w, "loop_closed": n,
                               "loop_rate_per_hour": round(n / w, 2)}
    return {"per_window": per_window,
            "last_loop_close_age_min": (round(last_close_age_min, 1)
                                        if last_close_age_min is not None else None)}


def grade(*, gh_windows: dict[str, Any], loop_windows: dict[str, Any],
          target_per_hour: float, primary_window: int) -> dict[str, Any]:
    """ON_TRACK / WARMING_UP / BELOW_TARGET on the primary window's COMPLETED rate.

    We grade on the *completed* (productive) rate, not the raw close rate: closing
    a ticket as not-planned drains the backlog but is not the "resolved 10/hour"
    the goal means. The loop-attributed rate is reported alongside so a high
    gh-rate carried by humans (loop wedged) is visible, not hidden by a green card.
    """
    key = f"{primary_window}h"
    gw = (gh_windows.get("per_window") or {}).get(key, {})
    completed_rate = float(gw.get("completed_rate_per_hour") or 0.0)
    raw_rate = float(gw.get("rate_per_hour") or 0.0)
    last_close_min = (loop_windows or {}).get("last_loop_close_age_min")
    recently_active = last_close_min is not None and last_close_min <= _WARMUP_RECENT_MIN

    short_key = "1h"
    short = (gh_windows.get("per_window") or {}).get(short_key, {})
    short_completed = float(short.get("completed_rate_per_hour") or 0.0)

    gap = round(target_per_hour - completed_rate, 2)
    if completed_rate >= target_per_hour:
        verdict, ok = "ON_TRACK", True
        reason = (f"{completed_rate}/h completed over the trailing {primary_window}h "
                  f"meets the {target_per_hour}/h target")
    elif recently_active and short_completed >= target_per_hour:
        # The loop is closing again (a fresh attributed close) and the most recent
        # hour already clears target — the longer window still carries a stall.
        verdict, ok = "WARMING_UP", True
        reason = (f"recovering: 1h completed rate {short_completed}/h already meets "
                  f"{target_per_hour}/h (loop closed {last_close_min}m ago); the "
                  f"{primary_window}h window {completed_rate}/h still carries an earlier stall")
    else:
        verdict, ok = "BELOW_TARGET", False
        reason = (f"{completed_rate}/h completed over the trailing {primary_window}h "
                  f"is {gap}/h short of the {target_per_hour}/h target"
                  + ("" if last_close_min is not None
                     else "; loop has NO attributed close on record — is it spawning?"))
    return {"verdict": verdict, "ok": ok, "reason": reason,
            "primary_window_hours": primary_window,
            "target_per_hour": target_per_hour,
            "completed_rate_per_hour": completed_rate,
            "raw_rate_per_hour": raw_rate,
            "gap_per_hour": gap,
            "short_window_completed_rate_per_hour": short_completed}


def build_payload(*, root: Path, closed_issues: list[dict[str, Any]],
                  progress_records: list[dict[str, Any]], now_ts: float,
                  target_per_hour: float, primary_window: int,
                  windows: tuple[int, ...] = DEFAULT_WINDOWS) -> dict[str, Any]:
    fetch_error = None
    if closed_issues and isinstance(closed_issues[0], dict) and closed_issues[0].get("_error"):
        fetch_error = closed_issues[0]["_error"]
        closed_issues = []

    gh_windows = windowed_closes(closed_issues, now_ts=now_ts, windows=windows)
    loop_windows = windowed_loop_closes(progress_records, now_ts=now_ts, windows=windows)
    g = grade(gh_windows=gh_windows, loop_windows=loop_windows,
              target_per_hour=target_per_hour, primary_window=primary_window)

    if fetch_error:
        g = {**g, "verdict": "AUDIT_ERROR", "ok": False,
             "reason": f"could not fetch closed issues from gh: {fetch_error}"}

    return {
        "schema": SCHEMA,
        "ok": g["ok"],
        "verdict": g["verdict"],
        "reason": g["reason"],
        "workspace": str(root),
        "target_per_hour": target_per_hour,
        "primary_window_hours": primary_window,
        "completed_rate_per_hour": g["completed_rate_per_hour"],
        "raw_rate_per_hour": g["raw_rate_per_hour"],
        "gap_per_hour": g["gap_per_hour"],
        "gh": gh_windows,
        "loop": loop_windows,
        "fetch_error": fetch_error,
    }


def collect(root: Path, *, target_per_hour: float, primary_window: int,
            closed_limit: int = 400, fetcher: ClosedFetcher | None = None,
            now_ts: float | None = None) -> dict[str, Any]:
    import time
    now_ts = time.time() if now_ts is None else now_ts
    fetch = fetcher or fetch_closed_issues
    closed = fetch(root, closed_limit)
    progress = read_progress_records(root / RUNS_DIRNAME)
    return build_payload(root=root, closed_issues=closed, progress_records=progress,
                         now_ts=now_ts, target_per_hour=target_per_hour,
                         primary_window=primary_window)


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def render(p: dict[str, Any]) -> str:
    gw = (p.get("gh") or {}).get("per_window") or {}
    lw = (p.get("loop") or {}).get("per_window") or {}
    lines = [
        f"dispatch throughput: {p.get('verdict')} ({'ok' if p.get('ok') else 'ACTION'})",
        f"  {p.get('reason')}",
        f"  target={p.get('target_per_hour')}/h  primary={p.get('primary_window_hours')}h"
        f"  completed={p.get('completed_rate_per_hour')}/h (raw {p.get('raw_rate_per_hour')}/h)",
        "  window | closed | completed | /h    | loop-closed | loop /h",
    ]
    for key in ("1h", "3h", "6h", "12h", "24h"):
        g = gw.get(key)
        if not g:
            continue
        l = lw.get(key) or {}
        lines.append(
            f"  {key:<6} | {g.get('closed'):>6} | {g.get('completed'):>9} | "
            f"{g.get('completed_rate_per_hour'):>5} | {l.get('loop_closed', '-'):>11} | "
            f"{l.get('loop_rate_per_hour', '-')}")
    last = (p.get("loop") or {}).get("last_loop_close_age_min")
    lines.append(f"  loop last attributed close: "
                 + (f"{last}m ago" if last is not None else "none on record"))
    return "\n".join(lines)


def render_md_block(p: dict[str, Any]) -> str:
    """A markdown section foldable into the committed dispatch-status doc."""
    gw = (p.get("gh") or {}).get("per_window") or {}
    lw = (p.get("loop") or {}).get("per_window") or {}
    out = [
        "## Throughput (closed issues per hour)",
        "",
        f"`verdict` = **{p.get('verdict')}** — {p.get('reason')}",
        "",
        f"Target **{p.get('target_per_hour')}/h** on the trailing "
        f"**{p.get('primary_window_hours')}h** window (graded on the *completed* rate — "
        "not-planned closes drain backlog but are not productive resolution).",
        "",
        "| window | closed | completed | completed /h | loop-closed | loop /h |",
        "|---|---|---|---|---|---|",
    ]
    for key in ("1h", "3h", "6h", "12h", "24h"):
        g = gw.get(key)
        if not g:
            continue
        l = lw.get(key) or {}
        out.append(
            f"| {key} | {g.get('closed')} | {g.get('completed')} | "
            f"{g.get('completed_rate_per_hour')} | {l.get('loop_closed', '-')} | "
            f"{l.get('loop_rate_per_hour', '-')} |")
    last = (p.get("loop") or {}).get("last_loop_close_age_min")
    out += ["",
            f"Loop's last attributed close: "
            + (f"{last} min ago." if last is not None else "**none on record**.")
            + " A gh-rate far above the loop-rate means humans/peers are draining the "
            "backlog, not the dispatcher."]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Closed-issues-per-hour throughput meter for the dispatch loop (read-only).")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--target-per-hour", type=float, default=DEFAULT_TARGET_PER_HOUR,
                    help=f"close-rate target (default: {DEFAULT_TARGET_PER_HOUR}/h)")
    ap.add_argument("--primary-window", type=int, default=DEFAULT_PRIMARY_WINDOW,
                    help=f"window (hours) graded against the target (default: {DEFAULT_PRIMARY_WINDOW})")
    ap.add_argument("--closed-limit", type=int, default=400,
                    help="max closed issues to fetch from gh (default: 400)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args(argv)

    root = Path(args.workspace).resolve() if args.workspace else repo_root()
    payload = collect(root, target_per_hour=args.target_per_hour,
                      primary_window=args.primary_window, closed_limit=args.closed_limit)
    print(json.dumps(payload, indent=2) if args.json else render(payload))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
