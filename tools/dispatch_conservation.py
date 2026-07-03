#!/usr/bin/env python3
"""Conservation ledger for dispatch worker-units: spent = accounted + leaked.

The fleet's promise is *work conservation*: if N open issues meet N worker-units
inside a window, every unit must end in exactly one accountable outcome — a
witnessed ship, a graded refusal (no-commit with a reason), or a spawn failure —
and anything else is a LEAK the operator cannot see today. `dispatch_throughput`
measures a close RATE; `dispatch_status` diagnoses the current tick; nothing
answers the identity question "we spent K units this window — where did each
go?". This tool folds the durable `.dispatch-runs` artifacts into that identity:

    units_spent(window) = shipped_witnessed
                        + committed_unwitnessed
                        + no_commit{self_modify|policy_block|auth_wall|
                                    off_trunk|banner_noop|unknown}
                        + spawn_failed
                        + leaked_unswept          <- the number this tool exists for

plus the issue side (closes attributed in the window, contract holds, and
re-storm churn: issues burning 2+ units in one window). Read-only: it parses
artifacts directly and never imports the hot dispatcher modules
(issue_resolve_dispatch / dispatch_preflight), so it stays runnable while a
peer edits them and cannot perturb what it measures.

Sources (the on-disk contract, all under --runs-dir):
  - worker logs   {resolve|repair}-<issue>-<YYYYMMDD-HHMMSS>.log  (stamp = UTC
    spawn time; first line `# fak-spawn ... lane=<L> backend=<B> ...`)
  - sidecars      .pid (live only; pruned once dead), .backend, .witness
    ({"claim": CLAIM_WITNESSED|CLAIM_UNWITNESSED|CLAIM_NO_COMMIT, ...,
      "reason": <bucket>} — written by the live witness sweep)
  - progress.jsonl  fleet-issue-resolve-progress/1 rows (closed_now per tick)
  - contract-holds.jsonl  {utc, issue, score, reason}

Honesty rules: a `.pid` we cannot disprove is LIVE (never leaked); a graded
witness is final; a dead worker with a real log and no witness is
`leaked_unswept` — either the witness sweep has not reached it yet or its
artifacts were pruned first. Coverage is reported, never assumed: the tool
prints how many in-window units carry no witness so silence reads as a number,
not as success.

Usage:
    python tools/dispatch_conservation.py                 # human summary, 6h window
    python tools/dispatch_conservation.py --window-h 24 --json
    python tools/dispatch_conservation.py --fail-on-leak 0   # CI gate: exit 1 on any leak
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

SCHEMA = "fleet-dispatch-conservation/1"
DEFAULT_RUNS_DIRNAME = ".dispatch-runs"
DEFAULT_WINDOW_H = 6.0

# Log-name grammar shared with the dispatcher (kept as a literal copy, not an
# import: the dispatcher module is a live-edited surface).
_LOG_RE = re.compile(r"^(resolve|repair)-(\d+)-(\d{8})-(\d{6})\.log$")
_SPAWN_HEADER_RE = re.compile(r"^# fak-spawn \S+ issue=\d+ lane=(\S+) backend=(\S+)")

# Closed outcome vocabulary. Every in-window unit folds to exactly one token.
OUTCOME_LIVE = "live"
OUTCOME_WITNESSED = "shipped_witnessed"
OUTCOME_UNWITNESSED = "committed_unwitnessed"
OUTCOME_NO_COMMIT = "no_commit"
OUTCOME_SPAWN_FAILED = "spawn_failed"
OUTCOME_LEAKED = "leaked_unswept"

# The witness sweep's no-commit reason buckets (classify_no_commit_reason).
NO_COMMIT_REASONS = ("self_modify", "policy_block", "auth_wall",
                     "off_trunk", "banner_noop", "unknown")


def parse_log_stamp_utc(name: str) -> float | None:
    """Spawn time (epoch seconds) from a worker log name, or None when the name
    is not a worker log. The stamp is authoritative for windowing: log mtime
    moves every time the worker writes, but the unit was SPENT at spawn."""
    m = _LOG_RE.match(name)
    if not m:
        return None
    try:
        dt = datetime.strptime(m.group(3) + m.group(4), "%Y%m%d%H%M%S")
    except ValueError:
        return None
    return dt.replace(tzinfo=timezone.utc).timestamp()


def read_spawn_header(log: Path) -> dict[str, str]:
    """lane/backend from the `# fak-spawn` first line ("" when absent), plus
    whether the log ever grew past the header — the spawn-failure signal: a
    header-only (or empty) log is a worker that died at/before exec, which the
    5-second spawn probe only catches when it exits inside the probe window."""
    out = {"lane": "", "backend": "", "body": False}
    try:
        with log.open("r", encoding="utf-8", errors="replace") as fh:
            first = fh.readline()
            m = _SPAWN_HEADER_RE.match(first.strip())
            if m:
                out["lane"], out["backend"] = m.group(1), m.group(2)
            elif first.strip():
                out["body"] = True  # no header but real content: still a run
            for line in fh:
                if line.strip():
                    out["body"] = True
                    break
    except OSError:
        pass
    return out


def read_witness(log: Path) -> dict[str, Any] | None:
    """The worker's `.witness` verdict sidecar, or None when the sweep has not
    graded it (or the record is unreadable — treated the same, honestly)."""
    try:
        raw = log.with_suffix(".witness").read_text(encoding="utf-8")
        rec = json.loads(raw)
    except (OSError, ValueError):
        return None
    return rec if isinstance(rec, dict) else None


def default_alive_pids() -> set[int] | None:
    """PIDs alive right now, or None when the host cannot be scanned (no
    psutil): None means "cannot disprove liveness", which classifies every
    .pid-bearing unit as live — the conservative direction (never leaked)."""
    try:
        import psutil  # type: ignore
    except ImportError:
        return None
    try:
        return {p.pid for p in psutil.process_iter()}
    except Exception:
        return None


def classify_unit(log: Path, *, alive: set[int] | None) -> dict[str, Any]:
    """Fold one worker log + sidecars into the closed outcome vocabulary.

    Precedence: a graded witness is final (the sweep only grades dead pids);
    else an alive .pid is live; else header-only means spawn-failed; else the
    unit leaked — dead, real work attempted, never graded."""
    name = log.name
    m = _LOG_RE.match(name)
    kind, issue = m.group(1), int(m.group(2))
    header = read_spawn_header(log)
    unit: dict[str, Any] = {
        "log": name, "kind": kind, "issue": issue,
        "lane": header["lane"], "backend": header["backend"],
        "spawned_utc": datetime.fromtimestamp(
            parse_log_stamp_utc(name), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    witness = read_witness(log)
    if witness is not None:
        claim = str(witness.get("claim", ""))
        if claim == "CLAIM_WITNESSED":
            unit["outcome"] = OUTCOME_WITNESSED
            unit["sha"] = witness.get("sha")
        elif claim == "CLAIM_UNWITNESSED":
            unit["outcome"] = OUTCOME_UNWITNESSED
            unit["sha"] = witness.get("sha")
        else:
            unit["outcome"] = OUTCOME_NO_COMMIT
            reason = str(witness.get("reason") or "unknown")
            unit["reason"] = reason if reason in NO_COMMIT_REASONS else "unknown"
        return unit

    pid_file = log.with_suffix(".pid")
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            pid = None
        # alive=None means the host could not be scanned: cannot disprove
        # liveness -> live (never invent a leak from a blind probe).
        if pid is not None and (alive is None or pid in alive):
            unit["outcome"] = OUTCOME_LIVE
            unit["pid"] = pid
            return unit

    if not header["body"]:
        unit["outcome"] = OUTCOME_SPAWN_FAILED
        return unit
    unit["outcome"] = OUTCOME_LEAKED
    return unit


def collect_units(runs_dir: Path, *, since_ts: float,
                  alive: set[int] | None) -> list[dict[str, Any]]:
    """Every worker-unit spent inside the window (spawn stamp >= since_ts),
    classified. Sorted oldest-first so reports read chronologically."""
    units: list[dict[str, Any]] = []
    if not runs_dir.is_dir():
        return units
    for log in runs_dir.glob("*.log"):
        stamp = parse_log_stamp_utc(log.name)
        if stamp is None or stamp < since_ts:
            continue
        unit = classify_unit(log, alive=alive)
        unit["_stamp"] = stamp
        units.append(unit)
    units.sort(key=lambda u: u["_stamp"])
    for u in units:
        del u["_stamp"]
    return units


def _parse_iso_utc(ts: str) -> float | None:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def windowed_closes(progress_path: Path, *, since_ts: float) -> dict[str, Any]:
    """Issue closes the loop attributed inside the window (sum of closed_now
    over in-window progress rows), plus the newest open/baseline picture."""
    out = {"closed_in_window": 0, "rows_in_window": 0,
           "open_now": None, "baseline_open": None}
    try:
        lines = progress_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return out
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        ts = _parse_iso_utc(str(rec.get("utc", "")))
        if ts is None or ts < since_ts:
            continue
        out["rows_in_window"] += 1
        closed = rec.get("closed_now")
        if isinstance(closed, int) and closed > 0:
            out["closed_in_window"] += closed
        if isinstance(rec.get("open_now"), int):
            out["open_now"] = rec["open_now"]
        if isinstance(rec.get("baseline_open"), int):
            out["baseline_open"] = rec["baseline_open"]
    return out


def windowed_contract_holds(holds_path: Path, *, since_ts: float) -> dict[str, Any]:
    """Contract holds recorded inside the window — issues the gate kept OUT of
    dispatch (they spent no unit, but they are demand the window did not serve)."""
    issues: set[int] = set()
    rows = 0
    try:
        lines = holds_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        ts = rec.get("ts")
        if not isinstance(ts, (int, float)):
            ts = _parse_iso_utc(str(rec.get("utc", "")))
        if ts is None or ts < since_ts:
            continue
        rows += 1
        if isinstance(rec.get("issue"), int):
            issues.add(rec["issue"])
    return {"rows": rows, "distinct_issues": len(issues)}


def fold_conservation(units: list[dict[str, Any]], closes: dict[str, Any],
                      holds: dict[str, Any], *, window_h: float,
                      now_iso: str) -> dict[str, Any]:
    """The conservation identity over one window. Pure: data in, report out."""
    resolve = [u for u in units if u["kind"] == "resolve"]
    repair = [u for u in units if u["kind"] == "repair"]
    by_outcome: dict[str, int] = {}
    no_commit: dict[str, int] = {}
    for u in resolve:
        by_outcome[u["outcome"]] = by_outcome.get(u["outcome"], 0) + 1
        if u["outcome"] == OUTCOME_NO_COMMIT:
            no_commit[u["reason"]] = no_commit.get(u["reason"], 0) + 1

    live = by_outcome.get(OUTCOME_LIVE, 0)
    spent = len(resolve) - live
    witnessed = by_outcome.get(OUTCOME_WITNESSED, 0)
    unwitnessed = by_outcome.get(OUTCOME_UNWITNESSED, 0)
    refused = by_outcome.get(OUTCOME_NO_COMMIT, 0)
    spawn_failed = by_outcome.get(OUTCOME_SPAWN_FAILED, 0)
    leaked = by_outcome.get(OUTCOME_LEAKED, 0)

    # Re-storm churn: issues that burned 2+ finished units in ONE window are
    # the cooldown-loop signature (a failed issue re-entering with no
    # escalation) — each extra unit is capacity another issue never got.
    attempts: dict[int, int] = {}
    for u in resolve:
        if u["outcome"] != OUTCOME_LIVE:
            attempts[u["issue"]] = attempts.get(u["issue"], 0) + 1
    churned = sorted((n, c) for n, c in attempts.items() if c >= 2)

    verdict = "CONSERVED" if leaked == 0 else "LEAKING"
    return {
        "schema": SCHEMA,
        "utc": now_iso,
        "window_h": window_h,
        "verdict": verdict,
        "units": {
            "resolve_total": len(resolve),
            "live": live,
            "spent": spent,
            "shipped_witnessed": witnessed,
            "committed_unwitnessed": unwitnessed,
            "no_commit": refused,
            "no_commit_reasons": dict(sorted(no_commit.items())),
            "spawn_failed": spawn_failed,
            "leaked_unswept": leaked,
            "repair_total": len(repair),
        },
        "identity_holds": spent == witnessed + unwitnessed + refused + spawn_failed + leaked,
        "yield": {
            "witnessed_per_spent": round(witnessed / spent, 4) if spent else None,
            "issues_closed_in_window": closes["closed_in_window"],
            "open_now": closes["open_now"],
            "baseline_open": closes["baseline_open"],
        },
        "churn": {
            "issues_with_2plus_units": len(churned),
            "worst": [{"issue": n, "units": c} for n, c in
                      sorted(churned, key=lambda x: -x[1])[:5]],
        },
        "contract_holds": holds,
        "leaked_units": [
            {k: u.get(k) for k in ("log", "issue", "lane", "backend", "spawned_utc")}
            for u in resolve if u["outcome"] == OUTCOME_LEAKED
        ],
    }


def render(report: dict[str, Any]) -> str:
    u = report["units"]
    y = report["yield"]
    lines = [
        # ASCII-only: the Windows console renders this under cp1252.
        f"dispatch conservation -- {report['verdict']}  window={report['window_h']}h  {report['utc']}",
        (f"  units: {u['resolve_total']} resolve ({u['live']} live) + "
         f"{u['repair_total']} repair; spent={u['spent']}"),
        (f"  spent = {u['shipped_witnessed']} shipped + {u['committed_unwitnessed']} unwitnessed-commit + "
         f"{u['no_commit']} no-commit + {u['spawn_failed']} spawn-failed + "
         f"{u['leaked_unswept']} LEAKED  (identity {'holds' if report['identity_holds'] else 'BROKEN'})"),
    ]
    if u["no_commit_reasons"]:
        lines.append("  no-commit reasons: " + ", ".join(
            f"{k}={v}" for k, v in u["no_commit_reasons"].items()))
    lines.append(
        f"  yield: witnessed/spent={y['witnessed_per_spent']}  closes-in-window={y['issues_closed_in_window']}"
        + (f"  open_now={y['open_now']} (baseline {y['baseline_open']})"
           if y["open_now"] is not None else ""))
    ch = report["churn"]
    if ch["issues_with_2plus_units"]:
        worst = ", ".join(f"#{w['issue']}x{w['units']}" for w in ch["worst"])
        lines.append(f"  churn: {ch['issues_with_2plus_units']} issue(s) burned 2+ units ({worst})")
    holds = report["contract_holds"]
    if holds["rows"]:
        lines.append(f"  contract holds: {holds['rows']} row(s), {holds['distinct_issues']} distinct issue(s) kept out of dispatch")
    for leak in report["leaked_units"]:
        lines.append(f"  LEAK {leak['log']}  issue=#{leak['issue']} lane={leak['lane']} spawned={leak['spawned_utc']}")
    if report["verdict"] == "CONSERVED":
        lines.append("  every finished unit is accounted; leaks would be listed above")
    return "\n".join(lines)


def main(argv: list[str] | None = None, *,
         alive_provider: Callable[[], set[int] | None] = default_alive_pids,
         now_ts: float | None = None) -> int:
    ap = argparse.ArgumentParser(description="conservation ledger for dispatch worker-units")
    ap.add_argument("--runs-dir", default=None,
                    help=f"dispatch runs dir (default: <repo>/{DEFAULT_RUNS_DIRNAME})")
    ap.add_argument("--window-h", type=float, default=DEFAULT_WINDOW_H,
                    help=f"window in hours (default {DEFAULT_WINDOW_H})")
    ap.add_argument("--json", action="store_true", help="emit the machine-readable report")
    ap.add_argument("--fail-on-leak", type=int, default=None, metavar="N",
                    help="exit 1 when leaked_unswept exceeds N (default: report-only)")
    args = ap.parse_args(argv)

    runs_dir = Path(args.runs_dir) if args.runs_dir else Path.cwd() / DEFAULT_RUNS_DIRNAME
    now = now_ts if now_ts is not None else time.time()
    since = now - args.window_h * 3600
    now_iso = datetime.fromtimestamp(now, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    units = collect_units(runs_dir, since_ts=since, alive=alive_provider())
    closes = windowed_closes(runs_dir / "progress.jsonl", since_ts=since)
    holds = windowed_contract_holds(runs_dir / "contract-holds.jsonl", since_ts=since)
    report = fold_conservation(units, closes, holds,
                               window_h=args.window_h, now_iso=now_iso)

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render(report))
    if args.fail_on_leak is not None and report["units"]["leaked_unswept"] > args.fail_on_leak:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
