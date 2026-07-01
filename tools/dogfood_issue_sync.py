#!/usr/bin/env python3
"""Sync recurring dogfood scorecard ACTION/debt into stable GitHub issues (#800).

The recent-feature dogfood packet (``tools/recent_feature_dogfood.py``) lets a
scorecard report ACTION/debt and STILL pass, as long as the machine payload is
well formed -- the right *local* gate, but recurring debt then leaves no durable
backlog trail. This helper reads a dogfood ``report.json``, finds the scorecard
probes whose payload is in an ACTION/debt state, and renders ONE stable GitHub
issue per scorecard, upserted by a hidden HTML-comment marker so a re-run EDITS
the same issue instead of opening a duplicate every run.

Offline + dry-run by DEFAULT: it prints the issues it WOULD create/update and
touches no network. ``--sync`` is the explicit opt-in that actually calls ``gh``.

    # plan only (no network) -- newest report.json if --report is omitted
    python tools/dogfood_issue_sync.py
    python tools/dogfood_issue_sync.py --report .fak/recent-feature-dogfood/<stamp>/report.json
    python tools/dogfood_issue_sync.py --report <path> --json

    # actually create/update the stable issues (explicit network opt-in)
    python tools/dogfood_issue_sync.py --report <path> --sync
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import glob
import json
import os
import subprocess
from dispatch_worker import install_no_window_subprocess_defaults
import sys
from pathlib import Path
from typing import Any
install_no_window_subprocess_defaults(subprocess)

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except (AttributeError, ValueError):
    pass

SCHEMA = "dogfood-issue-sync/1"
# Hidden marker stamped in each synced issue body; the upsert finds the existing
# issue by this exact string so a re-run edits rather than duplicates.
MARKER = "dogfood-issue-sync"
TRIAGE_LABELS = ["needs-triage", "triage-only"]
DEFAULT_MAX_REPORT_AGE_SECONDS = 24 * 60 * 60


def _parse_duration_seconds(value: str) -> float:
    s = value.strip().lower()
    if not s:
        raise argparse.ArgumentTypeError("duration must not be empty")
    unit = s[-1]
    scale = {"s": 1, "m": 60, "h": 60 * 60, "d": 24 * 60 * 60}.get(unit)
    if scale is not None:
        number = s[:-1]
    else:
        scale = 1
        number = s
    try:
        seconds = float(number) * scale
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid duration {value!r}") from exc
    if seconds < 0:
        raise argparse.ArgumentTypeError("duration must be non-negative")
    return seconds


def _iso_utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _format_duration(seconds: int | float) -> str:
    seconds = int(max(0, seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {sec}s" if sec else f"{minutes}m"
    hours, minute = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h {minute}m" if minute else f"{hours}h"
    days, hour = divmod(hours, 24)
    return f"{days}d {hour}h" if hour else f"{days}d"


def _report_freshness(path: str, *, max_age_seconds: float = DEFAULT_MAX_REPORT_AGE_SECONDS,
                      allow_stale: bool = False, now: datetime | None = None) -> dict[str, Any]:
    stamp = os.path.getmtime(path)
    now_dt = now or datetime.now(timezone.utc)
    age_seconds = int(max(0, now_dt.timestamp() - stamp))
    max_age = int(max_age_seconds)
    return {
        "timestamp": _iso_utc(stamp),
        "source": "mtime",
        "age_seconds": age_seconds,
        "age": _format_duration(age_seconds),
        "max_age_seconds": max_age,
        "max_age": _format_duration(max_age),
        "stale": age_seconds > max_age,
        "stale_allowed": bool(allow_stale),
    }


def _stale_report_message(freshness: dict[str, Any]) -> str:
    return (
        "selected dogfood report is stale "
        f"(age {freshness['age']} > max {freshness['max_age']}); "
        "rerun dogfood or pass --allow-stale-report to sync anyway"
    )


def _debt_count(payload: dict[str, Any]) -> int | None:
    """A scorecard exposes its debt under a key named ``debt`` or ``*_debt``
    (slop_debt, dogfood_debt, ...). Return the first such integer, else None."""
    for k, v in payload.items():
        if k == "debt" or k.endswith("_debt"):
            try:
                return int(v)
            except (TypeError, ValueError):
                return None
    return None


def _is_actionable(payload: dict[str, Any]) -> bool:
    """An ACTION verdict OR a positive debt count makes a scorecard worth a
    backlog issue. A grade-A / zero-debt / non-ACTION scorecard is healthy and
    gets no issue (the sync never files against a clean scorecard)."""
    if str(payload.get("verdict", "")).upper() == "ACTION":
        return True
    d = _debt_count(payload)
    return d is not None and d > 0


def _is_scorecard(payload: Any) -> bool:
    """A scorecard payload carries a ``grade`` or a ``*_debt`` key. Non-scorecard
    probe payloads (vcache index, etc.) are ignored."""
    return isinstance(payload, dict) and ("grade" in payload or _debt_count(payload) is not None)


def render_body(key: str, probe: dict[str, Any], payload: dict[str, Any], out_dir: str) -> str:
    """The stable issue body: the marker (for idempotent upsert) plus the current
    score / grade / debt / next-action / evidence / reproduce command."""
    grade = payload.get("grade", "?")
    score = payload.get("score", payload.get("coverage", "n/a"))
    debt = _debt_count(payload)
    verdict = payload.get("verdict", "")
    finding = payload.get("finding") or payload.get("recommendation") or "(see scorecard output)"
    evidence = out_dir or "(report out_dir)"
    cmd = " ".join(probe.get("command") or [])
    return "\n".join([
        f"<!-- {MARKER}:{key} -->",
        "_Auto-filed by `tools/dogfood_issue_sync.py` from a recent-feature dogfood run. "
        "Re-running the sync EDITS this issue; it is not a duplicate._",
        "",
        f"The **{key}** scorecard is in an ACTION/debt state in the latest dogfood pass.",
        "",
        f"- **grade:** {grade}",
        f"- **score:** {score}",
        f"- **debt:** {debt if debt is not None else 'n/a'}",
        f"- **verdict:** {verdict or 'n/a'}",
        f"- **suggested next action (worst-first):** {finding}",
        f"- **evidence:** `{evidence}`",
        f"- **reproduce:** `{cmd}`",
        "- dispatchability: `triage_only`",
        "",
        "Close when the scorecard returns to grade A / zero debt — the sync will not "
        "reopen a healthy scorecard.",
    ])


def plan_issues(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Pure fold: ``report.json`` -> the stable issues to upsert, one per
    actionable scorecard. No network — this is the unit the test exercises."""
    out: list[dict[str, Any]] = []
    out_dir = report.get("out_dir") or ""
    for probe in report.get("probes", []) or []:
        payload = probe.get("payload")
        if not _is_scorecard(payload) or not _is_actionable(payload):
            continue
        key = probe.get("key") or payload.get("schema") or "scorecard"
        out.append({
            "key": key,
            "title": f"dogfood: {key} reports scorecard ACTION/debt",
            "body": render_body(key, probe, payload, out_dir),
            "marker": f"<!-- {MARKER}:{key} -->",
            "labels": list(TRIAGE_LABELS),
        })
    return out


# --- gh I/O (only reached under --sync) ---------------------------------------

def _find_existing(marker: str) -> int | None:
    """The open issue whose body carries ``marker``, or None. Searches by the
    marker so the upsert is idempotent across runs."""
    try:
        r = subprocess.run(
            ["gh", "issue", "list", "--state", "open", "--search", marker,
             "--json", "number,body", "--limit", "50"],
            capture_output=True, text=True, timeout=60)
        for it in json.loads(r.stdout or "[]"):
            if marker in (it.get("body") or ""):
                return int(it["number"])
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired):
        return None
    return None


def _sync_issue(issue: dict[str, Any], label: str) -> dict[str, Any]:
    """Create or update the stable issue via ``gh``. Returns the action taken."""
    labels = _merge_labels(issue.get("labels") or [], label)
    num = _find_existing(issue["marker"])
    if num is not None:
        cmd = ["gh", "issue", "edit", str(num), "--body", issue["body"]]
        for lab in labels:
            cmd += ["--add-label", lab]
        subprocess.run(cmd, check=True, timeout=60)
        return {"key": issue["key"], "action": "updated", "number": num}
    cmd = ["gh", "issue", "create", "--title", issue["title"], "--body", issue["body"]]
    for lab in labels:
        cmd += ["--label", lab]
    r = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60)
    return {"key": issue["key"], "action": "created", "url": (r.stdout or "").strip()}


def _merge_labels(base: list[str], extra: str = "") -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for label in [*base, extra]:
        label = (label or "").strip()
        if not label or label in seen:
            continue
        seen.add(label)
        out.append(label)
    return out


def _newest_report() -> str | None:
    reps = sorted(glob.glob(".fak/recent-feature-dogfood/*/report.json"), key=os.path.getmtime)
    return reps[-1] if reps else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", default="", help="dogfood report.json (default: newest under .fak/recent-feature-dogfood/)")
    ap.add_argument("--sync", action="store_true", help="actually create/update issues via gh (explicit network opt-in)")
    ap.add_argument("--max-report-age", type=_parse_duration_seconds, default=DEFAULT_MAX_REPORT_AGE_SECONDS,
                    help="stale report threshold before --sync is refused (default: 24h; suffix s/m/h/d)")
    ap.add_argument("--allow-stale-report", action="store_true",
                    help="allow --sync even when the selected report is older than --max-report-age")
    ap.add_argument("--label", default="", help="label to put on newly-created issues (must already exist)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args(argv)

    path = args.report or _newest_report() or ""
    if not path or not Path(path).is_file():
        print("no dogfood report.json found — pass --report <path>", file=sys.stderr)
        return 2
    try:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"could not read {path}: {exc}", file=sys.stderr)
        return 2

    issues = plan_issues(report)
    freshness = _report_freshness(path, max_age_seconds=args.max_report_age,
                                  allow_stale=args.allow_stale_report)
    doc = {"schema": SCHEMA, "report": path, "mode": "sync" if args.sync else "dry-run",
           "report_freshness": freshness,
           "actionable": len(issues), "issues": [{k: i[k] for k in ("key", "title")} for i in issues],
           "results": []}

    if args.sync and freshness["stale"] and not args.allow_stale_report:
        doc["refused"] = True
        doc["error"] = "stale_report"
        message = _stale_report_message(freshness)
        if args.json:
            print(json.dumps(doc, indent=2))
        else:
            print(f"dogfood-issue-sync: REFUSED  report={path}")
            print(f"  report timestamp: {freshness['timestamp']} (source={freshness['source']})")
            print(f"  report age: {freshness['age']}  max={freshness['max_age']}  stale=yes")
            print(f"  {message}")
        print(f"dogfood-issue-sync: {message}", file=sys.stderr)
        return 2

    results = [_sync_issue(i, args.label) for i in issues] if args.sync else []
    doc["results"] = results

    if args.json:
        print(json.dumps(doc, indent=2))
    else:
        mode = "SYNC" if args.sync else "DRY-RUN (no network; pass --sync to create/update)"
        print(f"dogfood-issue-sync: {mode}  report={path}")
        print(f"  report timestamp: {freshness['timestamp']} (source={freshness['source']})")
        stale = "yes" if freshness["stale"] else "no"
        print(f"  report age: {freshness['age']}  max={freshness['max_age']}  stale={stale}")
        if freshness["stale"]:
            suffix = " --allow-stale-report override active" if args.allow_stale_report else ""
            print(f"  STALE report: {_stale_report_message(freshness)}{suffix}")
        print(f"  actionable scorecards: {len(issues)}")
        for i in issues:
            print(f"    - {i['title']}")
        for r in results:
            tail = r.get("url") or f"#{r.get('number')}"
            print(f"    {r['action']}: {r['key']} -> {tail}")
        if not issues:
            print("  (no scorecard is in an ACTION/debt state — nothing to file)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
