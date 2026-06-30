#!/usr/bin/env python3
"""stale_work_watchdog.py -- find (and optionally GC) THIS fak clone's own stale work.

fak's CLAUDE.md / AGENTS.md mandate a single shared trunk worktree, and every agent
session leaves per-session ephemera behind in *gitignored* state dirs:

  * .dos/markers/*.jsonl       -- wait-marker liveness rungs, one file per session
  * .dos/streams/*.jsonl       -- per-session event streams
  * .dos/stop-failures/*.json  -- stop-hook failure counters, one per session
  * tools/_watchdog/*.log|*.err|*.jsonl -- resume/retry/rehome watchdog logs

Nothing prunes these in THIS clone. The host's `DOS-cleanup-sweep` task sweeps a
*different* repo (dos-kernel-public), and `FakFleetJanitor` reaps idle GCP VMs --
neither touches `C:\\work\\fak\\.dos`. So the dirs grow without bound (174 MB and
climbing, oldest file only as old as the clone itself: nothing is reaping them).

This watchdog is the missing janitor, plus a read-only stale-work report:

  1. AGE-GC  -- gitignored per-session ephemera older than --max-age-days.
                DRY-RUN by default; --live deletes only the over-age files, and
                only ones provably inside the known ephemeral dirs.
  2. STUCK   -- stop-failure counters at/above --stuck-threshold consecutive
                failures: a session wedged in its Stop hook. REPORTED, never
                deleted (a human should look).
  3. WIP     -- count + oldest mtime of the shared trunk's uncommitted changes.
                REPORTED only -- never staged or committed. A path-scoped commit
                in this shared tree sweeps peers' in-flight files (see AGENTS.md
                shared-tree rules), so this layer NEVER mutates git state.

Exit codes: 0 = clean / swept OK | 2 = stale work found AND --fail-on-stale set
(so a CI step or alerting wrapper can gate on it). A plain scheduled `--live`
janitor run exits 0 after a successful sweep, keeping the task's LastResult clean.

Run:    python tools/stale_work_watchdog.py              # dry-run report
        python tools/stale_work_watchdog.py --live       # + GC over-age ephemera
        python tools/stale_work_watchdog.py --json
        python tools/stale_work_watchdog.py --fail-on-stale   # nonzero if stale
Test:   python tools/stale_work_watchdog_test.py
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from dispatch_worker import install_no_window_subprocess_defaults
install_no_window_subprocess_defaults(subprocess)
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
DEFAULT_REPO = HERE.parent  # tools/ -> repo root

# (label, repo-relative glob). Every entry MUST live under a gitignored state dir
# whose files are pure per-session ephemera -- safe to age-GC. The sweep refuses to
# delete anything whose resolved parent is not one of these dirs (belt-and-braces).
EPHEMERAL_GLOBS: tuple[tuple[str, str], ...] = (
    ("markers", ".dos/markers/*.jsonl"),
    ("streams", ".dos/streams/*.jsonl"),
    ("stop-failures", ".dos/stop-failures/*.json"),
    ("watchdog-logs", "tools/_watchdog/*.log"),
    ("watchdog-errs", "tools/_watchdog/*.err"),
    ("watchdog-jsonl", "tools/_watchdog/*.jsonl"),
)

# Resolved ephemeral parent dirs the sweep is allowed to unlink inside.
EPHEMERAL_DIRS: tuple[str, ...] = (".dos/markers", ".dos/streams",
                                   ".dos/stop-failures", "tools/_watchdog")


@dataclass
class StaleReport:
    repo: str
    now: float
    max_age_days: int
    stuck_threshold: int
    wip_stale_hours: int
    age_candidates: list[dict] = field(default_factory=list)
    stuck_sessions: list[dict] = field(default_factory=list)
    wip: dict = field(default_factory=dict)
    swept: dict = field(default_factory=lambda: {"files": 0, "bytes": 0})

    @property
    def age_files(self) -> int:
        return len(self.age_candidates)

    @property
    def age_bytes(self) -> int:
        return sum(c["size"] for c in self.age_candidates)

    @property
    def has_stale(self) -> bool:
        return bool(self.age_candidates or self.stuck_sessions
                    or self.wip.get("stale"))


def _stat_age_days(st: os.stat_result, now: float) -> float:
    return max(0.0, (now - st.st_mtime) / 86400.0)


def scan_ephemera(repo: Path, now: float, max_age_days: int) -> list[dict]:
    """Over-age gitignored ephemera, worst (oldest) first. Pure scan, no deletes."""
    out: list[dict] = []
    cutoff = max_age_days
    for label, glob in EPHEMERAL_GLOBS:
        for p in repo.glob(glob):
            try:
                st = p.stat()
            except OSError:
                continue
            if not p.is_file():
                continue
            age = _stat_age_days(st, now)
            if age >= cutoff:
                out.append({
                    "path": str(p),
                    "rel": p.relative_to(repo).as_posix(),
                    "label": label,
                    "age_days": round(age, 1),
                    "size": st.st_size,
                })
    out.sort(key=lambda c: c["age_days"], reverse=True)
    return out


def scan_stuck(repo: Path, threshold: int) -> list[dict]:
    """Stop-failure counters at/above `threshold` consecutive failures."""
    out: list[dict] = []
    for p in (repo / ".dos" / "stop-failures").glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        consec = int(data.get("consecutive", 0) or 0)
        if consec >= threshold:
            out.append({
                "session": p.stem,
                "rel": p.relative_to(repo).as_posix(),
                "consecutive": consec,
                "total": int(data.get("total", 0) or 0),
            })
    out.sort(key=lambda c: c["consecutive"], reverse=True)
    return out


def _git_porcelain(repo: Path) -> list[str]:
    try:
        cp = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return []
    if cp.returncode != 0:
        return []
    return [ln for ln in cp.stdout.splitlines() if ln.strip()]


def scan_wip(repo: Path, now: float, wip_stale_hours: int,
             porcelain: Callable[[Path], list[str]] = _git_porcelain) -> dict:
    """Report (never mutate) the shared trunk's uncommitted changes + their oldest age."""
    lines = porcelain(repo)
    paths: list[str] = []
    for ln in lines:
        # porcelain: XY <path>  (rename shows "orig -> new"; take the new path)
        rest = ln[3:] if len(ln) > 3 else ln
        rest = rest.split(" -> ")[-1].strip().strip('"')
        if rest:
            paths.append(rest)
    oldest_age_h = 0.0
    oldest_path = None
    for rel in paths:
        fp = repo / rel
        try:
            st = fp.stat()
        except OSError:
            continue
        age_h = max(0.0, (now - st.st_mtime) / 3600.0)
        if age_h > oldest_age_h:
            oldest_age_h, oldest_path = age_h, rel
    return {
        "count": len(paths),
        "oldest_age_hours": round(oldest_age_h, 1),
        "oldest_path": oldest_path,
        "stale": oldest_age_h >= wip_stale_hours and len(paths) > 0,
        "threshold_hours": wip_stale_hours,
    }


def _is_inside_ephemeral(repo: Path, p: Path) -> bool:
    try:
        rel = p.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError:
        return False
    return any(rel.startswith(d + "/") for d in EPHEMERAL_DIRS)


def sweep(repo: Path, candidates: list[dict], live: bool) -> dict:
    """Delete over-age ephemera iff `live`. Refuses anything outside the ephemeral dirs."""
    files = 0
    nbytes = 0
    for c in candidates:
        p = Path(c["path"])
        if not _is_inside_ephemeral(repo, p):
            # Defensive: a glob can only ever match inside the ephemeral dirs, but
            # never unlink something that somehow resolved elsewhere.
            continue
        if live:
            try:
                size = p.stat().st_size
                p.unlink()
            except OSError:
                continue
            files += 1
            nbytes += size
        else:
            files += 1
            nbytes += c["size"]
    return {"files": files, "bytes": nbytes}


def build_report(repo: Path, now: float, max_age_days: int, stuck_threshold: int,
                 wip_stale_hours: int, live: bool,
                 porcelain: Callable[[Path], list[str]] = _git_porcelain) -> StaleReport:
    rep = StaleReport(repo=str(repo), now=now, max_age_days=max_age_days,
                      stuck_threshold=stuck_threshold, wip_stale_hours=wip_stale_hours)
    rep.age_candidates = scan_ephemera(repo, now, max_age_days)
    rep.stuck_sessions = scan_stuck(repo, stuck_threshold)
    rep.wip = scan_wip(repo, now, wip_stale_hours, porcelain)
    rep.swept = sweep(repo, rep.age_candidates, live)
    return rep


def _fmt_mb(nbytes: int) -> str:
    return f"{nbytes / 1e6:.1f} MB"


def render_human(rep: StaleReport, live: bool) -> str:
    lines: list[str] = []
    head = "stale-work watchdog  (LIVE sweep)" if live else "stale-work watchdog  (dry-run)"
    lines.append(head)
    lines.append(f"  repo: {rep.repo}")
    # AGE-GC
    by_label: dict[str, list[int]] = {}
    for c in rep.age_candidates:
        b = by_label.setdefault(c["label"], [0, 0])
        b[0] += 1
        b[1] += c["size"]
    verb = "swept" if live else "would sweep"
    lines.append(f"  AGE-GC (> {rep.max_age_days}d): {verb} {rep.age_files} files, {_fmt_mb(rep.age_bytes)}")
    for label in sorted(by_label):
        n, b = by_label[label]
        lines.append(f"      {label:16s} {n:5d} files  {_fmt_mb(b)}")
    if rep.age_candidates[:3]:
        oldest = rep.age_candidates[0]
        lines.append(f"      oldest: {oldest['rel']}  ({oldest['age_days']}d)")
    # STUCK
    if rep.stuck_sessions:
        lines.append(f"  STUCK (>= {rep.stuck_threshold} consecutive stop-failures): {len(rep.stuck_sessions)} sessions")
        for s in rep.stuck_sessions[:5]:
            lines.append(f"      {s['session']}  consecutive={s['consecutive']} total={s['total']}")
        lines.append("      -> report only; a human should inspect these sessions")
    else:
        lines.append(f"  STUCK: none (>= {rep.stuck_threshold} consecutive)")
    # WIP
    w = rep.wip
    flag = "  <-- STALE" if w.get("stale") else ""
    lines.append(f"  WIP (shared trunk): {w.get('count', 0)} uncommitted, "
                 f"oldest {w.get('oldest_age_hours', 0)}h{flag}")
    if w.get("oldest_path"):
        lines.append(f"      oldest: {w['oldest_path']}")
    lines.append("      -> report only; NEVER auto-committed (shared-tree peer-sweep hazard)")
    lines.append(f"  result: {'STALE WORK FOUND' if rep.has_stale else 'clean'}")
    if not live and rep.age_candidates:
        lines.append("  (re-run with --live to GC the over-age ephemera)")
    return "\n".join(lines)


def render_json(rep: StaleReport, live: bool) -> str:
    return json.dumps({
        "repo": rep.repo,
        "live": live,
        "max_age_days": rep.max_age_days,
        "age_gc": {
            "files": rep.age_files,
            "bytes": rep.age_bytes,
            "swept": rep.swept,
            "candidates": rep.age_candidates,
        },
        "stuck": rep.stuck_sessions,
        "wip": rep.wip,
        "has_stale": rep.has_stale,
    }, indent=2)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=str(DEFAULT_REPO),
                    help="repo root to garden (default: this clone)")
    ap.add_argument("--max-age-days", type=int, default=7,
                    help="GC gitignored ephemera older than this (default 7)")
    ap.add_argument("--stuck-threshold", type=int, default=3,
                    help="flag stop-failure sessions at/above N consecutive (default 3)")
    ap.add_argument("--wip-stale-hours", type=int, default=24,
                    help="flag uncommitted work older than this many hours (default 24)")
    ap.add_argument("--live", action="store_true",
                    help="actually delete over-age ephemera (default: report only)")
    ap.add_argument("--fail-on-stale", action="store_true",
                    help="exit 2 if any stale work is found (for CI / alerting)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)

    repo = Path(args.repo).resolve()
    now = time.time()
    rep = build_report(repo, now, args.max_age_days, args.stuck_threshold,
                       args.wip_stale_hours, args.live)

    print(render_json(rep, args.live) if args.json else render_human(rep, args.live))

    if args.fail_on_stale and rep.has_stale:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
