#!/usr/bin/env python3
"""loop-audit-signal: turn a DARK background loop into a specific, deduped GitHub
issue the dispatch loop can resolve — the loop-fleet counterpart of score-signal
(tools/score_signal.py). score-signal externalizes a scorecard regression; this
externalizes the loop fleet's OWN liveness gap.

The fleet already has a rich READ: `fak loop health --json` folds the loop ledger +
registry into a per-loop `dark` verdict (with the #4989 OS-scheduler reconcile that
distinguishes a genuinely-dark loop from one whose mapped OS task fired but wrote no
ledger row). What was missing is the FEED: nothing turned that verdict into a tracked
ticket, so a rostered loop could go dark and stay dark with no issue. This tool closes
that gap — child of the operator epic #4954, filed as #5151.

Scope (v1, #5151): file on DARK — a registered loop known to the schedule that is not
observed ticking. SPINNING (up but not producing witnessed-done runs, #4956) is plumbed
behind --include-spin but OFF by default until #4956 lands its keep-rate cut, so v1
signals only the crisp, unambiguous gap.

Dedup, cap, and dry-run default mirror score-signal exactly: at most ONE open issue per
loop (anchored on a stable `<!-- fak-loop-audit: <loop_id> -->` marker, scoped to this
tool's own label), a worst-first --max-issues cap to bound a storm, and dry-run unless
--live is passed. The pure plan/dedup core is proven by tools/loop_audit_signal_test.py.

    python tools/loop_audit_signal.py                     # dry-run, fold live
    python tools/loop_audit_signal.py --from health.json  # dry-run over a captured fold
    python tools/loop_audit_signal.py --live              # actually file (armed)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except (AttributeError, ValueError):
    pass

_CREATE_NO_WINDOW = 0x08000000


def _win_creationflags() -> int:
    return _CREATE_NO_WINDOW if os.name == "nt" else 0


SCHEMA = "fak-loop-audit-signal/1"
SIGNAL_LABEL = "loop-audit"
HEALTH_LABEL = "loop-health"
# The dedup anchor. Loop ids are ledger LoopIDs / registry JobIDs — kebab/dotted/
# slashed tokens (e.g. "learning-docs-freshness", "nightrun.collect"). The class is
# deliberately permissive so any real loop id round-trips through the marker.
MARKER_RE = re.compile(r"fak-loop-audit:\s*([A-Za-z0-9_.\-/]+)")

DEFAULTS = {
    "max_issues": 5,          # hard cap on issues filed per run (anti-storm)
    "issue_scan_limit": 800,  # open loop-audit issues fetched for the dedup index
    "spin_min_runs": 5,       # SPINNING needs at least this many ended runs to judge
    "spin_max_keep": 0.0,     # ...and a keep-rate <= this (0.0 = zero witnessed dones)
}

# Loops live in cmd/fak (the drivers + `fak loop`), internal/loopmgr (the fold), and
# .github/workflows (the schedulers). Route the ticket to the `cmd` lane — that is
# where a dark loop's driver/registry is re-armed.
LANE = "cmd"


# ============================================================================
# Pure helpers (no I/O) — unit-tested directly by loop_audit_signal_test.py.
# ============================================================================
def marker(loop_id: str) -> str:
    return f"<!-- fak-loop-audit: {loop_id} -->"


def _row_never_ticked(row: dict[str, Any]) -> bool:
    """A registered loop with no ledger tick at all — dark by the never-ticked rule,
    not by age. last_tick_unix_nano is 0/absent and it has never ledgered."""
    return int(row.get("last_tick_unix_nano", 0) or 0) == 0


def dark_candidates(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Rows that are the canonical DARK loop: dark AND registered (known to a
    schedule) AND NOT os_fired_no_ledger_row (#4989 already explains that softer
    state — a loop whose OS task fired 0x0 within cadence is not a false alarm to
    re-file). Worst-first: never-ticked loops before aged ones, then by age desc."""
    out = []
    for row in report.get("rows", []) or []:
        if not row.get("dark"):
            continue
        if not row.get("registered"):
            continue  # an ad-hoc ledger entry is not a scheduled loop
        if row.get("os_fired_no_ledger_row"):
            continue  # #4989: OS proved it fired; not a clean DARK to file
        out.append(row)
    out.sort(key=lambda r: (0 if _row_never_ticked(r) else 1,
                            -int(r.get("age_seconds", 0) or 0)))
    return out


def spin_candidates(report: dict[str, Any], *, min_runs: int,
                    max_keep: float) -> list[dict[str, Any]]:
    """Rows that are UP but not producing witnessed-done runs (#4956 SPINNING): a
    live loop with enough ended runs to judge and a keep-rate at/below the floor.
    keep_rate is -1 when runs==0 (no denominator) — those are excluded by the runs
    gate, never mistaken for a 0% keep. Worst-first: most ended runs (most wasted
    work) first. Gated behind --include-spin; off in v1 (#5151)."""
    out = []
    for row in report.get("rows", []) or []:
        if row.get("dark"):
            continue
        if row.get("state") != "live":
            continue
        runs = int(row.get("runs", 0) or 0)
        keep = float(row.get("keep_rate", -1) if row.get("keep_rate") is not None else -1)
        if runs < min_runs:
            continue
        if keep < 0:
            continue
        if keep > max_keep:
            continue
        out.append(row)
    out.sort(key=lambda r: -int(r.get("runs", 0) or 0))
    return out


def plan(report: dict[str, Any], open_ids: set[str], *, max_issues: int,
         today: str, include_spin: bool = False,
         spin_min_runs: int = DEFAULTS["spin_min_runs"],
         spin_max_keep: float = DEFAULTS["spin_max_keep"],
         ) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """DARK candidates (then optionally SPINNING) -> dedup vs OPEN issues -> CAP.
    Deterministic: dark worst-first, spin worst-first, deduped by loop id, then the
    combined list is capped so dark (listed first) survives the cap over spin. One
    open issue per loop: a loop is dark OR spinning, never both, so a single marker
    per loop id is exact."""
    stats = {"already-open": 0, "within-run-dup": 0, "over-cap": 0,
             "dark": 0, "spinning": 0}
    ranked: list[tuple[str, dict[str, Any]]] = [("dark", r) for r in dark_candidates(report)]
    if include_spin:
        ranked += [("spinning", r) for r in spin_candidates(
            report, min_runs=spin_min_runs, max_keep=spin_max_keep)]

    out: list[dict[str, Any]] = []
    seen_run: set[str] = set()
    for reason, row in ranked:
        loop_id = str(row.get("loop_id") or "")
        if not loop_id:
            continue
        if loop_id in seen_run:
            stats["within-run-dup"] += 1
            continue
        seen_run.add(loop_id)
        if loop_id in open_ids:
            stats["already-open"] += 1
            continue
        stats[reason] += 1
        out.append(render_issue(row, reason, report, today))

    if len(out) > max_issues:
        stats["over-cap"] = len(out) - max_issues
        out = out[:max_issues]
    return out, stats


def _tick_phrase(row: dict[str, Any]) -> str:
    if _row_never_ticked(row):
        return "never ticked (registered to a schedule, never observed firing)"
    age = int(row.get("age_seconds", 0) or 0)
    days = age // 86400
    hrs = (age % 86400) // 3600
    span = f"{days}d {hrs}h" if days else f"{hrs}h"
    return f"last ticked {span} ago (age {age}s past cadence)"


def _cadence_phrase(row: dict[str, Any]) -> str:
    sec = int(row.get("cadence_seconds", 0) or 0)
    src = str(row.get("cadence_source") or "")
    if sec <= 0:
        return "unknown cadence"
    unit = f"{sec}s"
    if sec % 86400 == 0:
        unit = f"{sec // 86400}d"
    elif sec % 3600 == 0:
        unit = f"{sec // 3600}h"
    return f"{unit} cadence" + (f" (from {src})" if src else "")


def render_issue(row: dict[str, Any], reason: str, report: dict[str, Any],
                 today: str) -> dict[str, Any]:
    """Build one deduped issue for a dark (or spinning) loop. The body names the loop,
    the evidence (state/cadence/last-tick/runs), the re-witness command, and the
    `#N`-stamp closure contract, so a worker (or a human) can act without re-deriving
    the fold."""
    loop_id = str(row.get("loop_id") or "")
    runs = int(row.get("runs", 0) or 0)
    witnessed = int(row.get("witnessed", 0) or 0)
    keep = row.get("keep_rate", -1)
    keep_s = "n/a (no ended runs)" if (keep is None or keep < 0) else f"{float(keep):.0%}"

    if reason == "dark":
        title = f"loop-audit: {loop_id} is DARK — {_tick_phrase(row)}"
        state_line = (f"**State:** DARK — {_tick_phrase(row)}; {_cadence_phrase(row)}.\n")
        why = ("A loop registered to a schedule is not observed ticking. Its scheduled "
               "work is not running, and any telemetry it produces is going stale — the "
               "exact silent-loop failure a health fold exists to surface.\n")
        do = (
            "1. Confirm it is still dark: `fak loop health --json` (this loop's row) "
            "or `fak loop health --check` (exits 3 while any loop is dark).\n"
            "2. Find the loop's driver + schedule — its registry entry "
            "(`fak loop health --registry`) and the workflow/OS task that should fire "
            "it — and re-arm whatever stopped it.\n"
            "3. If the loop was intentionally retired, remove it from the loop registry "
            "so it stops reading DARK (a retired loop should leave the roster, not "
            "linger as a false alarm).\n"
            "4. Ship a commit citing this issue's `#N` in the subject with a "
            f"`(fak {LANE})` trailer, so the witness can bind and close it.\n")
        witness = ("`fak loop health --check` passes (this loop no longer dark), or the "
                   "loop is removed from the registry with the reason recorded.\n")
    else:  # spinning
        title = (f"loop-audit: {loop_id} is SPINNING — {runs} ended runs, "
                 f"{witnessed} witnessed ({keep_s} kept)")
        state_line = (f"**State:** SPINNING — live and ticking, but {witnessed}/{runs} "
                      f"ended runs reached a witnessed-done verdict ({keep_s} keep-rate); "
                      f"{_cadence_phrase(row)}.\n")
        why = ("The loop is up and firing on cadence but its ended runs are not being "
               "independently witnessed as done — it is talking, not proven done "
               "(#4956). Cadence liveness alone hides this; only the keep-rate shows it.\n")
        do = (
            "1. Re-fold: `fak loop health --json` — confirm the loop's keep-rate is "
            "still at/below floor over enough ended runs.\n"
            "2. Trace why runs end unwitnessed — a missing/failing witness step, a "
            "run that ends before it produces the witnessed artifact, or a witness "
            "the loop never emits.\n"
            "3. Fix the smallest responsible cause so ended runs reach a witnessed-done "
            "verdict; change no unrelated loop.\n"
            "4. Ship a commit citing this issue's `#N` with a "
            f"`(fak {LANE})` trailer.\n")
        witness = ("`fak loop health --json` shows this loop's keep-rate risen above the "
                   "floor over its ended runs.\n")

    ts = int(report.get("ts_unix_nano", 0) or 0)
    fold_when = ""
    if ts:
        fold_when = dt.datetime.fromtimestamp(
            ts / 1e9, dt.timezone.utc).strftime("%Y-%m-%d %H:%MZ")

    body = (
        f"> Auto-filed by **loop-audit-signal** (`tools/loop_audit_signal.py`, {today}) "
        f"from the `fak loop health` fold"
        + (f" @ {fold_when}" if fold_when else "")
        + ". A background loop the fleet rosters is not healthy — **needs a worker**; "
        "close as `wontfix` if the loop is intentionally idle and retire it from the "
        "registry.\n\n"
        f"**Loop:** `{loop_id}`\n"
        + state_line
        + f"**Runs:** {runs} ended, {witnessed} witnessed-done ({keep_s} kept).\n\n"
        "### Why this is next\n"
        + why
        + "\n### What to do\n"
        + do
        + "\n### Witness\n"
        + witness
        + "\n### Lane\n"
        + LANE + "\n\n"
        "### Path hints\n"
        "- `cmd/fak/loop.go` (the `fak loop health` fold + `runLoopHealth`)\n"
        "- `internal/loopmgr/health.go` (the per-loop verdict)\n"
        "- the loop registry (`fak loop health --registry`) + the workflow/OS task that "
        "fires this loop\n\n"
        "### Boundary notes\n"
        "Public loop-health verdict only; do not paste private operator telemetry.\n\n"
        "### Closure binding\n"
        f"Resolving commit cites this issue's #N in the subject and carries `(fak {LANE})`.\n"
        "\n---\n"
        "_The loop roster is folded by `fak loop health` (`internal/loopmgr`). After "
        "this issue is closed, a future recurrence for the same loop files a FRESH "
        "issue — only on an armed `--live` run; the scheduled CI run is dry-run. While "
        "this issue stays open, the loop is deduped by the marker below and never "
        "re-filed._\n"
        + marker(loop_id)
    )

    return {
        "title": title,
        "body": body,
        "labels": [SIGNAL_LABEL, HEALTH_LABEL],
        "loop_id": loop_id,
        "reason": reason,
    }


def open_issue_ids(issues: list[dict[str, Any]]) -> set[str]:
    """The loop ids already tracked by an OPEN loop-audit issue, parsed from the
    dedup marker in each body. Scoping the fetch to this tool's own label (below)
    keeps this index exact no matter how large the total backlog grows."""
    out: set[str] = set()
    for it in issues:
        m = MARKER_RE.search(str(it.get("body") or ""))
        if m:
            out.add(m.group(1))
    return out


# ============================================================================
# I/O boundary — gh + the loop-health fold. Thin wrappers so the logic above stays
# testable.
# ============================================================================
def gh_json(args: list[str], timeout: int = 60) -> Any:
    proc = subprocess.run(["gh", *args], capture_output=True, text=True,
                          encoding="utf-8", timeout=timeout,
                          creationflags=_win_creationflags())
    if proc.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args)} -> {proc.returncode}: "
                           f"{proc.stderr.strip()[:300]}")
    out = proc.stdout.strip()
    return json.loads(out) if out else []


def fetch_open_issues(limit: int) -> list[dict[str, Any]]:
    """Open issues carrying the loop-audit label — the dedup index. Scoped to the
    tool's OWN label so dedup stays exact as the backlog grows; `gh issue list
    --label X` returns [] cleanly when the label does not exist yet (fresh repo)."""
    return gh_json([
        "issue", "list", "--state", "open", "--label", SIGNAL_LABEL,
        "--limit", str(limit), "--json", "number,title,body",
    ])


def ensure_labels() -> None:
    """Idempotently create the marker labels so `gh issue create` never fails on a
    missing label. Best-effort but not silent — an auth/permission failure here would
    otherwise resurface as a confusing per-issue 'label not found'."""
    wanted = [
        (SIGNAL_LABEL, "b60205",
         "Auto-filed by loop-audit-signal (tools/loop_audit_signal.py) from a dark/"
         "spinning background loop; needs a worker"),
        (HEALTH_LABEL, "0e8a16", "Background-loop liveness/progress health"),
    ]
    for name, color, desc in wanted:
        try:
            proc = subprocess.run(
                ["gh", "label", "create", name, "--color", color,
                 "--description", desc, "--force"],
                capture_output=True, text=True, encoding="utf-8", timeout=30,
                creationflags=_win_creationflags())
        except (OSError, subprocess.TimeoutExpired) as e:
            print(f"warning: could not run `gh label create {name}`: {e}",
                  file=sys.stderr)
            continue
        if proc.returncode != 0:
            print(f"warning: could not ensure '{name}' label "
                  f"(issue creation may fail): {proc.stderr.strip()[:200]}",
                  file=sys.stderr)


def create_issue(issue: dict[str, Any]) -> str:
    args = ["issue", "create", "--title", issue["title"], "--body", issue["body"]]
    for lab in issue["labels"]:
        args += ["--label", lab]
    proc = subprocess.run(["gh", *args], capture_output=True, text=True,
                          encoding="utf-8", creationflags=_win_creationflags())
    if proc.returncode != 0:
        raise RuntimeError(f"gh issue create -> {proc.returncode}: "
                           f"{proc.stderr.strip()[:300]}")
    return proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""


def load_health(from_arg: str, root: Path, timeout: int, fak_bin: str) -> dict[str, Any]:
    """The folded loop-health payload: read it from --from (a file or '-' for stdin),
    else fold it live via `fak loop health --json`. Raises RuntimeError on any failure
    so the caller exits 2 cleanly."""
    if from_arg == "-":
        raw = sys.stdin.read()
    elif from_arg:
        raw = Path(from_arg).read_text(encoding="utf-8")
    else:
        proc = subprocess.run(
            [fak_bin, "loop", "health", "--json"],
            cwd=str(root), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
            creationflags=_win_creationflags())
        raw = proc.stdout
        if not raw.strip():
            tail = (proc.stderr or "").strip().splitlines()[-1:] or [""]
            raise RuntimeError(f"`{fak_bin} loop health --json` emitted no JSON "
                               f"(exit {proc.returncode}): {tail[0][:200]}")
    try:
        payload = json.loads(raw)
    except ValueError as e:
        raise RuntimeError(f"loop-health payload is not JSON: {e}") from e
    if not isinstance(payload, dict):
        raise RuntimeError("loop-health payload is not an object")
    return payload


# ============================================================================
# Driver.
# ============================================================================
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="loop-audit-signal: a DARK/SPINNING background loop -> a deduped "
                    "GitHub issue.")
    ap.add_argument("--workspace", default=".", help="repo root. Default: cwd.")
    ap.add_argument("--from", dest="from_arg", default="",
                    help="read a pre-folded `fak loop health --json` from this file "
                         "('-' = stdin). Default: fold it live. NOTE: a STALE fold "
                         "under --live can re-file a loop already re-armed; prefer a "
                         "live fold when arming --live.")
    ap.add_argument("--fak-bin", default=os.environ.get("FAK_BIN", "fak"),
                    help="the fak binary to fold with (default: $FAK_BIN or 'fak').")
    ap.add_argument("--max-issues", type=int,
                    help=f"hard cap on issues filed, worst-first "
                         f"(default {DEFAULTS['max_issues']}).")
    ap.add_argument("--include-spin", action="store_true",
                    help="also file SPINNING loops (live but not producing witnessed-"
                         "done runs, #4956). OFF by default (v1 files DARK only, #5151).")
    ap.add_argument("--spin-min-runs", type=int, default=DEFAULTS["spin_min_runs"],
                    help=f"SPINNING needs at least this many ended runs to judge "
                         f"(default {DEFAULTS['spin_min_runs']}).")
    ap.add_argument("--spin-max-keep", type=float, default=DEFAULTS["spin_max_keep"],
                    help=f"...and a keep-rate at/below this "
                         f"(default {DEFAULTS['spin_max_keep']}).")
    ap.add_argument("--live", action="store_true",
                    help="actually create the issues (default: dry-run).")
    ap.add_argument("--json", action="store_true", help="machine-readable output.")
    ap.add_argument("--timeout", type=int, default=120,
                    help="loop-health fold timeout seconds (when folding live).")
    args = ap.parse_args(argv)

    root = Path(args.workspace).resolve()
    today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    max_issues = args.max_issues if args.max_issues is not None else DEFAULTS["max_issues"]

    if args.live and args.from_arg:
        print("warning: --from with --live trusts a possibly STALE fold; a loop already "
              "re-armed could be re-filed. Prefer a live fold when arming --live.",
              file=sys.stderr)

    try:
        report = load_health(args.from_arg, root, args.timeout, args.fak_bin)
    except (OSError, RuntimeError, subprocess.SubprocessError) as e:
        print(f"refuse: could not load the loop-health payload: {e}", file=sys.stderr)
        return 2

    # The OPEN-issue dedup index is mandatory: with no way to know what is already
    # tracked, --live could re-file an open loop every tick. Refuse rather than storm.
    try:
        issues = fetch_open_issues(DEFAULTS["issue_scan_limit"])
        open_ids = open_issue_ids(issues)
    except Exception as e:  # noqa: BLE001
        print(f"refuse: cannot fetch open issues for the dedup index ({e})",
              file=sys.stderr)
        return 2

    to_file, stats = plan(
        report, open_ids, max_issues=max_issues, today=today,
        include_spin=args.include_spin, spin_min_runs=args.spin_min_runs,
        spin_max_keep=args.spin_max_keep)

    errors: list[str] = []
    filed: list[dict[str, Any]] = []
    if args.live and to_file:
        ensure_labels()
        for issue in to_file:
            try:
                url = create_issue(issue)
            except Exception as e:  # noqa: BLE001
                errors.append(f"create[{issue['loop_id']}]: {e}")
                continue
            filed.append({**issue, "issue_url": url})

    rollup = report.get("rollup") or {}
    result = {
        "schema": SCHEMA,
        "date": today,
        "mode": "live" if args.live else "dry-run",
        "rollup": {k: rollup.get(k) for k in ("loops", "live", "stale", "dark",
                                              "os_fired_no_ledger_row", "witness_collapse")
                   if k in rollup},
        "open_loop_issues": sorted(open_ids),
        "skipped": stats,
        "planned": [{"title": i["title"], "loop_id": i["loop_id"], "reason": i["reason"],
                     "labels": i["labels"]} for i in to_file],
        "filed": [{"title": f["title"], "loop_id": f["loop_id"],
                   "issue_url": f.get("issue_url", "")} for f in filed],
        "errors": errors,
    }

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    # Human report.
    rlp = result["rollup"]
    print(f"loop-audit-signal {today} — {result['mode']}  "
          f"(loops: {rlp.get('loops', '?')}, dark: {rlp.get('dark', '?')}, "
          f"os-fired-no-row: {rlp.get('os_fired_no_ledger_row', 0)})")
    # Only the true drop reasons belong under "deduped/dropped" — `dark`/`spinning`
    # are planned tallies, not drops, so surfacing them here would misread as loss.
    drop_keys = ("already-open", "within-run-dup", "over-cap")
    sk = ", ".join(f"{k}={stats[k]}" for k in drop_keys if stats.get(k)) or "none"
    print(f"  deduped/dropped: {sk}")
    if result["open_loop_issues"]:
        print(f"  already tracked (open): {', '.join(result['open_loop_issues'])}")
    if not to_file:
        print("  → no NEW unhealthy loop. Nothing to file.")
    else:
        verb = "FILED" if args.live else "would file"
        print(f"  → {verb} {len(to_file)} issue(s) (cap {max_issues}):")
        for i in to_file:
            f = next((x for x in filed if x["loop_id"] == i["loop_id"]), None)
            mark = ""
            if args.live:
                mark = f"  {f['issue_url']}" if f else "  (create failed)"
            print(f"     [{i['reason']:>8}] {i['title']}{mark}")
    if errors:
        print("  errors:")
        for e in errors:
            print(f"     ! {e}")
    if not args.live and to_file:
        print("\n  dry-run — file these for real with:  "
              "python tools/loop_audit_signal.py --live")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
