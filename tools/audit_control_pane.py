#!/usr/bin/env python3
"""audit_control_pane.py -- one rollup over the ~24 tools/*_audit.py auditors.

The repo grew a family of standalone auditors -- ``security_audit.py``,
``gofmt_debt_audit.py``, ``plan_audit.py``, ``history_leak_audit.py``,
``permission_source_audit.py`` and ~20 more -- each a self-contained
``argparse main()`` with its own paired ``*_test.py``. They share NO
entrypoint: an operator can only discover them by ``ls`` and must run each by
name to answer the one cross-cutting question "is anything failing an audit
right now?". Nothing folds the family into a single verdict.

This is that fold -- the audit FAMILY control pane, the sibling of
``scorecard_control_pane.py`` (which folds the scorecard family) and
``fresh_status.py`` (which folds git + benchmarks + work + industry). That one
answers "is the repo getting better or worse on quality?"; this one answers
"does any auditor flag a problem this minute?".

  python tools/audit_control_pane.py            # human rollup
  python tools/audit_control_pane.py --json      # machine control-pane payload
  python tools/audit_control_pane.py --list      # discovery only -- list the
                                                 # auditors, run nothing (fast)
  python tools/audit_control_pane.py --check     # CI gate: non-zero on any
                                                 # auditor that reports a finding
  python tools/audit_control_pane.py --only security_audit,gofmt_debt_audit
                                                 # run a named subset

The verdict contract -- the part the issue (#1137) says the fold must define
first, because most auditors do NOT yet expose a uniform ``--json`` verdict:

  * The UNIVERSAL signal is the PROCESS EXIT CODE. By repo convention every
    auditor exits 0 = clean / no finding, and non-zero = a finding (or that it
    could not run). All 24 honor this -- it is the one contract that spans the
    19 that have ``--json`` and the 5 that do not. The fold runs each auditor
    bare (no ``--json``, no ``--check`` -- not every auditor has them) and reads
    its exit code.
  * When an auditor's stdout DOES parse as a control-pane JSON envelope, the
    fold enriches the row with its ``verdict``/``reason``/``ok`` -- but the
    pass/fail decision still rests on the exit code, so a non-JSON auditor is a
    first-class citizen, never silently dropped.

Per-auditor row verdict (mirrors the scorecard_control_pane / fresh_status
ladder so a loop runner reads one envelope shape):

  * OK     -- exit 0. No finding.
  * ACTION -- exit non-zero AND not a known degrade code. A real finding the
              operator must look at.
  * SKIP   -- the auditor could not run here for an environmental reason and
              degrades softly (timed out within the bound, or its interpreter /
              a host-only probe is unavailable). A SKIP never trips the rollup
              -- the same SOFT-pane discipline ``fresh_status`` gives its work /
              industry panes. ``fresh_status`` folds ``plan_audit`` exactly this
              way: a slow / absent sub-tool degrades to SKIP rather than failing
              the parent.

The rollup verdict: ACTION if any auditor row is ACTION; otherwise OK (SKIPs
are reported but never trip it). ``--check`` exits non-zero ONLY on a rollup
ACTION -- the same advisory contract as the scorecard ratchet and fresh_status
(a finding fails the gate; a soft skip does not).

Pure-stdlib Python, repo-root resolved like the other honesty gates, ASCII-only
source (the provenance / leak gate forbids non-ASCII). The discovery + fold are
split from the live subprocess runner so the fold is unit-testable
(tools/audit_control_pane_test.py).
"""
from __future__ import annotations

import argparse
import json
import subprocess
from dispatch_worker import install_no_window_subprocess_defaults
install_no_window_subprocess_defaults(subprocess)
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = "fak-audit-control-pane/1"
AUDITS_GLOB = "*_audit.py"

# Auditors whose name matches the glob but are NOT cross-cutting "is anything
# failing right now?" auditors, or that are unsafe / pointless to run in the
# rollup. Empty by default -- every tools/*_audit.py is in scope -- but kept as
# the single seam for excluding one if it ever needs host state the rollup box
# does not have. (Names are the stem, e.g. "session_audit".)
EXCLUDE: frozenset[str] = frozenset()

# Per-auditor wall-clock bound. Generous -- the point is to degrade a hung /
# pathological auditor to SKIP, not to race a slow-but-honest one. fresh_status
# folds plan_audit with the same degrade-to-skip-on-timeout discipline.
DEFAULT_TIMEOUT = 90


def repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    return here.parent.parent


def _git_line(args: list[str], root: Path) -> str:
    try:
        p = subprocess.run(["git", *args], cwd=str(root), capture_output=True,
                           text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return ""
    if p.returncode != 0:
        return ""
    return p.stdout.strip()


def head_commit(root: Path) -> str:
    return _git_line(["rev-parse", "--short", "HEAD"], root) or "unknown"


# --- discovery (pure -- the tested surface) ---------------------------------

def discover(root: Path, *, only: list[str] | None = None,
             exclude: frozenset[str] = EXCLUDE) -> list[str]:
    """Discover the auditor stems under ``tools/`` in deterministic order.

    Matches ``tools/*_audit.py``, drops any ``*_test.py`` (a paired test is not
    an auditor) and anything in ``exclude``. ``only`` (a list of stems) narrows
    the set to a named subset, preserving discovery order; an ``only`` name that
    is not a discovered auditor is ignored (the caller's typo never invents a
    run). Returns the stems (``security_audit``), not paths.
    """
    tools = root / "tools"
    stems: list[str] = []
    for path in sorted(tools.glob(AUDITS_GLOB)):
        stem = path.stem
        if stem.endswith("_test"):
            continue
        if stem in exclude:
            continue
        stems.append(stem)
    if only:
        wanted = set(only)
        stems = [s for s in stems if s in wanted]
    return stems


# --- per-auditor row + the fold (pure -- the tested surface) ----------------

def classify_row(stem: str, *, returncode: int | None, timed_out: bool,
                 unrunnable: str = "", payload: dict[str, Any] | None = None
                 ) -> dict[str, Any]:
    """Classify one auditor outcome into a control-pane row (pure -- no I/O).

    The decision rests on the PROCESS EXIT CODE -- the one contract all 24
    auditors share. ``payload`` (a parsed control-pane JSON envelope, when the
    auditor emitted one) only ENRICHES the row's reason; it never overrides the
    exit-code verdict, so a non-JSON auditor is a first-class row.

      * timed_out / unrunnable -> SKIP (soft; never trips the rollup)
      * returncode == 0        -> OK
      * returncode != 0        -> ACTION (a finding to look at)
    """
    enrich = ""
    if isinstance(payload, dict):
        pv = payload.get("verdict")
        pr = payload.get("reason")
        if isinstance(pv, str) and pv:
            enrich = pv
        if isinstance(pr, str) and pr:
            enrich = f"{enrich}: {pr}" if enrich else pr

    if timed_out:
        return {
            "key": stem, "label": stem, "ok": True, "verdict": "SKIP",
            "returncode": None,
            "reason": "timed out within the per-auditor bound; skipped (soft -- "
                      "does not trip the rollup)",
        }
    if unrunnable:
        return {
            "key": stem, "label": stem, "ok": True, "verdict": "SKIP",
            "returncode": returncode,
            "reason": f"could not run here ({unrunnable}); skipped (soft)",
        }
    if returncode == 0:
        return {
            "key": stem, "label": stem, "ok": True, "verdict": "OK",
            "returncode": 0,
            "reason": enrich or "clean (exit 0)",
        }
    base = f"reported a finding (exit {returncode})"
    return {
        "key": stem, "label": stem, "ok": False, "verdict": "ACTION",
        "returncode": returncode,
        "reason": f"{base}: {enrich}" if enrich else base,
    }


def fold(rows: list[dict[str, Any]], *, workspace: str, commit: str,
         generated_at: str) -> dict[str, Any]:
    """Fold per-auditor rows into one control-pane payload + rollup verdict.

    ACTION rows trip the rollup to ACTION; SKIP rows are reported but never do
    (the SOFT-pane discipline). The envelope shape (schema/ok/verdict/finding/
    reason/next_action) mirrors scorecard_control_pane.fold and
    fresh_status.fold so a loop runner reads one shape across all three.
    """
    actionable = [r for r in rows if r.get("verdict") == "ACTION"]
    skipped = [r for r in rows if r.get("verdict") == "SKIP"]
    green = [r for r in rows if r.get("verdict") == "OK"]

    if actionable:
        ok, verdict, finding = False, "ACTION", "auditor_finding"
        names = ", ".join(r["label"] for r in actionable)
        reason = (f"{len(actionable)} auditor(s) report a finding ({names}); "
                  f"{len(green)} clean, {len(skipped)} skipped")
        first = actionable[0]
        next_action = (f"investigate {first['label']} first -- run "
                       f"`python tools/{first['label']}.py` for its full report")
    else:
        ok, verdict, finding = True, "OK", "all_clear"
        reason = f"{len(green)} auditor(s) clean"
        if skipped:
            reason += (f"; {len(skipped)} skipped ("
                       + ", ".join(r["label"] for r in skipped) + ")")
        next_action = "rollup is green; no auditor reports a finding"

    return {
        "schema": SCHEMA,
        "ok": ok,
        "verdict": verdict,
        "finding": finding,
        "reason": reason,
        "next_action": next_action,
        "workspace": workspace,
        "commit": commit,
        "generated_at": generated_at,
        "counts": {
            "total": len(rows),
            "clean": len(green),
            "action": len(actionable),
            "skipped": len(skipped),
        },
        "auditors": rows,
        "auditor_order": [r["key"] for r in rows],
    }


# --- live runner ------------------------------------------------------------

def run_auditor(root: Path, stem: str, *, python: str, timeout: int
                ) -> dict[str, Any]:
    """Run ``tools/<stem>.py`` bare and classify the outcome into a row.

    Runs the auditor with NO flags -- the exit code is the universal contract,
    and not every auditor has ``--json`` / ``--check``. If stdout happens to
    parse as a control-pane JSON envelope the row is enriched, but the verdict
    is the exit code. A timeout / OS error degrades the row to a soft SKIP.
    """
    script = root / "tools" / f"{stem}.py"
    if not script.exists():
        return classify_row(stem, returncode=None, timed_out=False,
                             unrunnable=f"missing tools/{stem}.py")
    try:
        proc = subprocess.run(
            [python, str(script)],
            cwd=str(root), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return classify_row(stem, returncode=None, timed_out=True)
    except (OSError, subprocess.SubprocessError) as exc:
        return classify_row(stem, returncode=None, timed_out=False,
                            unrunnable=str(exc))
    payload: dict[str, Any] | None = None
    try:
        maybe = json.loads(proc.stdout)
        if isinstance(maybe, dict):
            payload = maybe
    except ValueError:
        payload = None
    return classify_row(stem, returncode=proc.returncode, timed_out=False,
                        payload=payload)


def collect(root: Path, *, python: str = "", timeout: int = DEFAULT_TIMEOUT,
            only: list[str] | None = None) -> list[dict[str, Any]]:
    """Discover and run every auditor, returning the per-auditor rows."""
    python = python or sys.executable
    rows: list[dict[str, Any]] = []
    for stem in discover(root, only=only):
        rows.append(run_auditor(root, stem, python=python, timeout=timeout))
    return rows


# --- render -----------------------------------------------------------------

def render(payload: dict[str, Any]) -> str:
    counts = payload.get("counts", {})
    lines = [
        f"audit control pane -- {payload['verdict']} ({payload['finding']})  "
        f"@{payload['commit']}",
        f"  {counts.get('total', 0)} auditors: {counts.get('clean', 0)} clean, "
        f"{counts.get('action', 0)} action, {counts.get('skipped', 0)} skipped",
        f"  generated {payload['generated_at']}",
        "",
    ]
    mark = {"OK": "ok ", "SKIP": " - ", "ACTION": "XX "}
    for key in payload["auditor_order"]:
        r = next((x for x in payload["auditors"] if x["key"] == key), None)
        if r is None:
            continue
        lines.append(f"  {mark.get(r['verdict'], ' ? ')} {r['label']:<32} "
                     f"{r['reason']}")
    lines.extend(["", f"  -> {payload['next_action']}"])
    return "\n".join(lines)


def render_list(stems: list[str]) -> str:
    lines = [f"{len(stems)} auditors discovered under tools/{AUDITS_GLOB}:", ""]
    lines.extend(f"  {s}" for s in stems)
    return "\n".join(lines)


# --- main -------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Rollup over the tools/*_audit.py auditor family "
                    "(is anything failing an audit right now?).")
    ap.add_argument("--workspace", default="",
                    help="workspace root (default: repo root)")
    ap.add_argument("--json", action="store_true",
                    help="emit machine-readable JSON")
    ap.add_argument("--list", action="store_true",
                    help="discovery only: list the auditors and exit (runs nothing)")
    ap.add_argument("--check", action="store_true",
                    help="CI gate: exit non-zero only if an auditor reports a finding")
    ap.add_argument("--only", default="",
                    help="comma-separated subset of auditor stems to run "
                         "(e.g. security_audit,gofmt_debt_audit)")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                    help="per-auditor timeout seconds (timeout -> soft SKIP)")
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    root = Path(args.workspace).resolve() if args.workspace else repo_root()
    only = [s.strip() for s in args.only.split(",") if s.strip()] or None

    if args.list:
        stems = discover(root, only=only)
        if args.json:
            print(json.dumps({"schema": SCHEMA, "count": len(stems),
                              "auditors": stems}, indent=2))
        else:
            print(render_list(stems))
        return 0

    now = datetime.now(timezone.utc)
    rows = collect(root, timeout=args.timeout, only=only)
    payload = fold(rows, workspace=str(root), commit=head_commit(root),
                   generated_at=now.strftime("%Y-%m-%dT%H:%M:%SZ"))

    if args.check:
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(render(payload))
        # Advisory: non-zero ONLY on a rollup ACTION (a real finding). A soft
        # SKIP never trips the gate -- the same contract as fresh_status --check.
        return 0 if payload["verdict"] != "ACTION" else 1

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(render(payload))

    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
