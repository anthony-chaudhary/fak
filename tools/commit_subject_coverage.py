#!/usr/bin/env python3
"""Witness-gradeable commit-subject coverage — the watched-number twin of commit_stamp_doctor.

The DOS commit-audit witness grades a commit as diff-witnessed only when the
subject is `type(scope): <verb> <what>` with a recognized leading verb; a
noun-led subject (`feat(model): MiniMax witness ...`) ABSTAINs, and an ABSTAIN on
a landed commit is immutable. The closure auditor measured the cost: ABSTAIN is
the DOMINANT cause of CLAIMED_CLOSED -- ~13 of 43 closed-but-unwitnessed issues
have a resolving commit that touched real source yet the subject phrasing made the
witness abstain (docs/notes/DOS-EFFECTIVE-USAGE-AUDIT and the closure audit).

`tools/check_commit_msg.py` already nudges ONE commit toward the gradeable shape at
commit time (the advisory commit-msg hook). What was missing is the COVERAGE
number: what fraction of recent ship commits are witness-gradeable? Without it the
ABSTAIN-with-source rate is guessed, not watched -- exactly the gap
`commit_stamp_doctor` closed for `(fak <leaf>)` trailer coverage. This is its twin
for the subject-verb half.

It reuses `check_commit_msg.verdict` verbatim (single source of truth for the verb
set), folds it over the last N commit subjects (merges / reverts / version-bumps
excluded -- they make no gradeable claim), and reports the gradeable fraction plus
the ABSTAIN-prone subjects to fix. Pure git-log read; no gh / dos / network.

    python tools/commit_subject_coverage.py                  # the card
    python tools/commit_subject_coverage.py --json
    python tools/commit_subject_coverage.py --min-coverage 80   # advisory floor (exit 1 if below)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
from dispatch_worker import install_no_window_subprocess_defaults
import sys
from pathlib import Path
from typing import Any, Callable
install_no_window_subprocess_defaults(subprocess)

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except (AttributeError, ValueError):
    pass

SCHEMA = "fleet-commit-subject-coverage/1"
DEFAULT_LAST = 50
# A version-bump subject ('v0.32.0: ...') is a release, not a phase claim; the
# witness classifies it as a release-bump, so it is exempt from the denominator.
_VERSION_RE = re.compile(r"^v\d+\.\d+\.\d+")


def repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    return here.parent.parent


def _load_check_commit_msg():
    """Import check_commit_msg as the single source of truth for verdict()/verbs."""
    path = Path(__file__).resolve().parent / "check_commit_msg.py"
    spec = importlib.util.spec_from_file_location("check_commit_msg", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ccm = _load_check_commit_msg()


def is_exempt(subject: str) -> bool:
    """Merge / revert / fixup / squash / version-bump subjects make no gradeable claim."""
    return subject.startswith(_ccm.EXEMPT_PREFIXES) or bool(_VERSION_RE.match(subject))


def coverage(subjects: list[str]) -> dict[str, Any]:
    """Fold check_commit_msg.verdict over subjects -> gradeable fraction + the abstains."""
    total = 0
    gradeable = 0
    abstains: list[dict[str, str]] = []
    for raw in subjects:
        s = _ccm.first_line(raw)
        if not s or is_exempt(s):
            continue
        total += 1
        why = _ccm.verdict(s)
        if why is None:
            gradeable += 1
        else:
            abstains.append({"subject": s, "reason": why})
    frac = round(gradeable / total, 4) if total else None
    return {
        "total": total,
        "gradeable": gradeable,
        "abstain": len(abstains),
        "coverage": frac,
        "abstain_subjects": abstains,
    }


SubjectFetcher = Callable[[Path, int], list[str]]


def recent_subjects(root: Path, n: int) -> list[str]:
    """The last n non-merge commit subjects via git log (newest first)."""
    try:
        proc = subprocess.run(
            ["git", "log", f"-{int(n)}", "--no-merges", "--pretty=format:%s"],
            cwd=str(root), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if proc.returncode != 0:
        return []
    return [ln for ln in (proc.stdout or "").splitlines() if ln.strip()]


def build_payload(*, root: str, cov: dict[str, Any], min_coverage: float | None) -> dict[str, Any]:
    frac = cov.get("coverage")
    pct = None if frac is None else round(frac * 100, 1)
    if frac is None:
        ok, verdict_str = True, "NO_GRADEABLE_COMMITS"
        reason = "no witness-gradeable ship commits in the window (all merges/reverts/bumps?)"
    elif min_coverage is not None and pct < min_coverage:
        ok, verdict_str = False, "BELOW_FLOOR"
        reason = (f"{pct}% of recent ship subjects are witness-gradeable, below the "
                  f"{min_coverage}% floor — {cov['abstain']} ABSTAIN-prone subject(s) "
                  f"(noun-led / non-conventional) the witness cannot grade")
    else:
        ok, verdict_str = True, "OK"
        reason = (f"{pct}% of recent ship subjects are witness-gradeable "
                  f"({cov['gradeable']}/{cov['total']})")
    return {
        "schema": SCHEMA,
        "ok": ok,
        "verdict": verdict_str,
        "reason": reason,
        "workspace": root,
        "coverage_pct": pct,
        "min_coverage": min_coverage,
        **cov,
    }


def collect(root: Path, *, last: int, min_coverage: float | None,
            fetcher: SubjectFetcher | None = None) -> dict[str, Any]:
    fetch = fetcher or recent_subjects
    cov = coverage(fetch(root, last))
    return build_payload(root=str(root), cov=cov, min_coverage=min_coverage)


def render(p: dict[str, Any]) -> str:
    lines = [
        f"commit-subject coverage: {p.get('verdict')} ({'ok' if p.get('ok') else 'ACTION'})",
        f"  {p.get('reason')}",
        f"  gradeable={p.get('gradeable')}/{p.get('total')}  "
        f"coverage={p.get('coverage_pct')}%  abstain={p.get('abstain')}",
    ]
    abstains = p.get("abstain_subjects") or []
    if abstains:
        lines.append("  ABSTAIN-prone (fix: lead the description with a recognized verb):")
        for a in abstains[:12]:
            lines.append(f"    - {a['subject'][:78]}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Witness-gradeable commit-subject coverage meter (read-only).")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--last", type=int, default=DEFAULT_LAST,
                    help=f"commits to scan (default {DEFAULT_LAST})")
    ap.add_argument("--min-coverage", type=float, default=None,
                    help="advisory floor pct; exit 1 if below (for a `|| echo` CI warning)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args(argv)

    root = Path(args.workspace).resolve() if args.workspace else repo_root()
    payload = collect(root, last=max(1, args.last), min_coverage=args.min_coverage)
    print(json.dumps(payload, indent=2) if args.json else render(payload))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
