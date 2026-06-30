#!/usr/bin/env python3
"""commit_stamp_doctor — is the fleet's recent history verifiable by the DOS referee?

`dos doctor` reports a single binary line ("none of your last 50 commits name a unit
of work"). This is the per-commit complement: it scans the last N commits and reports
how many carry a ship-stamp `dos verify` / the goal-gate Stop hook can actually BIND,
so the fleet can watch the number move instead of guessing.

Background (DOS-INSTALL-AUDIT-2026-06-19): fleet writes Conventional-Commits subjects
(`feat(scope): …`, `fix(scope): …`), which a start-anchored ship grammar cannot see.
With `[stamp] trailer_stamp = true` in `dos.toml`, a `(fak <leaf>)` trailer at the END
of the subject is bound as a direct ship of leaf `<leaf>` (docs/289). This tool mirrors
the oracle's two bindable shapes plus the release anchor; it is a fast heuristic health
check, NOT the oracle itself (`dos verify fak <leaf>` is the source of truth for whether
a given leaf shipped).

A commit counts toward the denominator unless it is BOOKKEEPING (a merge, a bulk
`… snapshot:`, or a `docs/_plans:`/`docs/dispatch:`-style rollup) — those name work as
narrative and were never meant to be a ship attribution.

Usage:
    python tools/commit_stamp_doctor.py                # last 50, report-only
    python tools/commit_stamp_doctor.py -n 100         # last 100
    python tools/commit_stamp_doctor.py --json
    python tools/commit_stamp_doctor.py --min-coverage 60   # exit 1 if below (gating)

Exit codes: 0 = ok (or report-only); 1 = coverage below --min-coverage; 2 = git error.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from dispatch_worker import install_no_window_subprocess_defaults
import sys
from pathlib import Path
install_no_window_subprocess_defaults(subprocess)

# The two bindable per-leaf ship shapes the oracle recognizes for fleet's stamp
# convention (dos.toml [stamp]: style=grep, subject_dirs=[], trailer_stamp=true):
#   1. trailer (docs/289): subject TAIL is `(fak <leaf>)` / `(fak: <leaf>)` /
#      `(refs fak <leaf>)`. The leaf token is alnum-led, may carry `.`/`-`/`_`.
#   2. direct start-prefix (legacy): `fak/<leaf>: …`.
_TRAILER_RE = re.compile(r"\((?:refs\s+)?fak[ :]+[A-Za-z0-9][\w.\-]*\)\s*$", re.IGNORECASE)
_DIRECT_RE = re.compile(r"^fak/[A-Za-z0-9][\w.\-]*:", re.IGNORECASE)
# A `vX.Y.Z:` release commit bundles several ships — recognized, but not a per-leaf
# stamp, so it is reported on its own line rather than counted as a leaf ship.
_RELEASE_RE = re.compile(r"^v\d+\.\d+\.\d+:")
# Bookkeeping subjects name phase/leaf ids as narrative and must never count as a ship
# (DOS FQ-77). Excluded from the denominator entirely.
_BOOKKEEPING_RE = re.compile(
    r"^(?:Merge\b|[^:]*\bsnapshot:|docs/(?:_plans|_soaks|fanout|dispatch|dispatch-loop):)",
    re.IGNORECASE,
)

# Extract the <leaf> token from a stamp so it can be checked against the declared
# lanes — a `(fak gatway)` typo binds `dos verify fak gatway` to a phantom unit
# nobody queries, the silent half of the trust gap this tool exists to close.
_TRAILER_LEAF_RE = re.compile(r"\((?:refs\s+)?fak[ :]+([A-Za-z0-9][\w.\-]*)\)\s*$", re.IGNORECASE)
_DIRECT_LEAF_RE = re.compile(r"^fak/([A-Za-z0-9][\w.\-]*):", re.IGNORECASE)


def stamp_leaf(subject: str) -> str | None:
    """The lowercased <leaf> token from a `(fak <leaf>)` trailer or `fak/<leaf>:`
    prefix, else None (the subject carries no per-leaf stamp)."""
    m = _TRAILER_LEAF_RE.search(subject) or _DIRECT_LEAF_RE.match(subject)
    return m.group(1).lower() if m else None


def declared_lanes(repo_root=None) -> set:
    """The lane/leaf names DOS recognizes for this workspace, read from dos.toml
    `[lanes]` (concurrent + exclusive + autopick + the `[lanes.trees]` keys). A
    stamp leaf NOT in this set is most likely a typo or a non-lane label — `dos
    verify fak <leaf>` would bind it to a unit no one queries by name. Returns an
    empty set when dos.toml is unreadable, so the off-lane check is SKIPPED (never
    failed) on a parse error."""
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parent.parent
    try:
        import tomllib  # py3.11+
        data = tomllib.loads((root / "dos.toml").read_text(encoding="utf-8-sig"))
    except Exception:
        return set()
    lanes = data.get("lanes", {})
    names = set()
    for key in ("concurrent", "exclusive", "autopick"):
        names.update(str(x).lower() for x in lanes.get(key, []) if isinstance(x, str))
    trees = lanes.get("trees", {})
    if isinstance(trees, dict):
        names.update(str(k).lower() for k in trees)
    return names


def cmd_demo_leaves(repo_root=None) -> set:
    """Lowercased names of the real `cmd/<name>/` directories.

    Settling #518: the `cmd` lane owns `cmd/**` as ONE tree (`cmd = ["cmd/**"]`),
    so a demo/binary under `cmd/<name>/` has no `internal/<name>/` package and is
    NOT its own declared lane. Yet the fleet stamps such ships `(fak <name>)` to
    keep per-demo attribution in the subject (e.g. `(fak turntaxdemo)`). That leaf
    binds to the `cmd` lane's tree, so it is a legitimate ship — not the typo the
    off-lane warning hunts for. Convention (AGENTS.md): stamp a `cmd/` demo with
    its directory name; this set recognizes any leaf that maps to a real `cmd/<name>`
    dir, so a RESIDUAL off-lane entry reliably means a typo or a non-lane label, not
    a `cmd/` demo. Returns an empty set when `cmd/` is absent (check then SKIPPED)."""
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parent.parent
    cmd_dir = root / "cmd"
    if not cmd_dir.is_dir():
        return set()
    return {p.name.lower() for p in cmd_dir.iterdir() if p.is_dir()}


def _git_log(n: int) -> list[tuple[str, str]]:
    """Return [(short_sha, subject), …] for the last n commits, newest first."""
    out = subprocess.run(
        ["git", "log", f"-{n}", "--format=%h%x09%s"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if out.returncode != 0:
        sys.stderr.write(out.stderr)
        raise SystemExit(2)
    rows = []
    for line in out.stdout.splitlines():
        if "\t" in line:
            sha, subj = line.split("\t", 1)
            rows.append((sha.strip(), subj.strip()))
    return rows


def classify(subject: str) -> str:
    """One of: stamped-trailer, stamped-direct, release, bookkeeping, unstamped."""
    if _BOOKKEEPING_RE.match(subject):
        return "bookkeeping"
    if _DIRECT_RE.match(subject):
        return "stamped-direct"
    if _TRAILER_RE.search(subject):
        return "stamped-trailer"
    if _RELEASE_RE.match(subject):
        return "release"
    return "unstamped"


def audit(n: int) -> dict:
    rows = [(sha, subj, classify(subj)) for sha, subj in _git_log(n)]
    kinds = [k for _, _, k in rows]
    stamped = kinds.count("stamped-trailer") + kinds.count("stamped-direct")
    release = kinds.count("release")
    bookkeeping = kinds.count("bookkeeping")
    unstamped = kinds.count("unstamped")
    denom = len(rows) - bookkeeping  # bookkeeping is not a unit of work
    coverage = round(100.0 * (stamped + release) / denom, 1) if denom else 0.0
    # Off-lane trailers: stamped, but the leaf binds to nothing the taxonomy knows.
    # A leaf is RECOGNIZED if it is a declared lane OR a real `cmd/<leaf>/` demo dir
    # (#518: cmd/ demos stamp their dir name and bind to the cmd lane's `cmd/**` tree).
    # What's left after both is a genuine typo / non-lane label — the real signal here.
    lanes = declared_lanes()
    cmd_demos = cmd_demo_leaves()
    recognized = lanes | cmd_demos
    off_lane = []
    if lanes:
        for sha, subj, k in rows:
            if k in ("stamped-trailer", "stamped-direct"):
                leaf = stamp_leaf(subj)
                if leaf and leaf not in recognized:
                    off_lane.append({"sha": sha, "leaf": leaf, "subject": subj})
    return {
        "scanned": len(rows),
        "stamped": stamped,
        "stamped_trailer": kinds.count("stamped-trailer"),
        "stamped_direct": kinds.count("stamped-direct"),
        "release": release,
        "bookkeeping": bookkeeping,
        "unstamped": unstamped,
        "denominator": denom,
        "coverage_pct": coverage,
        "known_lanes": len(lanes),
        "cmd_demos_recognized": len(cmd_demos),
        "off_lane_trailers": off_lane[:15],
        "unstamped_samples": [
            {"sha": sha, "subject": subj}
            for sha, subj, k in rows if k == "unstamped"
        ][:15],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--num", type=int, default=50, help="commits to scan (default 50)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--min-coverage", type=float, default=None,
                    help="exit 1 if bindable-stamp coverage is below this percent")
    args = ap.parse_args()

    rep = audit(args.num)
    if args.json:
        print(json.dumps(rep, indent=2))
    else:
        print(f"commit-stamp doctor - last {rep['scanned']} commits")
        print(f"  bindable ship-stamp coverage : {rep['coverage_pct']}%  "
              f"({rep['stamped'] + rep['release']}/{rep['denominator']} non-bookkeeping)")
        print(f"    trailer  (fak <leaf>)      : {rep['stamped_trailer']}")
        print(f"    direct   fak/<leaf>:       : {rep['stamped_direct']}")
        print(f"    release  vX.Y.Z:           : {rep['release']}")
        print(f"  unstamped (referee-blind)    : {rep['unstamped']}")
        print(f"  bookkeeping (excluded)       : {rep['bookkeeping']}")
        if rep["off_lane_trailers"]:
            print(f"  off-lane stamps (typo? not in {rep['known_lanes']} lanes "
                  f"+ {rep['cmd_demos_recognized']} cmd/ demos):")
            for s in rep["off_lane_trailers"]:
                print(f"    {s['sha']}  (fak {s['leaf']})  {s['subject'][:56]}")
        if rep["unstamped_samples"]:
            print("  examples a referee cannot bind (add a `(fak <leaf>)` trailer):")
            for s in rep["unstamped_samples"]:
                print(f"    {s['sha']}  {s['subject'][:72]}")
    if args.min_coverage is not None and rep["coverage_pct"] < args.min_coverage:
        sys.stderr.write(
            f"\nFAIL: coverage {rep['coverage_pct']}% < required {args.min_coverage}%\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
