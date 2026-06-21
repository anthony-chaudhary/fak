#!/usr/bin/env python3
"""Doc-placement gate: dated research/strategy docs belong in docs/notes/, not the repo root.

The repo root is a public front door. Dated strategy/SOTA/plan/forensic notes
(`PLAN-*`, `GLM*`, `VISUALS-*`, `*-AUDIT-*`, the SOTA memos, …) belong under
`docs/notes/` and are reached through `INDEX.md`, NOT scattered at the root. This
check enforces that mechanically so the front door cannot silently re-clutter as
many agents commit in parallel.

Rule: a top-level `*.md` file is allowed ONLY if it is one of the known
entry-point / meta files (ALLOWED_ROOT_MD). Any other root-level `.md` is a
placement violation and is directed to `docs/notes/`.

Modes:
  --audit-staged   scan staged additions/renames (the pre-commit hook).
  --audit-tree     scan the whole tracked tree (CI / DoD readiness backstop).

Exit: 0 = clean, 1 = violation, 2 = could not run (git error). The pre-commit
hook treats anything but 1 as fail-open so a broken check never wedges commits.

Escape hatch (staged mode): set ALLOW_ROOT_DOC=1 to override once (a genuine new
root/meta doc the allowlist doesn't yet name — then add it to ALLOWED_ROOT_MD).
"""
from __future__ import annotations
import argparse
import os
import re
import subprocess
import sys

# Root-level .md files that are legitimately entry points or project meta.
# Keep this list small and deliberate; everything else goes to docs/notes/.
ALLOWED_ROOT_MD = {
    # entry points
    "README.md", "START-HERE.md", "INSTALL.md", "INDEX.md",
    # contributor + agent contract
    "CONTRIBUTING.md", "CLA.md", "AGENTS.md", "CLAUDE.md",
    # policy / security
    "SECURITY.md", "PUBLIC-SCRUB-POLICY.md",
    # public front-door docs: deliberately at the root because README.md and
    # AGENTS.md link them by their root path (the module hoist made the repo root
    # the public home). Moving these to docs/notes/ would break those links.
    "ARCHITECTURE.md", "EXTENDING.md", "GETTING-STARTED.md", "GPU.md",
    "POLICY.md", "PARTITION.md", "STATUS.md", "CLAIMS.md",
    "SOTA-COMPARISON.md", "DOGFOOD-CLAUDE.md",
    # benchmark front-door set (the single-source-of-truth deck the README cites)
    "BENCHMARK-AUTHORITY.md", "BENCHMARK-GALLERY.md",
    "BENCHMARK-GOVERNANCE.md", "BENCHMARK-TEMPLATE.md",
    # common OSS root/meta docs (allowed if a maintainer adds one)
    "CODE_OF_CONDUCT.md", "CHANGELOG.md", "GOVERNANCE.md", "MAINTAINERS.md",
    "ROADMAP.md", "AUTHORS.md", "NOTICE.md", "SUPPORT.md", "HISTORY.md",
}
NOTES_DIR = "docs/notes"


def _git(args: list[str], root: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", root] + args, capture_output=True, text=True)


def _violations(names) -> list[str]:
    return sorted(
        n for n in names
        if n.endswith(".md") and "/" not in n and n not in ALLOWED_ROOT_MD
    )


def _staged_paths(root: str) -> list[str] | None:
    # Added (A) or renamed-INTO (R) staged files; the new path is the last field.
    r = _git(["diff", "--cached", "--name-status", "--diff-filter=AR"], root)
    if r.returncode != 0:
        return None
    out = []
    for line in r.stdout.splitlines():
        parts = line.split("\t")
        if parts:
            out.append(parts[-1])
    return out


def _tracked_paths(root: str) -> list[str] | None:
    r = _git(["ls-files"], root)
    if r.returncode != 0:
        return None
    return r.stdout.split()


def _staged_added(root: str) -> list[str]:
    r = _git(["diff", "--cached", "--name-status", "--diff-filter=A"], root)
    if r.returncode != 0:
        return []
    return [ln.split("\t")[-1] for ln in r.stdout.splitlines() if ln.strip()]


def _unindexed_new_notes(root: str) -> list[str]:
    """New docs/notes/ dated notes (just added) whose basename is absent from INDEX.md."""
    try:
        with open(os.path.join(root, "INDEX.md"), encoding="utf-8") as fh:
            index = fh.read()
    except OSError:
        return []
    out = []
    for p in _staged_added(root):
        if not (p.startswith("docs/notes/") and p.endswith(".md")):
            continue
        base = p.rsplit("/", 1)[-1]
        if base == "README.md":
            continue
        is_dated = bool(re.search(r"20\d\d-\d\d-\d\d", base)) or base.startswith("PLAN-")
        if is_dated and base not in index:
            out.append(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--audit-staged", action="store_true", help="scan staged additions (pre-commit)")
    g.add_argument("--audit-tree", action="store_true", help="scan the whole tracked tree (CI/DoD)")
    ap.add_argument("--root", default=".", help="repo root (default: cwd)")
    args = ap.parse_args()
    root = os.path.abspath(args.root)

    if args.audit_staged:
        if os.environ.get("ALLOW_ROOT_DOC") == "1":
            print("doc-placement: skipped (ALLOW_ROOT_DOC=1).")
            return 0
        names = _staged_paths(root)
        scope = "staged additions"
    else:
        names = _tracked_paths(root)
        scope = "tracked tree"

    if names is None:
        print("DOC_PLACEMENT (warn): git not available; check skipped.", file=sys.stderr)
        return 2

    bad = _violations(names)
    # INDEX-completeness (staged only): a new docs/notes/ dated note must be in INDEX.md.
    unindexed = _unindexed_new_notes(root) if args.audit_staged else []

    if not bad and not unindexed:
        print(f"doc-placement: clean ({scope}); root *.md restricted to the front-door set.")
        return 0

    if bad:
        print(
            f"DOC_PLACEMENT: {len(bad)} dated/research doc(s) placed at the repo root — "
            f"they belong under {NOTES_DIR}/ (reached via INDEX.md):",
            file=sys.stderr,
        )
        for b in bad:
            print(f"  {b}  ->  {NOTES_DIR}/{b}", file=sys.stderr)
        print(
            f"  fix: git mv <doc> {NOTES_DIR}/  &&  add an entry to INDEX.md.\n"
            f"  the root holds only entry-point/meta files "
            f"(README, START-HERE, INSTALL, INDEX, CONTRIBUTING, CLA, SECURITY, "
            f"PUBLIC-SCRUB-POLICY, AGENTS, CLAUDE).",
            file=sys.stderr,
        )
    if unindexed:
        print(
            f"DOC_PLACEMENT: {len(unindexed)} new docs/notes/ note(s) not listed in INDEX.md "
            f"(keep the curated hub complete):",
            file=sys.stderr,
        )
        for u in unindexed:
            print(f"  {u}  —  add a one-line entry to INDEX.md", file=sys.stderr)
    if args.audit_staged:
        print("  override once: ALLOW_ROOT_DOC=1 <git cmd>   (a genuine new root/meta doc, "
              "or a deliberately-unindexed note).", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
