#!/usr/bin/env python3
"""Broken-link gate: a front-door doc must not ship a dead relative link.

The parallel fleet deletes/moves docs and leaves dangling links behind; a public
visitor hits the front door first, so a dead link there is an embarrassment. This
resolves every relative markdown link in the front-door set and refuses any that
does not point at a real file or directory.

Scope (front door): README, START-HERE, INSTALL, INDEX, AGENTS, CLAUDE,
CONTRIBUTING, SECURITY, docs/index.md, docs/FAQ.md.

Modes:
  --audit-staged   check front-door files that are staged (added/modified) — the
                   gate: if you touch a front-door file, it must be link-clean.
  --audit-tree     check the whole front-door set (CI / DoD).
Exit: 0 clean, 1 dead link, 2 could-not-run (hook fails open on 2).
Escape (staged): ALLOW_BAD_LINK=1.
"""
from __future__ import annotations
import argparse
import os
import posixpath
import re
import subprocess
import sys

FRONT_DOOR = [
    "README.md", "START-HERE.md", "INSTALL.md", "INDEX.md", "AGENTS.md",
    "CLAUDE.md", "CONTRIBUTING.md", "SECURITY.md",
    "docs/index.md", "docs/FAQ.md",
]
LINK_RE = re.compile(r"\]\(([^)]+)\)")


def _git(args, root):
    return subprocess.run(["git", "-C", root] + args, capture_output=True, text=True)


def _staged_frontdoor(root):
    r = _git(["diff", "--cached", "--name-status", "--diff-filter=ACMR"], root)
    if r.returncode != 0:
        return None
    paths = {ln.split("\t")[-1] for ln in r.stdout.splitlines() if ln.strip()}
    return [f for f in FRONT_DOOR if f in paths]


def _dead_links(root, f):
    fp = os.path.join(root, f)
    try:
        with open(fp, encoding="utf-8") as fh:
            txt = fh.read()
    except OSError:
        return []  # file not present (e.g. deleted) — nothing to check
    d = posixpath.dirname(f)
    out = []
    for link in LINK_RE.findall(txt):
        if link.startswith(("http://", "https://", "mailto:", "#", "/", "data:")):
            continue
        path = link.split("#")[0].split("?")[0]
        if not path:
            continue
        tgt = posixpath.normpath(posixpath.join(d, path))
        if os.path.exists(os.path.join(root, tgt)):
            continue
        out.append((link, tgt))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--audit-staged", action="store_true")
    g.add_argument("--audit-tree", action="store_true")
    ap.add_argument("--root", default=".")
    a = ap.parse_args()
    root = os.path.abspath(a.root)

    if a.audit_staged and os.environ.get("ALLOW_BAD_LINK") == "1":
        print("broken-link: skipped (ALLOW_BAD_LINK=1).")
        return 0

    if a.audit_staged:
        files = _staged_frontdoor(root)
        if files is None:
            print("BROKEN_LINK (warn): git not available; check skipped.", file=sys.stderr)
            return 2
        scope = "staged front-door files"
    else:
        files = [f for f in FRONT_DOOR if os.path.exists(os.path.join(root, f))]
        scope = "front-door set"

    findings = []
    for f in files:
        for link, tgt in _dead_links(root, f):
            findings.append((f, link, tgt))
    if not findings:
        print(f"broken-link: clean ({scope}).")
        return 0

    print(f"BROKEN_LINK: {len(findings)} dead relative link(s) in front-door docs:", file=sys.stderr)
    for f, link, tgt in findings:
        print(f"  {f}: ]({link})  ->  missing {tgt}", file=sys.stderr)
    print("  fix: repoint to the real path, or remove the link (the target may have "
          "been deleted/moved).", file=sys.stderr)
    if a.audit_staged:
        print("  override once: ALLOW_BAD_LINK=1 <git cmd>.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
