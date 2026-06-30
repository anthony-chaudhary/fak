#!/usr/bin/env python3
"""Reciprocal index-sync gate: INDEX.md and llms.txt must not drift from the tree (#511).

The two curated maps — INDEX.md (the human index) and llms.txt (the answer-engine
doc map) — are hand-maintained, so they rot in two directions: a doc gets moved or
deleted and leaves a DANGLING link behind, or a new dated note lands and is never
listed (an ORPHAN). This gate checks the index<->tree relationship BOTH ways (the
"reciprocal" orphan gate):

  DANGLING (index -> tree): every local .md link in INDEX.md and llms.txt must
                            resolve to a real file. A dead link at the front door
                            is an embarrassment a visitor hits first.
  ORPHAN   (tree -> index): every dated note under docs/notes/ must be listed in
                            INDEX.md by basename. INDEX.md's stated contract is "if
                            a doc exists, it is reachable from here"; an unlisted
                            dated note breaks that contract.

llms.txt is a *curated* answer-engine map (not exhaustive), so the orphan direction
applies only to INDEX.md; both maps get the dangling direction. The llms-full.txt
companion has its own drift gate (tools/gen_llms_full.py --check).

Link resolution uses working-tree existence — the same convention as check_links.py
(the BROKEN_LINK gate) — so a link to an untracked-but-present file resolves (that
file is about to be committed by the session that added the link).

Modes:
  --audit-staged   gate when an index file or a docs/notes note is staged
                   (the pre-commit hook, sibling of DOC_PLACEMENT / BROKEN_LINK)
  --audit-tree     gate the whole tree (CI / DoD readiness)
Exit: 0 = clean, 1 = violation, 2 = could not run (the hook fails open on 2).
Escape (staged): ALLOW_INDEX_DRIFT=1.
"""
from __future__ import annotations
import argparse
import os
import re
import subprocess
from dispatch_worker import install_no_window_subprocess_defaults
install_no_window_subprocess_defaults(subprocess)
import sys

INDEX_FILES = ["INDEX.md", "llms.txt"]
NOTES_DIR = "docs/notes"
LINK_RE = re.compile(r"\]\(([^)]+)\)")


def _git(args, root):
    return subprocess.run(["git", "-C", root] + args, capture_output=True, text=True)


def _staged(root):
    r = _git(["diff", "--cached", "--name-only"], root)
    if r.returncode != 0:
        return None
    return set(r.stdout.split())


def _staged_added(root):
    r = _git(["diff", "--cached", "--name-only", "--diff-filter=A"], root)
    if r.returncode != 0:
        return []
    return [ln for ln in r.stdout.split() if ln.strip()]


def _resolves(root, idx_dir, ref):
    """True if a relative link target points at a real working-tree path."""
    return os.path.exists(os.path.normpath(os.path.join(root, idx_dir, ref)))


def _links(txt):
    """De-duplicated local .md link targets in a markdown body (order-preserving)."""
    seen = set()
    out = []
    for link in LINK_RE.findall(txt):
        if link.startswith(("http://", "https://", "mailto:", "#", "/")):
            continue
        path = link.split("#")[0].split("?")[0]
        if not path or not path.endswith(".md"):
            continue
        if path in seen:
            continue
        seen.add(path)
        out.append(path)
    return out


def dangling(root, idx_file):
    """(idx_file, link) pairs for .md links in idx_file that do not resolve."""
    fp = os.path.join(root, idx_file)
    try:
        with open(fp, encoding="utf-8") as fh:
            txt = fh.read()
    except OSError:
        return []
    idx_dir = os.path.dirname(idx_file)
    return [(idx_file, link) for link in _links(txt) if not _resolves(root, idx_dir, link)]


def _is_dated_note(rel):
    base = rel.rsplit("/", 1)[-1]
    return (base != "README.md" and rel.endswith(".md")
            and (bool(re.search(r"20\d\d-\d\d-\d\d", base)) or base.startswith("PLAN-")))


def orphans(root, new_notes=None):
    """Dated notes whose basename is not referenced anywhere in INDEX.md.

    new_notes=None  -> every tracked dated note (the --audit-tree / CI scope).
    new_notes=[...] -> only those notes (the --audit-staged scope: notes this commit
                       adds, so the gate blocks a commit that lands an unlisted note
                       without re-flagging pre-existing ones).
    """
    fp = os.path.join(root, "INDEX.md")
    try:
        with open(fp, encoding="utf-8") as fh:
            idx = fh.read()
    except OSError:
        return []
    if new_notes is None:
        r = _git(["ls-files", NOTES_DIR], root)
        if r.returncode != 0:
            return []
        notes = [n for n in r.stdout.split() if _is_dated_note(n)]
    else:
        notes = [n for n in new_notes if _is_dated_note(n)]
    return sorted(n for n in notes if n.rsplit("/", 1)[-1] not in idx)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--audit-staged", action="store_true",
                   help="gate staged index/note changes (pre-commit)")
    g.add_argument("--audit-tree", action="store_true",
                   help="gate the whole tree (CI / DoD)")
    ap.add_argument("--root", default=".", help="repo root (default: cwd)")
    args = ap.parse_args()
    root = os.path.abspath(args.root)

    if args.audit_staged and os.environ.get("ALLOW_INDEX_DRIFT") == "1":
        print("index-sync: skipped (ALLOW_INDEX_DRIFT=1).")
        return 0

    staged = None
    if args.audit_staged:
        staged = _staged(root)
        if staged is None:
            print("INDEX_SYNC (warn): git not available; check skipped.", file=sys.stderr)
            return 2
        # Only this commit can affect index drift when an index file or a note is
        # staged; otherwise skip so an unrelated commit is never wedged.
        relevant = any(s in INDEX_FILES or s.startswith(NOTES_DIR + "/") for s in staged)
        if not relevant:
            print("index-sync: clean (no index/note file staged).")
            return 0

    bad_dangling, bad_orphans = [], []

    # DANGLING: a staged edit to an index file can introduce a dead link (staged
    # mode); in tree mode both maps are always checked.
    for idx in INDEX_FILES:
        if args.audit_staged and idx not in staged:
            continue
        bad_dangling += dangling(root, idx)

    # ORPHAN: in tree mode, every dated note must be listed; in staged mode, only
    # notes this commit adds (don't re-flag a pre-existing unlisted note).
    if args.audit_staged:
        new_notes = [n for n in _staged_added(root) if n.startswith(NOTES_DIR + "/")]
        bad_orphans = orphans(root, new_notes=new_notes)
    else:
        bad_orphans = orphans(root)

    if not bad_dangling and not bad_orphans:
        print("index-sync: clean (no dangling links, no unlisted dated notes).")
        return 0

    if bad_dangling:
        print(f"INDEX_SYNC: {len(bad_dangling)} dangling .md link(s) in the curated index:",
              file=sys.stderr)
        for idx, link in bad_dangling:
            print(f"  {idx}: ]({link})  ->  missing file", file=sys.stderr)
        print("  fix: repoint or remove the link, or add the target file.", file=sys.stderr)
    if bad_orphans:
        print(f"INDEX_SYNC: {len(bad_orphans)} dated note(s) under {NOTES_DIR}/ not listed in INDEX.md:",
              file=sys.stderr)
        for o in bad_orphans:
            print(f"  {o}  —  add a one-line entry to INDEX.md", file=sys.stderr)
        print("  fix: add an entry to INDEX.md (its contract: every doc is reachable from here).",
              file=sys.stderr)
    if args.audit_staged:
        print("  override once: ALLOW_INDEX_DRIFT=1 <git cmd>.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
