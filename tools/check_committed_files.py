#!/usr/bin/env python3
"""File-admission gate: keep build artifacts, junk, and oversized blobs out of the tree.

A public repo's history is forever (especially after a squash). This refuses to
commit regenerable build/runtime junk (caches, compiled binaries, logs, demo
outputs, editor scratch) and any oversized blob, so the tree and history stay lean.

Rules:
  * HARD junk  — caches/compiled/OS-cruft/editor-scratch — ALWAYS refused.
  * SOFT junk  — *.log / *.tmp / report.json / agent-report.json — refused UNLESS
    under a data dir (fak/experiments/, fak/testdata/) where such files are real
    committed evidence, or on the small kept-exception allowlist.
  * Large file — anything larger than MAX_BYTES (default 5 MiB) — refused (use
    Git LFS, trim it, or override).

Modes: --audit-staged (pre-commit, staged additions) | --audit-tree (CI/DoD).
Exit: 0 clean, 1 violation, 2 could-not-run (hook fails open on 2).
Escape (staged): ALLOW_STRAY_FILE=1.  Override the size cap: --max-bytes N.
"""
from __future__ import annotations
import argparse
import os
import re
import subprocess
import sys

DEFAULT_MAX_BYTES = 5 * 1024 * 1024

# Always-junk: never legitimately committed.
HARD_JUNK = [
    re.compile(r"(^|/)__pycache__/"),
    re.compile(r"(^|/)\.pytest_cache/"),
    re.compile(r"(^|/)\.ruff_cache/"),
    re.compile(r"(^|/)node_modules/"),
    re.compile(r"\.(pyc|pyo|class|o|a|obj)$"),
    re.compile(r"\.(exe|dll|so|dylib)$"),
    re.compile(r"(^|/)coverage\.out$"),
    re.compile(r"\.coverprofile$"),
    re.compile(r"(^|/)\.DS_Store$"),
    re.compile(r"(^|/)Thumbs\.db$"),
    re.compile(r"\.(swp|swo)$"),
    re.compile(r"~$"),
]
# Junk unless under a data dir or explicitly kept.
SOFT_JUNK = [
    re.compile(r"\.log$"),
    re.compile(r"\.tmp$"),
    re.compile(r"(^|/)(report|agent-report)\.json$"),
]
EXEMPT_DATA_DIRS = ("fak/experiments/", "fak/testdata/")
# Specific tracked files that trip a soft rule but are intentionally kept.
KEEP_EXCEPTIONS = {
    "fak/demorace-err.log",  # cited as evidence in docs/benchmarking/FINAL-ANALYSIS.md
}


def _git(args, root):
    return subprocess.run(["git", "-C", root] + args, capture_output=True, text=True)


def _staged_paths(root):
    r = _git(["diff", "--cached", "--name-status", "--diff-filter=AR"], root)
    if r.returncode != 0:
        return None
    return [ln.split("\t")[-1] for ln in r.stdout.splitlines() if ln.strip()]


def _tracked(root):
    r = _git(["ls-files"], root)
    return r.stdout.split() if r.returncode == 0 else None


def _classify(path, root, max_bytes):
    """Return a violation reason string, or None if the path is allowed."""
    if path in KEEP_EXCEPTIONS:
        size = _size(root, path)
        return None if size is None or size <= max_bytes else f"large file ({size//1024} KiB > {max_bytes//1024} KiB)"
    for rx in HARD_JUNK:
        if rx.search(path):
            return "build artifact / cache / compiled output"
    if not path.startswith(EXEMPT_DATA_DIRS):
        for rx in SOFT_JUNK:
            if rx.search(path):
                return "log / temp / demo-output (regenerable)"
    size = _size(root, path)
    if size is not None and size > max_bytes:
        return f"oversized blob ({size // 1024} KiB > {max_bytes // 1024} KiB)"
    return None


def _size(root, path):
    fp = os.path.join(root, path)
    try:
        return os.path.getsize(fp)
    except OSError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--audit-staged", action="store_true")
    g.add_argument("--audit-tree", action="store_true")
    ap.add_argument("--root", default=".")
    ap.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    a = ap.parse_args()
    root = os.path.abspath(a.root)

    if a.audit_staged and os.environ.get("ALLOW_STRAY_FILE") == "1":
        print("file-admission: skipped (ALLOW_STRAY_FILE=1).")
        return 0

    names = _staged_paths(root) if a.audit_staged else _tracked(root)
    scope = "staged additions" if a.audit_staged else "tracked tree"
    if names is None:
        print("FILE_ADMISSION (warn): git not available; check skipped.", file=sys.stderr)
        return 2

    bad = []
    for n in sorted(set(names)):
        reason = _classify(n, root, a.max_bytes)
        if reason:
            bad.append((n, reason))
    if not bad:
        print(f"file-admission: clean ({scope}).")
        return 0

    print(f"FILE_ADMISSION: {len(bad)} file(s) that should not be committed:", file=sys.stderr)
    for n, why in bad:
        print(f"  {n}  —  {why}", file=sys.stderr)
    print("  fix: drop it (it is regenerable), gitignore it, or move data under "
          "fak/experiments|testdata/. Oversized blobs: use Git LFS or trim.", file=sys.stderr)
    if a.audit_staged:
        print("  override once: ALLOW_STRAY_FILE=1 <git cmd>.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
