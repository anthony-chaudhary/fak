#!/usr/bin/env python3
"""File-admission gate: keep build artifacts, junk, and oversized blobs out of the tree.

A public repo's history is forever (especially after a squash). This refuses to
commit regenerable build/runtime junk (caches, compiled binaries, logs, demo
outputs, editor scratch) and any oversized blob, so the tree and history stay lean.

Rules:
  * SECRET — a cloud service-account key (*.sa.json), a -sa-key/-gcp-key JSON, or
    anything under a secrets/ dir — ALWAYS refused (path-based, fail-closed). A
    private key must never enter a forever-history public tree; rotate it and keep
    it in a secret store / gitignored dir.
  * PRIVATE-only — the operator's private lab GPU-server *connection* subsystem
    (the private control bridge client + its lab orchestrator) — ALWAYS refused in
    the PUBLIC tree; it lives only in the private canonical repo. This includes
    the sunset Python bench_slack bridge path; do not resurrect it here.
  * HARD junk  — caches/compiled/OS-cruft/editor-scratch — ALWAYS refused.
  * SOFT junk  — *.log / *.tmp / report.json / agent-report.json — refused UNLESS
    under a data dir (fak/experiments/, fak/testdata/) where such files are real
    committed evidence, or on the small kept-exception allowlist.
  * Large file — anything larger than MAX_BYTES (default 10 MiB) — refused (use
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
from dispatch_worker import install_no_window_subprocess_defaults
import sys
install_no_window_subprocess_defaults(subprocess)

# 10 MiB: large enough to admit the 1440p hero-video.mp4 (~9.4 MiB), which the live
# Pages hero + README embed by raw URL (so it cannot be dropped), while still refusing
# genuinely oversized blobs from a forever-history public tree.
DEFAULT_MAX_BYTES = 10 * 1024 * 1024

# Always-junk: never legitimately committed.
HARD_JUNK = [
    re.compile(r"(^|/)__pycache__/"),
    re.compile(r"(^|/)\.pytest_cache/"),
    re.compile(r"(^|/)\.ruff_cache/"),
    re.compile(r"(^|/)node_modules/"),
    re.compile(r"\.(pyc|pyo|class|o|a|obj)$"),
    re.compile(r"\.(exe|dll|so|dylib)$"),
    re.compile(r"^coverage$"),
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
# The Go module + experiments/ are at the repo ROOT now (post-hoist), so the data
# dirs where soft-junk names (report.json, *.log) are real artifacts live at root.
EXEMPT_DATA_DIRS = ("experiments/", "testdata/", "internal/", "fak/experiments/", "fak/testdata/")
# Specific tracked files that trip a soft rule but are intentionally kept.
KEEP_EXCEPTIONS = {
    "fak/demorace-err.log",  # cited as evidence in docs/benchmarking/FINAL-ANALYSIS.md
}

# Private-only: paths that must NEVER be tracked in the PUBLIC tree. The operator's
# lab GPU-server *connection* code — the private control bridge client and its bench
# orchestrator — speaks a private lab protocol and lives ONLY in the private
# canonical repo (PUBLIC-SCRUB-POLICY.md PRIVATE-ONLY list). Under the hard-cut
# model the public tree is edited directly, so the export-time scrubber's
# DELETE_PATHS never run as a public gate, and connection code using placeholder
# ids sails past the secret-needle scan — which is exactly how internal/dgxbridge +
# cmd/dgxbridge leaked once. This is the public-tree enforcement of the same
# move-to-private intent, keyed on the FLATTENED public path (no fak/ prefix): any
# cmd//internal/ package carrying the `dgx` token (so a NEW dedicated connection
# tool, e.g. cmd/dgxconn, is covered without an edit here) plus the named Slack-
# housekeeping sibling. The historical public Python bridge, tools/bench_slack.py,
# is also refused after its sunset: restoring it would be both a new Python tool
# under the ratchet and private control-plane code in the public tree. A match is
# ALWAYS refused, at commit-time and in CI: move it to the private repo. (Scope is
# the CONNECTION subsystem; the lab automation under tools/*dgx* and the dgx
# result dirs are a separate, larger relocation.)
#
# Sibling gate: tools/repo_guard.py is the WRITE-TIME filesystem boundary
# (OUT_OF_TREE_WRITE) and is content-blind — it judges only WHERE a path resolves; THIS
# gate is the COMMIT-TIME content-placement boundary that judges WHAT a path is. They are
# complementary halves of the public/private model (fak public, fak-private private).
PRIVATE_ONLY = [
    (re.compile(r"^(cmd|internal)/[^/]*dgx[^/]*/"),
     "private lab GPU-server connection subsystem — belongs in the private repo, not the public tree"),
    (re.compile(r"^(cmd|internal)/(?=[^/]*slack)(?=[^/]*(bridge|control|gc))[^/]*/"),
     "private lab private control bridge subsystem — belongs in the private repo, not the public tree"),
    (re.compile(r"^tools/bench_slack(_test)?\.py$"),
     "sunset Python Slack/DGX bridge — belongs in the private repo, not the public tree"),
]

# Secret files: credentials / private keys must NEVER enter a forever-history
# public tree. Path-based and fail-closed (it fires even when the bytes are
# unreadable), so it catches a key the content gates (check_secret_shapes.py, the
# scrub leak-gate) would miss by filename. Mirrors the *.sa.json / secrets/
# .gitignore globs and the private repo's convention — tools/create_gcp_admin_sa.sh
# writes secrets/gcp/<sa>.sa.json. A match is ALWAYS refused: rotate the key and
# keep it in a secret store or a gitignored dir, never in git.
SECRET_FILES = [
    (re.compile(r"(^|/)secrets/"),
     "secrets dir — credentials never belong in git; keep them gitignored / in a secret store"),
    (re.compile(r"\.sa\.json$"),
     "GCP service-account key (*.sa.json) — never commit a key; rotate it and keep it gitignored"),
    (re.compile(r"-(sa|gcp)-key\.json$"),
     "cloud service-account key — never commit a key; rotate it and keep it gitignored"),
]


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
    # Secrets first: a credential / key file is refused regardless of any other rule.
    for rx, why in SECRET_FILES:
        if rx.search(path):
            return why
    # Privacy next: a private-only path is refused regardless of size/junk rules.
    for rx, why in PRIVATE_ONLY:
        if rx.search(path):
            return why
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
