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
  * BY-MACHINE private-by-default (STAGED-ONLY) — a NEW addition under
    experiments/benchmark/runs/by-machine/ (with or without the fak/ prefix) is a
    raw per-machine benchmark run drop: regenerable harness output that routinely
    carries infra tells (cloud instance names/zones, credential paths, VM hostnames,
    accelerator SKUs, GPU topology). It is PRIVATE-BY-DEFAULT and refused at
    commit-time only. This does NOT run in --audit-tree: the ~50 grandfathered
    evidence artifacts already tracked under by-machine/ (gitignore is inert for
    tracked paths) must keep CI green. To PUBLISH a scrubbed run artifact, promote
    it deliberately with ALLOW_STRAY_FILE=1.
  * Large file — anything larger than MAX_BYTES (default 25 MiB) — refused (use
    Git LFS, trim it, or override).

Modes: --audit-staged (pre-commit, staged additions) | --audit-tree (CI/DoD).
Exit: 0 clean, 1 violation, 2 could-not-run (hook fails open on 2).
Escape (staged): ALLOW_STRAY_FILE=1 (skips ALL staged checks, incl. by-machine).
Override the size cap: --max-bytes N.
"""
from __future__ import annotations
import argparse
import os
import re
import subprocess
from dispatch_worker import install_no_window_subprocess_defaults
import sys
install_no_window_subprocess_defaults(subprocess)

# 25 MiB: comfortably admits legitimate committed assets (the 1440p hero-video.mp4 at
# ~9.4 MiB the live Pages hero + README embed by raw URL, a model card, a fixture corpus, a
# demo capture) that a 10 MiB cap was false-blocking, while still refusing genuinely
# oversized blobs from a forever-history public tree. Kept in lockstep with the Go twin
# (internal/hooks/gate_fileadmission.go fileAdmissionMaxBytes) so the pre-commit Go gate and
# this CI checker never disagree; --max-bytes still overrides per-invocation.
DEFAULT_MAX_BYTES = 25 * 1024 * 1024

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
    "fak/demorace-err.log",  # cited as evidence in docs/benchmarks/FINAL-ANALYSIS.md
}

# One-run worker fuel is control-plane residue, not a reusable project prompt. Keep this
# fallback in parity with internal/hooks/gate_fileadmission.go for fresh clones without fak.
GENERATED_CLAUDE_PROMPT = re.compile(
    r"(?i)(?:^|/)(?:frontdoor-\d+-recovery|resfleet-\d+|resolve-issue-\d+-continuation)\.md$"
)

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

# By-machine private-by-default: raw per-machine benchmark run drops live under
# experiments/benchmark/runs/by-machine/<machine>/<UTC>-<label>/ (or the fak/-nested
# form in the private superrepo). They are regenerable harness output and routinely
# carry infrastructure tells — cloud instance names + zones, credential file paths,
# VM hostnames + accelerator SKUs, private multi-GPU topology — so commit 62bed967e
# widened .gitignore to make the WHOLE tree private-by-default. This is the commit-time
# backstop that ignore rule cannot provide: a `git add -f` of a new raw drop, or a
# silent revert of the ignore line, would re-clutter the public tree (this exact
# regression class re-added 192 nightrun files once). It is STAGED-ONLY by design —
# _is_private_bymachine_addition is consulted ONLY on the staged-additions branch of
# main(), NEVER in _classify — because ~50 grandfathered evidence artifacts are already
# legitimately tracked under by-machine/ (gitignore is inert for tracked paths) and the
# --audit-tree gate MUST stay green over them. Deliberate promotion of a scrubbed
# artifact still works via the ALLOW_STRAY_FILE=1 escape (which skips all staged checks).
_PRIVATE_BYMACHINE = re.compile(r"^(fak/)?experiments/benchmark/runs/by-machine/")


def _is_private_bymachine_addition(path: str) -> bool:
    """True iff a NEWLY-ADDED path is a raw per-machine benchmark run drop that is
    private-by-default. Applied only on the staged-additions branch (--audit-staged);
    it is deliberately NOT part of _classify, so --audit-tree never fires it on the
    grandfathered evidence files still legitimately tracked under by-machine/."""
    return bool(_PRIVATE_BYMACHINE.match(path))

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
    if path.lower().startswith(".claude/goal-prompts/") and GENERATED_CLAUDE_PROMPT.search(path):
        return (
            "generated one-run .claude goal prompt; park under ignored scratch/private "
            "storage or delete after the run"
        )
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

    # A non-positive --max-bytes falls back to the default, matching the Go twin's
    # gateEnvInt (FAK_MAX_FILE_BYTES) which treats <=0/garbage as "unset" rather than a
    # 0-byte cap that would refuse every file. Keeps the two enforcers in lockstep on
    # degenerate input; ALLOW_STRAY_FILE=1 is the way to actually disable the check.
    if a.max_bytes <= 0:
        a.max_bytes = DEFAULT_MAX_BYTES

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
        # STAGED-ONLY: a new raw by-machine run drop that _classify admits (it is under
        # the experiments/ data dir, so soft-junk rules don't apply) is still refused at
        # commit-time. Guarded by a.audit_staged so --audit-tree never fires it on the
        # grandfathered evidence files. Fallback (only when _classify is silent) so a
        # more-specific junk/secret reason still wins the message.
        if reason is None and a.audit_staged and _is_private_bymachine_addition(n):
            reason = ("raw benchmark run drop under experiments/benchmark/runs/by-machine/ "
                      "is PRIVATE-BY-DEFAULT (regenerable harness output with infra tells) — "
                      "promote a scrubbed artifact deliberately with ALLOW_STRAY_FILE=1, or "
                      "keep it gitignored")
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
