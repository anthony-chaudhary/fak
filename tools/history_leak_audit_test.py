#!/usr/bin/env python3
"""Tests for tools/history_leak_audit.py.

Builds a throwaway git repo with a gitignored sidecar carrying a FAKE needle, commits
content that contains it, and asserts the history sweep flags HISTORY-DIRTY (exit 1) —
then a clean repo (exit 0) and the SELF_REFERENTIAL exemption. Pure stdlib.

Run: `python tools/history_leak_audit_test.py`  (exit 0 = all pass).
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("history_leak_audit",
                                              os.path.join(HERE, "history_leak_audit.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# A fake needle that is NOT a real secret (so this test file never trips a real gate).
NEEDLE = "ZZ-FAKE-OPERATOR-NEEDLE-" + "47q"

failures: list[str] = []


def check(label, cond, detail=""):
    print(f"  [{'ok  ' if cond else 'FAIL'}] {label}")
    if not cond:
        failures.append(label + (f" :: {detail}" if detail else ""))


def _git(repo, *args):
    env = dict(os.environ)
    env.update(GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t",
               GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t",
               GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull)
    return subprocess.run(["git", "-C", repo, *args], capture_output=True,
                          text=True, encoding="utf-8", errors="replace", env=env)


def _sidecar(root: Path):
    p = root / "tools" / "_registry" / "scrub_needles.private.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('{"schema":"fleet-scrub-needles/1","audit_needles":["%s"],"export_audit_needles":[]}' % NEEDLE,
                 encoding="utf-8")


def _repo(tmp, files: dict):
    root = Path(tmp)
    _git(str(root), "init", "-b", "main")
    # The sidecar holds REAL needles and is gitignored in the real repo, so it is
    # never a committed blob. Mirror that here (a COMMITTED sidecar would correctly
    # be flagged as a real leak — the tool must not exempt it).
    (root / ".gitignore").write_text("tools/_registry/\n", encoding="utf-8")
    _sidecar(root)
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    _git(str(root), "add", "-A")
    _git(str(root), "commit", "-q", "-m", "fixture")
    return root


def test_dirty():
    print("dirty history (needle in a committed doc):")
    with tempfile.TemporaryDirectory() as tmp:
        root = _repo(tmp, {"notes.md": f"the value is {NEEDLE} here\n"})
        rc = mod.audit_history(str(root), ref="main")
        check("exit 1 on a history hit", rc == 1)


def test_clean():
    print("clean history (no needle):")
    with tempfile.TemporaryDirectory() as tmp:
        root = _repo(tmp, {"notes.md": "nothing sensitive here\n"})
        rc = mod.audit_history(str(root), ref="main")
        check("exit 0 when clean", rc == 0)


def test_self_referential():
    print("self-referential exemption (needle only in the policy file):")
    with tempfile.TemporaryDirectory() as tmp:
        # PUBLIC-SCRUB-POLICY.md is SELF_REFERENTIAL: it may name needles.
        root = _repo(tmp, {"PUBLIC-SCRUB-POLICY.md": f"denylist includes {NEEDLE}\n"})
        rc = mod.audit_history(str(root), ref="main")
        check("exit 0 — self-referential file is exempt", rc == 0)


def test_degraded():
    print("degraded mode (no sidecar -> literal tier skipped, no false clean claim):")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _git(str(root), "init", "-b", "main")
        (root / "notes.md").write_text(f"value {NEEDLE}\n", encoding="utf-8")
        _git(str(root), "add", "-A")
        _git(str(root), "commit", "-q", "-m", "fixture")
        # no sidecar -> real needles unavailable -> literal tier skipped -> exit 0,
        # but the run is DEGRADED (shape-only), which is reported honestly.
        rc = mod.audit_history(str(root), ref="main")
        check("exit 0 without sidecar (shape-only); needle not literal-matched", rc == 0)


def main():
    test_dirty()
    test_clean()
    test_self_referential()
    test_degraded()
    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        return 1
    print("history_leak_audit_test: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
