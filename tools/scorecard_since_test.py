#!/usr/bin/env python3
"""Unit tests for tools/scorecard_since.py (the Python shift-left skip-gate).

Run: python3 tools/scorecard_since_test.py  (exits non-zero on any failure).
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scorecard_since as ss  # noqa: E402

FAILS: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILS.append(msg)


def test_match_glob() -> None:
    cases = [
        ("internal/uiquality/", "internal/uiquality/x.go", True),
        ("internal/uiquality/", "internal/uiquality", True),
        ("internal/uiquality/", "internal/uiqualityscore.go", False),
        ("tools/*.py", "tools/docs_scorecard.py", True),
        ("tools/*.py", "tools/sub/x.py", False),
        ("**/*.go", "internal/x/y.go", True),
        ("**/*.go", "main.go", True),
        ("docs/**/*.md", "docs/a/b/c.md", True),
        ("docs/**/*.md", "docs/x.md", True),
        ("docs/**/*.md", "notdocs/x.md", False),
        ("**/AGENTS.md", "AGENTS.md", True),
        ("**/AGENTS.md", "a/b/AGENTS.md", True),
        ("CLAUDE.md", "CLAUDE.md", True),
        ("CLAUDE.md", "AGENTS.md", False),
        ("internal\\uiquality\\", "internal/uiquality/x.go", True),
        ("", "anything", False),
    ]
    for g, p, want in cases:
        got = ss.match_glob(g, p)
        check(got == want, f"match_glob({g!r}, {p!r}) = {got}, want {want}")


def test_touched() -> None:
    changed = ["a/x.go", "README.md", "docs/y.md"]
    got = ss.touched(changed, ["**/*.go", "docs/**/*.md"])
    check(got == ["a/x.go", "docs/y.md"], f"touched = {got}")


def _git(dir_: str, *args: str) -> None:
    env = dict(os.environ,
               GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t",
               GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
    proc = subprocess.run(["git", "-C", dir_, *args],
                          capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"git {args}: {proc.stderr}")


def _write(dir_: str, rel: str, body: str) -> None:
    p = Path(dir_) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body)


def test_changed_paths_and_skip() -> None:
    if not _have_git():
        print("scorecard_since_test: git unavailable, skipping git-backed cases")
        return
    with tempfile.TemporaryDirectory() as d:
        _write(d, "keep.go", "package a\n")
        _write(d, "edit.go", "package a\n")
        _git(d, "init", "-q")
        _git(d, "add", "-A")
        _git(d, "commit", "-qm", "base")

        # One tracked edit + one untracked new file; keep.go unchanged.
        _write(d, "edit.go", "package a\n// x\n")
        _write(d, "new/added.go", "package b\n")

        changed = ss.changed_paths(d, "HEAD")
        check(changed == ["edit.go", "new/added.go"], f"changed_paths = {changed}")

        # Blank since → [].
        check(ss.changed_paths(d, "  ") == [], "blank since should be []")

        # Bad ref raises.
        try:
            ss.changed_paths(d, "nope-not-a-ref")
            check(False, "bad ref should raise")
        except RuntimeError:
            pass

        # since_skip: corpus (*.go) WAS touched → None (full scan), stderr note.
        out, err = io.StringIO(), io.StringIO()
        rc = ss.since_skip("t", d, "HEAD", ["**/*.go"], out=out, err=err)
        check(rc is None, f"touched corpus should full-scan (None), got {rc}")
        check("changed since HEAD" in err.getvalue(), f"expected scan note, got {err.getvalue()!r}")

        # since_skip: corpus (*.md) NOT touched → 0 (skip), stdout note.
        out, err = io.StringIO(), io.StringIO()
        rc = ss.since_skip("t", d, "HEAD", ["**/*.md"], out=out, err=err)
        check(rc == 0, f"untouched corpus should skip (0), got {rc}")
        check("unchanged since HEAD" in out.getvalue(), f"expected skip note, got {out.getvalue()!r}")

        # since_skip: bad ref → None (full scan), never a false "unchanged".
        out, err = io.StringIO(), io.StringIO()
        rc = ss.since_skip("t", d, "nope-not-a-ref", ["**/*.go"], out=out, err=err)
        check(rc is None, f"bad ref should full-scan (None), got {rc}")
        check("diff failed" in err.getvalue(), f"expected diff-failed note, got {err.getvalue()!r}")


def _have_git() -> bool:
    try:
        subprocess.run(["git", "--version"], capture_output=True)
        return True
    except OSError:
        return False


def main() -> int:
    test_match_glob()
    test_touched()
    test_changed_paths_and_skip()
    if FAILS:
        print("scorecard_since_test: FAIL")
        for f in FAILS:
            print("  -", f)
        return 1
    print("scorecard_since_test OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
