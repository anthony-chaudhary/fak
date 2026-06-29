#!/usr/bin/env python3
"""Tests for the reciprocal index-sync gate (`check_index_sync.py`, #511).

Covers the link-parsing logic (`_links`), the DANGLING direction (`dangling`, for
both INDEX.md and llms.txt), the ORPHAN direction (`orphans`, in both --audit-tree
and --audit-staged scopes), the dated-note classifier (`_is_dated_note`), and the
CLI exit codes (0 clean / 1 violation / escape hatch) against a throwaway git repo.
Closes with a LIVE regression that the real repo's curated maps ship no drift.

Run: `python tools/check_index_sync_test.py`  (exit 0 = all pass),
or `python -m pytest tools/check_index_sync_test.py -q`.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_index_sync as ix  # noqa: E402

ROOT = str(Path(__file__).resolve().parent.parent)
PY = sys.executable


def _doc(root: Path, rel: str, body: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")


def _sh(args, cwd: str) -> None:
    subprocess.run(args, cwd=cwd, check=True, capture_output=True)


# --- link parsing ----------------------------------------------------------

def test_links_picks_only_local_md() -> None:
    txt = ("[a](a.md) [ext](https://x/y.md) [anchor](#top) [mail](mailto:x@y) "
           "[dir](docs/notes/) [pic](pic.png) [b](b.md) [b-again](b.md) [slash](/abs.md)")
    assert ix._links(txt) == ["a.md", "b.md"]


def test_is_dated_note() -> None:
    assert ix._is_dated_note("docs/notes/NOTE-2026-06-23.md")
    assert ix._is_dated_note("docs/notes/PLAN-foo.md")
    assert not ix._is_dated_note("docs/notes/README.md")
    assert not ix._is_dated_note("docs/notes/random.md")
    assert not ix._is_dated_note("docs/notes/pic.png")


# --- DANGLING direction (index -> tree) ------------------------------------

def test_dangling_detected_in_index() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _doc(root, "INDEX.md", "- [gone](gone.md) - [here](here.md)")
        _doc(root, "here.md", "x")
        assert ix.dangling(str(root), "INDEX.md") == [("INDEX.md", "gone.md")]


def test_dangling_detected_in_llms_txt() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _doc(root, "llms.txt", "- [x](a.md) and [dead](deep/dead.md)")
        _doc(root, "a.md", "x")
        assert ix.dangling(str(root), "llms.txt") == [("llms.txt", "deep/dead.md")]


def test_no_dangling_when_all_resolve() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _doc(root, "INDEX.md", "- [a](a.md) - [b](sub/b.md)")
        _doc(root, "a.md", "x")
        _doc(root, "sub/b.md", "y")
        assert ix.dangling(str(root), "INDEX.md") == []


# --- ORPHAN direction (tree -> INDEX.md) -----------------------------------

def _repo(td: str) -> Path:
    """A temp git repo so `git ls-files` / staged diff behave as in the wild."""
    root = Path(td)
    _sh(["git", "init", "-q"], td)
    _sh(["git", "config", "user.email", "t@t"], td)
    _sh(["git", "config", "user.name", "t"], td)
    return root


def test_orphan_tree_mode_flags_unlisted_dated_note() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = _repo(td)
        _doc(root, "INDEX.md", "# index\n(no notes listed)")
        _doc(root, "docs/notes/NOTE-2026-06-23.md", "x")
        _sh(["git", "-C", td, "add", "."], td)
        assert ix.orphans(td) == ["docs/notes/NOTE-2026-06-23.md"]


def test_orphan_tree_mode_clean_when_listed() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = _repo(td)
        _doc(root, "INDEX.md", "- [n](docs/notes/NOTE-2026-06-23.md)")
        _doc(root, "docs/notes/NOTE-2026-06-23.md", "x")
        _doc(root, "docs/notes/README.md", "not dated, ignored")
        _sh(["git", "-C", td, "add", "."], td)
        assert ix.orphans(td) == []


def test_orphan_staged_mode_only_flags_new_note() -> None:
    # A pre-existing unlisted note (already tracked) must NOT be re-flagged in
    # staged mode; only a note THIS commit adds is this commit's responsibility.
    with tempfile.TemporaryDirectory() as td:
        root = _repo(td)
        _doc(root, "INDEX.md", "# index")
        _doc(root, "docs/notes/OLD-2026-01-01.md", "already tracked, unlisted")
        _sh(["git", "-C", td, "add", "."], td)
        _sh(["git", "-C", td, "commit", "-q", "-m", "base"], td)
        # now stage a NEW unlisted note
        _doc(root, "docs/notes/NEW-2026-06-23.md", "new, unlisted")
        _sh(["git", "-C", td, "add", "docs/notes/NEW-2026-06-23.md"], td)
        new = ["docs/notes/NEW-2026-06-23.md"]
        assert ix.orphans(td, new_notes=new) == ["docs/notes/NEW-2026-06-23.md"]
        # and the OLD unlisted note is not in scope for staged mode
        assert "docs/notes/OLD-2026-01-01.md" not in ix.orphans(td, new_notes=new)


# --- CLI exit codes (subprocess, against a temp repo) ----------------------

def test_cli_tree_mode_clean_exit_0() -> None:
    with tempfile.TemporaryDirectory() as td:
        _repo(td)
        _doc(Path(td), "INDEX.md", "# index\n- [a](a.md)")
        _doc(Path(td), "llms.txt", "# llms\n- [a](a.md)")
        _doc(Path(td), "a.md", "x")
        _sh(["git", "-C", td, "add", "."], td)
        r = subprocess.run([PY, str(Path(ROOT) / "tools/check_index_sync.py"),
                            "--audit-tree", "--root", td], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr


def test_cli_tree_mode_dangling_exit_1() -> None:
    with tempfile.TemporaryDirectory() as td:
        _repo(td)
        _doc(Path(td), "INDEX.md", "- [dead](dead.md)")
        _doc(Path(td), "llms.txt", "# llms")
        _sh(["git", "-C", td, "add", "."], td)
        r = subprocess.run([PY, str(Path(ROOT) / "tools/check_index_sync.py"),
                            "--audit-tree", "--root", td], capture_output=True, text=True)
        assert r.returncode == 1
        assert "dead.md" in r.stderr


def test_cli_staged_mode_skips_unrelated_commit() -> None:
    # An unrelated staged file (.py) must not trip the gate (it can't cause drift).
    with tempfile.TemporaryDirectory() as td:
        _repo(td)
        _doc(Path(td), "INDEX.md", "# index")
        _doc(Path(td), "llms.txt", "# llms")
        _doc(Path(td), "tools/foo.py", "x = 1")
        _sh(["git", "-C", td, "add", "tools/foo.py"], td)
        r = subprocess.run([PY, str(Path(ROOT) / "tools/check_index_sync.py"),
                            "--audit-staged", "--root", td], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr


def test_cli_staged_mode_escape_hatch() -> None:
    with tempfile.TemporaryDirectory() as td:
        _repo(td)
        _doc(Path(td), "INDEX.md", "- [dead](dead.md)")
        _doc(Path(td), "llms.txt", "# llms")
        _sh(["git", "-C", td, "add", "INDEX.md"], td)
        env = {**__import__("os").environ, "ALLOW_INDEX_DRIFT": "1"}
        r = subprocess.run([PY, str(Path(ROOT) / "tools/check_index_sync.py"),
                            "--audit-staged", "--root", td], capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stderr


# --- LIVE regression: the real repo's curated maps are drift-free ----------

def test_live_repo_has_no_index_drift() -> None:
    """The real INDEX.md / llms.txt ship no dangling link, and every tracked dated
    note is listed. This is the gate that would have caught #511's drift class."""
    dang = ix.dangling(ROOT, "INDEX.md") + ix.dangling(ROOT, "llms.txt")
    assert not dang, "dangling links in the curated index:\n" + "\n".join(
        f"{i}:{line}" for i, line in dang)
    orph = ix.orphans(ROOT)
    assert not orph, "unlisted dated notes:\n" + "\n".join(orph)


def _run() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"ok   {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run())
