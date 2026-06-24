#!/usr/bin/env python3
"""Tests for the node-local <-> repo memory mirror (tools/sync_memory.py).

The load-bearing invariant is store RESOLUTION: the mirror must follow the SAME
store the running harness writes to. The fleet runs every agent under its own
relocated config home via ``CLAUDE_CONFIG_DIR`` (``~/.claude-gem5-netra``, …), so
the old hardcoded ``~/.claude`` resolved to the WRONG (stale/foreign) or an empty
store on every such node — the regression this locks. Plus the small pure file
helpers (md_files / differ / copy_dir / status), all stdlib + tmpdir, no harness.

Run: `python tools/sync_memory_test.py`  (exit 0 = all pass).
"""
from __future__ import annotations

import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sync_memory as sm  # noqa: E402


@contextmanager
def env(**overrides: str | None):
    """Set/clear env vars for the block, restore exactly afterward."""
    saved = {k: os.environ.get(k) for k in overrides}
    try:
        for k, v in overrides.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def main() -> int:
    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if not cond and detail else ""))
        if not cond:
            failures.append(name)

    root = Path("C:/work/fak") if os.name == "nt" else Path("/work/fak")
    slug = "C--work-fak" if os.name == "nt" else "-work-fak"

    # 1) CLAUDE_MEMORY_DIR wins outright over everything else.
    with env(CLAUDE_MEMORY_DIR="/explicit/mem", CLAUDE_CONFIG_DIR="/some/config"):
        got = sm.default_home_memory(root)
        check("CLAUDE_MEMORY_DIR wins outright", got == Path("/explicit/mem"), str(got))

    # 2) CLAUDE_CONFIG_DIR (relocated config home) is honored — the bug fix. The
    #    store hangs under the RELOCATED home, not the vanilla ~/.claude.
    with env(CLAUDE_MEMORY_DIR=None, CLAUDE_CONFIG_DIR="/home/u/.claude-gem5-netra"):
        got = sm.default_home_memory(root)
        want = Path("/home/u/.claude-gem5-netra") / "projects" / slug / "memory"
        check("CLAUDE_CONFIG_DIR honored", got == want, str(got))
        check("relocated home is NOT vanilla ~/.claude", ".claude/projects" not in got.as_posix(),
              got.as_posix())

    # 3) neither set -> vanilla ~/.claude default (back-compat).
    with env(CLAUDE_MEMORY_DIR=None, CLAUDE_CONFIG_DIR=None):
        got = sm.default_home_memory(root)
        want = Path.home() / ".claude" / "projects" / slug / "memory"
        check("vanilla default falls back to ~/.claude", got == want, str(got))

    # 4) slug derivation: every non-alphanumeric in the repo path becomes '-'.
    with env(CLAUDE_MEMORY_DIR=None, CLAUDE_CONFIG_DIR="/c"):
        got = sm.default_home_memory(Path("/a.b/c-d"))
        check("slug maps non-alnum to '-'", got == Path("/c/projects/-a-b-c-d/memory"), str(got))

    # ---- pure file helpers over a tmp tree ----
    with tempfile.TemporaryDirectory() as td:
        home = Path(td) / "home"
        repo = Path(td) / "repo"
        home.mkdir()
        (home / "alpha.md").write_text("one", encoding="utf-8")
        (home / "beta.md").write_text("two", encoding="utf-8")
        (home / "README.md").write_text("doc", encoding="utf-8")  # excluded
        (home / "notes.txt").write_text("nope", encoding="utf-8")  # non-md, ignored

        # 5) md_files: only *.md, README.md excluded.
        names = set(sm.md_files(home))
        check("md_files keeps only *.md, drops README", names == {"alpha.md", "beta.md"}, str(names))
        check("md_files of missing dir is empty", sm.md_files(repo) == {}, str(sm.md_files(repo)))

        # 6) differ: missing dest differs; identical does not; changed does.
        check("differ true when dest missing", sm.differ(home / "alpha.md", repo / "alpha.md") is True)

        # 7) copy_dir: copies the two facts, never README, creates the dest.
        rc = sm.copy_dir(home, repo, prune=False)
        check("copy_dir returns 0", rc == 0, str(rc))
        copied = set(sm.md_files(repo))
        check("copy_dir mirrors facts only", copied == {"alpha.md", "beta.md"}, str(copied))
        check("copy_dir never mirrors README", not (repo / "README.md").exists())

        # 8) idempotent: second copy detects identical content, differ() is False.
        check("differ false after copy", sm.differ(home / "alpha.md", repo / "alpha.md") is False)

        # 9) prune removes a dest *.md absent from source; keeps present ones.
        (repo / "orphan.md").write_text("stale", encoding="utf-8")
        sm.copy_dir(home, repo, prune=True)
        after = set(sm.md_files(repo))
        check("prune drops orphan", "orphan.md" not in after, str(after))
        check("prune keeps real facts", {"alpha.md", "beta.md"} <= after, str(after))

        # 10) copy_dir from an empty source returns 1 (nothing to copy).
        rc = sm.copy_dir(repo / "empty", repo / "dest", prune=False)
        check("copy_dir from empty source returns 1", rc == 1, str(rc))

        # 11) status is read-only and returns 0.
        check("status returns 0", sm.status(home, repo) == 0)

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        return 1
    print("sync_memory_test: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
