#!/usr/bin/env python3
"""Tests for tools/dos_sync.py — the .dos work-product backup tool.

Pure stdlib, no git, no network. Builds a synthetic .dos/ tree in a tempdir and
exercises the safety-critical behaviors the audit required:
  * REFUSES a source dir not named `.dos` (cannot sweep a repo root).
  * selects only depth-1 *.md; ignores metrics/ + streams/ runtime subtrees.
  * push copies work-product; pull refuses to overwrite newer-local without --force.
"""
from __future__ import annotations

import os
import tempfile
import time

import dos_sync as mod

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = ""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [ok  ] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


def _write(path: str, text: str, mtime: float | None = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    if mtime is not None:
        os.utime(path, (mtime, mtime))


def make_dos(tmp: str) -> str:
    """A synthetic .dos/ with two work-product .md + runtime noise that must be skipped."""
    dos = os.path.join(tmp, ".dos")
    _write(os.path.join(dos, "awesome-list-prs-DRAFT.md"), "# draft\n")
    _write(os.path.join(dos, "release-notes-v0.30.0.md"), "# notes\n")
    # runtime that MUST be excluded:
    _write(os.path.join(dos, "metrics", "observations.jsonl"), '{"op":"OBSERVE"}\n')
    _write(os.path.join(dos, "streams", "abc.jsonl"), '{"op":"STEP"}\n')
    # a non-md depth-1 file that must be ignored:
    _write(os.path.join(dos, "project.json"), "{}\n")
    return dos


def test_refuses_non_dos_root():
    print("refuses a source that isn't named .dos:")
    with tempfile.TemporaryDirectory() as tmp:
        # point at the repo root itself, not .dos
        got = mod.resolve_dos_dir(tmp)
        check("resolve_dos_dir(repo_root) -> None (refused)", got is None)
        dos = make_dos(tmp)
        check("resolve_dos_dir(.dos) -> path", mod.resolve_dos_dir(dos) == os.path.abspath(dos))


def test_selects_only_depth1_md():
    print("selects only depth-1 *.md, excludes runtime + non-md:")
    with tempfile.TemporaryDirectory() as tmp:
        dos = make_dos(tmp)
        files = mod.durable_md(dos)
        check("exactly the two work-product .md", files == ["awesome-list-prs-DRAFT.md", "release-notes-v0.30.0.md"],
              f"got {files}")
        check("metrics/ excluded", not any("observations" in f for f in files))
        check("streams/ excluded", not any("abc" in f for f in files))
        check("project.json (non-md) excluded", "project.json" not in files)


def test_push_then_pull_roundtrip():
    print("push copies work-product; pull restores:")
    with tempfile.TemporaryDirectory() as tmp:
        dos = make_dos(tmp)
        archive = os.path.join(tmp, "archive")
        rc = mod.cmd_push(dos, archive, dry=False)
        check("push rc==0", rc == 0)
        check("archive has both .md",
              os.path.exists(os.path.join(archive, "awesome-list-prs-DRAFT.md"))
              and os.path.exists(os.path.join(archive, "release-notes-v0.30.0.md")))
        check("archive did NOT get metrics/streams",
              not os.path.exists(os.path.join(archive, "metrics"))
              and not os.path.exists(os.path.join(archive, "streams")))
        # wipe a local file, pull it back
        os.remove(os.path.join(dos, "release-notes-v0.30.0.md"))
        rc = mod.cmd_pull(dos, archive, force=False, dry=False)
        check("pull rc==0", rc == 0)
        check("local restored", os.path.exists(os.path.join(dos, "release-notes-v0.30.0.md")))


def test_pull_refuses_newer_local():
    print("pull refuses to clobber a newer-local file unless --force:")
    with tempfile.TemporaryDirectory() as tmp:
        dos = make_dos(tmp)
        archive = os.path.join(tmp, "archive")
        mod.cmd_push(dos, archive, dry=False)
        # make local strictly newer than archive
        local = os.path.join(dos, "awesome-list-prs-DRAFT.md")
        future = time.time() + 100
        _write(local, "# LOCAL EDIT newer than archive\n", mtime=future)
        mod.cmd_pull(dos, archive, force=False, dry=False)
        with open(local, encoding="utf-8") as f:
            body = f.read()
        check("newer-local NOT overwritten without --force", "LOCAL EDIT" in body, f"got {body!r}")
        mod.cmd_pull(dos, archive, force=True, dry=False)
        with open(local, encoding="utf-8") as f:
            body2 = f.read()
        check("--force DOES overwrite", "LOCAL EDIT" not in body2)


def test_dry_run_writes_nothing():
    print("dry-run writes nothing:")
    with tempfile.TemporaryDirectory() as tmp:
        dos = make_dos(tmp)
        archive = os.path.join(tmp, "archive")
        mod.cmd_push(dos, archive, dry=True)
        check("dry-run created no archive", not os.path.exists(archive))


def main() -> int:
    test_refuses_non_dos_root()
    test_selects_only_depth1_md()
    test_push_then_pull_roundtrip()
    test_pull_refuses_newer_local()
    test_dry_run_writes_nothing()
    print(f"\ndos_sync_test: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
