#!/usr/bin/env python3
"""sync_memory.py — mirror the node-local Claude Code auto-memory store into the
repo (so it ships via git) and back (to seed a fresh node).

The Claude Code harness keeps auto-memory under ``~/.claude/projects/<slug>/memory/``,
which never leaves the machine. The repo carries a committed mirror at
``.claude/memory/`` so that hard-won fleet knowledge ships with the tree like any other
tracked file. This script copies the ``*.md`` memory files between the two stores.

    python tools/sync_memory.py --push    # home  -> repo   (run before committing memory)
    python tools/sync_memory.py --pull    # repo  -> home   (run when seeding a new node)
    python tools/sync_memory.py --status  # show what differs, change nothing (default)

Only ``*.md`` files are synced. By default neither direction deletes files on the
destination — pass ``--prune`` to also remove destination ``*.md`` files that are absent
from the source (off by default so a stale mirror never silently drops a peer's
still-referenced memory). The home directory is derived from this repo's path the same
way the harness slugifies it; override with ``--home <dir>`` or ``CLAUDE_MEMORY_DIR``.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path


def repo_root() -> Path:
    # tools/sync_memory.py -> repo root is the parent of tools/.
    return Path(__file__).resolve().parent.parent


def default_home_memory(root: Path) -> Path:
    """Mirror the harness slug: non-alphanumerics in the project path become '-'.

    e.g. C:\\projects\\fleet  ->  C--projects-fleet
    """
    env = os.environ.get("CLAUDE_MEMORY_DIR")
    if env:
        return Path(env)
    slug = re.sub(r"[^A-Za-z0-9]", "-", str(root))
    return Path.home() / ".claude" / "projects" / slug / "memory"


# README.md documents the mirror itself; it is not a memory fact, so it is never
# synced into the harness store (a --pull must not land it where the harness would
# load it as a memory).
_EXCLUDE = {"README.md"}


def md_files(d: Path) -> dict[str, Path]:
    if not d.is_dir():
        return {}
    return {p.name: p for p in sorted(d.glob("*.md")) if p.name not in _EXCLUDE}


def differ(a: Path, b: Path) -> bool:
    if not b.exists():
        return True
    return a.read_bytes() != b.read_bytes()


def status(home: Path, mirror: Path) -> int:
    h, m = md_files(home), md_files(mirror)
    only_home = sorted(set(h) - set(m))
    only_repo = sorted(set(m) - set(h))
    changed = sorted(n for n in (set(h) & set(m)) if differ(h[n], m[n]))
    print(f"home  : {home}  ({len(h)} md)")
    print(f"repo  : {mirror}  ({len(m)} md)")
    if only_home:
        print(f"  only in home (would --push): {', '.join(only_home)}")
    if only_repo:
        print(f"  only in repo (would --pull): {', '.join(only_repo)}")
    if changed:
        print(f"  differing content        : {', '.join(changed)}")
    if not (only_home or only_repo or changed):
        print("  in sync.")
    return 0


def copy_dir(src: Path, dst: Path, prune: bool) -> int:
    s = md_files(src)
    if not s:
        print(f"nothing to copy: no *.md under {src}", file=sys.stderr)
        return 1
    dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    for name, path in s.items():
        target = dst / name
        if differ(path, target):
            target.write_bytes(path.read_bytes())
            print(f"  copied {name}")
            copied += 1
    if prune:
        for name, path in md_files(dst).items():
            if name not in s:
                path.unlink()
                print(f"  pruned {name}")
    if copied == 0:
        print("  already up to date.")
    else:
        print(f"  {copied} file(s) updated -> {dst}")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--push", action="store_true", help="home -> repo mirror")
    g.add_argument("--pull", action="store_true", help="repo mirror -> home")
    g.add_argument("--status", action="store_true", help="report differences only (default)")
    ap.add_argument("--prune", action="store_true",
                    help="also delete destination *.md absent from source")
    ap.add_argument("--home", help="override the node-local memory dir")
    args = ap.parse_args(argv)

    root = repo_root()
    mirror = root / ".claude" / "memory"
    home = Path(args.home) if args.home else default_home_memory(root)

    if args.push:
        print(f"push: {home} -> {mirror}")
        return copy_dir(home, mirror, args.prune)
    if args.pull:
        print(f"pull: {mirror} -> {home}")
        return copy_dir(mirror, home, args.prune)
    return status(home, mirror)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
