#!/usr/bin/env python3
"""Bump fleet's version-marker file(s) in one shot — the /release Step 4 helper.

Fleet's only load-bearing version marker is the bare-semver `VERSION` file at the
repo root (fak/ carries no version constant today). This prints a JSON report —
`{new_version, dry_run, targets}` — where `targets` maps each marker to
`{ok, changed, path, ...}`, exactly the shape `.claude/skills/release/SKILL.md`
§ "Expected behavior of helpers.release_bump" documents. Idempotent (new == current
is a clean no-op). Pure stdlib.

**Single-writer gate.** Bumping `VERSION` is the moment the release mutates a
single-writer resource, so by default this refuses unless the caller holds the
release lock (`tools/release_lock.py`) — the mechanical enforcement of
`docs/production-readiness.md` item 2 ("single-writer releases"). The /release
skill acquires the lock first (Step 0.5); within one session owner identity is the
shared `$CLAUDE_CODE_SESSION_ID`, so no token need be threaded by hand. `--no-lock`
bypasses the gate for genuinely solo / CI use; a `--dry-run` never needs the lock.

Usage:
  python tools/release_bump.py 0.2.0                 # requires a held release lock
  python tools/release_bump.py 0.2.0 --owner <tok>   # gate on a specific owner
  python tools/release_bump.py 0.2.0 --dry-run       # no lock needed
  python tools/release_bump.py 0.2.0 --no-lock       # solo/CI escape hatch
  python tools/release_bump.py 0.2.0 --skip version
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

try:  # same tools/ dir; import works when run as `python tools/release_bump.py`
    import release_lock
except ImportError:  # pragma: no cover - defensive
    release_lock = None  # type: ignore[assignment]

SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def bump_plain_file(root: Path, rel: str, new: str, *, dry_run: bool) -> dict:
    path = root / rel
    if not path.exists():
        return {"path": rel, "ok": False, "reason": f"{rel} not found"}
    old = path.read_text(encoding="utf-8").strip()
    target = f"{new}\n"
    changed = path.read_text(encoding="utf-8") != target
    if changed and not dry_run:
        path.write_text(target, encoding="utf-8")
    return {"path": rel, "old": old, "new": new, "changed": changed, "ok": True}


def main() -> int:
    ap = argparse.ArgumentParser(description="Bump fleet version-marker file(s).")
    ap.add_argument("version", help="New semver, e.g. 0.2.0 (no leading v)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip", action="append", default=[], choices=["version"], help="Skip a target (repeatable)")
    ap.add_argument("--owner", default=None, help="release-lock owner to gate on (default: session id)")
    ap.add_argument("--no-lock", action="store_true", help="bypass the single-writer release lock (solo/CI only)")
    args = ap.parse_args()

    new_version = args.version.lstrip("v")
    if not SEMVER_RE.match(new_version):
        print(f"ERROR: {args.version!r} is not a valid semver (expected X.Y.Z)", file=sys.stderr)
        return 2

    root = repo_root()

    # Single-writer gate: a real bump mutates VERSION, so require the release lock.
    # A dry-run mutates nothing, so it never needs the lock.
    if not args.no_lock and not args.dry_run:
        if release_lock is None:
            print(json.dumps({"ok": False, "reason": "release_lock helper unavailable; pass --no-lock to override"}, indent=2))
            return 4
        owner = args.owner or release_lock.default_owner()
        held, why = release_lock.held_by(root, owner)
        if not held:
            print(json.dumps({
                "ok": False,
                "reason": f"release lock not held ({why})",
                "hint": "acquire it first: python tools/release_lock.py acquire   (or pass --no-lock for solo/CI)",
                "owner_checked": owner,
            }, indent=2))
            return 4
    actions = {
        "version": lambda: bump_plain_file(root, "VERSION", new_version, dry_run=args.dry_run),
    }

    report: dict = {"new_version": new_version, "dry_run": args.dry_run, "targets": {}}
    any_error = False
    for key, fn in actions.items():
        if key in args.skip:
            report["targets"][key] = {"skipped": True, "reason": "--skip"}
            continue
        try:
            result = fn()
        except Exception as exc:  # surface but don't crash — the skill can still commit manually
            result = {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}
        report["targets"][key] = result
        if not result.get("ok", True):
            any_error = True

    print(json.dumps(report, indent=2))
    return 1 if any_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
