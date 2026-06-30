#!/usr/bin/env python3
"""Single-writer release lock for fleet — the /release critical-section guard.

Fleet is a live multi-session cluster (several `claude.exe` sessions, each its own
dos MCP server, often releasing in parallel). The `curate-cluster` skill rightly
says you need *no* lock to commit curation work: those commits are commutative and
markdown self-heals, so git's own index serialization is enough. **Releases are the
exception.** A release mutates two single-writer resources — the bare `VERSION`
marker and the monotone `vX.Y.Z` tag sequence — and two concurrent `/release` runs
race them: both bump `VERSION`, both try to create the same tag, then collide on
`git push`. `docs/production-readiness.md` item 2 ("single-writer releases … a
release lock so concurrent sessions don't race VERSION/tag") is exactly this gap.

This helper is that lock. It is deliberately small and host-agnostic:

  acquire   atomically create `.release.lock` (O_EXCL); refuse if a *live* lock is
            held by another session; steal a *stale* one (past its TTL) or with
            --force. The lock records its owner, so the holder can re-prove identity
            across separate `python` invocations.
  verify    exit 0 iff the lock is held by --owner and not expired (used by
            release_bump.py to gate the VERSION mutation on lock ownership).
  guard     compare the *actually staged* set (`git diff --cached`) against the
            release's *declared* snapshot — the mechanical net under the "never
            `git add -A`/`-u`/`.`" rule: a sweep that grabbed a peer's in-flight
            file shows up here as a `foreign` staged path and fails the gate.
  status    print the current lock (held / stale / free) for humans.
  release   remove the lock iff owned by --owner (or --force).

**Owner identity.** `--owner` defaults to `$FAK_RELEASE_OWNER`, else
`$CLAUDE_CODE_SESSION_ID` (the harness sets one per session — stable across Bash
calls, unique per session), else a host+user+pid fallback. So within one session
acquire → bump → release auto-share an owner with no token to thread by hand.

**Staleness is TTL-based, not PID-based — on purpose.** `os.kill(pid, 0)` is a
liveness probe on POSIX but *terminates* the process on Windows (this is a Windows
host), so we never signal. A crashed release's lock simply expires after its TTL
(default 30 min, `--ttl` to shrink); `--force` steals one sooner. The recorded
pid/host are diagnostics only.

Pure stdlib; off the request path (tooling seam). Exit codes: 0 ok, 2 usage,
3 contended/denied (held by another / not owner / foreign staged), 4 internal.

Usage:
  python tools/release_lock.py acquire --ttl 1800
  python tools/release_lock.py acquire --snapshot VERSION --snapshot docs/foo.md
  python tools/release_lock.py verify
  python tools/release_lock.py guard --allow VERSION --allow docs/releases/v0.5.0.md
  python tools/release_lock.py status
  python tools/release_lock.py release
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import socket
import subprocess
from dispatch_worker import install_no_window_subprocess_defaults
import sys
import time
from pathlib import Path
install_no_window_subprocess_defaults(subprocess)

LOCK_NAME = ".release.lock"
DEFAULT_TTL = 1800  # 30 min — a crashed release auto-recovers after this.
DEFAULT_NOTES_DIR = "docs/releases"

# Exit codes (shared with release_bump.py's gate).
EXIT_OK, EXIT_USAGE, EXIT_DENIED, EXIT_INTERNAL = 0, 2, 3, 4


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def lock_path(root: Path) -> Path:
    override = os.environ.get("FAK_RELEASE_LOCK_ROOT")
    if override:
        return Path(override).resolve() / LOCK_NAME
    return root / LOCK_NAME


def now() -> float:
    return time.time()


def host() -> str:
    try:
        return socket.gethostname()
    except OSError:
        return "?"


def git(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=str(root), stderr=subprocess.DEVNULL, text=True, encoding="utf-8"
        )
    except (subprocess.CalledProcessError, OSError):
        return ""


def default_owner() -> str:
    """A token that is stable across this session's separate `python` invocations."""
    sid = os.environ.get("FAK_RELEASE_OWNER") or os.environ.get("CLAUDE_CODE_SESSION_ID")
    if sid:
        return sid.strip()
    # Last resort: unique per host+user but NOT stable across calls (pid varies).
    # Real sessions hit the env path above; this only bites bare-shell manual use.
    return f"{os.environ.get('USERNAME') or os.environ.get('USER') or 'user'}@{host()}:{os.getpid()}"


def read_lock(root: Path) -> dict | None:
    p = lock_path(root)
    try:
        text = p.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return None
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None  # corrupt → treated as stealable by acquire


def is_stale(lock: dict, at: float | None = None) -> tuple[bool, str]:
    at = now() if at is None else at
    expires = lock.get("expires_at")
    if not isinstance(expires, (int, float)):
        # No/garbled expiry → reconstruct from acquired_at+ttl, else call it stale.
        acquired = lock.get("acquired_at")
        ttl = lock.get("ttl")
        if isinstance(acquired, (int, float)) and isinstance(ttl, (int, float)):
            expires = acquired + ttl
        else:
            return True, "no expiry recorded"
    return (at >= expires, "expired" if at >= expires else "live")


def held_by(root: Path, owner: str | None, at: float | None = None) -> tuple[bool, str]:
    """True iff a live lock exists and (owner is None or matches the holder)."""
    lock = read_lock(root)
    if lock is None:
        return False, "no lock present"
    stale, why = is_stale(lock, at)
    if stale:
        return False, f"stale ({why})"
    if owner is not None and lock.get("owner") != owner:
        return False, "held by another session"
    return True, "held"


def lock_state(root: Path, at: float | None = None) -> dict:
    at = now() if at is None else at
    lock = read_lock(root)
    if lock is None:
        return {"held": False, "lock": None}
    stale, why = is_stale(lock, at)
    acquired = lock.get("acquired_at")
    expires = lock.get("expires_at")
    return {
        "held": not stale,
        "stale": stale,
        "reason": why,
        "lock": lock,
        "age_s": round(at - acquired, 1) if isinstance(acquired, (int, float)) else None,
        "remaining_s": round(expires - at, 1) if isinstance(expires, (int, float)) else None,
    }


# --------------------------------------------------------------------------- verbs


def acquire(root: Path, *, ttl: int, owner: str, snapshot: list[str], note: str | None,
            steal_stale: bool, force: bool) -> tuple[dict, int]:
    started = now()
    branch = git(root, "rev-parse", "--abbrev-ref", "HEAD").strip()
    head = git(root, "rev-parse", "--short", "HEAD").strip()
    payload = {
        "owner": owner,
        "pid": os.getpid(),
        "host": host(),
        "branch": branch or None,
        "git_user": git(root, "config", "user.name").strip() or None,
        "head_sha": head or None,
        "acquired_at": started,
        "ttl": ttl,
        "expires_at": started + ttl,
        "snapshot": [s.replace("\\", "/") for s in snapshot],
    }
    if note:
        payload["note"] = note
    data = (json.dumps(payload, indent=2) + "\n").encode("utf-8")

    path = lock_path(root)
    stole: dict | None = None
    for _ in range(5):
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            existing = read_lock(root)
            if existing is None:
                stale, why = True, "corrupt lock file"
            else:
                stale, why = is_stale(existing, started)
            if force or (steal_stale and stale):
                stole = {**(existing or {}), "_stolen_because": "force" if (force and not stale) else why}
                try:
                    os.unlink(str(path))
                except FileNotFoundError:
                    pass
                continue  # race back to the O_EXCL create
            return {"ok": False, "reason": "held", "holder": existing}, EXIT_DENIED
        else:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            out = {"ok": True, "lock": payload}
            if stole is not None:
                out["stole"] = stole
            return out, EXIT_OK
    return {"ok": False, "reason": "contended", "detail": "lost the steal race 5x"}, EXIT_DENIED


def release(root: Path, *, owner: str | None, force: bool) -> tuple[dict, int]:
    lock = read_lock(root)
    if lock is None:
        return {"ok": True, "released": False, "reason": "no lock present"}, EXIT_OK
    if not force and (owner is None or lock.get("owner") != owner):
        return {"ok": False, "reason": "not owner", "holder": lock,
                "hint": "pass --owner <token> or --force"}, EXIT_DENIED
    try:
        os.unlink(str(lock_path(root)))
    except FileNotFoundError:
        pass
    return {"ok": True, "released": True, "was": lock}, EXIT_OK


def staged_paths(root: Path) -> list[str]:
    raw = git(root, "diff", "--cached", "--name-only", "-z")
    return [p.replace("\\", "/") for p in raw.split("\0") if p]


def classify_staged(staged: list[str], allowed: list[str], notes_dir: str) -> dict:
    """Partition staged paths into allowed vs foreign.

    A path is allowed if it equals (or fnmatch-matches) any --allow entry, is the
    VERSION marker, or is a release note under notes_dir. Everything else is
    `foreign` — the signature of a `git add -A` sweep that grabbed a peer's file.
    """
    notes_dir = notes_dir.replace("\\", "/").rstrip("/")
    patterns = [a.replace("\\", "/") for a in allowed]
    foreign: list[str] = []
    for raw in staged:
        p = raw.replace("\\", "/")
        if p == "VERSION":
            continue
        if p.startswith(notes_dir + "/") and p.endswith(".md"):
            continue
        if any(p == pat or fnmatch.fnmatch(p, pat) for pat in patterns):
            continue
        foreign.append(p)
    return {"ok": not foreign, "staged": staged, "foreign": foreign}


def guard(root: Path, *, allow: list[str], notes_dir: str, from_lock: bool) -> tuple[dict, int]:
    allowed = list(allow)
    if from_lock:
        lock = read_lock(root)
        if lock and isinstance(lock.get("snapshot"), list):
            allowed += [str(s) for s in lock["snapshot"]]
    result = classify_staged(staged_paths(root), allowed, notes_dir)
    if not result["ok"]:
        result["reason"] = (
            f"{len(result['foreign'])} staged path(s) outside the declared release snapshot "
            "- looks like a `git add -A/-u/.` sweep. Unstage them (git restore --staged <path>) "
            "and stage only the snapshot explicitly."
        )
    return result, EXIT_OK if result["ok"] else EXIT_DENIED


# --------------------------------------------------------------------------- cli


def _emit(payload: dict, code: int) -> int:
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return code


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Single-writer release lock for fleet.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("acquire", help="atomically take the release lock")
    a.add_argument("--ttl", type=int, default=DEFAULT_TTL, help=f"lock lifetime in seconds (default {DEFAULT_TTL})")
    a.add_argument("--owner", default=None, help="owner token (default: session id)")
    a.add_argument("--snapshot", action="append", default=[], help="a path this release will stage (repeatable)")
    a.add_argument("--note", default=None, help="free-text note recorded in the lock")
    a.add_argument("--no-steal-stale", action="store_true", help="do NOT auto-steal an expired lock")
    a.add_argument("--force", action="store_true", help="steal even a live lock (use sparingly)")

    v = sub.add_parser("verify", help="exit 0 iff the lock is held by --owner")
    v.add_argument("--owner", default=None, help="owner token to check (default: session id)")

    g = sub.add_parser("guard", help="fail if staged files escape the declared snapshot (no -A sweeps)")
    g.add_argument("--allow", action="append", default=[], help="an in-scope path/glob (repeatable)")
    g.add_argument("--notes-dir", default=DEFAULT_NOTES_DIR, help=f"release-notes dir (default {DEFAULT_NOTES_DIR})")
    g.add_argument("--no-from-lock", action="store_true", help="ignore the snapshot recorded in the lock")

    sub.add_parser("status", help="print the current lock state")

    r = sub.add_parser("release", help="release the lock if owned")
    r.add_argument("--owner", default=None, help="owner token (default: session id)")
    r.add_argument("--force", action="store_true", help="release regardless of owner")

    args = ap.parse_args(argv)
    root = repo_root()

    if args.cmd == "acquire":
        owner = args.owner or default_owner()
        payload, code = acquire(
            root, ttl=args.ttl, owner=owner, snapshot=args.snapshot, note=args.note,
            steal_stale=not args.no_steal_stale, force=args.force,
        )
        return _emit(payload, code)

    if args.cmd == "verify":
        owner = args.owner or default_owner()
        ok, why = held_by(root, owner)
        return _emit({"ok": ok, "owner": owner, "reason": why}, EXIT_OK if ok else EXIT_DENIED)

    if args.cmd == "guard":
        payload, code = guard(root, allow=args.allow, notes_dir=args.notes_dir, from_lock=not args.no_from_lock)
        return _emit(payload, code)

    if args.cmd == "status":
        return _emit(lock_state(root), EXIT_OK)

    if args.cmd == "release":
        owner = None if args.force else (args.owner or default_owner())
        payload, code = release(root, owner=owner, force=args.force)
        return _emit(payload, code)

    ap.error(f"unknown command {args.cmd!r}")
    return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
