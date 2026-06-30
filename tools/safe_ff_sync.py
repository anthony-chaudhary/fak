#!/usr/bin/env python3
"""Safe fast-forward sync for fleet's shared multi-session worktree.

`C:\\work\\fleet` is ONE git clone shared by many concurrent Claude sessions. The
branch ref, index, and worktree are shared, and the worktree almost always carries
100+ files of in-flight peer work (modified / deleted / untracked). The hazard,
recorded live (see the `fleet-shared-worktree-git-hazard` note): a plain
`git pull --ff-only` on a *behind + dirty* branch moved the local ref forward but
**aborted the worktree update** when an untracked peer file blocked the checkout —
leaving HEAD ahead of a stale worktree, i.e. ~115 phantom changes (a version skew).

This helper performs the ONE fast-forward that cannot cause that skew. It only
fast-forwards when **every path the ff would write is already byte-identical in the
worktree to the remote-tracking version**. In that case the ff rewrites nothing the
worktree doesn't already have, so no peer edit can be clobbered and no checkout can
be aborted half-way. If ANY ff-path diverges locally, it REFUSES and prints the
divergent set — sync is left to a human / a quiescent moment, exactly as the
shared-clone rule prescribes. Novel peer work on *other* paths is never touched.

It is deliberately conservative and NEVER runs `git pull`, `git stash`,
`git reset --hard`, `git clean`, or `git add`. It operates only on the
already-fetched remote-tracking ref (with `--fetch` it will `git fetch <remote>
<branch>` first; otherwise it trusts whatever you last fetched).

  check   (default) report whether a safe ff is possible: the write-set, the
          identical paths, and any divergent paths. Read-only — mutates nothing.
  apply   perform the safe ff — only if `check` would pass. For each ff-path it
          clears the identical-but-locally-dirty tracked files back to HEAD and
          drops the identical untracked additions, then `git merge --ff-only`.

This is the single-writer release lock's git-hygiene sibling: `release_lock.py`
serializes *releases*; this serializes nothing but refuses the *one* sync move that
silently corrupts a shared worktree. Pure stdlib; host-agnostic.

Exit codes: 0 ok (safe / applied / already in sync), 2 usage, 3 unsafe/refused,
4 internal. JSON to stdout with --json.

Usage:
  python tools/safe_ff_sync.py check
  python tools/safe_ff_sync.py check --branch fak-v0.1 --remote origin --json
  python tools/safe_ff_sync.py apply --fetch
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

_CREATE_NO_WINDOW = 0x08000000


def _win_creationflags() -> int:
    return _CREATE_NO_WINDOW if os.name == "nt" else 0


# Exit codes (mirror release_lock.py's vocabulary).
EXIT_OK, EXIT_USAGE, EXIT_UNSAFE, EXIT_INTERNAL = 0, 2, 3, 4


class GitError(RuntimeError):
    pass


def git(args, repo=".", check=True, binary=False):
    """Run a git command; return stdout (str, or bytes if binary)."""
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        check=False,
        creationflags=_win_creationflags(),
    )
    if check and proc.returncode != 0:
        raise GitError(
            f"git {' '.join(args)} -> {proc.returncode}: "
            f"{proc.stderr.decode('utf-8', 'replace').strip()}"
        )
    return proc.stdout if binary else proc.stdout.decode("utf-8", "replace")


def current_branch(repo):
    name = git(["rev-parse", "--abbrev-ref", "HEAD"], repo).strip()
    if name == "HEAD":
        raise GitError("detached HEAD — no branch to sync")
    return name


def rev(repo, ref):
    return git(["rev-parse", "--verify", ref], repo).strip()


def is_ancestor(repo, a, b):
    """True iff commit `a` is an ancestor of `b` (so a->b is a fast-forward)."""
    proc = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", a, b],
        capture_output=True,
        creationflags=_win_creationflags(),
    )
    if proc.returncode not in (0, 1):
        raise GitError(
            f"merge-base --is-ancestor failed: "
            f"{proc.stderr.decode('utf-8', 'replace').strip()}"
        )
    return proc.returncode == 0


def blob_at(repo, ref, path):
    """Bytes of `path` at `ref`, or None if it does not exist there."""
    proc = subprocess.run(
        ["git", "-C", str(repo), "show", f"{ref}:{path}"],
        capture_output=True,
        creationflags=_win_creationflags(),
    )
    return proc.stdout if proc.returncode == 0 else None


def worktree_bytes(repo, path):
    """Bytes of `path` in the worktree, or None if absent."""
    fp = Path(repo) / path
    try:
        return fp.read_bytes()
    except (FileNotFoundError, NotADirectoryError, IsADirectoryError):
        return None


def ff_write_set(repo, head, target):
    """Paths git would write going HEAD -> target, as (status, path) pairs.

    Renames/copies are surfaced as their own status so the caller refuses them
    (they are rare in a pure ff set and not worth a fragile decomposition)."""
    out = git(["diff", "--name-status", "-z", head, target], repo)
    fields = out.split("\0")
    entries = []
    i = 0
    while i < len(fields):
        status = fields[i]
        if not status:
            i += 1
            continue
        code = status[0]
        if code in ("R", "C"):
            # status, old, new  (two path fields follow)
            old, new = fields[i + 1], fields[i + 2]
            entries.append((status, old))
            entries.append((status, new))
            i += 3
        else:
            entries.append((code, fields[i + 1]))
            i += 2
    return entries


def classify(repo, head, target, entries):
    """Split ff-write paths into identical (safe) vs divergent (unsafe).

    A path is SAFE when applying the ff cannot clobber divergent local content:
      M (changed)  worktree bytes == target bytes  (we'll reset->HEAD then ff)
      A (added)    worktree absent, or worktree bytes == target bytes (rm then ff)
      D (deleted)  worktree absent, or worktree bytes == HEAD bytes (clean delete)
    Anything else (R/C/T, or any byte divergence) is UNSAFE.
    """
    identical, divergent = [], []
    for status, path in entries:
        wt = worktree_bytes(repo, path)
        if status == "M":
            tgt = blob_at(repo, target, path)
            safe = wt is not None and tgt is not None and wt == tgt
        elif status == "A":
            tgt = blob_at(repo, target, path)
            safe = wt is None or (tgt is not None and wt == tgt)
        elif status == "D":
            base = blob_at(repo, head, path)
            safe = wt is None or (base is not None and wt == base)
        else:
            safe = False
        (identical if safe else divergent).append({"status": status, "path": path})
    return identical, divergent


def assess(repo, remote, branch, do_fetch):
    """Read-only assessment of whether a safe ff is possible."""
    if do_fetch:
        git(["fetch", remote, branch], repo)
    target_ref = f"{remote}/{branch}"
    head = rev(repo, "HEAD")
    try:
        target = rev(repo, target_ref)
    except GitError:
        return {
            "ok": False, "state": "no-remote-ref", "target_ref": target_ref,
            "reason": f"remote-tracking ref {target_ref} not found — fetch first",
        }

    if head == target:
        return {"ok": True, "state": "in-sync", "head": head, "target": target}
    if is_ancestor(repo, target, head):
        return {"ok": False, "state": "ahead", "head": head, "target": target,
                "reason": "local branch is ahead of remote — nothing to ff (push instead)"}
    if not is_ancestor(repo, head, target):
        return {"ok": False, "state": "diverged", "head": head, "target": target,
                "reason": "local and remote have diverged — not a fast-forward; resolve by hand"}

    entries = ff_write_set(repo, head, target)
    identical, divergent = classify(repo, head, target, entries)
    return {
        "ok": not divergent,
        "state": "behind",
        "head": head,
        "target": target,
        "target_ref": target_ref,
        "write_count": len(entries),
        "identical": identical,
        "divergent": divergent,
        "reason": (
            "every ff-path is byte-identical to the remote — safe to fast-forward"
            if not divergent else
            f"{len(divergent)} path(s) diverge locally — refusing; "
            "sync at a quiescent moment or resolve by hand"
        ),
    }


def apply_ff(repo, remote, branch, info):
    """Perform the safe ff. Caller guarantees info['ok'] and state == 'behind'."""
    target_ref = info["target_ref"]
    # 1. Clear identical-but-locally-dirty tracked files back to HEAD so the merge
    #    won't refuse "local changes would be overwritten" (content is identical to
    #    the target, so the ff restores it byte-for-byte).
    m_paths = [e["path"] for e in info["identical"] if e["status"] == "M"]
    if m_paths:
        git(["checkout", "HEAD", "--", *m_paths], repo)
    # 2. Drop identical untracked additions so the ff can create them.
    for e in info["identical"]:
        if e["status"] == "A":
            fp = Path(repo) / e["path"]
            if fp.exists():
                fp.unlink()
    # 3. The one safe fast-forward — no pull, no stash, no reset.
    git(["merge", "--ff-only", target_ref], repo)
    return rev(repo, "HEAD")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Safe fast-forward sync for a shared worktree.")
    ap.add_argument("command", choices=["check", "apply"], nargs="?", default="check")
    ap.add_argument("--repo", default=".", help="repo path (default: cwd)")
    ap.add_argument("--remote", default="origin")
    ap.add_argument("--branch", default=None, help="default: current branch")
    ap.add_argument("--fetch", action="store_true",
                    help="git fetch <remote> <branch> before assessing")
    ap.add_argument("--json", action="store_true", help="emit the assessment as JSON")
    args = ap.parse_args(argv)

    try:
        repo = args.repo
        branch = args.branch or current_branch(repo)
        info = assess(repo, args.remote, branch, args.fetch)
        info["branch"] = branch

        if args.command == "apply":
            if info["state"] == "in-sync":
                pass  # nothing to do
            elif info["state"] == "behind" and info["ok"]:
                info["new_head"] = apply_ff(repo, args.remote, branch, info)
                info["applied"] = True
            else:
                info["applied"] = False  # refused — leave the tree untouched

        if args.json:
            print(json.dumps(info, indent=2))
        else:
            _print_human(info, args.command)

        if info["state"] == "in-sync":
            return EXIT_OK
        if args.command == "apply":
            return EXIT_OK if info.get("applied") else EXIT_UNSAFE
        return EXIT_OK if info["ok"] else EXIT_UNSAFE
    except GitError as e:
        print(f"safe_ff_sync: {e}", file=sys.stderr)
        return EXIT_INTERNAL


def _print_human(info, command):
    state = info["state"]
    if state == "in-sync":
        print("in sync — local branch already at the remote; nothing to do.")
        return
    if state in ("ahead", "diverged", "no-remote-ref"):
        print(f"{state}: {info['reason']}")
        return
    # behind
    n_id, n_div = len(info["identical"]), len(info["divergent"])
    verb = "applied" if info.get("applied") else ("SAFE" if info["ok"] else "REFUSED")
    print(f"[{verb}] behind {info['target_ref']} — {info['write_count']} ff-path(s): "
          f"{n_id} identical, {n_div} divergent")
    print(f"  {info['reason']}")
    for e in info["divergent"]:
        print(f"    DIVERGES  {e['status']}  {e['path']}")
    if info.get("applied"):
        print(f"  HEAD -> {info['new_head'][:12]} (novel local work on other paths preserved)")


if __name__ == "__main__":
    sys.exit(main())
