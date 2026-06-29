#!/usr/bin/env python3
r"""worker_worktree.py — per-worker git worktree isolation for dispatch workers,
reconciled with the trunk-only commit rule (#1334).

THE PROBLEM (from #1334)
------------------------
Every dispatch worker launches with ``cwd = repo root``
(``issue_resolve_dispatch.spawn_issue_worker`` runs in the trunk worktree). So N
concurrent workers share ONE working tree, ONE index, and ONE Go build cache on
the trunk:

  * a worker mid-edit leaves a half-built package that REDS another worker's
    ``go build`` / ``make ci`` (build-poisoning);
  * ``git commit -- <paths>`` from two workers races on the shared index;
  * uncommitted WIP from a stalled worker entangles the next worker's diff.

This is the dominant throughput killer past ~4 concurrent workers and the hard
blocker to scaling the spawn loop toward 100 (#1333).

THE TENSION THIS RECONCILES
---------------------------
``CLAUDE.md`` / ``AGENTS.md``: *never open a feature branch or new worktree — the
trunk guard refuses off-trunk commits (``OFF_TRUNK``)*. And git itself refuses two
worktrees checked out on the SAME branch. So a naive "give each worker a worktree
on ``main``" both trips ``OFF_TRUNK`` (a branch worktree) AND is refused by git
(``main`` is already checked out in the primary).

The reconciliation, the design content of the issue:

  1. Each worker EDITS in its own throwaway worktree checked out at a DETACHED
     HEAD pinned to the current trunk SHA — ``git worktree add --detach <dir>
     <trunk-sha>``. A detached worktree is NOT on ``main`` (so git does not refuse
     it) and is NOT on a feature branch (so it can never be the thing that trips
     ``OFF_TRUNK``). It is an isolated working tree + index; pointing ``GOCACHE`` /
     ``GOTMPDIR`` into it isolates the build too, so a broken build in one worker's
     worktree cannot red another's. (This is the exact pattern the repo's own
     ``fak-selfupdate-build-*`` worktrees already use — strong in-repo precedent.)

  2. The worker's change LANDS on the trunk through a SERIALIZED commit-to-trunk
     step (:func:`land_worktree_diff`): the worktree's diff is applied to the
     trunk worktree and committed there as a normal stamped, signed-off commit
     ``ON main``. Nothing ever commits off-trunk, so ``OFF_TRUNK`` never trips, and
     the trunk stays linear and guarded. Serialization is provided by the lane
     lease the dispatcher already holds (``issue_resolve_dispatch.acquire_lane_lease``)
     — one worker per leaf tree means at most one apply per tree at a time.

DONE-CONDITION (the issue's acceptance, restated as this module's contract)
---------------------------------------------------------------------------
  * During an N-worker wave, :func:`prepare_worker_worktree` yields N isolated
    detached worktrees (``git worktree list`` shows one per live worker).
  * A broken build in one worktree does not red another: each worker's
    ``GOCACHE`` / ``GOTMPDIR`` live INSIDE its own worktree (:func:`worktree_env`).
  * The trunk accumulates NO anonymous cross-worker WIP: a worker edits in its
    worktree, and the only thing that touches the trunk is the serialized
    :func:`land_worktree_diff`, which commits the worker's change as its OWN
    stamped commit.

SAFETY STANCE
-------------
Everything here is FAIL-OPEN and idempotent, the same discipline as
``issue_resolve_dispatch``'s lease/witness helpers: a git error never raises
through these functions (it is reported in the returned dict), so wiring the
isolation in can only ever ADD the isolation, never wedge the dispatcher. The
PURE planners (path/name/env composition) are unit-tested without touching git;
the git-touching create/reap/land functions take an injectable ``git`` runner so
the whole acquire→edit→land→reap path is exercised with a fake.

This module is the reusable PRIMITIVE. Wiring it into the live spawn (passing the
worktree dir as the worker's ``cwd`` and landing on exit) is the follow-on that
builds on it; the primitive lands first, tested, so the wiring has a witnessed
foundation. Pure stdlib; no deps.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any, Callable, Sequence

# Where per-worker worktrees live by default: a sibling scratch root OUTSIDE the
# repo tree so a worktree never shows up in the trunk's own `git status` and is
# never a candidate for `git commit -- <paths>` in the primary. The repo's
# worktree_doctor.py already recognises "scratchpad"/"pr-work" segments as
# disposable; we add our own marker segment so its --sweep-disposable reaps a
# leaked worker worktree too.
WORKTREE_ROOT_ENV = "FLEET_WORKER_WORKTREE_ROOT"
WORKTREE_MARKER = "fak-worker-wt"
# A worktree dir name is <marker>-<lane>-<short-key>; the key is hashed so an
# arbitrary issue/wave label can never inject a path separator or `..`.
_KEY_HASH_LEN = 12

GitRunner = Callable[[Path, list[str]], "tuple[int, str]"]


def _no_window_creationflags() -> int:
    """CREATE_NO_WINDOW on Windows so a git subprocess spawned from a detached
    dispatcher never flashes a console; 0 elsewhere. Mirrors
    dispatch_worker.no_window_creationflags without importing it (this module
    stays dependency-free so a worker can vendor it alone)."""
    return 0x08000000 if os.name == "nt" else 0


def _git(root: Path, args: list[str], *, timeout: int = 120) -> "tuple[int, str]":
    """Run one ``git`` subcommand under ``root``; return ``(rc, stdout)``. Never
    raises: an exec failure is reported as rc 127 so every caller fails open —
    the same contract as ``issue_resolve_dispatch._git_capture``."""
    kwargs: dict[str, Any] = {
        "cwd": str(root), "capture_output": True, "text": True,
        "encoding": "utf-8", "errors": "replace", "timeout": timeout,
    }
    if os.name == "nt":
        kwargs["creationflags"] = _no_window_creationflags()
    try:
        proc = subprocess.run(["git", *args], **kwargs)
    except (OSError, subprocess.SubprocessError):
        return 127, ""
    return proc.returncode, proc.stdout or ""


# --------------------------------------------------------------------------- #
# PURE planners — path / name / env composition. Unit-tested without git.
# --------------------------------------------------------------------------- #

def _safe_key(key: str) -> str:
    """A path-safe short token for an arbitrary worker key (issue number, wave id,
    pid). Hashed so a hostile/odd key (containing ``/``, ``\\``, ``..``) can never
    escape the worktree root or collide a sibling — the dir name stays a single
    flat segment."""
    raw = (str(key) or "worker").encode("utf-8", errors="replace")
    return hashlib.sha1(raw).hexdigest()[:_KEY_HASH_LEN]


def worktree_dir_name(lane: str, key: str) -> str:
    """The flat directory name for one worker's worktree: ``<marker>-<lane>-<key>``.
    ``lane`` is sanitised to ``[A-Za-z0-9_.-]`` (fak lane names already are) and the
    key is hashed, so the result is always one safe path segment."""
    safe_lane = "".join(c if (c.isalnum() or c in "_.-") else "-"
                        for c in (str(lane) or "lane")) or "lane"
    return f"{WORKTREE_MARKER}-{safe_lane}-{_safe_key(key)}"


def default_worktree_root() -> Path:
    """The root directory under which per-worker worktrees are created.

    Honours ``FLEET_WORKER_WORKTREE_ROOT``; otherwise a per-OS scratch location
    OUTSIDE the repo (so a worktree is never inside the trunk tree). The chosen
    base mirrors worktree_doctor's archive-dir convention (LOCALAPPDATA on
    Windows, the system temp dir elsewhere)."""
    override = os.environ.get(WORKTREE_ROOT_ENV)
    if override:
        return Path(override)
    import tempfile
    base = os.environ.get("LOCALAPPDATA") if os.name == "nt" else None
    base = base or tempfile.gettempdir()
    return Path(base) / "Fleet" / "worker-worktrees"


def worktree_path(lane: str, key: str, *, root: Path | None = None) -> Path:
    """The absolute path one worker's isolated worktree will live at."""
    base = root if root is not None else default_worktree_root()
    return Path(base) / worktree_dir_name(lane, key)


def worktree_env(base_env: dict[str, str], wt_dir: Path) -> dict[str, str]:
    """The child env that isolates a worker's BUILD to its own worktree, on top of
    whatever ``base_env`` the dispatcher already composed (``child_env`` +
    account pins).

    Pointing ``GOCACHE`` and ``GOTMPDIR`` INSIDE the worktree is what makes "a
    broken build in one worker's worktree does not red another's" true: each
    worker compiles into its own cache, so a half-built package can never poison a
    sibling's ``go build`` / ``make ci``. ``DISPATCH_WORKSPACE`` is repointed at the
    worktree so a worker that reads it (the self-describing dispatch contract)
    operates on its isolated tree, not the shared trunk."""
    env = dict(base_env)
    wt = str(wt_dir)
    env["DISPATCH_WORKSPACE"] = wt
    env["FLEET_WORKER_WORKTREE"] = wt
    env["GOCACHE"] = str(Path(wt_dir) / ".gocache")
    env["GOTMPDIR"] = str(Path(wt_dir) / ".gotmp")
    return env


def parse_worktree_paths(porcelain: str) -> list[str]:
    """Parse the worktree paths out of ``git worktree list --porcelain`` — the
    pure half of :func:`count_worker_worktrees`, testable without git."""
    return [line[len("worktree "):].strip()
            for line in porcelain.splitlines()
            if line.startswith("worktree ")]


def is_worker_worktree(path: str) -> bool:
    """True when ``path`` is one of OUR per-worker worktrees (its basename carries
    the :data:`WORKTREE_MARKER` segment) — so an auditor can enumerate the live
    wave's isolated worktrees without trusting any worker's self-report."""
    name = os.path.basename(os.path.normpath(path or ""))
    return name.startswith(WORKTREE_MARKER + "-") or name == WORKTREE_MARKER


# --------------------------------------------------------------------------- #
# Git-touching create / reap / land — fail-open, injectable git runner.
# --------------------------------------------------------------------------- #

def trunk_head_sha(root: Path, *, git: GitRunner | None = None) -> str | None:
    """The current trunk HEAD sha to pin a detached worktree to. None on any git
    error (caller fails open: no worktree, worker runs in the shared trunk exactly
    as before this primitive)."""
    run = git or _git
    rc, out = run(root, ["rev-parse", "HEAD"])
    sha = out.strip()
    return sha if rc == 0 and sha else None


def prepare_worker_worktree(root: Path, lane: str, key: str, *,
                            base_sha: str | None = None,
                            wt_root: Path | None = None,
                            git: GitRunner | None = None) -> dict[str, Any]:
    """Create ONE worker's isolated, DETACHED worktree at the trunk HEAD.

    Returns ``{"ok", "path", "base_sha", ...}``. On ``ok`` the worker should run
    with ``cwd = path`` and the env from :func:`worktree_env`; on failure
    (``ok`` False) the dispatcher FAILS OPEN — it runs the worker in the shared
    trunk exactly as before, so a worktree-layer fault never wedges a spawn.

    Detached on purpose (the #1334 reconciliation): the worktree is pinned to a
    SHA, never a branch, so git does not refuse it (``main`` stays singly checked
    out) and it can never be the thing that trips ``OFF_TRUNK``. Idempotent: if the
    target path already holds this worktree it is reported ``reused`` rather than
    re-added (a re-dispatch of the same key never errors)."""
    run = git or _git
    base = base_sha or trunk_head_sha(root, git=run)
    if not base:
        return {"ok": False, "path": None, "base_sha": None,
                "reason": "could not resolve trunk HEAD (git error) — fail open"}
    wt = worktree_path(lane, key, root=wt_root)
    if wt.exists():
        # Already prepared (a retry / re-dispatch). Confirm git still tracks it; if
        # so, reuse it rather than erroring on `worktree add` over an existing dir.
        rc, out = run(root, ["worktree", "list", "--porcelain"])
        tracked = rc == 0 and any(os.path.normcase(os.path.normpath(p))
                                  == os.path.normcase(os.path.normpath(str(wt)))
                                  for p in parse_worktree_paths(out))
        if tracked:
            return {"ok": True, "path": str(wt), "base_sha": base, "reused": True}
    try:
        wt.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return {"ok": False, "path": str(wt), "base_sha": base,
                "reason": f"could not create worktree root: {exc} — fail open"}
    rc, out = run(root, ["worktree", "add", "--detach", str(wt), base])
    if rc != 0:
        return {"ok": False, "path": str(wt), "base_sha": base,
                "reason": f"git worktree add failed (rc {rc}): {out.strip()[-200:]} "
                          "— fail open", "detail": out.strip()[-500:]}
    return {"ok": True, "path": str(wt), "base_sha": base, "reused": False}


def reap_worker_worktree(root: Path, wt_path: str | Path, *,
                         git: GitRunner | None = None) -> dict[str, Any]:
    """Force-remove ONE worker's worktree after its change has LANDED (or it
    crashed). ``--force`` is honest here: the worktree is throwaway editing space,
    and its only durable output is the commit :func:`land_worktree_diff` already
    placed on the trunk — there is nothing to lose by removing it. Best-effort:
    a removal failure is reported, never raised, and a trailing ``worktree prune``
    clears the admin record so a half-removed dir never confuses the auditor."""
    run = git or _git
    p = str(wt_path)
    if not is_worker_worktree(p):
        # Guardrail: only ever reap OUR marker worktrees, never the primary or a
        # peer's scratch worktree (a defensive mirror of worktree_doctor's stance).
        return {"ok": False, "path": p, "removed": False,
                "reason": "refusing to reap a non-worker worktree"}
    rc, out = run(root, ["worktree", "remove", "--force", p])
    removed = rc == 0
    run(root, ["worktree", "prune"])
    return {"ok": removed, "path": p, "removed": removed,
            "detail": out.strip()[-300:] if not removed else None}


def land_worktree_diff(root: Path, wt_path: str | Path, *,
                       commit_msg_file: str | Path,
                       paths: Sequence[str] | None = None,
                       git: GitRunner | None = None) -> dict[str, Any]:
    """Land a worker's edits from its isolated worktree onto the TRUNK as one
    stamped, signed-off commit ON ``main`` — the serialized commit-to-trunk step
    that keeps ``OFF_TRUNK`` from ever tripping (#1334).

    The worker edited in ``wt_path`` (a detached worktree); this captures that
    worktree's diff against its base and applies it to the trunk worktree, then
    commits ``-s`` by explicit path with the worker's prepared message. Because
    the commit happens IN the trunk worktree on ``main``, it is a normal guarded
    commit — never an off-trunk one. The CALLER holds the lane lease, which
    serialises this so two workers never apply to the same leaf tree at once.

    Returns ``{"ok", "applied", "committed", ...}``. FAIL-OPEN: any git error is
    reported, never raised. ``paths`` (when given) scopes the trunk commit to the
    worker's declared file region — never an ``add -A`` of the shared tree."""
    run = git or _git
    wt = str(wt_path)
    # Capture the worker's tracked diff from its worktree (against HEAD = its base).
    rc, diff = run(Path(wt), ["diff", "HEAD"])
    if rc != 0:
        return {"ok": False, "applied": False, "committed": False,
                "reason": f"could not read worktree diff (rc {rc}) — fail open"}
    if not diff.strip():
        # The worker committed in-worktree, or made no tracked change. Nothing to
        # apply; the caller's commit-witness (dos commit-audit) is the arm that
        # decides whether the slot was productive.
        return {"ok": True, "applied": False, "committed": False,
                "reason": "no tracked diff in worktree to land"}
    # Apply the captured diff to the trunk worktree's working tree.
    proc = _git_apply(root, diff, git_run=run)
    if not proc.get("ok"):
        return {"ok": False, "applied": False, "committed": False,
                "reason": f"git apply to trunk failed: {proc.get('detail')}"}
    commit_args = ["commit", "-s", "-F", str(commit_msg_file)]
    if paths:
        commit_args += ["--", *list(paths)]
    rc, out = run(root, commit_args)
    return {"ok": rc == 0, "applied": True, "committed": rc == 0,
            "detail": out.strip()[-300:]}


def _git_apply(root: Path, diff: str, *, git_run: GitRunner) -> dict[str, Any]:
    """Apply a captured diff to ``root``'s working tree via ``git apply``. Kept
    separate so the apply step is injectable/testable; reads the patch from a temp
    file (a long diff exceeds argv limits) and removes it after."""
    import tempfile
    fd, patch = tempfile.mkstemp(prefix="fak-wt-land-", suffix=".patch")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(diff if diff.endswith("\n") else diff + "\n")
        rc, out = git_run(root, ["apply", "--whitespace=nowarn", patch])
        return {"ok": rc == 0, "detail": out.strip()[-300:]}
    finally:
        try:
            os.unlink(patch)
        except OSError:
            pass


def count_worker_worktrees(root: Path, *, git: GitRunner | None = None) -> dict[str, Any]:
    """Enumerate the live per-worker worktrees from ``git worktree list`` — the
    auditor's direct check of the #1334 done-condition ("``git worktree list``
    shows N isolated worktrees, one per live worker"), read from git, not from any
    worker's self-report. Returns ``{"count", "paths"}``."""
    run = git or _git
    rc, out = run(root, ["worktree", "list", "--porcelain"])
    if rc != 0:
        return {"count": 0, "paths": [], "error": out.strip()[-200:]}
    paths = [p for p in parse_worktree_paths(out) if is_worker_worktree(p)]
    return {"count": len(paths), "paths": sorted(paths)}
