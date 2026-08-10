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

This module is the reusable PRIMITIVE, and its :func:`main` CLI (``prepare`` /
``land`` / ``reap`` / ``list``) makes it RUNNABLE: a guarded fak instance, an
operator, or a self-source dispatch ticket that hits ``SELF_MODIFY_HOLD`` can now
drive the full acquire → edit → land → reap cycle from the shell instead of
following a prose pointer at "#1334" that resolved to nothing invokable —

    wt=$(python tools/worker_worktree.py --root . prepare --lane cmd --key 1338 | jq -r .path)
    # ...edit the self-source files INSIDE $wt (its own isolated detached worktree)...
    python tools/worker_worktree.py --root . land --worktree "$wt" \
        --msg-file msg.txt --path cmd/fak/foo.go
    python tools/worker_worktree.py --root . reap --worktree "$wt"

AUTO-wiring the live dispatch spawn (passing the worktree dir as each worker's
``cwd`` and landing on exit, so a self-source ticket flows through this WITHOUT
an operator in the loop) remains the follow-on (#1333's worktree-isolation
blocker) that builds on this runnable path. The primitive + CLI land first,
tested and dog-fooded, so that wiring has a witnessed foundation. Pure stdlib;
no deps.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
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
WARM_GOCACHE_ENV = "FAK_DISPATCH_WORKTREE_WARM_GOCACHE"
_CREATE_NO_WINDOW = 0x08000000


def no_window_creationflags() -> int:
    """Suppress helper console windows on Windows; creationflags must be zero on POSIX."""
    return _CREATE_NO_WINDOW if os.name == "nt" else 0
WORKTREE_MARKER = "fak-worker-wt"
# The gate that turns per-worker worktree isolation ON, shared verbatim with the Go
# spine: cmd/fak/dispatch_tick_worker.go:workerWorktreeEnabled reads the SAME env var
# with the SAME truthy/falsy grammar, so the native Go spawn site (#3168) and the two
# Python spawn sites (#3181) obey ONE flag under ONE fail-open contract. Default OFF
# restores the shared-trunk spawn byte-for-byte.
WORKTREE_ENABLE_ENV = "FLEET_WORKER_WORKTREE"
_ENABLE_OFF_VALUES = frozenset({"", "0", "off", "false", "no", "disable", "disabled"})
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


def worktree_isolation_enabled(environ: "dict[str, str] | None" = None) -> bool:
    """True when ``FLEET_WORKER_WORKTREE`` selects per-worker worktree isolation.

    Mirrors ``cmd/fak/dispatch_tick_worker.go:workerWorktreeEnabled`` EXACTLY — unset or
    an off-ish value (``0``/``off``/``false``/``no``/``disable``/``disabled``/empty) is
    OFF, any other value is ON — so the two Python spawn sites gate on the identical
    grammar as the Go spine. Pure: reads ``os.environ`` (or an injected mapping for
    tests) and touches no git."""
    env = os.environ if environ is None else environ
    raw = env.get(WORKTREE_ENABLE_ENV)
    if raw is None:
        return False
    return str(raw).strip().lower() not in _ENABLE_OFF_VALUES


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


def _warm_gocache_enabled(env: dict[str, str]) -> bool:
    return env.get(WARM_GOCACHE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _seed_worker_gocache(source: Path, target: Path) -> None:
    """Best-effort warm seed: hardlink immutable cache entries, copy if needed.

    Go's build cache publishes content-addressed entries atomically.  Linking those
    existing entries gives a worker warm reads without sharing its writable cache
    namespace; later worker misses and cache maintenance stay under ``target``.
    Any racing/locked entry is skipped so cache warming can never wedge dispatch.
    """
    if not source.is_dir() or source.resolve() == target.resolve():
        return

    def link_or_copy(src: str, dst: str) -> str:
        # Only two-hex shard entries are immutable cache objects.  Metadata such
        # as trim.txt remains a private copy because Go may rewrite it in place.
        immutable_entry = bool(re.fullmatch(r"[0-9a-f]{2}", Path(src).parent.name))
        if immutable_entry:
            try:
                os.link(src, dst)
                return dst
            except OSError:
                pass
        return shutil.copy2(src, dst)

    try:
        shutil.copytree(source, target, dirs_exist_ok=True, copy_function=link_or_copy)
    except OSError:
        # A live Go cache may trim entries while it is being walked.  A partial seed
        # is still useful; the worker's ordinary cache miss path fills the rest.
        pass


def worktree_env(base_env: dict[str, str], wt_dir: Path) -> dict[str, str]:
    """The child env that isolates a worker's BUILD to its own worktree, on top of
    whatever ``base_env`` the dispatcher already composed (``child_env`` +
    account pins).

    Pointing ``GOCACHE`` and ``GOTMPDIR`` INSIDE the worktree is what makes "a
    broken build in one worker's worktree does not red another's" true: each
    worker compiles into its own cache, so a half-built package can never poison a
    sibling's ``go build`` / ``make ci``.  When
    ``FAK_DISPATCH_WORKTREE_WARM_GOCACHE=1``, immutable entries from the incoming
    ``GOCACHE`` (or Go's default cache) are best-effort seeded into that private
    cache before launch; misses and all later writes remain private.
    ``DISPATCH_WORKSPACE`` is repointed at the
    worktree so a worker that reads it (the self-describing dispatch contract)
    operates on its isolated tree, not the shared trunk."""
    env = dict(base_env)
    worker_cache = Path(wt_dir) / ".gocache"
    if _warm_gocache_enabled(env):
        source = env.get("GOCACHE", "").strip()
        if not source:
            try:
                source = subprocess.run(
                    ["go", "env", "GOCACHE"], cwd=wt_dir.parent,
                    env=env, capture_output=True, text=True, check=True,
                    creationflags=no_window_creationflags(),
                ).stdout.strip()
            except (OSError, subprocess.SubprocessError):
                source = ""
        if source:
            _seed_worker_gocache(Path(source), worker_cache)
    wt = str(wt_dir)
    env["DISPATCH_WORKSPACE"] = wt
    env["FLEET_WORKER_WORKTREE"] = wt
    env["GOCACHE"] = str(worker_cache)
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


def isolate_spawn(root: Path, lane: str, key: str, cwd: str | Path,
                  env: dict[str, str], *, base_sha: str | None = None,
                  enabled: bool | None = None, wt_root: Path | None = None,
                  git: GitRunner | None = None) -> "tuple[str, dict[str, str], dict[str, Any]]":
    """Compose the ``(cwd, env)`` a dispatch worker should spawn under, applying #3181
    per-worker worktree isolation when the gate is on — the Python twin of the
    ``cmd/fak/dispatch_tick.go`` #3168 block, so all three spawn sites share ONE
    contract.

    Returns ``(spawn_cwd, spawn_env, info)``:

      * Isolation OFF (default) or ANY worktree step faults -> the caller's original
        ``(cwd, env)`` UNCHANGED. A worktree-layer fault therefore never wedges a spawn:
        the worker runs in the shared trunk exactly as before, so the flag-OFF path is
        byte-identical to today (the fail-open contract the Go spine already keeps).
      * Isolation ON and :func:`prepare_worker_worktree` succeeds -> the worker's own
        detached worktree path as ``cwd`` and :func:`worktree_env` layered onto ``env``
        (GOCACHE/GOTMPDIR redirected inside the worktree so a broken build there can
        never red a sibling).

    ``info`` carries ``enabled`` plus, on success, ``worktree``/``base_sha`` (the
    land+reap hook the witness sweep reads off the spawner's ``.worktree`` sidecar) or,
    on a fault, the ``failopen`` reason. ``git`` is the injectable runner so the whole
    decision is unit-testable without touching real git."""
    on = worktree_isolation_enabled() if enabled is None else enabled
    info: dict[str, Any] = {"enabled": bool(on), "worktree": None}
    if not on:
        return str(cwd), env, info
    res = prepare_worker_worktree(root, lane, str(key), base_sha=base_sha,
                                  wt_root=wt_root, git=git)
    if res.get("ok") and res.get("path"):
        wt = str(res["path"])
        info.update(worktree=wt, base_sha=res.get("base_sha"),
                    reused=bool(res.get("reused")))
        return wt, worktree_env(env, Path(wt)), info
    info["failopen"] = res.get("reason")
    return str(cwd), env, info


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
                       base_sha: str | None = None,
                       paths: Sequence[str] | None = None,
                       verify: "Callable[[Path], dict[str, Any]] | None" = None,
                       git: GitRunner | None = None) -> dict[str, Any]:
    """Land a worker's edits from its isolated worktree onto the TRUNK as one
    stamped, signed-off commit ON ``main`` — the serialized commit-to-trunk step
    that keeps ``OFF_TRUNK`` from ever tripping (#1334).

    The worker edited in ``wt_path`` (a detached worktree); this captures that
    worktree's FULL delta since its pinned base and applies it to the trunk
    worktree, then commits ``-s`` by explicit path with the worker's prepared
    message. Because the commit happens IN the trunk worktree on ``main``, it is a
    normal guarded commit — never an off-trunk one. The CALLER holds the lane
    lease, which serialises this so two workers never apply to the same leaf tree
    at once.

    ``base_sha`` (the SHA :func:`prepare_worker_worktree` pinned the worktree to)
    is the diff base. Diffing against the base — not ``HEAD`` — is load-bearing: a
    guarded worker follows the repo's standing "commit when green" instruction and
    will ``git commit`` INSIDE its detached worktree, which moves the worktree HEAD
    forward, so ``git diff HEAD`` would be EMPTY and the worker's whole change would
    silently evaporate. ``git diff <base_sha>`` captures the delta whether the worker
    committed it, staged it, or left it unstaged. When ``base_sha`` is None it falls
    back to ``HEAD`` (the pre-#1333 behavior, correct only for a worker that never
    commits in-worktree).

    ``verify`` is a build/adjudication WITNESS run IN THE ISOLATED WORKTREE before
    anything is applied to the trunk (``land_worktree_diff`` itself performs a
    mechanical apply+commit with no build, so without this an edit that breaks the
    build would land on ``main``). It receives the worktree Path and returns
    ``{"ok": bool, "detail": str}``; a non-ok result REFUSES the land (nothing is
    applied or committed). It is injectable so the primitive stays testable; the
    dispatcher supplies a real ``go build`` / ``make ci`` runner. When None the
    witness is skipped (the caller's downstream ``dos commit-audit`` is the only arm).

    Returns ``{"ok", "applied", "committed", ...}``. FAIL-OPEN on git errors: any
    git error is reported, never raised. ``paths`` (when given) scopes the trunk
    commit to the worker's declared file region — never an ``add -A`` of the tree.

    EXCEPT the apply-reject arm (#3207): a diff that no longer applies because the
    trunk moved inside the worker's region since ``base_sha`` is NOT a generic git
    hiccup to fail open on — failing open silently drops the worker's entire
    (already-verified) delta. That arm returns a STRUCTURED REFUSAL —
    ``{"reason": "COLLISION_RISK", "detail": <git evidence>, "next_action": ...}``
    — where ``reason`` is a recognized token from the closed refusal vocabulary
    (``dos check-reason COLLISION_RISK``: refusal, route-to-replan), so the
    caller's loop replans (re-pin the worktree onto current trunk HEAD, re-verify
    there, re-land) instead of losing the work. Contract and the chosen
    serialized-apply-with-auto-rebase-on-reject reconcile algorithm:
    ``docs/notes/WORKTREE-LAND-MERGE-RECONCILIATION-2026-07-10.md``.

    >>> def fake_git(root, args):
    ...     if args and args[0] == "diff":
    ...         return 0, "diff --git a/x b/x\\n@@\\n-old\\n+new\\n"
    ...     return 1, "error: patch does not apply"
    >>> r = land_worktree_diff(Path("."), "wt", commit_msg_file="m", git=fake_git)
    >>> (r["ok"], r["committed"], r["reason"])
    (False, False, 'COLLISION_RISK')
    >>> "patch does not apply" in r["detail"]
    True"""
    run = git or _git
    wt = str(wt_path)
    # Capture the worker's full delta since the pinned base (committed + staged +
    # unstaged), not just uncommitted (git diff HEAD misses in-worktree commits).
    diff_ref = base_sha or "HEAD"
    rc, diff = run(Path(wt), ["diff", diff_ref])
    if rc != 0:
        return {"ok": False, "applied": False, "committed": False,
                "reason": f"could not read worktree diff vs {diff_ref} (rc {rc}) — fail open"}
    if not diff.strip():
        # No net change since the base: the worker landed nothing. The caller's
        # commit-witness (dos commit-audit) decides whether the slot was productive.
        return {"ok": True, "applied": False, "committed": False,
                "reason": f"no net diff in worktree vs {diff_ref} to land"}
    # Build/adjudication witness IN THE WORKTREE before touching the trunk: an edit
    # that reds the build must never land on main. Refuse the land on a failed witness.
    if verify is not None:
        vres = verify(Path(wt))
        if not vres.get("ok"):
            return {"ok": False, "applied": False, "committed": False,
                    "reason": f"worktree verify failed, refusing to land: {vres.get('detail')}",
                    "verify": vres}
    # Apply the captured diff to the trunk worktree's working tree. A plain
    # `git apply` is all-or-nothing: on a reject NOTHING was applied and the
    # trunk stays clean — but the worker's whole verified delta is at stake.
    proc = _git_apply(root, diff, git_run=run)
    if not proc.get("ok"):
        # STRUCTURED REFUSAL, not a fail-open drop (#3207). A rejected apply
        # means the trunk moved inside the worker's diff region since base_sha —
        # a MATERIALIZED lease collision (`dos check-reason COLLISION_RISK`:
        # known, refusal, route-to-replan). The LANDING SESSION owns the
        # conflict: keep the worktree (never reap on refusal — the preserved
        # work is the evidence), re-pin it onto current trunk HEAD, re-run the
        # verify witness there, and re-land; on a genuine overlapping-region
        # conflict STOP and replan the region. Never a human merge queue, never
        # a silent drop. Contract + algorithm comparison:
        # docs/notes/WORKTREE-LAND-MERGE-RECONCILIATION-2026-07-10.md
        return {"ok": False, "applied": False, "committed": False,
                "reason": "COLLISION_RISK",
                "detail": f"git apply to trunk rejected (diff base {diff_ref}): "
                          f"{proc.get('detail')}",
                "next_action": ("replan: re-pin the worktree onto current trunk "
                                "HEAD, re-verify, re-land; STOP on a genuine "
                                "conflict")}
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


# --------------------------------------------------------------------------- #
# CLI — the RUNNABLE safe self-modify path (#1334 follow-on).
#
# Everything above is a library the dispatcher imports; nothing could RUN it, so
# the worktree-isolated escape that the SELF_MODIFY_HOLD refusals point guarded
# workers and self-source tickets at ("route it to a worktree-isolated path")
# was named everywhere and invokable nowhere. This entrypoint closes that gap: a
# guarded fak instance, an operator, or a ticket-runner can now drive the full
# acquire -> edit -> land -> reap cycle from the shell —
#
#     wt=$(python tools/worker_worktree.py --root . prepare --lane cmd --key 1338 | jq -r .path)
#     # ...edit the self-source files INSIDE $wt (its own detached worktree)...
#     python tools/worker_worktree.py --root . land --worktree "$wt" \
#         --msg-file msg.txt --path cmd/fak/foo.go
#     python tools/worker_worktree.py --root . reap --worktree "$wt"
#
# Each subcommand prints exactly one JSON object to stdout and exits 0 when the
# underlying primitive returned ok, non-zero otherwise, so a caller branches on
# the exit code without parsing. The primitives stay fail-open; the CLI only
# maps args -> primitive -> JSON, adding no new failure mode of its own.
# --------------------------------------------------------------------------- #

def _go_build_verify(wt: Path) -> dict[str, Any]:
    """Build/adjudication witness for :func:`land_worktree_diff`: run ``go build
    ./...`` IN the isolated worktree so a worker's edit that reds the build is
    refused before it can land on the trunk. The worktree's :func:`worktree_env`
    already redirects GOCACHE/GOTMPDIR inside it, so this build never poisons a
    sibling. A build tool that is absent (no ``go`` on PATH) fails OPEN (ok=True):
    the witness must not block landing on a host that cannot run it — the caller's
    downstream ``dos commit-audit`` remains the backstop."""
    try:
        proc = subprocess.run(
            ["go", "build", "./..."], cwd=str(wt),
            env={**os.environ, **worktree_env({}, wt)},
            capture_output=True, text=True,
            creationflags=_no_window_creationflags())
    except FileNotFoundError:
        return {"ok": True, "detail": "go not found — witness skipped (fail open)"}
    except OSError as exc:
        return {"ok": True, "detail": f"could not run go build ({exc}) — fail open"}
    if proc.returncode == 0:
        return {"ok": True, "detail": "go build ./... clean"}
    return {"ok": False, "detail": (proc.stderr or proc.stdout).strip()[-500:]}


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="worker_worktree",
        description="Runnable safe self-modify path (#1334): prepare an isolated "
                    "detached worktree at trunk HEAD, edit in it, land the diff "
                    "onto the trunk as one stamped signed-off commit, then reap.")
    parser.add_argument("--root", default=".",
                        help="repo root the worktree/commit operate on (default: cwd). "
                             "Give it BEFORE the subcommand.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_prep = sub.add_parser(
        "prepare", help="create one isolated detached worktree pinned at trunk HEAD")
    p_prep.add_argument("--lane", required=True,
                        help="lane name (labels the worktree dir)")
    p_prep.add_argument("--key", required=True,
                        help="worker key (issue number, wave id, pid) — hashed into the dir name")
    p_prep.add_argument("--base-sha", default="",
                        help="pin to this sha instead of resolving trunk HEAD")
    p_prep.add_argument("--wt-root", default="",
                        help="parent dir for the worktree (default: the scratch root)")

    p_land = sub.add_parser(
        "land", help="apply the worktree's diff onto the trunk and commit -s by path")
    p_land.add_argument("--worktree", required=True,
                        help="the prepared worktree path to land from")
    p_land.add_argument("--msg-file", required=True,
                        help="commit message file (git commit -s -F)")
    p_land.add_argument("--base-sha", default="",
                        help="the sha the worktree was pinned to (diff base). Required to "
                             "capture a worker that committed IN-worktree; without it the "
                             "diff is taken against HEAD and in-worktree commits are missed")
    p_land.add_argument("--verify", choices=["off", "go-build"], default="off",
                        help="build/adjudication witness run IN THE WORKTREE before landing; "
                             "'go-build' refuses the land if `go build ./...` reds (default off)")
    p_land.add_argument("--path", action="append", default=[], dest="paths",
                        help="scope the trunk commit to this path (repeatable); "
                             "omit to commit the whole applied diff")

    p_reap = sub.add_parser(
        "reap", help="force-remove a worker worktree after its change has landed")
    p_reap.add_argument("--worktree", required=True,
                        help="the worktree path to remove (must carry the worker marker)")

    sub.add_parser("list", help="enumerate live per-worker worktrees from git")

    args = parser.parse_args(list(argv) if argv is not None else None)
    root = Path(args.root)

    if args.cmd == "prepare":
        res = prepare_worker_worktree(
            root, args.lane, args.key,
            base_sha=args.base_sha or None,
            wt_root=Path(args.wt_root) if args.wt_root else None)
        # Hand the caller the build-isolating env additions so it can export them
        # for the edit step without re-deriving worktree_env itself.
        if res.get("ok") and res.get("path"):
            res["env"] = worktree_env({}, Path(res["path"]))
        ok = bool(res.get("ok"))
    elif args.cmd == "land":
        res = land_worktree_diff(
            root, args.worktree,
            commit_msg_file=args.msg_file,
            base_sha=args.base_sha or None,
            paths=args.paths or None,
            verify=_go_build_verify if args.verify == "go-build" else None)
        ok = bool(res.get("ok"))
    elif args.cmd == "reap":
        res = reap_worker_worktree(root, args.worktree)
        ok = bool(res.get("ok"))
    else:  # list
        res = count_worker_worktrees(root)
        ok = "error" not in res

    print(json.dumps(res, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
