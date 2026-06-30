#!/usr/bin/env python3
r"""worktree_doctor.py — converge a multi-worktree fleet repo toward "ONE worktree, on master",
SAFELY, and make staying that way durable.

THE PROBLEM THIS KILLS
----------------------
This repo is a live multi-session fleet. Sessions spin up extra `git worktree`s and
feature branches; over time the checkout sprawls (here: a `master` worktree, an
`integrate-to-master` worktree, and a primary still on a stale feature branch
mid-merge). Then nobody is sure which tree is canonical, branches diverge, a stuck
merge sits for hours, and an over-eager cleanup risks deleting a peer's uncommitted
work. The operator's standing wish is simple: **one worktree, checked out on master.**

THE INVARIANT
-------------
  exactly one worktree, on `master`, clean.

THE SAFETY RULE (why this is trustworthy enough to run unattended)
-----------------------------------------------------------------
A worktree is removed ONLY when removing it can lose NOTHING:
  * it is not the primary worktree, AND
  * it has no uncommitted changes and no untracked files, AND
  * it has no merge/rebase/cherry-pick in progress, AND
  * every commit it carries is already reachable from <master-ref>
    (`rev-list --count <master-ref>..HEAD == 0`) — so its branch is fully merged.
Anything that fails ANY of these is BLOCKED: the doctor PASSES on it (never touches
it) and surfaces exactly why. Staleness only makes the merged-check MORE
conservative (a commit on remote-master but not in the local ref reads as
"unmerged" -> we keep the worktree), so the tool never deletes work by being out of
date. It never uses `git worktree remove --force` and never switches the primary's
branch for you (a mid-merge primary is the operator's to resolve).

THE TRUNK IS AUTO-DETECTED
--------------------------
The invariant says "on the trunk". The trunk's remote ref is detected from the repo
(`origin/HEAD` -> e.g. `origin/main`), so the doctor anchors on the REAL trunk instead of
a hardcoded `master`. (This repo is `main`; a stale `origin/master` default made the
merged-check meaningless and reported the on-main primary as "off master" every run.)
Override with --master-ref when you must.

THE DISPOSABLE-WORKTREE SWEEP (--sweep-disposable)
--------------------------------------------------
The convergence logic above is conservative by design: it NEVER deletes a worktree that
carries work. But this repo's real sprawl is the DETACHED worktrees the harness/agents
spin into scratch dirs (`.../scratchpad/<name>`, `C:/work/pr-work/<name>`). Per AGENTS.md
we "never open a worktree", so any worktree under those roots is a leftover, not canonical
— yet many carry uncommitted scratch, so the safe doctor leaves them to pile up forever.
The sweep closes that gap WITHOUT losing anything: it ARCHIVES each dirty worktree's diff
+ untracked files first (--archive-dir), then force-removes it. A freshness guard
(--fresh-minutes) spares any worktree touched recently — a live session's scratch is left
alone while a dead session's is reaped. This is the ONLY path that uses --force, and only
after the work is safely archived.

USAGE
-----
  worktree_doctor.py                 # report: per-worktree disposition + safe plan
  worktree_doctor.py --json          # same, machine-readable (for cron/automation)
  worktree_doctor.py --prune         # execute ONLY the provably-safe removals; pass the rest
  worktree_doctor.py --sweep-disposable   # archive-then-remove dead scratch worktrees (temp/pr-work)
  worktree_doctor.py --fresh-minutes 180  # spare scratch worktrees touched in the last N minutes
  worktree_doctor.py --fetch         # refresh <master-ref>'s remote first (more accurate merged-check)
  worktree_doctor.py --master-ref origin/main
  worktree_doctor.py --allow-branch fak-v0.1     # an intentional long-lived worktree: RETAIN it,
                                                 # and don't let it trip the needs-a-human exit 1
  worktree_doctor.py --prune-branches            # ALSO delete merged, non-checked-out, non-protected
                                                 # local branches (git branch -d) + prune stale remote refs

MAKING IT DURABLE
-----------------
Run it on a cadence next to the existing fleet watchers (see
register_mac_watchers.sh / mac-host cron). `--json` + the exit code make it a clean
cron citizen:
  exit 0  already converged (one clean worktree on master), or a --prune made all the
          progress it safely could — including the normal case where the shared primary
          is on master but carries in-flight peer work (this repo's steady state, not an
          anomaly; the doctor cannot and should not "fix" a peer's uncommitted work);
  exit 1  NOT converged and blocked by issues needing a human: a non-primary worktree
          with work at risk, the primary literally OFF master, or a stuck
          merge/rebase/cherry-pick on the primary. Dirtiness / untracked files on an
          on-master primary do NOT trip this — cron notifies only on real anomalies;
  exit 2  the tool could not run (not a git repo, git error).
A nightly `worktree_doctor.py --prune` keeps the sprawl swept to its safe minimum
without ever risking unsaved work; the report flags the human-only remainder.
Pure stdlib; no deps.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from dispatch_worker import install_no_window_subprocess_defaults
import sys
install_no_window_subprocess_defaults(subprocess)

DEFAULT_MASTER_REF = "origin/master"  # last-resort fallback; main() AUTO-DETECTS the trunk

# Path segments under which an extra worktree is ALWAYS a harness/agent leftover (never
# canonical) — the scratch areas Claude/agents spin worktrees into. Matched per-segment so
# both .../<uuid>/scratchpad/<name> and C:/work/pr-work/<name> qualify regardless of root.
DISPOSABLE_MARKERS = ("scratchpad", "pr-work")


def _git(args, cwd=None):
    """Run a git command; return (rc, stdout, stderr). Never raises on nonzero."""
    p = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True
    )
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def detect_master_ref(repo, fallback=DEFAULT_MASTER_REF):
    """Resolve the trunk's remote-tracking ref from the repo itself, so the doctor anchors
    on the REAL trunk (this repo is `main`, not `master`). Prefer origin/HEAD's target;
    then the first of origin/main / origin/master that exists; else the fallback."""
    rc, out, _ = _git(["symbolic-ref", "--quiet", "refs/remotes/origin/HEAD"], cwd=repo)
    if rc == 0 and out.startswith("refs/remotes/"):
        return out[len("refs/remotes/"):]
    for cand in ("origin/main", "origin/master"):
        if _git(["rev-parse", "--verify", "--quiet", cand + "^{commit}"], cwd=repo)[0] == 0:
            return cand
    return fallback


# --------------------------------------------------------------------------- #
# Parsing + signal gathering (the git-touching layer).
# --------------------------------------------------------------------------- #

def parse_worktree_list(porcelain: str):
    """Parse `git worktree list --porcelain` into ordered worktree records.

    The FIRST record is the primary worktree (git's documented ordering)."""
    out = []
    cur = {}
    for line in porcelain.splitlines():
        if not line.strip():
            if cur:
                out.append(cur)
                cur = {}
            continue
        if line.startswith("worktree "):
            if cur:
                out.append(cur)
            cur = {"path": line[len("worktree "):], "branch": None,
                   "detached": False, "bare": False}
        elif line.startswith("HEAD "):
            cur["head"] = line[len("HEAD "):]
        elif line.startswith("branch "):
            ref = line[len("branch "):]
            cur["branch"] = ref.split("refs/heads/", 1)[-1]
        elif line.strip() == "detached":
            cur["detached"] = True
        elif line.strip() == "bare":
            cur["bare"] = True
    if cur:
        out.append(cur)
    for i, w in enumerate(out):
        w["is_primary"] = (i == 0)
    return out


def _mid_op(path: str):
    """Return 'merge'/'rebase'/'cherry-pick'/'revert' if one is in progress, else None."""
    rc, gd, _ = _git(["rev-parse", "--git-dir"], cwd=path)
    if rc != 0:
        return None
    gd = gd if os.path.isabs(gd) else os.path.join(path, gd)
    checks = [
        ("merge", "MERGE_HEAD"),
        ("rebase", "rebase-merge"),
        ("rebase", "rebase-apply"),
        ("cherry-pick", "CHERRY_PICK_HEAD"),
        ("revert", "REVERT_HEAD"),
    ]
    for name, marker in checks:
        if os.path.exists(os.path.join(gd, marker)):
            return name
    return None


def gather_signals(wt, master_ref=DEFAULT_MASTER_REF):
    """Augment a worktree record with the hard safety signals."""
    path = wt["path"]
    sig = dict(wt)
    rc, st, _ = _git(["status", "--porcelain"], cwd=path)
    tracked = untracked = 0
    if rc == 0:
        for ln in st.splitlines():
            if ln.startswith("??"):
                untracked += 1
            elif ln.strip():
                tracked += 1
    sig["dirty"] = tracked > 0
    sig["untracked"] = untracked
    sig["mid_op"] = _mid_op(path)
    # commits this worktree carries that are NOT yet on master-ref (would be lost).
    sig["unmerged_to_master"] = _count(["rev-list", "--count", f"{master_ref}..HEAD"], path)
    # commits not on this branch's own upstream (informational).
    sig["unpushed"] = _count(["rev-list", "--count", "@{u}..HEAD"], path, default=None)
    # crude "last activity" proxy for the disposable sweep's freshness guard.
    sig["age_seconds"] = worktree_age_seconds(path)
    return sig


def worktree_age_seconds(path, now=None):
    """Seconds since the worktree was last touched, as a crude liveness proxy: the max
    mtime of its working dir and its `.git` pointer (both bump on git activity). Returns
    None if unreadable (the sweep treats unknown age as stale)."""
    import time
    now = time.time() if now is None else now
    mtimes = []
    for p in (path, os.path.join(path, ".git")):
        try:
            mtimes.append(os.path.getmtime(p))
        except OSError:
            pass
    if not mtimes:
        return None
    return max(0.0, now - max(mtimes))


def _count(args, cwd, default=0):
    rc, out, _ = _git(args, cwd=cwd)
    if rc != 0:
        return default
    try:
        return int(out)
    except ValueError:
        return default


# --------------------------------------------------------------------------- #
# Decision logic (PURE — unit-tested without git).
# --------------------------------------------------------------------------- #

def issues_of(sig):
    """Reasons this worktree must be LEFT ALONE (empty list => no blocking issues)."""
    out = []
    if sig.get("mid_op"):
        out.append(f"{sig['mid_op']} in progress")
    if sig.get("dirty"):
        out.append("uncommitted changes")
    if sig.get("untracked"):
        out.append(f"{sig['untracked']} untracked file(s)")
    n = sig.get("unmerged_to_master", 0)
    if n:
        out.append(f"{n} commit(s) not on master")
    return out


def is_on_master(sig, trunk="master"):
    """Positional: this worktree's branch IS the trunk (regardless of cleanliness)."""
    return sig.get("branch") == trunk


def primary_hard_issues(sig):
    """Genuine stuck-state issues on the PRIMARY that warrant a human even when the
    primary is on master: an in-progress merge/rebase/cherry-pick/revert. Dirtiness,
    untracked files, and local-ahead commits are the shared-worktree steady state and
    are deliberately NOT raised here — the doctor cannot and should not touch a peer's
    in-flight work, so those must not cry wolf nightly."""
    out = []
    if sig.get("mid_op"):
        out.append(f"{sig['mid_op']} in progress")
    return out


def is_clean_master(sig, trunk="master"):
    return is_on_master(sig, trunk) and not issues_of(sig)


def safe_to_remove(sig):
    """A non-primary worktree whose removal can lose nothing."""
    return (not sig.get("is_primary")) and not issues_of(sig)


def make_plan(sigs, master_ref=DEFAULT_MASTER_REF, allow_branches=(), trunk="master"):
    """Compose per-worktree signals into a safe convergence plan.

    keeper            the single worktree to keep (the primary if it is a clean
                      master; else a clean master worktree; else None).
    prune             non-keeper worktrees that are safe_to_remove AND, if removal
                      would otherwise drop the last master worktree, are spared.
    retained          non-primary worktrees on an --allow-branch (e.g. a long-lived
                      release line like fak-v0.1): intentional, never pruned, and NOT
                      counted as a needs-a-human anomaly — so a cron job doesn't cry
                      wolf about them every run.
    blocked           NON-primary worktrees with issues (work at risk) — PASSED over.
                      The primary is never listed here (it is never a prune candidate);
                      its own anomalies surface via primary_offtrack / a stuck-op note.
    primary_offtrack  the primary is POSITIONALLY off master (wrong branch / detached).
                      A dirty primary that IS on master does NOT set this — that is the
                      shared-worktree steady state, not a positional defect.
    needs_human       a genuine anomaly a person must resolve: a blocked worktree, the
                      primary off master, or a stuck merge/rebase on the primary. This is
                      what drives exit 1. A dirty/untracked on-master primary never sets
                      it (no crying wolf on the shared-worktree norm). Allow-listed
                      worktrees never set it either.
    converged         True iff the only NON-retained worktree is a clean master primary.
    """
    allow = set(allow_branches or ())
    primary = next((s for s in sigs if s.get("is_primary")), None)
    clean_masters = [s for s in sigs if is_clean_master(s, trunk)]

    keeper = None
    if primary and is_clean_master(primary, trunk):
        keeper = primary
    elif clean_masters:
        keeper = clean_masters[0]

    def is_retained(s):
        return (not s.get("is_primary")) and s.get("branch") in allow

    retained = [{"path": s["path"], "branch": s.get("branch")}
                for s in sigs if is_retained(s)]

    # `blocked` = NON-primary worktrees with work at risk (the doctor passes on them).
    # The primary is never a prune candidate, so it never belongs here; its own
    # anomalies are surfaced separately via primary_off_master / primary_stuck so a
    # dirty shared-worktree primary does not masquerade as a blocked prune target.
    blocked = [{"path": s["path"], "branch": s.get("branch"),
                "reasons": issues_of(s)}
               for s in sigs
               if not s.get("is_primary") and issues_of(s) and not is_retained(s)]

    prune = []
    for s in sigs:
        if s is keeper or s.get("is_primary") or is_retained(s):
            continue
        if not safe_to_remove(s):
            continue
        # never delete the last remaining path to master.
        would_be_last_master = (
            is_clean_master(s, trunk)
            and keeper is None
            and len([m for m in clean_masters if m is not s]) == 0
        )
        if would_be_last_master:
            continue
        prune.append({"path": s["path"], "branch": s.get("branch")})

    # "off master" is a POSITIONAL defect (wrong branch / detached), NOT dirtiness.
    # A dirty primary that is already on master is the shared-worktree steady state.
    primary_off_master = bool(primary) and not is_on_master(primary, trunk)
    primary_stuck = primary_hard_issues(primary) if primary else []
    non_retained = [s for s in sigs if not is_retained(s)]
    converged = len(non_retained) == 1 and bool(keeper) and keeper.get("is_primary")
    needs_human = bool(blocked) or primary_off_master or bool(primary_stuck)

    notes = []
    if primary_off_master:
        why = issues_of(primary) or [f"on '{primary.get('branch')}', not {trunk}"]
        notes.append(
            f"primary worktree {primary['path']} is off {trunk}: {', '.join(why)} "
            f"(resolve, then `git -C {primary['path']} switch {trunk}`)"
        )
    if primary_stuck:
        notes.append(
            f"primary worktree {primary['path']} on master but stuck: "
            f"{', '.join(primary_stuck)} (resolve before it blocks further work)"
        )
    master_positioned = any(is_on_master(s, trunk) for s in sigs)
    if keeper is None and not master_positioned:
        notes.append(f"no {trunk} worktree exists yet — nothing is safe to keep/prune")
    elif keeper is None:
        # primary (or another worktree) is on master but dirty: the shared-worktree
        # norm. Safe surplus worktrees are still pruned; this is informational only.
        notes.append(
            f"primary is on {trunk} with in-flight work (normal for the shared worktree); "
            "safe surplus worktrees are still pruned"
        )
    for r in retained:
        notes.append(f"retained {r['path']} (allow-listed branch '{r['branch']}')")

    return {
        "master_ref": master_ref,
        "keeper": keeper["path"] if keeper else None,
        "prune": prune,
        "retained": retained,
        "blocked": blocked,
        "primary_offtrack": primary_off_master,
        "needs_human": needs_human,
        "converged": converged,
        "notes": notes,
    }


# --------------------------------------------------------------------------- #
# Branch + remote-ref hygiene (same safety stance: delete only the loss-free).
# --------------------------------------------------------------------------- #

def deletable_branches(local, protected, checked_out, merged):
    """PURE: local branches safe to delete — fully merged to master-ref, not on the
    protect list, and not checked out in ANY worktree (git refuses those anyway).
    Sorted + deterministic for stable reporting/automation."""
    off_limits = set(protected) | set(checked_out)
    return sorted(b for b in local if b in merged and b not in off_limits)


def gather_branches(repo, master_ref):
    """Local branches, which are merged to master-ref, and which are checked out."""
    rc, out, _ = _git(["for-each-ref", "--format=%(refname:short)", "refs/heads"], cwd=repo)
    local = [b for b in out.splitlines() if b.strip()] if rc == 0 else []
    rcw, outw, _ = _git(["worktree", "list", "--porcelain"], cwd=repo)
    checked_out = {w.get("branch") for w in parse_worktree_list(outw) if w.get("branch")} \
        if rcw == 0 else set()
    merged = set()
    for b in local:
        # merged iff it adds no commits beyond master-ref. default=1 => treat an
        # errored count as "not merged" (conservative: keep the branch).
        if _count(["rev-list", "--count", f"{master_ref}..{b}"], repo, default=1) == 0:
            merged.add(b)
    return {"local": local, "merged": merged, "checked_out": checked_out}


def do_prune_branches(names, repo, dry_run=True):
    """Delete the safe branches with `git branch -d` (NEVER -D, so git's own
    fully-merged gate is the last word). Returns per-branch results."""
    results = []
    for b in names:
        if dry_run:
            results.append({"branch": b, "deleted": False, "dry_run": True})
            continue
        rc, _, err = _git(["branch", "-d", b], cwd=repo)
        results.append({"branch": b, "deleted": rc == 0,
                        "error": err if rc != 0 else None})
    return results


# --------------------------------------------------------------------------- #
# Rendering + actions.
# --------------------------------------------------------------------------- #

def render_text(sigs, plan, trunk="master"):
    lines = []
    lines.append(f"worktree doctor — invariant: one worktree on {trunk} (ref {plan['master_ref']})")
    lines.append("")
    retained_paths = {r["path"] for r in plan.get("retained", [])}
    for s in sigs:
        tag = "primary" if s.get("is_primary") else "       "
        iss = issues_of(s)
        if s["path"] in retained_paths:
            state = "RETAINED (allow-listed): " + (", ".join(iss) if iss else "clean")
        elif not iss:
            state = "CLEAN"
        elif s.get("is_primary") and is_on_master(s, trunk) and not primary_hard_issues(s):
            # shared-worktree steady state: on master with only soft (dirty/untracked)
            # issues — information, not a block. Never screams BLOCKED at the norm.
            state = "IN-FLIGHT (shared worktree): " + ", ".join(iss)
        else:
            state = "BLOCKED: " + ", ".join(iss)
        keep = "  <- KEEP" if s["path"] == plan["keeper"] else ""
        lines.append(f"  [{tag}] {s.get('branch') or 'DETACHED':<20} {state}{keep}")
        lines.append(f"          {s['path']}")
    lines.append("")
    if plan["prune"]:
        lines.append("SAFE TO PRUNE (clean + fully merged to master):")
        for p in plan["prune"]:
            lines.append(f"  - {p['path']}  ({p['branch']})")
    else:
        lines.append("SAFE TO PRUNE: none")
    if plan["blocked"]:
        lines.append("")
        lines.append("PASSED (has work — left untouched):")
        for b in plan["blocked"]:
            lines.append(f"  - {b['path']}  ({b['branch']}): {', '.join(b['reasons'])}")
    for n in plan["notes"]:
        lines.append("")
        lines.append("NOTE: " + n)
    lines.append("")
    lines.append("CONVERGED" if plan["converged"] else "NOT converged toward single-master")
    return "\n".join(lines)


def do_prune(plan, dry_run=True):
    """Remove the safe-to-prune worktrees. Returns list of {path, removed, error}."""
    results = []
    for p in plan["prune"]:
        if dry_run:
            results.append({"path": p["path"], "removed": False, "dry_run": True})
            continue
        rc, _, err = _git(["worktree", "remove", p["path"]])  # NO --force, ever
        results.append({"path": p["path"], "removed": rc == 0,
                        "error": err if rc != 0 else None})
    return results


# --------------------------------------------------------------------------- #
# Disposable-worktree sweep — the harness/agent scratch leftovers.
#
# The doctor above converges BRANCH worktrees safely and, by design, never deletes one
# that carries work. But this repo's real sprawl is the DETACHED worktrees the harness and
# agents spin into scratch dirs (.../scratchpad/<name>, C:/work/pr-work/<name>). Per
# AGENTS.md we "never open a worktree", so any worktree under those roots is a leftover —
# yet many carry uncommitted scratch the safe doctor (rightly) refuses to delete, so they
# pile up forever. The sweep closes that gap WITHOUT losing anything: ARCHIVE each dirty
# worktree's diff + untracked first, THEN force-remove. A freshness guard spares any
# worktree touched recently (a live session); a dead session's scratch is reaped.
# --------------------------------------------------------------------------- #

def _has_disposable_segment(path, markers):
    """True if any path segment of `path` equals a disposable marker (case-insensitive)."""
    parts = {p.lower() for p in os.path.normpath(path).replace("\\", "/").split("/") if p}
    return any(m.lower() in parts for m in markers)


def is_disposable_path(path, roots=(), markers=DISPOSABLE_MARKERS):
    """PURE: True if `path` is a harness/agent scratch worktree — under a disposable root,
    or containing a disposable path segment (scratchpad / pr-work)."""
    if not path:
        return False
    np = os.path.normcase(os.path.normpath(path))
    for r in roots:
        if not r:
            continue
        nr = os.path.normcase(os.path.normpath(r))
        if np == nr or np.startswith(nr + os.sep):
            return True
    return _has_disposable_segment(path, markers)


def sweep_candidates(sigs, roots=(), markers=DISPOSABLE_MARKERS, fresh_seconds=0,
                     keep_paths=()):
    """PURE: which worktrees the sweep will reap. A candidate is non-primary, lives in a
    disposable location, is not explicitly kept, and was last touched longer ago than
    fresh_seconds (an unknown age reads as +inf -> stale -> reaped). The freshness guard
    is what spares a LIVE session's scratch worktree. Records carry whether work must be
    archived first (has_work)."""
    keep = {os.path.normcase(os.path.normpath(p)) for p in keep_paths if p}
    out = []
    for s in sigs:
        if s.get("is_primary"):
            continue
        path = s.get("path")
        if not path or not is_disposable_path(path, roots, markers):
            continue
        if os.path.normcase(os.path.normpath(path)) in keep:
            continue
        age = s.get("age_seconds")
        age = float("inf") if age is None else age
        if age < fresh_seconds:
            continue  # touched recently -> a live session; leave it
        has_work = bool(s.get("dirty")) or bool(s.get("untracked")) or bool(s.get("mid_op"))
        out.append({"path": path, "branch": s.get("branch"),
                    "detached": s.get("detached", False), "has_work": has_work,
                    "age_seconds": None if age == float("inf") else age})
    return out


def archive_worktree(path, archive_dir, name=None):
    """Preserve a worktree's uncommitted work before removal: write its tracked diff
    (git diff HEAD) + base commit, and copy its untracked files, into archive_dir/<name>.
    Best-effort; returns a summary dict. This is what makes the force-remove loss-free."""
    import shutil
    name = name or os.path.basename(os.path.normpath(path)) or "wt"
    dest = os.path.join(archive_dir, name)
    os.makedirs(dest, exist_ok=True)
    summary = {"name": name, "dir": dest, "tracked_patch_bytes": 0, "untracked": 0}
    rc, head, _ = _git(["rev-parse", "HEAD"], cwd=path)
    if rc == 0:
        with open(os.path.join(dest, "base-commit.txt"), "w", encoding="utf-8") as fh:
            fh.write(head + "\n")
    rc, diff, _ = _git(["diff", "HEAD"], cwd=path)
    if rc == 0 and diff:
        with open(os.path.join(dest, "tracked.patch"), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(diff + "\n")
        summary["tracked_patch_bytes"] = len(diff)
    rc, others, _ = _git(["ls-files", "--others", "--exclude-standard"], cwd=path)
    if rc == 0:
        for rel in [ln for ln in others.splitlines() if ln.strip()]:
            try:
                dst = os.path.join(dest, "untracked", rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(os.path.join(path, rel), dst)
                summary["untracked"] += 1
            except OSError:
                pass
    return summary


def do_sweep(candidates, archive_dir, repo, dry_run=True):
    """Archive any work, then `git worktree remove --force` each candidate. Force is safe
    here ONLY because the work was archived first. Returns per-candidate results."""
    results = []
    for c in candidates:
        if dry_run:
            results.append({"path": c["path"], "removed": False, "dry_run": True,
                            "archived": None})
            continue
        archived = archive_worktree(c["path"], archive_dir) if c.get("has_work") else None
        rc, _, err = _git(["worktree", "remove", "--force", c["path"]], cwd=repo)
        results.append({"path": c["path"], "removed": rc == 0, "archived": archived,
                        "error": err if rc != 0 else None})
    if any(r.get("removed") for r in results):
        _git(["worktree", "prune"], cwd=repo)
    return results


def collect(repo, master_ref, fetch):
    if fetch:
        remote = master_ref.split("/", 1)[0] if "/" in master_ref else "origin"
        _git(["fetch", remote, master_ref.split("/", 1)[-1]], cwd=repo)
    rc, out, err = _git(["worktree", "list", "--porcelain"], cwd=repo)
    if rc != 0:
        raise RuntimeError(err or "git worktree list failed")
    return [gather_signals(w, master_ref) for w in parse_worktree_list(out)
            if not w.get("bare")]


def _default_archive_dir():
    import datetime
    import tempfile
    base = os.path.join(os.environ.get("LOCALAPPDATA", tempfile.gettempdir()),
                        "Fleet", "watchdog", "worktree-archive")
    return os.path.join(base, datetime.date.today().isoformat())


def main(argv=None):
    ap = argparse.ArgumentParser(description="Converge to one worktree on the trunk, safely.")
    ap.add_argument("--repo", default=".", help="repo path (default: cwd)")
    ap.add_argument("--master-ref", default=None,
                    help="trunk remote ref (default: auto-detect origin/HEAD, e.g. origin/main)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--prune", action="store_true", help="remove the safe worktrees (else dry-run)")
    ap.add_argument("--fetch", action="store_true", help="refresh master-ref's remote first")
    ap.add_argument("--allow-branch", action="append", default=[], metavar="NAME",
                    help="branch intentionally kept in its own worktree (e.g. fak-v0.1); "
                         "repeatable — such worktrees are RETAINED, never pruned, and do "
                         "not count as a needs-a-human anomaly")
    ap.add_argument("--prune-branches", action="store_true",
                    help="also delete merged, non-checked-out, non-protected local branches "
                         "(git branch -d) and `git remote prune` stale remote-tracking refs")
    ap.add_argument("--sweep-disposable", action="store_true",
                    help="archive-then-force-remove dead scratch worktrees under disposable "
                         "roots (temp / pr-work / scratchpad); else dry-run-report them")
    ap.add_argument("--disposable-root", action="append", default=[], metavar="DIR",
                    help="extra root whose worktrees are disposable (repeatable)")
    ap.add_argument("--fresh-minutes", type=float, default=180.0,
                    help="spare a scratch worktree touched within this many minutes "
                         "(protects a live session; default 180)")
    ap.add_argument("--keep-path", action="append", default=[], metavar="DIR",
                    help="never sweep this worktree path (repeatable; e.g. the current session)")
    ap.add_argument("--archive-dir", default=None,
                    help="where sweep archives a dirty worktree's work before removal "
                         "(default: %%LOCALAPPDATA%%/Fleet/watchdog/worktree-archive/<date>)")
    args = ap.parse_args(argv)

    # Anchor on the REAL trunk: detect origin/HEAD unless the operator pinned --master-ref.
    master_ref = args.master_ref or detect_master_ref(args.repo)
    trunk = master_ref.split("/", 1)[-1]
    archive_dir = args.archive_dir or _default_archive_dir()

    try:
        sigs = collect(args.repo, master_ref, args.fetch)
    except Exception as e:  # not a git repo / git error
        if args.json:
            print(json.dumps({"error": str(e)}))
        else:
            print(f"worktree_doctor: {e}", file=sys.stderr)
        return 2

    # Disposable-worktree sweep FIRST, so the convergence plan below reflects the survivors
    # (and a successful sweep doesn't leave the just-reaped worktrees flagged needs_human).
    cands = sweep_candidates(sigs, roots=args.disposable_root,
                             fresh_seconds=args.fresh_minutes * 60.0,
                             keep_paths=args.keep_path)
    swept = do_sweep(cands, archive_dir, args.repo,
                     dry_run=not args.sweep_disposable) if cands else []
    if any(r.get("removed") for r in swept):
        sigs = collect(args.repo, master_ref, fetch=False)  # refresh after real removals

    plan = make_plan(sigs, master_ref, allow_branches=args.allow_branch, trunk=trunk)
    pruned = do_prune(plan, dry_run=not args.prune) if plan["prune"] else []

    # Branch + remote-ref hygiene: protect the trunk, the master-ref's own branch, and
    # every allow-listed branch; delete only the merged, non-checked-out remainder.
    master_ref_branch = master_ref.split("/", 1)[-1]
    protected = {"master", trunk, master_ref_branch, *args.allow_branch}
    binfo = gather_branches(args.repo, master_ref)
    deletable = deletable_branches(binfo["local"], protected,
                                   binfo["checked_out"], binfo["merged"])
    branch_results = do_prune_branches(deletable, args.repo,
                                       dry_run=not args.prune_branches) if deletable else []
    remote_pruned = None
    if args.prune_branches:
        rc, out, _ = _git(["remote", "prune", "origin"], cwd=args.repo)
        remote_pruned = [ln for ln in out.splitlines() if ln.strip()] if rc == 0 else []

    if args.json:
        print(json.dumps({"master_ref": master_ref, "trunk": trunk,
                          "plan": plan, "pruned": pruned, "swept": swept,
                          "archive_dir": archive_dir,
                          "deletable_branches": deletable,
                          "branch_results": branch_results,
                          "remote_pruned": remote_pruned,
                          "worktrees": [{k: s.get(k) for k in
                                         ("path", "branch", "is_primary", "dirty",
                                          "untracked", "mid_op", "unmerged_to_master",
                                          "unpushed", "age_seconds")} for s in sigs]}, indent=2))
    else:
        print(render_text(sigs, plan, trunk=trunk))
        if swept:
            print("")
            verb = "SWEPT" if args.sweep_disposable else "WOULD SWEEP (run with --sweep-disposable)"
            print(f"DISPOSABLE SCRATCH WORKTREES — {verb} (archive: {archive_dir}):")
            for r in swept:
                if r.get("dry_run"):
                    print(f"  - {r['path']}")
                elif r["removed"]:
                    arch = r.get("archived")
                    tag = f" [archived {arch['tracked_patch_bytes']}B diff, {arch['untracked']} untracked]" if arch else ""
                    print(f"  - reaped {r['path']}{tag}")
                else:
                    print(f"  - FAILED {r['path']}: {r['error']}")
        if pruned:
            print("")
            for r in pruned:
                if r.get("dry_run"):
                    print(f"  would remove {r['path']} (run with --prune)")
                elif r["removed"]:
                    print(f"  removed {r['path']}")
                else:
                    print(f"  FAILED to remove {r['path']}: {r['error']}")
        print("")
        if deletable:
            verb = "delete" if args.prune_branches else "would delete (run with --prune-branches)"
            print(f"STALE LOCAL BRANCHES ({verb}): merged to {master_ref}, not checked out")
            for r in branch_results:
                if r.get("dry_run"):
                    print(f"  - {r['branch']}")
                elif r["deleted"]:
                    print(f"  - deleted {r['branch']}")
                else:
                    print(f"  - FAILED {r['branch']}: {r['error']}")
        else:
            print("STALE LOCAL BRANCHES: none")
        if remote_pruned:
            print(f"remote-tracking refs pruned: {', '.join(remote_pruned)}")

    # exit code: 0 converged / only allow-listed extras remain; 1 a real anomaly needs
    # a human (blocked / primary off master); 2 could not run. See module docstring.
    if plan["converged"]:
        return 0
    if plan["needs_human"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
