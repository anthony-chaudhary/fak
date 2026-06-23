#!/usr/bin/env python3
r"""dispatch_parity_canary — the typed gate that holds opencode population at 1
until glm-5.2 is *proven* to drive the dispatch loop the way Claude does.

The `dos-dispatch-loop` skill's design principle is "the kernel decides, not
prose": the *verdict* is an exit code, so a weaker model can still be correct
**if** it branches on the int rather than re-narrating the verdict. Claude is
the reference implementation; glm-5.2 (the configured opencode model) is the
unproven one. The hedge is sound, but model parity over the whole loop length
is an assumption — and an opencode worker that mis-reads a verdict can ship a
red tree, re-storm a drained lane, strand uncommitted work, or double-commit
(issue #420). Each is caught downstream by `dos verify` / the git hooks / the
trunk guard, but every catch is a productivity loss, so we prove parity on ONE
low-risk lane *before* raising the opencode population.

This module is the gate that makes that proof a typed fact instead of a vibe.
It does NOT run a worker. It grades an *already-shipped* canary commit — the
artifact a `--max-ticks 1` opencode run on the `docs` lane left behind — against
the **parity bar**, the closed set of behaviors a loop worker must exhibit no
matter which model drives it:

  1. SHIPPED_A_UNIT     the canary produced a real commit (not an empty/--allow-empty)
  2. WITNESSED          `dos commit-audit <sha>` grades it OK / diff-witnessed —
                        the diff does the KIND of thing the subject claims, so the
                        worker shipped a real change, not a self-narrated one
  3. BY_PATHSPEC        it committed by explicit `git commit -- <paths>`, never a
                        blanket `git add -A` that would steal a sibling's WIP in
                        the shared multi-session tree
  4. SIGNED_OFF         `git commit -s` (the DCO sign-off the hook requires)
  5. ISSUE_BOUND        the subject cites `#<issue>` so the closure auditor can
                        bind the commit to its ticket (an unbound ship never closes)
  6. LANE_TREE_CLEAN    the worker stopped on a typed verdict and left the lane's
                        file tree clean — no stranded edits outside the unit it shipped

All six must hold ⇒ ``PARITY_PROVEN`` (exit 0). Any miss ⇒ a typed
``PARITY_UNPROVEN`` carrying the first failed rung, and the population-raise stays
gated. The whole point of #420 is that this verdict is computed from the SAME
non-forgeable surfaces the loop already trusts — git + `dos commit-audit` — not
from the worker's own transcript prose.

The behavioral rungs (3, 4, 5, 6) are read from the dispatch run log the
spawner already writes (`.dispatch-runs/resolve-<issue>-*.log`) — a structural
read of the git commands the worker actually ran, never its narration. Rungs 1
and 2 are read from git itself. A `--commit <sha>` with no run log can still be
graded on the git-only rungs (1, 2, 5) and reports the behavioral rungs as
``unobserved`` rather than guessing — a thin answer never masquerades as a
strong one.

Read-only. Emits JSON (``--json``) or a one-line card. Exit 0 iff
``PARITY_PROVEN``. The gate the opencode-population raise branches on:

    python tools/dispatch_parity_canary.py --commit 7528df3 --issue 545 \
        --backend opencode --log .dispatch-runs/resolve-545-20260623-162209.log

See docs/dispatch-loop.md (the loop) and issue #420 (the canary contract).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

SCHEMA = "fak-dispatch-parity-canary/1"

# The parity bar: the closed, ordered set of behaviors a dispatch-loop worker
# must exhibit regardless of which model drives it. Each rung names the typed
# PARITY_UNPROVEN_* reason emitted when it fails. Ordered cheapest/most-load-
# bearing first so the first miss is the most informative.
PARITY_BAR: tuple[tuple[str, str], ...] = (
    ("shipped_a_unit", "PARITY_UNPROVEN_NO_UNIT"),
    ("witnessed", "PARITY_UNPROVEN_UNWITNESSED"),
    ("issue_bound", "PARITY_UNPROVEN_UNBOUND"),
    ("by_pathspec", "PARITY_UNPROVEN_BLANKET_ADD"),
    ("signed_off", "PARITY_UNPROVEN_NO_SIGNOFF"),
    ("lane_tree_clean", "PARITY_UNPROVEN_TREE_DIRTY"),
)


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "dos.toml").is_file() or (parent / ".git").exists():
            return parent
    return here.parent.parent


def _git(root: Path, *args: str) -> tuple[int, str]:
    """Run a read-only git command; return (rc, stdout). Never raises."""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
        )
        return proc.returncode, (proc.stdout or "")
    except (OSError, ValueError) as exc:  # pragma: no cover - env-dependent
        return 1, f"<git-error: {exc}>"


# --------------------------------------------------------------------------
# Pure predicates over the commit subject and the worker's git-command stream.
# These are the heart of the gate; they are deliberately string-pure so the
# test can drive every rung without git/subprocess/network.
# --------------------------------------------------------------------------

def issue_bound(subject: str, issue: int | None) -> bool:
    """Rung 5: the subject cites `#<issue>` (or any `#N` when issue unset)."""
    if issue is not None:
        return bool(re.search(rf"#\b{issue}\b", subject))
    return bool(re.search(r"#\d+\b", subject))


# A worker that ran `git add -A` / `git add .` / `git add --all` / `git add -u`
# staged the whole shared tree — the exact blanket-add that steals a sibling's
# WIP. We refute parity on the presence of that command. Help-text echoes like
# `(use "git add <file>..." to update ...)` are not commands — they carry a
# quote/paren and never sit at the start of a command, so the leading-anchor
# match below ignores them.
_BLANKET_ADD = re.compile(r"(?m)^\s*git\s+add\s+(?:-A\b|--all\b|-u\b|\.(?:\s|$))")
# The shared-tree discipline is satisfied two ways: a pathspec on the commit
# itself (`git commit -- <path>`), OR an explicit `git add <named-path>` (a real
# path, not a flag and not the `<file>` placeholder) that stages only what the
# worker chose. Either proves it did not blanket-add.
_COMMIT_PATHSPEC = re.compile(r"(?m)^\s*git\s+commit\b[^\n]*\s--\s+\S")
_EXPLICIT_ADD = re.compile(r"(?m)^\s*git\s+add\s+(?!-|<)[^\s\"']+")
_COMMIT_SIGNOFF = re.compile(r"(?m)^\s*git\s+commit\b[^\n]*(?:-s\b|--signoff\b)")


def by_pathspec(git_cmds: str) -> bool:
    """Rung 3: staged explicitly (commit pathspec or named add), never blanket."""
    if _BLANKET_ADD.search(git_cmds):
        return False
    return bool(_COMMIT_PATHSPEC.search(git_cmds) or _EXPLICIT_ADD.search(git_cmds))


def signed_off(git_cmds: str) -> bool:
    """Rung 4: the commit carried a DCO sign-off."""
    return bool(_COMMIT_SIGNOFF.search(git_cmds))


def grade(
    *,
    shipped_a_unit: bool,
    witnessed: bool,
    subject: str,
    issue: int | None,
    git_cmds: str | None,
    lane_tree_clean: bool | None,
) -> dict[str, Any]:
    """Apply the parity bar. Returns the rung map + the first-failed verdict.

    ``git_cmds`` None ⇒ no run log: the behavioral rungs (by_pathspec,
    signed_off) report ``unobserved`` and cannot be claimed proven.
    ``lane_tree_clean`` None ⇒ likewise unobserved.
    """
    have_log = git_cmds is not None
    rungs: dict[str, Any] = {
        "shipped_a_unit": bool(shipped_a_unit),
        "witnessed": bool(witnessed),
        "issue_bound": issue_bound(subject, issue),
        "by_pathspec": by_pathspec(git_cmds) if have_log else None,
        "signed_off": signed_off(git_cmds) if have_log else None,
        "lane_tree_clean": (None if lane_tree_clean is None else bool(lane_tree_clean)),
    }

    failed: str | None = None
    reason: str | None = None
    for rung, token in PARITY_BAR:
        value = rungs[rung]
        if value is None:
            # Unobserved behavioral rung: parity is not PROVEN, but this is an
            # evidence gap, not a refutation. Stop here with a distinct verdict.
            failed = rung
            reason = "PARITY_UNOBSERVED"
            break
        if not value:
            failed = rung
            reason = token
            break

    proven = failed is None
    return {
        "proven": proven,
        "verdict": "PARITY_PROVEN" if proven else reason,
        "failed_rung": failed,
        "rungs": rungs,
    }


# --------------------------------------------------------------------------
# I/O layer: derive the rung inputs from git + the dispatch run log.
# --------------------------------------------------------------------------

# In an opencode/Claude run log the shell lines render the command after the
# prompt glyph + an ANSI reset, e.g. "\x1b[0m$ \x1b[0mgit commit -s -- a b".
# We pull every `git ...` invocation back out structurally (the command stream),
# never the surrounding prose.
_GIT_IN_LOG = re.compile(r"git\s+[a-z][\w-]*(?:[^\n]*)", re.IGNORECASE)


def git_cmds_from_log(text: str) -> str:
    """Flatten a run log to the newline-joined `git ...` commands it ran."""
    return "\n".join(m.group(0).rstrip() for m in _GIT_IN_LOG.finditer(text))


def commit_subject(root: Path, sha: str) -> str:
    rc, out = _git(root, "show", "-s", "--format=%s", sha)
    return out.strip() if rc == 0 else ""


def is_real_unit(root: Path, sha: str) -> bool:
    """Rung 1: the commit touched at least one file (not an empty commit)."""
    # diff-tree lists the changed paths for a single commit without the
    # `--name-only`/`-s` option clash `git show` trips on.
    rc, out = _git(root, "diff-tree", "--no-commit-id", "--name-only", "-r", sha)
    if rc != 0:
        return False
    return any(line.strip() for line in out.splitlines())


def witness_commit(root: Path, sha: str) -> dict[str, Any]:
    """Rung 2: grade via `dos commit-audit`. OK ∧ diff/data-witnessed ⇒ True.

    Shells to the `dos` CLI exactly as the close arm does — `dos commit-audit
    <sha> --workspace <root> --json`, which emits a JSON ARRAY (one row per
    audited sha). Returns {witnessed, verdict, witness} — fail-closed if dos is
    absent or the row is missing.
    """
    try:
        proc = subprocess.run(
            ["dos", "commit-audit", sha, "--workspace", str(root), "--json"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, ValueError) as exc:
        return {"witnessed": False, "verdict": "DOS_UNAVAILABLE", "witness": None,
                "error": str(exc)}
    if proc.returncode not in (0, 1) or not proc.stdout.strip():
        return {"witnessed": False, "verdict": "DOS_UNAVAILABLE", "witness": None}
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"witnessed": False, "verdict": "DOS_UNPARSEABLE", "witness": None}
    row = payload[0] if isinstance(payload, list) and payload else (
        payload if isinstance(payload, dict) else {})
    verdict = row.get("verdict")
    witness = row.get("witness")
    ok = verdict == "OK" and witness in ("diff-witnessed", "data-witnessed")
    return {"witnessed": bool(ok), "verdict": verdict, "witness": witness}


def lane_tree_clean(root: Path, lane_tree: str | None) -> bool | None:
    """Rung 6: the lane's file tree carries no uncommitted change right now.

    ``lane_tree`` is a path prefix (e.g. ``tools`` or ``docs``). None ⇒ the
    rung is unobserved (we don't know which tree the unit belonged to).
    A clean lane tree is the structural proof the worker stopped on a verdict
    and stranded nothing.
    """
    if not lane_tree:
        return None
    rc, out = _git(root, "status", "--porcelain", "--", lane_tree)
    if rc != 0:
        return None
    return not out.strip()


def evaluate(
    root: Path,
    *,
    commit: str,
    issue: int | None,
    backend: str,
    log_path: Path | None,
    lane_tree: str | None,
) -> dict[str, Any]:
    sha_rc, sha_out = _git(root, "rev-parse", "--short", commit)
    sha = sha_out.strip() if sha_rc == 0 else commit
    subject = commit_subject(root, commit)
    shipped = is_real_unit(root, commit)
    wit = witness_commit(root, commit)

    git_cmds: str | None = None
    if log_path is not None:
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
            git_cmds = git_cmds_from_log(text)
        except OSError:
            git_cmds = None

    tree_clean = lane_tree_clean(root, lane_tree)

    graded = grade(
        shipped_a_unit=shipped,
        witnessed=wit["witnessed"],
        subject=subject,
        issue=issue,
        git_cmds=git_cmds,
        lane_tree_clean=tree_clean,
    )

    return {
        "schema": SCHEMA,
        "workspace": str(root),
        "backend": backend,
        "commit": sha,
        "issue": issue,
        "subject": subject,
        "log": str(log_path) if log_path else None,
        "lane_tree": lane_tree,
        "ship_witness": wit,
        **graded,
        "ok": graded["proven"],
        "interpretation": _interpret(graded, backend),
    }


def _interpret(graded: dict[str, Any], backend: str) -> str:
    if graded["proven"]:
        return (
            f"PARITY_PROVEN — the {backend} canary cleared every rung of the bar "
            "(shipped a witnessed unit, bound to its issue, committed by pathspec "
            "with sign-off, lane tree clean). Safe to gate the population raise "
            "ON this verdict."
        )
    if graded["verdict"] == "PARITY_UNOBSERVED":
        return (
            f"PARITY_UNOBSERVED — rung '{graded['failed_rung']}' has no evidence "
            "(no run log / unknown lane tree). Parity is not refuted, but it is "
            "not proven either; supply --log / --lane-tree before raising the "
            "population."
        )
    return (
        f"{graded['verdict']} — rung '{graded['failed_rung']}' FAILED. The "
        f"{backend} worker did not match the reference loop discipline; keep "
        "opencode population at 1 and fix the gap before re-running the canary."
    )


def render(payload: dict[str, Any]) -> str:
    head = f"{payload['verdict']}  ({payload['backend']} canary on {payload['commit']}"
    if payload.get("issue"):
        head += f", #{payload['issue']}"
    head += ")"
    rungs = payload.get("rungs", {})
    marks = []
    for rung, _ in PARITY_BAR:
        v = rungs.get(rung)
        glyph = "?" if v is None else ("ok" if v else "MISS")
        marks.append(f"{rung}={glyph}")
    return head + "\n  " + "  ".join(marks) + "\n  " + payload["interpretation"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Typed parity gate: did the model's dispatch-loop canary "
                    "match the reference loop discipline? (issue #420)")
    ap.add_argument("--commit", required=True,
                    help="the shipped canary commit (sha / ref) to grade")
    ap.add_argument("--issue", type=int, default=None,
                    help="the issue the canary was supposed to resolve (for the "
                         "#N-bound rung)")
    ap.add_argument("--backend", default="opencode",
                    help="which backend drove the canary (default: opencode)")
    ap.add_argument("--log", default="",
                    help="the dispatch run log (.dispatch-runs/resolve-*.log) — "
                         "without it the behavioral rungs report unobserved")
    ap.add_argument("--lane-tree", default="",
                    help="the lane's file-tree prefix (e.g. docs) for the "
                         "tree-clean rung")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args(argv)

    root = Path(args.workspace).resolve() if args.workspace else repo_root()
    log_path = Path(args.log) if args.log else None
    payload = evaluate(
        root,
        commit=args.commit,
        issue=args.issue,
        backend=args.backend,
        log_path=log_path,
        lane_tree=args.lane_tree or None,
    )
    print(json.dumps(payload, indent=2) if args.json else render(payload))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
