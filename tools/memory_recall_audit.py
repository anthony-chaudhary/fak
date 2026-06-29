#!/usr/bin/env python3
"""DOS-witnessed agent-memory freshness auditor - the memory store's checking layer.

The agent-memory store is an ACCELERATION layer: it carries hard-won, per-project
knowledge forward from past sessions. But a memory is the least-trustworthy signal
in the stack - a frozen self-report from a past session, handed back to a future
one wearing the authority of a fact. Nothing re-checked those facts against ground
truth, so a memory whose named commit fell out of history, or whose claimed
file/flag no longer exists, would silently re-enter context as current truth.
``.claude/rsi-loop-dod.md`` forbids exactly this: *every acceleration layer has a
matching checking layer*. This auditor checks the CLAIMS the store carries.

It is the in-fleet fold of the DOS recall rung (``dos memory verify``): the kernel
re-probes each memory's checkable claims (a commit SHA via git ancestry, a code
token via a comment-aware working-tree grep, a tracked path via git history)
against ground truth NOW and returns a closed verdict per memory:

  RECALL_FRESH        every checkable claim still holds          (safe to surface)
  RECALL_STALE        a load-bearing claim no longer holds       (do NOT re-inject)
  RECALL_UNVERIFIABLE names no re-checkable artifact / all abstain (an opinion)

UNVERIFIABLE is the EXPECTED majority - most fleet memories are prose/positioning
notes with no bindable claim, and that is fine. The actionable signal is STALE.
So ``memory_freshness_rate = FRESH / (FRESH + STALE)`` ignores the UNVERIFIABLE
denominator, and the audit is ``ok`` unless at least one memory is STALE.

Read-only by construction: it NEVER edits, prunes, rewrites, or re-homes a memory
file (that would be unsafe in the live shared worktree). It only reads the store
through ``dos memory verify`` and shapes the standard fleet control-pane payload.

STORE LOCATION: the Claude Code auto-memory store is per-project and
machine-specific - NOT a repo-relative committed mirror. It lives under the
home directory at ``~/.claude/projects/<ns>/memory/``, where ``<ns>`` is the
absolute workspace path with the drive colon and path separators collapsed to
``-`` (e.g. ``C:\\work\\fak`` -> ``C--work-fak``). ``collect`` resolves this
default from ``--workspace``; pass ``--store`` to point at any other store. The
store is NOT shipped via git, so every node audits its OWN local memories.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Callable

SCHEMA = "fleet-memory-recall-audit/1"

FRESH = "RECALL_FRESH"
STALE = "RECALL_STALE"
UNVERIFIABLE = "RECALL_UNVERIFIABLE"

# Non-fact index/doc files in the store that carry no bindable claim of their own.
_NON_FACT = {"MEMORY", "MEMORY_archive", "README"}


def repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    return here.parent.parent


def project_namespace(workspace: Path) -> str:
    """Encode an absolute workspace path as a Claude project namespace.

    The auto-memory store keys each project by its absolute path with the drive
    colon and every path separator collapsed to ``-`` (e.g. ``C:\\work\\fak`` ->
    ``C--work-fak``). This mirrors the on-disk ``~/.claude/projects/<ns>`` layout.
    """
    abs_path = str(workspace.resolve())
    out = []
    for ch in abs_path:
        out.append("-" if ch in (":", "/", "\\") else ch)
    return "".join(out)


def default_store(workspace: Path) -> Path:
    """The per-project auto-memory store for ``workspace``.

    Resolves to ``~/.claude/projects/<ns>/memory`` - the real, machine-specific
    location of the Claude Code auto-memory store. It is NOT a committed,
    repo-relative mirror: each node has its OWN store under its home directory.
    """
    ns = project_namespace(workspace)
    return Path.home() / ".claude" / "projects" / ns / "memory"


def run_text(cmd: list[str], cwd: Path, *, timeout: int = 120) -> dict[str, Any]:
    """Run a command, return {stdout, stderr, returncode}.

    Force UTF-8 decode with replacement: dos emits non-ASCII verdict prose (em
    dashes, arrows) the Windows default cp1252 codec cannot decode, which would
    otherwise crash the subprocess reader thread mid-audit (windows-ps51 gotcha).
    """
    try:
        proc = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"stdout": "", "stderr": str(exc), "returncode": 1, "_error": str(exc)}
    return {"stdout": proc.stdout, "stderr": proc.stderr, "returncode": proc.returncode}


# ---------------------------------------------------------------------------
# I/O seam: run `dos memory verify --json` (tests inject a fake)
# ---------------------------------------------------------------------------

StoreVerifier = Callable[[Path, str], dict[str, Any]]


def verify_store(workspace: Path, store: str) -> dict[str, Any]:
    """Run the kernel's whole-store recall sweep; return {records|error}."""
    res = run_text(
        ["dos", "memory", "verify", "--workspace", str(workspace),
         "--store", store, "--json"],
        workspace,
    )
    if res.get("_error"):
        return {"error": f"dos memory verify failed to run: {res['_error']}"}
    text = (res.get("stdout") or "").strip()
    if not text:
        # A non-zero exit with no JSON is a tooling error, not a clean store.
        err = (res.get("stderr") or "").strip() or f"empty output (rc={res.get('returncode')})"
        return {"error": f"dos memory verify produced no JSON: {err[:200]}"}
    try:
        data = json.loads(text)
    except ValueError:
        # Tolerant: scan for the last JSON array line (a banner may precede it).
        data = None
        for line in reversed(text.splitlines()):
            line = line.strip()
            if line.startswith("["):
                try:
                    data = json.loads(line)
                    break
                except ValueError:
                    continue
        if data is None:
            return {"error": "dos memory verify output was not parseable JSON"}
    if not isinstance(data, list):
        return {"error": "dos memory verify did not return a list of records"}
    return {"records": data}


# ---------------------------------------------------------------------------
# Pure grader: verdict records -> the standard fleet control-pane payload
# ---------------------------------------------------------------------------

def _culprit_str(rec: dict[str, Any]) -> str:
    """One-line description of the claim that decided a STALE verdict."""
    culprit = rec.get("culprit")
    if isinstance(culprit, dict):
        claim = culprit.get("claim") or {}
        raw = claim.get("raw") or claim.get("target_file") or ""
        gt = culprit.get("ground_truth") or culprit.get("status") or ""
        return f"{raw} - {gt}".strip(" -")
    if culprit:
        return str(culprit)
    return rec.get("reason") or ""


def build_payload(*, workspace: str, records: list[dict[str, Any]],
                  error: str | None = None) -> dict[str, Any]:
    graded: list[dict[str, Any]] = []
    counts = {FRESH: 0, STALE: 0, UNVERIFIABLE: 0}
    for rec in records:
        name = str(rec.get("memory") or "")
        if name in _NON_FACT:
            continue  # index/readme: not a fact file
        verdict = str(rec.get("verdict") or UNVERIFIABLE)
        counts[verdict] = counts.get(verdict, 0) + 1
        graded.append({
            "memory": name,
            "verdict": verdict,
            "type": rec.get("type") or "",
            "culprit": _culprit_str(rec) if verdict == STALE else "",
            "reason": rec.get("reason") or "",
        })

    fresh, stale = counts.get(FRESH, 0), counts.get(STALE, 0)
    denom = fresh + stale
    freshness_rate = round(fresh / denom, 4) if denom else None

    # `ok` is read FIRST by fleet_control_pane.classify_loop_status, so it must
    # carry the real signal: the sweep ran AND no memory is STALE.
    if error:
        ok, verdict, finding = False, "AUDIT_ERROR", "tooling_error"
        reason = error
        next_action = "fix the dos-memory-verify read-back (run from repo ROOT), then re-run"
    elif stale > 0:
        ok, verdict, finding = False, "ACTION", "stale_memory"
        reason = (
            f"{stale} memory file(s) assert a claim that no longer holds against git / "
            f"the working tree (freshness_rate={freshness_rate})"
        )
        next_action = (
            "for each STALE memory: refresh the fact to current ground truth, retire it if "
            "superseded, or correct the named SHA/path/token - then re-mirror with "
            "tools/sync_memory.py. NEVER let a STALE memory be surfaced as current fact."
        )
    else:
        ok, verdict, finding = True, "OK", "memories_fresh"
        reason = (
            f"no STALE memory: {fresh} FRESH, {counts.get(UNVERIFIABLE, 0)} UNVERIFIABLE "
            f"(opinion/positioning notes, expected); freshness_rate={freshness_rate}"
        )
        next_action = "no memory-freshness action needed; re-run after the next memory edit"

    return {
        "schema": SCHEMA,
        "ok": ok,
        "verdict": verdict,
        "finding": finding,
        "reason": reason,
        "next_action": next_action,
        "workspace": workspace,
        "memory_freshness_rate": freshness_rate,
        "counts": counts,
        "totals": {"memories_audited": len(graded)},
        "memories": sorted(graded, key=_sort_key),
    }


def _sort_key(g: dict[str, Any]) -> tuple[int, str]:
    # Surface STALE first, then FRESH, then UNVERIFIABLE; alpha within a bucket.
    order = {STALE: 0, FRESH: 1, UNVERIFIABLE: 2}
    return (order.get(g.get("verdict"), 9), str(g.get("memory") or ""))


# ---------------------------------------------------------------------------
# Wiring + CLI
# ---------------------------------------------------------------------------

def collect(workspace: Path, *, store: str | None = None,
            verifier: StoreVerifier | None = None) -> dict[str, Any]:
    root = workspace.resolve()
    store_arg = store or str(default_store(root))
    do_verify = verifier or verify_store
    result = do_verify(root, store_arg)
    if "error" in result:
        return build_payload(workspace=str(root), records=[], error=result["error"])
    return build_payload(workspace=str(root), records=result.get("records") or [])


def render(payload: dict[str, Any]) -> str:
    counts = payload.get("counts") or {}
    lines = [
        f"memory-recall audit: {payload.get('verdict')} ({payload.get('finding')})",
        f"freshness_rate={payload.get('memory_freshness_rate')}  next: {payload.get('next_action')}",
        (
            f"verdicts: fresh={counts.get(FRESH, 0)} "
            f"stale={counts.get(STALE, 0)} "
            f"unverifiable={counts.get(UNVERIFIABLE, 0)}"
        ),
    ]
    stale = [g for g in payload.get("memories", []) if g.get("verdict") == STALE]
    if stale:
        lines.append("  STALE (do not surface as current fact):")
        for g in stale[:20]:
            lines.append(f"    {g['memory']:<40} {g.get('culprit') or g.get('reason')}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="DOS-witnessed agent-memory freshness auditor (read-only)."
    )
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument(
        "--store", default="",
        help="memory store dir (default: ~/.claude/projects/<ns>/memory for the workspace)",
    )
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args(argv)

    workspace = Path(args.workspace).resolve() if args.workspace else repo_root()
    payload = collect(workspace, store=args.store or None)

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(render(payload))

    # Exit non-zero ONLY on a real STALE memory (or a tooling error). An all-
    # UNVERIFIABLE store is a clean pass, not a failure.
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
