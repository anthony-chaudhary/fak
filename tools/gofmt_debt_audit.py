#!/usr/bin/env python3
"""gofmt format-debt auditor — the checking layer for the blocking gofmt CI gate.

The CI workflow (``.github/workflows/ci.yml``) carries a BLOCKING gofmt step
(issue #38): any committed ``fak/**/*.go`` file that ``gofmt -l`` would reformat
fails the build. That gate only fires when a push reaches CI, so unformatted code
can sit on the trunk for a while before anyone notices it has gone red. This is
the always-on half: a read-only fold the control-pane runs every tick that
surfaces gofmt debt on the trunk the moment it lands — before a push discovers it.

WHY IT CHECKS THE COMMITTED BLOB, NOT THE WORKING TREE
------------------------------------------------------
On the canonical Windows/OneDrive checkout ``core.autocrlf=true`` and
``.gitattributes`` forces ``eol=lf`` only for ``*.sh``/``*.golden`` — NOT ``*.go``.
So a working-tree ``.go`` file is CRLF-transformed on checkout, and a local
``gofmt -l .`` flags hundreds of files purely as a line-ending artifact that does
NOT reflect CI. The authoritative, CI-equivalent signal is the COMMITTED BLOB:
CI checks out LF on Linux, which is exactly ``git show HEAD:<f>`` (git stores LF).
So this audit formats ``git show HEAD:<f>`` and compares — the same bytes the
Linux gofmt gate sees. (Future format work should use the same "blob method":
``git show HEAD:./f | gofmt > f``; never trust local ``gofmt -w``/``-l`` here.)

Read-only by construction: it never formats and never commits. ACTION means the
trunk would FAIL the gofmt CI gate right now; the fix is the gated ``/gofmt-sweep``
pass (format the listed files via the blob method, commit lane-by-lane by explicit
path). This is the format layer's checking loop, the way ``readme_freshness_audit``
is the front door's and ``memory_recall_audit`` is the memory mirror's.

Run from the repo ROOT: ``python tools/gofmt_debt_audit.py [--json]``. The only
external tools are ``git`` and ``gofmt`` (both required; a missing one is reported
as a tooling error, not silently passed).
"""
from __future__ import annotations

import argparse
import json
import subprocess
from dispatch_worker import install_no_window_subprocess_defaults
from pathlib import Path
from typing import Any
install_no_window_subprocess_defaults(subprocess)

SCHEMA = "fleet-gofmt-debt-audit/1"

# The Go module whose committed .go files the blocking CI gate covers. Pathspec is
# git-glob (repo-root relative); the module IS the repo root (no fak/ subdir).
DEFAULT_GLOB = "**/*.go"
DEFAULT_HEAD = "HEAD"

# How many offending files to name in the human render / reason before eliding.
_LIST_CAP = 25


# ---------------------------------------------------------------------------
# Pure-ish primitives: each shells exactly one tool, so they are the testable
# seam (a test feeds bytes to is_clean and asserts; no git needed).
# ---------------------------------------------------------------------------

def gofmt_format(src: bytes) -> tuple[bytes | None, str | None]:
    """Return (gofmt(src), None), or (None, error) if gofmt could not parse/run.

    gofmt reads stdin and writes the canonical (LF) formatting to stdout. A
    non-zero exit means the source did not parse (gofmt prints the error to
    stderr) — surfaced as an error string, never a silent "clean".
    """
    try:
        proc = subprocess.run(
            ["gofmt"], input=src, capture_output=True, timeout=30,
        )
    except FileNotFoundError:
        return None, "gofmt not found on PATH (install Go / add it to PATH)"
    except subprocess.TimeoutExpired:
        return None, "gofmt timed out"
    if proc.returncode != 0:
        return None, (proc.stderr or b"").decode("utf-8", "replace").strip()[:400]
    return proc.stdout, None


def is_clean(src: bytes) -> tuple[bool | None, str]:
    """Is `src` already gofmt-canonical? (True/False), or (None, err) on parse fail."""
    formatted, err = gofmt_format(src)
    if err is not None:
        return None, err
    return (formatted == src), ""


# ---------------------------------------------------------------------------
# Git scan: enumerate the module's committed .go files and check each blob.
# ---------------------------------------------------------------------------

def _git(root: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args], cwd=str(root), capture_output=True, timeout=60,
    )


def list_go_files(root: Path, glob: str) -> tuple[list[str], str | None]:
    proc = _git(root, "ls-files", glob)
    if proc.returncode != 0:
        return [], (proc.stderr or b"").decode("utf-8", "replace").strip()[:400]
    files = [ln for ln in proc.stdout.decode("utf-8", "replace").splitlines() if ln]
    return files, None


def blob_bytes(root: Path, head: str, path: str) -> tuple[bytes | None, str | None]:
    """The committed bytes of `path` at `head` (LF, exactly what Linux CI sees)."""
    proc = _git(root, "show", f"{head}:{path}")
    if proc.returncode != 0:
        return None, (proc.stderr or b"").decode("utf-8", "replace").strip()[:200]
    return proc.stdout, None


def scan(root: Path, *, head: str = DEFAULT_HEAD,
         glob: str = DEFAULT_GLOB) -> dict[str, Any]:
    """Scan committed .go blobs for gofmt debt. Returns a structured result.

    {scanned, unclean: [path…], unparseable: [{path, error}…], error?}
    """
    files, err = list_go_files(root, glob)
    if err is not None:
        return {"scanned": 0, "unclean": [], "unparseable": [],
                "error": f"git ls-files {glob!r} failed: {err}"}

    unclean: list[str] = []
    unparseable: list[dict[str, str]] = []
    for path in files:
        src, gerr = blob_bytes(root, head, path)
        if gerr is not None:
            unparseable.append({"path": path, "error": f"git show: {gerr}"})
            continue
        clean, cerr = is_clean(src)
        if clean is None:
            # gofmt could not run/parse. If gofmt itself is missing, every file
            # reports the same error — bail with a single tooling error.
            if "gofmt not found" in cerr:
                return {"scanned": len(files), "unclean": [], "unparseable": [],
                        "error": cerr}
            unparseable.append({"path": path, "error": cerr})
        elif not clean:
            unclean.append(path)
    return {"scanned": len(files), "unclean": sorted(unclean),
            "unparseable": unparseable}


# ---------------------------------------------------------------------------
# Grader: fold the scan into the standard control-pane payload.
# ---------------------------------------------------------------------------

def build_payload(*, workspace: str, scan_result: dict[str, Any],
                  glob: str = DEFAULT_GLOB) -> dict[str, Any]:
    error = scan_result.get("error")
    unclean = list(scan_result.get("unclean") or [])
    unparseable = list(scan_result.get("unparseable") or [])
    scanned = int(scan_result.get("scanned") or 0)

    counts = {"scanned": scanned, "unclean": len(unclean),
              "unparseable": len(unparseable)}

    if error:
        ok, verdict, finding = False, "AUDIT_ERROR", "tooling_error"
        reason = error
        next_action = ("fix the git/gofmt read (run from repo ROOT with git + gofmt "
                       "on PATH), then re-run")
    elif unclean:
        ok, verdict, finding = False, "ACTION", "gofmt_debt"
        named = ", ".join(unclean[:_LIST_CAP])
        more = f" (+{len(unclean) - _LIST_CAP} more)" if len(unclean) > _LIST_CAP else ""
        reason = (f"{len(unclean)} committed .go file(s) are not gofmt-clean — the "
                  f"blocking gofmt CI gate (#38) would FAIL: {named}{more}")
        next_action = ("invoke /gofmt-sweep: format each listed file via the blob method "
                       "(git show HEAD:./f | gofmt > f), build, then commit lane-by-lane "
                       "by explicit path with a (fak <lane>) trailer — never `gofmt -w .` "
                       "(rewrites active WIP) and never `git add -A`")
    elif unparseable:
        # Committed .go that gofmt cannot parse is a real defect, but a rare one
        # and not what the gofmt gate measures; surface it as ACTION to investigate.
        ok, verdict, finding = False, "ACTION", "unparseable_go"
        named = ", ".join(u["path"] for u in unparseable[:_LIST_CAP])
        reason = f"{len(unparseable)} committed .go file(s) did not parse under gofmt: {named}"
        next_action = "inspect the listed file(s); a committed .go that gofmt cannot parse is a defect"
    else:
        ok, verdict, finding = True, "OK", "gofmt_clean"
        reason = f"all {scanned} committed {glob} file(s) are gofmt-clean; the gofmt CI gate would pass"
        next_action = "no format action needed; re-run after the next .go change lands on the trunk"

    return {
        "schema": SCHEMA,
        "ok": ok,
        "verdict": verdict,
        "finding": finding,
        "reason": reason,
        "next_action": next_action,
        "workspace": workspace,
        "glob": glob,
        "counts": counts,
        "unclean": unclean,
        "unparseable": unparseable,
    }


# ---------------------------------------------------------------------------
# Wiring + CLI
# ---------------------------------------------------------------------------

def repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    return here.parent.parent


def collect(workspace: Path, *, head: str = DEFAULT_HEAD,
            glob: str = DEFAULT_GLOB) -> dict[str, Any]:
    root = workspace.resolve()
    result = scan(root, head=head, glob=glob)
    return build_payload(workspace=str(root), scan_result=result, glob=glob)


def render(payload: dict[str, Any]) -> str:
    counts = payload.get("counts") or {}
    lines = [
        f"gofmt-debt audit: {payload.get('verdict')} ({payload.get('finding')})",
        (f"scanned={counts.get('scanned', 0)} unclean={counts.get('unclean', 0)} "
         f"unparseable={counts.get('unparseable', 0)}"),
        f"reason: {payload.get('reason')}",
        f"next: {payload.get('next_action')}",
    ]
    for p in (payload.get("unclean") or [])[:_LIST_CAP]:
        lines.append(f"  unclean  {p}")
    for u in (payload.get("unparseable") or [])[:_LIST_CAP]:
        lines.append(f"  noparse  {u['path']}: {u['error']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="gofmt format-debt auditor (read-only; checks committed blobs).")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--head", default=DEFAULT_HEAD, help="git rev to audit (default: HEAD)")
    ap.add_argument("--glob", default=DEFAULT_GLOB,
                    help=f"git pathspec of .go files (default: {DEFAULT_GLOB})")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args(argv)

    workspace = Path(args.workspace).resolve() if args.workspace else repo_root()
    payload = collect(workspace, head=args.head, glob=args.glob)

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(render(payload))

    # Exit non-zero when the trunk carries gofmt debt (or a tooling error). The
    # control pane reads `ok` first, but the nonzero exit alone classifies ACTION.
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
