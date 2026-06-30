#!/usr/bin/env python3
"""Hardware-tell DOC-CONTENT gate: refuse a staged *.md that ADDS a prose hardware tell.

The leak class this closes
--------------------------
`tools/scrub_hardware_names.py --check` (the `residual_hits` doc-content scan) runs only
POST-commit in `make ci` / `make hygiene`. So a committed doc carrying a prose hardware
tell (`on DGX`, `DGX hours`, `dgx3`, `da33`, `SXM4`, …) sails through `git commit` and only
detonates LATER as a red trunk in CI — a fleet-wide failure on a leak nobody in a green lane
authored (issue #1455). This gate moves the SAME scan to COMMIT time, so the author who adds
the tell is the one who is refused, before it lands.

Why it cannot drift from the scrubber
-------------------------------------
It does NOT define a second pattern list. It IMPORTS `tools/scrub_hardware_names.py` and reuses
its `residual_hits` — the very function `--check` uses — so the gate and the lint agree by
construction. In particular it inherits the FALSE-POSITIVE-safe masking landed in `21a993bf`:
filename link-text (`[DGX-OVERNIGHT-PLAN](…)`) is masked as an identifier, not a prose tell, so
this gate does not re-introduce the FP that issue #1455 is about.

Scope: only the lines this commit ADDS
---------------------------------------
`residual_hits` is run over the full STAGED blob of each added/modified `.md` (so fenced-code
context is fence-accurate), but a hit is reported ONLY if its line is in the staged ADDED set.
A pre-existing peer-authored tell on a line this commit merely sits next to is NOT the author's
to fix, and refusing it would re-create the very "red on a leak you didn't author" friction this
gate exists to remove.

Mode env  FLEET_HW_GUARD = block (default) | warn | off
Escape    FLEET_ALLOW_HW=1  (the meta-case: a commit about the scrubber itself, or a competitor
          citation). Shared with the commit-MESSAGE gate so one escape covers both.

Exit: 0 clean, 1 a real added tell (block mode), 2 could-not-run (the hook falls open on 2).
Fail-open: any internal error (git missing, scrubber unimportable, decode failure) → exit 2,
never blocks a commit because the checker itself broke.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _load_scrubber():
    """Import tools/scrub_hardware_names.py so the gate reuses its patterns + masking.

    Single source of truth: residual_hits / RESIDUAL_TELLS / _mask_inline_code all live there.
    Returns the module, or None if it cannot be loaded (gate then fails open).
    """
    script = Path(__file__).resolve().parent / "scrub_hardware_names.py"
    try:
        spec = importlib.util.spec_from_file_location("scrub_hardware_names", script)
        if not spec or not spec.loader:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception:  # any import/exec failure → fail open
        return None


def _git(args: list[str], root: str) -> subprocess.CompletedProcess:
    # Decode as UTF-8 explicitly (Windows locale is cp1252; staged docs carry ×, →, ✅, U+2581).
    # errors="replace" keeps the gate scanning instead of crashing on stray bytes.
    return subprocess.run(
        ["git", "-C", root] + args,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )


def _staged_md_paths(root: str) -> list[str] | None:
    """Added (A), modified (M), or renamed-into (R) staged *.md paths (the new path)."""
    r = _git(["diff", "--cached", "--name-status", "--diff-filter=ACMR"], root)
    if r.returncode != 0:
        return None
    out: list[str] = []
    for line in r.stdout.splitlines():
        parts = line.split("\t")
        if not parts:
            continue
        path = parts[-1].replace("\\", "/")
        if path.endswith(".md"):
            out.append(path)
    return out


def _staged_blob(root: str, path: str) -> str | None:
    """The STAGED content of a path (the bytes that will be committed: `git show :path`)."""
    r = _git(["show", f":{path}"], root)
    if r.returncode != 0:
        return None
    return r.stdout


def _added_lines(root: str, path: str) -> set[str] | None:
    """The set of ADDED line bodies for one path (the '+' side of its staged diff).

    Compared by stripped text so a hit on a line this commit introduced is reported, while a
    pre-existing peer-authored tell on an untouched line is not. A new file (--diff-filter=A
    with no index parent) shows every line as added, which is correct: all of it is the
    author's.
    """
    r = _git(["diff", "--cached", "--unified=0", "--no-color", "--", path], root)
    if r.returncode != 0:
        return None
    added: set[str] = set()
    for line in r.stdout.splitlines():
        if line.startswith("+++"):
            continue
        if line.startswith("+"):
            added.add(line[1:].strip())
    return added


def audit_staged(root: str, scrub) -> int:
    """Scan staged *.md additions for a prose hardware tell. Returns the exit code."""
    paths = _staged_md_paths(root)
    if paths is None:
        print("HARDWARE_TELL (warn): git not available; check skipped.", file=sys.stderr)
        return 2

    # The scrubber's own meta files legitimately carry every tell in their patterns/tests;
    # never flag them (the lint excludes generated docs; these are the rule's own source).
    self_ref = {
        "tools/scrub_hardware_names.py",
        "tools/scrub_hardware_names_test.py",
        "tools/check_hardware_tells.py",
        "tools/check_hardware_tells_test.py",
        "PUBLIC-SCRUB-POLICY.md",
    }

    findings: list[tuple[str, int, str]] = []
    for path in paths:
        if path in self_ref:
            continue
        blob = _staged_blob(root, path)
        if blob is None:
            continue  # unreadable staged blob → skip this file (fail open per file)
        added = _added_lines(root, path)
        if added is None:
            added = set()
        # residual_hits is the EXACT function --check uses (fence-aware, FP-safe masking).
        for ln, text in scrub.residual_hits(blob):
            if text.strip() in added:
                findings.append((path, ln, text))

    if not findings:
        print("hardware-tell (content): clean (no new prose A100/DGX/SXM4 tell in staged docs).")
        return 0

    print(
        f"HARDWARE_TELL: {len(findings)} staged doc line(s) ADD a private hardware tell "
        f"(a prose DGX/SXM4/dgxN/da33 name):",
        file=sys.stderr,
    )
    for path, ln, text in findings[:12]:
        print(f"  {path}:{ln}: {text.strip()[:80]}", file=sys.stderr)
    print(
        "  fix: python tools/scrub_hardware_names.py --apply " + (findings[0][0] if findings else "<file>")
        + "  (describe the box generically: GPU server / datacenter GPU), per PUBLIC-SCRUB-POLICY.md.",
        file=sys.stderr,
    )
    print(
        "  override once: FLEET_ALLOW_HW=1 <git cmd>  (a commit about the scrubber itself, "
        "or a competitor citation).",
        file=sys.stderr,
    )
    return 1


def main() -> int:
    try:  # Windows consoles default to cp1252; doc prose carries ×, →, ✅, etc.
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    ap = argparse.ArgumentParser(description=__doc__)
    # --audit-staged is the only mode; accept it (or its absence) uniformly so the hook's
    # `--audit-staged --root <root>` invocation matches the sibling checkers' contract.
    ap.add_argument("--audit-staged", action="store_true",
                    help="scan staged *.md additions for a prose hardware tell (pre-commit)")
    ap.add_argument("--root", default=".", help="repo root (default: cwd)")
    args = ap.parse_args()

    mode = os.environ.get("FLEET_HW_GUARD", "block")
    if mode == "off" or os.environ.get("FLEET_ALLOW_HW") == "1":
        print("hardware-tell (content): skipped (FLEET_HW_GUARD=off / FLEET_ALLOW_HW=1).")
        return 0

    root = os.path.abspath(args.root)

    scrub = _load_scrubber()
    if scrub is None:
        print("HARDWARE_TELL (warn): could not load scrub_hardware_names.py; check skipped.",
              file=sys.stderr)
        return 2

    try:
        rc = audit_staged(root, scrub)
    except Exception as exc:  # any unexpected failure → fail open, never wedge the commit
        print(f"HARDWARE_TELL (warn): check could not run ({exc}); skipped.", file=sys.stderr)
        return 2
    if rc == 1 and mode == "warn":
        return 0  # advisory: surface the finding (already printed) but do not block
    return rc


if __name__ == "__main__":
    sys.exit(main())
