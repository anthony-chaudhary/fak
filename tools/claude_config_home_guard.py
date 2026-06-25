#!/usr/bin/env python3
"""claude_config_home_guard.py -- regression guard against a bare ``~/.claude``
config-home hardcode in the tools tree (issue #606).

Why this exists
---------------
Claude Code can RELOCATE its state directory with the ``CLAUDE_CONFIG_DIR``
environment variable, and the fleet uses this heavily: every agent runs under its
own config home (``~/.claude-q-netra``, ``~/.claude-gem5-netra``, ...). So any tool
that resolves agent-memory or session-transcript paths under
``~/.claude/projects/<slug>/...`` MUST consult ``CLAUDE_CONFIG_DIR`` -- otherwise it
silently reads/writes the wrong (stale or foreign) store, or an empty one.

This bit us once: ``tools/sync_memory.py`` hardcoded ``~/.claude/projects/<slug>/
memory`` and mirrored the wrong store on every relocated node (fixed in 3eab69a,
precedence ``CLAUDE_MEMORY_DIR`` > ``CLAUDE_CONFIG_DIR`` > ``~/.claude``).
``tools/ctxcost.py`` / ``tools/ctxwin.py`` already had the correct pattern. A
one-time audit is clean today -- this guard keeps it clean by failing the build if
a FUTURE tool re-introduces the bare hardcode. It is a regression guard for a class
of bug, a sibling of the other honesty/lint gates (scrub_hardware_names,
skill_frontmatter_lint).

What it flags
-------------
A production ``tools/*.py`` that, on a single source line (comments and docstrings
masked out), BOTH:

  1. resolves a bare ``.claude`` config home anchored to the USER HOME -- a home
     token (``Path.home()``, ``expanduser(...)``, a ``"~/"`` literal, ``$HOME``, or a
     ``home`` variable) combined with a path component ``.claude`` that is *bare*
     (NOT ``.claude*`` -- the all-accounts glob is the legitimate relocation-aware
     pattern -- and NOT a per-account variant ``.claude-<x>`` / ``.claude.<x>``), AND
  2. steps that ``.claude`` straight into the agent STORE -- the next path
     component is ``projects`` or ``memory`` (i.e. the ``~/.claude/projects/...`` /
     ``~/.claude/memory`` store the bug lives in),

WITHOUT the file consulting ``CLAUDE_CONFIG_DIR`` (or ``CLAUDE_MEMORY_DIR``) anywhere.

What it deliberately does NOT flag
----------------------------------
  * The repo-root committed mirror ``<root>/.claude/memory`` (anchored to the repo
    root, not the user home -- e.g. ``tools/memory_read.py``, ``sync_memory.py``).
  * The ``glob("~/.claude*/projects/...")`` all-accounts scan (``.claude*`` is not a
    bare ``.claude``; it discovers every relocated-under-home account dir -- the
    pattern ``ctxwin.py`` / ``resume_sweep.py`` / ``peek_session.py`` use).
  * Prose / docstrings mentioning the path (masked before scanning).
  * Test files (``*_test.py``) -- they intentionally build fixture store paths
    under a tmpdir ``home`` to exercise the resolution; that is their job.

Modes
-----
  --check : lint; exit 1 if any tracked production tool carries the bad pattern.
  (default): same scan, but print the per-file hits and exit 0 (advisory listing).

Pure-stdlib, hermetic, repo root like the other honesty gates.
"""
from __future__ import annotations

import argparse
import io
import re
import subprocess
import sys
import tokenize
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# A bare ``.claude`` whose NEXT path component is the agent store (projects|memory).
# Two construction shapes: a literal/f-string path (``~/.claude/projects``) and a
# component join (``os.path.join(home, ".claude", "projects")`` / ``home / ".claude"
# / "projects"``). The negative lookahead ``(?![\w*.\-])`` keeps it BARE: it excludes
# ``.claude*`` (the all-accounts glob) and ``.claude-acct`` / ``.claude.json`` variants.
_STORE = r"(?:projects|memory)\b"
ADJ = re.compile(
    r"\.claude(?![\w*.\-])[\\/]\s*" + _STORE                        # "~/.claude/projects"
    + r"|[\"']\.claude[\"']\s*[,/]\s*[\(\s]*[\"']" + _STORE         # join(.., ".claude", "projects")
)

# A home anchor on the SAME line -- what makes a ``.claude`` the CONFIG HOME rather
# than the repo-root mirror. Broad on purpose; ADJ already constrains tightly.
HOME = re.compile(r"Path\.home\(\)|expanduser\(|[\"']~[\\/]|\bHOME\b|\bhome\b")

# The antidote: the file consults the relocated config home anywhere.
ANTIDOTE = re.compile(r"CLAUDE_CONFIG_DIR|CLAUDE_MEMORY_DIR")


def mask_noncode(src: str) -> list[str]:
    """Return ``src`` split into lines with comments and docstrings blanked to spaces
    (line numbers and columns preserved). Operational string literals -- the
    ``".claude"`` / ``"projects"`` args of a real path build -- are KEPT; only
    comments and standalone-expression docstrings are removed, so a doc mention of
    the path never trips the guard."""
    lines = src.splitlines()
    grid = [list(ln) for ln in lines]

    def clear(srow: int, scol: int, erow: int, ecol: int) -> None:
        for r in range(srow, erow + 1):
            if not (1 <= r <= len(grid)):
                continue
            row = grid[r - 1]
            a = scol if r == srow else 0
            b = ecol if r == erow else len(row)
            for c in range(a, min(b, len(row))):
                row[c] = " "

    prev = tokenize.NEWLINE
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return lines  # unparseable; fall back to raw lines (still scannable)
    for tok in toks:
        if tok.type == tokenize.COMMENT:
            clear(*tok.start, *tok.end)
        elif tok.type == tokenize.STRING and prev in (
            tokenize.NEWLINE, tokenize.NL, tokenize.INDENT,
            tokenize.DEDENT, tokenize.ENCODING,
        ):
            clear(*tok.start, *tok.end)  # docstring: a string as a bare statement
        if tok.type != tokenize.NL:
            prev = tok.type
    return ["".join(r) for r in grid]


def scan_text(src: str) -> list[tuple[int, str]]:
    """Return ``[(lineno, stripped_line), ...]`` for every bare-``~/.claude``-store
    hardcode in ``src`` -- but ONLY if the source never consults the relocated
    config home. A file with the antidote anywhere is exempt wholesale (it has
    already opted into relocation-awareness)."""
    masked = mask_noncode(src)
    if ANTIDOTE.search("\n".join(masked)):
        return []
    hits: list[tuple[int, str]] = []
    for i, line in enumerate(masked, 1):
        if ADJ.search(line) and HOME.search(line):
            hits.append((i, line.strip()))
    return hits


def production_tools() -> list[Path]:
    """git-tracked ``tools/*.py`` minus the ``*_test.py`` fixtures."""
    tracked = subprocess.run(
        ["git", "ls-files", "tools/*.py"],
        cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    out: list[Path] = []
    for rel in tracked:
        rel = rel.replace("\\", "/")
        if rel.endswith("_test.py"):
            continue
        out.append(REPO / rel)
    return out


def main(argv: list[str] | None = None) -> int:
    try:  # Windows consoles default to cp1252; output carries ~, →, ✅
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="lint; exit 1 on a bare ~/.claude config-home hardcode")
    ap.add_argument("files", nargs="*",
                    help="files to scan (default: tracked tools/*.py minus *_test.py)")
    args = ap.parse_args(argv)

    files = [Path(f) for f in args.files] if args.files else production_tools()
    flagged = 0
    for f in files:
        if not f.exists():
            continue
        hits = scan_text(f.read_text(encoding="utf-8"))
        if not hits:
            continue
        flagged += 1
        rel = f.relative_to(REPO) if f.is_absolute() and str(f).startswith(str(REPO)) else f
        for n, line in hits[:6]:
            print(f"  {rel}:{n}: {line}")

    if flagged:
        print(f"\nclaude-config-home guard: FAIL -- {flagged} tool(s) hardcode the "
              f"~/.claude/<store> config home without consulting CLAUDE_CONFIG_DIR")
        print("fix: resolve the home as CLAUDE_MEMORY_DIR > CLAUDE_CONFIG_DIR > ~/.claude "
              "(see tools/sync_memory.py / ctxcost.py / ctxwin.py)")
        return 1 if args.check else 0
    print("claude-config-home guard: clean "
          "(no bare ~/.claude/<store> hardcode in the tools tree)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
