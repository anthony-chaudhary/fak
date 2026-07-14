#!/usr/bin/env python3
"""Shared shift-left skip-gate for the Python scorecards — the analog of Go's
internal/scdiff.

The scorecard family's post-hoc cost is whole-tree rescans: every card re-reads its
entire corpus each run, so an agent editing one file pays a full scan to learn "did
my edit move the number?". These cards are HOLISTIC (code-slop's duplication and
dead-code, code-quality's architecture and toolchain, docs' cross-doc link integrity
all read the whole corpus), so a partial scan of only the changed files would report
a WRONG number. The correct incremental move is therefore all-or-nothing: if a change
touched none of the card's corpus, its debt cannot have moved — skip the whole scan;
otherwise fall through to a full, correct scan.

`since_skip` is the reusable gate a card calls near the top of main(); it mirrors the
Go `uiQualitySinceSkip` shape (return a code to skip, or None to full-scan).
"""

from __future__ import annotations

import re
import subprocess
import sys


def no_window_creationflags() -> int:
    """CREATE_NO_WINDOW on Windows so a background git spawn under headless automation
    does not flash a console window; 0 on POSIX. Mirrors dispatch_worker.no_window_creationflags."""
    if sys.platform == "win32":
        return 0x08000000  # subprocess.CREATE_NO_WINDOW
    return 0


def _normalize(p: str) -> str:
    """Canonicalize to repo-relative slash form: backslashes to slashes, a leading
    './' stripped, surrounding whitespace trimmed."""
    p = p.strip().replace("\\", "/")
    return p[2:] if p.startswith("./") else p


def changed_paths(root, since: str) -> list[str]:
    """Repo-relative, slash-separated, sorted paths that differ between `since` and
    the working tree: `git diff --name-only <since>` (two-dot: ref vs working tree,
    so staged and unstaged edits both count) unioned with untracked files
    (`git ls-files --others --exclude-standard`, since a new file adds debt).

    A blank `since` returns []. A git failure (unresolvable ref) raises RuntimeError,
    so the caller full-scans rather than falsely reporting "unchanged" off a broken
    diff.
    """
    since = (since or "").strip()
    if not since:
        return []
    root = str(root)

    def git_lines(*args: str) -> list[str]:
        proc = subprocess.run(
            ["git", "-C", root, *args],
            capture_output=True, text=True,
            creationflags=no_window_creationflags(),
        )
        if proc.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)}: {proc.stderr.strip()}")
        return [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]

    changed: set[str] = set()
    for p in git_lines("diff", "--name-only", since, "--"):
        changed.add(_normalize(p))
    try:  # untracked files never appear in `git diff`; a failure here is non-fatal.
        for p in git_lines("ls-files", "--others", "--exclude-standard"):
            changed.add(_normalize(p))
    except RuntimeError:
        pass
    return sorted(c for c in changed if c)


_glob_cache: dict[str, re.Pattern] = {}


def _compile_glob(g: str) -> re.Pattern:
    """Translate a corpus glob into an anchored regexp (cached). '**/' → an optional
    run of leading segments, bare '**' → '.*' (crossing '/'), single '*' → '[^/]*'
    (segment-local), every other char literal. Mirrors internal/scdiff.compileGlob."""
    cached = _glob_cache.get(g)
    if cached is not None:
        return cached
    out: list[str] = ["^"]
    i = 0
    while i < len(g):
        if g[i] == "*" and i + 1 < len(g) and g[i + 1] == "*":
            if i + 2 < len(g) and g[i + 2] == "/":
                out.append("(?:.*/)?")
                i += 3
            else:
                out.append(".*")
                i += 2
        elif g[i] == "*":
            out.append("[^/]*")
            i += 1
        else:
            out.append(re.escape(g[i]))
            i += 1
    out.append("$")
    pat = re.compile("".join(out))
    _glob_cache[g] = pat
    return pat


def match_glob(g: str, p: str) -> bool:
    """Whether repo-relative slash path p matches corpus glob g. A trailing '/' is a
    directory-prefix match; otherwise the compiled-regexp vocabulary above applies."""
    g = _normalize(g)
    p = _normalize(p)
    if not g:
        return False
    if g.endswith("/"):
        return p.startswith(g) or p == g[:-1]
    return bool(_compile_glob(g).match(p))


def touched(changed: list[str], globs: list[str]) -> list[str]:
    """The subset of changed paths matching any corpus glob (changed order preserved)."""
    return [c for c in changed if any(match_glob(g, c) for g in globs)]


def since_skip(card: str, workspace, since: str, corpus_globs: list[str], *,
               out=None, err=None) -> int | None:
    """The reusable --since gate. Returns an exit code (0) to short-circuit the run
    with an "unchanged" report — i.e. none of the card's corpus is in the diff, so
    the debt provably cannot have moved. Returns None to signal "fall through to a
    full scan": either the corpus WAS touched, or the diff could not be computed (an
    unresolvable ref / git failure), in which case a full scan is the safe choice —
    silently reporting "unchanged" off a failed diff would be a false clean.
    """
    out = out or sys.stdout
    err = err or sys.stderr
    try:
        changed = changed_paths(workspace, since)
    except RuntimeError as exc:
        print(f"{card}: --since {since} diff failed ({exc}); scanning fully", file=err)
        return None
    hits = touched(changed, corpus_globs)
    if hits:
        print(f"{card}: {len(hits)} corpus file(s) changed since {since}; scanning", file=err)
        return None
    print(f"{card}: unchanged since {since} (no corpus file in the changed set)", file=out)
    return 0
