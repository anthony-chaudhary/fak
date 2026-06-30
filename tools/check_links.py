#!/usr/bin/env python3
"""Broken-link gate: a front-door doc must not ship a dead relative link.

The parallel fleet deletes/moves docs and leaves dangling links behind; a public
visitor hits the front door first, so a dead link there is an embarrassment. This
resolves every relative markdown link in the front-door set and refuses any that
does not point at a real file or directory.

It also catches the *inline-code* reference class — `` `fak/GROWTH.md §2` `` or
`` `POSITION-….md §5` `` — a doc citing a non-existent file as authority that the
markdown-link regex never sees (issue #288). The first whitespace-delimited token
of each inline-code span, when it names a repo `.md` file, must resolve too.

It also catches the *scrub-private* reference class — a public front-door doc
citing a file that is deleted from every public copy (``CLAUDE.md``, issue #258).
The target EXISTS in the canonical repo, so the dead-link checks above pass it,
but it ships a dead reference in the public export. Such a reference is flagged
wherever it appears in a public front-door doc — markdown link or inline code.

Scope (front door): README, START-HERE, INSTALL, INDEX, AGENTS, CLAUDE,
CONTRIBUTING, SECURITY, CLA, docs/index.md, docs/FAQ.md.

Modes:
  --audit-staged   check front-door files that are staged (added/modified) — the
                   gate: if you touch a front-door file, it must be link-clean.
  --audit-tree     check the whole front-door set (CI / DoD).
Exit: 0 clean, 1 dead link, 2 could-not-run (hook fails open on 2).
Escape (staged): ALLOW_BAD_LINK=1.
"""
from __future__ import annotations
import argparse
import os
import posixpath
import re
import subprocess
from dispatch_worker import install_no_window_subprocess_defaults
import sys
install_no_window_subprocess_defaults(subprocess)

FRONT_DOOR = [
    "README.md", "START-HERE.md", "INSTALL.md", "INDEX.md", "AGENTS.md",
    "CLAUDE.md", "CONTRIBUTING.md", "SECURITY.md", "CLA.md",
    "LEARNING-PATH.md", "docs/index.md", "docs/FAQ.md",
]
LINK_RE = re.compile(r"\]\(([^)]+)\)")
INLINE_RE = re.compile(r"`([^`]+)`")
MD_TOKEN_RE = re.compile(r"\A[\w./-]+\.md\Z")

# Front-door-relevant `.md` files deleted from every PUBLIC copy that have NO
# public role, so a public front-door doc must never cite them (issue #258:
# CONTRIBUTING.md -> CLAUDE.md). The target EXISTS in canonical, so a plain
# "does the target exist?" check passes it — but the reference is dead in the
# public export. See scrub_public_copy.py DELETE_PATHS.
#   - CLAUDE.md: "Claude Code thin pointer to AGENTS.md; private side only"
#     — private side only, so it is never the correct public reference target.
#   - PUBLIC-SCRUB-POLICY.md: the narrative map of what is hidden and why — pure
#     private (absent from this tree today; a forward guard if ever written).
# AGENTS.md is deliberately NOT here: it is the agent-orientation SOURCE (not a
# mirror), so whether the public copy ships an agent-orientation page — and what
# START-HERE/INDEX should point at — is a separate public/private policy decision.
SCRUB_PRIVATE_MD = {"CLAUDE.md", "PUBLIC-SCRUB-POLICY.md"}


def _git(args, root):
    return subprocess.run(["git", "-C", root] + args, capture_output=True, text=True)


def _resolves(root, d, ref):
    """True if a relative file reference points at a real path.

    Honors the repo convention that a `fak/X` label denotes X at the repo root
    (the repo root IS the fak Go module; there is no `fak/` subdirectory)."""
    cands = [posixpath.normpath(posixpath.join(d, ref)), ref]
    if ref.startswith("fak/"):
        cands.append(ref[len("fak/"):])
    return any(os.path.exists(os.path.join(root, c)) for c in cands)


def _staged_frontdoor(root):
    r = _git(["diff", "--cached", "--name-status", "--diff-filter=ACMR"], root)
    if r.returncode != 0:
        return None
    paths = {ln.split("\t")[-1] for ln in r.stdout.splitlines() if ln.strip()}
    return [f for f in FRONT_DOOR if f in paths]


def _dead_links(root, f):
    fp = os.path.join(root, f)
    try:
        with open(fp, encoding="utf-8") as fh:
            txt = fh.read()
    except OSError:
        return []  # file not present (e.g. deleted) — nothing to check
    d = posixpath.dirname(f)
    out = []
    for link in LINK_RE.findall(txt):
        if link.startswith(("http://", "https://", "mailto:", "#", "/", "data:")):
            continue
        path = link.split("#")[0].split("?")[0]
        if not path:
            continue
        if _resolves(root, d, path):
            continue
        out.append((link, path))
    return out


def _dead_inline_refs(root, f):
    """Inline-code references (`fak/GROWTH.md §2`) that name a repo `.md` file but
    do not resolve — the dead-reference class issue #288 was filed for: a doc
    citing a non-existent file as authority, invisible to LINK_RE."""
    fp = os.path.join(root, f)
    try:
        with open(fp, encoding="utf-8") as fh:
            txt = fh.read()
    except OSError:
        return []
    d = posixpath.dirname(f)
    out, seen = [], set()
    for span in INLINE_RE.findall(txt):
        parts = span.split()
        if not parts:
            continue
        ref = parts[0].split("#")[0]
        if not MD_TOKEN_RE.match(ref) or ref in seen:
            continue
        seen.add(ref)
        if _resolves(root, d, ref):
            continue
        out.append((span, ref))
    return out


def _targets_scrub_private(ref, d):
    """True if a markdown-link target or inline-code token resolves to a
    scrub-private `.md` file (deleted from the public copy). Honors the `fak/X`
    -> repo-root and subdir-relative conventions, matching _resolves, so
    `CLAUDE.md`, `./CLAUDE.md`, `fak/CLAUDE.md`, and a `docs/` doc's
    `../CLAUDE.md` are all recognized."""
    path = ref.split("#")[0].split("?")[0]
    if not path:
        return False
    cands = [posixpath.normpath(posixpath.join(d, path)), path]
    if path.startswith("fak/"):
        cands.append(path[len("fak/"):])
    return any(c in SCRUB_PRIVATE_MD for c in cands)


def _scrub_private_refs(root, f):
    """References to a scrub-private `.md` file. The target EXISTS in canonical,
    so _dead_links / _dead_inline_refs pass it — but it ships a DEAD reference in
    the public export (the issue #258 defect class: a public contributor doc
    pointing at a file the scrubber deletes). Catches both `[x](CLAUDE.md)` links
    and `` `CLAUDE.md` `` / `` `fak/CLAUDE.md` `` inline code. A scrub-private
    file citing another scrub-private file is not a public leak (both die
    together), so the scrub-private files themselves are exempt."""
    if posixpath.basename(f) in SCRUB_PRIVATE_MD:
        return []
    fp = os.path.join(root, f)
    try:
        with open(fp, encoding="utf-8") as fh:
            txt = fh.read()
    except OSError:
        return []
    d = posixpath.dirname(f)
    out = []
    for link in LINK_RE.findall(txt):
        if _targets_scrub_private(link, d):
            out.append((f"]({link})", link.split("#")[0].split("?")[0]))
    for span in INLINE_RE.findall(txt):
        parts = span.split()
        if parts and _targets_scrub_private(parts[0], d):
            out.append((f"`{span}`", parts[0]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--audit-staged", action="store_true")
    g.add_argument("--audit-tree", action="store_true")
    ap.add_argument("--root", default=".")
    a = ap.parse_args()
    root = os.path.abspath(a.root)

    if a.audit_staged and os.environ.get("ALLOW_BAD_LINK") == "1":
        print("broken-link: skipped (ALLOW_BAD_LINK=1).")
        return 0

    if a.audit_staged:
        files = _staged_frontdoor(root)
        if files is None:
            print("BROKEN_LINK (warn): git not available; check skipped.", file=sys.stderr)
            return 2
        scope = "staged front-door files"
    else:
        files = [f for f in FRONT_DOOR if os.path.exists(os.path.join(root, f))]
        scope = "front-door set"

    findings = []
    for f in files:
        for link, tgt in _dead_links(root, f):
            findings.append((f, f"]({link})", tgt))
        for span, ref in _dead_inline_refs(root, f):
            findings.append((f, f"`{span}`", ref))
        for cite, tgt in _scrub_private_refs(root, f):
            findings.append((f, cite, tgt))
    if not findings:
        print(f"broken-link: clean ({scope}).")
        return 0

    print(f"BROKEN_LINK: {len(findings)} dead reference(s) in front-door docs:", file=sys.stderr)
    for f, cite, tgt in findings:
        print(f"  {f}: {cite}  ->  missing {tgt}", file=sys.stderr)
    print("  fix: repoint to a file that ships publicly, or remove the reference "
          "(the target may be missing, moved, or private-side-only/scrubbed).",
          file=sys.stderr)
    if a.audit_staged:
        print("  override once: ALLOW_BAD_LINK=1 <git cmd>.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
