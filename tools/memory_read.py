#!/usr/bin/env python3
r"""memory_read.py — backend-agnostic reader for the committed fleet auto-memory mirror.

Claude Code workers are handed the node-local auto-memory store automatically each
session; opencode workers are NOT — they start cold every run unless their prompt
explicitly reads the committed mirror at ``.claude/memory/`` (issue #421). That mirror
ships with the tree like any other tracked file (kept in sync by ``sync_memory.py``),
so it is the one fleet-knowledge source both backends *can* reach the same way.

This helper IS that one read path: it loads ``MEMORY.md`` (the hot index) plus the
per-fact ``*.md`` files the index references, and prints a single bounded digest a
worker can be handed inline — so the read path is not agent-prompt-specific.

    python tools/memory_read.py                 # digest of .claude/memory/ (repo mirror)
    python tools/memory_read.py --store DIR      # read a different store
    python tools/memory_read.py --index-only     # MEMORY.md only, skip per-fact bodies
    python tools/memory_read.py --max-bytes N     # cap total digest size (default 60000)

Exit is ALWAYS 0 when the store is simply absent (a fresh node, or a scrubbed public
clone where ``.claude/memory/`` is not shipped): it prints a one-line "no mirror" note
so a worker prompt that pipes it in degrades to a harmless no-op rather than erroring.

Pure/testable: ``render_digest`` takes a directory and returns a string; the CLI wrapper
resolves the default store and prints it.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except (AttributeError, ValueError):
    pass

# The committed mirror, repo-root-relative (see .claude/memory/README.md). Same
# constant the recall auditor binds to, so both tools agree on where the store lives.
STORE_REL = ".claude/memory"

# Index/doc files that carry no bindable fact of their own — never expanded as facts.
_NON_FACT = {"MEMORY.md", "MEMORY_archive.md", "README.md"}

# A markdown link to a sibling .md file: [Title](file.md) — the index line shape.
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)#\s]+\.md)(?:#[^)]*)?\)")


def repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    return here.parent.parent


def default_store(root: Path | None = None) -> Path:
    return (root or repo_root()) / STORE_REL


def parse_index(index_text: str) -> list[tuple[str, str]]:
    """Extract ``(title, filename)`` for each per-fact file the index links to.

    Order-preserving and deduped on filename; the non-fact index/doc files are
    dropped so the digest expands only real memory facts.
    """
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for title, fname in _LINK_RE.findall(index_text):
        # Only same-directory fact files (no path components, not a non-fact doc).
        if "/" in fname or "\\" in fname or fname in _NON_FACT or fname in seen:
            continue
        seen.add(fname)
        out.append((title.strip(), fname))
    return out


def strip_frontmatter(text: str) -> str:
    """Return the fact body with a leading ``---`` YAML frontmatter block removed."""
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            nl = text.find("\n", end + 1)
            if nl != -1:
                return text[nl + 1:].lstrip("\n")
    return text


def render_digest(store_dir: Path, *, index_only: bool = False,
                  max_bytes: int = 60000) -> str:
    """Render the committed-memory digest for ``store_dir``. Pure modulo file reads.

    Returns a single string: a header, the MEMORY.md index verbatim, then each
    referenced per-fact body (frontmatter stripped), stopping once ``max_bytes`` of
    fact bodies have been emitted and noting how many facts were omitted.
    """
    index_path = store_dir / "MEMORY.md"
    if not index_path.is_file():
        return (f"(no committed memory mirror at {store_dir.as_posix()} — "
                "fresh node or scrubbed clone; nothing to orient from)\n")

    index_text = index_path.read_text(encoding="utf-8")
    parts = [
        f"# Fleet memory (committed mirror: {STORE_REL}) — read-only orientation",
        "",
        index_text.rstrip("\n"),
    ]
    if index_only:
        parts.append("")
        return "\n".join(parts) + "\n"

    facts = parse_index(index_text)
    parts += ["", "---", ""]
    budget = max_bytes
    emitted = 0
    omitted = 0
    for title, fname in facts:
        fpath = store_dir / fname
        if not fpath.is_file():
            omitted += 1
            continue
        body = strip_frontmatter(fpath.read_text(encoding="utf-8")).rstrip("\n")
        block = f"## {title} ({fname})\n\n{body}\n"
        if budget - len(block) < 0 and emitted > 0:
            omitted += 1
            continue
        parts.append(block)
        budget -= len(block)
        emitted += 1
    if omitted:
        parts.append(f"…({omitted} more fact file(s) omitted — read directly from "
                     f"{STORE_REL}/ if needed)")
    return "\n".join(parts).rstrip("\n") + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Print a bounded digest of the committed fleet auto-memory mirror "
                    "(MEMORY.md + the per-fact files it references) for either backend.")
    ap.add_argument("--store", default="", help=f"memory store dir (default: {STORE_REL})")
    ap.add_argument("--index-only", action="store_true",
                    help="emit MEMORY.md only, skip per-fact bodies")
    ap.add_argument("--max-bytes", type=int, default=60000,
                    help="cap total per-fact body bytes emitted (default 60000)")
    args = ap.parse_args(argv)

    store = Path(args.store).resolve() if args.store else default_store()
    sys.stdout.write(render_digest(store, index_only=args.index_only,
                                   max_bytes=args.max_bytes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
