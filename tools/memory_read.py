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

LESSONS LEDGER (#2141, gen/next — gated). A *lesson* is a hard-won recovery/scar
published ONCE into ``<store>/lessons/<slug>.md`` and auto-injected only to a peer
whose session CONTEXT matches the lesson's TRIGGER — before that peer hits the same
wall. Lessons live in a subdirectory ON PURPOSE: both digest renderers (this file and
``internal/memoryread``) drop index links carrying path components, so a published
lesson can never leak ungated through the untargeted fact walk — the only way out is
this trigger-matched, read-time-re-verified path. Lesson file schema (frontmatter):

    ---
    name: bash-git-hang-use-powershell
    description: git via Bash hangs on this host -- use PowerShell
    trigger:                 # ALL keys must match the session context (fail closed)
      host_os: windows        #   windows | linux | darwin
      host: some-node          #   exact hostname (case-insensitive)
      path_glob: cmd/fak/**    #   fnmatch against cwd relative to the repo root
      tool: bash               #   only matches when the caller passes --context tool=...
      refusal_token: OFF_TRUNK #   only matches when passed via --context refusal_token=...
    verify:                  # read-time re-verify (dos_recall-shaped): fail => WITHHELD
      path: docs/repo-guard.md #   repo-root-relative file that must still exist
      contains: OFF_TRUNK      #   optional substring that must still be present
    ---
    The lesson body a matching peer receives.

Gate (env ``FAK_LESSONS``, or ``--lessons``): ``shadow`` (default) computes the match
and prints a one-line would-inject readout but injects NO body; ``live`` injects the
matched, verify-fresh lesson bodies; ``off`` is silent. A lesson whose ``verify`` probe
fails is WITHHELD with a note, never asserted. Session context is auto-derived
(host, host_os, cwd-relative path) and extendable via repeated ``--context key=value``.
Promotion toward live-by-default needs shadow readouts showing precise matches on real
peer sessions; the invalidating assumption is that a trigger is stably computable from
session-start context alone.
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import platform
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

# Published fleet lessons live in a subdirectory of the mirror (see module docstring:
# the subdirectory is the containment — untargeted fact walks never expand it).
LESSONS_REL = "lessons"

# Shadow-first gate, mirroring memory_cotravel's FAK_MEMORY_COTRAVEL discipline:
# observe the match in the field before any body is injected.
_LESSONS_GATES = ("shadow", "live", "off")
_DEFAULT_LESSONS_GATE = "shadow"

# Trigger keys a lesson may bind (issue #2141: host, path glob, tool, refusal token).
# Any OTHER key fails the match closed — an unknown trigger withholds, never over-injects.
_TRIGGER_KEYS = {"host", "host_os", "path_glob", "tool", "refusal_token"}

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


def lessons_gate() -> str:
    g = (os.environ.get("FAK_LESSONS") or _DEFAULT_LESSONS_GATE).strip().lower()
    return g if g in _LESSONS_GATES else _DEFAULT_LESSONS_GATE


def default_context(root: Path | None = None) -> dict[str, str]:
    """The session-start context a lesson trigger is matched against.

    Auto-derived only from what is knowable BEFORE the wall is hit: hostname, host
    OS family, and cwd relative to the repo root. ``tool`` / ``refusal_token`` are
    never guessed — a caller that knows them passes ``--context``.
    """
    root = (root or repo_root()).resolve()
    osname = {"win32": "windows", "cygwin": "windows",
              "msys": "windows", "darwin": "darwin"}.get(sys.platform, "linux")
    cwd = Path.cwd().resolve()
    try:
        rel = cwd.relative_to(root).as_posix()
    except ValueError:
        rel = cwd.as_posix()
    return {"host": platform.node().strip().lower(), "host_os": osname,
            "path": rel if rel != "." else "."}


def parse_lesson_meta(text: str) -> dict[str, dict[str, str]]:
    """Extract ``trigger:``/``verify:`` mappings (plus top-level scalars under
    ``"meta"``) from a lesson's frontmatter. Deliberately tiny: flat two-level YAML
    only, which is all the published schema allows."""
    out: dict[str, dict[str, str]] = {}
    if not text.startswith("---"):
        return out
    end = text.find("\n---", 3)
    if end == -1:
        return out
    section: str | None = None
    for line in text[3:end].splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        key, sep, val = line.partition(":")
        if not sep:
            continue
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if indent == 0:
            if key in ("trigger", "verify") and not val:
                section = key
            else:
                section = None
                if val:
                    out.setdefault("meta", {})[key] = val
        elif section and key:
            out.setdefault(section, {})[key] = val
    return out


def trigger_matches(trigger: dict[str, str], ctx: dict[str, str]) -> bool:
    """True when EVERY trigger key matches the context (AND semantics, fail closed).

    An empty trigger never matches (a lesson without a trigger is not injectable —
    publish it as an ordinary indexed fact instead), and an unknown key or a context
    value the session did not supply withholds rather than over-injects."""
    if not trigger:
        return False
    for key, want in trigger.items():
        if key == "path_glob":
            path = ctx.get("path", "")
            if not any(fnmatch.fnmatch(c, want) for c in (path, path + "/")):
                return False
        elif key in _TRIGGER_KEYS:
            have = ctx.get(key, "")
            if not have or have.lower() != want.lower():
                return False
        else:
            return False
    return True


def verify_lesson(verify: dict[str, str], root: Path) -> tuple[bool, str]:
    """Read-time re-verify (the dos_recall shape): a failing probe means the lesson
    is WITHHELD, not asserted. ``path`` is repo-root-relative and must exist;
    ``contains`` (optional) must still appear in that file. No binding declared —
    nothing to re-verify — passes."""
    rel = verify.get("path", "")
    if not rel:
        return True, ""
    f = root / rel
    if not f.is_file():
        return False, f"verify.path missing: {rel}"
    needle = verify.get("contains", "")
    if needle:
        try:
            hay = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False, f"verify.path unreadable: {rel}"
        if needle not in hay:
            return False, f"verify.contains no longer in {rel}"
    return True, ""


def render_lessons(store_dir: Path, *, root: Path | None = None,
                   ctx: dict[str, str] | None = None, gate: str | None = None,
                   max_bytes: int = 60000) -> list[str]:
    """Digest lines for the published lessons ledger under ``store_dir/lessons/``.

    Honors the gate: ``off`` → nothing; ``shadow`` (default) → a one-line
    would-inject readout, no bodies; ``live`` → matched, verify-fresh lesson bodies.
    A matched lesson whose verify probe fails is withheld WITH a note in every
    non-off gate — the stale-withheld event is itself signal. Returns [] when no
    lessons are published, so a mirror without a ledger renders byte-identical to
    before this seam existed."""
    g = gate if gate in _LESSONS_GATES else lessons_gate()
    lessons_dir = store_dir / LESSONS_REL
    if g == "off" or not lessons_dir.is_dir():
        return []
    # verify.path is repo-root-relative; the mirror lives at <root>/.claude/memory,
    # so the store's root is two levels up. A --store fixture elsewhere resolves
    # against ITS OWN two-up dir — a dangling verify path then withholds (fail closed).
    root = (root or store_dir.parent.parent).resolve()
    ctx = ctx if ctx is not None else default_context(root)

    published = 0
    matched: list[str] = []
    withheld: list[str] = []
    blocks: list[str] = []
    budget = max_bytes
    for fpath in sorted(lessons_dir.glob("*.md")):
        try:
            text = fpath.read_text(encoding="utf-8")
        except OSError:
            continue
        published += 1
        meta = parse_lesson_meta(text)
        if not trigger_matches(meta.get("trigger", {}), ctx):
            continue
        ok, why = verify_lesson(meta.get("verify", {}), root)
        if not ok:
            withheld.append(f"{fpath.name} — {why}")
            continue
        matched.append(fpath.name)
        if g == "live":
            title = meta.get("meta", {}).get("description") or fpath.stem
            body = strip_frontmatter(text).rstrip("\n")
            block = f"### {title} ({LESSONS_REL}/{fpath.name})\n\n{body}\n"
            if budget - len(block) < 0 and blocks:
                blocks.append(f"…(lesson {fpath.name} past the {max_bytes}-byte "
                              f"lesson budget — read it from {STORE_REL}/{LESSONS_REL}/)")
                continue
            blocks.append(block)
            budget -= len(block)
    if published == 0:
        return []

    parts: list[str] = [""]
    if g == "live":
        parts.append(f"## Lessons for this context ({STORE_REL}/{LESSONS_REL}/)")
        parts.append("")
        parts += blocks or ["(no published lesson matches this session's context)"]
    else:
        would = ", ".join(matched) if matched else "none"
        parts.append(f"lessons ledger (shadow): {len(matched)} of {published} "
                     f"published lesson(s) match this session ({would}) — "
                     "set FAK_LESSONS=live to inject")
    for w in withheld:
        parts.append(f"(withheld stale lesson: {w})")
    return parts


def render_digest(store_dir: Path, *, index_only: bool = False,
                  max_bytes: int = 60000, ctx: dict[str, str] | None = None,
                  lessons: str | None = None) -> str:
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
    # Trigger-matched lessons render FIRST (right after the index): the whole point
    # is that a matching peer sees the scar BEFORE it starts working. --index-only
    # still renders them — the readout/injection is the seam, fact bodies are not.
    parts += render_lessons(store_dir, ctx=ctx, gate=lessons, max_bytes=max_bytes)
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
    ap.add_argument("--lessons", choices=_LESSONS_GATES, default=None,
                    help="lessons-ledger gate override (default: env FAK_LESSONS, "
                         f"else {_DEFAULT_LESSONS_GATE})")
    ap.add_argument("--context", action="append", default=[], metavar="KEY=VALUE",
                    help="extend/override the session context a lesson trigger is "
                         "matched against (e.g. --context tool=bash "
                         "--context refusal_token=OFF_TRUNK); repeatable")
    args = ap.parse_args(argv)

    ctx = None
    if args.context:
        ctx = default_context()
        for kv in args.context:
            key, sep, val = kv.partition("=")
            if sep and key.strip():
                ctx[key.strip()] = val.strip()
    store = Path(args.store).resolve() if args.store else default_store()
    sys.stdout.write(render_digest(store, index_only=args.index_only,
                                   max_bytes=args.max_bytes, ctx=ctx,
                                   lessons=args.lessons))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
