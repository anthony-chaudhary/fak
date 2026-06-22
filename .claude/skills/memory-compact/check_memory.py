#!/usr/bin/env python3
"""check_memory.py — the integrity witness for a Claude Code auto-memory store.

The don't-trust-the-narration doctrine turned on the memory store itself: do not
trust "I compacted it" — re-derive every invariant from disk and let the EXIT CODE
be the verdict.

Checks, against the memory dir (default: the dir this script's sibling MEMORY.md
lives in, or --dir):

  1. HARNESS CAP   MEMORY.md must be <= 200 lines AND <= 25_000 bytes. The Claude
     Code harness loads only the first 200 lines / 25KB of MEMORY.md at session
     start ("whichever comes first"); anything past that silently never loads.
     (We gate at 25_000, not 25*1024, to keep a safety margin under the real cap.)
  2. BIJECTION     Every topic `.md` on disk is referenced exactly once across the
     index files (MEMORY.md + any MEMORY_*.md sibling, e.g. MEMORY_archive.md).
     No orphan files (on disk, unreferenced), no orphan links (referenced, absent),
     no duplicate references.
  3. DANGLING      Reports `[[wiki-links]]` that resolve to no file on disk. These
     are TOLERATED by the memory spec (a `[[name]]` marks a write-later note, not
     an error), so this is a WARNING — it does not fail the run unless --strict.

Exit code: 0 = clean (caps + bijection pass). Non-zero = a hard invariant broke.
Pure stdlib; read-only (never writes). Run it after any compaction pass, and/or
wire it into a stop-hook so a session gets a verdict instead of eyeballing it.

Usage:
  python check_memory.py [--dir PATH] [--max-lines 200] [--max-bytes 25000] [--strict] [--json]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Windows consoles default to cp1252 and choke on non-Latin-1 glyphs. Force UTF-8
# so the witness prints the same everywhere (and never crashes on its own output).
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

# A markdown link to a local .md file: ](slug.md)
_LINK = re.compile(r"\]\(([A-Za-z0-9_./-]+\.md)\)")
# A wiki-link: [[slug]]  (no .md suffix by convention)
_WIKI = re.compile(r"\[\[([A-Za-z0-9_-]+)\]\]")
# Index files: MEMORY.md and any MEMORY_*.md sibling (archive tiers).
_INDEX_GLOB = "MEMORY*.md"


def _resolve_dir(arg: str | None) -> Path:
    if arg:
        return Path(arg).expanduser().resolve()
    # default: the dir this script sits next to MEMORY.md in, else cwd
    here = Path(__file__).resolve().parent
    if (here / "MEMORY.md").exists():
        return here
    return Path.cwd()


def check(mem_dir: Path, *, max_lines: int, max_bytes: int) -> dict:
    primary = mem_dir / "MEMORY.md"
    if not primary.exists():
        return {"ok": False, "fatal": f"no MEMORY.md in {mem_dir}"}

    index_files = sorted(mem_dir.glob(_INDEX_GLOB))
    index_names = {p.name for p in index_files}

    # --- 1. CAP (primary MEMORY.md only — that's the auto-loaded file) ---
    raw = primary.read_bytes()
    n_lines = raw.count(b"\n") + (0 if raw.endswith(b"\n") or not raw else 1)
    n_bytes = len(raw)
    cap_lines_ok = n_lines <= max_lines
    cap_bytes_ok = n_bytes <= max_bytes

    # --- 2. BIJECTION (across all index files) ---
    # references = every ](X.md) across the index files, excluding the index
    # files themselves (a header may link MEMORY_archive.md as a pointer) and the
    # literal `slug.md` format-example in the header.
    refs: list[str] = []
    for idx in index_files:
        for m in _LINK.finditer(idx.read_text(encoding="utf-8", errors="replace")):
            target = m.group(1)
            # strip any leading ./ and keep only the basename for matching
            name = target.split("/")[-1]
            if name in index_names or name == "slug.md":
                continue
            refs.append(name)

    on_disk = {
        p.name
        for p in mem_dir.glob("*.md")
        if p.name not in index_names
    }
    ref_counts: dict[str, int] = {}
    for r in refs:
        ref_counts[r] = ref_counts.get(r, 0) + 1

    ref_set = set(ref_counts)
    orphan_links = sorted(ref_set - on_disk)          # referenced, not on disk
    orphan_files = sorted(on_disk - ref_set)          # on disk, unreferenced
    dup_refs = sorted(n for n, c in ref_counts.items() if c > 1)

    bijection_ok = not orphan_links and not orphan_files and not dup_refs

    # --- 3. DANGLING wiki-links (advisory) ---
    disk_slugs = {p.stem for p in mem_dir.glob("*.md") if p.name not in index_names}
    dangling: dict[str, list[str]] = {}
    for p in mem_dir.glob("*.md"):
        text = p.read_text(encoding="utf-8", errors="replace")
        for m in _WIKI.finditer(text):
            slug = m.group(1)
            if slug not in disk_slugs:
                dangling.setdefault(slug, []).append(p.name)

    return {
        "dir": str(mem_dir),
        "cap": {
            "lines": n_lines, "max_lines": max_lines, "lines_ok": cap_lines_ok,
            "bytes": n_bytes, "max_bytes": max_bytes, "bytes_ok": cap_bytes_ok,
            "ok": cap_lines_ok and cap_bytes_ok,
        },
        "bijection": {
            "refs": len(ref_set), "disk": len(on_disk),
            "orphan_links": orphan_links, "orphan_files": orphan_files,
            "duplicate_refs": dup_refs, "ok": bijection_ok,
        },
        "dangling_wikilinks": {k: sorted(v) for k, v in sorted(dangling.items())},
        "index_files": sorted(index_names),
        "ok": (cap_lines_ok and cap_bytes_ok and bijection_ok),
    }


def _fmt_human(r: dict) -> str:
    if r.get("fatal"):
        return f"FATAL: {r['fatal']}"
    L = []
    c = r["cap"]
    L.append("=== HARNESS CAP (first 200 lines / 25KB load each session) ===")
    L.append(
        f"  lines: {c['lines']}/{c['max_lines']}  "
        + ("PASS" if c["lines_ok"] else f"FAIL +{c['lines']-c['max_lines']}")
    )
    L.append(
        f"  bytes: {c['bytes']}/{c['max_bytes']}  "
        + ("PASS" if c["bytes_ok"] else f"FAIL +{c['bytes']-c['max_bytes']}")
        + (f"  ({c['max_bytes']-c['bytes']}B margin)" if c["bytes_ok"] else "")
    )
    b = r["bijection"]
    L.append("=== BIJECTION (every topic .md referenced once across index files) ===")
    L.append(
        f"  refs {b['refs']} / disk {b['disk']}  | "
        f"orphan-links={len(b['orphan_links'])} "
        f"orphan-files={len(b['orphan_files'])} "
        f"dupes={len(b['duplicate_refs'])}  "
        + ("INTACT" if b["ok"] else "BROKEN")
    )
    for k, items in (
        ("orphan-link (referenced, not on disk)", b["orphan_links"]),
        ("orphan-file (on disk, unreferenced)", b["orphan_files"]),
        ("duplicate-ref (linked >1x)", b["duplicate_refs"]),
    ):
        for it in items:
            L.append(f"    ! {k}: {it}")
    d = r["dangling_wikilinks"]
    L.append(f"=== DANGLING [[wiki-links]] (advisory, tolerated) : {len(d)} ===")
    for slug, files in list(d.items())[:20]:
        L.append(f"    ~ [[{slug}]] <- {', '.join(files)}")
    if len(d) > 20:
        L.append(f"    ... and {len(d)-20} more")
    L.append("")
    L.append("VERDICT: " + ("CLEAN" if r["ok"] else "BROKEN (a hard invariant failed)"))
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Integrity witness for a Claude Code auto-memory store.")
    ap.add_argument("--dir", default=None, help="memory dir (default: sibling of MEMORY.md, else cwd)")
    ap.add_argument("--max-lines", type=int, default=200)
    ap.add_argument("--max-bytes", type=int, default=25_000)
    ap.add_argument("--strict", action="store_true", help="also fail on dangling wiki-links")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    mem_dir = _resolve_dir(a.dir)
    r = check(mem_dir, max_lines=a.max_lines, max_bytes=a.max_bytes)
    if a.json:
        print(json.dumps(r, indent=2))
    else:
        print(_fmt_human(r))

    if r.get("fatal"):
        return 2
    ok = r["ok"] and (not a.strict or not r["dangling_wikilinks"])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
