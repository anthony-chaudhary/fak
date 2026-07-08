#!/usr/bin/env python3
"""Cache-headline gate: a cache "win" number must name its plane + provenance.

Epic #1490 turns cache attribution honest by default: provider prompt-cache
rebates, fak-owned kernel KV reuse, and O(1) context savings are DIFFERENT
mechanisms with different trust, cost, and correctness semantics, and a single
blended headline hides which one fired (Next-50 item 46,
`docs/cache-frontier/DEFAULT-ENABLEMENT-NEXT-50.md` legacy assumption #1:
"Provider cache is 99% of the story"). This gate makes the legacy phrasing
mechanically un-shippable: a cache headline — "99% cache", "cache win" — that
omits which PLANE (provider / kernel / context / forecast) and provenance the
number belongs to fails review.

The rule is deliberately narrow to stay false-positive-free (the same design as
tools/check_provenance_labels.py): it watches a fixed set of KNOWN legacy-
headline shapes (HEADLINE_PATTERNS below) and flags only a line that carries one
of them WITHOUT any plane/provenance label (LABEL_RE) co-located. A labeled
headline — "provider prompt-cache rebate (OBSERVED)", "kernel KV reuse
(WITNESSED)", "50%-99% cache hit rate (provenance: SGLang KV plane)" — passes,
and the carve-outs (ALLOW_PATTERNS) let a line that quotes the legacy phrasing to
REMOVE it (a "legacy assumption to remove", a meta-critique) through. This is the
standing guard behind the epic's per-mechanism attribution work: once the honest
split lands, an un-linted blended "99% cache" re-introduces exactly the legacy the
epic removed.

It is NOT a semantic check — it cannot tell whether a cache number is TRUE (that
is owned by internal/vcachescore and the cachevalue reports). It only enforces the
label: a cache-win headline names its plane and provenance, or it does not ship.

Modes:
  --audit-staged   scan staged additions from the index (the pre-commit hook).
  --audit-tree     scan the whole tracked tree (CI / hygiene backstop).

Exit: 0 = clean, 1 = violation, 2 = could not run (git error). The pre-commit
hook treats anything but 1 as fail-open so a broken check never wedges commits.

Escape hatch (staged mode): set ALLOW_CACHE_HEADLINE_DRIFT=1 to override once
(e.g. a genuine new context) — then refine the LABEL_RE / carve-out patterns here
so the override is not needed again.
"""
from __future__ import annotations
import argparse
import os
import re
import subprocess
import sys

_CREATE_NO_WINDOW = 0x08000000


def _win_creationflags() -> int:
    return _CREATE_NO_WINDOW if os.name == "nt" else 0


# The KNOWN legacy cache-headline shapes. A line is a CANDIDATE if it matches one
# of these; it VIOLATES only when it ALSO lacks a plane/provenance label (below).
# Each is anchored to the issue's literal targets ("99% cache", "cache win") and
# the DEFAULT-ENABLEMENT-NEXT-50.md legacy phrasing ("cache is 99% of the story").
# `[\s-]+ wins?` / `\s*% cache` are word-boundaried so "cache window(ed)" and the
# hyphenated stat "99%-cache-hit" (a specific hit-rate, not a blended win) are NOT
# matched — those are legitimate and out of scope.
HEADLINE_PATTERNS = [
    # a bare "cache win" / "cache-win" / "cache wins" headline
    re.compile(r"\bcache[\s-]+wins?\b", re.I),
    # a percentage directly qualifying "cache" as a blended win: "99% cache"
    re.compile(r"\b\d{2,3}\s*%\s*cache\b", re.I),
    # the legacy "(provider) cache is 99% of the story" phrasing
    re.compile(r"\bcache\s+is\s+\d{1,3}\s*%", re.I),
    re.compile(r"\b\d{1,3}\s*%\s+of\s+the\s+story\b", re.I),
]

# A co-located plane/provenance label makes a headline honest (the issue's own
# bar: "'99% cache' … WITHOUT provider/kernel/context labels fails review"). The
# four named planes + the provenance verbs + the concept-table plane vocabulary
# (DEFAULT-ENABLEMENT-NEXT-50.md "Concept Refresh"). Broad on purpose: the safe
# direction is to fire ONLY on a truly unattributed headline.
LABEL_RE = re.compile(
    r"\b(?:"
    r"provider|kernel|context|forecast(?:ed)?|"          # the four named planes
    r"observed|witnessed|decision|measured|modeled|hypothesis|"  # provenance verbs
    r"provenance|per-plane|per-mechanism|plane|"          # attribution vocabulary
    r"o\(1\)|vcache|v-cache|radixkv|radix|paged|prefix|"  # concept-table planes
    r"radixattention|pagedattention|sglang|vllm|llama|"   # external-engine KV planes
    r"prompt-?cache|engine|rebate|cost[\s/-]*latency|"    # provider-plane vocabulary
    r"owner|attribution|reuse|kv"                         # split-owner / KV vocabulary
    r")\b",
    re.I,
)

# Carve-outs: a candidate line that is ALLOWED despite lacking a label. A line
# that QUOTES the legacy phrasing to REMOVE/critique it is not asserting a blended
# headline (mirrors check_provenance_labels' meta-critique carve-out).
ALLOW_PATTERNS = [
    re.compile(r"\blegacy\b", re.I),          # "legacy assumption to remove"
    re.compile(r"\bremove\b", re.I),
    re.compile(r"\bmislead", re.I),
    re.compile(r"\bhide[sn]?\b", re.I),        # "hides which mechanism fired"
    re.compile(r"\bblend(?:ed|s)?\b", re.I),   # naming the blended-number defect
    re.compile(r"\bfails? review\b", re.I),    # the lint's own spec wording
    re.compile(r"\bomits?\b", re.I),
]

FIX = ('name the plane + provenance the number belongs to — provider prompt-cache '
       '(OBSERVED rebate), fak kernel KV reuse (WITNESSED), or O(1) context '
       '(WITNESSED/FORECAST) — never a blended "99% cache" / "cache win"')

# Surfaces to scan. Front-facing docs + claims; NOT generated mirrors
# (llms-full.txt regenerates from llms.txt + docs) and NOT dated release notes
# under docs/releases/ (immutable history).
SCAN_GLOBS = (
    "*.md", "*.html", "*.txt",
)
SKIP_PREFIXES = (
    "docs/releases/",        # dated history, immutable
    "vendor/", "node_modules/",
)
SKIP_BASENAMES = (
    "llms-full.txt",             # generated mirror (regenerates from llms.txt + docs)
    "check_cache_headlines.py",  # this file's own doc-strings name the patterns
    "check_cache_headlines_test.py",  # its fixtures name the patterns
)


def _tracked_files(root: str) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z", *SCAN_GLOBS],
        cwd=root, capture_output=True, text=True, check=True,
        creationflags=_win_creationflags(),
    )
    return [p for p in out.stdout.split("\0") if p]


def _staged_files(root: str) -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=AM", "-z"],
        cwd=root, capture_output=True, text=True, check=True,
        creationflags=_win_creationflags(),
    )
    files = [p for p in out.stdout.split("\0") if p]
    keep = []
    for p in files:
        base = os.path.basename(p)
        if any(p.startswith(pre) for pre in SKIP_PREFIXES):
            continue
        if base in SKIP_BASENAMES:
            continue
        if any(_glob_match(base, g) for g in SCAN_GLOBS):
            keep.append(p)
    return keep


def _glob_match(name: str, glob: str) -> bool:
    # only "*.ext" globs are used here
    return name.endswith(glob[1:]) if glob.startswith("*") else name == glob


def _line_violates(line: str) -> bool:
    if not any(p.search(line) for p in HEADLINE_PATTERNS):
        return False
    if LABEL_RE.search(line):
        return False
    if any(p.search(line) for p in ALLOW_PATTERNS):
        return False
    return True


def scan_file(root: str, relpath: str) -> list[dict]:
    base = os.path.basename(relpath)
    if any(relpath.startswith(pre) for pre in SKIP_PREFIXES):
        return []
    if base in SKIP_BASENAMES:
        return []
    full = os.path.join(root, relpath)
    try:
        with open(full, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return []
    hits = []
    for i, line in enumerate(lines, 1):
        if _line_violates(line):
            hits.append({"file": relpath, "line": i,
                         "text": line.strip()[:160], "fix": FIX})
    return hits


_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def _staged_added_lines(root: str, relpath: str) -> list[tuple[int, str]]:
    """Return (new-line-number, text) for staged additions in relpath.

    Pre-commit must judge the index, not the working tree. Reading the whole file
    would let an old, unrelated line block a clean staged edit; reading the diff
    keeps the hook scoped to what the commit introduces.
    """
    out = subprocess.run(
        ["git", "diff", "--cached", "--unified=0", "--no-ext-diff", "--", relpath],
        cwd=root, capture_output=True, text=True, check=True,
        creationflags=_win_creationflags(),
    )
    lines: list[tuple[int, str]] = []
    new_line: int | None = None
    for raw in out.stdout.splitlines():
        m = _HUNK_RE.match(raw)
        if m:
            new_line = int(m.group(1))
            continue
        if new_line is None:
            continue
        if raw.startswith("+++"):
            continue
        if raw.startswith("+"):
            lines.append((new_line, raw[1:]))
            new_line += 1
            continue
        if raw.startswith("-"):
            continue
        if raw.startswith(" "):
            new_line += 1
    return lines


def scan_staged_file(root: str, relpath: str) -> list[dict]:
    base = os.path.basename(relpath)
    if any(relpath.startswith(pre) for pre in SKIP_PREFIXES):
        return []
    if base in SKIP_BASENAMES:
        return []
    hits = []
    for line_no, line in _staged_added_lines(root, relpath):
        if _line_violates(line):
            hits.append({"file": relpath, "line": line_no,
                         "text": line.strip()[:160], "fix": FIX})
    return hits


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--audit-tree", action="store_true",
                   help="scan the whole tracked tree (CI / hygiene)")
    g.add_argument("--audit-staged", action="store_true",
                   help="scan staged additions (pre-commit hook)")
    ap.add_argument("--root", default=".", help="repo root")
    args = ap.parse_args(argv)

    root = os.path.abspath(args.root)
    try:
        files = _staged_files(root) if args.audit_staged else _tracked_files(root)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"cache-headlines: could not run git: {e}", file=sys.stderr)
        return 2

    all_hits: list[dict] = []
    for rel in files:
        if args.audit_staged:
            all_hits.extend(scan_staged_file(root, rel))
        else:
            all_hits.extend(scan_file(root, rel))

    if not all_hits:
        scope = "staged" if args.audit_staged else "tracked tree"
        print(f"cache-headlines: clean ({scope}); every cache headline names its "
              "plane + provenance.")
        return 0

    if args.audit_staged and os.environ.get("ALLOW_CACHE_HEADLINE_DRIFT") == "1":
        print("cache-headlines: ALLOW_CACHE_HEADLINE_DRIFT=1 set — overriding "
              f"{len(all_hits)} hit(s) once.", file=sys.stderr)
        return 0

    print(f"CACHE_HEADLINE: {len(all_hits)} cache headline(s) without a "
          "plane/provenance label:", file=sys.stderr)
    for h in all_hits:
        print(f"  {h['file']}:{h['line']}: {h['text']}", file=sys.stderr)
        print(f"    fix: {h['fix']}", file=sys.stderr)
    print("  override once (staged): ALLOW_CACHE_HEADLINE_DRIFT=1 <git cmd>.",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
