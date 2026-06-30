#!/usr/bin/env python3
"""Provenance-label gate: a MODELED number must never be called "measured".

fak's brand is honest, evidence-backed claims — "the kernel that doesn't believe
the agents". The one defect that contradicts the whole pitch is a *modeled* number
(a closed-form geometry / arithmetic floor) presented as a *measured* (wall-clock,
empirical) result. That is exactly what happened to the front-page WebVoyager
9.7x: `internal/webbench/geometry.go::ComputeArms` is a pure integer formula, the
CLI prints "deterministic floor, no model", yet a dozen surfaces called it
"measured". This check makes that contradiction mechanically un-shippable.

The rule is deliberately narrow to stay false-positive-free: it watches a fixed
set of KNOWN-MODELED number families (declared in MODELED_CLAIMS below, each
anchored to its BENCHMARK-AUTHORITY.md row) and flags only a line that pairs that
number with the word "measured" as a POSITIVE assertion. A negation ("not a
measured ..."), a different genuinely-measured number on the same line (the
live-race 4.1x), or a line that merely critiques a false "measured" claim are all
allowed — the patterns below encode those carve-outs explicitly.

This is the durable counterpart to the d076c47 webbench cleanup: that swept the
existing leaks; this stops the next one. It is NOT a semantic check — it cannot
tell whether the geometry model itself is sound (that is owned by
internal/webbench tests). It only enforces the label: a known-modeled number is
labeled modeled, never measured.

Modes:
  --audit-staged   scan staged additions from the index (the pre-commit hook).
  --audit-tree     scan the whole tracked tree (CI / hygiene backstop).

Exit: 0 = clean, 1 = violation, 2 = could not run (git error). The pre-commit
hook treats anything but 1 as fail-open so a broken check never wedges commits.

Escape hatch (staged mode): set ALLOW_PROVENANCE_DRIFT=1 to override once (e.g. a
genuine new context where the word legitimately appears) — then refine the
carve-out patterns here so the override is not needed again.
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


# Each MODELED claim family: a human label, the number-tokens that identify it on
# a line, and the BENCHMARK-AUTHORITY.md row that establishes it as modeled. A
# line matches this family if it contains the family's WebVoyager/webbench context
# AND one of its number tokens.
MODELED_CLAIMS = [
    {
        "label": "webbench WebVoyager prefill geometry (9.7x / 8.8x)",
        # context tokens — at least one must be on the line for the family to apply
        "context": [r"webvoyager", r"webbench", r"643[\s-]*task", r"643\s*\)"],
        # number tokens — at least one must be on the line
        "numbers": [r"9\.7\s*[x×]", r"8\.8\s*[x×]", r"8\.8\s*[–-]\s*9\.7"],
        "authority_row": "README front-page webbench hero",
        "fix": 'call it "modeled" (closed-form geometry, no wall-clock); '
               'the 9.7x is the A/C ratio vs the naive re-prefill floor',
    },
]

# A line is a VIOLATION when it asserts "measured" for a modeled family. These
# carve-outs flip a hit back to allowed: the line ALSO labels the number modeled
# (an honest modeled-vs-measured contrast), the "measured" belongs to the separate
# genuinely-measured live-race 4.1x, a negation, or a meta-critique of a false
# claim. Each pattern below is anchored to a real allowed line in the tree.
ALLOW_PATTERNS = [
    # the number is explicitly labeled modeled on the same line -> honest contrast
    re.compile(r"\bmodeled\b", re.I),
    # "measured" attaches to the separate, genuinely-measured live-race 4.1x
    re.compile(r"measured\s*\*{0,2}\s*4\.1", re.I),
    re.compile(r"/\s*measured\s+4\.1", re.I),
    # negations: "not (a wall-clock) measured", "not a ... measurement"
    re.compile(r"not\s+(?:a\s+)?(?:wall-clock\s+)?measured", re.I),
    re.compile(r"not\s+a\s+wall-clock\s+measurement", re.I),
    re.compile(r"not\s+['\"]?measured", re.I),
    # meta-critiques that call OUT a false/mislabeled claim (don't assert it)
    re.compile(r"mislabel", re.I),
    re.compile(r"false\s+['\"]?measured", re.I),
    re.compile(r"fuses\s+two\s+unrelated", re.I),
    re.compile(r"naive\s+arm\s+is\s+modeled\s+from\s+the\s+measured", re.I),
    # commit-subject / changelog quotes of this very fix
    re.compile(r"from\s+['\"]?measured['\"]?\s+to\s+modeled", re.I),
    re.compile(r'"measured"\s*->\s*', re.I),
    # a future/aspirational "end-to-end measured cost" (a real pending measurement)
    re.compile(r"end-to-end\s+measured", re.I),
]

MEASURED_RE = re.compile(r"\bmeasured\b", re.I)

# Surfaces to scan. Front-facing docs + the asset source data; NOT generated
# mirrors (llms-full.txt, the FAQ JSON-LD block regenerate from these) and NOT
# dated release notes under docs/releases/ (immutable history).
SCAN_GLOBS = (
    "*.md", "*.html", "*.txt", "*.json",
)
SKIP_PREFIXES = (
    "docs/releases/",        # dated history, immutable
    "vendor/", "node_modules/",
)
SKIP_BASENAMES = (
    "llms-full.txt",         # generated mirror (regenerates from llms.txt + docs)
    "check_provenance_labels.py",  # this file's own doc-strings name the patterns
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


def _line_violates(line: str) -> tuple[bool, dict | None]:
    if not MEASURED_RE.search(line):
        return False, None
    if any(p.search(line) for p in ALLOW_PATTERNS):
        return False, None
    low = line.lower()
    for fam in MODELED_CLAIMS:
        has_ctx = any(re.search(c, low) for c in fam["context"])
        has_num = any(re.search(n, low) for n in fam["numbers"])
        if has_ctx and has_num:
            return True, fam
    return False, None


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
        bad, fam = _line_violates(line)
        if bad:
            hits.append({"file": relpath, "line": i,
                         "text": line.strip()[:160], "fix": fam["fix"]})
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
        bad, fam = _line_violates(line)
        if bad:
            hits.append({"file": relpath, "line": line_no,
                         "text": line.strip()[:160], "fix": fam["fix"]})
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
        print(f"provenance-labels: could not run git: {e}", file=sys.stderr)
        return 2

    all_hits: list[dict] = []
    for rel in files:
        if args.audit_staged:
            all_hits.extend(scan_staged_file(root, rel))
        else:
            all_hits.extend(scan_file(root, rel))

    if not all_hits:
        scope = "staged" if args.audit_staged else "tracked tree"
        print(f"provenance-labels: clean ({scope}); no MODELED number labeled \"measured\".")
        return 0

    if args.audit_staged and os.environ.get("ALLOW_PROVENANCE_DRIFT") == "1":
        print("provenance-labels: ALLOW_PROVENANCE_DRIFT=1 set — overriding "
              f"{len(all_hits)} hit(s) once.", file=sys.stderr)
        return 0

    print(f"PROVENANCE_LABEL: {len(all_hits)} MODELED number(s) labeled \"measured\":",
          file=sys.stderr)
    for h in all_hits:
        print(f"  {h['file']}:{h['line']}: {h['text']}", file=sys.stderr)
        print(f"    fix: {h['fix']}", file=sys.stderr)
    print("  override once (staged): ALLOW_PROVENANCE_DRIFT=1 <git cmd>.",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
