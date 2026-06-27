#!/usr/bin/env python3
"""check_brand_consistency.py — guard fak's PRIMARY product descriptor against re-drift.

fak once self-described four ways (agent kernel / Fused Agent Kernel /
*agent tool firewall* / *tool-call policy gateway*). The durable brand position
keeps ONE primary descriptor — "the Fused Agent Kernel" / "agent kernel" — and
retires the other two as PRIMARY descriptors. It still ALLOWS them as:
  - AEO keyword / topic / alternateName / Category synonym-list entries,
  - "also described as …" analogies,
  - references to the named `agent-kernel-video` asset and its card.

This check FAILS only when a RETIRED descriptor is used as the PRIMARY noun for
fak — a copula ("fak is an agent tool firewall") or a title/banner
("fak — agent tool firewall", "fak - Agent Tool Firewall"). It is deliberately
conservative: when a line carries any legitimate-use marker it is allowed, so a
flag is worth a human look rather than a wall of false positives.

See issue #591 (this guard) and #589 (the brand epic). The video-campaign
surfaces were re-themed to the agent-kernel/syscall spine in #592 and are now
scanned here like any other surface.

Run:  python tools/check_brand_consistency.py --audit-tree
Exit: 0 if clean, 1 if any primary-descriptor violation (so it can gate CI).
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The retired descriptors (match "tool-call" and "tool call").
RETIRED = r"(?:agent tool firewall|tool[- ]call policy gateway)"

# fak declared TO BE a retired descriptor: "fak is a/an/the X", or a
# title/banner "fak — X" / "fak - X" / "fak: X" (article optional there).
PRIMARY_RE = re.compile(
    r"\bfak\b[^.\n]{0,40}?(?:\bis\s+(?:an?|the)\s+|\s+[—:-]\s+(?:the\s+)?)" + RETIRED,
    re.IGNORECASE,
)

# Markers that make a retired descriptor a LEGIT secondary use, not a primary claim.
ALLOW_MARKERS = re.compile(
    r"(?i)also described as|alternatename|keywords?|topics?|category|aria-label|"
    r"\balt\b|explainer|reveal|\bcard\b|poster|\.mp4|\.gif|\.svg|agent-firewall|firewall card",
)

# Whole files / trees exempt: generated corpus, visual assets, and the
# generators that emit keyword/alternateName metadata.
EXEMPT_PREFIXES = ("visuals/",)
EXEMPT_FILES = {
    "llms-full.txt",                             # generated; mirrors source on regen
    "tools/check_brand_consistency.py",          # this file's own docstring examples
    "tools/check_brand_consistency_test.py",     # the test's synthetic samples
    "tools/gen_structured_data.py",              # emits alternateName/keywords lists
    # The Go twin of this checker (internal/hooks BRAND_CONSISTENCY gate) + its parity test
    # carry the SAME synthetic retired-descriptor samples, so they are exempt here too — both
    # checkers must agree, and neither should flag the other's fixtures. Kept in lockstep with
    # brandExemptFiles in internal/hooks/gate_brandconsistency.go.
    "internal/hooks/gate_brandconsistency.go",       # the Go gate's own docstring/regex source
    "internal/hooks/gate_brandconsistency_test.go",  # the Go parity test's golden vectors
}

# Reader-facing text surfaces only.
SCAN_EXT = (".md", ".txt", ".go", ".html", ".cff")


def tracked_files() -> list[str]:
    out = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True)
    return [p.strip() for p in out.splitlines() if p.strip()]


def audit() -> list[tuple[str, int, str]]:
    debt: list[tuple[str, int, str]] = []
    for rel in tracked_files():
        rel = rel.replace("\\", "/")
        if rel in EXEMPT_FILES or any(rel.startswith(p) for p in EXEMPT_PREFIXES):
            continue
        if not rel.endswith(SCAN_EXT):
            continue
        try:
            with open(os.path.join(ROOT, rel), encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except OSError:
            continue
        for i, line in enumerate(lines, 1):
            if PRIMARY_RE.search(line) and not ALLOW_MARKERS.search(line):
                debt.append((rel, i, line.strip()))
    return debt


def main() -> int:
    ap = argparse.ArgumentParser(description="Guard fak's primary descriptor against re-drift.")
    ap.add_argument("--audit-tree", action="store_true",
                    help="audit the git-tracked tree (default; immune to peer working-tree WIP)")
    ap.parse_args()

    debt = audit()
    if debt:
        print('brand-consistency: %d primary-descriptor violation(s) — retire '
              '"agent tool firewall" / "tool-call policy gateway" as the PRIMARY noun; '
              'keep "the Fused Agent Kernel" / "agent kernel":' % len(debt))
        for rel, ln, text in debt:
            print(f"  {rel}:{ln}: {text}")
        print("brand_debt:", len(debt))
        return 1
    print("brand-consistency: clean (primary descriptor held on the Fused Agent Kernel / agent kernel)")
    print("brand_debt: 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
