#!/usr/bin/env python3
"""Secret-shape gate: catch operator-leak SHAPES the literal needle list can't.

The leak-scan gate (scrub_public_copy.py) matches LITERAL needles pulled from the
private side. That structurally misses *novel* instances of the same leak class —
e.g. a real `C:/Users/<op>/...` path in forward-slash form, or a new `msl-<host>`
lab host, can slip past `--audit-tree` because its exact string was never on the
needle list. This gate matches the SHAPES instead, so a brand-new operator path or
`msl-*`/`*.lab` host is caught regardless.

It is deliberately scoped to the high-confidence, low-false-positive classes:
  * operator home paths  — C:\\Users\\<name> / C:/Users/<name> / /Users/<name>
    (the placeholder segments USER/You/runner/... are exempt).
  * internal lab hosts   — `msl-*` and `*.lab` (the `example.lab` placeholder is exempt).
Token/key shapes are left to the leak gate + its KEEP-listed fixtures.

Modes: --audit-staged (added lines only) | --audit-tree (tracked text files).
Exit: 0 clean, 1 shape hit, 2 could-not-run (hook fails open on 2).
Escape (staged): ALLOW_SECRET_SHAPE=1  (e.g. a new redaction-policy example).
"""
from __future__ import annotations
import argparse
import os
import re
import subprocess
from dispatch_worker import install_no_window_subprocess_defaults
install_no_window_subprocess_defaults(subprocess)
import sys

# user segments that are placeholders / system profiles, not a real operator name
PLACEHOLDER_USERS = {
    "user", "you", "runner", "public", "default", "all", "username",
    "administrator", "guest", "youruser", "name", "someone",
}
OP_PATH = re.compile(r"[A-Za-z]:[\\/][Uu]sers[\\/]([A-Za-z][A-Za-z0-9._-]{1,})")
MAC_PATH = re.compile(r"/Users/([A-Za-z][A-Za-z0-9._-]{1,})")
MSL_HOST = re.compile(r"\bmsl-[a-z0-9][a-z0-9-]*", re.IGNORECASE)
LAB_HOST = re.compile(r"\b[a-z0-9][a-z0-9.-]*\.lab\b", re.IGNORECASE)

# Files that legitimately contain these shapes (they define/test/document the rule).
SELF_REF = {
    "PUBLIC-SCRUB-POLICY.md",
    "tools/scrub_public_copy.py",
    "tools/check_secret_shapes.py",
    "tools/check_secret_shapes_test.py",
    "tools/operator_path_needle_test.py",
}
TEXT_EXT = (".md", ".txt", ".go", ".py", ".json", ".jsonl", ".sh", ".ps1",
            ".yml", ".yaml", ".toml", ".cff", ".html")


def _git(args, root):
    # Decode git output as UTF-8 explicitly: text=True alone uses the locale codec
    # (cp1252 on Windows), which crashes on UTF-8 bytes in git's output (a staged
    # diff's × — κ …, or the metaspace marker U+2581) and makes the gate die instead
    # of scan. errors="replace" keeps it scanning, never crashing.
    return subprocess.run(["git", "-C", root] + args, capture_output=True,
                          text=True, encoding="utf-8", errors="replace")


def _scan_text(text: str):
    """Yield (shape, matched) for each leak-shape hit in text."""
    for m in OP_PATH.finditer(text):
        if m.group(1).lower() not in PLACEHOLDER_USERS:
            yield ("operator-path", m.group(0))
    for m in MAC_PATH.finditer(text):
        if m.group(1).lower() not in PLACEHOLDER_USERS:
            yield ("operator-path", m.group(0))
    for m in MSL_HOST.finditer(text):
        yield ("internal-host", m.group(0))
    for m in LAB_HOST.finditer(text):
        host = m.group(0).lower()
        if not host.endswith("example.lab"):
            yield ("internal-host", m.group(0))


def _staged_added_lines(root):
    """Map file -> joined added lines (the '+' side of the staged diff)."""
    r = _git(["diff", "--cached", "--unified=0", "--no-color", "--diff-filter=ACMR"], root)
    if r.returncode != 0:
        return None
    out, cur = {}, None
    for line in r.stdout.splitlines():
        if line.startswith("+++ b/"):
            cur = line[6:]
            out.setdefault(cur, [])
        elif cur is not None and line.startswith("+") and not line.startswith("+++"):
            out[cur].append(line[1:])
    return {f: "\n".join(v) for f, v in out.items() if v}


def _tracked_text(root):
    r = _git(["ls-files"], root)
    if r.returncode != 0:
        return None
    files = [f for f in r.stdout.split() if f.endswith(TEXT_EXT)]
    out = {}
    for f in files:
        try:
            with open(os.path.join(root, f), encoding="utf-8") as fh:
                out[f] = fh.read()
        except (OSError, UnicodeDecodeError):
            continue
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--audit-staged", action="store_true")
    g.add_argument("--audit-tree", action="store_true")
    ap.add_argument("--root", default=".")
    a = ap.parse_args()
    root = os.path.abspath(a.root)

    if a.audit_staged and os.environ.get("ALLOW_SECRET_SHAPE") == "1":
        print("secret-shape: skipped (ALLOW_SECRET_SHAPE=1).")
        return 0

    blobs = _staged_added_lines(root) if a.audit_staged else _tracked_text(root)
    scope = "staged additions" if a.audit_staged else "tracked tree"
    if blobs is None:
        print("SECRET_SHAPE (warn): git not available; check skipped.", file=sys.stderr)
        return 2

    findings = []
    for f, text in blobs.items():
        if f in SELF_REF:
            continue
        for shape, hit in _scan_text(text):
            findings.append((f, shape, hit))
    if not findings:
        print(f"secret-shape: clean ({scope}).")
        return 0

    print(f"SECRET_SHAPE: {len(findings)} operator-leak shape(s) — novel form the needle list misses:", file=sys.stderr)
    seen = set()
    for f, shape, hit in findings:
        key = (f, hit)
        if key in seen:
            continue
        seen.add(key)
        print(f"  {f}: [{shape}] {hit}", file=sys.stderr)
    print("  fix: redact to the placeholder (C:\\Users\\USER, /Users/USER, "
          "gpu-server/example.lab) per PUBLIC-SCRUB-POLICY.md.", file=sys.stderr)
    if a.audit_staged:
        print("  override once: ALLOW_SECRET_SHAPE=1 <git cmd>  (a new redaction example only).", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
