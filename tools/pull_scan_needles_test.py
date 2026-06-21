#!/usr/bin/env python3
"""Tests for tools/pull_scan_needles.py -- the hard-cut scan-instructions puller.

Proves the self-healing contract the `scan-needles` control-pane loop relies on:
--check is OK when pulled OR no private repo is reachable, and ACTION (exit 1)
only when a private repo is reachable but the needles are not pulled. Also covers
the canonical-artifact preference, pull, status, and dump. Pure stdlib.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "pull_scan_needles.py")


def run(*args):
    p = subprocess.run([sys.executable, TOOL, *args], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return p.returncode, p.stdout + p.stderr


def write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main() -> int:
    failures = []

    def check(name, cond, detail=""):
        print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if not cond and detail else ""))
        if not cond:
            failures.append(name)

    with tempfile.TemporaryDirectory() as tmp:
        public = os.path.join(tmp, "public")
        os.makedirs(public)
        private = os.path.join(tmp, "private")
        os.makedirs(private)
        missing = os.path.join(tmp, "nope")
        write(os.path.join(private, "scrub_needles.json"),
              json.dumps({"schema": "fleet-scrub-needles/1", "audit_needles": ["AAA"],
                          "export_audit_needles": ["AAA", "BBB"]}))

        # 1) --check, no sidecar, NO private repo -> OK (no nag where you can't pull)
        rc, out = run("--check", "--json", "--public-dir", public, "--from", missing)
        check("check no-private exits 0", rc == 0, out)
        check("check no-private mode", '"shape-only-no-private"' in out, out)

        # 2) --check, no sidecar, private REACHABLE -> ACTION (exit 1) so recover fires
        rc, out = run("--check", "--json", "--public-dir", public, "--from", private)
        check("check pullable exits 1", rc == 1, out)
        check("check pullable mode", '"shape-only-pullable"' in out, out)

        # 3) pull reads the canonical scrub_needles.json artifact
        rc, out = run("--public-dir", public, "--from", private)
        check("pull exits 0", rc == 0, out)
        sidecar = os.path.join(public, "tools", "_registry", "scrub_needles.private.json")
        check("sidecar written", os.path.isfile(sidecar), out)
        if os.path.isfile(sidecar):
            data = json.load(open(sidecar, encoding="utf-8"))
            check("sidecar has export needles", data.get("export_audit_needles") == ["AAA", "BBB"], str(data))
            check("source is the canonical artifact", str(data.get("source", "")).endswith("scrub_needles.json"), str(data))

        # 4) --check after pull -> OK, full mode
        rc, out = run("--check", "--json", "--public-dir", public, "--from", private)
        check("check after pull exits 0", rc == 0, out)
        check("check after pull is full", '"full"' in out, out)

        # 5) --status reports pulled
        rc, out = run("--status", "--public-dir", public)
        check("status reports PULLED", rc == 0 and "PULLED" in out, out)

        # 6) --dump emits the canonical artifact
        rc, out = run("--dump", "--from", private)
        check("dump exits 0", rc == 0, out)
        check("dump contains needles", '"BBB"' in out, out)

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        return 1
    print("pull_scan_needles_test: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
