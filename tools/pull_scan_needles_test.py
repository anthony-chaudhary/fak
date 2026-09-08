#!/usr/bin/env python3
"""Tests for tools/pull_scan_needles.py -- the hard-cut scan-instructions puller.

Proves the self-healing contract the `scan-needles` control-pane loop relies on:
default discovery selects the authorized sibling, a retired-source sidecar is
rejected, a current loaded sidecar reports full mode, and an absent sidecar is
explicitly shape-only. Pure stdlib.
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
        public = os.path.join(tmp, "fak")
        os.makedirs(public)
        private = os.path.join(tmp, "fak-private")
        legacy = os.path.join(tmp, "fleet")
        missing = os.path.join(tmp, "unavailable")
        os.makedirs(private)
        os.makedirs(legacy)
        authorized_audit = ["fixture-authorized-audit"]
        authorized_export = authorized_audit + ["fixture-authorized-export"]
        legacy_audit = ["fixture-retired-audit"]
        legacy_export = legacy_audit + ["fixture-retired-export"]
        write(os.path.join(private, "scrub_needles.json"), json.dumps({
            "schema": "fleet-scrub-needles/1",
            "audit_needles": authorized_audit,
            "export_audit_needles": authorized_export,
        }))
        write(os.path.join(legacy, "scrub_needles.json"), json.dumps({
            "schema": "fleet-scrub-needles/1",
            "audit_needles": legacy_audit,
            "export_audit_needles": legacy_export,
        }))

        # 1) No sidecar is explicit shape-only, while default discovery still sees
        # the authorized sibling and never reports local paths or needle values.
        rc, out = run("--check", "--json", "--public-dir", public)
        payload = json.loads(out)
        check("absent sidecar exits 1", rc == 1, out)
        check("absent sidecar is shape-only", payload.get("pulled") is False and
              payload.get("mode") == "shape-only-pullable", out)
        check("absent receipt is value-free", tmp not in out and
              not any(value in out for value in authorized_export + legacy_export), out)

        rc, out = run("--check", "--json", "--public-dir", public, "--from", missing)
        payload = json.loads(out)
        check("absent companion exits 0", rc == 0, out)
        check("absent companion is explicit shape-only", payload.get("pulled") is False and
              payload.get("mode") == "shape-only-no-private", out)
        check("absent-companion receipt is value-free", tmp not in out, out)

        # 2) A sidecar attributed to the retired sibling is not accepted as full.
        sidecar = os.path.join(public, "tools", "_registry", "scrub_needles.private.json")
        write(sidecar, json.dumps({
            "schema": "fleet-scrub-needles/1",
            "source": os.path.join(legacy, "scrub_needles.json"),
            "audit_needles": legacy_audit,
            "export_audit_needles": legacy_export,
        }))
        rc, out = run("--check", "--json", "--public-dir", public)
        payload = json.loads(out)
        check("retired sidecar is rejected", rc == 1 and payload.get("pulled") is False and
              payload.get("mode") == "shape-only-pullable", out)
        check("retired receipt is value-free", tmp not in out and
              not any(value in out for value in legacy_export), out)

        # 3) Default pull selects fak-private, ignoring the present retired sibling.
        os.remove(sidecar)
        rc, out = run("--public-dir", public)
        check("default authorized pull exits 0", rc == 0, out)
        check("pull output is value-free", tmp not in out and
              not any(value in out for value in authorized_export + legacy_export), out)
        data = json.load(open(sidecar, encoding="utf-8"))
        sidecar_summary = {
            "source": data.get("source"),
            "has_digest": bool(data.get("source_digest")),
            "audit_count": len(data.get("audit_needles") or []),
            "export_count": len(data.get("export_audit_needles") or []),
        }
        check("authorized artifact wins", data.get("audit_needles") == authorized_audit and
              data.get("export_audit_needles") == authorized_export, str(sidecar_summary))
        check("sidecar stores portable source identity", data.get("source") == "scrub_needles.json",
              str(sidecar_summary))
        check("sidecar carries freshness digest", bool(data.get("source_digest")),
              str(sidecar_summary))

        # 4) A current, actually loaded sidecar reports pulled=true/full without values.
        rc, out = run("--check", "--json", "--public-dir", public)
        payload = json.loads(out)
        check("current sidecar exits 0", rc == 0, out)
        check("current sidecar is full", payload.get("pulled") is True and
              payload.get("mode") == "full", out)
        check("full receipt is value-free", tmp not in out and
              not any(value in out for value in authorized_export), out)

        # 5) Human status is also value-free.
        rc, out = run("--status", "--public-dir", public)
        check("status reports PULLED", rc == 0 and "PULLED" in out and tmp not in out, out)

        # 6) The explicit canonical-artifact export remains available.
        rc, out = run("--dump", "--from", private)
        dumped = json.loads(out)
        check("dump exits 0", rc == 0, out)
        check("dump preserves canonical fixture", dumped.get("audit_needles") == authorized_audit and
              dumped.get("export_audit_needles") == authorized_export, "unexpected fixture counts")

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        return 1
    print("pull_scan_needles_test: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
