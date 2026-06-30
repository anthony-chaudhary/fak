#!/usr/bin/env python3
"""One idempotent command to provision THIS public clone's gardening (hard cut).

    python tools/provision_public_gardening.py            # report current state
    python tools/provision_public_gardening.py --apply    # converge (idempotent)

Durability goal: every public clone / fresh box stands up the same gardening with
ONE command instead of a hand-run checklist, and re-running converges to the same
state. This wraps the existing tools in order:

  1. fleet_control_pane.py init            -- host-local config
  2. fleet_control_pane.py bootstrap --apply-- runtime dirs + trunk/leak guard + tick
  3. install_trunk_guard.py                -- arm the leak-scan + trunk git guards
  4. pull_scan_needles.py                  -- pull the private scan instructions
                                              (skipped, not failed, with no private repo)
  5. fak public-scrub audit-tree           -- the proof: tracked tree is clean

Report mode (default) runs only the read-only checks (steps 4-status + 5) and the
control-pane doctor, so it is safe to run anywhere. ``--apply`` runs the full
converging sequence. Pure stdlib. Exit 0 ok, 1 a step needs attention, 2 precondition.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable


def _run(cmd, **kw):
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", **kw)


def _step(label: str, cmd, ok_codes=(0,)) -> tuple[bool, str]:
    proc = _run(cmd)
    tail = (proc.stdout or proc.stderr).strip().splitlines()
    last = tail[-1] if tail else ""
    ok = proc.returncode in ok_codes
    print(f"  [{'ok ' if ok else 'ACT'}] {label}: {last[:100]}")
    return ok, last


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--apply", action="store_true",
                    help="run the converging setup (default: read-only report)")
    ap.add_argument("--from", dest="private", default=None,
                    help="private repo for the needle pull (default: sibling ../fleet)")
    args = ap.parse_args(argv)

    print("=" * 72)
    print(f"PROVISION PUBLIC GARDENING  {'(APPLY)' if args.apply else '(REPORT)'}  {ROOT}")
    print("=" * 72)

    needs = 0
    if args.apply:
        for label, cmd in [
            ("init", [PY, "tools/fleet_control_pane.py", "init"]),
            ("bootstrap", [PY, "tools/fleet_control_pane.py", "bootstrap", "--apply"]),
            ("trunk+leak guard", [PY, "tools/install_trunk_guard.py"]),
        ]:
            ok, _ = _step(label, cmd)
            needs += 0 if ok else 1
        pull_cmd = [PY, "tools/pull_scan_needles.py"]
        if args.private:
            pull_cmd += ["--from", args.private]
        ok, last = _step("pull scan instructions", pull_cmd, ok_codes=(0, 2))
        if "not found" in last:
            print("         (no private repo on this box -- leak scan stays shape-only)")

    # Read-only checks (always).
    _step("needle status", [PY, "tools/pull_scan_needles.py", "--check"])
    ok, _ = _step("leak-scan proof (audit-tree)", ["go", "run", "./cmd/fak",
                                                   "public-scrub", "audit-tree", "--root", "."])
    needs += 0 if ok else 1
    ok, _ = _step("control-pane doctor", [PY, "tools/fleet_control_pane.py", "doctor"],
                  ok_codes=(0, 1))  # doctor exit 1 == ACTION, not a hard failure

    if not args.apply:
        print("\nREPORT only. Re-run with --apply to converge (idempotent).")
    elif needs:
        print(f"\nApplied with {needs} step(s) needing attention -- see above.")
    else:
        print("\nApplied. Public gardening provisioned; leak scan clean.")
    return 1 if needs else 0


if __name__ == "__main__":
    raise SystemExit(main())
