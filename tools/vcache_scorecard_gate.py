#!/usr/bin/env python3
"""The repeatable vCache scorecard dogfood gate (issue #791, epic #788).

`fak vcache score` folds the vCache proof leaves into one 2x agent-dev gate. That
score is DETERMINISTIC over the built-in synthetic Zipf workload -- it issues no
provider calls and treats cache hits as rebates only -- so a regression that quietly
defeats anchor concentration, breaks the star-anchor proof, or drops the multiplier
below 2x is a green-looking change that has actually broken the headline claim.

The recent-feature dogfood PACKET (tools/recent_feature_dogfood.py) RUNS the score for
daily visibility; this is the dedicated GATE: a fast, standalone, deterministic check
that asserts the score's contract on BOTH paths and exits non-zero on a regression, so
it can sit in `make ci` next to the other scorecard gates rather than only in the daily
artifact trail.

It asserts two things, both with no network and no telemetry:

  1. The DEFAULT score is 2x-ready: schema == fak.vcache.score.v1, two_x_better is
     true, active_multiplier >= the threshold, and the planned star proof is PROVEN.
     `fak vcache score` exits 0.
  2. The NEGATIVE path holds: an unreachable threshold (--two-x 99) FAILS the gate --
     `fak vcache score` exits 1 and the payload reports two_x_better false. This proves
     the gate actually gates (an always-pass gate is a false green).

Exit 0 = the vCache scorecard floor holds. Exit 1 = a regression (with the failing
assertion named). Exit 2 = a harness error (the binary could not be run).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCORE_SCHEMA = "fak.vcache.score.v1"


def fak_cmd() -> list[str]:
    """Resolve the fak binary the gate drives, mirroring the dogfood packet's order:
    an explicit $FAK_BIN, then a built binary in the repo/tools/.bin, then $PATH, then
    a `go run` fallback so the gate works in a clean checkout with only the Go toolchain.
    """
    configured = os.environ.get("FAK_BIN", "").strip()
    if configured:
        return [configured]
    for rel in ("fak.exe", "fak", "tools/.bin/fak"):
        p = ROOT / rel
        if p.exists():
            return [str(p)]
    found = shutil.which("fak")
    if found:
        return [found]
    return ["go", "run", "./cmd/fak"]


def run_score(extra: list[str], timeout: int) -> tuple[int, dict]:
    """Run `fak vcache score --json [extra]` and return (exit_code, payload). A payload
    that does not parse as a JSON object is returned as {} so the caller's assertions
    fail with a clear message rather than a traceback.
    """
    cmd = fak_cmd() + ["vcache", "score", "--json"] + extra
    proc = subprocess.run(
        cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=timeout
    )
    try:
        payload = json.loads(proc.stdout)
    except (json.JSONDecodeError, ValueError):
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    return proc.returncode, payload


def check_default(threshold: float, timeout: int) -> list[str]:
    """Assert the default score is 2x-ready and the planned proof PROVEN. Returns a list
    of failure messages (empty == pass)."""
    fails: list[str] = []
    code, p = run_score([], timeout)
    if code != 0:
        fails.append(f"default `fak vcache score` exited {code}, want 0 (2x gate should pass)")
    if p.get("schema") != SCORE_SCHEMA:
        fails.append(f"default score schema = {p.get('schema')!r}, want {SCORE_SCHEMA!r}")
    if p.get("two_x_better") is not True:
        fails.append(f"default two_x_better = {p.get('two_x_better')!r}, want True")
    mult = p.get("active_multiplier", 0)
    if not isinstance(mult, (int, float)) or mult < threshold:
        fails.append(f"default active_multiplier = {mult!r}, want >= {threshold}")
    planned = p.get("planned") or {}
    status = str(planned.get("status", "")).upper()
    if status and status != "PROVEN":
        fails.append(f"default planned proof status = {status!r}, want PROVEN")
    return fails


def check_negative(timeout: int) -> list[str]:
    """Assert an unreachable threshold FAILS the gate -- the proof that the gate gates."""
    fails: list[str] = []
    code, p = run_score(["--two-x", "99"], timeout)
    if code != 1:
        fails.append(f"`fak vcache score --two-x 99` exited {code}, want 1 (unreachable gate must fail)")
    if p and p.get("two_x_better") is not False:
        fails.append(f"--two-x 99 two_x_better = {p.get('two_x_better')!r}, want False")
    return fails


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--two-x", type=float, default=2.0,
                    help="the multiplier floor the default score must clear (default 2.0)")
    ap.add_argument("--timeout", type=int, default=120, help="per-command timeout, seconds")
    ap.add_argument("--json", action="store_true", help="emit a machine-readable verdict")
    args = ap.parse_args(argv)

    try:
        fails = check_default(args.two_x, args.timeout) + check_negative(args.timeout)
    except (subprocess.SubprocessError, FileNotFoundError, OSError) as e:
        msg = f"vcache-scorecard-gate: could not run fak: {e}"
        if args.json:
            print(json.dumps({"schema": "fak.vcache-scorecard-gate.v1", "ok": False,
                              "error": str(e)}))
        else:
            print(msg, file=sys.stderr)
        return 2

    ok = not fails
    if args.json:
        print(json.dumps({"schema": "fak.vcache-scorecard-gate.v1", "ok": ok,
                          "two_x_threshold": args.two_x, "failures": fails}))
    else:
        if ok:
            print(f"vcache-scorecard-gate: OK -- vCache 2x floor holds (threshold {args.two_x}x)")
        else:
            print("vcache-scorecard-gate: REGRESSION", file=sys.stderr)
            for f in fails:
                print(f"  - {f}", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
