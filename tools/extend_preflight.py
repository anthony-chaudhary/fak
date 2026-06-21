#!/usr/bin/env python3
"""extend_preflight.py — am I set up to build an optimization on fak?

A one-command readiness gate for a contributor — a human, a Claude Code session, or a
Codex/Cursor/Aider run — who wants to land a subsystem optimization (a faster kernel, a
new device/quant backend, a smarter admission rung) as a first-class citizen. It checks
the environment and repo wiring that `fak/EXTENDING.md` depends on, then prints the
three-gate golden path so a fresh contributor knows exactly what to do next.

The point: agentic contributors are first-class here, so "am I set up correctly?" should
be a command, not tribal knowledge spread across five docs. This makes EXTENDING.md
*runnable*.

Read-only. Pure stdlib (no venv). Off the request path — a `tools/` diagnostic.

Usage:
  python tools/extend_preflight.py           # human-readable report
  python tools/extend_preflight.py --json     # machine-readable (for an agent tool)
  python tools/extend_preflight.py --quiet     # print only the problems

Exit code: 0 if every REQUIRED check passes; 1 otherwise. Warnings and info never fail
the exit (so this is safe to wire into a loop), but they're surfaced loudly.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# A ship-stamp trailer `(fak <leaf>)` at the END of a subject, or the legacy
# `fak/<leaf>: ...` start-prefix, or a release `vX.Y.Z:` anchor — the three forms DOS
# binds as a unit of work (see dos.toml [stamp], AGENTS.md "Verifiable commits").
STAMP_RE = re.compile(r"\(fak [a-z0-9][a-z0-9-]*\)\s*$|^[a-z0-9][a-z0-9-]*/[a-z0-9-]+:|^v\d+\.\d+\.\d+:")

ERROR, WARN, INFO = "error", "warn", "info"


def _git(*args: str) -> str:
    """Run a git command at the repo root; return stdout stripped, or '' on failure."""
    try:
        out = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=20
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def _exists(rel: str) -> bool:
    return (ROOT / rel).exists()


def _file_has(rel: str, needle: str) -> bool:
    p = ROOT / rel
    try:
        return needle in p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return False


def _check(name: str, level: str, ok: bool, detail: str, fix: str = "") -> dict:
    return {"name": name, "level": level, "ok": bool(ok), "detail": detail, "fix": fix}


def run_checks() -> list[dict]:
    checks: list[dict] = []

    # --- environment: the guards that enforce the rules below the agent layer ---------
    hooks = _git("config", "--get", "core.hooksPath")
    checks.append(_check(
        "git-guards-installed", ERROR,
        hooks == "tools/githooks",
        f"core.hooksPath = {hooks!r} (want 'tools/githooks': trunk guard + leak scan)",
        "python tools/install_trunk_guard.py",
    ))

    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    checks.append(_check(
        "on-master", WARN,
        branch == "master",
        f"on branch {branch!r} (the stay-on-master operator law; off-master is OFF_TRUNK)",
        "git switch master  # commit directly to master; never branch to dodge a dirty tree",
    ))

    subjects = _git("log", "-20", "--format=%s").splitlines()
    stamped = [s for s in subjects if STAMP_RE.search(s)]
    checks.append(_check(
        "ship-stamp-convention", WARN,
        len(stamped) > 0,
        f"{len(stamped)}/{len(subjects)} of the last commits carry a verifiable ship-stamp",
        "stamp subjects 'type(scope): ... (fak <leaf>)' so dos verify can bind your work",
    ))

    # --- Gate 1: plug in (the registration seams + scaffold + layering gate) ----------
    g1 = (
        _exists("tools/new_leaf.py")
        and _exists("fak/internal/architest")
        and _file_has("fak/internal/compute/compute.go", "func Register(")
    )
    checks.append(_check(
        "gate1-plug-in", ERROR, g1,
        "Register* seams present: new_leaf.py scaffold, internal/architest layering gate, "
        "compute.Register HAL seam",
        "see fak/EXTENDING.md 'Gate 1' and fak/ARCHITECTURE.md",
    ))

    # --- Gate 2: prove correct (the witness pattern + Reference/Approx contract) -------
    witnesses = sorted((ROOT / "fak" / "internal").glob("*/proofs_witness_test.go"))
    g2 = len(witnesses) > 0 and _file_has(
        "fak/internal/compute/compute.go", "CorrectnessClass"
    )
    checks.append(_check(
        "gate2-prove-correct", ERROR, g2,
        f"{len(witnesses)} proofs_witness_test.go witnesses; Reference/Approx correctness "
        "class wired (max|delta|=0 vs argmax-exact+cosine)",
        "see fak/EXTENDING.md 'Gate 2' and fak/docs/proofs/00-METHOD.md",
    ))

    # --- Gate 3: prove faster (the non-forgeable keep-bit) ----------------------------
    g3 = _exists("fak/cmd/rsicycle") and _exists("fak/internal/shipgate/shipgate.go")
    checks.append(_check(
        "gate3-prove-faster", ERROR, g3,
        "keep-bit wired: cmd/rsicycle one-shot + shipgate.Evaluate (KEEP only on "
        "strict-gain AND green AND clean)",
        "see fak/EXTENDING.md 'Gate 3' and fak/BENCHMARK-AUTHORITY.md",
    ))

    # --- the golden-path docs themselves ----------------------------------------------
    checks.append(_check(
        "golden-path-docs", ERROR,
        _exists("fak/EXTENDING.md") and _exists("CONTRIBUTING.md"),
        "fak/EXTENDING.md (on-ramp) + CONTRIBUTING.md (landing flow) present",
        "git pull --no-rebase  # fetch the contributor docs",
    ))

    # --- test path (informational: Windows hosts run the Go suite through WSL) ---------
    has_wsl = shutil.which("wsl") is not None
    has_testsh = _exists("fak/test.sh")
    if sys.platform == "win32":
        checks.append(_check(
            "test-path", INFO,
            has_wsl and has_testsh,
            f"WSL present={has_wsl}, fak/test.sh present={has_testsh} "
            "(Go tests run via WSL on this host: .\\fak\\test.ps1)",
            "install WSL + a distro; `go build`/`go vet` still work natively",
        ))
    else:
        checks.append(_check(
            "test-path", INFO, has_testsh,
            f"fak/test.sh present={has_testsh} (run the suite with ./fak/test.sh)",
            "",
        ))

    return checks


GOLDEN_PATH = [
    "Gate 1 - Plug in:  python tools/new_leaf.py <name> --tier <tier> --register   "
    "(or add a compute.Backend file in fak/internal/compute/); keep architest green: "
    ".\\fak\\test.ps1 ./internal/architest/",
    "Gate 2 - Prove correct:  ship a deterministic proofs_witness_test.go; declare your "
    "CorrectnessClass (Reference=max|delta|=0, Approx=argmax-exact+cosine); .\\fak\\test.ps1 "
    "./internal/<pkg>/",
    "Gate 3 - Prove faster:  go run ./cmd/rsicycle ...  (KEEP only on a measured strict gain; "
    "trace the number to BENCHMARK-AUTHORITY.md)",
    "Land it:  stay on master | git commit -- <paths> | stamp 'type(scope): ... (fak <leaf>)' | "
    "DCO + CLA on an external PR (CONTRIBUTING.md)",
]


def summarize(checks: list[dict]) -> dict:
    failed_required = [c for c in checks if c["level"] == ERROR and not c["ok"]]
    return {
        "ok": len(failed_required) == 0,
        "failed_required": [c["name"] for c in failed_required],
        "checks": checks,
        "golden_path": GOLDEN_PATH,
        "doc": "fak/EXTENDING.md",
    }


def render(result: dict, quiet: bool) -> str:
    lines = ["fak extension preflight - are you set up to build an optimization on fak?", ""]
    for c in result["checks"]:
        if quiet and c["ok"]:
            continue
        tag = "PASS" if c["ok"] else ("WARN" if c["level"] == WARN else
                                      "info" if c["level"] == INFO else "FAIL")
        lines.append(f"  [{tag:4}] {c['name']}: {c['detail']}")
        if not c["ok"] and c["fix"]:
            lines.append(f"         fix -> {c['fix']}")
    lines.append("")
    if result["ok"]:
        lines.append("READY. The three-gate golden path:")
    else:
        lines.append(f"NOT READY — required checks failed: {', '.join(result['failed_required'])}")
        lines.append("Fix the FAILs above, then re-run. The golden path once you're ready:")
    for step in result["golden_path"]:
        lines.append(f"  - {step}")
    lines.append("")
    lines.append("Full on-ramp: fak/EXTENDING.md")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--quiet", action="store_true", help="print only non-passing checks")
    args = ap.parse_args(argv)

    # Belt-and-suspenders for older Windows consoles (cp1252): never crash on a stray
    # non-ASCII glyph. The printed strings are already ASCII; this guards future edits.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    result = summarize(run_checks())
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(render(result, args.quiet))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
