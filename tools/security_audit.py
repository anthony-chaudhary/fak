#!/usr/bin/env python3
"""security_audit.py — on-demand security-posture auditor for the fleet-public repo.

The companion to .github/workflows/security-audit.yml (issue #54 / SOTA-parity E-004).
The workflow runs the *authoritative* scan — `govulncheck ./...` against the fak module,
which needs the Go toolchain and the live vulnerability DB. This tool runs the OTHER
half: a deterministic, no-toolchain, no-network check that the security automation is
actually WIRED — the same posture as public_readiness_audit.py, which verifies the leak
gate is present rather than re-running it.

It answers "is security-audit automation in place and consistent with this repo's stated
posture?" as a re-runnable gate, so a regression (someone deletes the govulncheck job, the
secret scanner, or adds an unpinned dependency) trips a check instead of going unnoticed.

Checks (each yields zero or more FAIL/WARN findings):
  govulncheck-ci    a .github/workflows/* invokes `govulncheck` (static analysis +
                    CVE/dependency scan in CI). FAIL if absent — the issue's top item.
  secret-leak-gate  the secret-leak scanner (tools/scrub_public_copy.py) exists AND a
                    workflow runs it AND the pre-commit hook exists. FAIL on a missing
                    scanner or CI gate (the #78-class regression, security-lint surface).
  dependency-surface  fak/go.mod's external `require`s are pinned by fak/go.sum. The
                    module ships zero external deps today; this stays correct — and FAILs
                    loudly — the moment an UNPINNED dependency is added (dependency scan).
  security-policy   a disclosure policy (SECURITY.md / .github/ / docs/) exists. FAIL if
                    missing — a security program needs a reporting channel.
  vet-lint          a workflow runs `go vet` (Go's built-in correctness/security lint).
                    WARN if absent.

Exit code is 1 if any FAIL, else 0. `--json` emits machine-readable output; `--check
NAME[,NAME]` runs a subset. `--run-govulncheck` additionally shells out to a locally
installed govulncheck against fak/ for an on-demand scan (best effort; CI is canonical).

Run: `python tools/security_audit.py`  (exit 0 = posture intact).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

# Disclosure-policy locations, in preference order.
SECURITY_POLICY_CANDIDATES = ["SECURITY.md", ".github/SECURITY.md", "docs/SECURITY.md"]

# The secret-leak scanner + its local hook. Both must survive for the gate to mean
# anything; public_readiness_audit.py guards the same pair from the readiness angle.
LEAK_SCANNER = "tools/scrub_public_copy.py"
PRECOMMIT_HOOK = "tools/githooks/pre-commit"

# The Go module lives at the REPOSITORY ROOT in the public layout (`go.mod`), and
# under `fak/` in the private/nested layout — check root first, then fall back. A
# single hardcoded `fak/go.mod` mis-fired a WARN on the public root-module clone
# (the module is at the root here; `fak/go.mod` does not exist). Each pair is tried
# in order; the first go.mod that exists is the module surface to audit.
GOMOD_CANDIDATES = [("go.mod", "go.sum"), ("fak/go.mod", "fak/go.sum")]


class Finding:
    def __init__(self, check: str, level: str, msg: str, where: str = ""):
        self.check = check
        self.level = level  # FAIL | WARN
        self.msg = msg
        self.where = where

    def as_dict(self):
        return {"check": self.check, "level": self.level, "msg": self.msg, "where": self.where}


def _read(root: Path, rel: str) -> str:
    p = root / rel
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8", errors="replace")


def workflow_files(root: Path) -> list[Path]:
    wf = root / ".github" / "workflows"
    if not wf.is_dir():
        return []
    return sorted(p for p in wf.iterdir()
                  if p.is_file() and p.suffix in (".yml", ".yaml"))


def _workflows_mentioning(root: Path, needle: str) -> list[str]:
    """Relative paths of workflow files whose text contains `needle` (comments excluded)."""
    hits = []
    for p in workflow_files(root):
        txt = p.read_text(encoding="utf-8", errors="replace")
        # strip whole-line YAML comments so a "# someday: govulncheck" note never counts.
        live = "\n".join(l for l in txt.splitlines() if not l.lstrip().startswith("#"))
        if needle in live:
            hits.append(p.relative_to(root).as_posix())
    return hits


def parse_external_requires(gomod_text: str, module_path: str) -> list[str]:
    """External module paths in a go.mod's `require`s (block or single-line).

    Excludes whole-line/// comments, the module's own path, and anything without a dotted
    domain in its first path segment (those are stdlib-shaped, never go.sum entries).
    """
    deps: list[str] = []
    in_block = False
    for raw in gomod_text.splitlines():
        line = raw.strip()
        if line.startswith("//"):
            continue
        if "//" in line:  # drop trailing inline comment
            line = line[: line.index("//")].strip()
        if not line:
            continue
        if in_block:
            if line.startswith(")"):
                in_block = False
                continue
            tok = line.split()
            if tok:
                deps.append(tok[0])
            continue
        if line.startswith("require") and "(" in line:
            in_block = True
            continue
        if line.startswith("require "):
            tok = line[len("require "):].split()
            if tok:
                deps.append(tok[0])
    out = []
    for d in deps:
        if d == module_path:
            continue
        first = d.split("/", 1)[0]
        if "." not in first:  # stdlib-shaped (e.g. "fmt") — never pinned in go.sum
            continue
        out.append(d)
    return out


def _module_path(gomod_text: str) -> str:
    for raw in gomod_text.splitlines():
        line = raw.strip()
        if line.startswith("module "):
            return line[len("module "):].strip()
    return ""


# --- individual checks ------------------------------------------------------

def check_govulncheck_ci(root: Path) -> list[Finding]:
    hits = _workflows_mentioning(root, "govulncheck")
    if not hits:
        return [Finding("govulncheck-ci", "FAIL",
                        "no .github/workflows/* runs govulncheck "
                        "(static analysis + CVE/dependency scan)", ".github/workflows/")]
    return []


def check_secret_leak_gate(root: Path) -> list[Finding]:
    out = []
    if not (root / LEAK_SCANNER).exists():
        out.append(Finding("secret-leak-gate", "FAIL",
                           "secret-leak scanner is missing", LEAK_SCANNER))
    elif not _workflows_mentioning(root, "scrub_public_copy"):
        # scanner present but no CI gate runs it — the #78-class regression.
        out.append(Finding("secret-leak-gate", "FAIL",
                           "scanner exists but no workflow runs it (no CI secret-leak gate)",
                           ".github/workflows/"))
    if not (root / PRECOMMIT_HOOK).exists():
        out.append(Finding("secret-leak-gate", "WARN",
                           "pre-commit leak hook missing (local gate only; CI still gates)",
                           PRECOMMIT_HOOK))
    return out


def check_dependency_surface(root: Path) -> list[Finding]:
    # Resolve the module surface from the first candidate go.mod that exists
    # (root for the public layout, fak/ for the nested/private one).
    gomod_rel = gosum_rel = ""
    gomod = ""
    for mod_rel, sum_rel in GOMOD_CANDIDATES:
        text = _read(root, mod_rel)
        if text:
            gomod, gomod_rel, gosum_rel = text, mod_rel, sum_rel
            break
    if not gomod:
        tried = ", ".join(m for m, _ in GOMOD_CANDIDATES)
        return [Finding("dependency-surface", "WARN", f"go.mod not found (tried: {tried})",
                        GOMOD_CANDIDATES[0][0])]
    deps = parse_external_requires(gomod, _module_path(gomod))
    if not deps:
        return []  # zero external deps — the repo's stated posture; nothing to pin.
    if not (root / gosum_rel).exists():
        return [Finding("dependency-surface", "FAIL",
                        f"{len(deps)} external dependency(ies) but no {gosum_rel} "
                        f"to pin them (unverified dependency surface)", gomod_rel)]
    return []


def check_security_policy(root: Path) -> list[Finding]:
    if any((root / c).exists() for c in SECURITY_POLICY_CANDIDATES):
        return []
    return [Finding("security-policy", "FAIL",
                    "no security-disclosure policy (SECURITY.md at root, .github/, or docs/)",
                    "")]


def check_vet_lint(root: Path) -> list[Finding]:
    if not _workflows_mentioning(root, "go vet"):
        return [Finding("vet-lint", "WARN",
                        "no workflow runs `go vet` (Go's built-in correctness/security lint)",
                        ".github/workflows/")]
    return []


CHECKS = {
    "govulncheck-ci": check_govulncheck_ci,
    "secret-leak-gate": check_secret_leak_gate,
    "dependency-surface": check_dependency_surface,
    "security-policy": check_security_policy,
    "vet-lint": check_vet_lint,
}


def run(root: Path, names: list[str]) -> list[Finding]:
    findings: list[Finding] = []
    for name in names:
        findings.extend(CHECKS[name](root))
    return findings


def run_govulncheck(root: Path) -> int:
    """Best-effort on-demand scan: shell out to a locally installed govulncheck.

    Returns its exit code (0 clean, non-zero = vuln found / not runnable). CI is the
    authoritative run; this is the convenience hook the issue calls "on demand".
    """
    if shutil.which("govulncheck") is None:
        print("govulncheck not installed — `go install golang.org/x/vuln/cmd/govulncheck@latest`",
              file=sys.stderr)
        return 127
    fak = root / "fak"
    if not fak.is_dir():
        print(f"fak module dir not found at {fak}", file=sys.stderr)
        return 2
    print(f"# govulncheck ./...  (cwd={fak})")
    proc = subprocess.run(["govulncheck", "./..."], cwd=fak)
    return proc.returncode


def main(argv=None):
    ap = argparse.ArgumentParser(description="on-demand security-posture auditor")
    ap.add_argument("--root", default=".", help="repo root (default: cwd)")
    ap.add_argument("--check", default="", help="comma-separated subset of checks")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--run-govulncheck", action="store_true",
                    help="also run a locally installed govulncheck against fak/ (best effort)")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    names = [c.strip() for c in args.check.split(",") if c.strip()] if args.check else list(CHECKS)
    bad = [c for c in names if c not in CHECKS]
    if bad:
        print(f"unknown check(s): {', '.join(bad)}", file=sys.stderr)
        return 2

    findings = run(root, names)
    fails = [f for f in findings if f.level == "FAIL"]
    warns = [f for f in findings if f.level == "WARN"]

    if args.json:
        print(json.dumps({
            "root": str(root),
            "checks": names,
            "fail": len(fails),
            "warn": len(warns),
            "findings": [f.as_dict() for f in findings],
        }, indent=2))
    else:
        if not findings:
            print(f"security-audit: posture INTACT across {len(names)} checks")
        else:
            for f in findings:
                loc = f" [{f.where}]" if f.where else ""
                print(f"{f.level:4} {f.check}: {f.msg}{loc}")
            print(f"\n{len(fails)} FAIL, {len(warns)} WARN across {len(names)} checks")

    rc = 1 if fails else 0
    if args.run_govulncheck:
        vc = run_govulncheck(root)
        if vc != 0:
            rc = rc or 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
