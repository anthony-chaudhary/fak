#!/usr/bin/env python3
"""Load-bearing test-coverage audit for the python tools/ cluster.

`code_quality_scorecard.py` grades the GO corpus only, so the python tools/ cluster
— ~150 scripts, many load-bearing (a SKILL.md helper, a CI gate, a dispatch-loop
arm) — has NO coverage gate. That is exactly how a 470-line ranking helper
(issue_triage.py) shipped with no test, invisible to every gate. This is the
cheap, STATIC complement: it never runs a test (zero CI-time cost), it just maps
which tools/ modules lack a sibling `<name>_test.py` and flags the LOAD-BEARING
ones — a module whose name is referenced by a SKILL.md or by ci.yml, i.e. one the
fleet actually depends on. The headline number is load-bearing coverage; a
load-bearing module with no test is the debt to retire.

Pure filesystem read; no test execution, no gh/dos/git/network.

    python tools/tool_coverage_audit.py                 # the card
    python tools/tool_coverage_audit.py --json
    python tools/tool_coverage_audit.py --min-coverage 90   # advisory floor (exit 1 if below)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except (AttributeError, ValueError):
    pass

SCHEMA = "fleet-tool-coverage-audit/1"
DEFAULT_MIN_COVERAGE = 90.0
# Modules that legitimately need no sibling unit test (config shims / data / entry
# stubs). Kept small and explicit so a real gap is never silently exempted.
_NO_TEST_OK = {"__init__"}


def repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    return here.parent.parent


# ---------------------------------------------------------------------------
# Discovery seams (real impls read the tree; tests pass literals)
# ---------------------------------------------------------------------------

def find_module_stems(tools_dir: Path) -> list[str]:
    """tools/*.py minus *_test.py minus __init__ — the auditable module set."""
    out = []
    for p in sorted(tools_dir.glob("*.py")):
        if p.name.endswith("_test.py") or p.stem in _NO_TEST_OK:
            continue
        out.append(p.stem)
    return out


def find_test_stems(tools_dir: Path) -> set[str]:
    """The stems that HAVE a sibling test: foo_test.py -> 'foo'."""
    return {p.name[: -len("_test.py")] for p in tools_dir.glob("*_test.py")}


def gather_refs_text(skills_dir: Path, *extra_files: Path) -> str:
    """Concatenated text where a module name counts as 'referenced' (load-bearing).

    SKILL.md bodies + named extra files (e.g. ci.yml). Best-effort: a missing
    path contributes empty text rather than erroring.
    """
    chunks: list[str] = []
    if skills_dir.exists():
        for sk in skills_dir.rglob("SKILL.md"):
            try:
                chunks.append(sk.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                pass
    for f in extra_files:
        try:
            chunks.append(f.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            pass
    return "\n".join(chunks)


# ---------------------------------------------------------------------------
# Pure auditor
# ---------------------------------------------------------------------------

def audit(module_stems: list[str], test_stems: set[str], refs_text: str) -> dict[str, Any]:
    """Map coverage; the headline is LOAD-BEARING coverage (referenced modules)."""
    rows = []
    for stem in module_stems:
        referenced = f"{stem}.py" in refs_text
        rows.append({
            "module": stem,
            "tested": stem in test_stems,
            "load_bearing": referenced,
        })

    total = len(rows)
    tested = sum(1 for r in rows if r["tested"])
    lb = [r for r in rows if r["load_bearing"]]
    lb_tested = sum(1 for r in lb if r["tested"])
    lb_untested = sorted(r["module"] for r in lb if not r["tested"])

    overall = round(tested / total * 100, 1) if total else None
    lb_cov = round(lb_tested / len(lb) * 100, 1) if lb else None
    return {
        "total_modules": total,
        "tested": tested,
        "overall_coverage_pct": overall,
        "load_bearing": len(lb),
        "load_bearing_tested": lb_tested,
        "load_bearing_coverage_pct": lb_cov,
        "load_bearing_untested": lb_untested,
        "debt": len(lb_untested),
    }


def build_payload(*, root: str, a: dict[str, Any], min_coverage: float | None) -> dict[str, Any]:
    lb_cov = a.get("load_bearing_coverage_pct")
    if lb_cov is None:
        ok, verdict = True, "NO_LOAD_BEARING_MODULES"
        reason = "no load-bearing tools/ modules found (nothing referenced by a SKILL.md / ci.yml)"
    elif min_coverage is not None and lb_cov < min_coverage:
        ok, verdict = False, "BELOW_FLOOR"
        reason = (f"load-bearing test coverage {lb_cov}% is below the {min_coverage}% floor — "
                  f"{a['debt']} load-bearing module(s) have NO sibling test: "
                  + ", ".join(a["load_bearing_untested"][:12]))
    else:
        ok, verdict = True, "OK"
        reason = (f"load-bearing test coverage {lb_cov}% "
                  f"({a['load_bearing_tested']}/{a['load_bearing']}); {a['debt']} untested")
    return {"schema": SCHEMA, "ok": ok, "verdict": verdict, "reason": reason,
            "workspace": root, "min_coverage": min_coverage, **a}


def collect(root: Path, *, min_coverage: float | None) -> dict[str, Any]:
    tools_dir = root / "tools"
    refs = gather_refs_text(root / ".claude" / "skills",
                            root / ".github" / "workflows" / "ci.yml")
    a = audit(find_module_stems(tools_dir), find_test_stems(tools_dir), refs)
    return build_payload(root=str(root), a=a, min_coverage=min_coverage)


def render(p: dict[str, Any]) -> str:
    lines = [
        f"tool test-coverage audit: {p.get('verdict')} ({'ok' if p.get('ok') else 'ACTION'})",
        f"  {p.get('reason')}",
        f"  modules={p.get('total_modules')} tested={p.get('tested')} "
        f"overall={p.get('overall_coverage_pct')}%  "
        f"load-bearing={p.get('load_bearing_tested')}/{p.get('load_bearing')} "
        f"({p.get('load_bearing_coverage_pct')}%)",
    ]
    lbu = p.get("load_bearing_untested") or []
    if lbu:
        lines.append("  load-bearing + UNTESTED (add a sibling _test.py):")
        for mod in lbu[:20]:
            lines.append(f"    - tools/{mod}.py")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Load-bearing test-coverage audit for the python tools/ cluster (read-only, static).")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--min-coverage", type=float, default=None,
                    help="advisory floor pct on LOAD-BEARING coverage; exit 1 if below")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args(argv)

    root = Path(args.workspace).resolve() if args.workspace else repo_root()
    payload = collect(root, min_coverage=args.min_coverage)
    print(json.dumps(payload, indent=2) if args.json else render(payload))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
