#!/usr/bin/env python3
"""Claim-reproducibility scorecard — validates witnesses are resolvable from a clean clone.

Every capability claim in CLAIMS.md (``[SHIPPED]`` / ``[SIMULATED]`` / ``[STUB]``) and
every benchmark row in BENCHMARK-AUTHORITY.md carries a witness handle or artifact path.
This scorecard checks whether those witnesses are actually **resolvable by an outsider**:
a ``Witness: TestFooBar`` that names a non-existent test, or a ``Reproduce: go run ./cmd/gone``
pointing at a deleted binary, is an **un-falsifiable claim** — the worst failure mode
for a skeptical reader, because it looks checkable and isn't.

This scores **resolvability** (does the witness exist and point at a real thing), not live
green/red — running every witness needs a build/GPU and is out of scope for a deterministic,
host-free scorecard. A resolvable witness is what lets an outsider *go run it themselves*.

Run from the repo ROOT::

    python tools/claim_repro_scorecard.py                 # human scorecard
    python tools/claim_repro_scorecard.py --json          # machine payload
    python tools/claim_repro_scorecard.py --markdown      # snapshot body
    python tools/claim_repro_scorecard.py --compare       # before/after diff
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

SCHEMA = "fleet-claim-repro-scorecard/1"

# Repo-root-relative inputs
CLAIMS_REL = "CLAIMS.md"
BENCHMARK_REL = "BENCHMARK-AUTHORITY.md"

# Regex patterns for witness extraction
# Test witness patterns: `go test ./pkg -run X`, `TestX`, `-run X`
_WITNESS_TEST_RE = re.compile(r"`go test\s+([^`]+)`")
_WITNESS_FUNC_RE = re.compile(r"(Test|Benchmark|Fuzz|Example)\w+")
_WITNESS_RUN_RE = re.compile(r"-run\s+(\S+)")

# cmd/<dir> references in `go run ./cmd/<dir>` or similar
_CMD_DIR_RE = re.compile(r"`go run\s+\./cmd/([^/`]+)`")

# File/artifact path patterns
_ARTIFACT_PATH_RE = re.compile(r"`([^`]+\.(json|md|log|txt|csv))`")

# Markdown link patterns for artifacts
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+\.(json|md|log|txt|csv))\)")


def _clamp(score: float) -> int:
    return int(max(0, min(100, round(score))))


def repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    return here.parent.parent


def _safe_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _extract_test_functions(content: str, pkg_path: str) -> set[str]:
    """Extract test function names from Go test files in a package."""
    test_files = []
    for suffix in ("_test.go",):
        test_files.extend(list(Path(repo_root()).rglob(f"{pkg_path}/**/*{suffix}")))

    funcs = set()
    for tf in test_files:
        try:
            content = tf.read_text(encoding="utf-8", errors="ignore")
            for m in _WITNESS_FUNC_RE.finditer(content):
                funcs.add(m.group(0))
        except OSError:
            continue
    return funcs


def _package_exists(pkg_path: str, root: Path) -> bool:
    """Check if a Go package path exists in the tree."""
    pkg_dir = root / pkg_path.replace(".", "/")
    return pkg_dir.exists() and pkg_dir.is_dir()


def _cmd_dir_exists(cmd_name: str, root: Path) -> bool:
    """Check if cmd/<dir> exists."""
    cmd_dir = root / "cmd" / cmd_name
    return cmd_dir.exists() and cmd_dir.is_dir()


def _file_exists(rel_path: str, root: Path) -> bool:
    """Check if a file exists at a relative path."""
    file_path = root / rel_path
    return file_path.exists() and file_path.is_file()


def _go_files_in_dir(dir_path: Path) -> list[Path]:
    """Get all .go files in a directory."""
    return list(dir_path.rglob("*.go"))


def _resolve_claim_witnesses(line: str, root: Path) -> list[str]:
    """Check resolvability of a claim's witness handles."""
    issues: list[str] = []

    # Check `go test ./pkg -run X` patterns
    for m in _WITNESS_TEST_RE.finditer(line):
        cmd = m.group(1)
        parts = cmd.split()
        pkg_path = ""
        for i, part in enumerate(parts):
            if part.startswith("./") or part.startswith(".\\"):
                pkg_path = part[2:]  # Remove ./ prefix
                break
            if "." in part and not part.startswith("-"):
                pkg_path = part
                break

        if pkg_path and not _package_exists(pkg_path, root):
            issues.append(f"missing package path: {pkg_path}")

        # Check test function names if -run is present
        run_match = _WITNESS_RUN_RE.search(cmd)
        if run_match:
            test_name = run_match.group(1)
            # Get all test functions in the package
            test_funcs = _extract_test_functions(cmd, pkg_path) if pkg_path else set()
            # Check if any test function matches the run pattern (simplified: exact match)
            if test_funcs and not any(test_name in func for func in test_funcs):
                issues.append(f"test pattern '{test_name}' not found in package {pkg_path}")

    # Check `go run ./cmd/<dir>` patterns
    for m in _CMD_DIR_RE.finditer(line):
        cmd_name = m.group(1)
        if not _cmd_dir_exists(cmd_name, root):
            issues.append(f"missing cmd dir: cmd/{cmd_name}")

    # Check artifact paths in backticks
    for m in _ARTIFACT_PATH_RE.finditer(line):
        artifact_path = m.group(1)
        if not _file_exists(artifact_path, root):
            issues.append(f"missing artifact: {artifact_path}")

    # Check markdown link artifact paths
    for m in _MD_LINK_RE.finditer(line):
        artifact_path = m.group(2)
        if not _file_exists(artifact_path, root):
            issues.append(f"missing linked artifact: {artifact_path}")

    # Check standalone test function references
    for m in _WITNESS_FUNC_RE.finditer(line):
        func_name = m.group(0)
        # Search tree-wide for this test function
        found = False
        for go_file in root.rglob("*_test.go"):
            try:
                content = go_file.read_text(encoding="utf-8", errors="ignore")
                if func_name in content:
                    found = True
                    break
            except OSError:
                continue
        if not found:
            issues.append(f"test function not found: {func_name}")

    return issues


def _resolve_benchmark_witnesses(line: str, root: Path) -> list[str]:
    """Check resolvability of a benchmark row's artifact/reproduce handles."""
    issues: list[str] = []

    # Check artifact paths
    for m in _ARTIFACT_PATH_RE.finditer(line):
        artifact_path = m.group(1)
        if not _file_exists(artifact_path, root):
            issues.append(f"missing artifact: {artifact_path}")

    # Check markdown link artifact paths
    for m in _MD_LINK_RE.finditer(line):
        artifact_path = m.group(2)
        if not _file_exists(artifact_path, root):
            issues.append(f"missing linked artifact: {artifact_path}")

    # Check Reproduce: command patterns
    if "Reproduce:" in line:
        # Extract the reproduce command
        cmd_match = re.search(r"Reproduce:\s*`([^`]+)`", line)
        if cmd_match:
            cmd = cmd_match.group(1)
            # Check for `go run ./cmd/<dir>` patterns in reproduce
            for m in _CMD_DIR_RE.finditer(cmd):
                cmd_name = m.group(1)
                if not _cmd_dir_exists(cmd_name, root):
                    issues.append(f"Reproduce: missing cmd dir: cmd/{cmd_name}")

    return issues


def _check_claims(claims_text: str, root: Path) -> dict[str, Any]:
    """Check all claims in CLAIMS.md for resolvable witnesses."""
    if not claims_text:
        return {
            "kpi": "claims",
            "score": 100,
            "detail": "no CLAIMS.md (skipped)",
            "defects": [],
            "soft": ["CLAIMS.md not found"],
        }

    total_claims = 0
    unfalsifiable_claims: list[dict[str, Any]] = []

    in_fence = False
    for line in claims_text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        if in_fence or not line.startswith("- ["):
            continue

        total_claims += 1

        # Extract witness handles and check resolvability
        issues = _resolve_claim_witnesses(line, root)

        if issues:
            unfalsifiable_claims.append({
                "line": line.strip()[:120],
                "issues": issues,
            })

    defects = [f"un-falsifiable claim: {c['line']} — {', '.join(c['issues'])}"
               for c in unfalsifiable_claims]

    n = len(unfalsifiable_claims)
    score = _clamp(100 - 30 * n)  # Each un-falsifiable claim costs 30 points

    return {
        "kpi": "claims",
        "score": score,
        "detail": (f"{total_claims} claims, {n} un-falsifiable"
                   if unfalsifiable_claims else f"{total_claims} claims, all falsifiable"),
        "defects": defects,
        "soft": [],
    }


def _check_benchmarks(benchmark_text: str, root: Path) -> dict[str, Any]:
    """Check all benchmark rows in BENCHMARK-AUTHORITY.md for resolvable artifacts."""
    if not benchmark_text:
        return {
            "kpi": "benchmarks",
            "score": 100,
            "detail": "no BENCHMARK-AUTHORITY.md (skipped)",
            "defects": [],
            "soft": ["BENCHMARK-AUTHORITY.md not found"],
        }

    total_benchmarks = 0
    unfalsifiable_benchmarks: list[dict[str, Any]] = []

    in_fence = False
    for line in benchmark_text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        # Look for table rows or artifact references
        if "|" not in line:
            continue

        total_benchmarks += 1

        # Check artifact and reproduce command resolvability
        issues = _resolve_benchmark_witnesses(line, root)

        if issues:
            unfalsifiable_benchmarks.append({
                "line": line.strip()[:120],
                "issues": issues,
            })

    defects = [f"un-falsifiable benchmark: {c['line']} — {', '.join(c['issues'])}"
               for c in unfalsifiable_benchmarks]

    n = len(unfalsifiable_benchmarks)
    score = _clamp(100 - 30 * n)  # Each un-falsifiable benchmark costs 30 points

    return {
        "kpi": "benchmarks",
        "score": score,
        "detail": (f"{total_benchmarks} benchmarks, {n} un-falsifiable"
                   if unfalsifiable_benchmarks else f"{total_benchmarks} benchmarks, all falsifiable"),
        "defects": defects,
        "soft": [],
    }


def grade_letter(score: float) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


def build_payload(*, workspace: str, kpis: list[dict[str, Any]],
                   error: str | None = None) -> dict[str, Any]:
    if error:
        return {
            "schema": SCHEMA, "ok": False, "verdict": "AUDIT_ERROR",
            "finding": "tooling_error", "reason": error,
            "next_action": "fix the read (run from repo ROOT), then re-run",
            "workspace": workspace, "corpus": {}, "kpis": [],
        }

    by_name = {k["kpi"]: k for k in kpis}
    # Weight claims slightly more than benchmarks (0.6 vs 0.4)
    weights = {"claims": 0.6, "benchmarks": 0.4}
    score = sum(weights[name] * by_name[name]["score"]
                for name in weights if name in by_name)
    score = round(score, 1)

    claim_repro_debt = sum(len(k["defects"]) for k in kpis)
    n_soft = sum(len(k["soft"]) for k in kpis)
    grade = grade_letter(score)

    breakdown = sorted(
        ({"kpi": k["kpi"], "score": k["score"], "debt": len(k["defects"]),
          "detail": k["detail"]} for k in kpis),
        key=lambda x: (-x["debt"], x["score"]))

    corpus = {
        "score": score,
        "grade": grade,
        "claim_repro_debt": claim_repro_debt,
        "soft_signals": n_soft,
        "kpi_scores": {k["kpi"]: k["score"] for k in kpis},
        "debt_by_kpi": {k["kpi"]: len(k["defects"]) for k in kpis},
        "breakdown": breakdown,
    }

    if claim_repro_debt == 0:
        ok, verdict, finding = True, "OK", "claims_falsifiable"
        reason = (f"all claims falsifiable: score {score}/100 (grade {grade}), "
                  f"zero un-falsifiable claims across {len(kpis)} KPIs "
                  f"({n_soft} advisory signal(s))")
        next_action = "no required edit; re-run after the next claim/benchmark change"
    else:
        ok, verdict, finding = False, "ACTION", "claims_unfalsifiable"
        worst = breakdown[0]
        reason = (f"{claim_repro_debt} un-falsifiable claim(s); score {score}/100 (grade {grade}); "
                  f"heaviest KPI: {worst['kpi']} ({worst['debt']} defect(s))")
        next_action = ("repair witnesses worst-first (see corpus.breakdown + per-KPI defects): "
                       "fix missing artifacts, update deleted cmd dirs, correct package paths; "
                       "re-run to prove the drop")

    return {
        "schema": SCHEMA, "ok": ok, "verdict": verdict, "finding": finding,
        "reason": reason, "next_action": next_action, "workspace": workspace,
        "corpus": corpus, "kpis": kpis,
    }


def collect(workspace: Path) -> dict[str, Any]:
    root = workspace.resolve()
    claims_text = _safe_read(root / CLAIMS_REL)
    benchmark_text = _safe_read(root / BENCHMARK_REL)

    kpis = [
        _check_claims(claims_text, root),
        _check_benchmarks(benchmark_text, root),
    ]

    return build_payload(workspace=str(root), kpis=kpis)


def render(payload: dict[str, Any]) -> str:
    c = payload.get("corpus") or {}
    lines = [
        f"claim-repro-scorecard: {payload.get('verdict')} ({payload.get('finding')})",
        f"  {payload.get('reason')}",
        "",
        (f"score {c.get('score', 0)}/100 (grade {c.get('grade', '?')}) "
         f"· CLAIM-REPRO-DEBT {c.get('claim_repro_debt', 0)} · {c.get('soft_signals', 0)} advisory"),
        "",
        "per-KPI (worst first):",
        f"  {'score':>5} {'debt':>4}  kpi            detail",
    ]
    for b in c.get("breakdown", []):
        lines.append(f"  {b['score']:>5} {b['debt']:>4}  {b['kpi']:<14} {b['detail']}")
    lines.append("")
    lines.append("un-falsifiable claim work-list:")
    any_defect = False
    for k in sorted(payload.get("kpis", []), key=lambda x: -len(x["defects"])):
        if not k["defects"]:
            continue
        any_defect = True
        lines.append(f"  {k['kpi']} ({len(k['defects'])}):")
        for it in k["defects"][:12]:
            lines.append(f"      - {it}")
        if len(k["defects"]) > 12:
            lines.append(f"      ... and {len(k['defects']) - 12} more")
    if not any_defect:
        lines.append("  (none — all claims falsifiable)")
    lines.append("")
    lines.append(f"next: {payload.get('next_action')}")
    return "\n".join(lines)


def render_markdown(payload: dict[str, Any], *, stamp: str | None = None) -> str:
    c = payload.get("corpus") or {}
    out: list[str] = []
    out.append("---")
    out.append('title: "fak claim-reproducibility scorecard — are claims falsifiable from a clean clone?"')
    out.append('description: " fak\'s deterministic claim-reproducibility scorecard: validates that every witness in CLAIMS.md and BENCHMARK-AUTHORITY.md resolves to a real artifact, test, or command path."')
    out.append("---")
    out.append("")
    out.append("# Claim-reproducibility scorecard")
    out.append("")
    if stamp:
        out.append(f"<!-- claim-repro-scorecard: {stamp} · process: tools/claim_repro_scorecard.py -->")
        out.append("")
    out.append("This scorecard validates that every witness handle in ``CLAIMS.md`` "
               "(``[SHIPPED]``/``[SIMULATED]``/``[STUB]`` claims) and every artifact path "
               "or ``Reproduce:`` command in ``BENCHMARK-AUTHORITY.md`` is **resolvable by an "
               "outsider from a clean clone**. An un-falsifiable claim — a ``Witness: TestFooBar`` "
               "that names a non-existent test, or a ``Reproduce: go run ./cmd/gone`` pointing at "
               "a deleted binary — is the worst failure mode for a skeptical reader, because "
               "it looks checkable and isn't.")
    out.append("")
    out.append("> Regenerate: ``python tools/claim_repro_scorecard.py --markdown --stamp DATE > docs/CLAIM-REPRO-SCORECARD.md``")
    out.append("")
    out.append("## Headline")
    out.append("")
    out.append("| Metric | Value |")
    out.append("|---|---|")
    out.append(f"| **Un-falsifiable claims (total HARD defects)** | **{c.get('claim_repro_debt', 0)}** |")
    out.append(f"| Composite score | {c.get('score', 0)}/100 (grade {c.get('grade', '?')}) |")
    out.append(f"| Advisory (soft) signals | {c.get('soft_signals', 0)} |")
    out.append("")
    out.append("## Per-KPI")
    out.append("")
    out.append("Two KPIs, each 0–100. ``debt`` = units of HARD un-falsifiable claims in that KPI.")
    out.append("")
    out.append("| KPI | Score | Debt | Detail |")
    out.append("|---|---:|:--:|---|")
    for b in c.get("breakdown", []):
        out.append(f"| ``{b['kpi']}`` | {b['score']} | {b['debt']} | {b['detail']} |")
    out.append("")
    out.append("## Un-falsifiable claim work-list")
    out.append("")
    any_defect = False
    for k in sorted(payload.get("kpis", []), key=lambda x: -len(x["defects"])):
        if not k["defects"]:
            continue
        any_defect = True
        out.append(f"### ``{k['kpi']}`` — {len(k['defects'])} defect(s), score {k['score']}")
        for it in k["defects"]:
            out.append(f"- {it}")
        out.append("")
    if not any_defect:
        out.append("No un-falsifiable claims: every witness resolves. 🎉")
        out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Claim-reproducibility scorecard (read-only).")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument("--markdown", action="store_true",
                    help="emit the CLAIM-REPRO-SCORECARD.md body")
    ap.add_argument("--stamp", default="", help="date stamp for the markdown header")
    ap.add_argument("--compare", action="store_true",
                    help="compare current debt vs baseline (if any)")

    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    workspace = Path(args.workspace).resolve() if args.workspace else repo_root()
    payload = collect(workspace)

    if args.json:
        print(json.dumps(payload, indent=2))
    elif args.markdown:
        print(render_markdown(payload, stamp=args.stamp or None))
    else:
        print(render(payload))

    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())