#!/usr/bin/env python3
"""Run subsystem boundary checks and emit auditable baseline evidence."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
from dispatch_worker import install_no_window_subprocess_defaults
import sys
import time
from pathlib import Path
from typing import Any

import fleet_version
install_no_window_subprocess_defaults(subprocess)


SCHEMA = "fak.subsystem-check-audit.v1"
BASELINE_SCHEMA = "fak.subsystem-check-baseline.v1"
ROOT = Path(__file__).resolve().parents[1]


CHECKS: list[dict[str, Any]] = [
    {
        "id": "arch-request-path",
        "subsystem": "Architecture / request-path hygiene",
        "kind": "structural-test",
        "cwd": "fak",
        "argv": ["go", "test", "./internal/architest"],
        "proves": "Layering, request-path exec ban, registered leaves, OpenAI client ownership, and oracle seam placement.",
        "does_not_prove": "Runtime correctness of the leaf subsystems.",
    },
    {
        "id": "abi-policy-floor",
        "subsystem": "ABI / policy / adjudication floor",
        "kind": "contract-test",
        "cwd": "fak",
        "argv": ["go", "test", "./internal/abi", "./internal/adjudicator", "./internal/policy"],
        "proves": "Closed ABI/reason vocabulary, default-deny behavior, policy load failure modes, and policy round trips.",
        "does_not_prove": "That every allow-listed dangerous operation is semantically safe.",
    },
    {
        "id": "syscall-boundary-tax",
        "subsystem": "Syscall boundary tax",
        "kind": "boundary-benchmark",
        "cwd": "fak",
        "argv": ["go", "test", "./internal/bench", "./internal/metrics"],
        "proves": "The resident decide path and benchmark gate are alive; vDSO ablation changes the measured path.",
        "does_not_prove": "Production throughput or model quality.",
    },
    {
        "id": "result-admission",
        "subsystem": "Result admission / context-MMU",
        "kind": "security-boundary-test",
        "cwd": "fak",
        "argv": ["go", "test", "./internal/ctxmmu", "./internal/normgate"],
        "proves": "Poison, secret, oversize, and gated page-in handling at the admission boundary.",
        "does_not_prove": "Complete prompt-injection detection.",
    },
    {
        "id": "trust-taint-witness",
        "subsystem": "Information-flow / provenance / plan-CFI / witness",
        "kind": "security-contract-test",
        "cwd": "fak",
        "argv": ["go", "test", "./internal/ifc", "./internal/provenance", "./internal/plancfi", "./internal/witness"],
        "proves": "Kernel-authored trust/taint, vDSO taint preservation, plan deviation escalation, and git-backed claim checks.",
        "does_not_prove": "That every real-world sink or plan has been modeled.",
    },
    {
        "id": "gateway-wire",
        "subsystem": "Gateway / wire boundary",
        "kind": "adapter-test",
        "cwd": "fak",
        "argv": ["go", "test", "./internal/gateway"],
        "proves": "HTTP, MCP, and OpenAI-compatible adapters route through the kernel mechanisms and fail as values.",
        "does_not_prove": "External provider availability, auth state, or price.",
    },
    {
        "id": "recall-cdb-boundary",
        "subsystem": "Durable recall / process boundary",
        "kind": "persistence-test",
        "cwd": "fak",
        "argv": ["go", "test", "./internal/recall", "./internal/cdb"],
        "proves": "Quarantine survives persist/reload, reload re-screens content, corrupt CAS fails closed, and sealed bytes stay sealed.",
        "does_not_prove": "That long-term memory improves task quality.",
    },
    {
        "id": "kv-cache-reuse",
        "subsystem": "KV quarantine / cache reuse",
        "kind": "cache-contract-test",
        "cwd": "fak",
        "argv": ["go", "test", "./internal/kvmmu", "./internal/radixkv", "./internal/cachemeta", "./internal/vdso"],
        "proves": "Poison-driven KV eviction, radix split reuse, cache metadata, and vDSO invalidation controls.",
        "does_not_prove": "Zero-copy integration with an external serving engine.",
    },
    {
        "id": "model-kernel",
        "subsystem": "Model kernel / loader correctness",
        "kind": "oracle-test",
        "cwd": "fak",
        "argv": ["go", "test", "./internal/model"],
        "proves": "Go model paths match their oracles and loader/quant/prefix semantics stay stable.",
        "does_not_prove": "Production task capability.",
    },
    {
        "id": "compute-hal",
        "subsystem": "Compute HAL / device abstraction",
        "kind": "runtime-boundary-test",
        "cwd": "fak",
        "argv": ["go", "test", "./internal/compute", "./internal/model", "-run", "HAL|Device|Evict|Clone|Q8Dispatch"],
        "proves": "The backend seam can drive the model loop and enforce correctness classes against CPU references.",
        "does_not_prove": "That a non-reference GPU backend is production-ready.",
    },
    {
        "id": "turn-tax-controls",
        "subsystem": "Turn-tax mechanism controls",
        "kind": "control-benchmark",
        "cwd": "fak",
        "argv": ["go", "test", "./internal/turnbench", "-run", "Run_|HappyPath|VDSO|Safety"],
        "proves": "Turn-tax classes are live events, vDSO is a real swap, safety is separated, and happy path saves zero.",
        "does_not_prove": "General task-quality improvement.",
    },
    {
        "id": "fleet-fanout-controls",
        "subsystem": "Fleet / fan-out anti-inflation controls",
        "kind": "control-benchmark",
        "cwd": "fak",
        "argv": ["go", "test", "./internal/turnbench", "-run", "Fleet_|Fanout"],
        "proves": "No-share and single-agent controls zero out cross-agent uplift, and fan-out prefix geometry is explicit.",
        "does_not_prove": "That production workloads are read-heavy enough to benefit.",
    },
    {
        "id": "phase1-capability-gate",
        "subsystem": "Phase 1 capability gate logic",
        "kind": "gate-test",
        "cwd": "fak",
        "argv": ["go", "test", "./internal/turnbench", "-run", "Phase1CapabilityGate"],
        "proves": "The Phase 1 gate requires live local-GPU 7-9B parity evidence and rejects stale or wrong-class evidence.",
        "does_not_prove": "That such hardware evidence exists today.",
    },
    {
        "id": "api-host-tooling",
        "subsystem": "API-host bridge tooling",
        "kind": "artifact-gate-test",
        "cwd": ".",
        "argv": [sys.executable, "-m", "pytest", "tools/api_host_*_test.py"],
        "proves": "Candidate host and bridge artifacts fail closed on auth, wire, staleness, and requirement drift.",
        "does_not_prove": "That a specific live provider is available or paid for.",
    },
    # dgx-artifact-integrity check removed from the public copy -- it ran
    # pytest tools/dgx_*_test.py (operator-private lab infra, excluded). See
    # PUBLIC-SCRUB-POLICY.md PRIVATE-ONLY list.
    {
        "id": "release-process-safety",
        "subsystem": "Release / process safety",
        "kind": "process-test",
        "cwd": ".",
        "argv": [sys.executable, "-m", "pytest", "tools/release_lock_test.py", "tools/safe_ff_sync_test.py"],
        "proves": "Release ownership and fast-forward sync workflows fail closed.",
        "does_not_prove": "Product performance or model quality.",
    },
]

PROFILE_IDS: dict[str, list[str]] = {
    "smoke": [
        "arch-request-path",
        "abi-policy-floor",
        "syscall-boundary-tax",
        "result-admission",
        "turn-tax-controls",
    ],
    "core": [
        "arch-request-path",
        "abi-policy-floor",
        "syscall-boundary-tax",
        "result-admission",
        "trust-taint-witness",
        "gateway-wire",
        "recall-cdb-boundary",
        "kv-cache-reuse",
        "model-kernel",
        "compute-hal",
        "turn-tax-controls",
        "fleet-fanout-controls",
        "phase1-capability-gate",
    ],
    "tooling": [
        "api-host-tooling",
        "release-process-safety",
    ],
}
PROFILE_IDS["all"] = PROFILE_IDS["core"] + PROFILE_IDS["tooling"]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def tail(text: str, n: int = 4000) -> str:
    return text[-n:]


def command_text(argv: list[str]) -> str:
    return " ".join(argv)


def repo_rel(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def expand_globs(root: Path, cwd_rel: str, argv: list[str]) -> list[str]:
    cwd = root / cwd_rel if cwd_rel else root
    expanded: list[str] = []
    for arg in argv:
        if not any(ch in arg for ch in "*?["):
            expanded.append(arg)
            continue
        matches = sorted(cwd.glob(arg.replace("\\", "/")))
        if not matches:
            expanded.append(arg)
            continue
        expanded.extend(repo_rel(cwd, match) for match in matches)
    return expanded


def run_git(root: Path, argv: list[str]) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            ["git", *argv],
            cwd=str(root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)
    return proc.returncode, proc.stdout.strip() or proc.stderr.strip()


def git_snapshot(root: Path) -> dict[str, Any]:
    head_rc, head = run_git(root, ["rev-parse", "HEAD"])
    branch_rc, branch = run_git(root, ["rev-parse", "--abbrev-ref", "HEAD"])
    status_rc, status = run_git(root, ["status", "--porcelain=v1"])
    rows = [line for line in status.splitlines() if line.strip()] if status_rc == 0 else []
    return {
        "head": head if head_rc == 0 else "",
        "branch": branch if branch_rc == 0 else "",
        "dirty": bool(rows) if status_rc == 0 else None,
        "dirty_count": len(rows) if status_rc == 0 else None,
        "status_sha256": digest(status),
        "status_sample": rows[:80],
        "status_error": "" if status_rc == 0 else status,
    }


def resolve_checks(profile: str, only: list[str] | None = None) -> list[dict[str, Any]]:
    index = {check["id"]: check for check in CHECKS}
    ids = only or PROFILE_IDS[profile]
    missing = [check_id for check_id in ids if check_id not in index]
    if missing:
        raise ValueError(f"unknown check id(s): {', '.join(missing)}")
    return [index[check_id] for check_id in ids]


def run_command(root: Path, check: dict[str, Any], timeout_s: int) -> dict[str, Any]:
    argv = list(check["argv"])
    cwd_rel = str(check.get("cwd") or "")
    cwd = root / cwd_rel if cwd_rel else root
    resolved_argv = expand_globs(root, cwd_rel, argv)
    base = {
        "id": check["id"],
        "subsystem": check["subsystem"],
        "kind": check["kind"],
        "cwd": cwd_rel,
        "command": command_text(argv),
        "argv": argv,
        "resolved_argv": resolved_argv,
        "proves": check["proves"],
        "does_not_prove": check["does_not_prove"],
        "started_at": utc_now(),
    }
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            resolved_argv,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
        )
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        combined = proc.stdout + proc.stderr
        return {
            **base,
            "status": "passed" if proc.returncode == 0 else "failed",
            "exit_code": proc.returncode,
            "elapsed_ms": elapsed_ms,
            "stdout_tail": tail(proc.stdout),
            "stderr_tail": tail(proc.stderr),
            "output_sha256": digest(combined),
        }
    except subprocess.TimeoutExpired as exc:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        return {
            **base,
            "status": "timeout",
            "exit_code": None,
            "elapsed_ms": elapsed_ms,
            "stdout_tail": tail(stdout),
            "stderr_tail": tail(stderr),
            "output_sha256": digest(stdout + stderr),
            "timeout_s": timeout_s,
        }
    except OSError as exc:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        message = str(exc)
        return {
            **base,
            "status": "error",
            "exit_code": None,
            "elapsed_ms": elapsed_ms,
            "stdout_tail": "",
            "stderr_tail": tail(message),
            "output_sha256": digest(message),
            "error_type": type(exc).__name__,
        }


def pending_result(check: dict[str, Any], status: str, reason: str) -> dict[str, Any]:
    return {
        "id": check["id"],
        "subsystem": check["subsystem"],
        "kind": check["kind"],
        "cwd": check.get("cwd", ""),
        "command": command_text(list(check["argv"])),
        "argv": list(check["argv"]),
        "resolved_argv": expand_globs(Path.cwd(), str(check.get("cwd") or ""), list(check["argv"])),
        "proves": check["proves"],
        "does_not_prove": check["does_not_prove"],
        "status": status,
        "exit_code": None,
        "elapsed_ms": 0,
        "reason": reason,
        "stdout_tail": "",
        "stderr_tail": "",
        "output_sha256": digest(reason),
    }


def compare_baseline(
    report: dict[str, Any],
    baseline: dict[str, Any] | None,
    duration_tolerance_ratio: float,
    duration_min_slack_ms: int,
    fail_on_duration_regression: bool,
) -> dict[str, Any]:
    comparison: dict[str, Any] = {
        "baseline_path": "",
        "status": "not_configured",
        "failures": [],
        "warnings": [],
        "duration_tolerance_ratio": duration_tolerance_ratio,
        "duration_min_slack_ms": duration_min_slack_ms,
        "fail_on_duration_regression": fail_on_duration_regression,
    }
    if baseline is None:
        return comparison
    comparison["status"] = "passed"
    comparison["baseline_schema"] = baseline.get("schema", "")
    if baseline.get("schema") != BASELINE_SCHEMA:
        comparison["failures"].append({"reason": "bad_baseline_schema", "expected": BASELINE_SCHEMA, "actual": baseline.get("schema", "")})
    if baseline.get("profile") and baseline.get("profile") != report.get("profile"):
        comparison["failures"].append({"reason": "profile_mismatch", "expected": baseline.get("profile"), "actual": report.get("profile")})

    current = {run["id"]: run for run in report["runs"]}
    expected = {row["id"]: row for row in baseline.get("checks", []) if isinstance(row, dict) and row.get("id")}
    for check_id, expected_row in expected.items():
        run = current.get(check_id)
        if run is None:
            comparison["failures"].append({"id": check_id, "reason": "missing_current_check"})
            continue
        if expected_row.get("command") and expected_row.get("command") != run.get("command"):
            comparison["failures"].append(
                {
                    "id": check_id,
                    "reason": "command_drift",
                    "expected": expected_row.get("command"),
                    "actual": run.get("command"),
                }
            )
        expected_status = expected_row.get("expected_status", "passed")
        if expected_status == "passed" and run.get("status") != "passed":
            comparison["failures"].append(
                {
                    "id": check_id,
                    "reason": "pass_to_nonpass",
                    "expected": "passed",
                    "actual": run.get("status"),
                    "exit_code": run.get("exit_code"),
                }
            )
        baseline_ms = expected_row.get("elapsed_ms")
        current_ms = run.get("elapsed_ms")
        if isinstance(baseline_ms, (int, float)) and baseline_ms > 0 and isinstance(current_ms, (int, float)):
            ratio_limit_ms = int(baseline_ms * (1.0 + duration_tolerance_ratio))
            slack_limit_ms = int(baseline_ms + duration_min_slack_ms)
            allowed_ms = max(ratio_limit_ms, slack_limit_ms)
            if current_ms > allowed_ms:
                finding = {
                    "id": check_id,
                    "reason": "duration_regression",
                    "baseline_ms": baseline_ms,
                    "actual_ms": current_ms,
                    "allowed_ms": allowed_ms,
                    "ratio_limit_ms": ratio_limit_ms,
                    "slack_limit_ms": slack_limit_ms,
                }
                if fail_on_duration_regression:
                    comparison["failures"].append(finding)
                else:
                    comparison["warnings"].append(finding)

    for check_id in sorted(set(current) - set(expected)):
        comparison["warnings"].append({"id": check_id, "reason": "new_check_not_in_baseline"})
    if comparison["failures"]:
        comparison["status"] = "failed"
    elif comparison["warnings"]:
        comparison["status"] = "passed_with_warnings"
    return comparison


def build_report(
    root: Path,
    profile: str,
    checks: list[dict[str, Any]],
    timeout_s: int,
    execute: bool,
    baseline: dict[str, Any] | None,
    baseline_path: str,
    duration_tolerance_ratio: float,
    duration_min_slack_ms: int,
    fail_on_duration_regression: bool,
) -> dict[str, Any]:
    runs = [
        run_command(root, check, timeout_s) if execute else pending_result(check, "not_run", "execute=false")
        for check in checks
    ]
    failed = [run for run in runs if run["status"] != "passed"]
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "app_version": fleet_version.app_version(root),
        "generated_at": utc_now(),
        "root": str(root),
        "profile": profile,
        "timeout_s": timeout_s,
        "git": git_snapshot(root),
        "summary": {
            "total": len(runs),
            "passed": len(runs) - len(failed),
            "failed": len(failed),
            "status": "failed" if failed else "passed",
        },
        "runs": runs,
    }
    comparison = compare_baseline(report, baseline, duration_tolerance_ratio, duration_min_slack_ms, fail_on_duration_regression)
    comparison["baseline_path"] = baseline_path
    report["comparison"] = comparison
    report["summary"]["baseline_failures"] = len(comparison["failures"])
    report["summary"]["baseline_warnings"] = len(comparison["warnings"])
    if comparison["failures"]:
        report["summary"]["status"] = "failed"
    return report


def baseline_from_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": BASELINE_SCHEMA,
        "generated_at": utc_now(),
        "generated_from_schema": report.get("schema", ""),
        "app_version": report.get("app_version", ""),
        "profile": report.get("profile", ""),
        "git": report.get("git", {}),
        "duration_policy": {
            "default_tolerance_ratio": report.get("comparison", {}).get("duration_tolerance_ratio", 0.5),
            "default_min_slack_ms": report.get("comparison", {}).get("duration_min_slack_ms", 2000),
            "duration_regression_is_failure": report.get("comparison", {}).get("fail_on_duration_regression", False),
        },
        "checks": [
            {
                "id": run["id"],
                "subsystem": run["subsystem"],
                "kind": run["kind"],
                "cwd": run["cwd"],
                "command": run["command"],
                "expected_status": "passed",
                "elapsed_ms": run["elapsed_ms"],
                "output_sha256": run["output_sha256"],
            }
            for run in report.get("runs", [])
        ],
    }


def load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_text(path: str, body: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")


def escape_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    git = report["git"]
    comparison = report["comparison"]
    lines = [
        "# Subsystem Check Audit",
        "",
        f"- Profile: `{report['profile']}`",
        f"- Generated: {report['generated_at']}",
        f"- App version: `{report['app_version']}`",
        f"- Git: `{git.get('branch', '')}` `{str(git.get('head', ''))[:12]}`; dirty={git.get('dirty')} dirty_count={git.get('dirty_count')}",
        f"- Checks passed: {summary['passed']}/{summary['total']}",
        f"- Baseline comparison: {comparison['status']} ({summary['baseline_failures']} failures, {summary['baseline_warnings']} warnings)",
        "",
        "| check | subsystem | status | elapsed ms | command |",
        "|---|---|---|---:|---|",
    ]
    for run in report["runs"]:
        lines.append(
            "| `{}` | {} | {} | {} | `{}` |".format(
                escape_cell(run["id"]),
                escape_cell(run["subsystem"]),
                escape_cell(run["status"]),
                escape_cell(run["elapsed_ms"]),
                escape_cell(run["command"]),
            )
        )
    if comparison["failures"]:
        lines.extend(["", "## Baseline Failures", ""])
        for failure in comparison["failures"]:
            lines.append(f"- `{failure.get('id', '')}` {failure.get('reason', '')}: `{json.dumps(failure, sort_keys=True)}`")
    if comparison["warnings"]:
        lines.extend(["", "## Baseline Warnings", ""])
        for warning in comparison["warnings"]:
            lines.append(f"- `{warning.get('id', '')}` {warning.get('reason', '')}: `{json.dumps(warning, sort_keys=True)}`")
    lines.append("")
    return "\n".join(lines)


def list_checks(profile: str, checks: list[dict[str, Any]]) -> str:
    lines = [f"Profile: {profile}", ""]
    for check in checks:
        lines.append(f"- {check['id']}: {check['subsystem']} ({command_text(check['argv'])})")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run FAK subsystem checks and emit auditable JSON/Markdown evidence")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--profile", choices=sorted(PROFILE_IDS), default="smoke")
    ap.add_argument("--only", action="append", default=[], help="Run only this check id; may be repeated")
    ap.add_argument("--timeout-s", type=int, default=240)
    ap.add_argument("--baseline", default="", help="Optional baseline JSON to compare against")
    ap.add_argument("--write-baseline", default="", help="Write a pass-only baseline JSON from this run")
    ap.add_argument("--duration-tolerance-ratio", type=float, default=0.50)
    ap.add_argument("--duration-min-slack-ms", type=int, default=2000)
    ap.add_argument("--fail-on-duration-regression", action="store_true")
    ap.add_argument("--out-json", default="")
    ap.add_argument("--out-md", default="")
    ap.add_argument("--index-only", action="store_true", help="Emit the selected check index without executing commands")
    ap.add_argument("--list", action="store_true", help="Print selected checks and exit")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).resolve()
    try:
        checks = resolve_checks(args.profile, args.only or None)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.list:
        print(list_checks(args.profile, checks))
        return 0
    baseline = load_json(args.baseline) if args.baseline else None
    report = build_report(
        root=root,
        profile=args.profile,
        checks=checks,
        timeout_s=args.timeout_s,
        execute=not args.index_only,
        baseline=baseline,
        baseline_path=args.baseline,
        duration_tolerance_ratio=args.duration_tolerance_ratio,
        duration_min_slack_ms=args.duration_min_slack_ms,
        fail_on_duration_regression=args.fail_on_duration_regression,
    )
    body = json.dumps(report, indent=2) + "\n"
    if args.out_json:
        write_text(args.out_json, body)
    else:
        print(body, end="")
    if args.out_md:
        write_text(args.out_md, markdown(report))
    if args.write_baseline:
        if report["summary"]["failed"] or report["summary"]["baseline_failures"]:
            print("refusing to write baseline from a failing run", file=sys.stderr)
            return 1
        write_text(args.write_baseline, json.dumps(baseline_from_report(report), indent=2) + "\n")
    return 0 if report["summary"]["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
