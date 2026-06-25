#!/usr/bin/env python3
"""Run one Codex shell command only after fak preflight admits its operation name."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any


PREFLIGHT_RE = re.compile(r"\bverdict=(?P<verdict>[A-Z_]+)\s+reason=(?P<reason>[A-Z_]+)\s+by=(?P<by>[A-Za-z0-9_-]+)")
DENIED_EXIT = 10
EXPECTATION_FAILED_EXIT = 11


def default_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_fak_command(root: Path) -> list[str]:
    exe = "fak.exe" if sys.platform == "win32" else "fak"
    local = root / exe
    if local.exists():
        return [str(local)]
    on_path = shutil.which("fak")
    if on_path:
        return [on_path]
    return ["go", "run", "./cmd/fak"]


def compact(text: str, limit: int = 800) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + "...[truncated]"


def parse_preflight(stdout: str, stderr: str, exit_code: int) -> dict[str, Any]:
    match = PREFLIGHT_RE.search(stdout)
    parsed = match.groupdict() if match else {}
    parsed["exit_code"] = exit_code
    parsed["stdout"] = compact(stdout)
    parsed["stderr"] = compact(stderr)
    return parsed


def fak_argv(args: argparse.Namespace) -> list[str]:
    base = [args.fak_bin] if args.fak_bin else default_fak_command(args.repo_root)
    return base + [
        "preflight",
        "--policy",
        args.policy,
        "--tool",
        args.tool,
        "--args",
        args.args_json,
    ]


def command_argv(args: argparse.Namespace) -> list[str]:
    command = list(args.command or [])
    if command and command[0] == "--":
        command = command[1:]
    return command


def command_digest(command: list[str]) -> str | None:
    if not command:
        return None
    payload = json.dumps(command, ensure_ascii=False, separators=(",", ":")).encode("utf-8", errors="replace")
    return hashlib.sha256(payload).hexdigest()


def command_identity(command: list[str], *, label: str | None, redact: bool) -> dict[str, Any]:
    digest = command_digest(command)
    out: dict[str, Any] = {
        "command_argc": len(command),
        "command_digest": digest,
        "command_executable": Path(command[0]).name if command else None,
        "command_label": label,
        "command_redacted": bool(redact),
    }
    if not redact:
        out["command"] = command
    return out


def run_gate(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    command = command_argv(args)
    preflight = subprocess.run(
        fak_argv(args),
        cwd=str(args.repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    parsed = parse_preflight(preflight.stdout, preflight.stderr, preflight.returncode)
    report: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "tool": args.tool,
        "policy": str(Path(args.policy).as_posix()),
        "preflight": parsed,
        "dry_run": bool(args.dry_run or not command),
        "executed": False,
        "expect_deny": bool(args.expect_deny),
    }
    report.update(command_identity(command, label=args.command_label, redact=args.redact_command))
    if args.expect_reason:
        report["expect_reason"] = args.expect_reason

    allowed = preflight.returncode == 0 and parsed.get("verdict") == "ALLOW"
    if not allowed:
        if args.expect_deny:
            if args.expect_reason and parsed.get("reason") != args.expect_reason:
                report["status"] = "DENIED_UNEXPECTED_REASON"
                return EXPECTATION_FAILED_EXIT, report
            report["status"] = "DENIED_EXPECTED"
            return 0, report
        report["status"] = "DENIED"
        return DENIED_EXIT, report

    if args.expect_deny:
        report["status"] = "UNEXPECTED_ALLOW"
        return EXPECTATION_FAILED_EXIT, report

    if args.dry_run or not command:
        report["status"] = "ALLOW"
        return 0, report

    report["executed"] = True
    if args.json:
        proc = subprocess.run(
            command,
            cwd=str(args.repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        report["command_exit_code"] = proc.returncode
        if args.include_command_output:
            report["command_stdout"] = compact(proc.stdout)
            report["command_stderr"] = compact(proc.stderr)
        else:
            report["command_output_dropped"] = ["stdout", "stderr"]
        report["status"] = "PASS" if proc.returncode == 0 else "COMMAND_FAILED"
        return proc.returncode, report

    print(f"fak gate: ALLOW {args.tool}; running {' '.join(command)}", file=sys.stderr)
    proc = subprocess.run(command, cwd=str(args.repo_root), check=False)
    report["command_exit_code"] = proc.returncode
    report["status"] = "PASS" if proc.returncode == 0 else "COMMAND_FAILED"
    return proc.returncode, report


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=default_repo_root())
    parser.add_argument("--fak-bin", help="use an already-built fak binary")
    parser.add_argument("--policy", default="examples/dev-agent-policy.json")
    parser.add_argument("--tool", required=True, help="policy operation name to preflight")
    parser.add_argument("--args-json", default="{}", help="JSON argument object for fak preflight")
    parser.add_argument("--dry-run", action="store_true", help="preflight only; do not execute the command")
    parser.add_argument("--expect-deny", action="store_true", help="succeed only when fak denies the operation; never run the command")
    parser.add_argument("--expect-reason", help="with --expect-deny, require this fak denial reason")
    parser.add_argument("--command-label", help="privacy-preserving label for the command identity in JSON reports")
    parser.add_argument("--redact-command", action="store_true", help="omit the raw command argv from JSON reports")
    parser.add_argument("--json", action="store_true", help="emit a machine-readable report")
    parser.add_argument("--include-command-output", action="store_true", help="copy captured command stdout/stderr into JSON reports")
    parser.add_argument("--out", type=Path, help="write the JSON report to this path")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="command to run after --")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    args.repo_root = args.repo_root.resolve()
    try:
        json.loads(args.args_json)
    except json.JSONDecodeError as exc:
        print(f"codex_fak_gate: --args-json is not valid JSON: {exc}", file=sys.stderr)
        return 2

    code, report = run_gate(args)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        verdict = report.get("preflight", {}).get("verdict", "UNKNOWN")
        reason = report.get("preflight", {}).get("reason", "UNKNOWN")
        print(f"fak gate: {report['status']} tool={args.tool} verdict={verdict} reason={reason}", file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
