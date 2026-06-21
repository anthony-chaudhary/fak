#!/usr/bin/env python3
"""Run fak's local laptop test lanes.

This is the operator entry point for Anthony's laptop-shaped hardware split:

  * cpu     -> the normal Go test runner on the CPU/Intel lane
  * nvidia  -> the WSL/Linux CUDA backend witness lane
  * all     -> CPU first, then NVIDIA

The script is deliberately thin. It delegates to the repo's existing runners
(`fak/test.ps1`, `fak/test.sh`, and `fak/internal/compute/build_cuda.sh`) and only
normalizes host selection, WSL invocation, and dry-run output.
"""

from __future__ import annotations

import argparse
import codecs
import datetime as dt
import functools
import hashlib
import json
import os
import platform as host_platform
import posixpath
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass


CPU_SMOKE_ARGS = (
    "-run",
    "TestHALSessionMatchesLegacyCPUReference|TestMatMulDelegatesVerbatim|TestQ8DispatchIsApproxAndGated|TestRegistryPickAndDefault",
    "./internal/compute/",
    "./internal/model/",
)
CPU_DEFAULT_ARGS = ("./...",)
NVIDIA_ACTIONS = ("build", "test", "bench")
OUTPUT_TAIL_CHARS = 20000
DEFAULT_PRECHECK_REPORT = "fak/experiments/gpu/laptop-check.json"
DEFAULT_VERIFY_CHECK_REPORT = "fak/experiments/gpu/laptop-post-setup.json"
DEFAULT_VERIFY_RUN_REPORT = "fak/experiments/gpu/laptop-all.json"
DEFAULT_CPU_CHECK_REPORT = "fak/experiments/gpu/laptop-cpu-check.json"
DEFAULT_CPU_RUN_REPORT = "fak/experiments/gpu/laptop-cpu.json"
VERIFY_CHECK_NAMES = ("cpu", "nvidia", "cuda-toolchain")
VERIFY_RUN_LABELS = ("cpu", "nvidia-setup", "nvidia")
VERIFY_CPU_CHECK_NAMES = ("cpu",)
VERIFY_CPU_RUN_LABELS = ("cpu",)
STATUS_SAMPLE_LIMIT = 20


@dataclass(frozen=True)
class Command:
    label: str
    argv: tuple[str, ...]
    cwd: Path
    env: dict[str, str]


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def host_report(root: Path | None = None) -> dict:
    return {
        "node": host_platform.node() or "unknown",
        "system": host_platform.system() or "unknown",
        "release": host_platform.release() or "unknown",
        "machine": host_platform.machine() or "unknown",
        "python": sys.executable or "unknown",
        "repo_root": str(root or repo_root()),
    }


def git_output(root: Path, *args: str) -> tuple[bool, str]:
    try:
        cp = subprocess.run(
            ("git", *args),
            cwd=str(root),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False, ""
    return cp.returncode == 0, cp.stdout.strip()


def repo_report(root: Path | None = None) -> dict:
    repo = root or repo_root()
    ok_head, head = git_output(repo, "rev-parse", "HEAD")
    ok_branch, branch = git_output(repo, "branch", "--show-current")
    if not branch:
        ok_branch, branch = git_output(repo, "rev-parse", "--abbrev-ref", "HEAD")
    ok_status, status = git_output(repo, "status", "--porcelain", "--untracked-files=all")
    status_lines = [line for line in status.splitlines() if line]
    status_fingerprint = hashlib.sha256(status.encode("utf-8")).hexdigest() if ok_status else "unknown"
    return {
        "git_available": bool(ok_head and ok_status),
        "head": head or "unknown",
        "branch": branch or ("unknown" if ok_branch else "unknown"),
        "dirty": bool(status_lines),
        "status_count": len(status_lines) if ok_status else -1,
        "status_fingerprint": status_fingerprint,
        "status_sample": status_lines[:STATUS_SAMPLE_LIMIT] if ok_status else [],
        "status_truncated": len(status_lines) > STATUS_SAMPLE_LIMIT if ok_status else False,
    }


def normalize_remainder(values: list[str]) -> tuple[str, ...]:
    if values and values[0] == "--":
        values = values[1:]
    return tuple(values)


def is_windows(platform: str | None = None) -> bool:
    return (platform or sys.platform).startswith("win")


def windows_drive_to_wsl_path(path: str | Path) -> str:
    raw = str(path)
    match = re.match(r"^([A-Za-z]):[\\/](.*)$", raw)
    if not match:
        raise ValueError(f"cannot map non-drive Windows path to WSL /mnt path: {raw!r}")
    drive = match.group(1).lower()
    tail = match.group(2).replace("\\", "/")
    return f"/mnt/{drive}/{tail}"


def host_env(base: dict[str, str], args: argparse.Namespace) -> dict[str, str]:
    env = dict(base)
    if args.fast:
        env["FAK_FAST"] = "1"
    if args.wsl_distro:
        env["FAK_WSL_DISTRO"] = args.wsl_distro
    return env


def cpu_args(args: argparse.Namespace) -> tuple[str, ...]:
    forwarded = normalize_remainder(args.go_test_args)
    if forwarded:
        return forwarded
    if args.smoke:
        return CPU_SMOKE_ARGS
    return CPU_DEFAULT_ARGS


def cpu_command(root: Path, args: argparse.Namespace, platform: str | None = None) -> Command:
    env = host_env(os.environ, args)
    forwarded = cpu_args(args)
    if is_windows(platform):
        ps1 = root / "fak" / "test.ps1"
        argv = (
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ps1),
            *forwarded,
        )
    else:
        test_sh = posixpath.join(str(root).replace("\\", "/"), "fak", "test.sh")
        argv = ("bash", test_sh, *forwarded)
    return Command("cpu", argv, root, env)


def clean_wsl_distro_name(value: str) -> str:
    return value.replace("\x00", "").lstrip("\ufeff\ufffe").strip()


def decode_wsl_list_output(raw: bytes) -> str:
    if raw.startswith(codecs.BOM_UTF16_LE) or raw.startswith(codecs.BOM_UTF16_BE):
        return raw.decode("utf-16", errors="replace")
    if b"\x00" in raw:
        return raw.decode("utf-16le", errors="replace")
    return raw.decode("utf-8", errors="replace")


@functools.lru_cache(maxsize=1)
def list_wsl_distros() -> tuple[str, ...]:
    try:
        cp = subprocess.run(
            ["wsl.exe", "--list", "--quiet"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ()
    if cp.returncode != 0:
        return ()
    text = decode_wsl_list_output(cp.stdout)
    return tuple(name for line in text.splitlines() if (name := clean_wsl_distro_name(line)))


def preferred_wsl_distro(args: argparse.Namespace, platform: str | None = None) -> str:
    distro = args.wsl_distro or os.environ.get("FAK_WSL_DISTRO", "")
    if distro or not is_windows(platform):
        return distro
    installed = list_wsl_distros()
    if "Ubuntu-24.04" in installed:
        return "Ubuntu-24.04"
    return ""


def wsl_prefix(args: argparse.Namespace, platform: str | None = None) -> tuple[str, ...]:
    distro = preferred_wsl_distro(args, platform)
    if distro:
        return ("wsl.exe", "-d", distro)
    return ("wsl.exe",)


def wsl_report(args: argparse.Namespace, platform: str | None = None) -> dict:
    host = platform or sys.platform
    if not is_windows(host):
        return {"platform": host, "source": "native", "distro": "not-applicable"}
    explicit = args.wsl_distro or os.environ.get("FAK_WSL_DISTRO", "")
    selected = preferred_wsl_distro(args, host)
    if explicit:
        return {"platform": host, "source": "explicit", "distro": selected or explicit}
    if selected:
        return {"platform": host, "source": "preferred", "distro": selected}
    return {"platform": host, "source": "default", "distro": "default"}


def run_probe(argv: tuple[str, ...], cwd: Path, env: dict[str, str], timeout: float = 20.0) -> tuple[bool, str]:
    try:
        cp = subprocess.run(
            list(argv),
            cwd=str(cwd),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    out = "\n".join(part for part in (cp.stdout.strip(), cp.stderr.strip()) if part)
    return cp.returncode == 0, out


def one_line_detail(text: str) -> str:
    return "; ".join(line.strip() for line in text.splitlines() if line.strip())


def go_env_probe_script() -> str:
    return (
        "export PATH=/usr/local/go/bin:$HOME/go/bin:$PATH; "
        "command -v go >/dev/null; "
        "go version; "
        "printf 'GOOS=%s GOARCH=%s\\n' \"$(go env GOOS)\" \"$(go env GOARCH)\""
    )


def tail_text(text: str, limit: int = OUTPUT_TAIL_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def write_report(path: str, report: dict, root: Path | None = None) -> None:
    if not path:
        return
    out = resolve_report_path(root or repo_root(), path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[report] wrote {out}", flush=True)


def resolve_report_path(root: Path, path: str) -> Path:
    report_path = Path(path)
    if report_path.is_absolute():
        return report_path
    return root / report_path


def read_json_report(path: Path) -> tuple[dict | None, list[str]]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        return None, [f"{path}: cannot read report: {exc}"]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, [f"{path}: invalid JSON: {exc}"]
    if not isinstance(data, dict):
        return None, [f"{path}: report root is not a JSON object"]
    return data, []


def normalized_report_path(path: str) -> str:
    return posixpath.normpath(path.replace("\\", "/")).lower()


def report_path_matches(path: str, default_path: str) -> bool:
    if not path:
        return False
    path_norm = normalized_report_path(path)
    default_norm = normalized_report_path(default_path)
    return path_norm == default_norm or path_norm.endswith("/" + default_norm)


def proof_mode(args: argparse.Namespace) -> str:
    explicit = getattr(args, "proof_mode", "")
    if explicit:
        return explicit
    if getattr(args, "cpu_only", False):
        return "cpu-only"
    lane = getattr(args, "lane", "")
    out = getattr(args, "out", "")
    if lane == "all":
        return "cpu+nvidia"
    if lane == "check":
        if getattr(args, "require_nvidia", False) or getattr(args, "require_cuda_toolchain", False):
            return "cpu+nvidia"
        if report_path_matches(out, DEFAULT_CPU_CHECK_REPORT):
            return "cpu-only"
    if lane == "cpu" and report_path_matches(out, DEFAULT_CPU_RUN_REPORT):
        return "cpu-only"
    return "manual"


def proof_stage(args: argparse.Namespace) -> str:
    explicit = getattr(args, "proof_stage", "")
    if explicit:
        return explicit
    lane = getattr(args, "lane", "unknown")
    mode = proof_mode(args)
    out = getattr(args, "out", "")
    if mode == "cpu-only":
        if lane == "check":
            return "cpu-check"
        if lane == "cpu":
            return "cpu-run"
    if mode == "cpu+nvidia":
        if lane == "all":
            return "run"
        if lane == "check":
            if getattr(args, "require_cuda_toolchain", False) or report_path_matches(out, DEFAULT_VERIFY_CHECK_REPORT):
                return "postcheck"
            return "precheck"
    return lane


def cpu_scope_name(args: argparse.Namespace) -> str:
    explicit = getattr(args, "cpu_scope", "")
    if explicit:
        return explicit
    forwarded = normalize_remainder(list(getattr(args, "go_test_args", ())))
    if forwarded == CPU_DEFAULT_ARGS:
        return "full"
    if forwarded:
        return "custom"
    return "smoke" if getattr(args, "smoke", False) else "full"


def proof_report(args: argparse.Namespace, kind: str) -> dict:
    proof = {
        "mode": proof_mode(args),
        "stage": proof_stage(args),
    }
    if kind == "run":
        proof["cpu_scope"] = cpu_scope_name(args) if getattr(args, "lane", "") in {"cpu", "all"} else "none"
    elif getattr(args, "cpu_scope", ""):
        proof["cpu_scope"] = getattr(args, "cpu_scope")
    return proof


def check_summary(checks: list[CheckResult], args: argparse.Namespace, rc: int) -> dict:
    proof = proof_report(args, "check")
    required = check_required_names(args)
    return {
        "ok": rc == 0,
        "proof_mode": proof["mode"],
        "proof_stage": proof["stage"],
        "required": sorted(required),
        "failed_required": sorted(check.name for check in checks if check.name in required and not check.ok),
        "ok_checks": sorted(check.name for check in checks if check.ok),
        "missing_checks": sorted(check.name for check in checks if not check.ok),
    }


def run_summary(plan: list[Command], args: argparse.Namespace, rc: int, results: list[dict]) -> dict:
    proof = proof_report(args, "run")
    command_results = results if results else [merged(command_report_base(cmd), {"skipped": bool(args.dry_run), "exit_code": None}) for cmd in plan]
    return {
        "ok": rc == 0,
        "proof_mode": proof["mode"],
        "proof_stage": proof["stage"],
        "cpu_scope": proof.get("cpu_scope", "none"),
        "command_count": len(command_results),
        "failed_commands": sorted(
            cmd.get("label", "")
            for cmd in command_results
            if cmd.get("exit_code") not in (0, None)
        ),
        "skipped_commands": sorted(
            cmd.get("label", "")
            for cmd in command_results
            if cmd.get("skipped") is True
        ),
    }


def expected_proof_mode(args: argparse.Namespace) -> str:
    return "cpu-only" if args.cpu_only else "cpu+nvidia"


def current_repo_errors(path: Path, report: dict, root: Path, allow_stale_repo: bool) -> list[str]:
    if allow_stale_repo:
        return []
    report_repo = report.get("repo")
    if not isinstance(report_repo, dict):
        return []
    report_head = report_repo.get("head")
    if not isinstance(report_head, str) or not report_head or report_head == "unknown":
        return []
    current = repo_report(root)
    if current.get("git_available") is not True:
        return [f"{path}: current repo metadata is unavailable; use --allow-stale-repo to inspect saved artifacts anyway"]
    current_head = current.get("head")
    if report_head != current_head:
        return [f"{path}: repo.head is {report_head!r}, current HEAD is {current_head!r}; use --allow-stale-repo for old artifacts"]
    report_fingerprint = report_repo.get("status_fingerprint")
    current_fingerprint = current.get("status_fingerprint")
    if not isinstance(report_fingerprint, str) or not report_fingerprint:
        return [f"{path}: repo.status_fingerprint is missing; rerun the laptop lane or use --allow-stale-repo for old artifacts"]
    if report_fingerprint != current_fingerprint:
        return [
            f"{path}: repo.status_fingerprint is {report_fingerprint!r}, "
            f"current fingerprint is {current_fingerprint!r}; use --allow-stale-repo for old artifacts"
        ]
    return []


def verify_common_report(path: Path, report: dict, kind: str, expected_mode: str) -> list[str]:
    errors: list[str] = []
    if report.get("schema") != "fak.laptop-test.v1":
        errors.append(f"{path}: unexpected schema {report.get('schema')!r}")
    if report.get("kind") != kind:
        errors.append(f"{path}: expected kind {kind!r}, got {report.get('kind')!r}")
    if report.get("exit_code") != 0:
        errors.append(f"{path}: report exit_code is {report.get('exit_code')!r}, want 0")
    host = report.get("host")
    if not isinstance(host, dict):
        errors.append(f"{path}: missing host metadata")
    else:
        for field in ("node", "system", "release", "machine", "python", "repo_root"):
            if not isinstance(host.get(field), str) or not host.get(field):
                errors.append(f"{path}: host.{field} is missing")
    repo = report.get("repo")
    if not isinstance(repo, dict):
        errors.append(f"{path}: missing repo metadata")
    else:
        if repo.get("git_available") is not True:
            errors.append(f"{path}: repo.git_available is not true")
        for field in ("head", "branch"):
            if not isinstance(repo.get(field), str) or not repo.get(field):
                errors.append(f"{path}: repo.{field} is missing")
        if not isinstance(repo.get("dirty"), bool):
            errors.append(f"{path}: repo.dirty is missing")
        if not isinstance(repo.get("status_count"), int) or repo.get("status_count") < 0:
            errors.append(f"{path}: repo.status_count is missing")
        if not isinstance(repo.get("status_fingerprint"), str) or not repo.get("status_fingerprint"):
            errors.append(f"{path}: repo.status_fingerprint is missing")
        if not isinstance(repo.get("status_sample"), list):
            errors.append(f"{path}: repo.status_sample is missing")
        if not isinstance(repo.get("status_truncated"), bool):
            errors.append(f"{path}: repo.status_truncated is missing")
    wsl = report.get("wsl")
    if not isinstance(wsl, dict):
        errors.append(f"{path}: missing WSL metadata")
    else:
        for field in ("platform", "source", "distro"):
            if not isinstance(wsl.get(field), str) or not wsl.get(field):
                errors.append(f"{path}: wsl.{field} is missing")
    proof = report.get("proof")
    if not isinstance(proof, dict):
        errors.append(f"{path}: missing proof metadata")
    else:
        mode = proof.get("mode")
        if not isinstance(mode, str) or not mode:
            errors.append(f"{path}: proof.mode is missing")
        elif expected_mode and mode != expected_mode:
            errors.append(f"{path}: proof.mode is {mode!r}, want {expected_mode!r}")
        if not isinstance(proof.get("stage"), str) or not proof.get("stage"):
            errors.append(f"{path}: proof.stage is missing")
    summary = report.get("summary")
    if not isinstance(summary, dict):
        errors.append(f"{path}: missing summary metadata")
    else:
        if not isinstance(summary.get("ok"), bool):
            errors.append(f"{path}: summary.ok is missing")
        if not isinstance(summary.get("proof_mode"), str) or not summary.get("proof_mode"):
            errors.append(f"{path}: summary.proof_mode is missing")
        if not isinstance(summary.get("proof_stage"), str) or not summary.get("proof_stage"):
            errors.append(f"{path}: summary.proof_stage is missing")
    return errors


def verify_check_report(
    path: Path,
    report: dict,
    required_names: tuple[str, ...] = VERIFY_CHECK_NAMES,
    expected_mode: str = "",
) -> list[str]:
    errors = verify_common_report(path, report, "check", expected_mode)
    checks_raw = report.get("checks")
    if not isinstance(checks_raw, list):
        return errors + [f"{path}: checks is not a list"]
    checks = {check.get("name"): check for check in checks_raw if isinstance(check, dict)}
    for name in required_names:
        check = checks.get(name)
        if check is None:
            errors.append(f"{path}: missing check {name!r}")
            continue
        if check.get("ok") is not True:
            errors.append(f"{path}: check {name!r} is not OK: {check.get('detail', '')}")
        if name == "cpu":
            detail = check.get("detail")
            if not isinstance(detail, str) or "GOOS=" not in detail or "GOARCH=" not in detail:
                errors.append(f"{path}: check 'cpu' is missing GOOS/GOARCH detail")
    return errors


def verify_run_report(
    path: Path,
    report: dict,
    required_labels: tuple[str, ...] = VERIFY_RUN_LABELS,
    expected_mode: str = "",
    require_full_cpu: bool = False,
) -> list[str]:
    errors = verify_common_report(path, report, "run", expected_mode)
    proof = report.get("proof")
    if isinstance(proof, dict):
        cpu_scope = proof.get("cpu_scope")
        if not isinstance(cpu_scope, str) or not cpu_scope:
            errors.append(f"{path}: proof.cpu_scope is missing")
        elif cpu_scope not in {"smoke", "full", "custom", "none"}:
            errors.append(f"{path}: proof.cpu_scope is unexpected: {cpu_scope!r}")
        elif require_full_cpu and cpu_scope != "full":
            errors.append(f"{path}: proof.cpu_scope is {cpu_scope!r}, want 'full'")
    if report.get("dry_run") is True:
        errors.append(f"{path}: run report is from --dry-run")
    commands_raw = report.get("commands")
    if not isinstance(commands_raw, list):
        return errors + [f"{path}: commands is not a list"]
    commands = {cmd.get("label"): cmd for cmd in commands_raw if isinstance(cmd, dict)}
    for label in required_labels:
        cmd = commands.get(label)
        if cmd is None:
            errors.append(f"{path}: missing command label {label!r}")
            continue
        if cmd.get("skipped") is True:
            errors.append(f"{path}: command {label!r} was skipped")
        if cmd.get("exit_code") != 0:
            errors.append(f"{path}: command {label!r} exit_code is {cmd.get('exit_code')!r}, want 0")
    return errors


def verify_check_names(args: argparse.Namespace) -> tuple[str, ...]:
    return VERIFY_CPU_CHECK_NAMES if args.cpu_only else VERIFY_CHECK_NAMES


def verify_run_labels(args: argparse.Namespace) -> tuple[str, ...]:
    return VERIFY_CPU_RUN_LABELS if args.cpu_only else VERIFY_RUN_LABELS


def run_verify(root: Path, args: argparse.Namespace) -> int:
    check_path = resolve_report_path(root, args.check_report)
    run_path = resolve_report_path(root, args.run_report)
    expected_mode = expected_proof_mode(args)

    errors: list[str] = []
    check_report, check_errors = read_json_report(check_path)
    errors.extend(check_errors)
    if check_report is not None:
        errors.extend(verify_check_report(check_path, check_report, verify_check_names(args), expected_mode))
        errors.extend(current_repo_errors(check_path, check_report, root, args.allow_stale_repo))

    run_report, run_errors = read_json_report(run_path)
    errors.extend(run_errors)
    if run_report is not None:
        errors.extend(verify_run_report(run_path, run_report, verify_run_labels(args), expected_mode, args.full_cpu))
        errors.extend(current_repo_errors(run_path, run_report, root, args.allow_stale_repo))

    if errors:
        for error in errors:
            print(f"[verify] FAIL {error}", file=sys.stderr)
        return 1
    print(f"[verify] OK check={check_path}", flush=True)
    print(f"[verify] OK run={run_path}", flush=True)
    return 0


def status_scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None or value == "":
        return "?"
    return str(value)


def status_list(value: object) -> str:
    if not isinstance(value, list):
        return status_scalar(value)
    if not value:
        return "-"
    return ",".join(str(item) for item in value)


def short_repo_head(repo: object) -> str:
    if not isinstance(repo, dict):
        return "unknown"
    head = repo.get("head")
    if not isinstance(head, str) or not head:
        return "unknown"
    return head[:12]


def command_status(command: dict) -> str:
    label = status_scalar(command.get("label"))
    if command.get("skipped") is True:
        return f"{label}:SKIP"
    exit_code = command.get("exit_code")
    if exit_code == 0:
        return f"{label}:OK"
    return f"{label}:FAIL({status_scalar(exit_code)})"


def check_status(check: dict) -> str:
    name = status_scalar(check.get("name"))
    return f"{name}:{'OK' if check.get('ok') is True else 'MISSING'}"


def status_report_errors(path: Path, report: dict, expected_kind: str, expected_mode: str, root: Path, allow_stale_repo: bool) -> list[str]:
    errors: list[str] = []
    if report.get("schema") != "fak.laptop-test.v1":
        errors.append(f"{path}: unexpected schema {report.get('schema')!r}")
    if report.get("kind") != expected_kind:
        errors.append(f"{path}: expected kind {expected_kind!r}, got {report.get('kind')!r}")

    proof = report.get("proof")
    if not isinstance(proof, dict):
        errors.append(f"{path}: missing proof metadata")
    else:
        mode = proof.get("mode")
        if mode != expected_mode:
            errors.append(f"{path}: proof.mode is {mode!r}, want {expected_mode!r}")

    summary = report.get("summary")
    if not isinstance(summary, dict):
        errors.append(f"{path}: missing summary metadata")
    elif summary.get("ok") is not True:
        errors.append(f"{path}: summary.ok is {summary.get('ok')!r}")

    if report.get("exit_code") != 0:
        errors.append(f"{path}: report exit_code is {report.get('exit_code')!r}, want 0")
    if expected_kind == "run" and report.get("dry_run") is True:
        errors.append(f"{path}: run report is from --dry-run")
    errors.extend(current_repo_errors(path, report, root, allow_stale_repo))
    return errors


def print_status_report(label: str, path: Path, report: dict, errors: list[str]) -> None:
    proof = report.get("proof") if isinstance(report.get("proof"), dict) else {}
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    repo = report.get("repo") if isinstance(report.get("repo"), dict) else {}
    host = report.get("host") if isinstance(report.get("host"), dict) else {}
    wsl = report.get("wsl") if isinstance(report.get("wsl"), dict) else {}

    state = "OK" if not errors else "FAIL"
    proof_text = f"{status_scalar(proof.get('mode'))}/{status_scalar(proof.get('stage'))}"
    print(
        f"[status:{label}] {state} path={path} proof={proof_text} "
        f"summary_ok={status_scalar(summary.get('ok'))} exit_code={status_scalar(report.get('exit_code'))} "
        f"generated={status_scalar(report.get('generated_utc'))}",
        flush=True,
    )
    print(
        f"[status:{label}] repo=head={short_repo_head(repo)} branch={status_scalar(repo.get('branch'))} "
        f"dirty={status_scalar(repo.get('dirty'))} status_count={status_scalar(repo.get('status_count'))} "
        f"host={status_scalar(host.get('system'))}/{status_scalar(host.get('machine'))} "
        f"wsl={status_scalar(wsl.get('source'))}/{status_scalar(wsl.get('distro'))}",
        flush=True,
    )
    sample = repo.get("status_sample")
    if isinstance(sample, list) and sample:
        suffix = " ..." if repo.get("status_truncated") is True else ""
        print(f"[status:{label}] dirty_sample={'; '.join(str(item) for item in sample)}{suffix}", flush=True)

    if label == "check":
        checks = [check_status(check) for check in report.get("checks", []) if isinstance(check, dict)]
        print(
            f"[status:check] checks={','.join(checks) if checks else '-'} "
            f"required={status_list(report.get('required'))} "
            f"failed_required={status_list(summary.get('failed_required'))}",
            flush=True,
        )
    elif label == "run":
        commands = [command_status(command) for command in report.get("commands", []) if isinstance(command, dict)]
        cpu_scope = proof.get("cpu_scope", summary.get("cpu_scope"))
        print(
            f"[status:run] commands={','.join(commands) if commands else '-'} "
            f"cpu_scope={status_scalar(cpu_scope)} dry_run={status_scalar(report.get('dry_run'))} "
            f"failed_commands={status_list(summary.get('failed_commands'))} "
            f"skipped_commands={status_list(summary.get('skipped_commands'))}",
            flush=True,
        )


def run_status(root: Path, args: argparse.Namespace) -> int:
    expected_mode = expected_proof_mode(args)
    targets = (
        ("check", resolve_report_path(root, args.check_report)),
        ("run", resolve_report_path(root, args.run_report)),
    )

    print(
        f"[status] mode={expected_mode} check={targets[0][1]} run={targets[1][1]}",
        flush=True,
    )
    all_errors: list[str] = []
    for label, path in targets:
        report, read_errors = read_json_report(path)
        if read_errors:
            all_errors.extend(read_errors)
            print(f"[status:{label}] FAIL path={path}", flush=True)
            continue
        assert report is not None
        errors = status_report_errors(path, report, label, expected_mode, root, args.allow_stale_repo)
        all_errors.extend(errors)
        print_status_report(label, path, report, errors)

    if all_errors:
        for error in all_errors:
            print(f"[status] FAIL {error}", file=sys.stderr)
        return 1
    return 0


def with_args(args: argparse.Namespace, **updates: object) -> argparse.Namespace:
    values = vars(args).copy()
    values.update(updates)
    return argparse.Namespace(**values)


def accept_fail(phase: str, rc: int) -> int:
    print(f"[accept] FAIL phase={phase} exit_code={rc}", file=sys.stderr, flush=True)
    return rc


def print_accept_success(root: Path, args: argparse.Namespace) -> None:
    print("[accept] OK laptop hardware proof verified", flush=True)
    print(f"[accept] report precheck={resolve_report_path(root, args.precheck_report)}", flush=True)
    print(f"[accept] report run={resolve_report_path(root, args.run_report)}", flush=True)
    print(f"[accept] report postcheck={resolve_report_path(root, args.check_report)}", flush=True)


def print_accept_cpu_success(root: Path, args: argparse.Namespace) -> None:
    print("[accept] OK laptop CPU/Intel proof verified", flush=True)
    print(f"[accept] report check={resolve_report_path(root, args.check_report)}", flush=True)
    print(f"[accept] report run={resolve_report_path(root, args.run_report)}", flush=True)


def run_accept_cpu(root: Path, args: argparse.Namespace, platform: str | None = None) -> int:
    cpu_scope = "full" if args.full_cpu else "smoke"
    print("[accept] preflight: CPU/Intel target", flush=True)
    check_args = with_args(
        args,
        lane="check",
        require_nvidia=False,
        require_cuda_toolchain=False,
        proof_mode="cpu-only",
        proof_stage="cpu-check",
        cpu_scope=cpu_scope,
        out=args.check_report,
    )
    rc = run_checks(root, check_args, platform)
    if rc != 0:
        return accept_fail("cpu-preflight", rc)

    cpu_scope_label = "full CPU suite" if args.full_cpu else "CPU smoke"
    print(f"[accept] run: {cpu_scope_label}", flush=True)
    run_args = with_args(
        args,
        lane="cpu",
        smoke=not args.full_cpu,
        setup=False,
        proof_mode="cpu-only",
        proof_stage="cpu-run",
        cpu_scope=cpu_scope,
        out=args.run_report,
    )
    plan = build_plan(root, run_args, platform)
    rc = run_plan(plan, dry_run=False, platform=platform, args=run_args)
    if rc != 0:
        return accept_fail("cpu-run", rc)

    print("[accept] verify CPU reports", flush=True)
    rc = run_verify(root, args)
    if rc != 0:
        return accept_fail("cpu-verify", rc)
    print_accept_cpu_success(root, args)
    return 0


def run_accept(root: Path, args: argparse.Namespace, platform: str | None = None) -> int:
    if args.dry_run:
        raise SystemExit("accept lane writes hardware proof reports and does not support --dry-run")
    if args.cpu_only:
        return run_accept_cpu(root, args, platform)

    cpu_scope = "full" if args.full_cpu else "smoke"
    print("[accept] preflight: CPU and NVIDIA passthrough", flush=True)
    precheck_args = with_args(
        args,
        lane="check",
        require_nvidia=True,
        require_cuda_toolchain=False,
        proof_mode="cpu+nvidia",
        proof_stage="precheck",
        cpu_scope=cpu_scope,
        out=args.precheck_report,
    )
    rc = run_checks(root, precheck_args, platform)
    if rc != 0:
        return accept_fail("preflight", rc)

    cpu_scope_label = "full CPU suite" if args.full_cpu else "CPU smoke"
    print(f"[accept] run: {cpu_scope_label}, CUDA setup, CUDA tests", flush=True)
    run_args = with_args(
        args,
        lane="all",
        smoke=not args.full_cpu,
        setup=True,
        nvidia_action="test",
        proof_mode="cpu+nvidia",
        proof_stage="run",
        cpu_scope=cpu_scope,
        out=args.run_report,
    )
    plan = build_plan(root, run_args, platform)
    rc = run_plan(plan, dry_run=False, platform=platform, args=run_args)
    if rc != 0:
        return accept_fail("run", rc)

    print("[accept] post-setup: CPU, NVIDIA passthrough, CUDA toolchain", flush=True)
    postcheck_args = with_args(
        args,
        lane="check",
        require_nvidia=True,
        require_cuda_toolchain=True,
        proof_mode="cpu+nvidia",
        proof_stage="postcheck",
        cpu_scope=cpu_scope,
        out=args.check_report,
    )
    rc = run_checks(root, postcheck_args, platform)
    if rc != 0:
        return accept_fail("post-setup", rc)

    print("[accept] verify reports", flush=True)
    rc = run_verify(root, args)
    if rc != 0:
        return accept_fail("verify", rc)
    print_accept_success(root, args)
    return 0


def wsl_check(args: argparse.Namespace, script: str, platform: str | None = None) -> tuple[str, ...]:
    return (*wsl_prefix(args, platform), "bash", "-lc", script)


def check_cpu(root: Path, args: argparse.Namespace, platform: str | None = None) -> CheckResult:
    env = host_env(os.environ, args)
    if is_windows(platform):
        ps1 = root / "fak" / "test.ps1"
        if not ps1.exists():
            return CheckResult("cpu", False, f"missing {ps1}")
        ok, detail = run_probe(
            wsl_check(args, go_env_probe_script(), platform),
            root,
            env,
        )
        if ok:
            return CheckResult("cpu", True, "WSL Go available: " + one_line_detail(detail))
        return CheckResult("cpu", False, "WSL Go unavailable: " + detail)

    sh = root / "fak" / "test.sh"
    if not sh.exists():
        return CheckResult("cpu", False, f"missing {sh}")
    ok, detail = run_probe(("bash", "-lc", go_env_probe_script()), root / "fak", env)
    if ok:
        return CheckResult("cpu", True, one_line_detail(detail))
    return CheckResult("cpu", False, "go unavailable: " + detail)


def check_nvidia(root: Path, args: argparse.Namespace, platform: str | None = None) -> CheckResult:
    host_platform = platform or sys.platform
    if host_platform == "darwin":
        return CheckResult("nvidia", False, "CUDA/NVIDIA lane requires Windows+WSL or Linux; current host is darwin")

    env = host_env(os.environ, args)
    if is_windows(host_platform):
        script = (
            "set -e; "
            "command -v nvidia-smi >/dev/null; nvidia-smi -L; "
            "test -r /usr/lib/wsl/lib/libcuda.so"
        )
        ok, detail = run_probe(wsl_check(args, script, host_platform), root, env)
        if ok:
            return CheckResult("nvidia", True, detail)
        return CheckResult("nvidia", False, "WSL NVIDIA passthrough check failed: " + detail)

    script = (
        "set -e; "
        "command -v nvidia-smi >/dev/null; nvidia-smi -L"
    )
    ok, detail = run_probe(("bash", "-lc", script), root / "fak", env)
    if ok:
        return CheckResult("nvidia", True, detail)
    return CheckResult("nvidia", False, "NVIDIA check failed: " + detail)


def check_cuda_toolchain(root: Path, args: argparse.Namespace, platform: str | None = None) -> CheckResult:
    host_platform = platform or sys.platform
    if host_platform == "darwin":
        return CheckResult("cuda-toolchain", False, "CUDA toolkit lane requires Windows+WSL or Linux; current host is darwin")

    env = host_env(os.environ, args)
    script = (
        "set -e; "
        "test -x ${CUDA_HOME:-$HOME/cudaenv}/bin/nvcc; "
        "${CUDA_HOME:-$HOME/cudaenv}/bin/nvcc --version | tail -2"
    )
    if is_windows(host_platform):
        ok, detail = run_probe(wsl_check(args, script, host_platform), root, env)
    else:
        ok, detail = run_probe(("bash", "-lc", script), root / "fak", env)
    if ok:
        return CheckResult("cuda-toolchain", True, detail)
    return CheckResult("cuda-toolchain", False, "CUDA toolkit unavailable; run the nvidia lane with --setup: " + detail)


def check_required_names(args: argparse.Namespace) -> set[str]:
    required = {"cpu"}
    if args.require_nvidia:
        required.add("nvidia")
    if args.require_cuda_toolchain:
        required.add("cuda-toolchain")
    return required


def check_report(checks: list[CheckResult], args: argparse.Namespace, rc: int, platform: str | None = None, root: Path | None = None) -> dict:
    required = check_required_names(args)
    return {
        "schema": "fak.laptop-test.v1",
        "kind": "check",
        "generated_utc": utc_now(),
        "platform": platform or sys.platform,
        "host": host_report(root),
        "repo": repo_report(root),
        "wsl": wsl_report(args, platform),
        "proof": proof_report(args, "check"),
        "summary": check_summary(checks, args, rc),
        "exit_code": rc,
        "required": sorted(required),
        "checks": [
            {"name": check.name, "ok": check.ok, "required": check.name in required, "detail": check.detail}
            for check in checks
        ],
    }


def run_checks(root: Path, args: argparse.Namespace, platform: str | None = None) -> int:
    checks = [check_cpu(root, args, platform), check_nvidia(root, args, platform), check_cuda_toolchain(root, args, platform)]
    for check in checks:
        status = "OK" if check.ok else "MISSING"
        print(f"[check:{check.name}] {status} {check.detail}", flush=True)
    required = check_required_names(args)
    failed = [check.name for check in checks if check.name in required and not check.ok]
    rc = 0
    if failed:
        print("required checks failed: " + ", ".join(failed), file=sys.stderr)
        rc = 1
    write_report(args.out, check_report(checks, args, rc, platform, root), root)
    return rc


def nvidia_commands(root: Path, args: argparse.Namespace, platform: str | None = None) -> list[Command]:
    if args.nvidia_action not in NVIDIA_ACTIONS:
        raise ValueError(f"unsupported NVIDIA action: {args.nvidia_action}")
    host_platform = platform or sys.platform
    if host_platform == "darwin" and not args.dry_run:
        raise SystemExit("nvidia lane requires Windows+WSL or Linux with CUDA; this host is darwin")

    env = host_env(os.environ, args)
    commands: list[Command] = []
    actions: list[tuple[str, tuple[str, ...]]] = []
    if args.setup:
        actions.append(("nvidia-setup", ("bash", "internal/compute/setup_cuda_wsl.sh")))
    actions.append(("nvidia", ("bash", "internal/compute/build_cuda.sh", args.nvidia_action)))
    if args.nvidia_action == "bench":
        if args.model_dir:
            label, argv = actions[-1]
            actions[-1] = (label, (*argv, args.model_dir, str(args.decode_steps)))
        elif args.decode_steps != 128:
            label, argv = actions[-1]
            actions[-1] = (label, (*argv, "internal/model/.cache/smollm2-135m", str(args.decode_steps)))

    if is_windows(host_platform):
        try:
            fak_dir = windows_drive_to_wsl_path(root / "fak")
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        prefix = wsl_prefix(args, host_platform)
        for label, argv in actions:
            script = "set -euo pipefail; cd " + shlex.quote(fak_dir) + "; " + " ".join(shlex.quote(a) for a in argv)
            commands.append(Command(label, (*prefix, "bash", "-lc", script), root, env))
        return commands

    fak_dir = root / "fak"
    for label, argv in actions:
        commands.append(Command(label, argv, fak_dir, env))
    return commands


def build_plan(root: Path, args: argparse.Namespace, platform: str | None = None) -> list[Command]:
    lane = args.lane
    plan: list[Command] = []
    if lane in {"cpu", "all"}:
        plan.append(cpu_command(root, args, platform))
    if lane in {"nvidia", "all"}:
        plan.extend(nvidia_commands(root, args, platform))
    return plan


def format_command(cmd: Command, platform: str | None = None) -> str:
    if is_windows(platform):
        return subprocess.list2cmdline(list(cmd.argv))
    return " ".join(shlex.quote(part) for part in cmd.argv)


def command_report_base(cmd: Command, platform: str | None = None) -> dict:
    return {
        "label": cmd.label,
        "cwd": str(cmd.cwd),
        "argv": list(cmd.argv),
        "command": format_command(cmd, platform),
    }


def merged(base: dict, extra: dict) -> dict:
    out = dict(base)
    out.update(extra)
    return out


def plan_report(plan: list[Command], args: argparse.Namespace, rc: int, results: list[dict], platform: str | None = None) -> dict:
    root = repo_root()
    return {
        "schema": "fak.laptop-test.v1",
        "kind": "run",
        "generated_utc": utc_now(),
        "platform": platform or sys.platform,
        "host": host_report(root),
        "repo": repo_report(root),
        "wsl": wsl_report(args, platform),
        "proof": proof_report(args, "run"),
        "summary": run_summary(plan, args, rc, results),
        "lane": args.lane,
        "dry_run": bool(args.dry_run),
        "exit_code": rc,
        "commands": results if results else [merged(command_report_base(cmd, platform), {"skipped": bool(args.dry_run)}) for cmd in plan],
    }


def run_command_streamed(cmd: Command) -> tuple[int, str, str]:
    out_tail = ""
    try:
        with subprocess.Popen(
            list(cmd.argv),
            cwd=str(cmd.cwd),
            env=cmd.env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        ) as proc:
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="", flush=True)
                out_tail = tail_text(out_tail + line)
            return proc.wait(), out_tail, ""
    except OSError as exc:
        msg = str(exc)
        print(msg, file=sys.stderr, flush=True)
        return 127, "", msg


def run_plan(plan: list[Command], dry_run: bool, platform: str | None = None, args: argparse.Namespace | None = None) -> int:
    results: list[dict] = []
    rc = 0
    for cmd in plan:
        print(f"[{cmd.label}] cwd={cmd.cwd}", flush=True)
        print(format_command(cmd, platform), flush=True)
        if dry_run:
            results.append(merged(command_report_base(cmd, platform), {"skipped": True, "exit_code": None}))
            continue
        start = time.monotonic()
        if args is not None and args.out:
            returncode, stdout_tail, stderr_tail = run_command_streamed(cmd)
        else:
            cp = subprocess.run(list(cmd.argv), cwd=str(cmd.cwd), env=cmd.env, check=False)
            returncode, stdout_tail, stderr_tail = cp.returncode, "", ""
        elapsed_ms = int((time.monotonic() - start) * 1000)
        results.append(merged(command_report_base(cmd, platform), {
            "skipped": False,
            "exit_code": returncode,
            "duration_ms": elapsed_ms,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
        }))
        if returncode != 0:
            rc = returncode
            break
    if args is not None:
        write_report(args.out, plan_report(plan, args, rc, results, platform))
    return rc


def parse_args(argv: list[str]) -> argparse.Namespace:
    forwarded: list[str] = []
    if "--" in argv:
        split = argv.index("--")
        forwarded = argv[split + 1 :]
        argv = argv[:split]

    ap = argparse.ArgumentParser(description="Run fak laptop CPU/Intel and NVIDIA/CUDA test lanes")
    ap.add_argument("lane", nargs="?", choices=("cpu", "nvidia", "all", "check", "verify", "status", "accept"), default="cpu")
    ap.add_argument("--smoke", action="store_true", help="CPU lane: run focused compute/model smoke tests instead of ./...")
    ap.add_argument("--fast", action="store_true", help="CPU lane: forward FAK_FAST=1 to fak/test.sh via WSL")
    ap.add_argument("--full-cpu", action="store_true", help="accept lane: run the full CPU ./... suite instead of the focused smoke tests")
    ap.add_argument("--cpu-only", action="store_true", help="accept/verify/status lane: prove only the CPU/Intel lane, without requiring NVIDIA")
    ap.add_argument("--require-nvidia", action="store_true", help="check lane: fail if NVIDIA/WSL GPU passthrough is unavailable")
    ap.add_argument("--require-cuda-toolchain", action="store_true", help="check lane: fail if the CUDA nvcc toolchain is unavailable")
    ap.add_argument("--setup", action="store_true", help="NVIDIA lane: run the no-sudo CUDA setup before build/test")
    ap.add_argument("--nvidia-action", choices=NVIDIA_ACTIONS, default="test", help="NVIDIA lane action for build_cuda.sh")
    ap.add_argument("--model-dir", default="", help="NVIDIA bench model dir (only with --nvidia-action bench)")
    ap.add_argument("--decode-steps", type=int, default=128, help="NVIDIA bench decode steps")
    ap.add_argument("--wsl-distro", default="", help="WSL distro override; also sets FAK_WSL_DISTRO")
    ap.add_argument("--dry-run", action="store_true", help="print commands without executing them")
    ap.add_argument("--out", default="", help="write a JSON report for the check/run")
    ap.add_argument("--precheck-report", default=None, help="accept lane: initial NVIDIA passthrough report path")
    ap.add_argument("--check-report", default=None, help="verify/status lane: check report path")
    ap.add_argument("--run-report", default=None, help="verify/status lane: run report path")
    ap.add_argument("--allow-stale-repo", action="store_true", help="verify/status lane: allow reports from a different git HEAD")
    args = ap.parse_args(argv)
    if args.precheck_report is None:
        args.precheck_report = DEFAULT_PRECHECK_REPORT
    if args.check_report is None:
        args.check_report = DEFAULT_CPU_CHECK_REPORT if args.cpu_only else DEFAULT_VERIFY_CHECK_REPORT
    if args.run_report is None:
        args.run_report = DEFAULT_CPU_RUN_REPORT if args.cpu_only else DEFAULT_VERIFY_RUN_REPORT
    args.go_test_args = tuple(forwarded)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(argv if argv is not None else sys.argv[1:]))
    if args.lane == "check":
        return run_checks(repo_root(), args)
    if args.lane == "verify":
        return run_verify(repo_root(), args)
    if args.lane == "status":
        return run_status(repo_root(), args)
    if args.lane == "accept":
        return run_accept(repo_root(), args)
    plan = build_plan(repo_root(), args)
    return run_plan(plan, args.dry_run, args=args)


if __name__ == "__main__":
    raise SystemExit(main())
