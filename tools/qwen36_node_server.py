#!/usr/bin/env python3
"""Start a Qwen3.6 GGUF llama-server on a local GPU/CPU test bed.

This helper is meant to be run on the node itself. It keeps the serving setup
small and reproducible: pick conservative Mac, NVIDIA, Vulkan, or CPU defaults,
bind to the node's Tailscale IP when available, and expose an OpenAI-compatible
/v1 API that tools/qwen36_surface_smoke.py can exercise from the driver.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


DEFAULT_MODEL = "lmstudio-community/Qwen3.6-27B-GGUF:Q4_K_M"
DEFAULT_PORT = 8131
SAFE_LOCALHOSTS = {"127.0.0.1", "::1", "localhost"}
PUBLIC_BINDS = {"0.0.0.0", "::", ""}


@dataclass(frozen=True)
class Profile:
    name: str
    ctx_size: int
    n_gpu_layers: int
    extra_args: tuple[str, ...] = ()


PROFILES = {
    "mac": Profile("mac", ctx_size=32768, n_gpu_layers=99),
    "nvidia": Profile("nvidia", ctx_size=8192, n_gpu_layers=20),
    "vulkan": Profile("vulkan", ctx_size=8192, n_gpu_layers=20, extra_args=("--fit", "on")),
    "cpu": Profile("cpu", ctx_size=4096, n_gpu_layers=0),
}


def default_cpu_threads(cpu_count: int | None = None) -> int:
    """Bounded CPU thread count that leaves headroom on a shared host.

    llama.cpp's unset --threads default grabs ALL logical cores. On a box that is
    also running the agent fleet (many resident workers), an all-cores resident
    CPU-profile server over-subscribes the machine and contributes to host CPU
    over-pin. For the cpu profile we therefore emit an EXPLICIT, bounded thread
    count that leaves two cores free. GPU profiles offload decode, so they keep
    llama.cpp's own default (no --threads emitted) — pass --threads to override.
    """
    n = cpu_count if cpu_count is not None else (os.cpu_count() or 4)
    return max(1, n - 2)


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def nvidia_smi_report(timeout_s: float = 5.0) -> dict[str, Any]:
    exe = shutil.which("nvidia-smi")
    if not exe and os.name == "nt":
        candidate = r"C:\Windows\System32\nvidia-smi.exe"
        if os.path.exists(candidate):
            exe = candidate
    if not exe:
        return {"found": False, "ok": False, "detail": "nvidia-smi was not found on PATH"}
    cmd = [
        exe,
        "--query-gpu=name,driver_version,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"found": True, "ok": False, "command": cmd, "detail": str(exc)}
    gpus: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 4:
            gpus.append({
                "name": parts[0],
                "driver_version": parts[1],
                "memory_total_mib": parts[2],
                "memory_free_mib": parts[3],
            })
    detail = proc.stderr.strip() or proc.stdout.strip()
    return {
        "found": True,
        "ok": proc.returncode == 0 and bool(gpus),
        "command": cmd,
        "exit_code": proc.returncode,
        "gpus": gpus,
        "detail": detail[:1000],
    }


def windows_video_controller_names(timeout_s: float = 5.0) -> list[str]:
    if platform.system() != "Windows":
        return []
    try:
        proc = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name }",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def has_windows_amd_gpu() -> bool:
    return any(
        "amd" in name.lower() or "radeon" in name.lower()
        for name in windows_video_controller_names()
    )


def detect_profile(system: str | None = None, has_nvidia: bool | None = None, has_amd_gpu: bool | None = None) -> str:
    system = system or platform.system()
    if has_nvidia is None:
        has_nvidia = command_exists("nvidia-smi")
    if has_amd_gpu is None:
        has_amd_gpu = system == "Windows" and has_windows_amd_gpu()
    if system == "Darwin":
        return "mac"
    if has_nvidia:
        return "nvidia"
    if has_amd_gpu:
        return "vulkan"
    return "cpu"


def find_llama_server(explicit: str = "") -> str:
    if explicit:
        looks_like_path = os.path.isabs(explicit) or os.path.sep in explicit
        if os.path.altsep:
            looks_like_path = looks_like_path or os.path.altsep in explicit
        if looks_like_path:
            if os.path.exists(explicit):
                return explicit
            raise FileNotFoundError(f"llama-server path does not exist: {explicit}")
        path = shutil.which(explicit)
        if path:
            return path
        raise FileNotFoundError(f"llama-server executable was not found on PATH: {explicit}")
    for name in ("llama-server", "llama-server.exe"):
        path = shutil.which(name)
        if path:
            return path
    raise FileNotFoundError("llama-server was not found on PATH; install llama.cpp first")


def tailscale_ip(timeout_s: float = 5.0) -> str:
    exe = shutil.which("tailscale")
    if not exe and os.name == "nt":
        candidate = r"C:\Program Files\Tailscale\tailscale.exe"
        if os.path.exists(candidate):
            exe = candidate
    if not exe:
        return ""
    try:
        proc = subprocess.run(
            [exe, "ip", "-4"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if proc.returncode != 0:
        return ""
    return next((line.strip() for line in proc.stdout.splitlines() if line.strip()), "")


def select_bind_host(bind: str, explicit_host: str = "", tailnet_ip: str = "") -> str:
    if explicit_host:
        return explicit_host
    if bind == "localhost":
        return "127.0.0.1"
    if bind == "tailnet":
        if not tailnet_ip:
            raise RuntimeError("tailscale ip -4 did not return an address; use --bind localhost or --host")
        return tailnet_ip
    if bind == "auto":
        return tailnet_ip or "127.0.0.1"
    raise ValueError(f"unknown bind mode: {bind}")


def validate_bind_host(host: str, allow_public_bind: bool) -> None:
    if host in PUBLIC_BINDS and not allow_public_bind:
        raise ValueError("refusing public bind; use a Tailscale IP, localhost, or --allow-public-bind")


def base_url(host: str, port: int) -> str:
    display_host = f"[{host}]" if ":" in host and not host.startswith("[") else host
    return f"http://{display_host}:{port}/v1"


def split_extra_args(values: list[str]) -> list[str]:
    args: list[str] = []
    for value in values:
        args.extend(shlex.split(value, posix=os.name != "nt"))
    return args


def build_command(
    llama_server: str,
    model: str,
    host: str,
    port: int,
    profile: Profile,
    ctx_size: int,
    n_gpu_layers: int,
    threads: int,
    extra_args: list[str],
) -> list[str]:
    cmd = [
        llama_server,
        "-hf",
        model,
        "--host",
        host,
        "--port",
        str(port),
        "--ctx-size",
        str(ctx_size or profile.ctx_size),
        "--n-gpu-layers",
        str(n_gpu_layers if n_gpu_layers >= 0 else profile.n_gpu_layers),
    ]
    # Always emit an explicit, bounded thread count for the CPU profile so a
    # resident CPU server never silently runs on all cores and over-pins a
    # shared host; an explicit --threads still wins. GPU profiles keep llama's
    # own default (decode is offloaded).
    effective_threads = threads
    if effective_threads <= 0 and profile.name == "cpu":
        effective_threads = default_cpu_threads()
    if effective_threads > 0:
        cmd.extend(["--threads", str(effective_threads)])
    cmd.extend(profile.extra_args)
    cmd.extend(extra_args)
    return cmd


def format_command(cmd: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(cmd)
    try:
        import shlex

        return shlex.join(cmd)
    except AttributeError:
        return " ".join(cmd)


def json_get(url: str, timeout_s: float) -> tuple[int, dict[str, Any] | None, str]:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = None
            return int(resp.status), parsed if isinstance(parsed, dict) else None, raw[:1000]
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        return int(exc.code), None, raw[:1000]
    except OSError as exc:
        return 0, None, str(exc)


def wait_for_models(url: str, deadline_s: float, interval_s: float = 1.0) -> dict[str, Any]:
    deadline = time.time() + deadline_s
    last: dict[str, Any] = {"http_status": 0, "detail": "not probed"}
    while time.time() < deadline:
        status, data, body = json_get(url.rstrip("/") + "/models", timeout_s=2.0)
        model_ids = [m.get("id") for m in (data or {}).get("data", []) if isinstance(m, dict)]
        last = {"http_status": status, "models": model_ids, "body_excerpt": body}
        if status == 200 and model_ids:
            return {"ready": True, **last}
        time.sleep(interval_s)
    return {"ready": False, **last}


def stop_process(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def build_runtime_plan(args: argparse.Namespace, require_llama: bool) -> dict[str, Any]:
    failures: list[str] = []
    checks: list[dict[str, Any]] = []

    profile_name = detect_profile() if args.profile == "auto" else args.profile
    profile = PROFILES[profile_name]
    checks.append({"name": "profile", "ok": True, "value": profile.name})

    nvidia_required = profile.name == "nvidia"
    nvidia = nvidia_smi_report() if nvidia_required else {"found": False, "ok": True, "detail": "not required for profile"}
    if nvidia_required and not nvidia.get("ok"):
        failures.append(str(nvidia.get("detail") or "nvidia-smi did not report a usable NVIDIA GPU"))
    checks.append({
        "name": "nvidia_smi",
        "ok": bool(nvidia.get("ok")),
        "required": nvidia_required,
        "found": bool(nvidia.get("found")),
        "gpus": nvidia.get("gpus", []),
        "detail": nvidia.get("detail", ""),
    })

    tail_ip = tailscale_ip()
    tailnet_required = args.bind == "tailnet" and not args.host
    checks.append({
        "name": "tailscale_ip",
        "ok": bool(tail_ip) or not tailnet_required,
        "required": tailnet_required,
        "value": tail_ip,
    })

    host = ""
    bind_error = ""
    try:
        host = select_bind_host(args.bind, explicit_host=args.host, tailnet_ip=tail_ip)
    except (RuntimeError, ValueError) as exc:
        bind_error = str(exc)
        failures.append(bind_error)
    checks.append({
        "name": "bind_host",
        "ok": bool(host) and not bind_error,
        "required": True,
        "value": host,
        "detail": bind_error,
    })

    public_bind_error = ""
    if host:
        try:
            validate_bind_host(host, args.allow_public_bind)
        except ValueError as exc:
            public_bind_error = str(exc)
            failures.append(public_bind_error)
    checks.append({
        "name": "safe_bind",
        "ok": not public_bind_error,
        "required": True,
        "value": host,
        "detail": public_bind_error,
    })

    llama = args.llama_server or "llama-server"
    llama_found = False
    llama_error = ""
    try:
        llama = find_llama_server(args.llama_server)
        llama_found = True
    except FileNotFoundError as exc:
        llama_error = str(exc)
        if require_llama:
            failures.append(llama_error)
    checks.append({
        "name": "llama_server",
        "ok": llama_found or not require_llama,
        "required": require_llama,
        "value": llama,
        "found": llama_found,
        "detail": llama_error,
    })

    url = base_url(host, args.port) if host else ""
    cmd = build_command(
        llama_server=llama,
        model=args.model,
        host=host or "<bind-host>",
        port=args.port,
        profile=profile,
        ctx_size=args.ctx_size,
        n_gpu_layers=args.n_gpu_layers,
        threads=args.threads,
        extra_args=split_extra_args(args.extra_arg),
    )
    return {
        "ok": not failures,
        "failures": failures,
        "profile": profile.name,
        "base_url": url,
        "tailscale_ip": tail_ip,
        "bind_host": host,
        "llama_server": llama,
        "llama_server_found": llama_found,
        "checks": checks,
        "command": cmd,
        "command_line": format_command(cmd),
    }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Start llama-server for Qwen3.6 node surface testing")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--profile", choices=["auto", "mac", "nvidia", "vulkan", "cpu"], default="auto")
    ap.add_argument("--bind", choices=["auto", "tailnet", "localhost"], default="auto")
    ap.add_argument("--host", default="", help="explicit host/IP to bind; overrides --bind")
    ap.add_argument("--allow-public-bind", action="store_true")
    ap.add_argument("--llama-server", default="", help="path to llama-server; defaults to PATH lookup")
    ap.add_argument("--ctx-size", type=int, default=0)
    ap.add_argument("--n-gpu-layers", type=int, default=-1)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--extra-arg", action="append", default=[], help="additional llama-server argument(s)")
    ap.add_argument("--dry-run", action="store_true", help="print the command and exit without starting")
    ap.add_argument("--preflight", action="store_true", help="check node prerequisites and print the launch plan")
    ap.add_argument("--health-only", action="store_true", help="check an already-running server and exit")
    ap.add_argument("--wait-s", type=float, default=180.0)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require_llama = args.preflight or not (args.dry_run or args.health_only)
    plan = build_runtime_plan(args, require_llama=require_llama)

    if args.preflight:
        print(json.dumps(plan, indent=2))
        return 0 if plan["ok"] else 2

    if not plan["ok"] and not args.dry_run and not args.health_only:
        for failure in plan["failures"]:
            print(failure, file=sys.stderr)
        print(json.dumps(plan, indent=2))
        return 2

    if args.health_only:
        if not plan["base_url"]:
            print(json.dumps(plan, indent=2))
            return 2
        result = wait_for_models(plan["base_url"], deadline_s=1.0, interval_s=0.1)
        print(json.dumps({"base_url": plan["base_url"], **result}, indent=2))
        return 0 if result.get("ready") else 1

    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return 0 if plan["ok"] else 2

    proc = subprocess.Popen(plan["command"])
    try:
        result = wait_for_models(plan["base_url"], deadline_s=args.wait_s)
        print(json.dumps({"base_url": plan["base_url"], **result}, indent=2))
        if not result.get("ready"):
            stop_process(proc)
            return 1
        return proc.wait()
    except KeyboardInterrupt:
        stop_process(proc)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
