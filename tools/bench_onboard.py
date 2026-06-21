#!/usr/bin/env python3
"""bench_onboard.py -- register a new benchmark machine.

This script detects hardware, generates specs.json, and optionally runs a smoke benchmark.

Usage:
  python tools/bench_onboard.py --interactive
  python tools/bench_onboard.py --machine-id my-node --tags gpu,a100,linux
"""
import argparse
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "fak" / "experiments" / "benchmark"
MACHINES_DIR = BENCHMARK_DIR / "machines"


def run_command(cmd: List[str]) -> Optional[str]:
    """Run command and return stdout, or None on error."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def detect_cpu() -> Dict[str, Any]:
    """Detect CPU information."""
    cpu = {
        "model": platform.processor(),
        "architecture": platform.machine().lower(),
        "cores_physical": os.cpu_count() or 1,
        "cores_logical": os.cpu_count() or 1
    }

    # Try to get more accurate core counts
    if cpu["architecture"] == "x86_64" or cpu["architecture"] == "amd64":
        # Try lscpu on Linux/WSL
        if shutil_which("lscpu"):
            out = run_command(["lscpu"])
            if out:
                m = re.search(r"^Core\(s\) per socket:\s*(\d+)", out, re.M)
                if m:
                    cpu["cores_physical"] = int(m.group(1))
                m = re.search(r"^CPU\(s\):\s*(\d+)", out, re.M)
                if m:
                    cpu["cores_logical"] = int(m.group(1))

    # Try sysctl on macOS
    elif cpu["architecture"] == "arm64":
        out = run_command(["sysctl", "-n", "hw.physicalcpu"])
        if out:
            cpu["cores_physical"] = int(out)
        out = run_command(["sysctl", "-n", "hw.logicalcpu"])
        if out:
            cpu["cores_logical"] = int(out)

    return cpu


def detect_gpu() -> List[Dict[str, Any]]:
    """Detect GPU information."""
    gpus = []

    # Try nvidia-smi
    if shutil_which("nvidia-smi"):
        out = run_command(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
        if out:
            for line in out.splitlines():
                parts = line.split(",")
                if len(parts) >= 2:
                    name = parts[0].strip()
                    mem_str = parts[1].strip()
                    # Parse memory (e.g., "12288 MiB" -> 12288)
                    mem_match = re.search(r"(\d+)", mem_str)
                    mem_gb = int(mem_match.group(1)) / 1024 if mem_match else 0

                    gpu_info = {
                        "model": name,
                        "memory_gb": round(mem_gb, 1)
                    }

                    # Try to get compute capability
                    cuda_out = run_command(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"])
                    if cuda_out:
                        lines = cuda_out.splitlines()
                        if len(lines) > 0:
                            gpu_info["compute_capability"] = lines[0].strip()

                    gpus.append(gpu_info)

    # Try rocm-smi for AMD GPUs
    if not gpus and shutil_which("rocm-smi"):
        out = run_command(["rocm-smi", "--showproductname"])
        if out:
            # Parse ROCm output (simplified)
            m = re.search(r"Card series:\s*(.+)", out)
            if m:
                gpus.append({"model": m.group(1).strip()})

    # Try system_profiler on macOS
    if not gpus and platform.system() == "Darwin":
        out = run_command(["system_profiler", "SPDisplaysDataType"])
        if out:
            m = re.search(r"Chip:\s*(.+)", out)
            if m:
                gpus.append({"model": m.group(1).strip(), "memory_gb": "unified"})

    return gpus


def detect_ram() -> float:
    """Detect total RAM in GB."""
    ram_gb = 0

    if platform.system() == "Linux":
        # Read /proc/meminfo
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(re.search(r"\d+", line).group())
                        ram_gb = kb / (1024 * 1024)
                        break
        except (OSError, AttributeError):
            pass
    elif platform.system() == "Darwin":
        out = run_command(["sysctl", "-n", "hw.memsize"])
        if out:
            ram_gb = int(out) / (1024**3)
    elif platform.system() == "Windows":
        # wmic.exe is removed on Win11 24H2+ (build 26200); use the supported CIM API.
        out = run_command([
            "powershell", "-NoProfile", "-NonInteractive", "-Command",
            "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory",
        ])
        if out:
            m = re.search(r"\d+", out)
            if m:
                ram_gb = int(m.group()) / (1024**3)

    return round(ram_gb, 1)


def detect_os() -> Dict[str, str]:
    """Detect OS information."""
    return {
        "name": platform.system(),
        "version": platform.version(),
        "kernel": platform.release()
    }


def detect_runtime() -> Dict[str, Optional[str]]:
    """Detect runtime versions."""
    runtime = {}

    # Go version
    go_out = run_command(["go", "version"])
    if go_out:
        runtime["go_version"] = go_out.split()[2]  # "go version go1.26.0 ..."

    # Python version
    runtime["python_version"] = platform.python_version()

    # CUDA version
    if shutil_which("nvidia-smi"):
        cuda_out = run_command(["nvidia-smi"])
        if cuda_out:
            m = re.search(r"CUDA Version:\s*([\d.]+)", cuda_out)
            if m:
                runtime["cuda_driver_version"] = m.group(1)

    return runtime


def shutil_which(name: str) -> bool:
    """Check if command exists (like shutil.which)."""
    try:
        return subprocess.run(
            ["which", name],
            capture_output=True,
            timeout=1
        ).returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def generate_specs(machine_id: str, tags: List[str]) -> Dict:
    """Generate machine specs dict."""
    return {
        "$schema": "benchmark/machine-specs.v1",
        "machine_id": machine_id,
        "hostname": platform.node(),
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "hardware": {
            "cpu": detect_cpu(),
            "gpu": detect_gpu(),
            "ram_gb": detect_ram()
        },
        "os": detect_os(),
        "runtime": detect_runtime(),
        "tags": tags
    }


def interactive_onboard() -> Dict:
    """Interactive machine onboarding."""
    print("=== Benchmark Machine Onboarding ===")
    print()

    # Suggest machine ID
    hostname = platform.node().lower().replace(".", "-").replace("_", "-")
    default_id = re.sub(r"[^a-z0-9-]", "-", hostname)

    machine_id = input(f"Machine ID [{default_id}]: ").strip() or default_id
    machine_id = re.sub(r"[^a-z0-9-]", "-", machine_id.lower())

    print()
    print("Detected hardware:")
    cpu = detect_cpu()
    print(f"  CPU: {cpu.get('model', '?')}")
    print(f"  Cores: {cpu.get('cores_logical', '?')}")

    gpus = detect_gpu()
    if gpus:
        for gpu in gpus:
            print(f"  GPU: {gpu.get('model', '?')}")
    else:
        print("  GPU: none detected")

    ram = detect_ram()
    print(f"  RAM: {ram} GB")

    print()
    tags_input = input("Tags (comma-separated, e.g., gpu,a100,linux): ").strip()
    tags = [t.strip() for t in tags_input.split(",")] if tags_input else []

    # Auto-add some tags based on detection
    if gpus and "gpu" not in tags:
        tags.append("gpu")
    arch = cpu.get("architecture", "")
    if arch and arch not in tags:
        tags.append(arch)
    os_name = platform.system().lower()
    if os_name and os_name not in tags:
        tags.append(os_name)

    return generate_specs(machine_id, tags)


def save_specs(specs: Dict, overwrite: bool = False) -> bool:
    """Save specs to machines directory."""
    machine_id = specs["machine_id"]
    dest_dir = MACHINES_DIR / machine_id
    dest_path = dest_dir / "specs.json"

    if dest_path.exists() and not overwrite:
        print(f"[ERROR] Machine {machine_id} already registered at {dest_path}", file=sys.stderr)
        print("Use --replace to overwrite", file=sys.stderr)
        return False

    dest_dir.mkdir(parents=True, exist_ok=True)

    with open(dest_path, "w", encoding="utf-8") as f:
        json.dump(specs, f, indent=2)

    print(f"[bench_onboard] Saved specs to {dest_path}", file=sys.stderr)
    return True


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--interactive", action="store_true",
                   help="Interactive onboarding mode")
    ap.add_argument("--machine-id", help="Machine ID (slug format)")
    ap.add_argument("--tags", help="Comma-separated tags")
    ap.add_argument("--replace", action="store_true",
                   help="Overwrite existing specs")
    args = ap.parse_args(argv)

    if args.interactive:
        specs = interactive_onboard()
    else:
        if not args.machine_id:
            print("[ERROR] --machine-id is required for non-interactive mode", file=sys.stderr)
            return 1

        tags = []
        if args.tags:
            tags = [t.strip() for t in args.tags.split(",")]

        specs = generate_specs(args.machine_id, tags)

    # Display specs before saving
    print()
    print("Generated specs:")
    print(json.dumps(specs, indent=2))
    print()

    if not args.interactive:
        # Auto-save in non-interactive mode
        if save_specs(specs, args.replace):
            print(f"[bench_onboard] Machine '{specs['machine_id']}' registered successfully", file=sys.stderr)
            print("[bench_onboard] Run 'python tools/bench_catalog.py build' to update catalog", file=sys.stderr)
            return 0
        return 1

    # Interactive mode: confirm before saving
    confirm = input("Save these specs? [Y/n]: ").strip().lower()
    if confirm and confirm not in ["y", "yes"]:
        print("[bench_onboard] Aborted", file=sys.stderr)
        return 1

    if save_specs(specs, args.replace):
        print(f"[bench_onboard] Machine '{specs['machine_id']}' registered successfully", file=sys.stderr)
        print("[bench_onboard] Run 'python tools/bench_catalog.py build' to update catalog", file=sys.stderr)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
