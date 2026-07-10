#!/usr/bin/env python3
"""GPU-free validator for the declared CUDA SASS matrix and PTX floor."""
from __future__ import annotations

import argparse
from pathlib import Path


def validate(root: Path) -> list[str]:
    errors: list[str] = []
    arch_file = root / "internal/compute/cuda_arch.txt"
    arches = [x.strip() for x in arch_file.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not arches or len(arches) != len(set(arches)):
        errors.append("cuda_arch.txt must contain a non-empty unique architecture list")
    if any(not x.startswith("sm_") or not x[3:].isdigit() for x in arches):
        errors.append("cuda_arch.txt entries must use sm_<digits>")

    files = {
        "linux": (root / "internal/compute/build_cuda.sh").read_text(encoding="utf-8"),
        "windows": (root / "tools/build_cuda_windows.ps1").read_text(encoding="utf-8-sig"),
        "docker": (root / "Dockerfile.cuda").read_text(encoding="utf-8"),
        "docs": (root / "docs/cuda-dev.md").read_text(encoding="utf-8"),
    }
    required = {
        "linux": ("cuda_arch.txt", "code=${arch}", "code=compute_${PTX_CC}"),
        "windows": ("cuda_arch.txt", "code=${item}", "code=compute_${cc}"),
        "docker": ("cuda_arch.txt", "code=${arch}", "code=compute_${cc}"),
        "docs": ("internal/compute/cuda_arch.txt", "cuda-build-sm100", "cuda-build-sm120"),
    }
    for name, needles in required.items():
        for needle in needles:
            if needle not in files[name]:
                errors.append(f"{name}: missing arch-matrix contract {needle!r}")
    if arches and arches[-1] != "sm_120":
        errors.append(f"PTX floor must follow highest declared arch sm_120, got {arches[-1]!r}")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args()
    errors = validate(args.root)
    if errors:
        print("cuda-arch-matrix: FAIL")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("cuda-arch-matrix: OK (declared SASS set + compute_120 PTX floor; Linux/Windows/Docker/docs agree)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
