import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _minor(version: str) -> tuple[int, int]:
    major, minor, *_ = version.split(".")
    return int(major), int(minor)


def test_blackwell_arches_are_first_class_single_arch_targets():
    arches = (ROOT / "internal/compute/cuda_arch.txt").read_text().split()
    assert "sm_100" in arches
    assert "sm_120" in arches
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "cuda-build-sm100:" in makefile
    assert "FAK_CUDA_ARCH=sm_100 bash internal/compute/build_cuda.sh build" in makefile
    assert "cuda-build-sm120:" in makefile
    assert "FAK_CUDA_ARCH=sm_120 bash internal/compute/build_cuda.sh build" in makefile


def test_build_entry_points_validate_the_declared_arch_set():
    build = (ROOT / "internal/compute/build_cuda.sh").read_text(encoding="utf-8")
    windows = (ROOT / "tools/build_cuda_windows.ps1").read_text(encoding="utf-8-sig")
    docker = (ROOT / "Dockerfile.cuda").read_text(encoding="utf-8")
    assert 'ARCH_FILE="$SCRIPT_DIR/cuda_arch.txt"' in build
    assert "unsupported CUDA arch" in build
    assert "internal\\compute\\cuda_arch.txt" in windows
    assert "internal/compute/cuda_arch.txt" in docker


def test_declared_arches_use_blackwell_capable_cuda_toolchains():
    arches = (ROOT / "internal/compute/cuda_arch.txt").read_text().split()
    docker = (ROOT / "Dockerfile.cuda").read_text(encoding="utf-8")
    setup = (ROOT / "internal/compute/setup_cuda_wsl.sh").read_text(encoding="utf-8")

    docker_versions = re.findall(r"^FROM nvidia/cuda:([0-9.]+)-(?:devel|runtime)-", docker, re.MULTILINE)
    setup_versions = re.findall(
        r"(?:cuda-nvcc|cuda-cudart-dev|cuda-nvrtc-dev|libcublas-dev|cuda-cccl)=([0-9.]+)",
        setup,
    )
    assert docker_versions == ["12.8.1", "12.8.1"]
    assert setup_versions == ["12.8"] * 5

    minimum_by_arch = {"sm_100": (12, 8), "sm_120": (12, 8)}
    for arch in arches:
        minimum = minimum_by_arch.get(arch, (0, 0))
        assert all(_minor(version) >= minimum for version in docker_versions)
        assert all(_minor(version) >= minimum for version in setup_versions)


def test_default_build_is_fatbin_with_highest_arch_ptx_floor():
    arches = (ROOT / "internal/compute/cuda_arch.txt").read_text().split()
    build = (ROOT / "internal/compute/build_cuda.sh").read_text(encoding="utf-8")
    windows = (ROOT / "tools/build_cuda_windows.ps1").read_text(encoding="utf-8-sig")
    docker = (ROOT / "Dockerfile.cuda").read_text(encoding="utf-8")
    for arch in arches:
        cc = arch.removeprefix("sm_")
        # Each entry point derives one SASS gencode from every declared arch.
        assert "code=${arch}" in build
        assert "code=${item}" in windows
        assert "code=${arch}" in docker
        assert cc
    highest = arches[-1].removeprefix("sm_")
    assert 'code=compute_${PTX_CC}' in build
    assert 'code=compute_${cc}' in windows
    assert 'code=compute_${cc}' in docker
    assert highest == "120"
    assert 'ARCH="${FAK_CUDA_ARCH:-}"' in build


# The dry-run gate runs this file as a bare script (`python3 tools/…`), not under
# pytest; without a self-runner that invocation is a vacuous green (zero tests run),
# so a perversion of any contract above would gate nothing (#12488).
def _run_all() -> int:
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"ok   {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
