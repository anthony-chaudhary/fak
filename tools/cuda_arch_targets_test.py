from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


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
