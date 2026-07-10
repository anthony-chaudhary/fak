from pathlib import Path
import shutil

from tools.cuda_arch_matrix import validate

ROOT = Path(__file__).resolve().parents[1]


def fixture(tmp_path: Path) -> Path:
    for rel in ("internal/compute/cuda_arch.txt", "internal/compute/build_cuda.sh", "tools/build_cuda_windows.ps1", "Dockerfile.cuda", "docs/cuda-dev.md"):
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dst)
    return tmp_path


def test_arch_matrix_current_tree_is_valid(tmp_path):
    assert validate(fixture(tmp_path)) == []


def test_arch_matrix_rejects_missing_sm120(tmp_path):
    root = fixture(tmp_path)
    p = root / "internal/compute/cuda_arch.txt"
    p.write_text(p.read_text().replace("sm_120\n", ""))
    assert any("PTX floor" in e for e in validate(root))


def test_arch_matrix_rejects_missing_ptx_floor(tmp_path):
    root = fixture(tmp_path)
    p = root / "internal/compute/build_cuda.sh"
    p.write_text(p.read_text().replace('GENCODE+=(-gencode "arch=compute_${PTX_CC},code=compute_${PTX_CC}")', ""))
    assert any("compute_${PTX_CC}" in e for e in validate(root))


def test_arch_matrix_rejects_doc_drift(tmp_path):
    root = fixture(tmp_path)
    p = root / "docs/cuda-dev.md"
    p.write_text(p.read_text().replace("cuda-build-sm120", "cuda-build-future"))
    assert any("docs" in e for e in validate(root))
