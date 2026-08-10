package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRunCUDAArchCurrentTree(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := runCUDAArch(&out, &errOut, []string{"--root", filepath.Join("..", "..")}); code != 0 {
		t.Fatalf("runCUDAArch() code = %d, stdout = %q, stderr = %q", code, out.String(), errOut.String())
	}
	want := "cuda-arch-matrix: OK (declared SASS set + compute_120 PTX floor; Linux/Windows/Docker/docs agree)\n"
	if out.String() != want || errOut.Len() != 0 {
		t.Fatalf("stdout = %q, stderr = %q", out.String(), errOut.String())
	}
}

func TestRunCUDAArchFailureOutput(t *testing.T) {
	root := copyCUDAArchFixture(t)
	path := filepath.Join(root, "docs", "cuda-dev.md")
	contents, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	contents = []byte(strings.Replace(string(contents), "cuda-build-sm120", "cuda-build-future", 1))
	if err := os.WriteFile(path, contents, 0o644); err != nil {
		t.Fatal(err)
	}

	var out, errOut bytes.Buffer
	if code := runCUDAArch(&out, &errOut, []string{"--root", root}); code != 1 {
		t.Fatalf("runCUDAArch() code = %d, stdout = %q, stderr = %q", code, out.String(), errOut.String())
	}
	if got := out.String(); !strings.HasPrefix(got, "cuda-arch-matrix: FAIL\n") || !strings.Contains(got, `  - docs: missing arch-matrix contract "cuda-build-sm120"`) {
		t.Fatalf("stdout = %q", got)
	}
	if errOut.Len() != 0 {
		t.Fatalf("stderr = %q", errOut.String())
	}
}

func copyCUDAArchFixture(t *testing.T) string {
	t.Helper()
	sourceRoot := filepath.Join("..", "..")
	destRoot := t.TempDir()
	files := []string{"internal/compute/cuda_arch.txt", "internal/compute/build_cuda.sh", "tools/build_cuda_windows.ps1", "Dockerfile.cuda", "docs/cuda-dev.md"}
	for _, rel := range files {
		contents, err := os.ReadFile(filepath.Join(sourceRoot, filepath.FromSlash(rel)))
		if err != nil {
			t.Fatal(err)
		}
		dest := filepath.Join(destRoot, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(dest, contents, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return destRoot
}
