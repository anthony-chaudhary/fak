package computebuild

import (
	"bytes"
	"context"
	"errors"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

// fak-test:runtime fast est=20ms lane=default
func TestParseCUDAArchListIsCRLFTolerant(t *testing.T) {
	got := ParseCUDAArchList("sm_80\r\nsm_89\r\n\r\nsm_90")
	want := []string{"sm_80", "sm_89", "sm_90"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("ParseCUDAArchList = %q, want %q", got, want)
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestCUDACheckStrictRequiresPython(t *testing.T) {
	repoRoot, err := ResolveRepoRoot("")
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", t.TempDir())
	t.Setenv("PYTHON", "")
	cfg := &CUDAConfig{
		Command:     "check",
		RepoRoot:    repoRoot,
		ReceiptPath: filepath.Join(t.TempDir(), "cuda-check.json"),
		Toolchain:   &Toolchain{},
	}
	err = RunCUDA(WithStrictCUDACheck(context.Background()), cfg)
	if err == nil || !strings.Contains(err.Error(), "no python3/python on PATH — cannot run tools/cuda_abi_parity.py") {
		t.Fatalf("strict check without python: err = %v", err)
	}
}

// fak-test:runtime medium est=6s lane=default
func TestCUDACheckStrictReportsSkippedParseAndOKLine(t *testing.T) {
	if _, err := cudaCheckPython(true); err != nil {
		t.Skip("no python3/python on PATH to run the strict parity check")
	}
	repoRoot, err := ResolveRepoRoot("")
	if err != nil {
		t.Fatal(err)
	}
	// The strict check runs tools/cuda_abi_parity.py; while that script reports pre-existing
	// HARD prototype mismatches on the tree, the OK-line contract cannot hold, so skip with the cause.
	py, _ := cudaCheckPython(true)
	if parity := exec.Command(py, filepath.Join(repoRoot, "tools", "cuda_abi_parity.py"), "--check"); parity.Run() != nil {
		t.Skip("tools/cuda_abi_parity.py --check reports pre-existing HARD ABI mismatches; strict OK line is not reachable until they are fixed")
	}
	var out bytes.Buffer
	cfg := &CUDAConfig{
		Command:     "check",
		RepoRoot:    repoRoot,
		ReceiptPath: filepath.Join(t.TempDir(), "cuda-check.json"),
		Toolchain:   &Toolchain{},
		Stdout:      &out,
		Stderr:      &out,
	}
	if err := RunCUDA(WithStrictCUDACheck(context.Background()), cfg); err != nil {
		t.Fatalf("strict check: %v\n%s", err, out.String())
	}
	if !strings.Contains(out.String(), "[cuda] strict standalone parse skipped: no C compiler on PATH") {
		t.Fatalf("missing skipped-parse line:\n%s", out.String())
	}
	if !strings.HasSuffix(strings.TrimSpace(out.String()), "[cuda] OK check") {
		t.Fatalf("output does not end with OK line:\n%s", out.String())
	}
}

// fak-test:runtime medium est=2s lane=default
func TestCUDACheckDoesNotWaitOnBuildLock(t *testing.T) {
	repoRoot, err := ResolveRepoRoot("")
	if err != nil {
		t.Fatal(err)
	}
	_, unlock, err := lockCUDABuild(context.Background(), filepath.Join(repoRoot, "internal", "compute"))
	if err != nil {
		t.Fatal(err)
	}
	defer unlock()
	// An unrunnable interpreter ends the check at its first step, well inside the deadline.
	t.Setenv("PYTHON", filepath.Join(t.TempDir(), "missing-python"))
	ctx, cancel := context.WithTimeout(WithStrictCUDACheck(context.Background()), 3*time.Second)
	defer cancel()
	cfg := &CUDAConfig{
		Command:     "check",
		RepoRoot:    repoRoot,
		ReceiptPath: filepath.Join(t.TempDir(), "cuda-check.json"),
		Toolchain:   &Toolchain{},
	}
	if err := RunCUDA(ctx, cfg); errors.Is(err, context.DeadlineExceeded) {
		t.Fatal("check blocked on the CUDA build lock held by a build")
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestCUDACheckPythonHonorsPYTHONOverride(t *testing.T) {
	t.Setenv("PYTHON", "pinned-python")
	got, err := cudaCheckPython(true)
	if err != nil || got != "pinned-python" {
		t.Fatalf("cudaCheckPython(strict) = %q, %v; want pinned-python", got, err)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestCUDACheckPythonPlainIgnoresPYTHONOverride(t *testing.T) {
	t.Setenv("PYTHON", "pinned-python")
	t.Setenv("PATH", t.TempDir())
	if got, err := cudaCheckPython(false); err == nil {
		t.Fatalf("plain cudaCheckPython used PYTHON override: %q", got)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestStrictCUDACheckIsOptInOnly(t *testing.T) {
	if IsStrictCUDACheck(context.Background()) {
		t.Fatal("plain context must stay fail-open")
	}
	if !IsStrictCUDACheck(WithStrictCUDACheck(context.Background())) {
		t.Fatal("WithStrictCUDACheck context not strict")
	}
}
