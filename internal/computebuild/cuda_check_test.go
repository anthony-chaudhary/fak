package computebuild

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
)

const cudaCheckHelperExitEnv = "FAK_CUDA_CHECK_HELPER_EXIT"

func init() {
	if os.Getenv(cudaCheckHelperExitEnv) != "" {
		os.Exit(23)
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestCUDACheckFailsClosedOnABIParityFailure(t *testing.T) {
	repoRoot, err := ResolveRepoRoot("")
	if err != nil {
		t.Fatal(err)
	}
	pythonDir := t.TempDir()
	python := filepath.Join(pythonDir, "python3")
	if runtime.GOOS == "windows" {
		python += ".exe"
	}
	testBinary, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	testBinaryBytes, err := os.ReadFile(testBinary)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(python, testBinaryBytes, 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", pythonDir+string(os.PathListSeparator)+os.Getenv("PATH"))
	t.Setenv(cudaCheckHelperExitEnv, "1")

	receiptPath := filepath.Join(t.TempDir(), "cuda-check.json")
	cfg := &CUDAConfig{
		Command:     "check",
		RepoRoot:    repoRoot,
		ReceiptPath: receiptPath,
		Toolchain:   &Toolchain{},
	}
	err = RunCUDA(context.Background(), cfg)
	assertCUDACheckFailedReceipt(t, err, cfg, receiptPath, 23)
}

// fak-test:runtime medium est=6s lane=default
func TestCUDACheckFailsClosedOnHeaderSyntaxFailure(t *testing.T) {
	repoRoot, err := ResolveRepoRoot("")
	if err != nil {
		t.Fatal(err)
	}
	testBinary, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv(cudaCheckHelperExitEnv, "1")

	receiptPath := filepath.Join(t.TempDir(), "cuda-check.json")
	cfg := &CUDAConfig{
		Command:     "check",
		RepoRoot:    repoRoot,
		ReceiptPath: receiptPath,
		Toolchain:   &Toolchain{CC: testBinary},
	}
	err = RunCUDA(context.Background(), cfg)
	assertCUDACheckFailedReceipt(t, err, cfg, receiptPath, 23)
}

// fak-test:runtime medium est=6s lane=default
func TestCUDACheckSuccessReceipt(t *testing.T) {
	repoRoot, err := ResolveRepoRoot("")
	if err != nil {
		t.Fatal(err)
	}
	receiptPath := filepath.Join(t.TempDir(), "cuda-check.json")
	cfg := &CUDAConfig{
		Command:     "check",
		RepoRoot:    repoRoot,
		ReceiptPath: receiptPath,
		Toolchain:   &Toolchain{},
	}
	if err := RunCUDA(context.Background(), cfg); err != nil {
		t.Fatal(err)
	}
	if cfg.Receipt == nil || cfg.Receipt.Outcome != "success" || cfg.Receipt.ExitCode != 0 {
		t.Fatalf("receipt = %#v", cfg.Receipt)
	}
	if len(cfg.Receipt.Phases) != 1 || cfg.Receipt.Phases[0].Name != "preflight_check" || cfg.Receipt.Phases[0].Outcome != "success" {
		t.Fatalf("phases = %#v", cfg.Receipt.Phases)
	}
}

func assertCUDACheckFailedReceipt(t *testing.T, runErr error, cfg *CUDAConfig, receiptPath string, wantExit int) {
	t.Helper()
	var exitErr *exec.ExitError
	if !errors.As(runErr, &exitErr) || exitErr.ExitCode() != wantExit {
		t.Fatalf("exit error = %v", runErr)
	}
	if cfg.Receipt == nil || cfg.Receipt.Outcome != "failed" || cfg.Receipt.ExitCode != wantExit {
		t.Fatalf("receipt = %#v", cfg.Receipt)
	}
	if len(cfg.Receipt.Phases) != 1 {
		t.Fatalf("phases = %#v", cfg.Receipt.Phases)
	}
	phase := cfg.Receipt.Phases[0]
	if phase.Name != "preflight_check" || phase.Outcome != "failed" || phase.ExitCode == nil || *phase.ExitCode != wantExit {
		t.Fatalf("phase = %#v", phase)
	}

	raw, err := os.ReadFile(receiptPath)
	if err != nil {
		t.Fatal(err)
	}
	var durable ComputeBuildReceipt
	if err := json.Unmarshal(raw, &durable); err != nil {
		t.Fatal(err)
	}
	if durable.Outcome != "failed" || durable.ExitCode != wantExit || len(durable.Phases) != 1 || durable.Phases[0].Outcome != "failed" {
		t.Fatalf("durable receipt = %#v", durable)
	}
}
