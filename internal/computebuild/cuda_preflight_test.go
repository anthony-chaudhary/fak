package computebuild

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"
	"time"
)

type cudaPreflightInvocation struct {
	Stage       string `json:"stage"`
	InputExists bool   `json:"input_exists"`
	InputAbs    bool   `json:"input_abs"`
}

// Reuse the running Go test executable as each external tool. Its init runs
// before test flag parsing, so the fixture needs no interpreter or compiler.
func init() {
	trace := os.Getenv("FAK_CUDA_PREFLIGHT_TEST_TRACE")
	if trace == "" {
		return
	}
	name := strings.TrimSuffix(filepath.Base(os.Args[0]), ".exe")
	var stage, input string
	switch name {
	case "python3":
		stage = "abi"
		if len(os.Args) > 1 {
			input = os.Args[1]
		}
	case "go":
		stage = "arch"
	case "cuda-preflight-cc":
		stage = "header"
		if len(os.Args) > 1 {
			input = os.Args[len(os.Args)-1]
		}
	default:
		return
	}
	call := cudaPreflightInvocation{Stage: stage, InputExists: true, InputAbs: true}
	if stage != "arch" {
		info, err := os.Stat(input)
		call.InputExists = err == nil && !info.IsDir()
		call.InputAbs = filepath.IsAbs(input)
	}
	f, err := os.OpenFile(trace, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0600)
	if err != nil {
		os.Exit(98)
	}
	if err := json.NewEncoder(f).Encode(call); err != nil {
		_ = f.Close()
		os.Exit(98)
	}
	if err := f.Close(); err != nil {
		os.Exit(98)
	}
	if !call.InputExists {
		os.Exit(31)
	}
	if os.Getenv("FAK_CUDA_PREFLIGHT_TEST_FAIL") == stage {
		if stage == "abi" {
			os.Exit(17)
		}
		os.Exit(23)
	}
	os.Exit(0)
}

// fak-test:justify why=regression when=changed:internal/computebuild/**
// fak-test:runtime fast est=1s
func TestCUDACheckPropagatesPreflightFailures(t *testing.T) {
	for _, tc := range []struct {
		stage string
		code  int
		calls []string
	}{
		{stage: "abi", code: 17, calls: []string{"abi"}},
		{stage: "header", code: 23, calls: []string{"abi", "arch", "header"}},
	} {
		t.Run(tc.stage, func(t *testing.T) {
			cfg, trace := newCUDAPreflightFixture(t)
			t.Setenv("FAK_CUDA_PREFLIGHT_TEST_FAIL", tc.stage)
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			err := RunCUDA(ctx, cfg)
			var exitErr *exec.ExitError
			if !errors.As(err, &exitErr) || exitErr.ExitCode() != tc.code {
				t.Fatalf("preflight result = %v, want tool exit %d", err, tc.code)
			}
			calls := readCUDAPreflightTrace(t, trace)
			stages := make([]string, len(calls))
			for i, call := range calls {
				stages[i] = call.Stage
			}
			if !reflect.DeepEqual(stages, tc.calls) {
				t.Fatalf("tools invoked = %v, want %v", stages, tc.calls)
			}
			receipt := readCUDAPreflightReceipt(t, cfg.ReceiptPath)
			if receipt.Outcome != "failed" || receipt.ExitCode != tc.code {
				t.Fatalf("receipt outcome=%s exit=%d, want failed/%d", receipt.Outcome, receipt.ExitCode, tc.code)
			}
			if len(receipt.Phases) != 1 || receipt.Phases[0].Name != "preflight_check" || receipt.Phases[0].Outcome != "failed" {
				t.Fatalf("failed preflight was not retained as the sole receipt phase: %v", receipt.Phases)
			}
		})
	}
}

// fak-test:justify why=regression when=changed:internal/computebuild/**
// fak-test:runtime fast est=1s
func TestCUDACheckNormalizesRelativeRootBeforeToolExecution(t *testing.T) {
	cfg, trace := newCUDAPreflightFixture(t)
	cwd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	cfg.RepoRoot, err = filepath.Rel(cwd, cfg.RepoRoot)
	if err != nil {
		t.Fatal(err)
	}
	cfg.PkgDir = ""
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := RunCUDA(ctx, cfg); err != nil {
		t.Fatalf("relative-root preflight: %v", err)
	}
	calls := readCUDAPreflightTrace(t, trace)
	var stages []string
	for _, call := range calls {
		stages = append(stages, call.Stage)
		if !call.InputExists || !call.InputAbs {
			t.Fatalf("%s received missing or relative input: %+v", call.Stage, call)
		}
	}
	if !reflect.DeepEqual(stages, []string{"abi", "arch", "header"}) {
		t.Fatalf("tools invoked = %v, want all three preflights", stages)
	}
	receipt := readCUDAPreflightReceipt(t, cfg.ReceiptPath)
	if receipt.Outcome != "success" || receipt.ExitCode != 0 {
		t.Fatalf("successful tools produced outcome=%s exit=%d", receipt.Outcome, receipt.ExitCode)
	}
}

func newCUDAPreflightFixture(t *testing.T) (*CUDAConfig, string) {
	t.Helper()
	root := t.TempDir()
	repo := filepath.Join(root, "repo")
	bin := filepath.Join(root, "bin")
	for _, dir := range []string{bin, filepath.Join(repo, "tools"), filepath.Join(repo, "internal", "compute")} {
		if err := os.MkdirAll(dir, 0755); err != nil {
			t.Fatal(err)
		}
	}
	for _, path := range []string{"go.mod", "tools/cuda_abi_parity.py", "internal/compute/cuda_backend.h"} {
		if err := os.WriteFile(filepath.Join(repo, filepath.FromSlash(path)), nil, 0600); err != nil {
			t.Fatal(err)
		}
	}
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	helper, err := os.ReadFile(executable)
	if err != nil {
		t.Fatal(err)
	}
	toolPath := func(name string) string {
		if runtime.GOOS == "windows" {
			name += ".exe"
		}
		return filepath.Join(bin, name)
	}
	for _, name := range []string{"python3", "go", "cuda-preflight-cc"} {
		if err := os.WriteFile(toolPath(name), helper, 0755); err != nil {
			t.Fatal(err)
		}
	}
	trace := filepath.Join(root, "calls.jsonl")
	t.Setenv("PATH", bin)
	t.Setenv("FAK_CUDA_PREFLIGHT_TEST_TRACE", trace)
	t.Setenv("FAK_CUDA_PREFLIGHT_TEST_FAIL", "")
	return &CUDAConfig{
		Command:     "check",
		RepoRoot:    repo,
		ReceiptPath: filepath.Join(root, "receipt.json"),
		Toolchain:   &Toolchain{CC: toolPath("cuda-preflight-cc")},
		Stdout:      io.Discard,
		Stderr:      io.Discard,
	}, trace
}

func readCUDAPreflightTrace(t *testing.T, path string) []cudaPreflightInvocation {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	var calls []cudaPreflightInvocation
	decoder := json.NewDecoder(f)
	for {
		var call cudaPreflightInvocation
		if err := decoder.Decode(&call); err != nil {
			if errors.Is(err, io.EOF) {
				return calls
			}
			t.Fatal(err)
		}
		calls = append(calls, call)
	}
}

func readCUDAPreflightReceipt(t *testing.T, path string) ComputeBuildReceipt {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var receipt ComputeBuildReceipt
	if err := json.Unmarshal(data, &receipt); err != nil {
		t.Fatal(err)
	}
	return receipt
}
