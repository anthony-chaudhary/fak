package computebuild

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

// fak-test:runtime medium est=5s lane=default
func TestRunCUDACheckPropagatesCommandFailures(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	root, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	binDir := filepath.Join(root, "fake tools")
	if err := os.MkdirAll(binDir, 0755); err != nil {
		t.Fatal(err)
	}
	ext := ""
	if runtime.GOOS == "windows" {
		ext = ".exe"
	}
	// A native Go helper replaces all three commands. No Python, C compiler,
	// CUDA toolkit, shell script, or recursive go test runs inside RunCUDA.
	source := filepath.Join(root, "fake-cuda-check.go")
	if err := os.WriteFile(source, []byte(cudaCheckFakeToolSource), 0644); err != nil {
		t.Fatal(err)
	}
	tool := filepath.Join(root, "fake-cuda-check"+ext)
	cmd := exec.CommandContext(ctx, "go", "build", "-o", tool, source)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("build native CUDA check fixture: %v\n%s", err, out)
	}
	toolBytes, err := os.ReadFile(tool)
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"python3", "go", "cc"} {
		if err := os.WriteFile(filepath.Join(binDir, name+ext), toolBytes, 0755); err != nil {
			t.Fatal(err)
		}
	}
	t.Setenv("PATH", binDir)
	if runtime.GOOS == "windows" {
		t.Setenv("PATHEXT", ".EXE")
	}
	for _, tc := range []struct {
		name     string
		failTool string
		exitCode int
		calls    string
		message  string
	}{
		{name: "all_pass", calls: "python3\ngo\ncc\n"},
		{name: "abi_failure", failTool: "python3", exitCode: 17, calls: "python3\n", message: "CUDA ABI parity check"},
		{name: "architecture_failure", failTool: "go", exitCode: 19, calls: "python3\ngo\n", message: "go test ./internal/cudaarch"},
		{name: "header_failure", failTool: "cc", exitCode: 23, calls: "python3\ngo\ncc\n", message: "CUDA standalone header check"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			repo := filepath.Join(root, tc.name)
			pkg := filepath.Join(repo, "internal", "compute")
			for _, path := range []string{filepath.Join(repo, "tools", "cuda_abi_parity.py"), filepath.Join(pkg, "cuda_backend.h")} {
				if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(path, []byte("fixture\n"), 0644); err != nil {
					t.Fatal(err)
				}
			}
			logPath := filepath.Join(repo, "commands.log")
			t.Setenv("FAK_CUDA_CHECK_FIXTURE_ROOT", repo)
			t.Setenv("FAK_CUDA_CHECK_FIXTURE_LOG", logPath)
			t.Setenv("FAK_CUDA_CHECK_FIXTURE_FAIL", tc.failTool)
			cfg := &CUDAConfig{
				Command:     "check",
				RepoRoot:    repo,
				Toolchain:   &Toolchain{CC: filepath.Join(binDir, "cc"+ext)},
				ReceiptPath: filepath.Join(repo, "receipt.json"),
				Stdout:      io.Discard,
				Stderr:      io.Discard,
			}
			err := RunCUDA(ctx, cfg)
			wantOutcome, wantError := "success", ""
			if tc.exitCode == 0 {
				if err != nil {
					t.Fatalf("RunCUDA: %v", err)
				}
			} else {
				wantOutcome = "failed"
				var exitErr *exec.ExitError
				if !errors.As(err, &exitErr) || exitErr.ExitCode() != tc.exitCode {
					t.Fatalf("RunCUDA error = %v, want wrapped exit code %d", err, tc.exitCode)
				}
				wantError = err.Error()
				if !strings.Contains(wantError, tc.message) {
					t.Fatalf("RunCUDA error = %q, want %q", wantError, tc.message)
				}
			}
			calls, err := os.ReadFile(logPath)
			if err != nil || string(calls) != tc.calls {
				t.Fatalf("command order = %q, err = %v, want %q", calls, err, tc.calls)
			}
			data, err := os.ReadFile(cfg.ReceiptPath)
			if err != nil {
				t.Fatal(err)
			}
			var persisted ComputeBuildReceipt
			if err := json.Unmarshal(data, &persisted); err != nil {
				t.Fatal(err)
			}
			for _, receipt := range []*ComputeBuildReceipt{cfg.Receipt, &persisted} {
				if receipt == nil || receipt.Outcome != wantOutcome || receipt.ExitCode != tc.exitCode || receipt.Error != wantError {
					t.Fatalf("receipt = %+v, want outcome %s exit %d error %q", receipt, wantOutcome, tc.exitCode, wantError)
				}
				if len(receipt.Phases) != 1 {
					t.Fatalf("phases = %+v, want one preflight_check", receipt.Phases)
				}
				phase := receipt.Phases[0]
				if phase.Name != "preflight_check" || phase.Outcome != wantOutcome || phase.ExitCode == nil || *phase.ExitCode != tc.exitCode || phase.Error != wantError {
					t.Fatalf("preflight phase = %+v", phase)
				}
			}
		})
	}
}

const cudaCheckFakeToolSource = `package main

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
)

func main() {
	name := strings.TrimSuffix(filepath.Base(os.Args[0]), ".exe")
	root := os.Getenv("FAK_CUDA_CHECK_FIXTURE_ROOT")
	pkg := filepath.Join(root, "internal", "compute")
	wantDir := root
	var wantArgs []string
	switch name {
	case "python3":
		wantArgs = []string{filepath.Join(root, "tools", "cuda_abi_parity.py"), "--check"}
	case "go":
		wantArgs = []string{"test", "./internal/cudaarch"}
	case "cc":
		wantDir = pkg
		wantArgs = []string{"-x", "c", "-std=c11", "-fsyntax-only", "-Wall", "-Werror", filepath.Join(pkg, "cuda_backend.h")}
	default:
		os.Exit(90)
	}
	dir, err := os.Getwd()
	if err != nil || dir != wantDir || !reflect.DeepEqual(os.Args[1:], wantArgs) {
		os.Exit(91)
	}
	log, err := os.OpenFile(os.Getenv("FAK_CUDA_CHECK_FIXTURE_LOG"), os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		os.Exit(92)
	}
	if _, err := log.WriteString(name + "\n"); err != nil {
		os.Exit(93)
	}
	if err := log.Close(); err != nil {
		os.Exit(94)
	}
	if os.Getenv("FAK_CUDA_CHECK_FIXTURE_FAIL") == name {
		os.Exit(map[string]int{"python3": 17, "go": 19, "cc": 23}[name])
	}
}
`
