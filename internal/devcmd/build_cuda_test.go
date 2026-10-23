package devcmd

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestCUDABuildHelpAndInvalid(t *testing.T) {
	var stdout, stderr bytes.Buffer

	// 1. Help flag should return exit code 0
	code := RunBuildCUDA(&stdout, &stderr, []string{"--help"})
	if code != 0 {
		t.Fatalf("expected 0 for --help, got %d", code)
	}
	if !strings.Contains(stderr.String(), "usage: fak-dev build-cuda") {
		t.Fatalf("expected usage message, got %q", stderr.String())
	}

	// 2. Empty args should return exit code 2
	stdout.Reset()
	stderr.Reset()
	code = RunBuildCUDA(&stdout, &stderr, []string{})
	if code != 2 {
		t.Fatalf("expected 2 for empty args, got %d", code)
	}

	// 3. Unknown subcommand should return exit code 2
	stdout.Reset()
	stderr.Reset()
	code = RunBuildCUDA(&stdout, &stderr, []string{"unknown"})
	if code != 2 {
		t.Fatalf("expected 2 for unknown subcommand, got %d", code)
	}
	if !strings.Contains(stderr.String(), "unknown subcommand") {
		t.Fatalf("expected unknown subcommand error, got %q", stderr.String())
	}
}

func TestCUDABuildFlagParsingAndJSON(t *testing.T) {
	var stdout, stderr bytes.Buffer

	// Test "build" with nonexistent repo root and json flag
	code := RunBuildCUDA(&stdout, &stderr, []string{
		"build",
		"-json",
		"-arch", "sm_89",
		"-cuda-home", "/nonexistent/cuda",
		"-nccl",
		"-repo-root", "/nonexistent/repo",
	})

	if code != 1 {
		t.Fatalf("expected exit code 1 on failed build, got %d", code)
	}

	var res CUDABuildResult
	if err := json.Unmarshal(stdout.Bytes(), &res); err != nil {
		t.Fatalf("failed to decode JSON output: %v\nstdout: %s", err, stdout.String())
	}

	if res.Schema != "fak.cuda-build.v1" {
		t.Errorf("expected schema fak.cuda-build.v1, got %q", res.Schema)
	}
	if res.Command != "build" {
		t.Errorf("expected command build, got %q", res.Command)
	}
	if res.Arch != "sm_89" {
		t.Errorf("expected arch sm_89, got %q", res.Arch)
	}
	if res.Success {
		t.Errorf("expected success false, got true")
	}
	if res.Error == "" {
		t.Errorf("expected non-empty error message in JSON")
	}
}

func TestCUDABuildBinaryPositionalArgs(t *testing.T) {
	var stdout, stderr bytes.Buffer

	// Test "binary" with positional args in json mode
	code := RunBuildCUDA(&stdout, &stderr, []string{
		"binary",
		"-json",
		"-arch", "sm_89",
		"-repo-root", "/nonexistent/repo",
		"./cmd/fak",
		"bin/fak.exe",
	})

	if code != 1 {
		t.Fatalf("expected exit code 1, got %d", code)
	}

	var res CUDABuildResult
	if err := json.Unmarshal(stdout.Bytes(), &res); err != nil {
		t.Fatalf("failed to decode JSON output: %v", err)
	}
	if res.Command != "binary" {
		t.Fatalf("expected command binary, got %s", res.Command)
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestCUDABuildExplicitToolkitSelectsNVCC(t *testing.T) {
	root := t.TempDir()
	ext := ""
	if runtime.GOOS == "windows" {
		ext = ".exe"
	}
	ambient := filepath.Join(root, "ambient CUDA")
	selected := filepath.Join(root, "selected CUDA")
	for _, home := range []string{ambient, selected} {
		bin := filepath.Join(home, "bin")
		if err := os.MkdirAll(bin, 0755); err != nil {
			t.Fatal(err)
		}
		// Discovery accepts this file, but the OS rejects it before any tool
		// code runs. Its launch error identifies the actual version-probe path.
		if err := os.WriteFile(filepath.Join(bin, "nvcc"+ext), nil, 0755); err != nil {
			t.Fatal(err)
		}
	}
	repo := filepath.Join(root, "repo")
	pkg := filepath.Join(repo, "internal", "compute")
	if err := os.MkdirAll(pkg, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(pkg, "cuda_arch.txt"), []byte("sm_80\nsm_89\nsm_90\nsm_100\nsm_120\n"), 0644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("CUDA_HOME", ambient)
	t.Setenv("CUDA_PATH", ambient)
	t.Setenv("PATH", filepath.Join(ambient, "bin"))
	// Keep Windows discovery from executing an installed Visual Studio tool.
	t.Setenv("ProgramFiles", root)
	t.Setenv("ProgramFiles(x86)", root)
	for _, tc := range []struct {
		name string
		home string
		nvcc string
	}{
		{name: "existing_selected_compiler", home: selected},
		{name: "missing_selected_compiler", home: filepath.Join(root, "missing CUDA")},
		{name: "explicit_nvcc_wins", home: selected, nvcc: filepath.Join(root, "explicit nvcc"+ext)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			args := []string{"build", "--json", "--arch", "sm_103", "--cuda-home", tc.home,
				"--repo-root", repo, "--receipt", filepath.Join(root, tc.name+".json")}
			wantNVCC := filepath.Join(tc.home, "bin", "nvcc"+ext)
			if tc.nvcc != "" {
				args = append(args, "--nvcc", tc.nvcc)
				wantNVCC = tc.nvcc
			}
			if code := RunBuildCUDA(&stdout, &stderr, args); code != 1 {
				t.Fatalf("exit = %d, want launch failure; stdout=%s stderr=%s", code, stdout.String(), stderr.String())
			}
			var result CUDABuildResult
			if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
				t.Fatalf("decode result: %v; stdout=%s", err, stdout.String())
			}
			if result.Success || !strings.Contains(result.Error, "nvcc --version failed") || !strings.Contains(result.Error, wantNVCC) {
				t.Fatalf("result = %+v, want version-probe launch failure at %q", result, wantNVCC)
			}
		})
	}
}
