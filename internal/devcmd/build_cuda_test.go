package devcmd

import (
	"bytes"
	"encoding/json"
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
