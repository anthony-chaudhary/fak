package devcmd

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/computebuild"
)

// fak-test:runtime fast est=10ms
func TestCUDABuildBinarySmoke(t *testing.T) {
	for _, tc := range []struct {
		name      string
		flags     []string
		wantSmoke bool
	}{
		{name: "default", wantSmoke: true},
		{name: "explicit-on", flags: []string{"--smoke=true"}, wantSmoke: true},
		{name: "explicit-off", flags: []string{"--smoke=false"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			root := t.TempDir()
			if err := os.WriteFile(filepath.Join(root, "go.mod"), []byte("module cuda-smoke-test\n"), 0600); err != nil {
				t.Fatal(err)
			}
			args := append([]string{"binary", "--repo-root", root}, tc.flags...)
			var stdout, stderr bytes.Buffer
			called := false
			code := runBuildCUDA(&stdout, &stderr, args, func(ctx context.Context, cfg *computebuild.CUDAConfig) error {
				called = true
				cfg.Toolchain = &computebuild.Toolchain{}
				// Missing binary arguments stop the real orchestrator after defaults, before any tool execution.
				err := computebuild.RunCUDA(ctx, cfg)
				if err == nil || cfg.Receipt == nil {
					t.Fatal("expected orchestrator binary validation failure with receipt")
				}
				if cfg.Smoke != tc.wantSmoke {
					t.Errorf("orchestrator Smoke = %v, want %v", cfg.Smoke, tc.wantSmoke)
				}
				return err
			})
			if !called || code != 1 {
				t.Fatalf("runner called = %v, exit code = %d; want true, 1", called, code)
			}
		})
	}
}

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
