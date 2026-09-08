package devcmd

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestVulkanBuildHelpAndInvalid(t *testing.T) {
	var stdout, stderr bytes.Buffer

	// 1. Help flag should return exit code 0
	code := RunBuildVulkan(&stdout, &stderr, []string{"--help"})
	if code != 0 {
		t.Fatalf("expected 0 for --help, got %d", code)
	}
	if !strings.Contains(stderr.String(), "usage: fak-dev build-vulkan") {
		t.Fatalf("expected usage message, got %q", stderr.String())
	}

	// 2. Empty args should return exit code 2
	stdout.Reset()
	stderr.Reset()
	code = RunBuildVulkan(&stdout, &stderr, []string{})
	if code != 2 {
		t.Fatalf("expected 2 for empty args, got %d", code)
	}

	// 3. Unknown subcommand should return exit code 2
	stdout.Reset()
	stderr.Reset()
	code = RunBuildVulkan(&stdout, &stderr, []string{"nonexistent"})
	if code != 2 {
		t.Fatalf("expected 2 for unknown subcommand, got %d", code)
	}
	if !strings.Contains(stderr.String(), "unknown subcommand") {
		t.Fatalf("expected unknown subcommand error, got %q", stderr.String())
	}
}

func TestVulkanBuildFlagParsingAndJSON(t *testing.T) {
	var stdout, stderr bytes.Buffer

	// Test with invalid repo root and json flag
	code := RunBuildVulkan(&stdout, &stderr, []string{
		"lib",
		"-json",
		"-cxx", "custom-g++",
		"-ar", "custom-ar",
		"-repo-root", "/nonexistent/repo",
	})

	// Since /nonexistent/repo doesn't have vulkan_shim.cpp, it should fail with code 1
	if code != 1 {
		t.Fatalf("expected exit code 1 on failed build, got %d", code)
	}

	var res VulkanBuildResult
	if err := json.Unmarshal(stdout.Bytes(), &res); err != nil {
		t.Fatalf("failed to decode JSON output: %v\nstdout: %s", err, stdout.String())
	}

	if res.Schema != "fak.vulkan-build.v1" {
		t.Errorf("expected schema fak.vulkan-build.v1, got %q", res.Schema)
	}
	if res.Command != "lib" {
		t.Errorf("expected command lib, got %q", res.Command)
	}
	if res.Success {
		t.Errorf("expected success false, got true")
	}
	if res.Error == "" {
		t.Errorf("expected non-empty error message in JSON")
	}
}

func TestVulkanBuildBinaryPositionalArgs(t *testing.T) {
	var stdout, stderr bytes.Buffer

	// Test "binary" with positional args in json mode
	code := RunBuildVulkan(&stdout, &stderr, []string{
		"binary",
		"-json",
		"-smoke=false",
		"-compare-receipt", "/nonexistent/prior-receipt.json",
		"-git-commit", strings.Repeat("0", 40),
		"-repo-root", "/nonexistent/repo",
		"./cmd/fak",
		"bin/fak.exe",
	})

	if code != 1 {
		t.Fatalf("expected exit code 1, got %d", code)
	}

	var res VulkanBuildResult
	if err := json.Unmarshal(stdout.Bytes(), &res); err != nil {
		t.Fatalf("failed to decode JSON output: %v", err)
	}
	if res.Command != "binary" {
		t.Fatalf("expected command binary, got %s", res.Command)
	}
	if res.ReceiptSchema != "" || res.StableIdentitySHA256 != "" || res.ReproducibilityStatus != "" {
		t.Fatalf("failed CLI result retained success-only provenance: %+v", res)
	}
	if res.GitRef != "HEAD" || res.ReceiptPath == "" {
		t.Fatalf("failed CLI result lost compatibility routing fields: %+v", res)
	}
}
