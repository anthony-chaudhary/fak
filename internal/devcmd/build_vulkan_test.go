package devcmd

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
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
}

func TestVulkanBuildProvenanceFlags(t *testing.T) {
	var stdout, stderr bytes.Buffer

	// Test passing -commit and -ref with invalid repo root in json mode
	code := RunBuildVulkan(&stdout, &stderr, []string{
		"binary",
		"-json",
		"-commit", "1234567890abcdef1234567890abcdef12345678",
		"-ref", "refs/heads/main",
		"-smoke=false",
		"-repo-root", "/nonexistent/repo",
		"-out-pkg", "./cmd/fake",
		"-out-bin", "bin/fake.exe",
	})

	if code != 1 {
		t.Fatalf("expected exit code 1, got %d", code)
	}

	var res VulkanBuildResult
	if err := json.Unmarshal(stdout.Bytes(), &res); err != nil {
		t.Fatalf("failed to decode JSON: %v", err)
	}
	if res.Command != "binary" {
		t.Errorf("expected command binary, got %s", res.Command)
	}
	if res.Success {
		t.Errorf("expected failure on nonexistent repo")
	}
}

func TestVulkanBuildBinaryDirtyRefusalCLI(t *testing.T) {
	// Create synthetic git repo
	dir := t.TempDir()

	git := func(args ...string) string {
		cmd := exec.Command("git", args...)
		cmd.Dir = dir
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("git %v failed: %v (%s)", args, err, string(out))
		}
		return strings.TrimSpace(string(out))
	}

	git("init")
	git("config", "user.name", "fak-test")
	git("config", "user.email", "fak-test@example.com")
	git("config", "commit.gpgsign", "false")

	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte("module fak-cli-test\n\ngo 1.26\n"), 0644); err != nil {
		t.Fatalf("write go.mod: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "README.md"), []byte("# CLI Test\n"), 0644); err != nil {
		t.Fatalf("write README.md: %v", err)
	}
	git("add", "go.mod", "README.md")
	git("commit", "-m", "init")

	// Make dirty
	if err := os.WriteFile(filepath.Join(dir, "README.md"), []byte("# Dirty modification\n"), 0644); err != nil {
		t.Fatalf("dirty write: %v", err)
	}

	var stdout, stderr bytes.Buffer
	code := RunBuildVulkan(&stdout, &stderr, []string{
		"binary",
		"-json",
		"-repo-root", dir,
		"-out-pkg", "./cmd/fake",
		"-out-bin", filepath.Join(dir, "bin", "fake.exe"),
		"-smoke=false",
	})

	if code != 1 {
		t.Fatalf("expected exit code 1 for dirty repo, got %d", code)
	}

	var res VulkanBuildResult
	if err := json.Unmarshal(stdout.Bytes(), &res); err != nil {
		t.Fatalf("failed decoding JSON output: %v\nOutput: %s", err, stdout.String())
	}
	if res.Success {
		t.Errorf("expected res.Success=false, got true")
	}
	if !strings.Contains(res.Error, "dirty") && !strings.Contains(res.Error, "uncommitted") {
		t.Errorf("expected dirty error message, got %q", res.Error)
	}
}

func TestVulkanBuildBinaryCommitMismatchProvenanceCLI(t *testing.T) {
	dir := t.TempDir()

	git := func(args ...string) string {
		cmd := exec.Command("git", args...)
		cmd.Dir = dir
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("git %v failed: %v (%s)", args, err, string(out))
		}
		return strings.TrimSpace(string(out))
	}

	git("init")
	git("config", "user.name", "fak-test")
	git("config", "user.email", "fak-test@example.com")
	git("config", "commit.gpgsign", "false")

	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte("module fak-cli-test\n\ngo 1.26\n"), 0644); err != nil {
		t.Fatalf("write go.mod: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "README.md"), []byte("# CLI Test\n"), 0644); err != nil {
		t.Fatalf("write README.md: %v", err)
	}
	git("add", "go.mod", "README.md")
	git("commit", "-m", "init")

	var stdout, stderr bytes.Buffer
	code := RunBuildVulkan(&stdout, &stderr, []string{
		"binary",
		"-json",
		"-repo-root", dir,
		"-commit", "0000000000000000000000000000000000000000",
		"-out-pkg", "./cmd/fake",
		"-out-bin", filepath.Join(dir, "bin", "fake.exe"),
		"-smoke=false",
	})

	if code != 1 {
		t.Fatalf("expected exit code 1 for commit mismatch, got %d", code)
	}

	var res VulkanBuildResult
	if err := json.Unmarshal(stdout.Bytes(), &res); err != nil {
		t.Fatalf("failed decoding JSON output: %v\nOutput: %s", err, stdout.String())
	}
	if res.Success {
		t.Errorf("expected res.Success=false, got true")
	}
	if !strings.Contains(res.Error, "commit") {
		t.Errorf("expected commit mismatch error message, got %q", res.Error)
	}
}
