package main

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// binPath is the real cmd/fak-selfupdate binary compiled once in TestMain and then
// driven as a subprocess. Running the shipped binary (rather than calling an
// in-process helper) is what witnesses the delegation contract in main.go:
// `func main() { selfupdatecmd.Run(os.Args[1:]) }`.
var binPath string

func TestMain(m *testing.M) {
	dir, err := os.MkdirTemp("", "fak-selfupdate-bin-")
	if err != nil {
		panic(err)
	}
	suffix := ""
	if runtime.GOOS == "windows" {
		suffix = ".exe"
	}
	binPath = filepath.Join(dir, "fak-selfupdate"+suffix)
	build := exec.Command("go", "build", "-o", binPath, ".")
	build.Stdout = os.Stderr
	build.Stderr = os.Stderr
	if err := build.Run(); err != nil {
		os.RemoveAll(dir)
		panic("build cmd/fak-selfupdate: " + err.Error())
	}
	code := m.Run()
	os.RemoveAll(dir)
	os.Exit(code)
}

// runBin executes the compiled binary with args and returns (exitCode, stdout, stderr).
func runBin(t *testing.T, args ...string) (int, string, string) {
	t.Helper()
	if binPath == "" {
		t.Fatal("binPath is empty; TestMain did not build the binary")
	}
	cmd := exec.Command(binPath, args...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	err := cmd.Run()
	code := 0
	if err != nil {
		var exitErr *exec.ExitError
		if ok := asExitError(err, &exitErr); ok {
			code = exitErr.ExitCode()
		} else {
			t.Fatalf("run %q: %v", args, err)
		}
	}
	return code, stdout.String(), stderr.String()
}

func asExitError(err error, target **exec.ExitError) bool {
	if e, ok := err.(*exec.ExitError); ok {
		*target = e
		return true
	}
	return false
}

// TestUnknownFlagIsRefusedWithUsage witnesses that os.Args reaches the shared
// flag parser: an undefined flag is refused, the usage banner is written to
// STDERR, stdout stays clean, and the process exits non-zero (flag.ExitOnError
// uses code 2). This fails if main.go stops forwarding os.Args or if the
// delegate swallows the parse error.
func TestUnknownFlagIsRefusedWithUsage(t *testing.T) {
	code, stdout, stderr := runBin(t, "--definitely-not-a-selfupdate-flag")
	if code == 0 {
		t.Fatalf("expected non-zero exit for unknown flag, got 0\nstderr:\n%s", stderr)
	}
	if !strings.Contains(stderr, "flag provided but not defined") {
		t.Fatalf("stderr missing flag-parse diagnostic; got:\n%s", stderr)
	}
	if !strings.Contains(stderr, "Usage of self-update:") {
		t.Fatalf("stderr missing usage banner; got:\n%s", stderr)
	}
	if stdout != "" {
		t.Fatalf("usage must go to stderr, stdout must stay empty; got stdout:\n%q", stdout)
	}
}

// TestApplyWithoutBuildGCIsRefused witnesses that the delegated flag validation
// runs and reports on stderr, and that stdout is not polluted by a diagnostic.
func TestApplyWithoutBuildGCIsRefused(t *testing.T) {
	_, stdout, stderr := runBin(t, "--apply")
	if !strings.Contains(stderr, "--apply requires --build-gc") {
		t.Fatalf("stderr missing --apply/--build-gc diagnostic; got:\n%s", stderr)
	}
	if stdout != "" {
		t.Fatalf("refusal must not write stdout; got %q", stdout)
	}
}

// TestMutuallyExclusiveFlagsAreRefused witnesses a second delegated validation
// branch (--build-gc vs --check) end-to-end through the real binary.
func TestMutuallyExclusiveFlagsAreRefused(t *testing.T) {
	_, stdout, stderr := runBin(t, "--build-gc", "--check")
	if !strings.Contains(stderr, "--build-gc and --check are mutually exclusive") {
		t.Fatalf("stderr missing mutual-exclusion diagnostic; got:\n%s", stderr)
	}
	if stdout != "" {
		t.Fatalf("refusal must not write stdout; got %q", stdout)
	}
}

// TestBuildGCJSONIsSuccessfulDelegation witnesses the positive path: a
// deterministic, offline subcommand (build-worktree GC plan) runs to completion
// in the shipped binary, exits 0, stays silent on stderr, and emits exactly one
// JSON receipt of the documented schema on stdout. This is the runtime proof
// that the binary actually executes and delegates for real.
func TestBuildGCJSONIsSuccessfulDelegation(t *testing.T) {
	root := t.TempDir()
	code, stdout, stderr := runBin(t, "--build-gc", "--json", "--root", root)
	if code != 0 {
		t.Fatalf("expected exit 0 from --build-gc --json, got %d\nstderr:\n%s", code, stderr)
	}
	if strings.TrimSpace(stderr) != "" {
		t.Fatalf("expected empty stderr on success, got:\n%s", stderr)
	}
	var receipt struct {
		Schema string `json:"schema"`
		Mode   string `json:"mode"`
	}
	if err := json.Unmarshal([]byte(stdout), &receipt); err != nil {
		t.Fatalf("stdout is not a single JSON receipt: %v\nstdout:\n%s", err, stdout)
	}
	if receipt.Schema != "fak.self-update.build-gc/v1" {
		t.Fatalf("receipt schema = %q, want fak.self-update.build-gc/v1", receipt.Schema)
	}
	if receipt.Mode != "plan" {
		t.Fatalf("receipt mode = %q, want plan (dry-run default)", receipt.Mode)
	}
}
