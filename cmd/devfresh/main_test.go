package main

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// binPath is the freshly built cmd/devfresh binary under test. It is compiled
// once in TestMain so every case below drives the REAL binary as a subprocess
// and asserts observable behaviour (exit code, stdout, stderr) rather than
// calling internal helpers. That is what witnesses missing_tests and
// unproven_runtime together, and it cannot pass when the binary is deleted.
var binPath string

func TestMain(m *testing.M) {
	dir, err := os.MkdirTemp("", "devfresh-bin-")
	if err != nil {
		panic(err)
	}
	defer os.RemoveAll(dir)
	binPath = filepath.Join(dir, "devfresh")
	if isWindows() {
		binPath += ".exe"
	}
	build := exec.Command("go", "build", "-o", binPath, ".")
	build.Stderr = os.Stderr
	if err := build.Run(); err != nil {
		panic("go build cmd/devfresh: " + err.Error())
	}
	os.Exit(m.Run())
}

func isWindows() bool {
	return os.PathSeparator == '\\'
}

// runBin executes the built binary with args and returns exit code, stdout,
// and stderr. A non-zero exit is data, not a test failure.
func runBin(t *testing.T, args ...string) (int, string, string) {
	t.Helper()
	cmd := exec.Command(binPath, args...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	err := cmd.Run()
	code := 0
	if err != nil {
		exitErr, ok := err.(*exec.ExitError)
		if !ok {
			t.Fatalf("running %s %v: %v", binPath, args, err)
		}
		code = exitErr.ExitCode()
	}
	return code, stdout.String(), stderr.String()
}

// TestNoArgsPrintsUsageToStderr: the delegation contract requires an empty
// invocation to print the usage banner on STDERR (stdout stays empty) and
// exit non-zero. Asserting stdout is empty is what makes this non-vacuous.
func TestNoArgsPrintsUsageToStderr(t *testing.T) {
	code, stdout, stderr := runBin(t)
	if code == 0 {
		t.Fatalf("no args: exit code = 0, want non-zero")
	}
	if stdout != "" {
		t.Errorf("no args: stdout = %q, want empty", stdout)
	}
	if !strings.Contains(stderr, "usage: devfresh") {
		t.Errorf("no args: stderr = %q, want usage banner", stderr)
	}
}

// TestUnknownFlagIsRejected: flag.ExitOnError must reject an unknown flag with
// a non-zero exit and a message on stderr, never a silent success.
func TestUnknownFlagIsRejected(t *testing.T) {
	code, stdout, stderr := runBin(t, "--definitely-not-a-flag")
	if code == 0 {
		t.Fatalf("unknown flag: exit code = 0, want non-zero")
	}
	if stdout != "" {
		t.Errorf("unknown flag: stdout = %q, want empty", stdout)
	}
	if stderr == "" {
		t.Errorf("unknown flag: stderr empty, want a diagnostic")
	}
}

// TestSelfcheckPasses drives the built-in witness and requires its exact
// success line on stdout.
func TestSelfcheckPasses(t *testing.T) {
	code, stdout, stderr := runBin(t, "--selfcheck")
	if code != 0 {
		t.Fatalf("--selfcheck: exit code = %d (stderr=%q), want 0", code, stderr)
	}
	if !strings.Contains(stdout, "devfresh selfcheck: PASS") {
		t.Errorf("--selfcheck: stdout = %q, want PASS line", stdout)
	}
}

// TestCleanFileExitsZero: a file with no freshness defects must exit 0 and
// print nothing. This proves the "successful delegation" path, not just errors.
func TestCleanFileExitsZero(t *testing.T) {
	path := filepath.Join(t.TempDir(), "clean.md")
	if err := os.WriteFile(path, []byte("Nothing version-dependent here.\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	code, stdout, stderr := runBin(t, path)
	if code != 0 {
		t.Fatalf("clean file: exit code = %d (stderr=%q), want 0", code, stderr)
	}
	if stdout != "" {
		t.Errorf("clean file: stdout = %q, want empty", stdout)
	}
}

// TestUnpointedClaimFailsAndJSON: a document asserting a version-dependent
// fact without a freshness pointer must be reported and exit 1. The JSON form
// must round-trip to a non-empty findings slice, so the assertion binds to the
// engine's real output shape rather than a substring.
func TestUnpointedClaimFailsAndJSON(t *testing.T) {
	path := filepath.Join(t.TempDir(), "stale.md")
	if err := os.WriteFile(path, []byte("The latest Codex release is available.\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	code, stdout, stderr := runBin(t, "--json", path)
	if code != 1 {
		t.Fatalf("unpointed claim: exit code = %d (stderr=%q), want 1", code, stderr)
	}
	var findings []map[string]any
	if err := json.Unmarshal([]byte(stdout), &findings); err != nil {
		t.Fatalf("unpointed claim: stdout not JSON: %v (%q)", err, stdout)
	}
	if len(findings) == 0 {
		t.Fatalf("unpointed claim: no findings reported in %q", stdout)
	}
}

// TestMissingFileExitsTwo: the engine's read-error class maps to exit 2 with a
// message on stderr, and never a successful exit.
func TestMissingFileExitsTwo(t *testing.T) {
	missing := filepath.Join(t.TempDir(), "does-not-exist.md")
	code, stdout, stderr := runBin(t, missing)
	if code != 2 {
		t.Fatalf("missing file: exit code = %d (stderr=%q), want 2", code, stderr)
	}
	if stdout != "" {
		t.Errorf("missing file: stdout = %q, want empty", stdout)
	}
	if !strings.Contains(stderr, "devfresh:") {
		t.Errorf("missing file: stderr = %q, want a devfresh diagnostic", stderr)
	}
}
