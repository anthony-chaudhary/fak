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

// binPath is the freshly compiled cmd/agenticbench binary. Building it once in
// TestMain and driving it as a subprocess is the only way to witness the real
// runtime behaviour of a package that is purely a func main().
var binPath string

func TestMain(m *testing.M) {
	dir, err := os.MkdirTemp("", "agenticbench-bin")
	if err != nil {
		os.Stderr.WriteString("mkdirtemp: " + err.Error() + "\n")
		os.Exit(1)
	}
	binPath = filepath.Join(dir, "agenticbench.exe")
	build := exec.Command("go", "build", "-o", binPath, ".")
	build.Stderr = os.Stderr
	if err := build.Run(); err != nil {
		os.Stderr.WriteString("go build: " + err.Error() + "\n")
		os.Exit(1)
	}
	code := m.Run()
	os.RemoveAll(dir)
	os.Exit(code)
}

// runBin executes the compiled binary and returns exit code, stdout, stderr.
func runBin(t *testing.T, args ...string) (int, string, string) {
	t.Helper()
	cmd := exec.Command(binPath, args...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	err := cmd.Run()
	code := 0
	if err != nil {
		ee, ok := err.(*exec.ExitError)
		if !ok {
			t.Fatalf("run %v: %v", args, err)
		}
		code = ee.ExitCode()
	}
	return code, stdout.String(), stderr.String()
}

// emptyRoot creates a temp dir with no benchmark artifacts, so the rollup is
// deterministic: every child is MISSING and no result claim is allowed.
func emptyRoot(t *testing.T) string {
	t.Helper()
	return t.TempDir()
}

// TestRollupOnEmptyRootWritesJSON is the positive delegation witness: with a
// real -root the binary runs the engine, prints the epic rollup to stdout, and
// reports its human summary on stderr.
func TestRollupOnEmptyRootWritesJSON(t *testing.T) {
	code, stdout, stderr := runBin(t, "-root", emptyRoot(t))
	if code != 0 {
		t.Fatalf("exit = %d, want 0\nstderr:\n%s", code, stderr)
	}
	var report struct {
		Schema string `json:"schema"`
		Epic   int    `json:"epic"`
		Status string `json:"status"`
	}
	if err := json.Unmarshal([]byte(stdout), &report); err != nil {
		t.Fatalf("stdout is not valid JSON: %v\nstdout:\n%s", err, stdout)
	}
	if report.Schema != "fak.agentic-benchmark-epic-rollup.v1" {
		t.Errorf("schema = %q, want fak.agentic-benchmark-epic-rollup.v1", report.Schema)
	}
	if report.Epic != 868 {
		t.Errorf("epic = %d, want 868", report.Epic)
	}
	if !strings.Contains(stderr, "== agenticbench ==") {
		t.Errorf("stderr missing summary banner, got:\n%s", stderr)
	}
	if !strings.Contains(stderr, "epic         : #868") {
		t.Errorf("stderr missing epic line, got:\n%s", stderr)
	}
}

// TestUnknownFlagIsUsageError pins the flag-parsing contract: an unrecognized
// flag exits 2, writes nothing to stdout, and prints usage to stderr.
func TestUnknownFlagIsUsageError(t *testing.T) {
	code, stdout, stderr := runBin(t, "-definitely-not-a-flag")
	if code != 2 {
		t.Fatalf("exit = %d, want 2", code)
	}
	if stdout != "" {
		t.Errorf("stdout = %q, want empty", stdout)
	}
	if !strings.Contains(stderr, "flag provided but not defined") {
		t.Errorf("stderr missing flag error, got:\n%s", stderr)
	}
	if !strings.Contains(stderr, "Usage of") {
		t.Errorf("stderr missing usage block, got:\n%s", stderr)
	}
}

// TestStrictFailsWhenGateIncomplete witnesses the -strict exit path: the epic
// gate cannot be complete on an empty root, so -strict must exit 2 after still
// emitting the rollup to stdout.
func TestStrictFailsWhenGateIncomplete(t *testing.T) {
	code, stdout, _ := runBin(t, "-root", emptyRoot(t), "-strict")
	if code != 2 {
		t.Fatalf("exit = %d, want 2", code)
	}
	var report struct {
		ResultClaimAllowed bool `json:"result_claim_allowed"`
	}
	if err := json.Unmarshal([]byte(stdout), &report); err != nil {
		t.Fatalf("stdout is not valid JSON: %v\nstdout:\n%s", err, stdout)
	}
	if report.ResultClaimAllowed {
		t.Errorf("result_claim_allowed = true on empty root, want false")
	}
}

// TestWriteFailureExitsOne witnesses the fatal() path: an -out target that is a
// directory cannot be written, so the engine error must surface on stderr with
// exit 1 and nothing on stdout.
func TestWriteFailureExitsOne(t *testing.T) {
	root := emptyRoot(t)
	code, stdout, stderr := runBin(t, "-root", root, "-out", root)
	if code != 1 {
		t.Fatalf("exit = %d, want 1", code)
	}
	if stdout != "" {
		t.Errorf("stdout = %q, want empty", stdout)
	}
	if !strings.Contains(stderr, "agenticbench: ") {
		t.Errorf("stderr missing fatal prefix, got:\n%s", stderr)
	}
}

// TestOutAndMDFilesAreWritten witnesses the file-output delegation: when -out
// and -md name writable paths the rollup is redirected to files and stdout is
// left empty.
func TestOutAndMDFilesAreWritten(t *testing.T) {
	root := emptyRoot(t)
	jsonPath := filepath.Join(t.TempDir(), "rollup.json")
	mdPath := filepath.Join(t.TempDir(), "rollup.md")
	code, stdout, stderr := runBin(t, "-root", root, "-out", jsonPath, "-md", mdPath)
	if code != 0 {
		t.Fatalf("exit = %d, want 0\nstderr:\n%s", code, stderr)
	}
	if stdout != "" {
		t.Errorf("stdout = %q, want empty when -out is set", stdout)
	}
	jb, err := os.ReadFile(jsonPath)
	if err != nil {
		t.Fatalf("read -out file: %v", err)
	}
	if !strings.Contains(string(jb), "fak.agentic-benchmark-epic-rollup.v1") {
		t.Errorf("-out file missing rollup schema:\n%s", jb)
	}
	mb, err := os.ReadFile(mdPath)
	if err != nil {
		t.Fatalf("read -md file: %v", err)
	}
	if len(mb) == 0 {
		t.Errorf("-md file is empty")
	}
}
