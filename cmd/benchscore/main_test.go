package main

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// binPath is the freshly compiled cmd/benchscore binary driven as a subprocess
// by every test below. Building it once in TestMain witnesses that the package
// compiles AND that the real executable runs, rather than asserting on an
// in-process imitation of main().
var binPath string

func TestMain(m *testing.M) {
	dir, err := os.MkdirTemp("", "benchscore-bin-")
	if err != nil {
		panic(err)
	}
	binPath = filepath.Join(dir, "benchscore.exe")
	build := exec.Command("go", "build", "-o", binPath, ".")
	if out, err := build.CombinedOutput(); err != nil {
		os.RemoveAll(dir)
		panic("go build ./cmd/benchscore failed: " + err.Error() + "\n" + string(out))
	}
	code := m.Run()
	os.RemoveAll(dir)
	os.Exit(code)
}

// runBin executes the compiled binary with args in dir, returning exit code,
// stdout, and stderr as separate strings. A non-zero exit is not an error.
func runBin(t *testing.T, dir string, args ...string) (int, string, string) {
	t.Helper()
	cmd := exec.Command(binPath, args...)
	cmd.Dir = dir
	var stdout, stderr strings.Builder
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	err := cmd.Run()
	code := 0
	if err != nil {
		exitErr, ok := err.(*exec.ExitError)
		if !ok {
			t.Fatalf("running %q: unexpected error %v", args, err)
		}
		code = exitErr.ExitCode()
	}
	return code, stdout.String(), stderr.String()
}

// writeScore drops a score.json under root/<subdir> with the given body.
func writeScore(t *testing.T, root, subdir, body string) {
	t.Helper()
	dir := filepath.Join(root, subdir)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "score.json"), []byte(body), 0o644); err != nil {
		t.Fatalf("write score.json: %v", err)
	}
}

func TestNoArgsOnMissingDefaultRootExitsNonZero(t *testing.T) {
	// Empty working directory => the default root cannot be walked; the
	// delegation must surface the error on stderr and exit non-zero rather
	// than silently printing an empty report as success.
	dir := t.TempDir()
	code, stdout, stderr := runBin(t, dir)
	if code == 0 {
		t.Fatalf("expected non-zero exit on missing default root, got 0; stdout=%q", stdout)
	}
	if stdout != "" {
		t.Errorf("expected empty stdout on hard error, got %q", stdout)
	}
	if !strings.Contains(stderr, "benchscore:") {
		t.Errorf("expected stderr to carry the benchscore: error prefix, got %q", stderr)
	}
}

func TestUnknownFlagPrintsUsageToStderr(t *testing.T) {
	dir := t.TempDir()
	code, stdout, stderr := runBin(t, dir, "-definitely-not-a-flag")
	if code == 0 {
		t.Fatalf("expected non-zero exit for unknown flag, got 0")
	}
	if stdout != "" {
		t.Errorf("expected empty stdout, got %q", stdout)
	}
	if !strings.Contains(stderr, "flag provided but not defined") {
		t.Errorf("expected flag-usage diagnostic on stderr, got %q", stderr)
	}
}

func TestCleanScanRendersMarkdownAndExitsZero(t *testing.T) {
	dir := t.TempDir()
	root := filepath.Join(dir, "runs")
	writeScore(t, root, "run-a", `{"schema":"fak.arm64-qkernel-score.v1"}`)

	code, stdout, stderr := runBin(t, dir, "-root", root)
	if code != 0 {
		t.Fatalf("expected exit 0 for warning-only scan, got %d; stderr=%q", code, stderr)
	}
	if !strings.Contains(stdout, "# Benchmark score matrix") {
		t.Errorf("expected markdown matrix header on stdout, got %q", stdout)
	}
	if !strings.Contains(stdout, "rows: 0") {
		t.Errorf("expected zero-row report, got %q", stdout)
	}
}

func TestJSONFlagEmitsParseableReport(t *testing.T) {
	dir := t.TempDir()
	root := filepath.Join(dir, "runs")
	writeScore(t, root, "run-a", `{"schema":"fak.arm64-qkernel-score.v1"}`)

	code, stdout, stderr := runBin(t, dir, "-root", root, "-json")
	if code != 0 {
		t.Fatalf("expected exit 0, got %d; stderr=%q", code, stderr)
	}
	var report struct {
		Schema string `json:"schema"`
		Root   string `json:"root"`
	}
	if err := json.Unmarshal([]byte(stdout), &report); err != nil {
		t.Fatalf("stdout is not valid JSON: %v; got %q", err, stdout)
	}
	if report.Schema != "fak.benchscore-report.v1" {
		t.Errorf("schema = %q, want fak.benchscore-report.v1", report.Schema)
	}
}

func TestMalformedScoreFileExitsOneWithIssue(t *testing.T) {
	dir := t.TempDir()
	root := filepath.Join(dir, "runs")
	writeScore(t, root, "run-bad", `{"schema": "fak.arm64-qkernel-score.v1",`) // truncated JSON

	code, stdout, stderr := runBin(t, dir, "-root", root)
	if code != 1 {
		t.Fatalf("expected exit 1 for a report carrying issues, got %d; stderr=%q", code, stderr)
	}
	if !strings.Contains(stdout, "## Issues") {
		t.Errorf("expected markdown issues section, got %q", stdout)
	}
}
