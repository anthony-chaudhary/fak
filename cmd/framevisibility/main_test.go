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

// binPath is the real compiled cmd/framevisibility binary, built once in
// TestMain. Every case drives it as a subprocess so the tests witness the
// binary's observable behaviour (exit code, stdout, stderr), not just the
// internal Run function.
var binPath string

func TestMain(m *testing.M) {
	dir, err := os.MkdirTemp("", "framevisibility-bin")
	if err != nil {
		panic(err)
	}
	binPath = filepath.Join(dir, "framevisibility.exe")
	build := exec.Command("go", "build", "-o", binPath, ".")
	build.Stderr = os.Stderr
	if err := build.Run(); err != nil {
		panic("build failed: " + err.Error())
	}
	code := m.Run()
	os.RemoveAll(dir)
	os.Exit(code)
}

func runBin(t *testing.T, args ...string) (int, string, string) {
	t.Helper()
	var stdout, stderr bytes.Buffer
	cmd := exec.Command(binPath, args...)
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

// TestUnknownFlagExitsTwoAndKeepsStdoutClean proves the binary forwards
// os.Args to the delegate and maps a flag-parse failure to exit code 2 with
// diagnostics on stderr only.
func TestUnknownFlagExitsTwoAndKeepsStdoutClean(t *testing.T) {
	code, stdout, stderr := runBin(t, "-unrecognized-flag")
	if code != 2 {
		t.Fatalf("exit code = %d, want 2 (stderr: %q)", code, stderr)
	}
	if stdout != "" {
		t.Fatalf("stdout = %q, want empty on a flag error", stdout)
	}
	if !strings.Contains(stderr, "unrecognized-flag") {
		t.Fatalf("stderr = %q, want it to name the bad flag", stderr)
	}
}

// TestMissingHomesExitsOne proves the delegate's fold error is surfaced and
// mapped to exit code 1.
func TestMissingHomesExitsOne(t *testing.T) {
	missing := filepath.Join(t.TempDir(), "does-not-exist")
	code, stdout, stderr := runBin(t, "-homes", missing)
	if code != 1 {
		t.Fatalf("exit code = %d, want 1 (stderr: %q)", code, stderr)
	}
	if stdout != "" {
		t.Fatalf("stdout = %q, want empty on a fold error", stdout)
	}
	if !strings.Contains(stderr, "fold:") {
		t.Fatalf("stderr = %q, want the delegate's fold: diagnostic", stderr)
	}
}

// TestSuccessfulDelegationFoldsSyntheticCorpus drives the real binary against
// a synthetic transcript tree and asserts the observable fold output and the
// written JSON report. This is the successful-delegation witness.
func TestSuccessfulDelegationFoldsSyntheticCorpus(t *testing.T) {
	root := t.TempDir()
	project := "C--work-fak"
	session := "session-1"
	dir := filepath.Join(root, ".claude-test", "projects", project, session)
	if err := os.MkdirAll(filepath.Join(dir, "subagents"), 0o755); err != nil {
		t.Fatal(err)
	}
	master := filepath.Join(filepath.Dir(dir), session+".jsonl")
	if err := os.WriteFile(master, []byte(`{"type":"assistant","message":{"content":[{"type":"text","text":"done"}]}}`+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "subagents", "agent-a.jsonl"), []byte(`{"type":"assistant","message":{"content":[{"type":"text","text":"done"}]}}`+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	outFile := filepath.Join(root, "out.json")

	code, stdout, stderr := runBin(t, "-homes", root, "-project", project, "-out", outFile)
	if code != 0 {
		t.Fatalf("exit code = %d, want 0 (stderr: %q)", code, stderr)
	}
	if !strings.Contains(stdout, "homes=1 sessions=1") {
		t.Fatalf("stdout = %q, want the folded homes/sessions summary", stdout)
	}
	if !strings.Contains(stdout, "wrote "+outFile) {
		t.Fatalf("stdout = %q, want it to name the written report", stdout)
	}

	blob, err := os.ReadFile(outFile)
	if err != nil {
		t.Fatalf("read report %s: %v", outFile, err)
	}
	var report struct {
		Totals struct {
			Homes    int `json:"homes"`
			Sessions int `json:"sessions"`
		} `json:"totals"`
	}
	if err := json.Unmarshal(blob, &report); err != nil {
		t.Fatalf("report is not valid JSON: %v", err)
	}
	if report.Totals.Homes != 1 || report.Totals.Sessions != 1 {
		t.Fatalf("report totals = %+v, want 1 home / 1 session", report.Totals)
	}
}

// TestNoQualifyingCorpusExitsZero proves that a valid-but-empty corpus is a
// successful fold, not an error: the binary still writes its report.
func TestNoQualifyingCorpusExitsZero(t *testing.T) {
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, ".claude-empty", "projects", "C--work-fak"), 0o755); err != nil {
		t.Fatal(err)
	}
	outFile := filepath.Join(root, "out.json")

	code, stdout, stderr := runBin(t, "-homes", root, "-project", "C--work-fak", "-out", outFile)
	if code != 0 {
		t.Fatalf("exit code = %d, want 0 (stderr: %q)", code, stderr)
	}
	if !strings.Contains(stdout, "sessions=0") {
		t.Fatalf("stdout = %q, want a zero-session summary", stdout)
	}
	if _, err := os.Stat(outFile); err != nil {
		t.Fatalf("expected report %s: %v", outFile, err)
	}
}
