package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// fak-test:runtime fast est=50ms
func TestRunTrajectoryAuditPiRoot(t *testing.T) {
	root := t.TempDir()
	t.Setenv("HOME", root)
	t.Setenv("USERPROFILE", root)
	t.Setenv("CLAUDE_CONFIG_DIR", filepath.Join(root, "claude"))
	t.Setenv("CODEX_HOME", filepath.Join(root, "codex"))
	t.Setenv("OPENCODE_HOME", filepath.Join(root, "opencode"))
	t.Setenv("XDG_DATA_HOME", filepath.Join(root, "data"))

	fixtureRoot := filepath.Join("..", "..", "internal", "trajectory", "testdata", "audit", "pi", "sessions")
	if _, err := os.Stat(filepath.Join(fixtureRoot, "2026-11-01", "fixture-pi-session.jsonl")); err != nil {
		t.Skipf("pi fixture not present: %v", err)
	}
	jsonlPath := filepath.Join(root, "report.jsonl")
	markdownPath := filepath.Join(root, "report.md")
	var stdout, stderr bytes.Buffer
	rc := runTrajectory(&stdout, &stderr, []string{"audit", "--since", "0", "--pi-root", fixtureRoot, "--jsonl", jsonlPath, "--md", markdownPath})
	if rc != 0 {
		t.Fatalf("exit=%d, want 0; stderr=%s", rc, stderr.String())
	}
	markdown, err := os.ReadFile(markdownPath)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(stderr.String(), "trajectory audit: sessions=1 exact_usage=8 refused=0 schema_drift=0 schema_breaking=0") {
		t.Fatalf("summary line missing or wrong: %s", stderr.String())
	}
	for _, want := range []string{
		"| pi | `pi/sessions` | true | 1/0/0/1 |",
		"per-assistant-turn usage buckets are disjoint",
	} {
		if !strings.Contains(string(markdown), want) {
			t.Fatalf("markdown missing %q:\n%s", want, string(markdown))
		}
	}
}

// fak-test:runtime fast est=20ms
func TestRunTrajectoryAuditPiRootAbsent(t *testing.T) {
	root := t.TempDir()
	t.Setenv("HOME", root)
	t.Setenv("USERPROFILE", root)
	t.Setenv("CLAUDE_CONFIG_DIR", filepath.Join(root, "claude"))
	t.Setenv("CODEX_HOME", filepath.Join(root, "codex"))
	t.Setenv("OPENCODE_HOME", filepath.Join(root, "opencode"))
	t.Setenv("XDG_DATA_HOME", filepath.Join(root, "data"))

	missing := filepath.Join(root, "no-such-pi-root")
	var stdout, stderr bytes.Buffer
	rc := runTrajectory(&stdout, &stderr, []string{"audit", "--since", "0", "--pi-root", missing})
	if rc != 0 {
		t.Fatalf("exit=%d, want 0; stderr=%s", rc, stderr.String())
	}
	out := stdout.String()
	// An absent root reports a present=false denominator with zero coverage,
	// exactly like the other optional harness roots.
	if !strings.Contains(out, "| pi | `pi/sessions` | false | 0/0/0/0 |") {
		t.Fatalf("absent pi root did not report an empty denominator row:\n%s", out)
	}
	if !strings.Contains(stderr.String(), "refused=0") {
		t.Fatalf("unexpected refusals: %s", stderr.String())
	}
}
