package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRunTrajectoryAuditPinnedCrossHarness(t *testing.T) {
	temp := t.TempDir()
	jsonlPath := filepath.Join(temp, "audit.jsonl")
	markdownPath := filepath.Join(temp, "audit.md")
	claudeRoot := filepath.Join("..", "..", "internal", "trajectory", "testdata", "audit", "claude", "projects")
	codexRoot := filepath.Join("..", "..", "internal", "trajectory", "testdata", "audit", "codex", "sessions")
	var stdout, stderr bytes.Buffer
	rc := runTrajectory(&stdout, &stderr, []string{
		"audit", "--since", "0", "--claude-root", claudeRoot, "--codex-root", codexRoot,
		"--jsonl", jsonlPath, "--md", markdownPath,
	})
	if rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, stderr.String())
	}
	jsonl, err := os.ReadFile(jsonlPath)
	if err != nil {
		t.Fatal(err)
	}
	markdown, err := os.ReadFile(markdownPath)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{`"schema":"fak-trajectory-audit/1"`, `"kind":"source_denominator"`, `"source":"claude"`, `"source":"codex"`, `"rank":1`} {
		if !bytes.Contains(jsonl, []byte(want)) {
			t.Fatalf("JSONL missing %q:\n%s", want, jsonl)
		}
	}
	if !bytes.Contains(markdown, []byte("Highest-cost bottleneck: claude/`claude-session`")) {
		t.Fatalf("markdown missing deterministic top row:\n%s", markdown)
	}
	if !strings.Contains(stderr.String(), "sessions=2 exact_usage=6 refused=0") {
		t.Fatalf("stderr = %q", stderr.String())
	}

	baselineJSONL := filepath.Join(temp, "with-baseline.jsonl")
	stderr.Reset()
	rc = runTrajectory(&stdout, &stderr, []string{
		"audit", "--since", "0", "--claude-root", claudeRoot, "--codex-root", codexRoot,
		"--baseline", jsonlPath, "--jsonl", baselineJSONL,
	})
	if rc != 0 {
		t.Fatalf("baseline rc=%d stderr=%s", rc, stderr.String())
	}
	withBaseline, err := os.ReadFile(baselineJSONL)
	if err != nil {
		t.Fatal(err)
	}
	if count := bytes.Count(withBaseline, []byte(`"kind":"baseline_delta"`)); count != 4 {
		t.Fatalf("baseline delta rows = %d, want 4:\n%s", count, withBaseline)
	}
}

func TestRunTrajectoryAuditUnsupportedShapeWritesArtifactAndFails(t *testing.T) {
	temp := t.TempDir()
	jsonlPath := filepath.Join(temp, "audit.jsonl")
	claudeRoot := filepath.Join("..", "..", "internal", "trajectory", "testdata", "audit", "unsupported", "claude", "projects")
	missingCodex := filepath.Join(temp, "missing-codex")
	var stdout, stderr bytes.Buffer
	rc := runTrajectory(&stdout, &stderr, []string{
		"audit", "--since", "0", "--claude-root", claudeRoot, "--codex-root", missingCodex, "--jsonl", jsonlPath,
	})
	if rc != 1 {
		t.Fatalf("rc=%d, want 1; stderr=%s", rc, stderr.String())
	}
	artifact, err := os.ReadFile(jsonlPath)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(artifact, []byte(`"kind":"refusal"`)) || !strings.Contains(stderr.String(), "TRAJECTORY_SCHEMA_REFUSED") {
		t.Fatalf("missing visible unsupported shape; stderr=%s artifact=%s", stderr.String(), artifact)
	}
}

func TestRunTrajectoryAuditRefusalMatrix(t *testing.T) {
	fixtureRoot := filepath.Join("..", "..", "internal", "trajectory", "testdata", "audit", "issue-8493", "refusals")
	missing := filepath.Join(t.TempDir(), "missing")
	tests := []struct {
		name       string
		claudeRoot string
		codexRoot  string
		code       string
	}{
		{"Claude duplicate mismatch", filepath.Join(fixtureRoot, "claude-duplicate", "projects"), missing, "claude_duplicate_usage_mismatch"},
		{"Codex cumulative decrease", missing, filepath.Join(fixtureRoot, "codex-decreasing", "sessions"), "codex_total_usage_decreased"},
		{"malformed JSON", filepath.Join(fixtureRoot, "malformed", "claude", "projects"), missing, "malformed_json"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			assertTrajectoryAuditRefusal(t, test.claudeRoot, test.codexRoot, test.code)
		})
	}

	t.Run("oversized line", func(t *testing.T) {
		root := t.TempDir()
		path := filepath.Join(root, "fak", "oversized.jsonl")
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		seed, err := os.ReadFile(filepath.Join(fixtureRoot, "oversized", "claude", "projects", "fak", "oversized.jsonl"))
		if err != nil {
			t.Fatal(err)
		}
		file, err := os.Create(path)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := file.Write(bytes.TrimSpace(seed)); err != nil {
			file.Close()
			t.Fatal(err)
		}
		chunk := bytes.Repeat([]byte("x"), 1024*1024)
		for written := len(seed); written <= 32*1024*1024; written += len(chunk) {
			if _, err := file.Write(chunk); err != nil {
				file.Close()
				t.Fatal(err)
			}
		}
		if err := file.Close(); err != nil {
			t.Fatal(err)
		}
		assertTrajectoryAuditRefusal(t, root, missing, "line_too_large")
	})
}

func assertTrajectoryAuditRefusal(t *testing.T, claudeRoot, codexRoot, code string) {
	t.Helper()
	jsonlPath := filepath.Join(t.TempDir(), "audit.jsonl")
	var stdout, stderr bytes.Buffer
	rc := runTrajectory(&stdout, &stderr, []string{
		"audit", "--since", "0", "--claude-root", claudeRoot, "--codex-root", codexRoot, "--jsonl", jsonlPath,
	})
	if rc != 1 {
		t.Fatalf("rc=%d, want 1; stderr=%s", rc, stderr.String())
	}
	artifact, err := os.ReadFile(jsonlPath)
	if err != nil {
		t.Fatalf("read refusal artifact: %v; stderr=%s", err, stderr.String())
	}
	for _, want := range []string{`"schema":"fak-trajectory-audit/1"`, `"kind":"refusal"`, `"code":"` + code + `"`} {
		if !bytes.Contains(artifact, []byte(want)) {
			t.Fatalf("artifact missing %q; stderr=%s artifact=%s", want, stderr.String(), artifact)
		}
	}
	if !strings.Contains(stderr.String(), "TRAJECTORY_SCHEMA_REFUSED") {
		t.Fatalf("missing refusal status; stderr=%s", stderr.String())
	}
}

func TestParseTrajectoryAuditSinceDays(t *testing.T) {
	duration, err := parseTrajectoryAuditSince("7d")
	if err != nil {
		t.Fatal(err)
	}
	if duration.Hours() != 168 {
		t.Fatalf("duration = %s", duration)
	}
}
