package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRunRequiresOutputs(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if got := run(&stdout, &stderr, []string{"--config", "config.json"}); got != 2 || !strings.Contains(stderr.String(), "--report REPORT.json") {
		t.Fatalf("code=%d stderr=%q", got, stderr.String())
	}
}

func TestRunRejectsUnknownConfig(t *testing.T) {
	dir := t.TempDir()
	config := filepath.Join(dir, "config.json")
	if err := os.WriteFile(config, []byte(`{"unknown":true}`), 0o600); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	got := run(&stdout, &stderr, []string{"--config", config, "--report", filepath.Join(dir, "report.json"), "--archive", filepath.Join(dir, "archive.json")})
	if got != 1 || !strings.Contains(stderr.String(), "unknown field") {
		t.Fatalf("code=%d stderr=%q", got, stderr.String())
	}
}
