package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestSessionObserveZeroConfigurationUsesCurrentWorkspace(t *testing.T) {
	root := t.TempDir()
	fixture, err := os.ReadFile(filepath.Join("..", "..", "internal", "session", "testdata", "compactaudit", "healthy-repeated-fire.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	rollout := filepath.Join(root, "rollout.jsonl")
	if err := os.WriteFile(rollout, fixture, 0o600); err != nil {
		t.Fatal(err)
	}
	now := func() time.Time { return time.Date(2026, 6, 20, 12, 0, 0, 0, time.Local) }
	getwd := func() (string, error) { return `C:\work\fak`, nil }
	var stdout, stderr bytes.Buffer
	if rc := runSessionObserveAt(&stdout, &stderr, []string{"--root", root}, now, getwd); rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, stderr.String())
	}
	for _, want := range []string{"Codex context — WORKING", "calendar days", `C:\work\fak`, "daily (UTC)", "resident shed"} {
		if !strings.Contains(stdout.String(), want) {
			t.Fatalf("output missing %q:\n%s", want, stdout.String())
		}
	}
}

func TestSessionObserveJSONIsAggregateAndScrubbed(t *testing.T) {
	root := t.TempDir()
	fixture, err := os.ReadFile(filepath.Join("..", "..", "internal", "session", "testdata", "compactaudit", "healthy-repeated-fire.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "rollout.jsonl"), fixture, 0o600); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	now := func() time.Time { return time.Date(2026, 6, 20, 12, 0, 0, 0, time.Local) }
	if rc := runSessionObserveAt(&stdout, &stderr, []string{"--root", root, "--cwd", `C:\work\fak`, "--json"}, now, os.Getwd); rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, stderr.String())
	}
	got := stdout.String()
	if !strings.Contains(got, `"measured_fires"`) || strings.Contains(got, root) || strings.Contains(got, `"sessions": [`) {
		t.Fatalf("unexpected JSON:\n%s", got)
	}
}
