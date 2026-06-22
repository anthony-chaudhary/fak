package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestListTranscriptsDiscovers verifies `fak debug --list` finds a real-shaped
// transcript under ~/.claude*/projects/<ns>/<uuid>.jsonl and prints the exact
// attach command — the answer to "fak debug ran a demo, where is MY session?".
func TestListTranscriptsDiscovers(t *testing.T) {
	tmp := t.TempDir()
	t.Setenv("HOME", tmp)        // os.UserHomeDir on Linux/WSL (where the suite runs)
	t.Setenv("USERPROFILE", tmp) // os.UserHomeDir on Windows
	t.Setenv("CLAUDE_CONFIG_DIR", "")

	dir := filepath.Join(tmp, ".claude-variant", "projects", "C--work-proj")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(dir, "sess-abc123.jsonl")
	if err := os.WriteFile(p, []byte("{}\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	var buf bytes.Buffer
	listTranscripts(&buf)
	out := buf.String()
	if !strings.Contains(out, "sess-abc123.jsonl") {
		t.Errorf("listing did not surface the transcript:\n%s", out)
	}
	if !strings.Contains(out, "fak debug --session") {
		t.Errorf("listing did not print the attach command:\n%s", out)
	}
}

// TestListTranscriptsEmpty verifies the no-transcripts case is a helpful message,
// not a silent nothing.
func TestListTranscriptsEmpty(t *testing.T) {
	tmp := t.TempDir()
	t.Setenv("HOME", tmp)
	t.Setenv("USERPROFILE", tmp)
	t.Setenv("CLAUDE_CONFIG_DIR", "")

	var buf bytes.Buffer
	listTranscripts(&buf)
	if !strings.Contains(buf.String(), "no Claude Code transcripts found") {
		t.Errorf("want a not-found hint, got:\n%s", buf.String())
	}
}
