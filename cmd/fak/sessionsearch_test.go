package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRunSessionSearch(t *testing.T) {
	t.Run("json empty results array", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		rc := runSessionSearch(&stdout, &stderr, []string{"--query", "fak", "--json"})
		if rc != 0 {
			t.Fatalf("expected rc 0, got %d, stderr: %s", rc, stderr.String())
		}
		out := strings.TrimSpace(stdout.String())
		if out != "[]" {
			t.Errorf("expected empty json array '[]', got: %q", out)
		}
		var parsed []any
		if err := json.Unmarshal([]byte(out), &parsed); err != nil {
			t.Fatalf("failed to unmarshal json array: %v", err)
		}
		if parsed == nil {
			t.Errorf("expected parsed json to be non-nil slice, got nil")
		}
	})

	t.Run("default text", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		rc := runSessionSearch(&stdout, &stderr, []string{"--query", "fak"})
		if rc != 0 {
			t.Fatalf("expected rc 0, got %d, stderr: %s", rc, stderr.String())
		}
		if !strings.Contains(stdout.String(), `sessionsearch: query "fak" matched 0 hit(s)`) {
			t.Errorf("expected matching query text, got: %s", stdout.String())
		}
	})

	t.Run("with journal file", func(t *testing.T) {
		tmpDir := t.TempDir()
		journalPath := filepath.Join(tmpDir, "journal.jsonl")
		content := `{"kind":"spawn","call_id":"c1","session":"interactive-42","tool":"slow_fetch","at_unix_ms":1700000000000,"deadline_ms":30000}
{"kind":"kill","call_id":"c1","session":"interactive-42","reason":"TOOL_DEADLINE_EXCEEDED","at_unix_ms":1700000032000}`
		if err := os.WriteFile(journalPath, []byte(content), 0o600); err != nil {
			t.Fatalf("failed to write journal: %v", err)
		}
		var stdout, stderr bytes.Buffer
		rc := runSessionSearch(&stdout, &stderr, []string{"--journal", journalPath, "--query", "deadline", "--json"})
		if rc != 0 {
			t.Fatalf("expected rc 0, got %d, stderr: %s", rc, stderr.String())
		}
		if !strings.Contains(stdout.String(), "TOOL_DEADLINE_EXCEEDED") {
			t.Errorf("expected matched event in json output, got: %s", stdout.String())
		}
	})

	t.Run("invalid flag", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		rc := runSessionSearch(&stdout, &stderr, []string{"--invalid-flag-xyz"})
		if rc != 2 {
			t.Fatalf("expected rc 2 on invalid flag, got %d", rc)
		}
	})
}
