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
	t.Chdir(t.TempDir())

	t.Run("json empty results array", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		rc := runSessionSearch(&stdout, &stderr, []string{"--query", "fak", "--json"})
		if rc != 0 {
			t.Fatalf("expected rc 0, got %d, stderr: %s", rc, stderr.String())
		}
		var env sessionSearchResultsEnvelope
		if err := json.Unmarshal(stdout.Bytes(), &env); err != nil {
			t.Fatalf("failed to unmarshal json envelope: %v, output: %s", err, stdout.String())
		}
		if env.Schema != "fak.sessionsearch-results/1" {
			t.Errorf("expected schema %q, got %q", "fak.sessionsearch-results/1", env.Schema)
		}
		if env.Query != "fak" {
			t.Errorf("expected query %q, got %q", "fak", env.Query)
		}
		if env.Total != 0 {
			t.Errorf("expected total 0, got %d", env.Total)
		}
		if env.Hits == nil || len(env.Hits) != 0 {
			t.Errorf("expected empty non-nil hits slice, got %#v", env.Hits)
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
		var env sessionSearchResultsEnvelope
		if err := json.Unmarshal(stdout.Bytes(), &env); err != nil {
			t.Fatalf("failed to unmarshal json envelope: %v, output: %s", err, stdout.String())
		}
		if env.Schema != "fak.sessionsearch-results/1" {
			t.Errorf("expected schema %q, got %q", "fak.sessionsearch-results/1", env.Schema)
		}
		if env.Query != "deadline" {
			t.Errorf("expected query %q, got %q", "deadline", env.Query)
		}
		if env.Total != 1 {
			t.Errorf("expected total 1, got %d", env.Total)
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

func TestRunSessionSearch_DefaultJournalAndSchemaEnvelope(t *testing.T) {
	tmpDir := t.TempDir()
	t.Chdir(tmpDir)

	journalDir := filepath.Join(tmpDir, ".fak", "toolproc")
	if err := os.MkdirAll(journalDir, 0o755); err != nil {
		t.Fatalf("failed to create default journal dir: %v", err)
	}

	content := `{"kind":"spawn","call_id":"c1","session":"sess-1","tool":"read_file","at_unix_ms":1700000000000}
{"kind":"exit","call_id":"c1","session":"sess-1","status":"ok","at_unix_ms":1700000001000}`
	journalPath := filepath.Join(journalDir, "journal.jsonl")
	if err := os.WriteFile(journalPath, []byte(content), 0o600); err != nil {
		t.Fatalf("failed to write default journal: %v", err)
	}

	var stdout, stderr bytes.Buffer
	rc := runSessionSearch(&stdout, &stderr, []string{"--query", "read_file", "--json"})
	if rc != 0 {
		t.Fatalf("expected rc 0, got %d, stderr: %s", rc, stderr.String())
	}

	var env sessionSearchResultsEnvelope
	if err := json.Unmarshal(stdout.Bytes(), &env); err != nil {
		t.Fatalf("failed to unmarshal json envelope: %v\noutput: %s", err, stdout.String())
	}

	if env.Schema != "fak.sessionsearch-results/1" {
		t.Errorf("expected schema %q, got %q", "fak.sessionsearch-results/1", env.Schema)
	}
	if env.Query != "read_file" {
		t.Errorf("expected query %q, got %q", "read_file", env.Query)
	}
	if env.Total != 1 {
		t.Errorf("expected total 1, got %d", env.Total)
	}
	if len(env.Hits) != 1 {
		t.Fatalf("expected 1 hit, got %d", len(env.Hits))
	}
	if !strings.Contains(env.Hits[0].Doc.Text, "read_file") {
		t.Errorf("expected hit doc text to contain %q, got %q", "read_file", env.Hits[0].Doc.Text)
	}
}
