package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestSessionAuditDiscoverAuditAndDeep(t *testing.T) {
	root := t.TempDir()
	sessionPath := writeSessionAuditJSONL(t, filepath.Join(root, "C--work-fak", "session-a.jsonl"), []map[string]any{
		sessionAuditAssistant("msg-1", 100, "Read"),
		map[string]any{
			"type":      "user",
			"timestamp": "2026-06-20T00:01:00.000Z",
			"message": map[string]any{
				"content": "Run the audit",
			},
		},
	})

	var stdout, stderr bytes.Buffer
	rc := runSessionAudit(&stdout, &stderr, []string{"discover", "--root", root, "--all", "--max", "1"})
	if rc != 0 {
		t.Fatalf("discover rc=%d stderr=%s", rc, stderr.String())
	}
	if !strings.Contains(stdout.String(), "1 sessions") || !strings.Contains(stdout.String(), "C--work-fak/session-a.jsonl") {
		t.Fatalf("unexpected discover output:\n%s", stdout.String())
	}

	stdout.Reset()
	stderr.Reset()
	jsonOut := filepath.Join(t.TempDir(), "audit.json")
	mdOut := filepath.Join(t.TempDir(), "audit.md")
	rc = runSessionAudit(&stdout, &stderr, []string{"audit", "--root", root, "--all", "--json", jsonOut, "--md", mdOut})
	if rc != 0 {
		t.Fatalf("audit rc=%d stderr=%s", rc, stderr.String())
	}
	if !strings.Contains(stdout.String(), "Session-Transcript Audit") {
		t.Fatalf("audit did not render markdown:\n%s", stdout.String())
	}
	if _, err := os.Stat(mdOut); err != nil {
		t.Fatalf("markdown output not written: %v", err)
	}
	var payload struct {
		Aggregate struct {
			NSessions int `json:"n_sessions"`
		} `json:"aggregate"`
	}
	raw, err := os.ReadFile(jsonOut)
	if err != nil {
		t.Fatalf("json output not written: %v", err)
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		t.Fatalf("bad json output: %v\n%s", err, raw)
	}
	if payload.Aggregate.NSessions != 1 {
		t.Fatalf("json sessions = %d, want 1", payload.Aggregate.NSessions)
	}

	stdout.Reset()
	stderr.Reset()
	rc = runSessionAudit(&stdout, &stderr, []string{"deep", sessionPath})
	if rc != 0 {
		t.Fatalf("deep rc=%d stderr=%s", rc, stderr.String())
	}
	if !strings.Contains(stdout.String(), "# Trajectory: session-a") || !strings.Contains(stdout.String(), "Run the audit") {
		t.Fatalf("unexpected deep output:\n%s", stdout.String())
	}
}

func TestSessionAuditWarnsWhenSubagentsExcluded(t *testing.T) {
	root := t.TempDir()
	writeSessionAuditJSONL(t, filepath.Join(root, "C--work-fak", "session-a.jsonl"), []map[string]any{
		sessionAuditAssistant("top", 100, ""),
	})
	writeSessionAuditJSONL(t, filepath.Join(root, "C--work-fak", "session-a", "subagents", "worker.jsonl"), []map[string]any{
		sessionAuditAssistant("sub", 2_000, ""),
	})
	var stdout, stderr bytes.Buffer
	rc := runSessionAudit(&stdout, &stderr, []string{"audit", "--root", root, "--all"})
	if rc != 0 {
		t.Fatalf("audit rc=%d stderr=%s", rc, stderr.String())
	}
	if !strings.Contains(stdout.String(), "NOTE: +1 subagent transcripts uncounted") {
		t.Fatalf("subagent warning missing:\n%s", stdout.String())
	}
}

func sessionAuditAssistant(id string, out int64, tool string) map[string]any {
	content := []any{}
	if tool != "" {
		content = append(content, map[string]any{"type": "tool_use", "name": tool, "input": map[string]any{}})
	}
	return map[string]any{
		"type":      "assistant",
		"timestamp": "2026-06-20T00:00:00.000Z",
		"message": map[string]any{
			"id":    id,
			"model": "claude-sonnet-4-5",
			"usage": map[string]any{
				"input_tokens":                10,
				"output_tokens":               out,
				"cache_read_input_tokens":     20,
				"cache_creation_input_tokens": 5,
			},
			"content": content,
		},
	}
}

func writeSessionAuditJSONL(t *testing.T, path string, records []map[string]any) string {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o777); err != nil {
		t.Fatal(err)
	}
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	enc := json.NewEncoder(f)
	for _, r := range records {
		if err := enc.Encode(r); err != nil {
			t.Fatal(err)
		}
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	return path
}
