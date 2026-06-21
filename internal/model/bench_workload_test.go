package model

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLoadBenchWorkload(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "workload.json")
	body := `{
	  "schema": "fak.agent-workload.v1",
	  "cases": [
	    {"name": "live-fak", "prompt_tokens": 1501, "completion_tokens": 184, "turns": 7, "tool_calls": 6}
	  ]
	}`
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	w, err := LoadBenchWorkload(path)
	if err != nil {
		t.Fatal(err)
	}
	if got := len(w.Cases); got != 1 {
		t.Fatalf("cases = %d, want 1", got)
	}
	if got := w.Cases[0].PromptTokens; got != 1501 {
		t.Fatalf("prompt tokens = %d, want 1501", got)
	}
}

func TestLoadBenchWorkloadRejectsInvalidCases(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "workload.json")
	body := `{"cases":[{"name":"bad","prompt_tokens":0,"completion_tokens":1}]}`
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	_, err := LoadBenchWorkload(path)
	if err == nil || !strings.Contains(err.Error(), "prompt_tokens") {
		t.Fatalf("expected prompt token validation error, got %v", err)
	}
}
