package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRunIdeaScoutDryRunWithFixtureCandidates(t *testing.T) {
	dir := t.TempDir()
	configPath := filepath.Join(dir, "topics.json")
	if err := os.WriteFile(configPath, []byte(`{
		"topics": [{"key":"fixture","github":"fixture","terms":["agent","tool","policy"],"area":"trust-floor"}],
		"thresholds": {"min_score": 1, "max_issues": 1}
	}`), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}
	candidatesPath := filepath.Join(dir, "candidates.json")
	if err := os.WriteFile(candidatesPath, []byte(`[
		{"source":"github","source_id":"github:o/r","url":"https://github.com/o/r","title":"o/r","summary":"agent tool policy","published":"2026-06-29T00:00:00Z","topic":"fixture","extra":{"stars":300,"pushed_at":"2026-06-30T00:00:00Z","language":"Go"}},
		{"source":"github","source_id":"github:o/dupe","url":"https://github.com/o/dupe","title":"dupe","summary":"agent tool policy","published":"2026-06-29T00:00:00Z","topic":"fixture"}
	]`), 0o644); err != nil {
		t.Fatalf("write candidates: %v", err)
	}
	issuesPath := filepath.Join(dir, "issues.json")
	if err := os.WriteFile(issuesPath, []byte(`[{"number":1,"title":"dupe","body":"manual issue"}]`), 0o644); err != nil {
		t.Fatalf("write issues: %v", err)
	}

	var stdout, stderr bytes.Buffer
	code := runIdeaScout(&stdout, &stderr, []string{
		"--workspace", dir,
		"--config", configPath,
		"--candidates", candidatesPath,
		"--issues", issuesPath,
		"--json",
		"--today", "2026-06-30",
	})
	if code != 0 {
		t.Fatalf("runIdeaScout code=%d stderr=%s stdout=%s", code, stderr.String(), stdout.String())
	}
	var result struct {
		Mode               string `json:"mode"`
		CandidatesGathered int    `json:"candidates_gathered"`
		Planned            []struct {
			Title    string   `json:"title"`
			SourceID string   `json:"source_id"`
			Labels   []string `json:"labels"`
		} `json:"planned"`
		Skipped map[string]int `json:"skipped"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		t.Fatalf("stdout is not JSON: %v\n%s", err, stdout.String())
	}
	if result.Mode != "dry-run" || result.CandidatesGathered != 2 {
		t.Fatalf("result mode/gathered = %#v", result)
	}
	if len(result.Planned) != 1 || result.Planned[0].SourceID != "github:o/r" {
		t.Fatalf("planned = %#v, want only github:o/r", result.Planned)
	}
	if result.Skipped["title-near"] != 1 {
		t.Fatalf("skipped = %#v, want title-near=1", result.Skipped)
	}
	if strings.Contains(stdout.String(), `"body"`) {
		t.Fatalf("JSON plan should omit full issue bodies from the summary output: %s", stdout.String())
	}
	if _, err := os.Stat(filepath.Join(dir, ".idea-scout", "seen.json")); !os.IsNotExist(err) {
		t.Fatalf("dry-run wrote seen cache, stat err=%v", err)
	}
}
