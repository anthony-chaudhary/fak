package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/scratchjanitor"
)

func TestRunScratchJanitorJSONShape(t *testing.T) {
	root := t.TempDir()
	session := makeCLIScratchSession(t, root, "project", "old")
	old := time.Now().Add(-scratchjanitor.DefaultMaxAge - time.Hour)
	if err := os.Chtimes(session, old, old); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	exitCode := runScratchJanitor(&stdout, &stderr, []string{
		"--root", root,
		"--json",
	})
	if exitCode != 0 {
		t.Fatalf("exit = %d, stderr = %s", exitCode, stderr.String())
	}

	var result map[string]json.RawMessage
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		t.Fatalf("decode output %q: %v", stdout.String(), err)
	}
	for _, field := range []string{"mode", "candidates", "actions"} {
		if _, ok := result[field]; !ok {
			t.Fatalf("output missing %q: %s", field, stdout.String())
		}
	}
	var candidates []scratchjanitor.Candidate
	if err := json.Unmarshal(result["candidates"], &candidates); err != nil {
		t.Fatal(err)
	}
	if len(candidates) != 1 || candidates[0].ScratchpadPath != session {
		t.Fatalf("candidates = %#v, want %q", candidates, session)
	}
	if _, err := os.Stat(session); err != nil {
		t.Fatalf("dry-run removed session: %v", err)
	}
}

func TestRunScratchJanitorResumeReferenceExclusion(t *testing.T) {
	root := t.TempDir()
	session := makeCLIScratchSession(t, root, "project", "referenced")
	old := time.Now().Add(-scratchjanitor.DefaultMaxAge - time.Hour)
	if err := os.Chtimes(session, old, old); err != nil {
		t.Fatal(err)
	}
	resumePath := filepath.Join(t.TempDir(), "resume.json")
	packet := map[string]any{
		"scan": map[string]any{
			"candidates": []map[string]string{
				{"scratchpad_path": filepath.Join(session, ".")},
			},
		},
	}
	data, err := json.Marshal(packet)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(resumePath, data, 0o600); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	exitCode := runScratchJanitor(&stdout, &stderr, []string{
		"--root", root,
		"--resume-json", resumePath,
		"--json",
	})
	if exitCode != 0 {
		t.Fatalf("exit = %d, stderr = %s", exitCode, stderr.String())
	}
	var result scratchjanitor.Result
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if len(result.Candidates) != 0 {
		t.Fatalf("referenced session selected: %#v", result.Candidates)
	}
}

func makeCLIScratchSession(t *testing.T, root, project, session string) string {
	t.Helper()
	path := filepath.Join(root, project, session)
	if err := os.MkdirAll(path, 0o755); err != nil {
		t.Fatal(err)
	}
	absolute, err := filepath.Abs(filepath.Clean(path))
	if err != nil {
		t.Fatal(err)
	}
	return filepath.Clean(absolute)
}
