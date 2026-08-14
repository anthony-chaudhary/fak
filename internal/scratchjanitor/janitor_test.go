package scratchjanitor

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestScanSelectsOnlyOldUnreferencedSessions(t *testing.T) {
	root := t.TempDir()
	now := time.Date(2026, time.August, 14, 12, 0, 0, 0, time.UTC)
	old := makeSession(t, root, "project-a", "old", now.Add(-8*24*time.Hour))
	fresh := makeSession(t, root, "project-a", "fresh", now.Add(-6*24*time.Hour))
	referenced := makeSession(t, root, "project-b", "referenced", now.Add(-9*24*time.Hour))

	result, err := Scan(Config{
		Root:       root,
		MaxAge:     DefaultMaxAge,
		Now:        func() time.Time { return now },
		Referenced: map[string]bool{filepath.Join(referenced, "."): true},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Mode != "dry-run" {
		t.Fatalf("mode = %q, want dry-run", result.Mode)
	}
	if len(result.Candidates) != 1 {
		t.Fatalf("candidates = %#v, want one", result.Candidates)
	}
	candidate := result.Candidates[0]
	if candidate.Project != "project-a" || candidate.Session != "old" {
		t.Fatalf("candidate = %#v, want project-a/old", candidate)
	}
	if candidate.Path != old || candidate.ScratchpadPath != old {
		t.Fatalf("candidate paths = %q, %q; want %q", candidate.Path, candidate.ScratchpadPath, old)
	}
	if candidate.AgeSeconds != int64((8*24*time.Hour)/time.Second) {
		t.Fatalf("age_seconds = %d", candidate.AgeSeconds)
	}
	if _, err := os.Stat(old); err != nil {
		t.Fatalf("dry-run removed old session: %v", err)
	}
	if _, err := os.Stat(fresh); err != nil {
		t.Fatalf("fresh session missing: %v", err)
	}
	if len(result.Actions) != 0 {
		t.Fatalf("dry-run actions = %#v, want none", result.Actions)
	}
}

func TestScanApplyRemovesCandidate(t *testing.T) {
	root := t.TempDir()
	now := time.Date(2026, time.August, 14, 12, 0, 0, 0, time.UTC)
	old := makeSession(t, root, "project", "session", now.Add(-8*24*time.Hour))

	result, err := Scan(Config{
		Root:   root,
		MaxAge: DefaultMaxAge,
		Now:    func() time.Time { return now },
		Apply:  true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Mode != "apply" {
		t.Fatalf("mode = %q, want apply", result.Mode)
	}
	if len(result.Actions) != 1 || result.Actions[0].Action != "removed" {
		t.Fatalf("actions = %#v, want one removal", result.Actions)
	}
	if _, err := os.Stat(old); !os.IsNotExist(err) {
		t.Fatalf("candidate still exists after apply: %v", err)
	}
}

func TestStrictAgeBoundaryIsExcluded(t *testing.T) {
	root := t.TempDir()
	now := time.Date(2026, time.August, 14, 12, 0, 0, 0, time.UTC)
	makeSession(t, root, "project", "boundary", now.Add(-DefaultMaxAge))

	result, err := Scan(Config{
		Root:   root,
		MaxAge: DefaultMaxAge,
		Now:    func() time.Time { return now },
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Candidates) != 0 {
		t.Fatalf("boundary candidate selected: %#v", result.Candidates)
	}
}

func makeSession(t *testing.T, root, project, session string, modTime time.Time) string {
	t.Helper()
	path := filepath.Join(root, project, session)
	if err := os.MkdirAll(path, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(path, modTime, modTime); err != nil {
		t.Fatal(err)
	}
	absolute, err := filepath.Abs(filepath.Clean(path))
	if err != nil {
		t.Fatal(err)
	}
	return filepath.Clean(absolute)
}
