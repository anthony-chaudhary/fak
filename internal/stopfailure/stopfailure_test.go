package stopfailure

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestResetAndArchiveCandidates(t *testing.T) {
	root := t.TempDir()
	stopDir := filepath.Join(root, ".dos", "stop-failures")
	streamDir := filepath.Join(root, ".dos", "streams")
	claudeHome := filepath.Join(root, "home")
	claudeProject := filepath.Join(claudeHome, ".claude", "projects", DefaultTranscriptNamespace)
	for _, dir := range []string{stopDir, streamDir, claudeProject} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
	}

	now := time.Date(2026, 6, 26, 12, 0, 0, 0, time.UTC)
	writeMarker(t, stopDir, "recent", `{"total":1,"consecutive":1}`, now.Add(-time.Hour))
	writeMarker(t, stopDir, "recent2", `{"total":1,"consecutive":1}`, now.Add(-2*time.Hour))
	writeMarker(t, stopDir, "stale", `{"total":2,"consecutive":2}`, now.Add(-8*time.Hour))
	writeMarker(t, stopDir, "claudeonly", `{"total":1,"consecutive":1}`, now.Add(-7*time.Hour))
	writeMarker(t, stopDir, "markeronly", `{"total":1,"consecutive":1}`, now.Add(-9*time.Hour))
	if err := os.WriteFile(filepath.Join(streamDir, "stale.jsonl"), []byte("{}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(claudeProject, "claudeonly.jsonl"), []byte("{}\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	opts := Options{
		Root:         root,
		Now:          now,
		RecentWindow: 6 * time.Hour,
		SinceWindow:  24 * time.Hour,
		ClaudeHome:   claudeHome,
	}
	plan, err := BuildPlan(opts)
	if err != nil {
		t.Fatal(err)
	}
	if got := plan.Counts[ActionRecentReview]; got != 2 {
		t.Fatalf("recent count = %d, want 2", got)
	}
	if got := plan.Counts[ActionStaleReset]; got != 2 {
		t.Fatalf("stale reset count = %d, want 2", got)
	}
	if got := plan.Counts[ActionStaleMarkerOnlyArchive]; got != 1 {
		t.Fatalf("marker-only archive count = %d, want 1", got)
	}

	reset, err := ResetStale(opts, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(reset.Updated) != 2 {
		t.Fatalf("reset updated = %d, want 2", len(reset.Updated))
	}
	assertConsecutive(t, stopDir, "stale", 0)
	assertConsecutive(t, stopDir, "claudeonly", 0)
	assertConsecutive(t, stopDir, "recent", 1)
	assertConsecutive(t, stopDir, "markeronly", 1)

	cleared, err := ClearReviewed(opts, []string{"recent"}, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(cleared.Updated) != 1 {
		t.Fatalf("clear updated = %d, want 1", len(cleared.Updated))
	}
	assertConsecutive(t, stopDir, "recent", 0)
	assertConsecutive(t, stopDir, "recent2", 1)

	archived, err := ArchiveMarkerOnly(opts, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(archived.Archived) != 1 {
		t.Fatalf("archived = %d, want 1", len(archived.Archived))
	}
	if _, err := os.Stat(filepath.Join(stopDir, "markeronly.json")); !os.IsNotExist(err) {
		t.Fatalf("active markeronly should be moved, err=%v", err)
	}
	if _, err := os.Stat(filepath.Join(stopDir, "archive", "markeronly.json")); err != nil {
		t.Fatalf("archive marker missing: %v", err)
	}
}

func writeMarker(t *testing.T, dir, session, body string, mtime time.Time) {
	t.Helper()
	path := filepath.Join(dir, session+".json")
	if err := os.WriteFile(path, []byte(body+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(path, mtime, mtime); err != nil {
		t.Fatal(err)
	}
}

func assertConsecutive(t *testing.T, dir, session string, want int) {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(dir, session+".json"))
	if err != nil {
		t.Fatal(err)
	}
	var doc map[string]int
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatal(err)
	}
	if got := doc["consecutive"]; got != want {
		t.Fatalf("%s consecutive=%d, want %d", session, got, want)
	}
}
