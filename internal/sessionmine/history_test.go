package sessionmine

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestExploreIndexAggregatesFiltersAndDrillsDown(t *testing.T) {
	root := t.TempDir()
	index := filepath.Join(root, "index.json")
	secretPath := filepath.Join(root, "SECRET-source.jsonl")
	state := IndexState{Schema: indexSchema, UpdatedAt: "2026-08-17T22:00:00Z", Seen: map[string]bool{}, Files: map[string]IndexedFile{
		sourceFingerprint("codex", secretPath):            {Provider: "codex", Size: 10, ModUnix: 1, Session: Session{ID: "codex-a", Provider: "codex", EndedAt: "2026-08-17T20:00:00Z", DurationMS: 100, ToolCalls: 3, ToolResults: 2, ToolErrors: 1, Trajectory: []string{"git_status", "view_image"}}},
		sourceFingerprint("claude", "other-secret.jsonl"): {Provider: "claude", Size: 20, ModUnix: 2, Session: Session{ID: "claude-b", Provider: "claude", EndedAt: "2026-08-17T21:00:00Z", DurationMS: 300, ToolCalls: 5, ToolResults: 4, ToolErrors: 2, Trajectory: []string{"read_file", "edit_file"}}},
	}}
	if err := writeIndexAtomic(index, state); err != nil {
		t.Fatal(err)
	}
	got, err := ExploreIndex(index, HistoryOptions{Limit: 10})
	if err != nil {
		t.Fatal(err)
	}
	if got.Metrics.Sessions != 2 || got.Metrics.ToolCalls != 8 || got.Metrics.ToolErrors != 3 || got.Metrics.P50DurationMS != 100 || got.Sessions[0].ID != "claude-b" {
		t.Fatalf("aggregate=%+v", got)
	}
	filtered, err := ExploreIndex(index, HistoryOptions{Provider: "codex", MinErrors: 1})
	if err != nil {
		t.Fatal(err)
	}
	if len(filtered.Sessions) != 1 || filtered.Sessions[0].ID != "codex-a" {
		t.Fatalf("filtered=%+v", filtered)
	}
	detail, err := ExploreIndex(index, HistoryOptions{SessionID: "claude-b"})
	if err != nil {
		t.Fatal(err)
	}
	if detail.Session == nil || strings.Join(detail.Session.Trajectory, ",") != "read_file,edit_file" || len(detail.Sessions) != 0 {
		t.Fatalf("detail=%+v", detail)
	}
	data := string(mustRead(t, index))
	if strings.Contains(data, secretPath) {
		t.Fatalf("source path leaked: %s", data)
	}
}

func TestExploreIndexRejectsUnknownSessionAndSchema(t *testing.T) {
	root := t.TempDir()
	index := filepath.Join(root, "index.json")
	if err := writeIndexAtomic(index, IndexState{Schema: indexSchema, Files: map[string]IndexedFile{}, Seen: map[string]bool{}}); err != nil {
		t.Fatal(err)
	}
	if _, err := ExploreIndex(index, HistoryOptions{SessionID: "missing"}); !errors.Is(err, ErrSessionNotFound) {
		t.Fatalf("missing err=%v", err)
	}
	if err := os.WriteFile(index, []byte(`{"schema":"future/9"}`), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := ExploreIndex(index, HistoryOptions{}); err == nil {
		t.Fatal("expected schema rejection")
	}
}

func TestExploreIndexFiltersByTrajectoryStep(t *testing.T) {
	root := t.TempDir()
	index := filepath.Join(root, "index.json")
	state := IndexState{Schema: indexSchema, Files: map[string]IndexedFile{
		"a": {Provider: "codex", Session: Session{ID: "a", Provider: "codex", ToolCalls: 2, Trajectory: []string{"git_status", "view_image"}}},
		"b": {Provider: "claude", Session: Session{ID: "b", Provider: "claude", ToolCalls: 3, Trajectory: []string{"read_file", "edit_file"}}},
	}, Seen: map[string]bool{}}
	if err := writeIndexAtomic(index, state); err != nil {
		t.Fatal(err)
	}
	got, err := ExploreIndex(index, HistoryOptions{Tool: "view_image"})
	if err != nil {
		t.Fatal(err)
	}
	if got.Metrics.Sessions != 1 || got.Metrics.ToolCalls != 2 || len(got.Sessions) != 1 || got.Sessions[0].ID != "a" {
		t.Fatalf("filtered=%+v", got)
	}
	missing, err := ExploreIndex(index, HistoryOptions{Tool: "View_Image"})
	if err != nil {
		t.Fatal(err)
	}
	if missing.Metrics.Sessions != 0 || len(missing.Sessions) != 0 {
		t.Fatalf("exact filter=%+v", missing)
	}
}
