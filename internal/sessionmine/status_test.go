package sessionmine

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestInspectIndexHealthStates(t *testing.T) {
	now := time.Date(2026, 8, 18, 0, 0, 0, 0, time.UTC)
	missing := InspectIndexHealth(filepath.Join(t.TempDir(), "none"), now)
	if missing.Verdict != "RED" || missing.Reason != "index_missing" {
		t.Fatalf("%+v", missing)
	}
	dir := t.TempDir()
	p := filepath.Join(dir, "index.json")
	state := IndexState{Schema: indexSchema, UpdatedAt: now.Add(-time.Hour).Format(time.RFC3339), Files: map[string]IndexedFile{"a": {Provider: "codex", Session: Session{ID: "s"}}}}
	b, _ := json.Marshal(state)
	if err := os.WriteFile(p, b, 0600); err != nil {
		t.Fatal(err)
	}
	partial := InspectIndexHealth(p, now)
	if partial.Verdict != "WARN" || partial.Reason != "partial_provider_coverage" {
		t.Fatalf("%+v", partial)
	}
	state.Files["b"] = IndexedFile{Provider: "claude", Session: Session{ID: "c"}}
	b, _ = json.Marshal(state)
	_ = os.WriteFile(p, b, 0600)
	healthy := InspectIndexHealth(p, now)
	if healthy.Verdict != "GREEN" || healthy.Reason != "healthy" {
		t.Fatalf("%+v", healthy)
	}
}
func TestInspectIndexHealthRejectsUnsupportedSchema(t *testing.T) {
	p := filepath.Join(t.TempDir(), "index.json")
	_ = os.WriteFile(p, []byte(`{"schema":"newer/2"}`), 0600)
	got := InspectIndexHealth(p, time.Now())
	if got.Verdict != "RED" || got.Reason != "index_invalid" {
		t.Fatalf("%+v", got)
	}
}
