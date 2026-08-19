package sessionmine

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func writeHealthIndex(t *testing.T, path string, state IndexState) {
	t.Helper()
	b, err := json.Marshal(state)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, b, 0o600); err != nil {
		t.Fatal(err)
	}
}

func TestInspectIndexHealthStates(t *testing.T) {
	now := time.Date(2026, 8, 18, 0, 0, 0, 0, time.UTC)
	missing := InspectIndexHealth(filepath.Join(t.TempDir(), "none"), now)
	if missing.Verdict != "RED" || missing.Reason != "index_missing" {
		t.Fatalf("%+v", missing)
	}
	dir := t.TempDir()
	p := filepath.Join(dir, "index.json")
	state := IndexState{Schema: indexSchema, UpdatedAt: now.Add(-time.Hour).Format(time.RFC3339), Files: map[string]IndexedFile{"a": {Provider: "codex", Session: Session{ID: "s"}}}}
	writeHealthIndex(t, p, state)
	partial := InspectIndexHealth(p, now)
	if partial.Verdict != "WARN" || partial.Reason != "partial_provider_coverage" {
		t.Fatalf("%+v", partial)
	}
	state.Files["b"] = IndexedFile{Provider: "claude", Session: Session{ID: "c"}}
	writeHealthIndex(t, p, state)
	healthy := InspectIndexHealth(p, now)
	if healthy.Verdict != "GREEN" || healthy.Reason != "healthy" {
		t.Fatalf("%+v", healthy)
	}
}

func TestInspectIndexHealthSourceCensusAndReceipt(t *testing.T) {
	now := time.Date(2026, 8, 18, 0, 0, 0, 0, time.UTC)
	dir := t.TempDir()
	codex := filepath.Join(dir, "codex")
	claude := filepath.Join(dir, "claude")
	if err := os.MkdirAll(codex, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(claude, 0o755); err != nil {
		t.Fatal(err)
	}
	accepted := filepath.Join(codex, "accepted.jsonl")
	rejected := filepath.Join(codex, "rejected.jsonl")
	_ = os.WriteFile(accepted, []byte("{}\n"), 0o600)
	_ = os.WriteFile(rejected, []byte("bad\n"), 0o600)
	index := filepath.Join(dir, "index.json")
	state := IndexState{Schema: indexSchema, UpdatedAt: now.Add(-time.Hour).Format(time.RFC3339), Files: map[string]IndexedFile{sourceFingerprint("codex", accepted): {Provider: "codex", Session: Session{ID: "s"}, Malformed: 1}}}
	writeHealthIndex(t, index, state)
	receipt := RefreshReceipt{Schema: "fak-session-history-refresh/1", CompletedAt: now.Format(time.RFC3339), Outcome: "ok", ParsedFiles: 1}
	if err := writeRefreshReceipt(index, receipt); err != nil {
		t.Fatal(err)
	}
	got := InspectIndexHealthWithOptions(IndexHealthOptions{IndexPath: index, CodexRoot: codex, ClaudeRoot: claude, Now: now})
	if got.Reason != "partial_provider_coverage" || got.LastRefresh.State != "recorded" || got.LastRefresh.Outcome != "ok" {
		t.Fatalf("%+v", got)
	}
	var cp SourceHealth
	for _, p := range got.Providers {
		if p.Provider == "codex" {
			cp = p
		}
	}
	if cp.Files != 2 || cp.AcceptedFiles != 1 || cp.RejectedFiles != 1 || cp.MalformedRows != 1 || cp.State != "partial" {
		t.Fatalf("%+v", cp)
	}
	if got.Providers[0].State != "empty" {
		t.Fatalf("providers=%+v", got.Providers)
	}
}

func TestInspectIndexHealthReportsLiveContentionWithoutPaths(t *testing.T) {
	now := time.Now().UTC()
	dir := t.TempDir()
	index := filepath.Join(dir, "index.json")
	writeHealthIndex(t, index, IndexState{Schema: indexSchema, UpdatedAt: now.Format(time.RFC3339), Files: map[string]IndexedFile{}})
	b, _ := json.Marshal(refreshLock{PID: os.Getpid(), StartedAt: now.Format(time.RFC3339)})
	if err := os.WriteFile(refreshLockPath(index), b, 0o600); err != nil {
		t.Fatal(err)
	}
	got := InspectIndexHealth(index, now)
	if got.Reason != "refresh_in_progress" || got.Contention.State != "live" || got.Contention.OwnerPID != os.Getpid() {
		t.Fatalf("%+v", got)
	}
	encoded, _ := json.Marshal(got)
	if string(encoded) == "" || containsString(string(encoded), dir) {
		t.Fatalf("status leaked source/index path: %s", encoded)
	}
}

func TestInspectIndexHealthRejectsUnsupportedSchema(t *testing.T) {
	p := filepath.Join(t.TempDir(), "index.json")
	_ = os.WriteFile(p, []byte(`{"schema":"newer/2"}`), 0o600)
	got := InspectIndexHealth(p, time.Now())
	if got.Verdict != "RED" || got.Reason != "index_invalid" {
		t.Fatalf("%+v", got)
	}
}
func containsString(s, sub string) bool {
	if len(sub) == 0 {
		return false
	}
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
