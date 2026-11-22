package superstream

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func writeSpecFile(t *testing.T, dir, name string, spec StreamSpec) string {
	t.Helper()
	path := filepath.Join(dir, name)
	b, err := json.MarshalIndent(spec, "", "  ")
	if err != nil {
		t.Fatalf("marshal spec: %v", err)
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatalf("write spec %s: %v", path, err)
	}
	return path
}

func readSpecFile(t *testing.T, path string) StreamSpec {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read spec %s: %v", path, err)
	}
	var spec StreamSpec
	if err := json.Unmarshal(b, &spec); err != nil {
		t.Fatalf("parse spec %s: %v", path, err)
	}
	return spec
}

func TestStateFilePathStripsExtension(t *testing.T) {
	cases := map[string]string{
		"a.json":                       "a.state.json",
		"mystream.super.json":          "mystream.super.state.json",
		filepath.Join("sub", "s.json"): filepath.Join("sub", "s.state.json"),
		"noext":                        "noext.state.json",
	}
	for in, want := range cases {
		if got := StateFilePath(in); got != want {
			t.Fatalf("StateFilePath(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestSaveLoadStateRoundTrip(t *testing.T) {
	dir := t.TempDir()
	spec := StreamSpec{
		ID:     "stream-roundtrip",
		Intent: "roundtrip",
		Queue: []WorkItem{
			{ID: "task-1", Title: "first", Lane: "docs", Witness: "go test ./docs"},
			{ID: "task-2", Title: "second", Lane: "gateway"},
		},
	}
	specPath := writeSpecFile(t, dir, "stream.json", spec)

	state := NewStreamState(spec.NormalizedSpec())
	state.Queue[0].Status = ItemCompleted
	state.Queue[0].CommitSHA = "abc123"
	state.Queue[0].TurnsSpent = 3
	state.Queue[0].TokensSpent = 900
	state.TotalTurnsSpent = 3
	state.TotalTokensSpent = 900
	// Stored counters are never trusted: LoadState must recompute them from
	// the reconciled queue statuses, so a bogus 99 must read back as 1.
	state.CompletedCount = 99

	now := time.Date(2026, 11, 21, 12, 0, 0, 0, time.UTC)
	if err := SaveState(specPath, state, now); err != nil {
		t.Fatalf("SaveState: %v", err)
	}

	got, found, err := LoadState(specPath, spec)
	if err != nil {
		t.Fatalf("LoadState: %v", err)
	}
	if !found {
		t.Fatal("found = false, want true")
	}
	if got.CompletedCount != 1 {
		t.Fatalf("CompletedCount = %d, want 1", got.CompletedCount)
	}
	if got.Queue[0].Status != ItemCompleted {
		t.Fatalf("Queue[0].Status = %s, want %s", got.Queue[0].Status, ItemCompleted)
	}
	if got.Queue[0].CommitSHA != "abc123" {
		t.Fatalf("Queue[0].CommitSHA = %q, want abc123", got.Queue[0].CommitSHA)
	}
	if got.Queue[0].TurnsSpent != 3 || got.Queue[0].TokensSpent != 900 {
		t.Fatalf("Queue[0] spend = %d turns / %d tokens, want 3 / 900", got.Queue[0].TurnsSpent, got.Queue[0].TokensSpent)
	}
	if got.TotalTurnsSpent != 3 || got.TotalTokensSpent != 900 {
		t.Fatalf("totals = %d turns / %d tokens, want 3 / 900", got.TotalTurnsSpent, got.TotalTokensSpent)
	}
	if got.Queue[1].Status != ItemPending {
		t.Fatalf("Queue[1].Status = %s, want %s", got.Queue[1].Status, ItemPending)
	}
}

func TestLoadStateMissingIsFresh(t *testing.T) {
	dir := t.TempDir()
	spec := StreamSpec{
		ID:     "stream-fresh",
		Intent: "fresh",
		Queue:  []WorkItem{{ID: "task-1", Lane: "docs"}},
	}
	specPath := filepath.Join(dir, "absent.json")

	got, found, err := LoadState(specPath, spec)
	if err != nil {
		t.Fatalf("LoadState: %v", err)
	}
	if found {
		t.Fatal("found = true, want false for a missing state file")
	}
	want := NewStreamState(spec.NormalizedSpec())
	if len(got.Queue) != len(want.Queue) || got.StreamID != want.StreamID {
		t.Fatalf("fresh state = %+v, want %+v", got, want)
	}
	if got.CompletedCount != 0 || got.ActiveIndex != 0 {
		t.Fatalf("fresh state counts = %d completed at index %d, want 0 at 0", got.CompletedCount, got.ActiveIndex)
	}
}

func TestLoadStateReconcilesEditedSpec(t *testing.T) {
	dir := t.TempDir()
	spec := StreamSpec{
		ID:     "stream-edited",
		Intent: "edited",
		Queue: []WorkItem{
			{ID: "task-1", Lane: "docs"},
			{ID: "task-2", Lane: "gateway"},
			{ID: "task-3", Lane: "engine"},
		},
	}
	specPath := writeSpecFile(t, dir, "edited.json", spec)

	state := NewStreamState(spec.NormalizedSpec())
	state.Queue[0].Status = ItemCompleted
	state.Queue[0].CommitSHA = "deadbeef"
	state.CompletedCount = 1
	state.ActiveIndex = 1
	if err := SaveState(specPath, state, time.Now()); err != nil {
		t.Fatalf("SaveState: %v", err)
	}

	// The spec is edited after the state was written: one item is added and
	// task-2 moves to a different lane.
	spec.Queue = append(spec.Queue, WorkItem{ID: "task-4", Lane: "docs"})
	spec.Queue[1].Lane = "engine"
	edit := writeSpecFile(t, dir, "edited.json", spec)

	got, found, err := LoadState(edit, spec)
	if err != nil {
		t.Fatalf("LoadState: %v", err)
	}
	if !found {
		t.Fatal("found = false, want true")
	}
	if len(got.Queue) != 4 {
		t.Fatalf("queue length = %d, want 4 (the edited spec owns the shape)", len(got.Queue))
	}
	if got.Queue[0].Status != ItemCompleted || got.Queue[0].CommitSHA != "deadbeef" {
		t.Fatalf("task-1 = %+v, want execution fields preserved", got.Queue[0])
	}
	if got.Queue[1].Lane != "engine" {
		t.Fatalf("task-2 lane = %q, want engine (from the edited spec)", got.Queue[1].Lane)
	}
	if got.Queue[3].ID != "task-4" || got.Queue[3].Status != ItemPending {
		t.Fatalf("task-4 = %+v, want a pending appended item", got.Queue[3])
	}
	if got.CompletedCount != 1 {
		t.Fatalf("CompletedCount = %d, want 1", got.CompletedCount)
	}
	if got.ActiveIndex != 1 {
		t.Fatalf("ActiveIndex = %d, want 1 (preserved)", got.ActiveIndex)
	}
}

func TestLoadStateCorruptRefused(t *testing.T) {
	spec := StreamSpec{
		ID:     "stream-corrupt",
		Intent: "corrupt",
		Queue:  []WorkItem{{ID: "task-1", Lane: "docs"}},
	}

	t.Run("schema tampered", func(t *testing.T) {
		dir := t.TempDir()
		specPath := writeSpecFile(t, dir, "corrupt.json", spec)
		state := NewStreamState(spec.NormalizedSpec())
		if err := SaveState(specPath, state, time.Now()); err != nil {
			t.Fatalf("SaveState: %v", err)
		}
		envPath := StateFilePath(specPath)
		raw, err := os.ReadFile(envPath)
		if err != nil {
			t.Fatalf("read state: %v", err)
		}
		tampered := strings.Replace(string(raw), StateSchema, "fak.superstream.state.v0", 1)
		if tampered == string(raw) {
			t.Fatalf("tamper failed: schema token absent from %s", raw)
		}
		if err := os.WriteFile(envPath, []byte(tampered), 0o644); err != nil {
			t.Fatalf("write tampered state: %v", err)
		}
		if _, _, err := LoadState(specPath, spec); err == nil {
			t.Fatal("LoadState accepted a state with a foreign schema")
		}
	})

	t.Run("garbage json", func(t *testing.T) {
		dir := t.TempDir()
		specPath := writeSpecFile(t, dir, "garbage.json", spec)
		if err := os.WriteFile(StateFilePath(specPath), []byte("{not json"), 0o644); err != nil {
			t.Fatalf("write garbage state: %v", err)
		}
		if _, _, err := LoadState(specPath, spec); err == nil {
			t.Fatal("LoadState accepted unparseable state")
		}
	})

	t.Run("stream id mismatch", func(t *testing.T) {
		dir := t.TempDir()
		specPath := writeSpecFile(t, dir, "mismatch.json", spec)
		other := NewStreamState(StreamSpec{ID: "stream-other", Intent: "other"}.NormalizedSpec())
		if err := SaveState(specPath, other, time.Now()); err != nil {
			t.Fatalf("SaveState: %v", err)
		}
		_, _, err := LoadState(specPath, spec)
		if err == nil {
			t.Fatal("LoadState accepted state for another stream")
		}
		if !strings.Contains(err.Error(), "stream-other") || !strings.Contains(err.Error(), "stream-corrupt") {
			t.Fatalf("error %q must name both ids", err)
		}
	})
}

func TestRegistryServesWithoutSpecFiles(t *testing.T) {
	t.Chdir(t.TempDir())

	spec := StreamSpec{
		ID:     "stream-served",
		Intent: "served",
		Queue:  []WorkItem{{ID: "task-1", Lane: "docs"}, {ID: "task-2", Lane: "gateway"}},
	}
	specPath := writeSpecFile(t, ".", "served.json", spec)
	state := NewStreamState(spec.NormalizedSpec())
	state.Queue[0].Status = ItemCompleted
	state.CompletedCount = 1
	state.TotalTurnsSpent = 4
	state.ActiveIndex = 1

	now := time.Now()
	if err := SaveState(specPath, state, now); err != nil {
		t.Fatalf("SaveState: %v", err)
	}
	if err := UpsertRegistryEntry(SummaryFor(specPath, spec.NormalizedSpec(), state, now), now); err != nil {
		t.Fatalf("UpsertRegistryEntry: %v", err)
	}

	// Neither the spec nor the state file exists any more: the served read
	// must still answer, because it reads only the registry.
	if err := os.Remove(specPath); err != nil {
		t.Fatalf("remove spec: %v", err)
	}
	if err := os.Remove(StateFilePath(specPath)); err != nil {
		t.Fatalf("remove state: %v", err)
	}

	entries, err := LoadRegistry()
	if err != nil {
		t.Fatalf("LoadRegistry: %v", err)
	}
	if len(entries) != 1 {
		t.Fatalf("entries = %d, want 1", len(entries))
	}
	e := entries[0]
	if e.StreamID != "stream-served" || e.SpecPath != specPath {
		t.Fatalf("entry = %+v, want stream-served at %s", e, specPath)
	}
	if e.Completed != 1 || e.TotalItems != 2 || e.ActiveItemID != "task-2" {
		t.Fatalf("entry counters = %d/%d active %q, want 1/2 active task-2", e.Completed, e.TotalItems, e.ActiveItemID)
	}
}

func TestLoadRegistryMissingIsEmpty(t *testing.T) {
	t.Chdir(t.TempDir())
	entries, err := LoadRegistry()
	if err != nil {
		t.Fatalf("LoadRegistry: %v", err)
	}
	if entries != nil {
		t.Fatalf("entries = %v, want nil for a missing registry", entries)
	}
}

func TestUpsertRegistryReplacesByID(t *testing.T) {
	t.Chdir(t.TempDir())
	now := time.Now()

	base := RegistrySummary{StreamID: "stream-a", SpecPath: "a.json", TotalItems: 3, Completed: 0}
	if err := UpsertRegistryEntry(base, now); err != nil {
		t.Fatalf("UpsertRegistryEntry: %v", err)
	}
	base.Completed = 2
	if err := UpsertRegistryEntry(base, now); err != nil {
		t.Fatalf("UpsertRegistryEntry (update): %v", err)
	}

	entries, err := LoadRegistry()
	if err != nil {
		t.Fatalf("LoadRegistry: %v", err)
	}
	if len(entries) != 1 {
		t.Fatalf("entries = %d, want 1 (replace by stream id)", len(entries))
	}
	if entries[0].Completed != 2 {
		t.Fatalf("Completed = %d, want 2", entries[0].Completed)
	}

	second := RegistrySummary{StreamID: "stream-b", SpecPath: "b.json", TotalItems: 1, Closed: true}
	if err := UpsertRegistryEntry(second, now); err != nil {
		t.Fatalf("UpsertRegistryEntry (second): %v", err)
	}
	entries, err = LoadRegistry()
	if err != nil {
		t.Fatalf("LoadRegistry: %v", err)
	}
	if len(entries) != 2 {
		t.Fatalf("entries = %d, want 2 distinct streams", len(entries))
	}
}

func TestLoadRegistryCorruptRefused(t *testing.T) {
	t.Chdir(t.TempDir())
	if err := os.MkdirAll(filepath.Dir(RegistryFilePath()), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(RegistryFilePath(), []byte(`{"schema":"wrong","streams":[]}`), 0o644); err != nil {
		t.Fatalf("write registry: %v", err)
	}
	if _, err := LoadRegistry(); err == nil {
		t.Fatal("LoadRegistry accepted a foreign schema")
	}
}
