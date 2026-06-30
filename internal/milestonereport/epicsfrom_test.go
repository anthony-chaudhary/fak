package milestonereport

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// seedPath resolves the committed seed relative to the module root (this test file
// lives two levels down at internal/milestonereport).
func seedPath(t *testing.T) string {
	t.Helper()
	return filepath.Join("..", "..", filepath.FromSlash(DefaultTrackedEpicsRel))
}

// TestSeedMirrorsTrackedEpics proves the committed seed loads to EXACTLY the in-code
// TrackedEpics default — so `--epics-from docs/milestones/tracked-epics.json` is
// behavior-identical to the zero-config default. If a tracked epic is added to one
// surface and not the other, this fails.
func TestSeedMirrorsTrackedEpics(t *testing.T) {
	f, err := LoadEpicsFile(seedPath(t))
	if err != nil {
		t.Fatalf("load seed: %v", err)
	}
	if !reflect.DeepEqual(f.Specs, TrackedEpics) {
		t.Fatalf("seed specs drifted from the in-code TrackedEpics:\n  seed: %+v\n  code: %+v", f.Specs, TrackedEpics)
	}
	if f.HasCounts() {
		t.Fatalf("seed must be specs-only (resolve live), not carry pre-resolved counts: %+v", f.Counts)
	}
}

// TestLoadEpicsFileCustomOverrides proves a hand-authored data file overrides the
// tracked set — the reviewable-diff property: editing the file changes WHICH epics
// the report tracks, with no code edit.
func TestLoadEpicsFileCustomOverrides(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "custom.json")
	body := `{
  "specs": [
    {"Number": 42, "Title": "answer", "Label": "life"},
    {"Number": 7,  "Title": "luck"}
  ]
}`
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	f, err := LoadEpicsFile(path)
	if err != nil {
		t.Fatalf("load custom: %v", err)
	}
	want := []EpicSpec{
		{Number: 42, Title: "answer", Label: "life"},
		{Number: 7, Title: "luck"},
	}
	if !reflect.DeepEqual(f.Specs, want) {
		t.Fatalf("custom specs = %+v, want %+v", f.Specs, want)
	}
	if f.HasCounts() {
		t.Fatalf("specs-only file must not report counts")
	}
}

// TestEpicsFilePreResolvedCountsFoldsOffline proves a file carrying a pre-resolved
// `counts` block folds DETERMINISTICALLY with no `gh` — the hermetic override the
// acceptance criterion requires. The fold result must match the same specs+counts
// folded directly through CountsFromSpecs.
func TestEpicsFilePreResolvedCountsFoldsOffline(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "offline.json")
	body := `{
  "specs": [
    {"Number": 100, "Title": "by-label", "Label": "track-x"},
    {"Number": 200, "Title": "by-checklist"}
  ],
  "counts": [
    {"Number": 100, "Closed": 3, "Total": 4, "Source": "label"},
    {"Number": 200, "Closed": 1, "Total": 2, "Source": "checklist"}
  ]
}`
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	f, err := LoadEpicsFile(path)
	if err != nil {
		t.Fatalf("load offline: %v", err)
	}
	if !f.HasCounts() {
		t.Fatalf("file with a counts block must report HasCounts()")
	}

	got := f.FoldOffline()
	// Must equal folding the same data straight through the pure interpreter — proving
	// FoldOffline is exactly CountsFromSpecs over the file's own data, no gh path.
	want := CountsFromSpecs(f.Specs, f.Counts)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("offline fold != CountsFromSpecs:\n  got:  %+v\n  want: %+v", got, want)
	}
	if got.Tracked != 2 || got.Measured != 2 {
		t.Fatalf("offline fold tracked/measured = %d/%d, want 2/2 (%+v)", got.Tracked, got.Measured, got)
	}
	byNum := map[int]EpicRow{}
	for _, row := range got.Rows {
		byNum[row.Number] = row
	}
	if r := byNum[100]; r.Source != "label" || r.Closed != 3 || r.Total != 4 {
		t.Fatalf("#100 offline fold = %+v, want label 3/4", r)
	}
	if r := byNum[200]; r.Source != "checklist" || r.Closed != 1 || r.Total != 2 {
		t.Fatalf("#200 offline fold = %+v, want checklist 1/2", r)
	}
}

// TestLoadEpicsFileErrors proves a missing file and an empty spec set are errors,
// not a silent fallback — a typo'd --epics-from path must fail loudly, never track
// the wrong set.
func TestLoadEpicsFileErrors(t *testing.T) {
	if _, err := LoadEpicsFile(filepath.Join(t.TempDir(), "nope.json")); err == nil {
		t.Fatalf("missing file must error")
	}
	dir := t.TempDir()
	empty := filepath.Join(dir, "empty.json")
	if err := os.WriteFile(empty, []byte(`{"specs": []}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadEpicsFile(empty); err == nil {
		t.Fatalf("empty spec set must error")
	}
}

// TestMarshalRoundTrip proves the seed can be regenerated from the in-code default and
// re-read identically — the inverse property that keeps the file and the code in sync.
func TestMarshalRoundTrip(t *testing.T) {
	src := EpicsFile{Schema: "fak-milestone-tracked-epics/1", Specs: TrackedEpics}
	b, err := MarshalEpicsFile(src)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "round.json")
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := LoadEpicsFile(path)
	if err != nil {
		t.Fatalf("reload: %v", err)
	}
	if !reflect.DeepEqual(got.Specs, TrackedEpics) {
		t.Fatalf("round-trip specs drifted: %+v", got.Specs)
	}
}
