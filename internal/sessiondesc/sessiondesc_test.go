package sessiondesc

import (
	"encoding/json"
	"strings"
	"testing"
)

// TestFoldJoinsExactIDsAcrossSpaces pins the core join: the SAME id observed
// via drive state, leaseref, and a harness binding folds to ONE descriptor
// with three bound spaces; ids present in only one source stay their own
// descriptors with typed misses.
func TestFoldJoinsExactIDsAcrossSpaces(t *testing.T) {
	src := Sources{
		DriveStatus: SourceObserved,
		Drive: []DriveRow{
			{TraceID: "sess-a", Run: "running", Generation: 2, Rev: 7},
			{TraceID: "sess-drive-only", Run: "paused"},
		},
		RefStatus: SourceObserved,
		Refs: []RefRow{
			{ID: "sess-a", Host: "node-1", PCBState: "RUNNING", UpdatedAt: 100, TTLSecs: 300},
			{ID: "sess-ref-only", Host: "node-2", PCBState: "STOPPED"},
		},
		HarnessStatus: SourceObserved,
		Harness: []HarnessRow{
			{SessionID: "sess-a", Agent: "claude", Identity: "worker7"},
		},
	}
	ds, err := Fold(src)
	if err != nil {
		t.Fatalf("Fold: %v", err)
	}
	if len(ds) != 3 {
		t.Fatalf("want 3 descriptors, got %d: %+v", len(ds), ds)
	}
	// Sorted by id: sess-a, sess-drive-only, sess-ref-only.
	a := ds[0]
	if a.ID != "sess-a" || a.BoundCount() != 3 {
		t.Fatalf("sess-a should bind drive+ref+harness, got %+v", a)
	}
	if a.Drive.Presence != Bound || a.Drive.Rev != 7 || a.Drive.Generation != 2 {
		t.Fatalf("sess-a drive key wrong: %+v", a.Drive)
	}
	if a.Ref.Presence != Bound || a.Ref.Host != "node-1" {
		t.Fatalf("sess-a ref key wrong: %+v", a.Ref)
	}
	if a.Harness.Presence != Bound || a.Harness.Agent != "claude" {
		t.Fatalf("sess-a harness key wrong: %+v", a.Harness)
	}
	if a.Census.Presence != AbsentNotObserved {
		t.Fatalf("census is reserved in v1, want ABSENT_NOT_OBSERVED, got %q", a.Census.Presence)
	}

	driveOnly := ds[1]
	if driveOnly.ID != "sess-drive-only" || driveOnly.Drive.Presence != Bound {
		t.Fatalf("drive-only descriptor wrong: %+v", driveOnly)
	}
	// Both other sources WERE observed and held nothing: a clean, typed miss.
	if driveOnly.Ref.Presence != AbsentNoBinding || driveOnly.Harness.Presence != AbsentNoBinding {
		t.Fatalf("drive-only misses must be ABSENT_NO_BINDING, got ref=%q harness=%q",
			driveOnly.Ref.Presence, driveOnly.Harness.Presence)
	}

	refOnly := ds[2]
	if refOnly.ID != "sess-ref-only" || refOnly.Ref.Presence != Bound || refOnly.BoundCount() != 1 {
		t.Fatalf("ref-only descriptor wrong: %+v", refOnly)
	}
}

// TestFoldNeverMergesDistinctSessions is the collision fence from #2214:
// two different ids never fold into one descriptor, however similar.
func TestFoldNeverMergesDistinctSessions(t *testing.T) {
	src := Sources{
		DriveStatus: SourceObserved,
		Drive:       []DriveRow{{TraceID: "sess-a"}},
		RefStatus:   SourceObserved,
		Refs:        []RefRow{{ID: "sess-A", Host: "node-1"}}, // case differs: a DIFFERENT session
	}
	ds, err := Fold(src)
	if err != nil {
		t.Fatalf("Fold: %v", err)
	}
	if len(ds) != 2 {
		t.Fatalf("byte-unequal ids must stay distinct descriptors, got %d: %+v", len(ds), ds)
	}
}

// TestFoldAbsenceVocabularyPerSourceStatus pins that a miss carries the reason
// its SOURCE dictates: unavailable != unconsulted != observed-empty.
func TestFoldAbsenceVocabularyPerSourceStatus(t *testing.T) {
	src := Sources{
		DriveStatus:   SourceObserved,
		Drive:         []DriveRow{{TraceID: "sess-a"}},
		RefStatus:     SourceUnavailable,  // consulted, FAILED
		HarnessStatus: SourceNotConsulted, // never consulted
	}
	ds, err := Fold(src)
	if err != nil {
		t.Fatalf("Fold: %v", err)
	}
	if len(ds) != 1 {
		t.Fatalf("want 1 descriptor, got %d", len(ds))
	}
	d := ds[0]
	if d.Ref.Presence != AbsentSourceUnavailable {
		t.Fatalf("unavailable source must yield ABSENT_SOURCE_UNAVAILABLE, got %q", d.Ref.Presence)
	}
	if d.Harness.Presence != AbsentNotObserved {
		t.Fatalf("unconsulted source must yield ABSENT_NOT_OBSERVED, got %q", d.Harness.Presence)
	}
}

// TestFoldRefusesEmptyAndDuplicateIDs: an unidentifiable row and a same-source
// duplicate are ERRORS, not best-effort folds — silently merging them is the
// mis-binding failure mode this schema exists to prevent.
func TestFoldRefusesEmptyAndDuplicateIDs(t *testing.T) {
	if _, err := Fold(Sources{DriveStatus: SourceObserved, Drive: []DriveRow{{TraceID: ""}}}); err == nil {
		t.Fatal("empty trace_id must refuse")
	}
	if _, err := Fold(Sources{RefStatus: SourceObserved, Refs: []RefRow{{ID: ""}}}); err == nil {
		t.Fatal("empty ref id must refuse")
	}
	if _, err := Fold(Sources{HarnessStatus: SourceObserved, Harness: []HarnessRow{{SessionID: ""}}}); err == nil {
		t.Fatal("empty harness session_id must refuse")
	}
	if _, err := Fold(Sources{
		DriveStatus: SourceObserved,
		Drive:       []DriveRow{{TraceID: "sess-a"}, {TraceID: "sess-a"}},
	}); err == nil {
		t.Fatal("duplicate drive rows for one id must refuse")
	}
}

// TestDescriptorGoldenJSON pins the v1 wire shape byte-for-byte (field names,
// presence tokens, schema tag) so an accidental rename is a red test, not a
// silent consumer break. The golden bytes are the contract the doc cites.
func TestDescriptorGoldenJSON(t *testing.T) {
	src := Sources{
		DriveStatus:   SourceObserved,
		Drive:         []DriveRow{{TraceID: "sess-a", Run: "running", ContinuationID: "sess-prev", ParentTrace: "sess-root", Generation: 3, Rev: 9}},
		RefStatus:     SourceObserved,
		Refs:          []RefRow{{ID: "sess-a", Host: "node-1", PCBState: "RUNNING", UpdatedAt: 1751328000, TTLSecs: 600}},
		HarnessStatus: SourceObserved,
		Harness:       []HarnessRow{{SessionID: "sess-a", Agent: "codex", Identity: "acct-3"}},
	}
	ds, err := Fold(src)
	if err != nil {
		t.Fatalf("Fold: %v", err)
	}
	got, err := json.Marshal(ds[0])
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	want := strings.Join([]string{
		`{"schema":"fak.session.descriptor.v1",`,
		`"id":"sess-a",`,
		`"drive":{"presence":"BOUND","trace_id":"sess-a","run":"running","continuation_id":"sess-prev","parent_trace":"sess-root","generation":3,"rev":9},`,
		`"ref":{"presence":"BOUND","id":"sess-a","host":"node-1","pcb_state":"RUNNING","updated_at":1751328000,"ttl_seconds":600},`,
		`"harness":{"presence":"BOUND","agent":"codex","identity":"acct-3"},`,
		`"census":{"presence":"ABSENT_NOT_OBSERVED"}}`,
	}, "")
	if string(got) != want {
		t.Fatalf("golden wire shape drifted:\n got: %s\nwant: %s", got, want)
	}
}

// TestFoldEmptySourcesIsEmptyNotError: nothing observed anywhere is a valid
// zero-session fact (the fail-closed empty view), not an error.
func TestFoldEmptySourcesIsEmptyNotError(t *testing.T) {
	ds, err := Fold(Sources{})
	if err != nil {
		t.Fatalf("Fold on empty sources: %v", err)
	}
	if len(ds) != 0 {
		t.Fatalf("want empty, got %+v", ds)
	}
}
