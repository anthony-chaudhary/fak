package sessionaudit

import (
	"encoding/json"
	"reflect"
	"strconv"
	"testing"
)

// TestBehaviorLensMatchesPythonReference pins the ported BehaviorLens detectors
// against the golden `behavior` dict that `python tools/session_audit.py deep`
// emits for the shared fixture testdata/behaviorlens.jsonl. The fixture exercises
// every detector: a successful read-storm (now intentionally benign), a verbatim
// repeat-failure loop (3× same Bash command+error), per-file mutation churn (5×
// Edit of one region), a timeout kill, a foreground sleep-poll, a ≥300s stall
// gap, and all three not-read sub-classes (post_resume / self_duplicate /
// true_never_read). The expected values below are copied verbatim from the
// Python reference output (see the issue #3097 acceptance witness).
func TestBehaviorLensMatchesPythonReference(t *testing.T) {
	s := Analyze("testdata/behaviorlens.jsonl")
	if s.Error != "" {
		t.Fatalf("analyze: %v", s.Error)
	}
	want := Behavior{
		ToolErrors:       map[string]int64{"Bash": 4, "Edit": 3},
		TimeoutKills:     1,
		SleepPolls:       1,
		EditChurn:        map[string]int64{"not_read": 3},
		NotReadClasses:   map[string]int64{"post_resume": 1, "self_duplicate": 1, "true_never_read": 1},
		RepeatFailures:   []RepeatFailureRow{{Tool: "Bash", Sig: "FAIL: boom", Count: 3}},
		MaxRepeatFailure: 3,
		FailureMass:      []RepeatFailureRow{{Tool: "Bash", Sig: "FAIL: boom", Count: 3}},
		MaxFailureMass:   3,
		FileChurn:        []FileChurnRow{{File: "/repo/C.go", Count: 5, DistinctRegions: 1, Reverts: 0}},
		MaxFileChurn:     5,
		SuccessLoops:     []SuccessLoopRow{},
		MaxSuccessLoop:   1,
		StallGaps:        1,
		MaxGapS:          600.0,
	}
	if !reflect.DeepEqual(s.Behavior, want) {
		t.Fatalf("behavior mismatch:\n got  %+v\n want %+v", s.Behavior, want)
	}
}

// TestFileChurnRevertArmRegionAware pins the #3943 fix: the lone-revert arm of
// the rewrite-loop detector is region-aware. A single revert amid all-distinct
// regions (distinct == count) is a long linear refactor that restores one
// earlier snippet once — healthy build-out, NOT a loop (the b72e2808 false
// positive). A revert WITH region reuse (distinct < count) stays flagged as
// genuine thrash (5c72b8ba). Mirrors the Python _churn_rows anchor cases so the
// Go and Python flag decisions agree byte-for-byte.
func TestFileChurnRevertArmRegionAware(t *testing.T) {
	edit := func(old, nw string) json.RawMessage {
		b, _ := json.Marshal(map[string]string{
			"file_path": "/repo/f.go", "old_string": old, "new_string": nw,
		})
		return b
	}
	churn := func(edits [][2]string) []FileChurnRow {
		l := newBehaviorLens()
		for _, e := range edits {
			l.noteToolUse("", "Edit", edit(e[0], e[1]), "")
		}
		return l.summary().FileChurn
	}

	// b72e2808 shape: count=5, distinct=5, reverts=1 -> NOT flagged.
	if rows := churn([][2]string{{"A", "x"}, {"B", "y"}, {"C", "z"}, {"D", "A"}, {"E", "w"}}); len(rows) != 0 {
		t.Fatalf("lone revert amid all-distinct regions must not flag, got %+v", rows)
	}

	// 5c72b8ba shape: count=5, distinct=4, reverts=2 -> flagged.
	thrash := churn([][2]string{{"A", "B"}, {"C", "D"}, {"B", "A"}, {"D", "C"}, {"A", "E"}})
	if len(thrash) != 1 {
		t.Fatalf("region-reuse thrash must flag exactly one file, got %+v", thrash)
	}
	if thrash[0].Count != 5 || thrash[0].DistinctRegions != 4 || thrash[0].Reverts != 2 {
		t.Fatalf("unexpected churn row: %+v", thrash[0])
	}

	// distinct-region build-out (6 distinct regions, 0 reverts) stays unflagged.
	if rows := churn([][2]string{{"a", "1"}, {"b", "2"}, {"c", "3"}, {"d", "4"}, {"e", "5"}, {"g", "6"}}); len(rows) != 0 {
		t.Fatalf("distinct-region build-out must not flag, got %+v", rows)
	}
}

// TestBehaviorLensEmptyTranscript proves the detectors default cleanly (empty
// maps, empty slices, zero maxima) on a transcript with no tool activity — the
// Python summary() emits the same shape via its `default=0` / dict() folds.
func TestBehaviorLensEmptyTranscript(t *testing.T) {
	recs := []map[string]any{
		assistantRecord("only-turn", 100, 0, 0),
	}
	b := Analyze(writeTranscript(t, recs)).Behavior
	if len(b.ToolErrors) != 0 || len(b.EditChurn) != 0 || len(b.NotReadClasses) != 0 {
		t.Fatalf("expected empty maps, got %+v", b)
	}
	if len(b.RepeatFailures) != 0 || len(b.FailureMass) != 0 || len(b.FileChurn) != 0 || len(b.SuccessLoops) != 0 {
		t.Fatalf("expected empty rows, got %+v", b)
	}
	if b.MaxRepeatFailure != 0 || b.MaxFailureMass != 0 || b.MaxFileChurn != 0 || b.MaxSuccessLoop != 0 ||
		b.StallGaps != 0 || b.MaxGapS != 0 || b.TimeoutKills != 0 || b.SleepPolls != 0 {
		t.Fatalf("expected zero maxima/counters, got %+v", b)
	}
}

func TestRepeatBenignToolFailuresDoNotScoreAsLoops(t *testing.T) {
	const calls = 12
	benign := newBehaviorLens()
	unsafe := newBehaviorLens()
	for i := 0; i < calls; i++ {
		benign.noteToolUse("read-"+strconv.Itoa(i), "Read", json.RawMessage(`{"file_path":"large.log","offset":1}`), `{"file_path":"large.log","offset":1}`)
		benign.noteToolResult("read-"+strconv.Itoa(i), true, "same transient failure")
		unsafe.noteToolUse("bash-"+strconv.Itoa(i), "Bash", json.RawMessage(`{"command":"exit 1"}`), `{"command":"exit 1"}`)
		unsafe.noteToolResult("bash-"+strconv.Itoa(i), true, "same transient failure")
	}

	gotBenign := benign.summary()
	if gotBenign.MaxRepeatFailure != 0 || gotBenign.MaxFailureMass != 0 || len(gotBenign.RepeatFailures) != 0 {
		t.Fatalf("12 repeated benign Read failures scored as a loop: %+v", gotBenign)
	}
	if gotBenign.ToolErrors["Read"] != calls {
		t.Fatalf("benign exclusion must not hide real error count: got %d want %d", gotBenign.ToolErrors["Read"], calls)
	}

	gotUnsafe := unsafe.summary()
	if gotUnsafe.MaxRepeatFailure != calls || gotUnsafe.MaxFailureMass != calls {
		t.Fatalf("12 repeated non-benign Bash failures scored repeat=%d mass=%d, want %d", gotUnsafe.MaxRepeatFailure, gotUnsafe.MaxFailureMass, calls)
	}
	if len(gotUnsafe.RepeatFailures) != 1 || gotUnsafe.RepeatFailures[0].Count != calls {
		t.Fatalf("non-benign repetition row = %+v, want one row scoring %d", gotUnsafe.RepeatFailures, calls)
	}
}
