package sessionaudit

import (
	"reflect"
	"testing"
)

// TestBehaviorLensMatchesPythonReference pins the ported BehaviorLens detectors
// against the golden `behavior` dict that `python tools/session_audit.py deep`
// emits for the shared fixture testdata/behaviorlens.jsonl. The fixture exercises
// every detector: a successful read-storm (success loop 8), a verbatim
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
		SuccessLoops:     []SuccessLoopRow{{Tool: "Read", Target: "/repo/R.go", Count: 8}},
		MaxSuccessLoop:   8,
		StallGaps:        1,
		MaxGapS:          600.0,
	}
	if !reflect.DeepEqual(s.Behavior, want) {
		t.Fatalf("behavior mismatch:\n got  %+v\n want %+v", s.Behavior, want)
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
