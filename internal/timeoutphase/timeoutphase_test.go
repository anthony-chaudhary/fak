package timeoutphase

import "testing"

// TestClassify_AllFivePhases is the closed-vocabulary coverage witness: one fixture per named
// phase, including the two that require no observed Stage marker (before_startup needs
// Started=true with no stage; unknown needs neither).
func TestClassify_AllFivePhases(t *testing.T) {
	cases := []struct {
		name string
		in   Attempt
		want Phase
	}{
		{"no evidence at all", Attempt{ID: "1"}, PhaseUnknown},
		{"started, no stage marker yet", Attempt{ID: "2", Started: true}, PhaseBeforeStartup},
		{"last stage edit", Attempt{ID: "3", Started: true, LastStage: StageEdit}, PhaseDuringEdit},
		{"last stage test", Attempt{ID: "4", Started: true, LastStage: StageTest}, PhaseDuringTests},
		{"last stage commit", Attempt{ID: "5", Started: true, LastStage: StageCommit}, PhaseDuringCommit},
		{"last stage push", Attempt{ID: "6", Started: true, LastStage: StagePush}, PhaseDuringPush},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := Classify(c.in)
			if got.Phase != c.want {
				t.Fatalf("Classify(%+v).Phase = %q, want %q", c.in, got.Phase, c.want)
			}
			if got.ID != c.in.ID {
				t.Fatalf("Classify(%+v).ID = %q, want %q", c.in, got.ID, c.in.ID)
			}
		})
	}
}

// TestClassify_UnrecognizedStageDoesNotCrash is the no-crash edge case: an attempt reporting a
// stage token this package does not recognize must still classify to a defined phase (falling
// back to the Started/no-stage-marker rule), never panic.
func TestClassify_UnrecognizedStageDoesNotCrash(t *testing.T) {
	got := Classify(Attempt{ID: "7", Started: true, LastStage: Stage("some-unknown-stage")})
	if got.Phase != PhaseBeforeStartup {
		t.Fatalf("unrecognized stage: Phase = %q, want %q (falls back as if no stage marker fired)", got.Phase, PhaseBeforeStartup)
	}

	got2 := Classify(Attempt{ID: "8", LastStage: Stage("bogus")})
	if got2.Phase != PhaseUnknown {
		t.Fatalf("unrecognized stage + not started: Phase = %q, want %q", got2.Phase, PhaseUnknown)
	}
}

// TestClassify_CopiesThroughFields confirms FailureClass/LastStage/TimestampUnix survive onto
// the Row unchanged, since Record/the CLI shell depend on them for rendering.
func TestClassify_CopiesThroughFields(t *testing.T) {
	got := Classify(Attempt{ID: "9", Started: true, LastStage: StageTest, FailureClass: "flaky_test", TimestampUnix: 4242})
	if got.FailureClass != "flaky_test" {
		t.Fatalf("FailureClass = %q, want %q", got.FailureClass, "flaky_test")
	}
	if got.LastStage != StageTest {
		t.Fatalf("LastStage = %q, want %q", got.LastStage, StageTest)
	}
	if got.TimestampUnix != 4242 {
		t.Fatalf("TimestampUnix = %d, want 4242", got.TimestampUnix)
	}
}

// TestRecord_EmitsAtLeastTwoDistinctPhases is the issue's explicit witness requirement (#1793):
// a fixture must emit at least two distinct timeout phases in one Report.
func TestRecord_EmitsAtLeastTwoDistinctPhases(t *testing.T) {
	rep := Record([]Attempt{
		{ID: "1", Started: false},
		{ID: "2", Started: true, LastStage: StageCommit},
		{ID: "3", Started: true, LastStage: StagePush},
	})
	if len(rep.Rows) != 3 {
		t.Fatalf("want 3 rows, got %d", len(rep.Rows))
	}
	distinct := map[Phase]bool{}
	for _, row := range rep.Rows {
		distinct[row.Phase] = true
	}
	if len(distinct) < 2 {
		t.Fatalf("want at least 2 distinct phases, got %v", distinct)
	}
	if rep.PhaseCount[PhaseUnknown] != 1 || rep.PhaseCount[PhaseDuringCommit] != 1 || rep.PhaseCount[PhaseDuringPush] != 1 {
		t.Fatalf("PhaseCount = %+v, want one each of unknown/during_commit/during_push", rep.PhaseCount)
	}
}

// TestRecord_EmptyInput confirms the zero-attempts case is a valid, non-nil empty report.
func TestRecord_EmptyInput(t *testing.T) {
	rep := Record(nil)
	if len(rep.Rows) != 0 {
		t.Fatalf("want 0 rows, got %d", len(rep.Rows))
	}
	if rep.PhaseCount == nil {
		t.Fatalf("want a non-nil (possibly empty) PhaseCount map")
	}
}
