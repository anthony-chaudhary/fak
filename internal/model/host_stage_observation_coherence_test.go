package model

import "testing"

// TestSessionHostStageObservationCoherence pins pub#14108: the session carries
// exactly one hostStages set — mutex-guarded, written by the v41 dense-host
// receipt path and read back by HostStages — plus the independent
// request-observer callback that stays outside that set. A redeclared
// hostStages field (the Oct 26 inert map vs the Oct 28 guarded pair) breaks
// the whole package build; driving both halves of the surviving surface keeps
// a future split-brain merge visible here instead of only at compile time.
func TestSessionHostStageObservationCoherence(t *testing.T) {
	s := &Session{}
	var observed []string
	s.hostStageObserver = func(stage string) { observed = append(observed, stage) }

	// Recorder half: a successful dense host note lands in the guarded set.
	s.noteV41DenseHostSuccess()
	if got := s.HostStages(); len(got) != 1 || got[0] != v41DenseHostStage {
		t.Fatalf("HostStages()=%v, want [%s]", got, v41DenseHostStage)
	}

	// Observer half: request observations stay on the session, outside the set.
	s.observeHostArm("embedding-panel")
	if len(observed) != 1 || observed[0] != "embedding-panel" {
		t.Fatalf("observer stages=%v, want [embedding-panel]", observed)
	}
	if got := s.HostStages(); len(got) != 1 || got[0] != v41DenseHostStage {
		t.Fatalf("observer callback leaked into HostStages(): %v", got)
	}

	// Idempotence: repeating the success note does not grow the set.
	s.noteV41DenseHostSuccess()
	if got := s.HostStages(); len(got) != 1 {
		t.Fatalf("HostStages() after repeat note=%v, want single entry", got)
	}
}
