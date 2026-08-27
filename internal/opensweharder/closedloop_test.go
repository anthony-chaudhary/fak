package opensweharder

import (
	"encoding/json"
	"path/filepath"
	"reflect"
	"testing"
)

func TestFrozenClosedLoop(t *testing.T) {
	f, err := LoadFixture(filepath.Join("testdata", "frozen_tasks.json"))
	if err != nil {
		t.Fatal(err)
	}
	got, err := Run(f)
	if err != nil {
		t.Fatal(err)
	}

	if got.Mode != ModeSimulated {
		t.Fatalf("mode = %q, want honest simulated label", got.Mode)
	}
	if got.OfferedTasks != 4 || got.AcceptedTasks != 3 {
		t.Fatalf("denominator offered=%d accepted=%d", got.OfferedTasks, got.AcceptedTasks)
	}
	if got.Baseline != (Score{Passed: 1, Accepted: 3}) {
		t.Fatalf("baseline = %+v", got.Baseline)
	}
	if got.Candidate != (Score{Passed: 3, Accepted: 3}) {
		t.Fatalf("candidate = %+v", got.Candidate)
	}
	if got.Reversed != got.Baseline || !got.ReversalOK {
		t.Fatalf("reversal=%+v baseline=%+v ok=%v", got.Reversed, got.Baseline, got.ReversalOK)
	}
	if got.Decision != "accept_counter_hypothesis" {
		t.Fatalf("decision = %q", got.Decision)
	}
	wantIDs := []string{"open-swe-001", "open-swe-002", "swe-smith-hard-001"}
	if !reflect.DeepEqual(got.AcceptedIDs, wantIDs) {
		t.Fatalf("accepted ids = %v", got.AcceptedIDs)
	}
	if got.Hypothesis.Baseline == "" || got.Hypothesis.CounterHypothesis == "" {
		t.Fatal("hypotheses must be explicit")
	}
	if _, err := json.Marshal(got); err != nil {
		t.Fatal(err)
	}
}

func TestRejectsBrokenReversal(t *testing.T) {
	f := Fixture{Schema: "fak-openswe-harder-fixture/1", Mode: ModeSimulated, Tasks: []Task{{ID: "x", Source: "swe-smith", Difficulty: 5, Accepted: true, BaselinePass: true, CandidatePass: true, ReversalPass: false}}}
	// LoadFixture enforces the frozen input contract; Run remains fail-closed at decision time.
	got, err := Run(f)
	if err != nil {
		t.Fatal(err)
	}
	if got.Decision != "reject_reversal_failed" || got.ReversalOK {
		t.Fatalf("got decision=%q reversal_ok=%v", got.Decision, got.ReversalOK)
	}
}

func TestRetainsBaselineWithoutGain(t *testing.T) {
	f := Fixture{Mode: ModeLive, Tasks: []Task{{ID: "x", Source: "open-swe", Difficulty: 2, Accepted: true, BaselinePass: true, CandidatePass: true, ReversalPass: true}}}
	got, err := Run(f)
	if err != nil {
		t.Fatal(err)
	}
	if got.Mode != ModeLive || got.Decision != "retain_baseline" {
		t.Fatalf("got mode=%q decision=%q", got.Mode, got.Decision)
	}
}
