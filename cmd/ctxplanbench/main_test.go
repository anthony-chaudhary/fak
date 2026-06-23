package main

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestReplayInvariants drives the full pipeline (ingest → recall reload → replay) over a
// small synthetic transcript of known shape and asserts the four invariants the measurement
// rests on: the O(1) bound holds, exact recall holds every turn, planned resident never
// exceeds the full transcript, and every forecast miss is SERVED (recoverable) — never lost.
// It is the test-form twin of the -selfcheck gate, so `go test ./cmd/ctxplanbench` (under
// WSL on this host) covers what an operator gets from `go run ./cmd/ctxplanbench -selfcheck`.
func TestReplayInvariants(t *testing.T) {
	tpath, cleanup := writeSelfcheckTranscript(t)
	defer cleanup()

	r, err := measureFile(context.Background(), tpath, 20, 6) // tight budget forces elision
	if err != nil {
		t.Fatalf("measureFile: %v", err)
	}
	if r.Turns < 3 {
		t.Fatalf("too few turns replayed (%d) — ingest/reload lost spans", r.Turns)
	}
	if r.PeakPlannedTok > r.Budget && r.OverBudgetTurns == 0 {
		t.Errorf("planned resident (%d) exceeded the budget (%d) without any pin-overrun turn — the O(1) bound broke", r.PeakPlannedTok, r.Budget)
	}
	if r.PlannedFaithfulTurns != r.Turns {
		t.Errorf("planned faithful %d/%d turns — exact recall did not hold every turn", r.PlannedFaithfulTurns, r.Turns)
	}
	if r.PlannedCumTok > r.LinearCumTok {
		t.Errorf("planned cum (%d) exceeded linear cum (%d) — planned resident never exceeds the full transcript", r.PlannedCumTok, r.LinearCumTok)
	}
	if r.CompactionLossTurns == 0 {
		t.Errorf("compaction never destroyed a fact — the tight budget did not force elision, so the contrast is vacuous")
	}
	// The recoverability contract: every fault must be served (a page fault, never a lost
	// fact). A refused fault is the gate holding on a sealed re-screen — not a lost fact,
	// but it must not happen on the benign synthetic transcript.
	if r.Served != r.Faults {
		t.Errorf("served (%d) != faults (%d) — a forecast miss was not recoverable", r.Served, r.Faults)
	}
	if r.FactsRecovered != r.Served {
		t.Errorf("factsRecovered (%d) != served (%d) — the quality tally drifted from the fault tally", r.FactsRecovered, r.Served)
	}
	if r.References == 0 {
		t.Errorf("zero references — the lexical back-reference signal did not fire on a transcript built to trigger it")
	}
}

// TestReplayLooseBudgetHoldsEverything checks the other end of the regime: a budget large
// enough to hold the whole transcript should make planned == linear (nothing elided) and
// produce zero faults — the degenerate "O(1) never engaged" case, which must still be
// faithful and never over-count a turn the workload did not contain.
func TestReplayLooseBudgetHoldsEverything(t *testing.T) {
	tpath, cleanup := writeSelfcheckTranscript(t)
	defer cleanup()

	r, err := measureFile(context.Background(), tpath, 100_000, 6)
	if err != nil {
		t.Fatalf("measureFile: %v", err)
	}
	if r.PlannedFaithfulTurns != r.Turns {
		t.Errorf("planned faithful %d/%d turns at a loose budget", r.PlannedFaithfulTurns, r.Turns)
	}
	if r.Faults != 0 {
		t.Errorf("loose budget still faulted (%d) — everything should have been resident", r.Faults)
	}
	if r.CompactionLossTurns != 0 {
		t.Errorf("loose budget still elided (%d turns) — nothing should have been cold", r.CompactionLossTurns)
	}
}

// writeSelfcheckTranscript materializes the synthetic known-shape transcript the selfcheck
// uses, so the test and the -selfcheck gate cannot drift apart. It returns the path and a
// cleanup.
func writeSelfcheckTranscript(t *testing.T) (string, func()) {
	t.Helper()
	const (
		durable = `{"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"u0","content":"preference: always deploy from the release branch, never main"}]}}`
		read    = `{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"u1","name":"Read"},{"type":"tool_result","tool_use_id":"u1","content":"auth token rotation runbook step one mint roll revoke"}]}}`
		noiseA  = `{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"u2","name":"Bash"},{"type":"tool_result","tool_use_id":"u2","content":"build log compiled 412 files quietly on tuesday afternoon"}]}}`
		noiseB  = `{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"u3","name":"Bash"},{"type":"tool_result","tool_use_id":"u3","content":"weather sunny light wind from the west nothing relevant"}]}}`
		edit    = `{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"u4","name":"Edit"},{"type":"tool_result","tool_use_id":"u4","content":"applied the auth token rotation runbook edit to the billing service"}]}}`
	)
	dir, err := os.MkdirTemp("", "ctxplanbench-test-*")
	if err != nil {
		t.Fatal(err)
	}
	tpath := filepath.Join(dir, "selfcheck.jsonl")
	if err := os.WriteFile(tpath, []byte(strings.Join([]string{durable, read, noiseA, noiseB, edit}, "\n")), 0o644); err != nil {
		os.RemoveAll(dir)
		t.Fatal(err)
	}
	return tpath, func() { os.RemoveAll(dir) }
}
