package main

import (
	"path/filepath"
	"testing"
)

// guardStopCountsFrom folds the guard-stops ledger into the operator-question counts
// that ride on the harness-resources row. It must agree with `fak guard-stops` (same
// fold) and be zero-safe: no ledger / unreadable / empty all read as an explicit zero,
// never a gap (#4348).
func TestGuardStopCountsFromLedger(t *testing.T) {
	dir := t.TempDir()
	ledger := filepath.Join(dir, "nested", "guard-stops.jsonl")
	rows := []guardStopRecord{
		// two turn-ends that asked a human instead of acting (operator-directed)
		{Ts: "2026-07-01T00:00:00Z", Disposition: string(stopDispOperatorDirectedEscalate), Kind: string(stopKindClean),
			Transcript: &guardStopTranscript{Read: true, OperatorDirected: true}},
		{Ts: "2026-07-01T00:01:00Z", Disposition: string(stopDispOperatorDirectedContinue), Kind: string(stopKindContinue), Blocked: true,
			Transcript: &guardStopTranscript{Read: true, OperatorDirected: true}},
		// one fail-open stop
		{Ts: "2026-07-01T00:02:00Z", Disposition: string(stopDispFailOpenGaugeUnavailable), Kind: string(stopKindFailOpen)},
		// a clean stop that touches neither count
		{Ts: "2026-07-01T00:03:00Z", Disposition: string(stopDispCleanCompletion), Kind: string(stopKindClean)},
	}
	for _, r := range rows {
		r := r
		if err := appendGuardStopRecord(ledger, r); err != nil {
			t.Fatalf("append: %v", err)
		}
	}

	got := guardStopCountsFrom(ledger)
	if got.OperatorDirected != 2 {
		t.Errorf("OperatorDirected = %d, want 2", got.OperatorDirected)
	}
	if got.FailOpen != 1 {
		t.Errorf("FailOpen = %d, want 1", got.FailOpen)
	}
	// The fold must match summarizeGuardStops directly (same ledger, same numbers).
	content, err := readGuardStopsLedger(ledger)
	if err != nil {
		t.Fatal(err)
	}
	sum := summarizeGuardStops(content, 0)
	if got.OperatorDirected != sum.OperatorDirected || got.FailOpen != sum.FailOpen {
		t.Errorf("fold diverged from summarizeGuardStops: got %+v, sum od=%d fo=%d", got, sum.OperatorDirected, sum.FailOpen)
	}
}

func TestGuardStopCountsFromZeroSafe(t *testing.T) {
	// No ledger path wired → explicit zero, not a failure.
	if got := guardStopCountsFrom(""); got.OperatorDirected != 0 || got.FailOpen != 0 {
		t.Errorf("empty path: got %+v, want zero", got)
	}
	// A path that does not exist → zero (a fresh session has recorded nothing).
	missing := filepath.Join(t.TempDir(), "does-not-exist.jsonl")
	if got := guardStopCountsFrom(missing); got.OperatorDirected != 0 || got.FailOpen != 0 {
		t.Errorf("missing ledger: got %+v, want zero", got)
	}
}
