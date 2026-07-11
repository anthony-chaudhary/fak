package main

import (
	"bytes"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/loopmgr"
	"github.com/anthony-chaudhary/fak/internal/seatpark"
)

// sweepSeatEvent is a dispatch-sweep run-end with the given park reason at tsUnix seconds.
func sweepSeatEvent(reason string, tsUnix int64) loopmgr.Event {
	return loopmgr.Event{
		LoopID:     dispatchSweepLoopID,
		Kind:       loopmgr.EventEnd,
		Reason:     reason,
		TSUnixNano: tsUnix * 1_000_000_000,
	}
}

func TestDeriveSweepSeatParkState_CountsSweepLoopNoSeatOnly(t *testing.T) {
	events := []loopmgr.Event{
		// a DIFFERENT loop's no-seat park must not leak into the sweep tail
		{LoopID: gardenDispatchLoopID, Kind: loopmgr.EventEnd, Reason: seatParkReasonNoSeat, TSUnixNano: 900 * 1_000_000_000},
		sweepSeatEvent(seatParkReasonNoSeat, 1000),
		sweepSeatEvent(string(seatpark.StatusParked), 1050), // a chosen wait — neutral in the tail
		sweepSeatEvent(seatParkReasonNoSeat, 1100),          // newest
	}
	parks, last := deriveSweepSeatParkState(events)
	if parks != 2 {
		t.Fatalf("parks = %d, want 2 (sweep-loop no-seat only; parked runs neutral)", parks)
	}
	if last != 1100 {
		t.Fatalf("lastParkUnix = %d, want 1100 (most recent no-seat)", last)
	}
}

func TestDeriveSweepSeatParkState_ProgressEndsTail(t *testing.T) {
	// A prior progress/other stop (Reason "") is a tail boundary: only runs AFTER it count.
	events := []loopmgr.Event{
		sweepSeatEvent(seatParkReasonNoSeat, 900), // before the boundary — not counted
		sweepSeatEvent("", 1000),                  // boundary
		sweepSeatEvent(seatParkReasonNoSeat, 1100),
	}
	if parks, _ := deriveSweepSeatParkState(events); parks != 1 {
		t.Fatalf("parks = %d, want 1 (only the run after the boundary)", parks)
	}
}

// TestRunDispatchSweep_ParksOnRecentNoSeatRefuse drives the sweep front-door: with a
// just-now no-seat park in the durable ledger, a LIVE sweep must PARK — return before
// RunSweep evaluates a single tick — rather than burst another spawn attempt against a
// wall only a peer finishing can move. Hermetic: the park gate returns before any tick.
func TestRunDispatchSweep_ParksOnRecentNoSeatRefuse(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "loops.jsonl")
	if _, err := loopmgr.Append(ledger, loopmgr.Event{
		LoopID: dispatchSweepLoopID, Kind: loopmgr.EventEnd,
		Status: loopmgr.StatusWitnessedDone, Reason: seatParkReasonNoSeat, Summary: "seed",
	}); err != nil {
		t.Fatalf("seed ledger: %v", err)
	}
	var stdout, stderr bytes.Buffer
	code := runDispatchSweep(&stdout, &stderr, []string{
		"--workspace", t.TempDir(), "--live", "--ledger", ledger,
	})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	if !bytes.Contains(stdout.Bytes(), []byte(string(seatpark.StatusParked))) {
		t.Fatalf("a recent no-seat refuse must PARK before running a tick, got: %s", stdout.String())
	}
}

// TestRunDispatchSweep_ExhaustsAfterBoundedParks: once the bounded budget of no-seat parks
// is spent, the sweep stops re-offering (SEAT_EXHAUSTED) rather than parking forever.
func TestRunDispatchSweep_ExhaustsAfterBoundedParks(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "loops.jsonl")
	for i := 0; i < seatpark.DefaultMaxParks; i++ {
		if _, err := loopmgr.Append(ledger, loopmgr.Event{
			LoopID: dispatchSweepLoopID, Kind: loopmgr.EventEnd,
			Status: loopmgr.StatusWitnessedDone, Reason: seatParkReasonNoSeat, Summary: "seed",
		}); err != nil {
			t.Fatalf("seed %d: %v", i, err)
		}
	}
	var stdout, stderr bytes.Buffer
	code := runDispatchSweep(&stdout, &stderr, []string{
		"--workspace", t.TempDir(), "--live", "--ledger", ledger,
	})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	if !bytes.Contains(stdout.Bytes(), []byte(string(seatpark.StatusExhausted))) {
		t.Fatalf("after %d bounded parks the sweep must EXHAUST, got: %s", seatpark.DefaultMaxParks, stdout.String())
	}
}

func TestDispatchSweepRefreshesRegistryOnlyOnFirstTick(t *testing.T) {
	for _, tc := range []struct {
		iter int
		want bool
	}{{0, true}, {1, false}, {2, false}, {99, false}} {
		if got := dispatchSweepRefresh(tc.iter); got != tc.want {
			t.Errorf("dispatchSweepRefresh(%d) = %v, want %v", tc.iter, got, tc.want)
		}
	}
}
