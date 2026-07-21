package gateway

import (
	"math"
	"testing"
)

// TestExpectedDrainTimeEqualInFlightHigherThroughputWins pins the core axis: at
// equal in-flight work, the higher-throughput worker drains sooner and is
// preferred.
func TestExpectedDrainTimeEqualInFlightHigherThroughputWins(t *testing.T) {
	fast := DrainLoad{InFlight: 10, Throughput: 10} // drains in 1.0s
	slow := DrainLoad{InFlight: 10, Throughput: 5}  // drains in 2.0s

	if got := ExpectedDrainTime(fast); got != 1.0 {
		t.Fatalf("fast drain = %v, want 1.0", got)
	}
	if got := ExpectedDrainTime(slow); got != 2.0 {
		t.Fatalf("slow drain = %v, want 2.0", got)
	}
	if !PreferByExpectedDrain(fast, slow) {
		t.Fatalf("equal in-flight, higher throughput should be preferred")
	}
	if PreferByExpectedDrain(slow, fast) {
		t.Fatalf("slower worker must not be preferred over faster at equal in-flight")
	}
}

// TestExpectedDrainTimeEqualThroughputMoreInFlightLoses pins monotonicity in
// load: at equal throughput, more in-flight work drains later and is not
// preferred.
func TestExpectedDrainTimeEqualThroughputMoreInFlightLoses(t *testing.T) {
	light := DrainLoad{InFlight: 4, Throughput: 2} // 2.0s
	heavy := DrainLoad{InFlight: 8, Throughput: 2} // 4.0s

	if !(ExpectedDrainTime(light) < ExpectedDrainTime(heavy)) {
		t.Fatalf("lighter queue must drain sooner: %v vs %v",
			ExpectedDrainTime(light), ExpectedDrainTime(heavy))
	}
	if !PreferByExpectedDrain(light, heavy) {
		t.Fatalf("equal throughput, more in-flight should not be preferred")
	}
}

// TestFasterWorkerWithMoreInFlightStillWins is the whole point of the term: a
// worker with MORE in-flight work can still be the better pick when it is fast
// enough to drain sooner than a lighter but slower peer.
func TestFasterWorkerWithMoreInFlightStillWins(t *testing.T) {
	fastBusy := DrainLoad{InFlight: 30, Throughput: 30} // 1.0s, more in-flight
	slowIdle := DrainLoad{InFlight: 5, Throughput: 2}   // 2.5s, less in-flight

	if !(ExpectedDrainTime(fastBusy) < ExpectedDrainTime(slowIdle)) {
		t.Fatalf("faster busy worker should drain sooner: %v vs %v",
			ExpectedDrainTime(fastBusy), ExpectedDrainTime(slowIdle))
	}
	if !PreferByExpectedDrain(fastBusy, slowIdle) {
		t.Fatalf("faster worker with more in-flight should still be preferred")
	}
	// A raw least-loaded comparison would (wrongly) pick slowIdle; the drain-time
	// score must rank fastBusy above it.
	if !(DrainPreferenceScore(fastBusy) > DrainPreferenceScore(slowIdle)) {
		t.Fatalf("drain score should rank the faster busy worker above the slow idle one")
	}
}

// TestZeroInFlightIsZeroDrain pins the idle case: no in-flight work drains in
// zero time regardless of speed, so an idle worker is maximally preferred.
func TestZeroInFlightIsZeroDrain(t *testing.T) {
	idle := DrainLoad{InFlight: 0, Throughput: 3}
	busy := DrainLoad{InFlight: 1, Throughput: 3}

	if got := ExpectedDrainTime(idle); got != 0 {
		t.Fatalf("zero in-flight drain = %v, want 0", got)
	}
	if !PreferByExpectedDrain(idle, busy) {
		t.Fatalf("idle worker should be preferred over a busy one")
	}
	if got := DrainPreferenceScore(idle); got != 1.0 {
		t.Fatalf("idle drain score = %v, want 1.0 (max)", got)
	}
	// Idle even when throughput is unknown: nothing to drain.
	idleUnknown := DrainLoad{InFlight: 0, Throughput: 0}
	if got := ExpectedDrainTime(idleUnknown); got != 0 {
		t.Fatalf("idle worker with unknown rate drain = %v, want 0", got)
	}
}

// TestNonPositiveThroughputFailsClosed pins the divide-by-zero guard: a zero or
// negative (or non-finite) throughput on a loaded worker makes it infinitely
// slow and never preferred — never a NaN/panic.
func TestNonPositiveThroughputFailsClosed(t *testing.T) {
	loaded := DrainLoad{InFlight: 5, Throughput: 1} // finite 5.0s
	for _, bad := range []DrainLoad{
		{InFlight: 5, Throughput: 0},
		{InFlight: 5, Throughput: -2},
		{InFlight: 5, Throughput: math.NaN()},
		{InFlight: 5, Throughput: math.Inf(1)},
	} {
		d := ExpectedDrainTime(bad)
		if !math.IsInf(d, 1) {
			t.Fatalf("bad throughput %+v drain = %v, want +Inf", bad, d)
		}
		if PreferByExpectedDrain(bad, loaded) {
			t.Fatalf("a fail-closed worker %+v must never be preferred", bad)
		}
		if !PreferByExpectedDrain(loaded, bad) {
			t.Fatalf("a finite-drain worker should be preferred over the fail-closed %+v", bad)
		}
		if got := DrainPreferenceScore(bad); got != 0 {
			t.Fatalf("fail-closed drain score = %v, want 0", got)
		}
	}
}

// TestExpectedDrainTiesAreNotStrictPreference keeps the comparison a clean
// less-than: equal drain times (and two infinitely slow workers) never flip a
// strict preference, so a caller can break ties on a lower-order axis and a
// stalled fleet does not flap.
func TestExpectedDrainTiesAreNotStrictPreference(t *testing.T) {
	a := DrainLoad{InFlight: 6, Throughput: 3} // 2.0s
	b := DrainLoad{InFlight: 4, Throughput: 2} // 2.0s
	if PreferByExpectedDrain(a, b) || PreferByExpectedDrain(b, a) {
		t.Fatalf("equal drain times must not be a strict preference either way")
	}

	deadA := DrainLoad{InFlight: 1, Throughput: 0}
	deadB := DrainLoad{InFlight: 9, Throughput: -1}
	if PreferByExpectedDrain(deadA, deadB) || PreferByExpectedDrain(deadB, deadA) {
		t.Fatalf("two infinitely slow workers must tie, not flap")
	}
}
