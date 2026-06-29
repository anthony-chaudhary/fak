package spec

import "testing"

// TestAcceptanceMeterObserveUnderCostCap is the observer-effect fence the
// docs/standards/observer-effect.md standard cites: any meter fak embeds on a hot
// path must cost less than a declared cap, and that cost must itself be MEASURED —
// you cannot honestly report an overhead you never bounded.
//
// The decode AcceptanceMeter is fak's shipped hot-path meter (#284). Its per-round
// Observe is a pure accumulator — a handful of integer adds, no I/O, no slice growth
// — so its declared cap is the tightest one a meter can have: ZERO heap allocations
// per sample. testing.AllocsPerRun is deterministic (same input → the same count on
// every run and every host), so this is a WITNESSED bound, not a noisy wall-clock
// one that would read OBSERVED. The companion metrics_test.go proves the un-metered
// fast path (a nil meter) is byte-identical to the metered run — zero behavioral
// cost; this pins the metered path's zero resource cost per observation.
func TestAcceptanceMeterObserveUnderCostCap(t *testing.T) {
	const maxAllocsPerObserve = 0.0 // declared cap: the meter allocates nothing per sample
	var m AcceptanceMeter
	got := testing.AllocsPerRun(1000, func() {
		m.Observe(4, 3, 4)
	})
	if got > maxAllocsPerObserve {
		t.Fatalf("AcceptanceMeter.Observe = %g allocs/op, want <= %g — the observer-effect fence: the meter's own per-sample cost must stay under its declared cap", got, maxAllocsPerObserve)
	}
	// Sanity: the accumulator still recorded every observed round, so the zero-cost
	// path is the real metering path, not a no-op masquerading as cheap.
	if m.Rounds() == 0 {
		t.Fatal("AcceptanceMeter recorded 0 rounds after Observe — the cap would be vacuous on a meter that does nothing")
	}
}
