package rungobs

import (
	"fmt"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/fusedturn"
)

// TestFusedTurnsBounded proves the per-turn latch map no longer grows without bound (#3451):
// after far more distinct turn keys than fusedTurnCap, residency stays O(cap) — while the
// fused-turn KPI counters (read from knownTurns/fusedTurns, not len(turns)) remain exact,
// because no evicted trace reappears in this drive so nothing is double-counted.
func TestFusedTurnsBounded(t *testing.T) {
	o := New()

	// Each trace emits a classical then a weight op back-to-back → a fully fused turn, minted
	// and latched before the next trace, so no trace is ever half-counted across an eviction.
	const flood = fusedTurnCap*2 + 137
	for i := 0; i < flood; i++ {
		tr := fmt.Sprintf("trace-%d", i)
		o.observeFusedOp(classifiedCall(tr, fusedturn.ClassClassical))
		o.observeFusedOp(classifiedCall(tr, fusedturn.ClassWeight))
	}

	o.mu.Lock()
	residency := len(o.turns)
	o.mu.Unlock()

	// Residency is O(cap), independent of the flood length.
	if residency > fusedTurnCap {
		t.Fatalf("turns len = %d, want <= cap %d — map is not bounded", residency, fusedTurnCap)
	}
	// Eviction actually fired (flood far exceeds cap), so the map is pinned at the cap.
	if residency != fusedTurnCap {
		t.Fatalf("turns len = %d, want == cap %d after a %d-key flood", residency, fusedTurnCap, flood)
	}

	// The KPI is metric-neutral: every distinct trace became a fused turn and eviction never
	// rolls back a counter, so both the denominator and numerator equal the trace count.
	snap := o.FusedSnapshot()
	if snap.Turns != int64(flood) {
		t.Fatalf("FusedSnapshot.Turns = %d, want %d — bounding the map must not change the denominator", snap.Turns, flood)
	}
	if snap.FusedTurns != int64(flood) {
		t.Fatalf("FusedSnapshot.FusedTurns = %d, want %d", snap.FusedTurns, flood)
	}
	if snap.Rate != 1.0 {
		t.Fatalf("FusedSnapshot.Rate = %v, want 1.0", snap.Rate)
	}
}
