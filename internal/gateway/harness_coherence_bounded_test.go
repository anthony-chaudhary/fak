package gateway

import (
	"fmt"
	"testing"
	"time"
)

// TestHarnessCoherenceCoordsBounded proves the per-trace coordinator map no longer grows without
// bound (#3450): past maxCoherenceSessions distinct traces the map evicts, it evicts the COLDEST
// (oldest last-served-turn) trace so a live session is never displaced, and residency stays O(cap)
// independent of how many total sessions were served.
func TestHarnessCoherenceCoordsBounded(t *testing.T) {
	m := newHarnessCoherenceMetrics(0)
	t0 := time.Unix(1_700_000_000, 0)

	obs := func(trace string, now time.Time) {
		m.observe(trace, now, "", false, "", false, false, 0, 0, 0)
	}

	// A trace we keep warm: observed first, then re-stamped fresh throughout the flood so its
	// last-served-turn is never the oldest — it must survive every eviction.
	obs("warm", t0)

	// Flood far past capacity with distinct cold traces, each seen exactly once at an ever-later
	// time (so cold-0 is the oldest overall). Re-stamp "warm" periodically to keep it fresh.
	const flood = maxCoherenceSessions*2 + 137
	for i := 0; i < flood; i++ {
		now := t0.Add(time.Duration(i+1) * time.Millisecond)
		obs(fmt.Sprintf("cold-%d", i), now)
		if i%100 == 0 {
			obs("warm", now)
		}
	}
	// Final warm stamp at the newest time so it is unambiguously not the coldest.
	obs("warm", t0.Add(time.Duration(flood+1)*time.Millisecond))

	// Residency is O(cap), independent of the flood length.
	if got := len(m.coords); got > maxCoherenceSessions {
		t.Fatalf("coords len = %d, want <= cap %d — map is not bounded", got, maxCoherenceSessions)
	}

	// The warm (most-recently-served) trace survived eviction.
	if _, ok := m.coords["warm"]; !ok {
		t.Fatalf("warm trace was evicted — eviction must drop the COLDEST trace, never a live one")
	}

	// The oldest cold trace was reclaimed (proves eviction actually fired, not just a big cap).
	if _, ok := m.coords["cold-0"]; ok {
		t.Fatalf("cold-0 (oldest trace) survived %d newer sessions — coldest-first eviction did not fire", flood)
	}

	// The aggregate served-turn denominator counted every observe, unaffected by eviction.
	// warm: 1 initial + ceil(flood/100) periodic + 1 final ; cold: flood.
	wantObserved := uint64(flood) + 1 + uint64((flood+99)/100) + 1
	if m.observedTurns != wantObserved {
		t.Fatalf("observedTurns = %d, want %d — eviction must not drop aggregate counters", m.observedTurns, wantObserved)
	}
}
