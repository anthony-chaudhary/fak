package session

import "testing"

// TestCostRingBoundedAndOrdered proves the ring retains exactly the last CostRingSize turns
// (never grows) and reports them most-recent-first via at().
func TestCostRingBoundedAndOrdered(t *testing.T) {
	var r CostRing
	// Push twice the capacity; only the last CostRingSize survive.
	for i := 1; i <= 2*CostRingSize; i++ {
		r = r.push(TurnCost{OutputTokens: i})
	}
	if r.Count != CostRingSize {
		t.Fatalf("Count = %d, want %d (ring must be bounded)", r.Count, CostRingSize)
	}
	// Latest is the last value pushed (2*CostRingSize); the i-th-back is one less.
	for i := 0; i < CostRingSize; i++ {
		got, ok := r.at(i)
		if !ok {
			t.Fatalf("at(%d) missing despite a full ring", i)
		}
		want := 2*CostRingSize - i
		if got.OutputTokens != want {
			t.Fatalf("at(%d).OutputTokens = %d, want %d", i, got.OutputTokens, want)
		}
	}
	// One past the live window is absent — never a zero-value slot.
	if _, ok := r.at(CostRingSize); ok {
		t.Fatalf("at(%d) should be out of range on a %d-entry ring", CostRingSize, CostRingSize)
	}
}

// TestCostSummaryEmptyAndSingle proves the summary degrades safely: an empty ring is all-zero
// (no divide-by-zero), and a single turn reports a Latest with no spike.
func TestCostSummaryEmptyAndSingle(t *testing.T) {
	var r CostRing
	if s := r.CostSummary(); s != (CostSummary{}) {
		t.Fatalf("empty ring summary = %+v, want zero", s)
	}
	r = r.push(TurnCost{OutputTokens: 100, ContextTokens: 50})
	s := r.CostSummary()
	if s.Latest != 150 || s.Count != 1 {
		t.Fatalf("single-turn summary = %+v, want Latest=150 Count=1", s)
	}
	if s.SpikeRatio != 0 || s.Previous != 0 || s.Delta != 0 {
		t.Fatalf("single-turn summary = %+v, want no spike/previous/delta (a first turn cannot spike)", s)
	}
}

// TestCostSummarySpikeRatio proves the cost-per-iteration runaway signal: a steady session
// reads SpikeRatio ~1.0, and a turn that explodes ~200x the prior window reads ~200.
func TestCostSummarySpikeRatio(t *testing.T) {
	var r CostRing
	for i := 0; i < 4; i++ {
		r = r.push(TurnCost{OutputTokens: 100}) // steady baseline
	}
	if s := r.CostSummary(); s.SpikeRatio != 1.0 {
		t.Fatalf("steady SpikeRatio = %v, want 1.0", s.SpikeRatio)
	}
	r = r.push(TurnCost{OutputTokens: 20000}) // the runaway turn: 200x the 100-token window
	s := r.CostSummary()
	if s.SpikeRatio != 200.0 {
		t.Fatalf("runaway SpikeRatio = %v, want 200.0", s.SpikeRatio)
	}
	if s.Latest != 20000 || s.Previous != 100 || s.Delta != 19900 {
		t.Fatalf("runaway summary = %+v, want Latest=20000 Previous=100 Delta=19900", s)
	}
}

// TestCostSummarySpikeOverMaxNotPrevious proves the spike is measured against the MAX of the
// prior window, not the immediately-previous turn — so a single normal turn wedged after a
// runaway cannot mask the spike (the ratio stays well below 1 only because the runaway is
// still the window max).
func TestCostSummarySpikeOverMaxNotPrevious(t *testing.T) {
	var r CostRing
	r = r.push(TurnCost{OutputTokens: 100})
	r = r.push(TurnCost{OutputTokens: 20000}) // runaway
	r = r.push(TurnCost{OutputTokens: 100})   // one normal turn after it
	s := r.CostSummary()
	// Latest=100, prior window max=20000 -> ratio is tiny, not 1.0 against the previous 20000.
	if s.SpikeRatio >= 1.0 {
		t.Fatalf("SpikeRatio = %v, want < 1 (latest is small vs the window max)", s.SpikeRatio)
	}
	if s.Previous != 20000 {
		t.Fatalf("Previous = %d, want 20000 (the immediately-prior turn)", s.Previous)
	}
}

// TestDebitUsageRecordsCostRing proves DebitUsage feeds the per-session ring carried on State
// and that Snapshot carries it out — the wiring #756's fak ps column reads. It also proves the
// ring fills for an UNBOUNDED-budget session (the runaway that never drains a cap), since the
// debit records cost independent of whether a budget axis is configured.
func TestDebitUsageRecordsCostRing(t *testing.T) {
	tbl := NewTable()
	const trace = "gw-cost"
	// Unbounded budget (the default) — no axis to drain, yet the cost ring must still fill.
	tbl.DebitUsage(trace, Usage{OutputTokens: 100, ContextTokens: 10})
	tbl.DebitUsage(trace, Usage{OutputTokens: 20000, ContextTokens: 10}) // runaway turn

	st := tbl.Get(trace)
	if st.Cost.Count != 2 {
		t.Fatalf("Cost.Count = %d, want 2 recorded turns", st.Cost.Count)
	}
	s := st.Cost.CostSummary()
	if s.Latest != 20010 {
		t.Fatalf("latest cost = %d, want 20010 (20000 out + 10 ctx)", s.Latest)
	}
	if s.SpikeRatio <= 100 {
		t.Fatalf("runaway SpikeRatio = %v, want a large (>100x) spike visible to fak ps", s.SpikeRatio)
	}

	// Snapshot carries the ring out (the scheduler / fak ps read).
	snap := tbl.Snapshot()
	if len(snap) != 1 || snap[0].Cost.Count != 2 {
		t.Fatalf("Snapshot cost = %+v, want one row carrying the 2-turn ring", snap)
	}
	if got := snap[0].Cost.CostSummary().SpikeRatio; got != s.SpikeRatio {
		t.Fatalf("Snapshot summary spike = %v, want %v (the same ring)", got, s.SpikeRatio)
	}
}

// TestDebitUsageZeroCostNoRecord proves a no-cost debit (both axes <= 0) records nothing — the
// early-return guard keeps an empty turn out of the ring.
func TestDebitUsageZeroCostNoRecord(t *testing.T) {
	tbl := NewTable()
	const trace = "gw-zero"
	tbl.DebitUsage(trace, Usage{}) // nothing to debit
	if c := tbl.Get(trace).Cost.Count; c != 0 {
		t.Fatalf("Cost.Count = %d after a zero debit, want 0", c)
	}
}
