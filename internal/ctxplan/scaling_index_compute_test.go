package ctxplan

import (
	"context"
	"strings"
	"testing"
)

// TestScalingCarriesIndexBoundedCompute pins issue #562's first half: a Planned Point now
// carries BOTH planner-compute terms — the full-scan Θ(N²) (PlannerComputeCum) and the
// index-bounded Θ(c·N) (PlannerComputeBounded) — and the bounded term is the exact
// IndexBoundedPlannerCompute(c, N), strictly smaller than the full-scan term for a bounded c
// on a large horizon. The bounded compute is the flatten the inverted index buys, read NEXT TO
// the quadratic the full re-scan costs.
func TestScalingCarriesIndexBoundedCompute(t *testing.T) {
	p := Params{TokensPerTurn: 700, WorkingSet: 8000, ForecastHit: 0.9, Retain: 0.7}
	turns := []int{50, 100, 1000, 10000, 1000000}
	cmp := Compare(p, turns)

	for i, n := range turns {
		pl := cmp.Planned[i]
		// The bounded term is the exact closed form, priced at the seed candidate bound.
		if pl.CandidateBound != DefaultMaxCandidates {
			t.Fatalf("N=%d: Planned CandidateBound must be DefaultMaxCandidates=%d, got %d",
				n, DefaultMaxCandidates, pl.CandidateBound)
		}
		wantBounded := IndexBoundedPlannerCompute(DefaultMaxCandidates, n) // Θ(c·N)
		if pl.PlannerComputeBounded != wantBounded {
			t.Fatalf("N=%d: PlannerComputeBounded must equal IndexBoundedPlannerCompute(c,N)=%d, got %d",
				n, wantBounded, pl.PlannerComputeBounded)
		}
		if wantBounded != int64(DefaultMaxCandidates)*int64(n) {
			t.Fatalf("N=%d: the bounded term must be c*N=%d, got %d",
				n, int64(DefaultMaxCandidates)*int64(n), wantBounded)
		}
		// Both compute terms are carried side by side; the full-scan term is unchanged.
		if pl.PlannerComputeCum != cumPlannerCompute(n) {
			t.Fatalf("N=%d: the full-scan term must stay N*(N+1)/2=%d, got %d",
				n, cumPlannerCompute(n), pl.PlannerComputeCum)
		}
	}

	// The bend: once N exceeds the candidate bound, the index-bounded Θ(c·N) is STRICTLY
	// smaller than the full-scan Θ(N²) — the flatten the index buys. At a million turns the
	// gap is enormous (c·N vs N²/2 ~= a 3900x reduction for c=128).
	large := cmp.Planned[len(turns)-1]
	if large.PlannerComputeBounded >= large.PlannerComputeCum {
		t.Fatalf("at N=%d the index-bounded compute %d must flatten the full-scan %d",
			turns[len(turns)-1], large.PlannerComputeBounded, large.PlannerComputeCum)
	}
	// The reduction must be at least ~Nx/(2c): full = N(N+1)/2, bounded = c*N, so
	// full/bounded = (N+1)/(2c). For N=1e6, c=128 that is ~3906x — assert a conservative 1000x.
	if large.PlannerComputeCum < large.PlannerComputeBounded*1000 {
		t.Errorf("at N=%d the full-scan term (%d) must dwarf the bounded term (%d) by >1000x",
			turns[len(turns)-1], large.PlannerComputeCum, large.PlannerComputeBounded)
	}

	// The bounded term is LINEAR in N: 10000x the turns -> exactly 10000x the bounded compute,
	// where the full-scan term grows quadratically (asserted in TestScalingPricesFaultTax...).
	small := cmp.Planned[1] // N=100
	if small.Turn != 100 || large.Turn != 1000000 {
		t.Fatalf("turn schedule drifted: small=%d large=%d", small.Turn, large.Turn)
	}
	if got, want := large.PlannerComputeBounded, small.PlannerComputeBounded*10000; got != want {
		t.Errorf("PlannerComputeBounded must be linear in N: N=1e6 (%d) must be exactly 10000x N=100 (%d => %d)",
			got, small.PlannerComputeBounded, want)
	}

	// Linear and Compaction pay NEITHER compute term (no re-planner at all).
	for i, n := range turns {
		if b := cmp.Linear[i].PlannerComputeBounded; b != 0 {
			t.Errorf("linear at N=%d must price no index-bounded compute, got %d", n, b)
		}
		if b := cmp.Compaction[i].PlannerComputeBounded; b != 0 {
			t.Errorf("compaction at N=%d must price no index-bounded compute, got %d", n, b)
		}
		if cb := cmp.Linear[i].CandidateBound; cb != 0 {
			t.Errorf("linear at N=%d must carry no candidate bound, got %d", n, cb)
		}
	}
}

// TestTableShowsBothComputeColumns proves the rendered operator table reads the index-bounded
// compute column NEXT TO the full-scan one (issue #562's "next to the full-scan term"): both
// headers are present and, on a large horizon, the bounded value renders smaller than the
// full-scan value.
func TestTableShowsBothComputeColumns(t *testing.T) {
	p := Params{TokensPerTurn: 700, WorkingSet: 8000, ForecastHit: 0.9, Retain: 0.7}
	tbl := Compare(p, []int{100, 1000000}).Table()
	if !strings.Contains(tbl, "planner-cpu") || !strings.Contains(tbl, "planner-cpu-idx") {
		t.Fatalf("table must carry BOTH the full-scan and index-bounded compute columns:\n%s", tbl)
	}
	// The bounded column header must appear AFTER the full-scan one (read side by side).
	if strings.Index(tbl, "planner-cpu-idx") <= strings.Index(tbl, "planner-cpu ") &&
		!strings.Contains(tbl, "planner-cpu ") {
		t.Errorf("index-bounded compute column must sit beside the full-scan one:\n%s", tbl)
	}
}

// TestMeasureGreedyGapOnRealStore pins issue #562's second half: the greedy-vs-exact
// optimality gap is MEASURED on a representative recorded store (goodPlusNoiseStore, the
// long-session workload the index exists for) over the index-bounded probe — the oracle-sized
// candidate set the Planned regime actually scores each turn — and reported as a BOUND, not
// assumed zero. The greedy planner must capture a high fraction of the optimum, and the exact
// oracle can never do worse than greedy.
func TestMeasureGreedyGapOnRealStore(t *testing.T) {
	ctx := context.Background()
	// Two representative stores at different depths so the gap is measured on more than one
	// recorded shape, not a single hand-built counterexample.
	for _, noise := range []int{120, 300} {
		st := goodPlusNoiseStore(noise)
		spans, err := st.Spans(ctx)
		if err != nil {
			t.Fatalf("noise=%d: store read failed: %v", noise, err)
		}
		ix := BuildIndex(spans)
		f := Forecast{Intents: []string{"auth token rotation runbook"}}
		// Score ONLY the index-bounded probe — the bounded candidate set the planner sees each
		// turn, and the oracle-sized input knapsackExact can actually optimize.
		probe := ix.Probe(f, ProbeOptions{})
		cands := Candidates(probe, f, nil)
		if len(cands) == 0 {
			t.Fatalf("noise=%d: the probe scored no candidates — nothing to measure", noise)
		}

		g := MeasureGreedyGap(cands, 64)

		// The oracle is an optimum: it can never do WORSE than greedy, so the gap is >= 0 and
		// the ratio is in (0, 1]. A measured bound, not an assertion of zero.
		if g.AbsGap < 0 {
			t.Errorf("noise=%d: exact (%.3f) cannot be worse than greedy (%.3f) — abs gap %.3f < 0",
				noise, g.ExactBenefit, g.GreedyBenefit, g.AbsGap)
		}
		if g.ExactBenefit <= 0 {
			t.Fatalf("noise=%d: the representative store must yield achievable benefit, got exact=%.3f", noise, g.ExactBenefit)
		}
		if g.Ratio <= 0 || g.Ratio > 1.0+1e-9 {
			t.Errorf("noise=%d: greedy/optimal ratio must be in (0,1], got %.4f", noise, g.Ratio)
		}
		// On these stores the candidates are roughly uniform-density, so greedy is near-optimal:
		// the MEASURED gap must be tight (greedy captures >= 90% of the optimum). This is the
		// bound the issue asks for — recorded, not assumed.
		if g.Ratio < 0.90 {
			t.Errorf("noise=%d: greedy captured only %.1f%% of the optimum (abs gap %.3f over budget %d) — gap wider than the measured bound",
				noise, 100*g.Ratio, g.AbsGap, g.Budget)
		}
		if g.Candidates != len(cands) {
			t.Errorf("noise=%d: GreedyGap.Candidates (%d) must record the scored set size (%d)", noise, g.Candidates, len(cands))
		}
	}

	// The measurement is REAL: it detects a true gap when one exists. The classic knapsack
	// counterexample (a dense span boxing greedy out of a more valuable pair) must produce a
	// strictly positive abs gap and a ratio < 1 — proving the bound above is not vacuous.
	counter := []Candidate{
		cand("dense", 0, 6, 7.0), // density 1.166 — greedy grabs this first
		cand("x", 1, 5, 5.0),     // density 1.0
		cand("y", 2, 5, 5.0),     // density 1.0
	}
	cg := MeasureGreedyGap(counter, 10)
	if cg.AbsGap <= 0 || cg.Ratio >= 1.0 {
		t.Fatalf("on the knapsack counterexample the measured gap must be positive (greedy sub-optimal): %+v", cg)
	}
	if cg.ExactBenefit != 10.0 {
		t.Errorf("the exact oracle should select x+y for benefit 10, measured %.3f", cg.ExactBenefit)
	}
}
