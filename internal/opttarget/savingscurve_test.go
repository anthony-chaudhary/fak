package opttarget

import "testing"

// loopTrace is the crafted saturating workload the knee tests key on: a cyclic
// scan over `distinct` keys repeated `repeats` times. Under LRU a cyclic scan
// is all-or-nothing — any budget below `distinct` evicts every key just before
// its reuse (0 hits), and any budget at or above it turns every access after
// the first pass into a hit (hits = distinct*(repeats-1)) — so reuse saturates
// at a KNOWN size and the expected knee is exact.
func loopTrace(distinct, repeats int) []int {
	tr := make([]int, 0, distinct*repeats)
	for r := 0; r < repeats; r++ {
		for k := 0; k < distinct; k++ {
			tr = append(tr, k)
		}
	}
	return tr
}

// distinctTrace has no reuse at all: every key is unique, so every budget's
// hit rate is 0 and the savings curve is flat.
func distinctTrace(n int) []int {
	tr := make([]int, n)
	for i := range tr {
		tr[i] = i
	}
	return tr
}

// episodicTrace mirrors the shape of rsiloop's reference trace: for each reuse
// distance d in 1..maxDistance it emits `episodes` runs of "touch target,
// touch d fillers, touch target again", so the hit rate climbs smoothly with
// budget instead of jumping. Filler keys live in a disjoint range so a filler
// is never the target being measured. Pure and deterministic.
func episodicTrace(maxDistance, episodes int) []int {
	var tr []int
	for d := 1; d <= maxDistance; d++ {
		for e := 0; e < episodes; e++ {
			target := (d*7 + e) % maxDistance
			tr = append(tr, target)
			for f := 0; f < d; f++ {
				tr = append(tr, 1000+((d+e+f)%maxDistance))
			}
			tr = append(tr, target)
		}
	}
	return tr
}

func intRange(lo, hi int) []int {
	var out []int
	for b := lo; b <= hi; b++ {
		out = append(out, b)
	}
	return out
}

// TestSavingsVsBudgetMonotone is witness (a): on synthetic traces with known
// reuse, hit rate (and the savings proxy with it) is monotonically
// non-decreasing as the budget grows — the LRU inclusion property the curve
// rests on. Where a trace has real reuse the curve must actually rise, not
// merely not fall.
func TestSavingsVsBudgetMonotone(t *testing.T) {
	cases := []struct {
		name     string
		trace    []int
		budgets  []int
		mustRise bool // the last point must strictly beat the first
	}{
		{"cyclic-loop", loopTrace(6, 20), intRange(1, 10), true},
		{"episodic-spread", episodicTrace(8, 5), intRange(1, 12), true},
		{"no-reuse-flat", distinctTrace(40), intRange(1, 8), false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			curve, err := SavingsVsBudget(c.trace, c.budgets)
			if err != nil {
				t.Fatalf("SavingsVsBudget: %v", err)
			}
			if got, want := len(curve.Points), len(c.budgets); got != want {
				t.Fatalf("len(Points) = %d, want %d", got, want)
			}
			if curve.TraceLen != len(c.trace) {
				t.Errorf("TraceLen = %d, want %d", curve.TraceLen, len(c.trace))
			}
			for i, p := range curve.Points {
				if p.Budget != c.budgets[i] {
					t.Errorf("Points[%d].Budget = %d, want %d", i, p.Budget, c.budgets[i])
				}
				if p.HitRate < 0 || p.HitRate > 1 {
					t.Errorf("Points[%d].HitRate = %v, want within [0,1]", i, p.HitRate)
				}
				if i == 0 {
					continue
				}
				prev := curve.Points[i-1]
				if p.HitRate < prev.HitRate {
					t.Errorf("HitRate fell at budget %d: %v -> %v (must be non-decreasing)", p.Budget, prev.HitRate, p.HitRate)
				}
				if p.Savings < prev.Savings {
					t.Errorf("Savings fell at budget %d: %v -> %v (must be non-decreasing)", p.Budget, prev.Savings, p.Savings)
				}
			}
			first, last := curve.Points[0], curve.Points[len(curve.Points)-1]
			if c.mustRise && last.HitRate <= first.HitRate {
				t.Errorf("curve never rose: HitRate %v at budget %d vs %v at budget %d", first.HitRate, first.Budget, last.HitRate, last.Budget)
			}
			if !c.mustRise && last.HitRate != 0 {
				t.Errorf("no-reuse trace hit rate = %v at budget %d, want 0", last.HitRate, last.Budget)
			}
		})
	}
}

// TestSavingsVsBudgetKnee is witness (b): on crafted traces whose reuse
// saturates at a known size, the detected knee is the expected budget — the
// saturation point when it sits on the requested grid, the first requested
// budget at or above it when it does not, and the smallest budget on a flat
// curve where more budget buys nothing.
func TestSavingsVsBudgetKnee(t *testing.T) {
	cases := []struct {
		name     string
		trace    []int
		budgets  []int
		wantKnee int
	}{
		// Reuse saturates at 6 distinct keys and 6 is on the grid: the marginal
		// gain collapses to 0 past budget 6, so 6 is the knee.
		{"saturation-on-grid", loopTrace(6, 20), []int{1, 2, 3, 4, 5, 6, 8, 10}, 6},
		// Saturation at 6 falls between requested budgets 4 and 8: the knee is
		// grid-limited to the first requested budget that covers the working set.
		{"saturation-off-grid", loopTrace(6, 20), []int{2, 4, 8, 16}, 8},
		// No reuse anywhere: a flat curve's knee is the smallest budget.
		{"flat-no-reuse", distinctTrace(40), []int{2, 4, 8}, 2},
		// Every budget already covers the 3-key working set: flat again, so the
		// smallest requested budget is the knee.
		{"already-saturated", loopTrace(3, 10), []int{4, 8, 16}, 4},
		// A single budget has no marginals and is its own knee.
		{"single-budget", loopTrace(4, 5), []int{7}, 7},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			curve, err := SavingsVsBudget(c.trace, c.budgets)
			if err != nil {
				t.Fatalf("SavingsVsBudget: %v", err)
			}
			if curve.Knee != c.wantKnee {
				t.Errorf("Knee = %d, want %d (curve: %+v)", curve.Knee, c.wantKnee, curve.Points)
			}
		})
	}
}

// TestSavingsVsBudgetExact pins the replay to hand-computed hit counts on tiny
// traces, so an LRU regression (admission, eviction order, the zero-budget
// floor) surfaces as an exact-count failure rather than only as a bent curve.
func TestSavingsVsBudgetExact(t *testing.T) {
	cases := []struct {
		name     string
		trace    []int
		budgets  []int
		wantHits []float64 // Savings per point; HitRate = Savings/len(trace)
	}{
		// Budget 1 thrashes between the two keys (0 hits); budget 2 holds both.
		{"two-key-alternation", []int{1, 2, 1, 2}, []int{1, 2}, []float64{0, 2}},
		// Budget 2 evicts key 1 before its reuse; budget 3 keeps it resident.
		{"evict-before-reuse", []int{1, 2, 3, 1}, []int{2, 3}, []float64{0, 1}},
		// A zero budget holds nothing; one entry turns the repeats into hits.
		{"zero-budget-floor", []int{5, 5, 5}, []int{0, 1}, []float64{0, 2}},
		// Saturated loop: hits = distinct*(repeats-1) once the budget covers it.
		{"loop-saturated", loopTrace(3, 4), []int{3}, []float64{9}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			curve, err := SavingsVsBudget(c.trace, c.budgets)
			if err != nil {
				t.Fatalf("SavingsVsBudget: %v", err)
			}
			for i, want := range c.wantHits {
				p := curve.Points[i]
				if p.Savings != want {
					t.Errorf("Points[%d].Savings = %v, want %v", i, p.Savings, want)
				}
				if wantRate := want / float64(len(c.trace)); p.HitRate != wantRate {
					t.Errorf("Points[%d].HitRate = %v, want %v", i, p.HitRate, wantRate)
				}
			}
		})
	}
}

// TestSavingsVsBudgetRefusals proves a malformed request is REFUSED, never
// silently lowered into a meaningless curve — the package's Validate rule
// applied to the curve inputs.
func TestSavingsVsBudgetRefusals(t *testing.T) {
	cases := []struct {
		name    string
		trace   []int
		budgets []int
	}{
		{"empty-trace", nil, []int{1, 2}},
		{"no-budgets", []int{1, 2, 1}, nil},
		{"negative-budget", []int{1, 2, 1}, []int{-1, 2}},
		{"duplicate-budgets", []int{1, 2, 1}, []int{2, 2}},
		{"decreasing-budgets", []int{1, 2, 1}, []int{4, 2}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if _, err := SavingsVsBudget(c.trace, c.budgets); err == nil {
				t.Fatalf("SavingsVsBudget(%s) = nil error, want refusal", c.name)
			}
		})
	}
}
