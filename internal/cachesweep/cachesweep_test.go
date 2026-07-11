package cachesweep

import (
	"math"
	"reflect"
	"testing"
)

// fixtureTrace is the hand-verified acceptance workload. It exercises the three behaviors
// the sweep must capture: an exact repeat (full reuse), a mid-run divergence that forces
// an edge SPLIT (partial reuse), and enough churn that a tight budget evicts and loses
// reuse a larger budget keeps.
//
//	a0 [1,2,3]  cold                       -> match 0
//	a1 [1,2,3]  exact repeat               -> match 3
//	a2 [1,2,4]  diverges at pos 2 (split)  -> match 2
//	a3 [1,2,3]  reuse the [1,2] fork       -> match 3 (unbounded)
//	a4 [1,2,4]  reuse the [1,2] fork       -> match 3 (unbounded)
//
// Unbounded reused = 0+3+2+3+3 = 11 of 15 requested  -> ceiling 11/15.
// The resident tree tops out at 4 cached tokens (edges [1,2]+[3]+[4]), so any budget ≥ 4
// never evicts and equals the ceiling; budget 2 evicts the cold fork repeatedly.
func fixtureTrace() Trace {
	seq := [][]int{{1, 2, 3}, {1, 2, 3}, {1, 2, 4}, {1, 2, 3}, {1, 2, 4}}
	tr := Trace{}
	for i, s := range seq {
		tr.Accesses = append(tr.Accesses, Access{Tokens: s, TimeNs: int64(i)})
	}
	return tr
}

func approx(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

func TestSweepCeilingCurveKnee(t *testing.T) {
	res := Sweep(fixtureTrace(), Options{Budgets: []int{2, 4, 8}})

	// Ceiling: the infinite-cache pass reuses 11 of 15 tokens.
	if res.Ceiling.ReusedTokens != 11 || res.Ceiling.TotalTokens != 15 {
		t.Fatalf("ceiling reused/total = %d/%d, want 11/15", res.Ceiling.ReusedTokens, res.Ceiling.TotalTokens)
	}
	if !approx(res.Ceiling.ReuseRatio, 11.0/15.0) {
		t.Fatalf("ceiling ratio = %v, want %v", res.Ceiling.ReuseRatio, 11.0/15.0)
	}
	if res.Ceiling.Budget != 0 {
		t.Fatalf("ceiling budget = %d, want 0 (unbounded)", res.Ceiling.Budget)
	}

	// Curve: budget 2 loses reuse to eviction (9/15); budgets 4 and 8 match the ceiling.
	if len(res.Curve) != 3 {
		t.Fatalf("curve has %d points, want 3", len(res.Curve))
	}
	wantReused := map[int]int64{2: 9, 4: 11, 8: 11}
	for _, p := range res.Curve {
		if got := wantReused[p.Budget]; p.ReusedTokens != got {
			t.Errorf("budget %d reused = %d, want %d", p.Budget, p.ReusedTokens, got)
		}
		if p.ReuseRatio > res.Ceiling.ReuseRatio+1e-9 {
			t.Errorf("budget %d ratio %v exceeds ceiling %v", p.Budget, p.ReuseRatio, res.Ceiling.ReuseRatio)
		}
	}
	// Budget 2 must actually evict (that is why it loses reuse); budget 8 must not.
	if res.Curve[0].Budget == 2 && res.Curve[0].Evictions == 0 {
		t.Errorf("budget 2 performed no evictions; expected pressure")
	}
	if last := res.Curve[len(res.Curve)-1]; last.Budget == 8 && last.Evictions != 0 {
		t.Errorf("budget 8 evicted %d; expected none (fits in budget)", last.Evictions)
	}

	// Knee: smallest budget reaching 99% of the ceiling. Budget 2 (0.60) is below the
	// 0.726 threshold; budget 4 (0.7333) clears it.
	if !res.KneeReached {
		t.Fatalf("knee not reached; expected budget 4 to clear 99%% of ceiling")
	}
	if res.Knee.Budget != 4 {
		t.Fatalf("knee budget = %d, want 4", res.Knee.Budget)
	}
}

// TestSweepDeterministic pins the load-bearing purity property: same input, same output,
// byte for byte (including eviction victim choices).
func TestSweepDeterministic(t *testing.T) {
	opt := Options{Budgets: []int{2, 3, 4, 8}, WriteDelayNs: 0}
	a := Sweep(fixtureTrace(), opt)
	b := Sweep(fixtureTrace(), opt)
	if !reflect.DeepEqual(a, b) {
		t.Fatalf("Sweep is not deterministic:\n a=%+v\n b=%+v", a, b)
	}
}

// TestWriteDelayWindow checks the optional visibility overlay: a re-request that arrives
// before the prefix's cache became visible counts as a miss even though it is resident.
//
//	3× [1,2,3] at t=0,1,2. First write of [1,2,3] completes at t=0.
//	delay 0 : a1 and a2 both reuse 3      -> reused 6/9
//	delay 2 : a1 (t=1 < 0+2) misses; a2 (t=2 == 0+2) reuses 3 -> reused 3/9
func TestWriteDelayWindow(t *testing.T) {
	tr := Trace{Accesses: []Access{
		{Tokens: []int{1, 2, 3}, TimeNs: 0},
		{Tokens: []int{1, 2, 3}, TimeNs: 1},
		{Tokens: []int{1, 2, 3}, TimeNs: 2},
	}}

	noDelay := Sweep(tr, Options{Budgets: []int{8}})
	if noDelay.Ceiling.ReusedTokens != 6 {
		t.Fatalf("no-delay reused = %d, want 6", noDelay.Ceiling.ReusedTokens)
	}

	delayed := Sweep(tr, Options{Budgets: []int{8}, WriteDelayNs: 2})
	if delayed.Ceiling.ReusedTokens != 3 {
		t.Fatalf("write-delay reused = %d, want 3 (a1 misses inside the window)", delayed.Ceiling.ReusedTokens)
	}
	if delayed.WriteDelayNs != 2 {
		t.Fatalf("result did not carry the write-delay knob: %d", delayed.WriteDelayNs)
	}
}

// TestSweepEmptyAndDegenerate covers the guards: an empty trace, and an all-unique trace
// with a zero ceiling (no reuse ⇒ no knee).
func TestSweepEmptyAndDegenerate(t *testing.T) {
	empty := Sweep(Trace{}, Options{Budgets: []int{4}})
	if empty.TotalTokens != 0 || empty.Ceiling.ReuseRatio != 0 || empty.KneeReached {
		t.Fatalf("empty trace produced non-zero result: %+v", empty)
	}

	unique := Trace{Accesses: []Access{
		{Tokens: []int{1}}, {Tokens: []int{2}}, {Tokens: []int{3}},
	}}
	res := Sweep(unique, Options{Budgets: []int{1, 2}})
	if res.Ceiling.ReusedTokens != 0 {
		t.Fatalf("all-unique ceiling reused = %d, want 0", res.Ceiling.ReusedTokens)
	}
	if res.KneeReached {
		t.Fatalf("all-unique trace has zero ceiling; there is no ROI knee")
	}
}

// TestNormalizeBudgets guards the dedup/drop/sort contract callers rely on.
func TestNormalizeBudgets(t *testing.T) {
	got := normalizeBudgets([]int{8, 2, 8, 0, -4, 4, 2})
	want := []int{2, 4, 8}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("normalizeBudgets = %v, want %v", got, want)
	}
}
