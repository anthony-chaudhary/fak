package cachesweep

import (
	"math"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/opttarget"
	"github.com/anthony-chaudhary/fak/internal/sweepcert"
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

func TestTraceDigest(t *testing.T) {
	base := Trace{Accesses: []Access{
		{Tokens: []int{1, 2, 3}, TimeNs: 10, CallerID: "c1"},
		{Tokens: []int{4, 5}, TimeNs: 20, CallerID: "c2"},
	}}
	d1 := TraceDigest(base)
	d2 := TraceDigest(base)
	if d1 != d2 || len(d1) != 64 {
		t.Fatalf("TraceDigest determinism failure: %q vs %q", d1, d2)
	}

	// Token sensitivity
	tMod := Trace{Accesses: []Access{
		{Tokens: []int{1, 2, 4}, TimeNs: 10, CallerID: "c1"},
		{Tokens: []int{4, 5}, TimeNs: 20, CallerID: "c2"},
	}}
	if TraceDigest(tMod) == d1 {
		t.Fatalf("TraceDigest insensitive to tokens")
	}

	// Timestamp sensitivity
	timeMod := Trace{Accesses: []Access{
		{Tokens: []int{1, 2, 3}, TimeNs: 11, CallerID: "c1"},
		{Tokens: []int{4, 5}, TimeNs: 20, CallerID: "c2"},
	}}
	if TraceDigest(timeMod) == d1 {
		t.Fatalf("TraceDigest insensitive to timestamps")
	}

	// CallerID sensitivity
	callerMod := Trace{Accesses: []Access{
		{Tokens: []int{1, 2, 3}, TimeNs: 10, CallerID: "c3"},
		{Tokens: []int{4, 5}, TimeNs: 20, CallerID: "c2"},
	}}
	if TraceDigest(callerMod) == d1 {
		t.Fatalf("TraceDigest insensitive to caller ID")
	}
}

func TestSweepCertificationMeasured(t *testing.T) {
	res := Sweep(fixtureTrace(), Options{Budgets: []int{2, 4, 8}})
	if res.KneeStatus != "measured" {
		t.Fatalf("KneeStatus = %q, want measured", res.KneeStatus)
	}
	if !strings.HasPrefix(res.EnvelopeDigest, "sha256:") {
		t.Fatalf("EnvelopeDigest = %q, want sha256: prefix", res.EnvelopeDigest)
	}
	wantSupporting := []string{"budget:2", "budget:4", "budget:8"}
	if !reflect.DeepEqual(res.SupportingPoints, wantSupporting) {
		t.Fatalf("SupportingPoints = %v, want %v", res.SupportingPoints, wantSupporting)
	}
	if res.Interval == nil || res.Interval.LowerPointID != "budget:2" || res.Interval.UpperPointID != "budget:4" {
		t.Fatalf("Interval = %+v, want lower budget:2 upper budget:4", res.Interval)
	}
}

func TestSweepCertificationOmission(t *testing.T) {
	res := Sweep(fixtureTrace(), Options{
		Budgets: []int{2, 8},
		Omissions: []BudgetOmission{
			{Budget: 4, Reason: "allocation failure"},
		},
	})
	if res.KneeStatus != "not_identifiable" {
		t.Fatalf("KneeStatus = %q, want not_identifiable", res.KneeStatus)
	}
	if len(res.Curve) != 2 {
		t.Fatalf("Curve has %d points, want 2", len(res.Curve))
	}
	if res.Interval == nil || res.Interval.LowerPointID != "budget:2" || res.Interval.UpperPointID != "budget:8" {
		t.Fatalf("Interval = %+v, want lower budget:2 upper budget:8", res.Interval)
	}
}

func TestSweepCertificationLeftRightCensored(t *testing.T) {
	// Left-censored: budget 4 already clears 99% of ceiling
	left := Sweep(fixtureTrace(), Options{Budgets: []int{4, 8}})
	if left.KneeStatus != "left_censored" {
		t.Fatalf("left KneeStatus = %q, want left_censored", left.KneeStatus)
	}
	if left.Interval == nil || left.Interval.UpperPointID != "budget:4" {
		t.Fatalf("left Interval = %+v, want upper budget:4", left.Interval)
	}

	// Right-censored: neither budget 2 nor 3 reaches threshold
	right := Sweep(fixtureTrace(), Options{Budgets: []int{2, 3}})
	if right.KneeStatus != "right_censored" {
		t.Fatalf("right KneeStatus = %q, want right_censored", right.KneeStatus)
	}
	if right.Interval == nil || right.Interval.LowerPointID != "budget:3" {
		t.Fatalf("right Interval = %+v, want lower budget:3", right.Interval)
	}
}

func TestSweepCertificationInvalidAxis(t *testing.T) {
	res := Sweep(fixtureTrace(), Options{Budgets: []int{4}})
	if res.KneeStatus != "invalid" {
		t.Fatalf("KneeStatus = %q, want invalid", res.KneeStatus)
	}
	if !strings.Contains(res.KneeReason, "at least two declared coordinates") {
		t.Fatalf("KneeReason = %q, want coordinate count error", res.KneeReason)
	}
}

func TestSweepEnvelopeBindsTraceDigestAndSemantics(t *testing.T) {
	baseTrace := fixtureTrace()
	baseOpts := Options{Budgets: []int{2, 4, 8}, WriteDelayNs: 0}
	baseRes := Sweep(baseTrace, baseOpts)
	if !strings.HasPrefix(baseRes.EnvelopeDigest, "sha256:") {
		t.Fatalf("base EnvelopeDigest = %q, want sha256: prefix", baseRes.EnvelopeDigest)
	}

	// 1. Altering trace tokens changes EnvelopeDigest
	tokenModTrace := fixtureTrace()
	tokenModTrace.Accesses[0].Tokens = []int{9, 9, 9}
	tokenModRes := Sweep(tokenModTrace, baseOpts)
	if tokenModRes.EnvelopeDigest == baseRes.EnvelopeDigest {
		t.Fatalf("EnvelopeDigest did not change when altering trace tokens: %q", tokenModRes.EnvelopeDigest)
	}

	// 2. Altering access count changes EnvelopeDigest
	countModTrace := fixtureTrace()
	countModTrace.Accesses = append(countModTrace.Accesses, Access{
		Tokens: []int{1, 2, 3},
		TimeNs: int64(len(countModTrace.Accesses)),
	})
	countModRes := Sweep(countModTrace, baseOpts)
	if countModRes.EnvelopeDigest == baseRes.EnvelopeDigest {
		t.Fatalf("EnvelopeDigest did not change when altering access count: %q", countModRes.EnvelopeDigest)
	}
	if countModRes.Accesses == baseRes.Accesses {
		t.Fatalf("access count was not altered: got %d want %d", countModRes.Accesses, len(countModTrace.Accesses))
	}

	// 3. Altering write delay changes EnvelopeDigest
	delayModOpts := baseOpts
	delayModOpts.WriteDelayNs = 10
	delayModRes := Sweep(baseTrace, delayModOpts)
	if delayModRes.EnvelopeDigest == baseRes.EnvelopeDigest {
		t.Fatalf("EnvelopeDigest did not change when altering write delay: %q", delayModRes.EnvelopeDigest)
	}
}

func TestSweepKneeCertificationInteriorMeasured(t *testing.T) {
	// fixtureTrace() ceiling is 11/15 (0.7333). With KneeFraction 0.99, threshold is 0.726.
	// Budget 2 yields 9/15 (0.600) < 0.726.
	// Budget 4 yields 11/15 (0.7333) >= 0.726 (interior crossing).
	// Budget 8 yields 11/15 (0.7333) >= 0.726.
	res := Sweep(fixtureTrace(), Options{Budgets: []int{2, 4, 8}})
	if res.KneeStatus != string(sweepcert.FindingMeasured) {
		t.Fatalf("KneeStatus = %q, want %q", res.KneeStatus, sweepcert.FindingMeasured)
	}
	if !res.KneeReached {
		t.Fatalf("expected KneeReached to be true")
	}
	if res.Knee.Budget != 4 {
		t.Fatalf("Knee budget = %d, want 4", res.Knee.Budget)
	}
	if res.Interval == nil {
		t.Fatalf("Interval is nil, want populated interval for measured knee")
	}
	if res.Interval.LowerPointID != "budget:2" || res.Interval.UpperPointID != "budget:4" {
		t.Fatalf("Interval = %+v, want LowerPointID: budget:2, UpperPointID: budget:4", res.Interval)
	}
	wantSupporting := []string{"budget:2", "budget:4", "budget:8"}
	if !reflect.DeepEqual(res.SupportingPoints, wantSupporting) {
		t.Fatalf("SupportingPoints = %v, want %v", res.SupportingPoints, wantSupporting)
	}
}

func TestSweepKneeCertificationLeftCensored(t *testing.T) {
	// First sampled budget (4) immediately clears the 99% ceiling threshold (0.726).
	// The crossing occurred at or before the first sampled coordinate.
	res := Sweep(fixtureTrace(), Options{Budgets: []int{4, 8}})
	if res.KneeStatus != string(sweepcert.FindingLeftCensored) {
		t.Fatalf("KneeStatus = %q, want %q", res.KneeStatus, sweepcert.FindingLeftCensored)
	}
	if !res.KneeReached {
		t.Fatalf("expected KneeReached to be true for left-censored crossing")
	}
	if res.Knee.Budget != 4 {
		t.Fatalf("Knee budget = %d, want 4", res.Knee.Budget)
	}
	if res.Interval == nil {
		t.Fatalf("Interval is nil, want populated interval for left-censored knee")
	}
	if res.Interval.UpperPointID != "budget:4" {
		t.Fatalf("Interval.UpperPointID = %q, want budget:4", res.Interval.UpperPointID)
	}
	if res.Interval.LowerPointID != "" {
		t.Fatalf("Interval.LowerPointID = %q, want empty (unbounded left)", res.Interval.LowerPointID)
	}
}

func TestSweepKneeCertificationRightCensored(t *testing.T) {
	// All sampled budgets (2, 3) fail to cross the 99% ceiling threshold (0.726).
	// Budget 2: 9/15 (0.600), Budget 3: 10/15 (0.6667).
	res := Sweep(fixtureTrace(), Options{Budgets: []int{2, 3}})
	if res.KneeStatus != string(sweepcert.FindingRightCensored) {
		t.Fatalf("KneeStatus = %q, want %q", res.KneeStatus, sweepcert.FindingRightCensored)
	}
	if res.KneeReached {
		t.Fatalf("KneeReached = true, want false when all points fail threshold")
	}
	if res.Interval == nil {
		t.Fatalf("Interval is nil, want populated interval for right-censored knee")
	}
	if res.Interval.LowerPointID != "budget:3" {
		t.Fatalf("Interval.LowerPointID = %q, want budget:3", res.Interval.LowerPointID)
	}
	if res.Interval.UpperPointID != "" {
		t.Fatalf("Interval.UpperPointID = %q, want empty (unbounded right)", res.Interval.UpperPointID)
	}
}

func TestSweepKneeCertificationMissingPointNotIdentifiable(t *testing.T) {
	// Declare budgets 2 and 8, with budget 4 declared as an explicit omission.
	res := Sweep(fixtureTrace(), Options{
		Budgets: []int{2, 8},
		Omissions: []BudgetOmission{
			{Budget: 4, Reason: "allocation failure"},
		},
	})
	if res.KneeStatus != string(sweepcert.FindingNotIdentifiable) {
		t.Fatalf("KneeStatus = %q, want %q", res.KneeStatus, sweepcert.FindingNotIdentifiable)
	}
	if res.Interval == nil {
		t.Fatalf("Interval is nil, want interval spanning the gap")
	}
	if res.Interval.LowerPointID != "budget:2" || res.Interval.UpperPointID != "budget:8" {
		t.Fatalf("Interval = %+v, want LowerPointID: budget:2, UpperPointID: budget:8", res.Interval)
	}

	// Verify absence of zero-filling:
	// The omitted point must NOT be inserted into Curve with a fabricated 0.0 reuse ratio.
	if len(res.Curve) != 2 {
		t.Fatalf("Curve has %d points, want exactly 2 measured points (no zero-filled omission)", len(res.Curve))
	}
	for _, p := range res.Curve {
		if p.Budget == 4 {
			t.Fatalf("omitted budget 4 was zero-filled into Curve: %+v", p)
		}
		if p.ReuseRatio <= 0 {
			t.Fatalf("measured point unexpectedly zero: %+v", p)
		}
	}
}

func TestSweepReconcileOpttargetNonEquivalence(t *testing.T) {
	// cachesweep and opttarget.SavingsVsBudget are explicitly non-equivalent:
	// 1. cachesweep replays hierarchical token sequences through a prefix tree (radixkv)
	//    with longest-prefix matching, edge splitting, and token-level LRU eviction to
	//    measure prefix-token reuse ratio.
	// 2. opttarget.SavingsVsBudget replays discrete scalar keys through a flat LRU cache
	//    with all-or-nothing key hit/miss accounting without prefix sharing or tree structure.

	// Trace with shared prefix [1, 2] and divergent suffixes [3] vs [4].
	tr := Trace{Accesses: []Access{
		{Tokens: []int{1, 2, 3}, TimeNs: 0},
		{Tokens: []int{1, 2, 4}, TimeNs: 1},
	}}

	sweepRes := Sweep(tr, Options{Budgets: []int{2, 4, 8}})

	// In cachesweep, access 1 reuses prefix [1, 2] (2 tokens) out of 3 tokens requested.
	if sweepRes.Ceiling.ReusedTokens != 2 {
		t.Fatalf("cachesweep ceiling reused = %d, want 2", sweepRes.Ceiling.ReusedTokens)
	}
	if !approx(sweepRes.Ceiling.ReuseRatio, 2.0/6.0) {
		t.Fatalf("cachesweep ceiling ratio = %v, want %v", sweepRes.Ceiling.ReuseRatio, 2.0/6.0)
	}
	// At budget 4, the resident tree holds [1, 2, 3] and [4] (4 tokens total), achieving full ceiling reuse.
	if sweepRes.Curve[1].ReusedTokens != 2 {
		t.Fatalf("cachesweep budget 4 reused = %d, want 2", sweepRes.Curve[1].ReusedTokens)
	}
	// cachesweep certifies the sweep envelope and status via sweepcert.
	if sweepRes.EnvelopeDigest == "" || sweepRes.KneeStatus == "" {
		t.Fatalf("cachesweep did not produce sweepcert certification: %+v", sweepRes)
	}

	// Model the same workload as discrete request keys in opttarget:
	// Access 0 is key 1, Access 1 is key 2 (two distinct requests).
	requestKeys := []int{1, 2}
	optBudgets := []int{1, 2, 3}
	optCurve, err := opttarget.SavingsVsBudget(requestKeys, optBudgets)
	if err != nil {
		t.Fatalf("opttarget.SavingsVsBudget failed: %v", err)
	}

	// In opttarget, discrete scalar keys have all-or-nothing hit semantics.
	// Since keys 1 and 2 are distinct, access 1 misses completely despite the shared [1, 2] prefix.
	for _, p := range optCurve.Points {
		if p.Savings != 0 || p.HitRate != 0 {
			t.Fatalf("opttarget discrete keys unexpectedly hit (savings=%v, hit_rate=%v); want 0 due to all-or-nothing semantics", p.Savings, p.HitRate)
		}
	}

	// Even if flattened into a scalar token stream, opttarget manages flat entry counts,
	// not token-tree capacities or prefix-sharing structures.
	flatTokens := []int{1, 2, 3, 1, 2, 4}
	flatCurve, err := opttarget.SavingsVsBudget(flatTokens, []int{1, 2, 3, 4})
	if err != nil {
		t.Fatalf("opttarget.SavingsVsBudget flatTokens failed: %v", err)
	}

	// At budget 2 in opttarget (2 entry capacity), flat LRU evicts tokens without
	// prefix hierarchy: hits = 0 out of 6.
	// In contrast, cachesweep at budget 2 prefix-caches [1, 2] and achieves 2 reused tokens.
	if flatCurve.Points[1].Savings != 0 {
		t.Fatalf("opttarget flat tokens at budget 2 savings = %v, want 0", flatCurve.Points[1].Savings)
	}

	// Furthermore, the knee criteria are mathematically distinct:
	// - cachesweep: smallest budget reaching KneeFraction (0.99) of infinite-cache ceiling.
	// - opttarget: last budget where marginal savings per unit budget >= 10% of peak marginal gain.
	if sweepRes.KneeFraction != DefaultKneeFraction {
		t.Fatalf("cachesweep knee fraction = %v, want %v", sweepRes.KneeFraction, DefaultKneeFraction)
	}
}
