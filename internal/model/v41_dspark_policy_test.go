package model

import (
	"errors"
	"math"
	"testing"
)

// dsparkCurveMid is the shared two-graph profile: a cheap small graph and a
// pricier large one, so marginal slot decisions cross a real boundary.
var dsparkCurveMid = []V41DSparkPerfPoint{
	{GraphSize: 4, BaselineMS: 100},
	{GraphSize: 8, BaselineMS: 120},
}

func assertSeqlens(t *testing.T, got []int, want ...int) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("seqlens = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("seqlens[%d] = %d, want %d (full: %v vs %v)", i, got[i], want[i], got, want)
		}
	}
}

func dsparkSum(seqlens []int) int {
	s := 0
	for _, v := range seqlens {
		s += v
	}
	return s
}

// TestV41DSparkVarlenZeroDraftDegenerate pins the zero-draft corner: with no
// draft positions there is nothing to allocate, so every request verifies its
// golden token alone, under both an empty and a populated profile.
func TestV41DSparkVarlenZeroDraftDegenerate(t *testing.T) {
	policy, err := NewV41DSparkVarlenPolicy(dsparkCurveMid)
	if err != nil {
		t.Fatalf("NewV41DSparkVarlenPolicy: %v", err)
	}
	reqs := []V41DSparkVarlenRequest{
		{DraftCount: 0, ScheduleCnt: 1, ProcessCnt: 1},
		{DraftCount: 0, HasDraftInfo: true, ScheduleCnt: 1, ProcessCnt: 1},
		{DraftCount: 0, ScheduleCnt: 1, ProcessCnt: 0},
	}
	seqlens, err := policy.Schedule(reqs)
	if err != nil {
		t.Fatalf("Schedule: %v", err)
	}
	assertSeqlens(t, seqlens, 1, 1, 1)

	bare, err := NewV41DSparkVarlenPolicy(nil)
	if err != nil {
		t.Fatalf("NewV41DSparkVarlenPolicy(empty): %v", err)
	}
	seqlens, err = bare.Schedule([]V41DSparkVarlenRequest{{DraftCount: 0}, {DraftCount: 0}})
	if err != nil {
		t.Fatalf("Schedule(empty profile): %v", err)
	}
	assertSeqlens(t, seqlens, 1, 1)

	none, err := bare.Schedule(nil)
	if err != nil || none != nil {
		t.Fatalf("Schedule(no requests) = (%v, %v), want (nil, nil)", none, err)
	}
}

// TestV41DSparkVarlenMissingRateFallback pins the conservative padding: a
// request without rate evidence pads 1.0 per position when drafts are known to
// exist and 0.0 otherwise, including the unpaired case where the counters
// themselves prove a draft exists.
func TestV41DSparkVarlenMissingRateFallback(t *testing.T) {
	policy, err := NewV41DSparkVarlenPolicy(dsparkCurveMid)
	if err != nil {
		t.Fatalf("NewV41DSparkVarlenPolicy: %v", err)
	}

	// Known-draft request pads high and takes both extra slots; the unknown
	// request pads low and takes none. Total work reads the same padded queue.
	got, err := policy.Schedule([]V41DSparkVarlenRequest{
		{DraftCount: 2, HasDraftInfo: true, ScheduleCnt: 1, ProcessCnt: 1},
		{DraftCount: 2, ScheduleCnt: 1, ProcessCnt: 1},
	})
	if err != nil {
		t.Fatalf("Schedule(missing rates): %v", err)
	}
	assertSeqlens(t, got, 3, 1)

	// Unpaired request (schedule_cnt == process_cnt+1) proves its own draft,
	// so its missing rates pad 1.0 and both extra slots land on it, ahead of
	// the explicit 0.2 rates.
	fresh, _ := NewV41DSparkVarlenPolicy(dsparkCurveMid)
	got, err = fresh.Schedule([]V41DSparkVarlenRequest{
		{DraftCount: 2, ScheduleCnt: 1, ProcessCnt: 1, AcceptRates: []float64{0.2, 0.2}, PrevAcceptRates: []float64{0.2, 0.2}},
		{DraftCount: 2, ScheduleCnt: 1, ProcessCnt: 0},
	})
	if err != nil {
		t.Fatalf("Schedule(unpaired missing rates): %v", err)
	}
	assertSeqlens(t, got, 1, 3)

	// Four-request mirror: one confident request versus three unknown ones.
	// The total-work decision stops at two extra slots, and both land on the
	// confident request; the pad-0 requests verify goldens only.
	fresh2, _ := NewV41DSparkVarlenPolicy(dsparkCurveMid)
	got, err = fresh2.Schedule([]V41DSparkVarlenRequest{
		{DraftCount: 2, ScheduleCnt: 1, ProcessCnt: 1, AcceptRates: []float64{0.9, 0.9}, PrevAcceptRates: []float64{0.9, 0.9}},
		{DraftCount: 2, ScheduleCnt: 1, ProcessCnt: 1},
		{DraftCount: 2, ScheduleCnt: 1, ProcessCnt: 1},
		{DraftCount: 2, ScheduleCnt: 1, ProcessCnt: 1},
	})
	if err != nil {
		t.Fatalf("Schedule(pad-0 mirror): %v", err)
	}
	assertSeqlens(t, got, 3, 1, 1, 1)

	// Same shape, but the three unknown requests are known to carry drafts, so
	// their conservative pad-1 rates outrank the explicit 0.9 rates and pull
	// the total to the profile ceiling; the extra slots land on the padded
	// requests by index order.
	fresh3, _ := NewV41DSparkVarlenPolicy(dsparkCurveMid)
	got, err = fresh3.Schedule([]V41DSparkVarlenRequest{
		{DraftCount: 2, ScheduleCnt: 1, ProcessCnt: 1, AcceptRates: []float64{0.9, 0.9}, PrevAcceptRates: []float64{0.9, 0.9}},
		{DraftCount: 2, HasDraftInfo: true, ScheduleCnt: 1, ProcessCnt: 1},
		{DraftCount: 2, HasDraftInfo: true, ScheduleCnt: 1, ProcessCnt: 1},
		{DraftCount: 2, HasDraftInfo: true, ScheduleCnt: 1, ProcessCnt: 1},
	})
	if err != nil {
		t.Fatalf("Schedule(pad-1 mirror): %v", err)
	}
	assertSeqlens(t, got, 1, 3, 3, 1)
}

// TestV41DSparkVarlenAgeSeparationCausality pins the N-2/N-1 split: the N-2
// estimates move how much total work is scheduled, while the N-1 estimates only
// redistribute the unchanged total across requests. The boundary between the
// two profiled graphs is steep enough that confident N-2 rates cross it and
// weak ones do not (candidates beyond the largest graph are unbounded and
// never scored).
func TestV41DSparkVarlenAgeSeparationCausality(t *testing.T) {
	curve := []V41DSparkPerfPoint{
		{GraphSize: 4, BaselineMS: 100},
		{GraphSize: 8, BaselineMS: 150},
	}
	mk := func(n2, n1 []float64) V41DSparkVarlenRequest {
		return V41DSparkVarlenRequest{
			DraftCount:      2,
			ScheduleCnt:     1,
			ProcessCnt:      1,
			AcceptRates:     n1,
			PrevAcceptRates: n2,
		}
	}
	run := func(reqs []V41DSparkVarlenRequest) []int {
		t.Helper()
		policy, err := NewV41DSparkVarlenPolicy(curve)
		if err != nil {
			t.Fatalf("NewV41DSparkVarlenPolicy: %v", err)
		}
		got, err := policy.Schedule(reqs)
		if err != nil {
			t.Fatalf("Schedule: %v", err)
		}
		return got
	}

	high := []float64{0.95, 0.95}
	low := []float64{0.5, 0.5}
	n1A := [][]float64{{0.9, 0.9}, {0.8, 0.8}, {0.1, 0.1}, {0.1, 0.1}}
	n1B := [][]float64{{0.1, 0.1}, {0.9, 0.9}, {0.8, 0.8}, {0.1, 0.1}}

	a := run([]V41DSparkVarlenRequest{mk(high, n1A[0]), mk(high, n1A[1]), mk(high, n1A[2]), mk(high, n1A[3])})
	b := run([]V41DSparkVarlenRequest{mk(high, n1B[0]), mk(high, n1B[1]), mk(high, n1B[2]), mk(high, n1B[3])})
	c := run([]V41DSparkVarlenRequest{mk(low, n1A[0]), mk(low, n1A[1]), mk(low, n1A[2]), mk(low, n1A[3])})

	// Identical N-1 evidence, different N-2: totals differ (8 vs 4). With
	// confident N-2 rates the marginal slots cross to the large graph; with
	// weak ones even four extra slots tie the golden candidate and lose.
	if dsparkSum(a) != 8 || dsparkSum(c) != 4 {
		t.Fatalf("N-2 causality lost: high-N2 total=%d (want 8), low-N2 total=%d (want 4)", dsparkSum(a), dsparkSum(c))
	}
	// Identical N-2 evidence, different N-1: totals match, distribution moves.
	if dsparkSum(a) != dsparkSum(b) {
		t.Fatalf("N-1 must not change total work: %d vs %d", dsparkSum(a), dsparkSum(b))
	}
	assertSeqlens(t, a, 3, 3, 1, 1)
	assertSeqlens(t, b, 1, 3, 3, 1)
	assertSeqlens(t, c, 1, 1, 1, 1)
}

// TestV41DSparkVarlenExactTargetVerification pins hand-computed allocations:
// the chosen total batch and which positions absorb the extra slots, including
// the rate-tie break toward lower request indices and the no-profile fallback.
func TestV41DSparkVarlenExactTargetVerification(t *testing.T) {
	curve := []V41DSparkPerfPoint{
		{GraphSize: 2, BaselineMS: 50},
		{GraphSize: 4, BaselineMS: 80},
		{GraphSize: 8, BaselineMS: 400},
	}
	// Two requests, two drafts each. N-2 rates 0.9/0.1 per request: the golden
	// candidate scores 2/50=0.04; the first extra slot crosses to graph 4 and
	// scores 2.9/80=0.03625 (rejected); two extras score 3.8/80=0.0475 (won);
	// further extras cross to graph 8 and lose. Exactly two slots are allocated.
	run := func(n1a, n1b []float64) []int {
		t.Helper()
		policy, err := NewV41DSparkVarlenPolicy(curve)
		if err != nil {
			t.Fatalf("NewV41DSparkVarlenPolicy: %v", err)
		}
		got, err := policy.Schedule([]V41DSparkVarlenRequest{
			{DraftCount: 2, ScheduleCnt: 1, ProcessCnt: 1, AcceptRates: n1a, PrevAcceptRates: []float64{0.9, 0.1}},
			{DraftCount: 2, ScheduleCnt: 1, ProcessCnt: 1, AcceptRates: n1b, PrevAcceptRates: []float64{0.9, 0.1}},
		})
		if err != nil {
			t.Fatalf("Schedule: %v", err)
		}
		return got
	}

	// Distinct N-1 rates: one slot per request.
	assertSeqlens(t, run([]float64{0.9, 0.1}, []float64{0.1, 0.9}), 2, 2)
	// Concentrated N-1 rates: both slots land on request 0.
	assertSeqlens(t, run([]float64{0.9, 0.9}, []float64{0.1, 0.1}), 3, 1)
	// Equal N-1 rates: the tie breaks toward the lower request index.
	assertSeqlens(t, run([]float64{0.5, 0.5}, []float64{0.5, 0.5}), 3, 1)

	// Without any profile the policy verifies every draft position.
	bare, err := NewV41DSparkVarlenPolicy(nil)
	if err != nil {
		t.Fatalf("NewV41DSparkVarlenPolicy(empty): %v", err)
	}
	got, err := bare.Schedule([]V41DSparkVarlenRequest{
		{DraftCount: 2, ScheduleCnt: 1, ProcessCnt: 1},
		{DraftCount: 3, ScheduleCnt: 1, ProcessCnt: 0},
	})
	if err != nil {
		t.Fatalf("Schedule(no profile): %v", err)
	}
	assertSeqlens(t, got, 3, 4)
}

// TestV41DSparkVarlenLearnedTerms pins the learned runtime corrections: the
// forward slope can revoke extra slots, the num-seqs-specific sampler estimate
// outranks the global fallback, and negative learned slopes clamp to zero.
func TestV41DSparkVarlenLearnedTerms(t *testing.T) {
	fiveGraph := []V41DSparkPerfPoint{
		{GraphSize: 2, BaselineMS: 50},
		{GraphSize: 4, BaselineMS: 80},
		{GraphSize: 8, BaselineMS: 400},
	}
	twoGraph := []V41DSparkPerfPoint{
		{GraphSize: 2, BaselineMS: 50},
		{GraphSize: 4, BaselineMS: 80},
	}
	reqs := func() []V41DSparkVarlenRequest {
		return []V41DSparkVarlenRequest{
			{DraftCount: 2, ScheduleCnt: 1, ProcessCnt: 1, AcceptRates: []float64{0.9, 0.1}, PrevAcceptRates: []float64{0.9, 0.1}},
			{DraftCount: 2, ScheduleCnt: 1, ProcessCnt: 1, AcceptRates: []float64{0.9, 0.1}, PrevAcceptRates: []float64{0.9, 0.1}},
		}
	}
	schedule := func(p *V41DSparkVarlenPolicy) []int {
		t.Helper()
		got, err := p.Schedule(reqs())
		if err != nil {
			t.Fatalf("Schedule: %v", err)
		}
		return got
	}

	// Baseline: two extra slots (3.8/80=0.0475 beats 2/50=0.04).
	base, err := NewV41DSparkVarlenPolicy(fiveGraph)
	if err != nil {
		t.Fatalf("NewV41DSparkVarlenPolicy: %v", err)
	}
	assertSeqlens(t, schedule(base), 2, 2)

	// Ten observed steps at graph 4 running 400ms over baseline learn a
	// 100ms/token slope; the penalized large graph loses and all extras are
	// revoked (3.8/480=0.0079 no longer beats 2/250=0.008).
	sloped, err := NewV41DSparkVarlenPolicy(fiveGraph)
	if err != nil {
		t.Fatalf("NewV41DSparkVarlenPolicy(sloped): %v", err)
	}
	for i := 0; i < 10; i++ {
		if err := sloped.UpdateRuntime(2, 4, 480, 0); err != nil {
			t.Fatalf("UpdateRuntime: %v", err)
		}
	}
	assertSeqlens(t, schedule(sloped), 1, 1)

	// Absurdly negative residuals would drive every candidate time negative
	// unless the learned slope clamps at zero; the allocation must stay intact.
	cheapLarge := []V41DSparkPerfPoint{
		{GraphSize: 2, BaselineMS: 50},
		{GraphSize: 4, BaselineMS: 30},
	}
	clamped, err := NewV41DSparkVarlenPolicy(cheapLarge)
	if err != nil {
		t.Fatalf("NewV41DSparkVarlenPolicy(clamped): %v", err)
	}
	for i := 0; i < 10; i++ {
		if err := clamped.UpdateRuntime(2, 4, -100, 0); err != nil {
			t.Fatalf("UpdateRuntime: %v", err)
		}
	}
	assertSeqlens(t, schedule(clamped), 2, 2)

	// Sampler preference: the marginal pair sums to one expected token, so a
	// zero sampler time rejects the crossing (3.0/80=0.0375 < 2/50=0.04) while
	// a heavy one accepts it (3.0/1080=0.00278 > 2/1050=0.0019). Five calm
	// observations at num_seqs=2 plus five heavy ones at num_seqs=3 leave the
	// global fallback median heavy, but the scheduling batch of two must read
	// its own calm estimate and reject the crossing.
	prefReqs := []V41DSparkVarlenRequest{
		{DraftCount: 2, ScheduleCnt: 1, ProcessCnt: 1, AcceptRates: []float64{0.9, 0.1}, PrevAcceptRates: []float64{0.9, 0.1}},
		{DraftCount: 2, ScheduleCnt: 1, ProcessCnt: 1, AcceptRates: []float64{0.9, 0.1}, PrevAcceptRates: []float64{0.1, 0.1}},
	}
	schedulePref := func(p *V41DSparkVarlenPolicy) []int {
		t.Helper()
		got, err := p.Schedule(prefReqs)
		if err != nil {
			t.Fatalf("Schedule: %v", err)
		}
		return got
	}
	preferred, err := NewV41DSparkVarlenPolicy(twoGraph)
	if err != nil {
		t.Fatalf("NewV41DSparkVarlenPolicy(preferred): %v", err)
	}
	for i := 0; i < 5; i++ {
		if err := preferred.UpdateRuntime(2, 4, 80, 0); err != nil {
			t.Fatalf("UpdateRuntime: %v", err)
		}
	}
	for i := 0; i < 5; i++ {
		if err := preferred.UpdateRuntime(3, 4, 80, 1000); err != nil {
			t.Fatalf("UpdateRuntime: %v", err)
		}
	}
	assertSeqlens(t, schedulePref(preferred), 1, 1)

	// Control: the same batch fed heavy observations through its own channel
	// crosses the boundary and takes both extra slots.
	heavy, err := NewV41DSparkVarlenPolicy(twoGraph)
	if err != nil {
		t.Fatalf("NewV41DSparkVarlenPolicy(heavy): %v", err)
	}
	for i := 0; i < 5; i++ {
		if err := heavy.UpdateRuntime(2, 4, 80, 1000); err != nil {
			t.Fatalf("UpdateRuntime: %v", err)
		}
	}
	assertSeqlens(t, schedulePref(heavy), 2, 2)
}

// TestV41DSparkVarlenCounterAndRateInvariants pins the fail-closed contract:
// counter violations, rate-length mismatches, non-finite rates, and batches
// beyond the profile are all refused with the typed sentinel.
func TestV41DSparkVarlenCounterAndRateInvariants(t *testing.T) {
	policy, err := NewV41DSparkVarlenPolicy(dsparkCurveMid)
	if err != nil {
		t.Fatalf("NewV41DSparkVarlenPolicy: %v", err)
	}
	valid := V41DSparkVarlenRequest{DraftCount: 1, ScheduleCnt: 1, ProcessCnt: 1, AcceptRates: []float64{0.5}}
	cases := map[string]V41DSparkVarlenRequest{
		"process ahead of schedule": {DraftCount: 1, ScheduleCnt: 1, ProcessCnt: 2},
		"schedule two ahead":        {DraftCount: 1, ScheduleCnt: 2, ProcessCnt: 0},
		"negative draft count":      {DraftCount: -1, ScheduleCnt: 0, ProcessCnt: 0},
		"rate length mismatch":      {DraftCount: 2, ScheduleCnt: 1, ProcessCnt: 1, AcceptRates: []float64{0.5}},
		"prev rate length mismatch": {DraftCount: 1, ScheduleCnt: 1, ProcessCnt: 1, AcceptRates: []float64{0.5}, PrevAcceptRates: []float64{0.5, 0.5}},
		"nan rate":                  {DraftCount: 1, ScheduleCnt: 1, ProcessCnt: 1, AcceptRates: []float64{math.NaN()}},
		"infinite prev rate":        {DraftCount: 1, ScheduleCnt: 1, ProcessCnt: 1, AcceptRates: []float64{0.5}, PrevAcceptRates: []float64{math.Inf(1)}},
	}
	for name, req := range cases {
		if _, err := policy.Schedule([]V41DSparkVarlenRequest{req}); err == nil {
			t.Errorf("%s: Schedule accepted an invalid request", name)
		} else if !errors.Is(err, ErrV41DSparkVarlenInvalidRequest) {
			t.Errorf("%s: error %v does not wrap %v", name, err, ErrV41DSparkVarlenInvalidRequest)
		}
	}

	// A batch wider than every profiled graph cannot be bounded.
	wide, err := NewV41DSparkVarlenPolicy([]V41DSparkPerfPoint{{GraphSize: 2, BaselineMS: 50}})
	if err != nil {
		t.Fatalf("NewV41DSparkVarlenPolicy(wide): %v", err)
	}
	if _, err := wide.Schedule([]V41DSparkVarlenRequest{valid, valid, valid, valid}); !errors.Is(err, ErrV41DSparkVarlenBatchExceedsProfile) {
		t.Errorf("wide batch: error %v, want %v", err, ErrV41DSparkVarlenBatchExceedsProfile)
	}

	// Observation channels reject the same garbage.
	if err := policy.UpdateRuntime(-1, 4, 100, 0); !errors.Is(err, ErrV41DSparkVarlenInvalidRequest) {
		t.Errorf("UpdateRuntime(negative num_seqs): error %v", err)
	}
	if err := policy.UpdateRuntime(2, 0, 100, 0); !errors.Is(err, ErrV41DSparkVarlenInvalidRequest) {
		t.Errorf("UpdateRuntime(zero graph): error %v", err)
	}
	if err := policy.UpdateRuntime(2, 4, math.NaN(), 0); !errors.Is(err, ErrV41DSparkVarlenInvalidRequest) {
		t.Errorf("UpdateRuntime(nan forward): error %v", err)
	}
	if err := policy.UpdateRuntime(2, 16, 100, 0); !errors.Is(err, ErrV41DSparkVarlenBatchExceedsProfile) {
		t.Errorf("UpdateRuntime(beyond profile): error %v", err)
	}

	// Profile curves must stay strictly increasing with finite baselines.
	if _, err := NewV41DSparkVarlenPolicy([]V41DSparkPerfPoint{{GraphSize: 4, BaselineMS: 100}, {GraphSize: 4, BaselineMS: 120}}); !errors.Is(err, ErrV41DSparkVarlenInvalidRequest) {
		t.Errorf("non-increasing curve accepted: %v", err)
	}
	if _, err := NewV41DSparkVarlenPolicy([]V41DSparkPerfPoint{{GraphSize: 4, BaselineMS: math.NaN()}}); !errors.Is(err, ErrV41DSparkVarlenInvalidRequest) {
		t.Errorf("nan baseline accepted: %v", err)
	}
	if err := policy.SetDecodePerfCurve(nil); err != nil {
		t.Errorf("SetDecodePerfCurve(empty): %v", err)
	}
}

// TestV41DSparkVarlenEstimatorUnits pins the sliding-window median estimators:
// minimum-sample gating, even/odd medians, ring wraparound, per-graph admission
// thresholds, and the 5000-observation window cap.
func TestV41DSparkVarlenEstimatorUnits(t *testing.T) {
	var sampler V41DSparkSamplerTimeEstimator
	if v, ok := sampler.Get(); ok || v != 0 {
		t.Fatalf("empty sampler Get = (%v, %v), want (0, false)", v, ok)
	}
	for _, v := range []float64{10, 20, 30, 40} {
		sampler.Update(v)
	}
	if _, ok := sampler.Get(); ok {
		t.Fatal("sampler reports before its minimum sample count")
	}
	sampler.Update(50)
	if v, ok := sampler.Get(); !ok || v != 30 {
		t.Fatalf("sampler median of 5 = (%v, %v), want (30, true)", v, ok)
	}
	sampler.Update(60)
	if v, _ := sampler.Get(); v != 35 {
		t.Fatalf("sampler even median = %v, want 35", v)
	}
	for _, v := range []float64{70, 80, 90} {
		sampler.Update(v)
	}
	if v, _ := sampler.Get(); v != 50 {
		t.Fatalf("sampler full-window median = %v, want 50", v)
	}
	sampler.Update(100) // wraps the ring, evicting the oldest sample
	if v, _ := sampler.Get(); v != 60 {
		t.Fatalf("sampler wrapped median = %v, want 60", v)
	}

	var slope V41DSparkForwardSlopeEstimator
	if slope.Get() != 0 {
		t.Fatalf("empty slope Get = %v, want 0", slope.Get())
	}
	for i := 0; i < 9; i++ {
		slope.Update(4, 2)
	}
	if slope.Get() != 0 {
		t.Fatalf("slope admits a graph below its minimum sample count: %v", slope.Get())
	}
	slope.Update(4, 2)
	if slope.Get() != 2 {
		t.Fatalf("single-group slope = %v, want 2", slope.Get())
	}
	for i := 0; i < 12; i++ {
		if i%2 == 0 {
			slope.Update(8, 1)
		} else {
			slope.Update(8, 3)
		}
	}
	if slope.Get() != 2 {
		t.Fatalf("two-group slope = %v, want 2 (medians 2 and 2)", slope.Get())
	}

	// Window cap: 5000 stale observations followed by 10 fresh ones retain only
	// the freshest 5000, so both groups stay admitted.
	capped := V41DSparkForwardSlopeEstimator{}
	for i := 0; i < 5000; i++ {
		capped.Update(4, 2)
	}
	for i := 0; i < 10; i++ {
		capped.Update(8, 100)
	}
	if v := capped.Get(); v != 51 {
		t.Fatalf("capped slope = %v, want 51 (median of 2 and 100)", v)
	}
}

// fak-test:runtime fast est=1ms
func TestV41DSparkVarlenIntegerBoundaries(t *testing.T) {
	t.Parallel()
	maxInt := int(^uint(0) >> 1)
	policy, err := NewV41DSparkVarlenPolicy(nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, reqs := range [][]V41DSparkVarlenRequest{
		{{DraftCount: maxInt}},
		{{DraftCount: maxInt - 1}, {DraftCount: 1}},
		{{DraftCount: maxInt - 3}, {DraftCount: 2}},
		{{DraftCount: maxInt - 1}, {DraftCount: maxInt - 1}},
	} {
		if _, err := policy.Schedule(reqs); !errors.Is(err, ErrV41DSparkVarlenInvalidRequest) {
			t.Fatalf("overflowing token count: got %v, want invalid-request refusal", err)
		}
	}
	// This boundary fits exactly and allocates only a one-element result:
	// an empty profile returns lengths without materializing draft positions.
	got, err := policy.Schedule([]V41DSparkVarlenRequest{{DraftCount: maxInt - 1}})
	if err != nil || len(got) != 1 || got[0] != maxInt {
		t.Fatalf("exact maximum: got %v, %v", got, err)
	}
	// Two golden tokens plus both drafts fit exactly; the second request
	// consumes the final representable slot after the first was accumulated.
	got, err = policy.Schedule([]V41DSparkVarlenRequest{{DraftCount: maxInt - 3}, {DraftCount: 1}})
	if err != nil || len(got) != 2 || got[0] != maxInt-2 || got[1] != 2 {
		t.Fatalf("exact aggregate maximum: got %v, %v", got, err)
	}
	for _, process := range []int{maxInt, maxInt - 1} {
		got, err := policy.Schedule([]V41DSparkVarlenRequest{{ScheduleCnt: maxInt, ProcessCnt: process}})
		if err != nil || len(got) != 1 || got[0] != 1 {
			t.Fatalf("valid maximum counters (process=%d): got %v, %v", process, got, err)
		}
	}
}

// fak-test:runtime fast est=1s lane=default
func TestV41DSparkRuntimeResidualOverflowIsAtomic(t *testing.T) {
	for _, sign := range []float64{-1, 1} {
		p, err := NewV41DSparkVarlenPolicy([]V41DSparkPerfPoint{{GraphSize: 1, BaselineMS: -sign * math.MaxFloat64}})
		if err != nil {
			t.Fatal(err)
		}
		if err := p.UpdateRuntime(1, 1, -sign*math.MaxFloat64, 3); err != nil {
			t.Fatal(err)
		}
		slopeBefore := append([]v41DSparkSlopeSample(nil), p.slope.window...)
		samplerBefore, fallbackBefore := *p.samplerBySeqs[1], *p.samplerFallback
		for _, seqs := range []int{1, 2} {
			if err := p.UpdateRuntime(seqs, 1, sign*math.MaxFloat64, 7); !errors.Is(err, ErrV41DSparkVarlenInvalidRequest) {
				t.Fatalf("sign %v: overflow error = %v", sign, err)
			}
			if len(p.slope.window) != len(slopeBefore) || p.slope.window[0] != slopeBefore[0] ||
				*p.samplerBySeqs[1] != samplerBefore || *p.samplerFallback != fallbackBefore || len(p.samplerBySeqs) != 1 {
				t.Fatal("rejected residual mutated observation state")
			}
		}
	}
	p, err := NewV41DSparkVarlenPolicy([]V41DSparkPerfPoint{{GraphSize: 2, BaselineMS: 10}})
	if err != nil {
		t.Fatal(err)
	}
	if err := p.UpdateRuntime(1, 2, -10, 0); err != nil {
		t.Fatal(err)
	}
	if got := p.slope.window[0].residual; got != -10 {
		t.Fatalf("finite negative residual = %v, want -10", got)
	}
}
