package model

import (
	"errors"
	"math"
	"testing"
	"time"
)

const draftDepthTol = 1e-9

// fak-test:runtime fast est=1ms
func TestDraftDepthRejectsForgedTableShape(t *testing.T) {
	t.Parallel()
	maxInt := int(^uint(0) >> 1)
	for _, rows := range []int{-1, 0, 1, DraftDepthMaxRows + 1, maxInt} {
		costs := DraftDepthCosts{Rows: rows}
		if rule, err := NewDraftDepthRule(costs, 1, nil, 1); !errors.Is(err, ErrDraftDepthVerifyTable) || rule != nil {
			t.Fatalf("Rows=%d: rule=%v error=%v, want table-shape refusal", rows, rule, err)
		}
		for _, k := range []int{1, DraftDepthMaxRows, maxInt - 1} {
			if got := costs.Row(k); got != 0 {
				t.Fatalf("Rows=%d Row(%d)=%v, want unpriceable zero", rows, k, got)
			}
		}
	}
	// The widest representable table remains legal and prices its final row.
	costs := DraftDepthCosts{Rows: DraftDepthMaxRows}
	for i := range costs.Verify {
		costs.Verify[i] = float64(i)
	}
	if _, err := NewDraftDepthRule(costs, DraftDepthMaxRows-1, nil, 1); err != nil {
		t.Fatal(err)
	}
	if got := costs.Row(DraftDepthMaxRows - 1); got != 1 {
		t.Fatalf("last valid row=%v, want 1", got)
	}
	if got := costs.Row(DraftDepthMaxRows); got != 0 {
		t.Fatalf("past last row=%v, want 0", got)
	}
}

func draftDepthApproxEq(t *testing.T, what string, got, want float64) {
	t.Helper()
	if math.Abs(got-want) > draftDepthTol {
		t.Fatalf("%s = %v, want %v", what, got, want)
	}
}

// draftDepthFixture is the measured one-regime verify table (ms per window
// width, index 0 unused) pinned by the adapted TensorFold cost model.
func draftDepthFixture() []float64 {
	return []float64{0, 9.85, 12.44, 14.72, 16.92, 18.51, 20.11, 21.7, 23.03, 24.36, 25.69, 26.95, 28.22, 29.33, 30.36, 31.22, 32.08}
}

// TestDraftDepthPooledCostsAndKeepStop pins the adapted TensorFold draft-depth
// cost model (zig/src/core/draft_depth.zig Costs/DepthRule @
// 9e8c368b0b4c2c7892a0c9f0f302d7c164016209): pooled marginal verify ms never
// add more ms as rows grow inside a kernel regime; draft 1 is always kept;
// low confidence rejects row 2 while high confidence accepts it; More charges
// the row plus its level; and the token/ms EMAs follow the cost model, never
// a wall clock.
func TestDraftDepthPooledCostsAndKeepStop(t *testing.T) {
	v := draftDepthFixture()

	t.Run("pooled costs never add more ms as rows grow", func(t *testing.T) {
		c, err := NewDraftDepthCosts(v, 0.6132, 16)
		if err != nil {
			t.Fatalf("NewDraftDepthCosts: %v", err)
		}
		if c.Rows != 16 {
			t.Fatalf("Rows = %d, want 16", c.Rows)
		}
		draftDepthApproxEq(t, "Level", c.Level, 0.6132)
		draftDepthApproxEq(t, "Verify[1]", c.Verify[1], 9.85)
		// The last marginal (0.86) does not rise above its predecessor, so it
		// pools alone and the widest window lands on its measured value.
		draftDepthApproxEq(t, "Verify[16]", c.Verify[16], 32.08)
		for k := 2; k < c.Rows; k++ {
			if c.Row(k) > c.Row(k-1)+1e-9 {
				t.Fatalf("row %d adds %.9f ms, more than row %d's %.9f ms", k, c.Row(k), k-1, c.Row(k-1))
			}
		}
		// Pool anchors: Row(k) prices verify[k+1]-verify[k]. Window 5's
		// marginal (1.59) rises into window 6's (1.60), so those two pool to
		// an average of 1.595 charged at Row(4) and Row(5); window 7 keeps
		// its own 1.59; and windows 11 and 12 (1.26, 1.27) pool to 1.265
		// charged at Row(10) and Row(11).
		draftDepthApproxEq(t, "Row(4)", c.Row(4), 1.595)
		draftDepthApproxEq(t, "Row(5)", c.Row(5), 1.595)
		draftDepthApproxEq(t, "Row(6)", c.Row(6), 1.59)
		draftDepthApproxEq(t, "Row(10)", c.Row(10), 1.265)
		draftDepthApproxEq(t, "Row(11)", c.Row(11), 1.265)
		draftDepthApproxEq(t, "Verify[4]", c.Verify[4], 16.92)
		draftDepthApproxEq(t, "Verify[7]", c.Verify[7], 21.7)
		draftDepthApproxEq(t, "Verify[11]", c.Verify[11], 26.955)
	})

	t.Run("keep stops at the priced row", func(t *testing.T) {
		c, err := NewDraftDepthCosts(v, 0.6132, 16)
		if err != nil {
			t.Fatalf("NewDraftDepthCosts: %v", err)
		}
		r, err := NewDraftDepthRule(c, 15, nil, 2.0)
		if err != nil {
			t.Fatalf("NewDraftDepthRule: %v", err)
		}
		// Row 1 always verifies, whatever the confidence.
		if !r.Keep(1, 0.0) {
			t.Fatalf("Keep(1, 0.0) = false, want row 1 always kept")
		}
		// Rate = 2.0/(12.44+0.6132); row 2 costs 2.28 ms, so
		// pow(run,2) >= rate*2.28 flips near run ~ 0.591.
		if !r.Keep(2, 0.99) {
			t.Fatalf("Keep(2, 0.99) = false, want high confidence to accept row 2")
		}
		if r.Keep(2, 0.1) {
			t.Fatalf("Keep(2, 0.1) = true, want low confidence to reject row 2")
		}
		if r.Keep(2, 0.59) {
			t.Fatalf("Keep(2, 0.59) = true, want below the priced threshold to reject")
		}
		if !r.Keep(2, 0.60) {
			t.Fatalf("Keep(2, 0.60) = false, want above the priced threshold to accept")
		}
		// More charges the row AND its level: rate*(2.28+0.6132) flips near run ~ 0.666.
		if r.More(1, 0.66) {
			t.Fatalf("More(1, 0.66) = true, want row+level cost to reject")
		}
		if !r.More(1, 0.67) {
			t.Fatalf("More(1, 0.67) = false, want row+level cost to accept")
		}
		if r.More(15, 1.0) {
			t.Fatalf("More(15, 1.0) = true, want nothing drafted past the bound")
		}
		// Done follows the round from the cost model, never a clock:
		// tokens 2 -> 2 + 0.125*(4-2) = 2.25;
		// ms 13.0532 -> 13.0532 + 0.125*(16.92 + 3*0.6132 - 13.0532) = 13.7665.
		r.Done(4, 4, 3)
		if r.Tokens() != 2.25 {
			t.Fatalf("Tokens = %v, want 2.25", r.Tokens())
		}
		draftDepthApproxEq(t, "MS", r.MS(), 13.7665)
	})

	t.Run("confidence floor overrides the cost test", func(t *testing.T) {
		c, err := NewDraftDepthCosts(v, 0.6132, 16)
		if err != nil {
			t.Fatalf("NewDraftDepthCosts: %v", err)
		}
		floor := 0.5
		r, err := NewDraftDepthRule(c, 15, &floor, 2.0)
		if err != nil {
			t.Fatalf("NewDraftDepthRule: %v", err)
		}
		if !r.Keep(1, 0.0) {
			t.Fatalf("Keep(1, 0.0) = false, want row 1 kept even under a floor")
		}
		if r.Keep(2, 0.49) {
			t.Fatalf("Keep(2, 0.49) = true, want below-floor to reject")
		}
		if !r.Keep(2, 0.5) {
			t.Fatalf("Keep(2, 0.5) = false, want at-floor to accept")
		}
		if r.More(1, 0.4999) {
			t.Fatalf("More(1, 0.4999) = true, want below-floor to reject")
		}
		if !r.More(1, 0.5) {
			t.Fatalf("More(1, 0.5) = false, want at-floor to accept")
		}
	})

	t.Run("pooling stops at a kernel regime", func(t *testing.T) {
		// Two regimes of 16 widths: rows 1-16 price themselves, row 17 steps
		// up into the next kernel regime and can never pool back across it.
		var v2 [33]float64
		small := []float64{46.33, 54.42, 60.47, 66.52, 71.58, 76.64, 81.3, 85.95, 90.52, 94.65, 98.78, 102.91, 107.04, 111.18, 115.31, 117.58}
		copy(v2[1:17], small)
		v2[17] = 122.58
		for r := 18; r <= 32; r++ {
			v2[r] = v2[r-1] + 2.4 + 0.1*float64(r%3)
		}
		c, err := NewDraftDepthCosts(v2[:], 3.5, 16)
		if err != nil {
			t.Fatalf("NewDraftDepthCosts: %v", err)
		}
		if c.Rows != 32 {
			t.Fatalf("Rows = %d, want 32", c.Rows)
		}
		alone, err := NewDraftDepthCosts(v2[:17], 3.5, 16)
		if err != nil {
			t.Fatalf("NewDraftDepthCosts(alone): %v", err)
		}
		for i := 1; i <= 16; i++ {
			if math.Abs(c.Verify[i]-alone.Verify[i]) > draftDepthTol {
				t.Fatalf("Verify[%d] = %v, want %v (first regime must ignore later data)", i, c.Verify[i], alone.Verify[i])
			}
		}
		draftDepthApproxEq(t, "Verify[16]", c.Verify[16], 117.58)
		draftDepthApproxEq(t, "Verify[17]", c.Verify[17], 122.58)
		draftDepthApproxEq(t, "Verify[32]", c.Verify[32], v2[32])
		for r := 19; r <= 32; r++ {
			if c.Row(r-1) > c.Row(r-2)+1e-9 {
				t.Fatalf("row %d adds %.9f ms, more than row %d's %.9f ms", r-1, c.Row(r-1), r-2, c.Row(r-2))
			}
		}
	})

	t.Run("closed refusals", func(t *testing.T) {
		if _, err := NewDraftDepthCosts([]float64{0, 9.85}, 0.6, 16); !errors.Is(err, ErrDraftDepthVerifyTable) {
			t.Fatalf("short table error = %v, want ErrDraftDepthVerifyTable", err)
		}
		if _, err := NewDraftDepthCosts(make([]float64, 34), 0.6, 16); !errors.Is(err, ErrDraftDepthVerifyTable) {
			t.Fatalf("oversize table error = %v, want ErrDraftDepthVerifyTable", err)
		}
		if _, err := NewDraftDepthCosts(v, 0.6, 0); !errors.Is(err, ErrDraftDepthRegimeInvalid) {
			t.Fatalf("zero regime error = %v, want ErrDraftDepthRegimeInvalid", err)
		}
		c, err := NewDraftDepthCosts(v, 0.6132, 16)
		if err != nil {
			t.Fatalf("NewDraftDepthCosts: %v", err)
		}
		if _, err := NewDraftDepthRule(c, 0, nil, 1.0); !errors.Is(err, ErrDraftDepthMostOutOfRange) {
			t.Fatalf("most=0 error = %v, want ErrDraftDepthMostOutOfRange", err)
		}
		if _, err := NewDraftDepthRule(c, 16, nil, 1.0); !errors.Is(err, ErrDraftDepthMostOutOfRange) {
			t.Fatalf("most=Rows error = %v, want ErrDraftDepthMostOutOfRange", err)
		}
		if _, err := NewDraftDepthRule(c, 15, nil, 0); !errors.Is(err, ErrDraftDepthPowerInvalid) {
			t.Fatalf("power=0 error = %v, want ErrDraftDepthPowerInvalid", err)
		}
		if _, err := NewDraftDepthRule(c, 15, nil, math.NaN()); !errors.Is(err, ErrDraftDepthPowerInvalid) {
			t.Fatalf("NaN power error = %v, want ErrDraftDepthPowerInvalid", err)
		}
	})
}

// fak-test:runtime fast est=1ms
// TestQwen38MTP_Adaptive_SplitRankCostParity witnesses the ticket's split-rank
// parity leaf: two governors fed identical cost-model observations with
// different rank splits emit identical depth sequences. Each governor carries
// its own DraftDepthRule built from the same measured table so the modeled
// pricing state compares independently. The ticket's fence forces the shape:
// the pre-existing speedup EMA adjudication may legitimately consume the split
// wall-clock latencies, so depth-sequence equality is witnessed on the
// explicit-Speedup sequence (where the clock is overridden), while the
// zero-Speedup sequence asserts only modeled-state parity. No assertion reads
// a wall clock; the witness is modeled pricing state and depth sequences.
// Software witness only, no throughput or hardware claim.
func TestQwen38MTP_Adaptive_SplitRankCostParity(t *testing.T) {
	cfg := DefaultQwen38AdaptiveConfig()
	cfg.ColdStartDepth = 2
	cfg.MaxDepth = 4
	cfg.HysteresisSteps = 3
	cfg.ProbeInterval = 5

	type splitObs struct {
		proposed, accepted int
		speedup            float64
	}
	withClock := func(o splitObs, step, target time.Duration) Qwen38AdaptiveStepObservation {
		return Qwen38AdaptiveStepObservation{
			ProposedTokens: o.proposed, AcceptedTokens: o.accepted, Speedup: o.speedup,
			StepLatency: step, TargetLatency: target,
		}
	}
	// Two ranks splitting the same logical work very differently on the
	// clock: a fast 10ms step against a 100ms target, a slow 300ms step
	// against a 10ms target. Any pricing path that reads the clock must
	// break the parity below.
	fastStep, fastTarget := 10*time.Millisecond, 100*time.Millisecond
	slowStep, slowTarget := 300*time.Millisecond, 10*time.Millisecond

	newPair := func() (*Qwen38MTPAdaptiveDepthGovernor, *DraftDepthRule) {
		c, err := NewDraftDepthCosts(draftDepthFixture(), 0.6132, 16)
		if err != nil {
			t.Fatalf("NewDraftDepthCosts: %v", err)
		}
		rule, err := NewDraftDepthRule(c, 15, nil, 2.0)
		if err != nil {
			t.Fatalf("NewDraftDepthRule: %v", err)
		}
		gov, err := NewQwen38MTPAdaptiveDepthGovernor(cfg)
		if err != nil {
			t.Fatalf("failed to create governor: %v", err)
		}
		gov.SetCostRule(rule)
		return gov, rule
	}

	// The full adjudication tour: hysteresis ramp 2->3->4, net-negative
	// escape to target-only, probe-interval holds, probe, resume.
	seq := []splitObs{
		{2, 2, 1.45}, {2, 2, 1.45}, {2, 2, 1.45},
		{3, 3, 1.60}, {3, 3, 1.60}, {3, 3, 1.60},
		{4, 0, 0.72},
		{0, 0, 1.0}, {0, 0, 1.0}, {0, 0, 1.0}, {0, 0, 1.0}, {0, 0, 1.0},
		{1, 1, 1.25},
	}
	wantDepths := []int{2, 2, 3, 3, 3, 4, 0, 0, 0, 0, 0, 1, 1}
	wantActions := []string{"hold", "hold", "ramp_up", "hold", "hold", "ramp_up",
		"escape", "hold", "hold", "hold", "hold", "probe", "resume"}

	assertModelParity := func(what string, a, b *Qwen38MTPAdaptiveDepthGovernor, ra, rb *DraftDepthRule) {
		if ra.Tokens() != rb.Tokens() {
			t.Fatalf("%s: Tokens fast=%v slow=%v, want equal modeled state", what, ra.Tokens(), rb.Tokens())
		}
		draftDepthApproxEq(t, what+": MS parity", ra.MS(), rb.MS())
		keepTrue, keepFalse := false, false
		for _, k := range []int{1, 2, 4, 8, 15} {
			for _, run := range []float64{0.0, 0.3, 0.5, 0.61, 0.67, 0.9, 1.0} {
				ka, kb := a.ModeledKeep(k, run), b.ModeledKeep(k, run)
				if ka != kb {
					t.Fatalf("%s: ModeledKeep(%d, %v) fast=%v slow=%v", what, k, run, ka, kb)
				}
				ma, mb := a.ModeledMore(k, run), b.ModeledMore(k, run)
				if ma != mb {
					t.Fatalf("%s: ModeledMore(%d, %v) fast=%v slow=%v", what, k, run, ma, mb)
				}
				keepTrue, keepFalse = keepTrue || ka, keepFalse || !ka
			}
		}
		if !keepTrue || !keepFalse {
			t.Fatalf("%s: keep grid degenerate (true=%v false=%v), parity not exercised", what, keepTrue, keepFalse)
		}
	}

	t.Run("explicit speedup: identical depth sequences across rank splits", func(t *testing.T) {
		govA, ruleA := newPair()
		govB, ruleB := newPair()
		if govA.CostRule() == govB.CostRule() {
			t.Fatal("governors share a cost rule; parity would be trivial")
		}
		for i, o := range seq {
			da, ra, err := govA.ObserveStep(withClock(o, fastStep, fastTarget))
			if err != nil {
				t.Fatalf("fast split step %d: %v", i+1, err)
			}
			db, rb, err := govB.ObserveStep(withClock(o, slowStep, slowTarget))
			if err != nil {
				t.Fatalf("slow split step %d: %v", i+1, err)
			}
			if da != db {
				t.Fatalf("step %d: fast split depth %d != slow split depth %d", i+1, da, db)
			}
			if da != wantDepths[i] {
				t.Fatalf("step %d: depth %d, want %d", i+1, da, wantDepths[i])
			}
			if ra.Action != rb.Action || ra.Action != wantActions[i] {
				t.Fatalf("step %d: actions %q/%q, want %q", i+1, ra.Action, rb.Action, wantActions[i])
			}
			if i == 0 {
				// Ground the pricing path absolutely: one drafting step folds
				// Done(3, 3, 1) into the fresh EMAs (tokens 2.0, ms 13.0532):
				// tokens -> 2 + 0.125*(3-2) = 2.125; ms -> 13.0532 +
				// 0.125*(14.72+0.6132-13.0532) = 13.3382. Neither split moves it.
				if ruleA.Tokens() != 2.125 || ruleB.Tokens() != 2.125 {
					t.Fatalf("step 1 Tokens fast=%v slow=%v, want 2.125", ruleA.Tokens(), ruleB.Tokens())
				}
				draftDepthApproxEq(t, "fast MS after step 1", ruleA.MS(), 13.3382)
				draftDepthApproxEq(t, "slow MS after step 1", ruleB.MS(), 13.3382)
			}
		}
		if ruleA.Tokens() == 2.0 {
			t.Fatal("rule state never left its cold-start anchor; pricing not exercised")
		}
		assertModelParity("after full sequence", govA, govB, ruleA, ruleB)
	})

	t.Run("zero speedup: modeled parity under a live clock split", func(t *testing.T) {
		// Fence-forced shape: with Speedup left at zero, ComputeSpeedup derives
		// the effective speedup from the split wall-clock latencies, which the
		// fenced speedup EMA adjudication may legitimately consume. Depth
		// sequences may diverge here; the honest witness is that the modeled
		// pricing state and its decisions still agree across the split.
		govA, ruleA := newPair()
		govB, ruleB := newPair()
		liveSeq := []splitObs{
			{2, 2, 0}, {2, 2, 0}, {2, 2, 0},
			{0, 0, 0}, {0, 0, 0},
			{1, 1, 0},
		}
		var firstA, firstB int
		for i, o := range liveSeq {
			da, _, err := govA.ObserveStep(withClock(o, fastStep, fastTarget))
			if err != nil {
				t.Fatalf("fast split step %d: %v", i+1, err)
			}
			db, _, err := govB.ObserveStep(withClock(o, slowStep, slowTarget))
			if err != nil {
				t.Fatalf("slow split step %d: %v", i+1, err)
			}
			if i == 0 {
				firstA, firstB = da, db
			}
		}
		// The split must have been live through the fenced adjudication: the
		// fast rank sees a 30x analytic speedup while the slow rank sees 0.1x,
		// so the first depths genuinely differ and this subtest is not vacuous.
		if firstA == firstB {
			t.Fatalf("analytic speedups did not split the adjudication: both first depths %d", firstA)
		}
		assertModelParity("under live clock split", govA, govB, ruleA, ruleB)
	})

	t.Run("reset between sequences keeps the split choosing alike", func(t *testing.T) {
		govA, ruleA := newPair()
		govB, ruleB := newPair()
		for _, o := range seq {
			if _, _, err := govA.ObserveStep(withClock(o, fastStep, fastTarget)); err != nil {
				t.Fatalf("fast split warm-up: %v", err)
			}
			if _, _, err := govB.ObserveStep(withClock(o, slowStep, slowTarget)); err != nil {
				t.Fatalf("slow split warm-up: %v", err)
			}
		}
		govA.Reset()
		govB.Reset()
		if govA.CurrentDepth() != cfg.ColdStartDepth || govB.CurrentDepth() != cfg.ColdStartDepth {
			t.Fatalf("after Reset depths = %d/%d, want cold start %d",
				govA.CurrentDepth(), govB.CurrentDepth(), cfg.ColdStartDepth)
		}
		if govA.CostRule() != ruleA || govB.CostRule() != ruleB {
			t.Fatal("Reset detached the attached cost rule")
		}
		// Second tour with the clock split swapped: parity must not depend on
		// which governor owns which clock. The rules survive Reset, so their
		// accumulated modeled state continues identically on both sides.
		seq2 := []splitObs{
			{2, 2, 1.45}, {2, 2, 1.45}, {2, 2, 1.45},
			{2, 0, 0.80},
			{0, 0, 1.0}, {0, 0, 1.0},
		}
		wantDepths2 := []int{2, 2, 3, 0, 0, 0}
		for i, o := range seq2 {
			da, _, err := govA.ObserveStep(withClock(o, slowStep, slowTarget))
			if err != nil {
				t.Fatalf("swapped split step %d: %v", i+1, err)
			}
			db, _, err := govB.ObserveStep(withClock(o, fastStep, fastTarget))
			if err != nil {
				t.Fatalf("swapped split step %d: %v", i+1, err)
			}
			if da != db || da != wantDepths2[i] {
				t.Fatalf("step %d: depths %d/%d, want %d", i+1, da, db, wantDepths2[i])
			}
		}
		assertModelParity("after reset and swapped split", govA, govB, ruleA, ruleB)
	})
}

// fak-test:runtime fast est=1ms
// TestQwen38MTP_Adaptive_FailClosedBoundaries witnesses two shipped
// fail-closed behaviors with no direct test yet: detaching the cost rule
// reverts modeled pricing to refusal while the governor's depth adjudication
// keeps working, and DraftDepthRule.Keep refuses rows outside the priced
// table (k < 1 or k >= Rows) even at a confidence the in-table rows accept.
func TestQwen38MTP_Adaptive_FailClosedBoundaries(t *testing.T) {
	cfg := DefaultQwen38AdaptiveConfig()
	c, err := NewDraftDepthCosts(draftDepthFixture(), 0.6132, 16)
	if err != nil {
		t.Fatalf("NewDraftDepthCosts: %v", err)
	}
	rule, err := NewDraftDepthRule(c, 15, nil, 2.0)
	if err != nil {
		t.Fatalf("NewDraftDepthRule: %v", err)
	}

	t.Run("detach reverts pricing fail-closed", func(t *testing.T) {
		gov, err := NewQwen38MTPAdaptiveDepthGovernor(cfg)
		if err != nil {
			t.Fatalf("failed to create governor: %v", err)
		}
		gov.SetCostRule(rule)
		if gov.CostRule() != rule {
			t.Fatal("CostRule did not return the attached rule")
		}
		// Priced decisions while attached: row 1 always kept, high confidence
		// accepts row 2 and the row+level price.
		if !gov.ModeledKeep(1, 0.0) || !gov.ModeledKeep(2, 0.99) || !gov.ModeledMore(1, 0.67) {
			t.Fatalf("attached pricing refused: keep(1/.0)=%v keep(2/.99)=%v more(1/.67)=%v",
				gov.ModeledKeep(1, 0.0), gov.ModeledKeep(2, 0.99), gov.ModeledMore(1, 0.67))
		}
		gov.SetCostRule(nil)
		if gov.CostRule() != nil {
			t.Fatal("CostRule = non-nil after SetCostRule(nil)")
		}
		if k1, k2, m1 := gov.ModeledKeep(1, 0.0), gov.ModeledKeep(2, 0.99), gov.ModeledMore(1, 0.67); k1 || k2 || m1 {
			t.Fatalf("detached pricing answered: keep(1/.0)=%v keep(2/.99)=%v more(1/.67)=%v", k1, k2, m1)
		}
		// Detach degrades pricing, not the governor: an explicit-speedup step
		// still adjudicates exactly as it does with the rule attached.
		depth, rec, err := gov.ObserveStep(Qwen38AdaptiveStepObservation{
			ProposedTokens: 3,
			AcceptedTokens: 2,
			Speedup:        1.10,
		})
		if err != nil {
			t.Fatalf("ObserveStep after detach: %v", err)
		}
		if depth != 2 || rec.Action != "hold" {
			t.Fatalf("post-detach step depth/action = %d/%q, want 2/hold", depth, rec.Action)
		}
	})

	t.Run("keep refuses out-of-table rows", func(t *testing.T) {
		// run = 0.99 accepts both row 1 (always) and row 2 in-table, so the
		// refusals below are the table fence, not low confidence.
		const run = 0.99
		if !rule.Keep(1, run) || !rule.Keep(2, run) {
			t.Fatalf("grounding lost: Keep(1, %v)=%v Keep(2, %v)=%v, want in-table acceptance",
				run, rule.Keep(1, run), run, rule.Keep(2, run))
		}
		if rule.Keep(0, run) {
			t.Fatalf("Keep(0, %v) = true, want k<1 refused fail-closed", run)
		}
		if rule.Keep(c.Rows, run) {
			t.Fatalf("Keep(%d, %v) = true, want k>=Rows refused fail-closed", c.Rows, run)
		}
		// The last in-table row (Rows-1 = 15) stays priced, not fenced: its
		// decision matches the in-table price exactly on both sides of the
		// threshold.
		for _, x := range []float64{0.1, 0.99} {
			want := math.Pow(x, 2.0) >= rule.Rate()*c.Row(15)
			if got := rule.Keep(15, x); got != want {
				t.Fatalf("Keep(15, %v) = %v, want %v (in-table pricing)", x, got, want)
			}
		}
		// Mirror through the governor while the rule is attached.
		gov, err := NewQwen38MTPAdaptiveDepthGovernor(cfg)
		if err != nil {
			t.Fatalf("failed to create governor: %v", err)
		}
		gov.SetCostRule(rule)
		if gov.ModeledKeep(0, run) || gov.ModeledKeep(c.Rows, run) {
			t.Fatalf("ModeledKeep answered out-of-table: keep(0)=%v keep(%d)=%v",
				gov.ModeledKeep(0, run), c.Rows, gov.ModeledKeep(c.Rows, run))
		}
	})
}
