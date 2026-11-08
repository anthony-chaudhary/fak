package model

import (
	"errors"
	"math"
	"testing"
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
