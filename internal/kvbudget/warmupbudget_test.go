package kvbudget

import "testing"

// TestDeriveBudgetFromMeasuredCapacity proves a measured usable byte figure maps
// to the expected token budget: floor((usable - reserve) / bytesPerToken). With
// 1000 usable bytes, a 0.25 reserve withholds floor(1000*0.25)=250, keeping 750,
// and at 10 bytes/token that is a 75-token budget.
func TestDeriveBudgetFromMeasuredCapacity(t *testing.T) {
	c := WarmupCapacity{UsableBytes: 1000, BytesPerToken: 10}
	d := c.DeriveTokenBudget(0.25)
	if !d.Derived() {
		t.Fatalf("Derived() = false, reason %q; want a derived budget", d.Reason)
	}
	if d.ReservedAmount != 250 {
		t.Errorf("ReservedAmount = %d, want 250", d.ReservedAmount)
	}
	if d.KeptAmount != 750 {
		t.Errorf("KeptAmount = %d, want 750", d.KeptAmount)
	}
	if d.TokenBudget != 75 {
		t.Errorf("TokenBudget = %d, want 75 (floor(750/10))", d.TokenBudget)
	}
}

// TestDeriveBudgetFloorDivision proves the budget floor-divides: 1005 usable at a
// zero reserve and 10 bytes/token yields floor(1005/10) = 100, never 100.5.
func TestDeriveBudgetFloorDivision(t *testing.T) {
	d := WarmupCapacity{UsableBytes: 1005, BytesPerToken: 10}.DeriveTokenBudget(0)
	if d.TokenBudget != 100 {
		t.Errorf("TokenBudget = %d, want 100 (floor(1005/10))", d.TokenBudget)
	}
	if d.ReservedAmount != 0 || d.KeptAmount != 1005 {
		t.Errorf("reserve at fraction 0: reserved=%d kept=%d, want 0/1005", d.ReservedAmount, d.KeptAmount)
	}
}

// TestDeriveBudgetMonotoneInMeasurement proves a larger measurement yields a
// proportionally larger budget: doubling usable bytes (same unit, same reserve)
// doubles the token budget.
func TestDeriveBudgetMonotoneInMeasurement(t *testing.T) {
	small := WarmupCapacity{UsableBytes: 2000, BytesPerToken: 10}.DeriveTokenBudget(0)
	large := WarmupCapacity{UsableBytes: 4000, BytesPerToken: 10}.DeriveTokenBudget(0)
	if small.TokenBudget != 200 {
		t.Fatalf("small TokenBudget = %d, want 200", small.TokenBudget)
	}
	if large.TokenBudget != 400 {
		t.Fatalf("large TokenBudget = %d, want 400", large.TokenBudget)
	}
	if !(large.TokenBudget > small.TokenBudget) {
		t.Errorf("budget not monotone: large %d !> small %d", large.TokenBudget, small.TokenBudget)
	}
	if large.TokenBudget != 2*small.TokenBudget {
		t.Errorf("budget not proportional: large %d != 2 * small %d", large.TokenBudget, small.TokenBudget)
	}
}

// TestDeriveBudgetReserveShrinksBudget proves a bigger reserve fraction shrinks
// the derived budget for the same measurement: 0 < 0.25 < 0.75 reserve gives a
// strictly decreasing budget.
func TestDeriveBudgetReserveShrinksBudget(t *testing.T) {
	c := WarmupCapacity{UsableBytes: 1000, BytesPerToken: 10}
	none := c.DeriveTokenBudget(0)      // floor(1000/10)      = 100
	quarter := c.DeriveTokenBudget(0.25) // floor(750/10)       = 75
	threeq := c.DeriveTokenBudget(0.75)  // floor(250/10)       = 25
	if none.TokenBudget != 100 || quarter.TokenBudget != 75 || threeq.TokenBudget != 25 {
		t.Fatalf("budgets = %d/%d/%d, want 100/75/25", none.TokenBudget, quarter.TokenBudget, threeq.TokenBudget)
	}
	if !(none.TokenBudget > quarter.TokenBudget && quarter.TokenBudget > threeq.TokenBudget) {
		t.Errorf("reserve did not shrink budget monotonically: %d %d %d", none.TokenBudget, quarter.TokenBudget, threeq.TokenBudget)
	}
}

// TestDeriveBudgetZeroMeasurementFailsClosed proves a zero or negative
// measurement fails closed to a zero, typed-reason budget — never a huge or a
// negative budget.
func TestDeriveBudgetZeroMeasurementFailsClosed(t *testing.T) {
	for _, usable := range []int64{0, -1, -4096} {
		d := WarmupCapacity{UsableBytes: usable, BytesPerToken: 10}.DeriveTokenBudget(0.10)
		if d.Derived() {
			t.Errorf("usable=%d derived a budget, want fail-closed", usable)
		}
		if d.TokenBudget != 0 {
			t.Errorf("usable=%d TokenBudget = %d, want 0", usable, d.TokenBudget)
		}
		if d.Reason != ReasonNoMeasuredCapacity {
			t.Errorf("usable=%d Reason = %q, want %q", usable, d.Reason, ReasonNoMeasuredCapacity)
		}
	}
}

// TestDeriveBudgetInvalidUnitFailsClosed proves a non-positive per-token size
// fails closed rather than dividing by zero or a negative unit.
func TestDeriveBudgetInvalidUnitFailsClosed(t *testing.T) {
	for _, per := range []int64{0, -8} {
		d := WarmupCapacity{UsableBytes: 1000, BytesPerToken: per}.DeriveTokenBudget(0.10)
		if d.Derived() || d.TokenBudget != 0 {
			t.Errorf("bytesPerToken=%d derived %d, want fail-closed 0", per, d.TokenBudget)
		}
		if d.Reason != ReasonInvalidUnitSize {
			t.Errorf("bytesPerToken=%d Reason = %q, want %q", per, d.Reason, ReasonInvalidUnitSize)
		}
	}
}

// TestDeriveBudgetReserveOutOfRangeFailsClosed proves a reserve fraction outside
// [0, 1) fails closed — a fraction of 1 (or more) would reserve everything, and
// a negative fraction would inflate the measurement.
func TestDeriveBudgetReserveOutOfRangeFailsClosed(t *testing.T) {
	for _, f := range []float64{-0.01, 1.0, 1.5} {
		d := WarmupCapacity{UsableBytes: 1000, BytesPerToken: 10}.DeriveTokenBudget(f)
		if d.Derived() || d.TokenBudget != 0 {
			t.Errorf("fraction=%v derived %d, want fail-closed 0", f, d.TokenBudget)
		}
		if d.Reason != ReasonReserveOutOfRange {
			t.Errorf("fraction=%v Reason = %q, want %q", f, d.Reason, ReasonReserveOutOfRange)
		}
	}
}

// TestDeriveBudgetBelowOneUnitFailsClosed proves a positive measurement that,
// after the reserve, no longer holds even one whole token rounds down to a zero
// admittable budget with the typed below-one-unit reason (never negative).
func TestDeriveBudgetBelowOneUnit(t *testing.T) {
	// 9 usable bytes < 10 bytes/token: below one token even at zero reserve.
	d := WarmupCapacity{UsableBytes: 9, BytesPerToken: 10}.DeriveTokenBudget(0)
	if d.Derived() || d.TokenBudget != 0 {
		t.Fatalf("below-one-unit derived %d, want 0", d.TokenBudget)
	}
	if d.Reason != ReasonBelowOneUnit {
		t.Errorf("Reason = %q, want %q", d.Reason, ReasonBelowOneUnit)
	}
	// A reserve can also push an otherwise-fitting measurement below one unit:
	// 10 usable, 0.5 reserve keeps 5 bytes < 10 bytes/token.
	d2 := WarmupCapacity{UsableBytes: 10, BytesPerToken: 10}.DeriveTokenBudget(0.5)
	if d2.Derived() || d2.TokenBudget != 0 || d2.Reason != ReasonBelowOneUnit {
		t.Errorf("reserve-below-one-unit = {budget %d reason %q}, want {0 %q}", d2.TokenBudget, d2.Reason, ReasonBelowOneUnit)
	}
}

// TestDeriveBudgetDefaultReserve proves the borrowed default reserve (0.10, the
// TGI 0.90 wiggle-room mirror) withholds a tenth: 1_000_000 usable bytes at 100
// bytes/token keeps 900_000 bytes -> a 9000-token budget.
func TestDeriveBudgetDefaultReserve(t *testing.T) {
	d := WarmupCapacity{UsableBytes: 1_000_000, BytesPerToken: 100}.DeriveTokenBudget(DefaultReserveFraction)
	if d.ReservedAmount != 100_000 || d.KeptAmount != 900_000 {
		t.Fatalf("default reserve: reserved=%d kept=%d, want 100000/900000", d.ReservedAmount, d.KeptAmount)
	}
	if d.TokenBudget != 9000 {
		t.Errorf("TokenBudget = %d, want 9000 (floor(900000/100))", d.TokenBudget)
	}
}

// TestDeriveBlockCapacity proves the block-measured probe multiplies the fitted
// blocks (after the reserve) by the block's token span, and fails closed the same
// way as the byte probe.
func TestDeriveBlockCapacity(t *testing.T) {
	// 100 fitted blocks, 0.10 reserve -> floor(100*0.10)=10 reserved, 90 kept;
	// each block holds 16 tokens -> 90*16 = 1440-token budget.
	d := WarmupBlockCapacity{FittedBlocks: 100, BlockTokens: 16}.DeriveTokenBudget(0.10)
	if !d.Derived() {
		t.Fatalf("Derived() = false, reason %q", d.Reason)
	}
	if d.ReservedAmount != 10 || d.KeptAmount != 90 {
		t.Errorf("reserve: reserved=%d kept=%d, want 10/90 blocks", d.ReservedAmount, d.KeptAmount)
	}
	if d.TokenBudget != 1440 {
		t.Errorf("TokenBudget = %d, want 1440 (90 blocks * 16 tokens)", d.TokenBudget)
	}

	// Monotone in fitted blocks.
	more := WarmupBlockCapacity{FittedBlocks: 200, BlockTokens: 16}.DeriveTokenBudget(0.10)
	if !(more.TokenBudget > d.TokenBudget) {
		t.Errorf("block budget not monotone: %d !> %d", more.TokenBudget, d.TokenBudget)
	}

	// Zero fitted blocks fails closed as no measured capacity.
	if z := (WarmupBlockCapacity{FittedBlocks: 0, BlockTokens: 16}).DeriveTokenBudget(0.10); z.Derived() || z.Reason != ReasonNoMeasuredCapacity {
		t.Errorf("zero blocks = {budget %d reason %q}, want fail-closed %q", z.TokenBudget, z.Reason, ReasonNoMeasuredCapacity)
	}

	// One fitted block always yields a positive budget: a valid reserve fraction
	// (< 1) floors to 0 reserved blocks on a single block, so the block count can
	// never be rounded below one by the reserve — one block worth of tokens is
	// always admittable.
	if b := (WarmupBlockCapacity{FittedBlocks: 1, BlockTokens: 16}).DeriveTokenBudget(0.9); !b.Derived() || b.TokenBudget != 16 {
		t.Errorf("one fitted block = {budget %d reason %q}, want a derived 16-token budget", b.TokenBudget, b.Reason)
	}
}
