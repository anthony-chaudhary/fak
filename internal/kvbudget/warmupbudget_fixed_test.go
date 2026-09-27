package kvbudget

import (
	"math"
	"testing"
)

// legacyTokenDerive is the pre-#13555 byte-probe derive, written out verbatim in
// the test so the per-token path is pinned against the formula it replaced —
// validation order (invalid unit, reserve range, no capacity), reserve
// floor(usable × f), then floor(kept / bytesPerToken) — not against itself.
func legacyTokenDerive(usable, perToken int64, fraction float64) DerivedBudget {
	if perToken <= 0 {
		return DerivedBudget{Reason: ReasonInvalidUnitSize}
	}
	if !(fraction >= 0 && fraction < 1 && !math.IsNaN(fraction)) {
		return DerivedBudget{Reason: ReasonReserveOutOfRange}
	}
	if usable <= 0 {
		return DerivedBudget{Reason: ReasonNoMeasuredCapacity}
	}
	reserved := int64(math.Floor(float64(usable) * fraction))
	if reserved < 0 {
		reserved = 0
	}
	if reserved > usable {
		reserved = usable
	}
	kept := usable - reserved
	if units := kept / perToken; units > 0 {
		return DerivedBudget{TokenBudget: units, ReservedAmount: reserved, KeptAmount: kept}
	}
	return DerivedBudget{Reason: ReasonBelowOneUnit, ReservedAmount: reserved, KeptAmount: kept}
}

// TestWarmupZeroFixedIsLegacyDerive proves FixedBytesPerStream ≤ 0 takes the old
// path byte-for-byte: every field of DerivedBudget (FixedStateAmount included,
// which stays 0) equals the legacy formula, whatever MaxConcurrentStreams says.
func TestWarmupZeroFixedIsLegacyDerive(t *testing.T) {
	cases := []struct {
		usable, perToken int64
		fraction         float64
	}{
		{1000, 10, 0.25},
		{1005, 10, 0},
		{1_000_000, 100, DefaultReserveFraction},
		{10_000_000_000, 13_760, 0.10},
		{10_000_000_000, 264_192, 0.10},
		{9, 10, 0},       // below one unit
		{10, 10, 0.5},    // reserve pushes below one unit
		{0, 10, 0.10},    // no capacity
		{-4096, 10, 0.1}, // negative capacity
		{1000, 0, 0.10},  // invalid unit
		{1000, -8, 0.10}, // invalid unit
		{1000, 10, 1.0},  // reserve out of range
		{1000, 10, -0.01},
		{1000, 10, math.NaN()},
	}
	for _, c := range cases {
		want := legacyTokenDerive(c.usable, c.perToken, c.fraction)
		for _, fixed := range []int64{0, -1} {
			for _, streams := range []int64{0, 8, -3} {
				capy := WarmupCapacity{UsableBytes: c.usable, BytesPerToken: c.perToken,
					FixedBytesPerStream: fixed, MaxConcurrentStreams: streams}
				if got := capy.DeriveTokenBudget(c.fraction); got != want {
					t.Errorf("usable=%d perToken=%d f=%v fixed=%d streams=%d: got %+v, want legacy %+v",
						c.usable, c.perToken, c.fraction, fixed, streams, got, want)
				}
			}
		}
	}
	// A literal anchor, so the legacy helper is not the only witness.
	if got := (WarmupCapacity{UsableBytes: 1000, BytesPerToken: 10}).DeriveTokenBudget(0.25); got !=
		(DerivedBudget{TokenBudget: 75, ReservedAmount: 250, KeptAmount: 750}) {
		t.Errorf("anchor derive = %+v, want {75 \"\" 250 750 0}", got)
	}
}

// TestWarmupFixedStateV4Flash pins the V4 Flash admission arithmetic: 10 GB
// usable at the default 0.10 reserve keeps 9,000,000,000 B; 8 concurrent streams
// each hold 60,893,696 B of context-independent state (fixed KV bound 11,366,912
// + compressor in-flight rows 49,526,784, spec [SW-VERIFIED]) ⇒ 487,149,568 B
// withheld; the rest at the 13,760 B/token slope is
// floor(8,512,850,432 / 13,760) = 618,666 tokens.
func TestWarmupFixedStateV4Flash(t *testing.T) {
	const (
		usable   = int64(10_000_000_000)
		perToken = int64(13_760)
		perFixed = int64(60_893_696)
		streams  = int64(8)
	)
	if perFixed != 11_366_912+49_526_784 {
		t.Fatalf("fixed per stream %d != KV bound + in-flight rows", perFixed)
	}
	reserved := int64(1_000_000_000) // floor(1e10 × 0.10)
	kept := usable - reserved
	fixed := streams * perFixed
	if fixed != 487_149_568 {
		t.Fatalf("hand fixed = %d, want 487149568", fixed)
	}
	budget := (kept - fixed) / perToken
	if budget != 618_666 || kept-fixed != 8_512_850_432 {
		t.Fatalf("hand budget = %d (kept-fixed %d), want 618666 (8512850432)", budget, kept-fixed)
	}
	got := WarmupCapacity{UsableBytes: usable, BytesPerToken: perToken,
		FixedBytesPerStream: perFixed, MaxConcurrentStreams: streams}.DeriveTokenBudget(DefaultReserveFraction)
	want := DerivedBudget{
		TokenBudget:      618_666,
		ReservedAmount:   1_000_000_000,
		KeptAmount:       8_512_850_432,
		FixedStateAmount: 487_149_568,
	}
	if got != want {
		t.Fatalf("DeriveTokenBudget = %+v, want %+v", got, want)
	}
	if !got.Derived() {
		t.Errorf("Derived() = false, want true")
	}
	// The fixed state strictly shrinks the budget relative to the per-token derive
	// of the same probe (floor(9e9 / 13,760) = 654,069).
	plain := WarmupCapacity{UsableBytes: usable, BytesPerToken: perToken}.DeriveTokenBudget(DefaultReserveFraction)
	if plain.TokenBudget != 654_069 || !(got.TokenBudget < plain.TokenBudget) {
		t.Errorf("per-token budget = %d, fixed-state budget = %d; want 654069 and strictly less",
			plain.TokenBudget, got.TokenBudget)
	}
}

// TestWarmupFixedStateScalesWithStreams checks the fold against the spec formula
// floor((kept − N×F) / bytesPerToken) across stream counts, and that admitting
// more concurrent streams never grows the budget.
func TestWarmupFixedStateScalesWithStreams(t *testing.T) {
	const usable, perToken, perFixed = int64(1_000_000), int64(100), int64(10_000)
	prev := int64(math.MaxInt64)
	for _, n := range []int64{1, 2, 7, 50, 89} {
		kept := usable - 100_000 // 0.10 reserve
		want := DerivedBudget{
			TokenBudget:      (kept - n*perFixed) / perToken,
			ReservedAmount:   100_000,
			KeptAmount:       kept - n*perFixed,
			FixedStateAmount: n * perFixed,
		}
		got := WarmupCapacity{UsableBytes: usable, BytesPerToken: perToken,
			FixedBytesPerStream: perFixed, MaxConcurrentStreams: n}.DeriveTokenBudget(0.10)
		if got != want {
			t.Errorf("streams=%d: got %+v, want %+v", n, got, want)
		}
		if got.TokenBudget > prev {
			t.Errorf("streams=%d budget %d grew past %d", n, got.TokenBudget, prev)
		}
		prev = got.TokenBudget
	}
}

// TestWarmupFixedStateUnboundedStreams proves a fixed per-stream charge with no
// stream bound fails closed: the withheld amount would be unbounded.
func TestWarmupFixedStateUnboundedStreams(t *testing.T) {
	for _, streams := range []int64{0, -1} {
		got := WarmupCapacity{UsableBytes: 10_000_000_000, BytesPerToken: 13_760,
			FixedBytesPerStream: 60_893_696, MaxConcurrentStreams: streams}.DeriveTokenBudget(0.10)
		if got != (DerivedBudget{Reason: ReasonUnboundedStreamState}) {
			t.Errorf("streams=%d: got %+v, want fail-closed %q with zero budget", streams, got, ReasonUnboundedStreamState)
		}
		if got.Derived() || got.TokenBudget != 0 {
			t.Errorf("streams=%d derived %d, want 0", streams, got.TokenBudget)
		}
	}
	if ReasonUnboundedStreamState != "unbounded_fixed_stream_state" {
		t.Errorf("ReasonUnboundedStreamState = %q, want unbounded_fixed_stream_state", ReasonUnboundedStreamState)
	}
}

// TestWarmupFixedStateBelowOneUnit proves the fixed state can exhaust the kept
// room: at or past it the derive fails closed with ReasonBelowOneUnit and records
// reserved / kept / fixed; a sliver under one token after the fixed state fails
// the same way.
func TestWarmupFixedStateBelowOneUnit(t *testing.T) {
	// 1000 usable, 0.10 reserve ⇒ reserved 100, kept 900.
	for _, c := range []struct {
		perFixed, streams, fixed int64
	}{
		{450, 2, 900},  // fixed == kept
		{450, 3, 1350}, // fixed > kept
		{900, 1, 900},
	} {
		got := WarmupCapacity{UsableBytes: 1000, BytesPerToken: 10,
			FixedBytesPerStream: c.perFixed, MaxConcurrentStreams: c.streams}.DeriveTokenBudget(0.10)
		want := DerivedBudget{Reason: ReasonBelowOneUnit, ReservedAmount: 100, KeptAmount: 900, FixedStateAmount: c.fixed}
		if got != want {
			t.Errorf("perFixed=%d streams=%d: got %+v, want %+v", c.perFixed, c.streams, got, want)
		}
	}
	// kept − fixed = 5 B < 10 B/token ⇒ zero whole tokens.
	got := WarmupCapacity{UsableBytes: 1000, BytesPerToken: 10,
		FixedBytesPerStream: 895, MaxConcurrentStreams: 1}.DeriveTokenBudget(0.10)
	if got.Derived() || got.TokenBudget != 0 || got.Reason != ReasonBelowOneUnit {
		t.Errorf("sub-token remainder = %+v, want fail-closed %q", got, ReasonBelowOneUnit)
	}
	// Exactly one token left is admitted.
	one := WarmupCapacity{UsableBytes: 1000, BytesPerToken: 10,
		FixedBytesPerStream: 890, MaxConcurrentStreams: 1}.DeriveTokenBudget(0.10)
	if one != (DerivedBudget{TokenBudget: 1, ReservedAmount: 100, KeptAmount: 10, FixedStateAmount: 890}) {
		t.Errorf("one-token remainder = %+v, want {1 \"\" 100 10 890}", one)
	}
	// streams × perFixed overflowing int64 saturates rather than wrapping negative
	// (which would inflate the budget).
	sat := WarmupCapacity{UsableBytes: 10_000_000_000, BytesPerToken: 13_760,
		FixedBytesPerStream: 1 << 62, MaxConcurrentStreams: 4}.DeriveTokenBudget(0.10)
	if sat.Derived() || sat.TokenBudget != 0 || sat.Reason != ReasonBelowOneUnit {
		t.Errorf("overflowing fixed state = %+v, want fail-closed %q", sat, ReasonBelowOneUnit)
	}
	if sat.FixedStateAmount != math.MaxInt64 {
		t.Errorf("overflowing FixedStateAmount = %d, want saturated %d", sat.FixedStateAmount, int64(math.MaxInt64))
	}
}

// TestWarmupFixedStateExistingReasonsWin proves the fixed path validates in the
// legacy order — invalid unit, then reserve range, then no capacity — and that
// each of those wins over the unbounded-streams refusal too.
func TestWarmupFixedStateExistingReasonsWin(t *testing.T) {
	cases := []struct {
		name             string
		usable, perToken int64
		fraction         float64
		want             Reason
	}{
		{"invalid_unit", 1000, 0, 0.10, ReasonInvalidUnitSize},
		{"negative_unit", 1000, -8, 0.10, ReasonInvalidUnitSize},
		{"unit_beats_reserve", 1000, 0, 1.5, ReasonInvalidUnitSize},
		{"unit_beats_capacity", 0, 0, 0.10, ReasonInvalidUnitSize},
		{"reserve_high", 1000, 10, 1.0, ReasonReserveOutOfRange},
		{"reserve_negative", 1000, 10, -0.01, ReasonReserveOutOfRange},
		{"reserve_nan", 1000, 10, math.NaN(), ReasonReserveOutOfRange},
		{"reserve_beats_capacity", 0, 10, 1.0, ReasonReserveOutOfRange},
		{"no_capacity", 0, 10, 0.10, ReasonNoMeasuredCapacity},
		{"negative_capacity", -4096, 10, 0.10, ReasonNoMeasuredCapacity},
	}
	for _, c := range cases {
		for _, streams := range []int64{8, 0} {
			got := WarmupCapacity{UsableBytes: c.usable, BytesPerToken: c.perToken,
				FixedBytesPerStream: 60_893_696, MaxConcurrentStreams: streams}.DeriveTokenBudget(c.fraction)
			if got != (DerivedBudget{Reason: c.want}) {
				t.Errorf("%s streams=%d: got %+v, want fail-closed %q", c.name, streams, got, c.want)
			}
		}
	}
}
