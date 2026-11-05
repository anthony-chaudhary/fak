package kvbudget

import (
	"math"
	"testing"
)

// fak-test:runtime fast est=10ms lane=default
// Estimate only: scalar arithmetic, with no allocation proportional to capacity.
func TestWarmupTokenBudgetFitsAdmissionInt(t *testing.T) {
	t.Parallel()
	limit := int64(math.MaxInt)
	lastWholeBlock := limit / 16
	for _, tc := range []struct {
		name     string
		blocks   bool
		usable   int64
		perUnit  int64
		fixed    int64
		fraction float64
		want     DerivedBudget
	}{
		{"ordinary blocks", true, 100, 16, 0, 0.1, DerivedBudget{TokenBudget: 1440, ReservedAmount: 10, KeptAmount: 90}},
		{"last whole block", true, lastWholeBlock, 16, 0, 0, DerivedBudget{TokenBudget: lastWholeBlock * 16, KeptAmount: lastWholeBlock}},
		{"first over int limit", true, lastWholeBlock + 1, 16, 0, 0, DerivedBudget{TokenBudget: limit, KeptAmount: lastWholeBlock + 1}},
		{"product wraps positive", true, math.MaxInt64, math.MaxInt64, 0, 0, DerivedBudget{TokenBudget: limit, KeptAmount: math.MaxInt64}},
		{"reserve before saturation", true, 100, math.MaxInt64, 0, 0.25, DerivedBudget{TokenBudget: limit, ReservedAmount: 25, KeptAmount: 75}},
		{"byte capacity", false, math.MaxInt64, 1, 0, 0, DerivedBudget{TokenBudget: limit, KeptAmount: math.MaxInt64}},
		{"fixed state capacity", false, math.MaxInt64, 1, 1, 0, DerivedBudget{TokenBudget: min(limit, int64(math.MaxInt64-1)), KeptAmount: math.MaxInt64 - 1, FixedStateAmount: 1}},
		{"invalid unit retains precedence", true, math.MaxInt64, 0, 0, math.NaN(), DerivedBudget{Reason: ReasonInvalidUnitSize}},
		{"fixed state consumes capacity", false, 1, 1, 1, 0, DerivedBudget{Reason: ReasonBelowOneUnit, KeptAmount: 1, FixedStateAmount: 1}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var got DerivedBudget
			if tc.blocks {
				got = (WarmupBlockCapacity{FittedBlocks: tc.usable, BlockTokens: tc.perUnit}).DeriveTokenBudget(tc.fraction)
			} else {
				got = (WarmupCapacity{UsableBytes: tc.usable, BytesPerToken: tc.perUnit,
					FixedBytesPerStream: tc.fixed, MaxConcurrentStreams: 1}).DeriveTokenBudget(tc.fraction)
			}
			if got != tc.want {
				t.Fatalf("derived=%+v, want %+v", got, tc.want)
			}
			if tc.want.Derived() && (!got.Derived() || int64(int(got.TokenBudget)) != got.TokenBudget) {
				t.Fatalf("successful budget must survive the gateway's int conversion: %+v", got)
			}
		})
	}
}
