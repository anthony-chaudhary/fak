package ggufload

import (
	"math"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestCeilDivInt64AvoidsOverflowAndPreservesLegacyInputs(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name string
		n, d int64
		want int64
	}{
		{"maximum_over_maximum", math.MaxInt64, math.MaxInt64, 1},
		{"maximum_over_two", math.MaxInt64, 2, math.MaxInt64/2 + 1},
		{"near_maximum_over_maximum", math.MaxInt64 - 1, math.MaxInt64, 1},
		{"near_maximum_exact", math.MaxInt64 - 1, 2, (math.MaxInt64 - 1) / 2},
		{"maximum_over_one", math.MaxInt64, 1, math.MaxInt64},
		{"rounded", 7, 3, 3},
		{"exact", 6, 3, 2},
		{"zero_numerator", 0, 3, 0},
		{"legacy_negative_fraction", -1, 3, 0},
		{"legacy_negative_exact", -6, 3, -1},
		{"minimum_over_one", math.MinInt64, 1, math.MinInt64},
		{"zero_denominator", 5, 0, 0},
		{"negative_denominator", 5, -2, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := ceilDivInt64(tc.n, tc.d); got != tc.want {
				t.Fatalf("ceilDivInt64(%d, %d) = %d, want %d", tc.n, tc.d, got, tc.want)
			}
		})
	}
}
