package compute

import (
	"math"
	"strconv"
	"testing"
)

// fak-test:runtime fast est=1ms
func TestGPUBudgetNumericOverflowUsesInvalidPolicy(t *testing.T) {
	const mib = int64(1) << 20
	const largestMiB = math.MaxInt64 / mib
	for _, tt := range []struct {
		name string
		mb   int64
		want int64
	}{
		{"largest_representable", largestMiB, largestMiB * mib},
		{"first_overflow", largestMiB + 1, 0},
		{"would_wrap_zero", 1 << 44, 0},
		{"would_wrap_small_positive", (1 << 44) + 1, 0},
		{"maximum_parsed_integer", math.MaxInt64, 0},
	} {
		t.Run(tt.name, func(t *testing.T) {
			raw := strconv.FormatInt(tt.mb, 10)
			// Both backend initializers use this resolver. Capacity must not
			// turn an invalid numeric setting into an automatic budget.
			for _, known := range []bool{false, true} {
				got := resolveGPUBudgetBytes(raw, 8<<30, known)
				if got != tt.want {
					t.Fatalf("numeric budget %s MiB, known=%v: got %d bytes, want %d", raw, known, got, tt.want)
				}
			}
		})
	}
}
