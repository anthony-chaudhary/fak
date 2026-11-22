package perfledger

import (
	"errors"
	"math"
	"slices"
	"testing"
)

func TestLogspaceIntSteps(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name  string
		lo    float64
		hi    float64
		steps int
		want  []int
	}{
		// Upstream ConcurrencyRamp default grid: [1, 1000] in 8 steps.
		{"upstream_default_ramp", 1, 1000, 8, []int{1, 3, 7, 19, 52, 139, 373, 1000}},
		{"two_steps_are_endpoints", 4, 64, 2, []int{4, 64}},
		{"powers_of_two", 1, 16, 5, []int{1, 2, 4, 8, 16}},
		{"rounding_duplicates_collapse", 1, 2, 5, []int{1, 2}},
		{"more_steps_than_integers", 1, 3, 10, []int{1, 2, 3}},
		{"fractional_lo_floored_at_one", 0.3, 10, 3, []int{1, 2, 10}},
		{"half_rounds_to_even_at_lo", 2.5, 100, 2, []int{2, 100}},
		{"half_rounds_to_even_at_hi", 1, 4.5, 2, []int{1, 4}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got, err := LogspaceIntSteps(tc.lo, tc.hi, tc.steps)
			if err != nil {
				t.Fatalf("LogspaceIntSteps(%v, %v, %d) error: %v", tc.lo, tc.hi, tc.steps, err)
			}
			if !slices.Equal(got, tc.want) {
				t.Fatalf("LogspaceIntSteps(%v, %v, %d) = %v, want %v", tc.lo, tc.hi, tc.steps, got, tc.want)
			}
		})
	}
}

func TestLogspaceIntStepsInvariants(t *testing.T) {
	t.Parallel()
	for lo := 1; lo <= 8; lo++ {
		for _, hi := range []int{lo + 1, 50, 1000, 100000} {
			if hi <= lo {
				continue
			}
			for steps := 2; steps <= 12; steps++ {
				got, err := LogspaceIntSteps(float64(lo), float64(hi), steps)
				if err != nil {
					t.Fatalf("(%d, %d, %d) error: %v", lo, hi, steps, err)
				}
				if len(got) > steps || len(got) < 2 {
					t.Fatalf("(%d, %d, %d) len %d outside [2, steps]: %v", lo, hi, steps, len(got), got)
				}
				if got[0] != lo || got[len(got)-1] != hi {
					t.Fatalf("(%d, %d, %d) endpoints %v, want %d..%d", lo, hi, steps, got, lo, hi)
				}
				for i := 1; i < len(got); i++ {
					if got[i] <= got[i-1] {
						t.Fatalf("(%d, %d, %d) not strictly ascending: %v", lo, hi, steps, got)
					}
				}
			}
		}
	}
}

func TestLogspaceIntStepsRefusals(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name  string
		lo    float64
		hi    float64
		steps int
	}{
		{"steps_one", 1, 10, 1},
		{"steps_zero", 1, 10, 0},
		{"steps_negative", 1, 10, -3},
		{"lo_zero", 0, 10, 4},
		{"lo_negative", -1, 10, 4},
		{"lo_equals_hi", 5, 5, 4},
		{"hi_below_lo", 10, 5, 4},
		{"lo_nan", math.NaN(), 10, 4},
		{"hi_nan", 1, math.NaN(), 4},
		{"hi_inf", 1, math.Inf(1), 4},
		{"lo_neg_inf", math.Inf(-1), 10, 4},
		{"hi_oversized", 1, 1e300, 4},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got, err := LogspaceIntSteps(tc.lo, tc.hi, tc.steps)
			if !errors.Is(err, ErrInvalidLogspaceGrid) {
				t.Fatalf("LogspaceIntSteps(%v, %v, %d) = %v, %v; want ErrInvalidLogspaceGrid", tc.lo, tc.hi, tc.steps, got, err)
			}
			if got != nil {
				t.Fatalf("refusal returned non-nil grid %v", got)
			}
		})
	}
}
