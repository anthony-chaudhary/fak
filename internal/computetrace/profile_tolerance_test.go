package computetrace

import (
	"math"
	"testing"
)

// Exact strict-comparator tolerance matrix, applied to the profile entry point.
func TestActivationProfileTolerancePolicy(t *testing.T) {
	t.Run("invalid tolerances are incomparable", func(t *testing.T) {
		invalid := []struct {
			name string
			tol  ActivationTolerance
		}{
			{"nan_atol", ActivationTolerance{Atol: math.NaN()}},
			{"nan_rtol", ActivationTolerance{Rtol: math.NaN()}},
			{"positive_inf_atol", ActivationTolerance{Atol: math.Inf(1)}},
			{"positive_inf_rtol", ActivationTolerance{Rtol: math.Inf(1)}},
			{"negative_inf_atol", ActivationTolerance{Atol: math.Inf(-1)}},
			{"negative_inf_rtol", ActivationTolerance{Rtol: math.Inf(-1)}},
			{"negative_atol", ActivationTolerance{Atol: -1}},
			{"negative_rtol", ActivationTolerance{Rtol: -1}},
			{"negative_subnormal_atol", ActivationTolerance{Atol: -math.SmallestNonzeroFloat64}},
			{"negative_subnormal_rtol", ActivationTolerance{Rtol: -math.SmallestNonzeroFloat64}},
			{"negative_atol_compensated", ActivationTolerance{Atol: -1, Rtol: 100}},
			{"negative_rtol_compensated", ActivationTolerance{Atol: 100, Rtol: -1}},
		}
		pairs := []struct {
			name     string
			ref, got float32
		}{
			{"equal", 1, 1},
			{"unequal", 1, 2},
			{"zero_actual", 1, 0},
		}
		for _, tc := range invalid {
			t.Run(tc.name, func(t *testing.T) {
				for _, pair := range pairs {
					t.Run(pair.name, func(t *testing.T) {
						ref := collect(indexedSample(0, 0, "q_proj", []float32{pair.ref}))
						got := collect(indexedSample(0, 0, "q_proj", []float32{pair.got}))
						res := ActivationErrorProfile(ref, got, tc.tol)
						if res.Verdict != TraceIncomparable || res.ComparedSamples != 0 || res.Reason == "" {
							t.Fatalf("invalid tolerance %+v: result=%+v, want INCOMPARABLE with reason and zero compared samples", tc.tol, res)
						}
					})
				}
			})
		}
	})

	t.Run("valid tolerances preserve comparison", func(t *testing.T) {
		negativeZero := math.Copysign(0, -1)
		cases := []struct {
			name     string
			tol      ActivationTolerance
			ref, got float32
			want     Verdict
		}{
			{"zero_equal", ActivationTolerance{}, 1, 1, TraceEqual},
			{"zero_unequal", ActivationTolerance{}, 1, 2, TraceDivergent},
			{"negative_zero_equal", ActivationTolerance{Atol: negativeZero, Rtol: negativeZero}, 1, 1, TraceEqual},
			{"negative_zero_unequal", ActivationTolerance{Atol: negativeZero, Rtol: negativeZero}, 1, 2, TraceDivergent},
			{"atol_boundary", ActivationTolerance{Atol: 1}, 1, 2, TraceEqual},
			{"atol_below_boundary", ActivationTolerance{Atol: math.Nextafter(1, 0)}, 1, 2, TraceDivergent},
			{"rtol_actual_boundary", ActivationTolerance{Rtol: 0.25}, 3, 4, TraceEqual},
			{"rtol_reverse_pair", ActivationTolerance{Rtol: 0.25}, 4, 3, TraceDivergent},
			{"rtol_negative_actual", ActivationTolerance{Rtol: 0.25}, -3, -4, TraceEqual},
			{"combined_boundary", ActivationTolerance{Atol: 0.5, Rtol: 0.125}, 3, 4, TraceEqual},
			{"positive_subnormal_atol", ActivationTolerance{Atol: math.SmallestNonzeroFloat64}, 0, math.SmallestNonzeroFloat32, TraceDivergent},
			{"positive_subnormal_rtol", ActivationTolerance{Rtol: math.SmallestNonzeroFloat64}, 1, 2, TraceDivergent},
			{"maximum_finite_atol", ActivationTolerance{Atol: math.MaxFloat64}, -math.MaxFloat32, math.MaxFloat32, TraceEqual},
			{"maximum_finite_rtol_zero_actual", ActivationTolerance{Rtol: math.MaxFloat64}, 1, 0, TraceDivergent},
			{"finite_limit_overflow", ActivationTolerance{Rtol: math.MaxFloat64}, 1, 2, TraceEqual},
			{"nan_reference", ActivationTolerance{Atol: math.MaxFloat64}, float32(math.NaN()), 1, TraceDivergent},
			{"nan_actual", ActivationTolerance{Atol: math.MaxFloat64}, 1, float32(math.NaN()), TraceDivergent},
			{"positive_inf_reference", ActivationTolerance{Atol: math.MaxFloat64}, float32(math.Inf(1)), 1, TraceDivergent},
			{"positive_inf_actual", ActivationTolerance{Atol: math.MaxFloat64}, 1, float32(math.Inf(1)), TraceDivergent},
			{"negative_inf_reference", ActivationTolerance{Atol: math.MaxFloat64}, float32(math.Inf(-1)), 1, TraceDivergent},
			{"negative_inf_actual", ActivationTolerance{Atol: math.MaxFloat64}, 1, float32(math.Inf(-1)), TraceDivergent},
		}
		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				ref := collect(indexedSample(0, 0, "q_proj", []float32{tc.ref}))
				got := collect(indexedSample(0, 0, "q_proj", []float32{tc.got}))
				res := ActivationErrorProfile(ref, got, tc.tol)
				if res.Verdict != tc.want || res.ComparedSamples != 1 || res.Reason != "" {
					t.Fatalf("valid tolerance %+v: result=%+v, want %s with one compared sample and no refusal reason", tc.tol, res, tc.want)
				}
			})
		}
	})
}
