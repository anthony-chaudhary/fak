package compute

import (
	"math"
	"strings"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestPrefillBatchNumericControls(t *testing.T) {
	defaultScale := float32(1 / math.Sqrt(2))
	for _, tc := range []struct {
		name      string
		theta     float64
		scale     float32
		reject    string
		wantTheta float64
		wantScale float32
	}{
		{"theta NaN", math.NaN(), 1, "RopeTheta", 0, 0},
		{"theta +Inf", math.Inf(1), 1, "RopeTheta", 0, 0},
		{"theta -Inf", math.Inf(-1), 1, "RopeTheta", 0, 0},
		{"scale NaN", 10000, float32(math.NaN()), "Scale", 0, 0},
		{"scale +Inf", 10000, float32(math.Inf(1)), "Scale", 0, 0},
		{"scale -Inf", 10000, float32(math.Inf(-1)), "Scale", 0, 0},
		{"zero defaults", 0, 0, "", 10000, defaultScale},
		{"finite negative defaults", -1, -1, "", 10000, defaultScale},
		{"finite positive controls", 500000, 0.25, "", 500000, 0.25},
	} {
		t.Run(tc.name, func(t *testing.T) {
			x := NewF32(Default(), []int{1, 2}, []float32{1, 2})
			w := NewF32(Default(), []int{2, 2}, []float32{1, 0, 0, 1})
			args := PrefillBatchArgs{
				X: x, Wq: w, Wk: w, Wv: w,
				NumHeads: 1, NumKVHeads: 1, HeadDim: 2,
				RopeTheta: tc.theta, Scale: tc.scale,
			}
			p, d, err := validatePrefillBatchArgs(&args)
			if tc.reject != "" {
				if err == nil || !strings.Contains(err.Error(), tc.reject) {
					t.Fatalf("nonfinite %s admitted or wrong error: %v", tc.reject, err)
				}
				return
			}
			if err != nil || p != 1 || d != 2 {
				t.Fatalf("finite controls rejected: P=%d D=%d error=%v", p, d, err)
			}
			if args.RopeTheta != tc.wantTheta || args.Scale != tc.wantScale {
				t.Fatalf("controls changed: theta=%g scale=%g, want %g/%g", args.RopeTheta, args.Scale, tc.wantTheta, tc.wantScale)
			}
		})
	}
}
