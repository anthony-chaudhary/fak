package main

import (
	"math"
	"testing"
)

// fak-test:runtime fast est=10ms

// TestCIWidthConvergence verifies the CI-width convergence stop decision
// against hand-computed width sequences, matching the source expected
// outputs from ai-dynamo/aiperf ci_width.py.
func TestCIWidthConvergence(t *testing.T) {
	cases := []struct {
		name            string
		values          []float64
		threshold       float64
		confidenceLevel float64
		minRuns         int
		wantConverged   bool
	}{
		{
			name:            "converging",
			values:          []float64{100, 100.5, 99.5, 100.2, 99.8},
			threshold:       0.10,
			confidenceLevel: 0.95,
			minRuns:         3,
			wantConverged:   true,
		},
		{
			name:            "not converging",
			values:          []float64{100, 110, 90, 105, 95},
			threshold:       0.10,
			confidenceLevel: 0.95,
			minRuns:         3,
			wantConverged:   false,
		},
		{
			name:            "too few runs",
			values:          []float64{100, 100.5},
			threshold:       0.10,
			confidenceLevel: 0.95,
			minRuns:         3,
			wantConverged:   false,
		},
		{
			name:            "zero mean",
			values:          []float64{0, 0, 0},
			threshold:       0.10,
			confidenceLevel: 0.95,
			minRuns:         3,
			wantConverged:   false,
		},
		{
			name:            "threshold sensitivity",
			values:          []float64{100, 100.5, 99.5, 100.2, 99.8},
			threshold:       0.001,
			confidenceLevel: 0.95,
			minRuns:         3,
			wantConverged:   false,
		},
		{
			name:            "minRuns sensitivity",
			values:          []float64{100, 100.5, 99.5},
			threshold:       0.10,
			confidenceLevel: 0.95,
			minRuns:         5,
			wantConverged:   false,
		},
		{
			name:            "high confidence widens the interval",
			values:          []float64{100, 102, 98, 101, 99},
			threshold:       0.10,
			confidenceLevel: 0.999,
			minRuns:         3,
			wantConverged:   false,
		},
		{
			name:            "same data converges at 0.95",
			values:          []float64{100, 102, 98, 101, 99},
			threshold:       0.10,
			confidenceLevel: 0.95,
			minRuns:         3,
			wantConverged:   true,
		},
		{
			name:            "empty values",
			values:          []float64{},
			threshold:       0.10,
			confidenceLevel: 0.95,
			minRuns:         3,
			wantConverged:   false,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c := NewCIWidthConvergence(tc.threshold, tc.confidenceLevel, tc.minRuns)
			if got := c.IsConverged(tc.values); got != tc.wantConverged {
				t.Fatalf("IsConverged(%v) = %v, want %v", tc.values, got, tc.wantConverged)
			}
		})
	}

	// Guard the math: bound the CI width ratio for the two 5-run cases
	// without asserting exact float equality.
	converging := NewCIWidthConvergence(0.10, 0.95, 3)
	if ratio := ciWidthRatioForTest(converging, []float64{100, 100.5, 99.5, 100.2, 99.8}); ratio >= 0.05 {
		t.Fatalf("converging case CI width ratio = %v, want < 0.05", ratio)
	}
	notConverging := NewCIWidthConvergence(0.10, 0.95, 3)
	if ratio := ciWidthRatioForTest(notConverging, []float64{100, 110, 90, 105, 95}); ratio <= 0.15 {
		t.Fatalf("not-converging case CI width ratio = %v, want > 0.15", ratio)
	}
}

// ciWidthRatioForTest recomputes the CI width ratio for the ratio-guard
// assertions; it mirrors the IsConverged math so the test can bound the
// intermediate value without asserting exact float equality.
func ciWidthRatioForTest(c *CIWidthConvergence, values []float64) float64 {
	mean := 0.0
	for _, v := range values {
		mean += v
	}
	mean /= float64(len(values))
	n := len(values)
	sumSquares := 0.0
	for _, v := range values {
		delta := v - mean
		sumSquares += delta * delta
	}
	std := math.Sqrt(sumSquares / float64(n-1))
	se := std / math.Sqrt(float64(n))
	tCrit := studentTQuantile((1+c.ConfidenceLevel)/2, float64(n-1))
	return (2 * tCrit * se) / math.Abs(mean)
}
