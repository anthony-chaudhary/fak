// fak-test:runtime fast est=10ms
package perfledger

import (
	"math"
	"testing"
)

// TestPercentileBand verifies the percentile-band metric dict (direct port of
// aiperf metric_dicts.metric_result_from_array's quantile core) returns the
// correct p1..p99 values for a known sample set and handles empty input.
//
// Every expected number is hand-computed from the source formula so a reviewer
// can verify each vector without running numpy: for samples 1..100 (n=100),
// rank r = q/100*(n-1) = q/100*99, and the value is the linear interpolation
// between floor(r) and min(floor(r)+1, n-1).
func TestPercentileBand(t *testing.T) {
	// samples 1..100, deliberately reversed to exercise the internal sort.
	samples := make([]float64, 100)
	for i := range samples {
		samples[i] = float64(100 - i) // 100, 99, ..., 1
	}

	got, err := ComputePercentileBand(samples)
	if err != nil {
		t.Fatalf("ComputePercentileBand(1..100) returned error: %v", err)
	}
	if got.Count != 100 {
		t.Errorf("Count = %d, want 100", got.Count)
	}

	// Hand-computed from the source formula (see comment above). Finite wants
	// are matched with a 1e-9 epsilon, matching the package convention, since
	// some results (e.g. p90 = 90.10000000000001) are not exactly representable.
	want := []struct {
		label string
		got   float64
		want  float64
	}{
		{"p1", got.P1, 1.99},    // r=0.99   -> 1 + 0.99*(2-1)
		{"p5", got.P5, 5.95},    // r=4.95   -> 5 + 0.95*(6-5)
		{"p10", got.P10, 10.9},  // r=9.9    -> 10 + 0.9*(11-10)
		{"p25", got.P25, 25.75}, // r=24.75  -> 25 + 0.75*(26-25)
		{"p50", got.P50, 50.5},  // r=49.5   -> 50 + 0.5*(51-50)
		{"p75", got.P75, 75.25}, // r=74.25  -> 75 + 0.25*(76-75)
		{"p90", got.P90, 90.1},  // r=89.1   -> 90 + 0.1*(91-90)
		{"p95", got.P95, 95.05}, // r=94.05  -> 95 + 0.05*(96-95)
		{"p99", got.P99, 99.01}, // r=98.01  -> 99 + 0.01*(100-99)
	}
	for _, w := range want {
		if math.Abs(w.got-w.want) > 1e-9 {
			t.Errorf("%s = %v, want %v", w.label, w.got, w.want)
		}
	}

	// Empty input must be refused, not panic: the source would index clean[0].
	if _, err := ComputePercentileBand(nil); err == nil {
		t.Error("ComputePercentileBand(nil) = nil error, want error for empty input")
	}
	if _, err := ComputePercentileBand([]float64{}); err == nil {
		t.Error("ComputePercentileBand([]) = nil error, want error for empty input")
	}
}
