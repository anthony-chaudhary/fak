// fak-test:runtime fast est=10ms
package perfledger

// Adversarial spec tests for the adjusted-percentile / derived-latency port
// (direct port of NVIDIA aiperf derived_latency.py). Every expected number is
// hand-computed in comments so a reviewer can verify each vector without
// running numpy.

import (
	"math"
	"testing"
)

// adjSameFloat matches a got value against a want: NaN wants require NaN,
// +Inf wants require +Inf, finite wants use a 1e-9 epsilon.
func adjSameFloat(t *testing.T, label string, got, want float64) {
	t.Helper()
	switch {
	case math.IsNaN(want):
		if !math.IsNaN(got) {
			t.Errorf("%s = %v, want NaN", label, got)
		}
	case math.IsInf(want, 1):
		if !math.IsInf(got, 1) {
			t.Errorf("%s = %v, want +Inf", label, got)
		}
	case math.Abs(got-want) > 1e-9:
		t.Errorf("%s = %v, want %v", label, got, want)
	}
}

// adjCheck asserts the full AdjustedPercentileBand field by field.
func adjCheck(t *testing.T, got *AdjustedPercentileBand, want struct {
	count int
	sum   float64
	avg   float64
	std   float64
	min   float64
	max   float64
	p1    float64
	p5    float64
	p10   float64
	p25   float64
	p50   float64
	p75   float64
	p90   float64
	p95   float64
	p99   float64
}) {
	t.Helper()
	if got == nil {
		t.Fatalf("got nil, want non-nil band")
	}
	if got.Count != want.count {
		t.Errorf("Count = %d, want %d", got.Count, want.count)
	}
	adjSameFloat(t, "SumMS", got.SumMS, want.sum)
	adjSameFloat(t, "AvgMS", got.AvgMS, want.avg)
	adjSameFloat(t, "StdMS", got.StdMS, want.std)
	adjSameFloat(t, "MinMS", got.MinMS, want.min)
	adjSameFloat(t, "MaxMS", got.MaxMS, want.max)
	adjSameFloat(t, "P1MS", got.P1MS, want.p1)
	adjSameFloat(t, "P5MS", got.P5MS, want.p5)
	adjSameFloat(t, "P10MS", got.P10MS, want.p10)
	adjSameFloat(t, "P25MS", got.P25MS, want.p25)
	adjSameFloat(t, "P50MS", got.P50MS, want.p50)
	adjSameFloat(t, "P75MS", got.P75MS, want.p75)
	adjSameFloat(t, "P90MS", got.P90MS, want.p90)
	adjSameFloat(t, "P95MS", got.P95MS, want.p95)
	adjSameFloat(t, "P99MS", got.P99MS, want.p99)
}

// TestAdjustedPercentiles verifies the failure-inflated percentile band against
// hand-computed vectors.
func TestAdjustedPercentiles(t *testing.T) {
	t.Run("two_errors_over_five_successes", func(t *testing.T) {
		// clean = [10,20,30,40,50] ms, errorCount = 2. Inflated multiset
		// sorted [10,20,30,40,50,+Inf,+Inf], n=7, nearest-rank
		// idx = round-half-to-even(q*6) clamped to [0,6]:
		//
		//	P1  0.06 -> 0 -> 10      P50 3.00 -> 3 -> 40
		//	P5  0.30 -> 0 -> 10      P75 4.50 -> 4 (even) -> 50
		//	P10 0.60 -> 1 -> 20      P90 5.40 -> 5 -> +Inf
		//	P25 1.50 -> 2 (even) -> 30  P95 5.70 -> 6 -> +Inf
		//	                              P99 5.94 -> 6 -> +Inf
		//
		// Count = 5 + 2 = 7. Sum/Avg/Max = +Inf (any error present),
		// Std = NaN (undefined with +Inf), Min = clean min 10.
		got := AdjustedPercentiles([]float64{10, 20, 30, 40, 50}, 2)
		adjCheck(t, got, struct {
			count int
			sum   float64
			avg   float64
			std   float64
			min   float64
			max   float64
			p1    float64
			p5    float64
			p10   float64
			p25   float64
			p50   float64
			p75   float64
			p90   float64
			p95   float64
			p99   float64
		}{
			count: 7, sum: math.Inf(1), avg: math.Inf(1), std: math.NaN(), min: 10, max: math.Inf(1),
			p1: 10, p5: 10, p10: 20, p25: 30, p50: 40, p75: 50,
			p90: math.Inf(1), p95: math.Inf(1), p99: math.Inf(1),
		})
	})

	t.Run("one_error_half_to_even_tie", func(t *testing.T) {
		// clean = [10,30] ms, errorCount = 1. Inflated sorted
		// [10,30,+Inf], n=3, idx = round-half-to-even(q*2):
		//
		//	P1/P5/P10 0.02/0.10/0.20 -> 0 -> 10
		//	P25 0.50 -> 0 (0 is even; round-half-up would give 1 -> 30) -> 10
		//	P50 1.00 -> 1 -> 30
		//	P75 1.50 -> 2 (2 is even) -> +Inf
		//	P90/P95/P99 1.80/1.90/1.98 -> 2 -> +Inf
		//
		// Count = 3. Sum/Avg/Max = +Inf, Std = NaN, Min = 10.
		got := AdjustedPercentiles([]float64{10, 30}, 1)
		adjCheck(t, got, struct {
			count int
			sum   float64
			avg   float64
			std   float64
			min   float64
			max   float64
			p1    float64
			p5    float64
			p10   float64
			p25   float64
			p50   float64
			p75   float64
			p90   float64
			p95   float64
			p99   float64
		}{
			count: 3, sum: math.Inf(1), avg: math.Inf(1), std: math.NaN(), min: 10, max: math.Inf(1),
			p1: 10, p5: 10, p10: 10, p25: 10, p50: 30, p75: math.Inf(1),
			p90: math.Inf(1), p95: math.Inf(1), p99: math.Inf(1),
		})
	})

	t.Run("no_errors_is_noop", func(t *testing.T) {
		// errorCount = 0 -> nothing to inflate -> nil (upstream no-op).
		if got := AdjustedPercentiles([]float64{10, 20, 30}, 0); got != nil {
			t.Errorf("zero errors: got %+v, want nil", got)
		}
		// Negative errorCount is also a no-op.
		if got := AdjustedPercentiles([]float64{10, 20, 30}, -1); got != nil {
			t.Errorf("negative errors: got %+v, want nil", got)
		}
	})

	t.Run("no_successes_is_noop", func(t *testing.T) {
		// No success-only distribution to inflate -> nil even with errors.
		if got := AdjustedPercentiles(nil, 3); got != nil {
			t.Errorf("empty clean: got %+v, want nil", got)
		}
		if got := AdjustedPercentiles([]float64{}, 3); got != nil {
			t.Errorf("empty clean slice: got %+v, want nil", got)
		}
	})

	t.Run("credit_to_start_derived_latency", func(t *testing.T) {
		// startNS / creditIssuedNS are parallel epoch-ns columns.
		//   rec0: 1.0e9 - 0.5e9 = +0.5e9 ns = 500 ms
		//   rec1: credit 0 (unissued) -> dropped
		//   rec2: 3.0e9 - 3.5e9 = -0.5e9 ns -> clamped to 0 -> 0 ms
		//   rec3: 4.0e9 - 3.9e9 = +0.1e9 ns = 100 ms
		// clean = [500, 0, 100] ms; sorted [0,100,500], n=3, linear rank
		// r = q*2:
		//
		//	P1  0.02 -> 0 + 0.02*100 = 2      P50 1.00 -> 100
		//	P5  0.10 -> 10                    P75 1.50 -> 100 + 0.5*400 = 300
		//	P10 0.20 -> 20                    P90 1.80 -> 100 + 0.8*400 = 420
		//	P25 0.50 -> 50                    P95 1.90 -> 460
		//	                                  P99 1.98 -> 492
		//
		// Sum=600, Avg=200, Min=0, Max=500. Population std: mean 200, squared
		// devs 40000+10000+90000 = 140000, /3, sqrt.
		got := CreditToStartLatency(
			[]int64{1_000_000_000, 2_000_000_000, 3_000_000_000, 4_000_000_000},
			[]int64{500_000_000, 0, 3_500_000_000, 3_900_000_000},
		)
		if got == nil {
			t.Fatalf("got nil, want non-nil")
		}
		if got.Count != 3 {
			t.Errorf("Count = %d, want 3", got.Count)
		}
		adjSameFloat(t, "SumMS", got.SumMS, 600)
		adjSameFloat(t, "AvgMS", got.AvgMS, 200)
		adjSameFloat(t, "StdMS", got.StdMS, math.Sqrt(140000.0/3.0))
		adjSameFloat(t, "MinMS", got.MinMS, 0)
		adjSameFloat(t, "MaxMS", got.MaxMS, 500)
		adjSameFloat(t, "P1MS", got.P1MS, 2)
		adjSameFloat(t, "P5MS", got.P5MS, 10)
		adjSameFloat(t, "P10MS", got.P10MS, 20)
		adjSameFloat(t, "P25MS", got.P25MS, 50)
		adjSameFloat(t, "P50MS", got.P50MS, 100)
		adjSameFloat(t, "P75MS", got.P75MS, 300)
		adjSameFloat(t, "P90MS", got.P90MS, 420)
		adjSameFloat(t, "P95MS", got.P95MS, 460)
		adjSameFloat(t, "P99MS", got.P99MS, 492)

		// Every credit unissued -> nothing to report -> nil.
		if g := CreditToStartLatency(
			[]int64{1_000_000_000, 2_000_000_000},
			[]int64{0, 0},
		); g != nil {
			t.Errorf("all unissued: got %+v, want nil", g)
		}
		// Empty input -> nil.
		if g := CreditToStartLatency(nil, nil); g != nil {
			t.Errorf("empty: got %+v, want nil", g)
		}
	})
}
