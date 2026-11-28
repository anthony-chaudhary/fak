// fak-test:runtime fast est=1s lane=default
package perfledger

import (
	"math"
	"testing"
)

func TestBenchmarkPoint(t *testing.T) {
	t.Run("n_runs < 3 sets low_confidence and computes exact 2D covariance + CI", func(t *testing.T) {
		pt2, err := ComputeBenchmarkPoint("c=4", []float64{100, 120}, []float64{10, 20}, 0.95)
		if err != nil {
			t.Fatalf("ComputeBenchmarkPoint(2): %v", err)
		}
		if !pt2.LowConfidence || pt2.NRuns != 2 {
			t.Fatalf("expected LowConfidence=true for n_runs=2, got %+v", pt2)
		}
		// Mean X = 110, Mean Y = 15, VarX = 200, VarY = 50, CovXY = 100.
		if math.Abs(pt2.XMean-110) > 1e-9 || math.Abs(pt2.YMean-15) > 1e-9 {
			t.Fatalf("means = (%v, %v), want (110, 15)", pt2.XMean, pt2.YMean)
		}
		if math.Abs(pt2.CovXY-100) > 1e-9 {
			t.Fatalf("CovXY = %v, want 100", pt2.CovXY)
		}
		if !(pt2.XCILow <= pt2.XMean && pt2.XMean <= pt2.XCIHigh) {
			t.Fatalf("X CI bounds violated: %+v", pt2)
		}
		if !(pt2.YCILow <= pt2.YMean && pt2.YMean <= pt2.YCIHigh) {
			t.Fatalf("Y CI bounds violated: %+v", pt2)
		}

		pt1, err := ComputeBenchmarkPoint("c=1", []float64{50}, []float64{5}, 0.95)
		if err != nil {
			t.Fatalf("ComputeBenchmarkPoint(1): %v", err)
		}
		if !pt1.LowConfidence || pt1.XCILow != 50 || pt1.XCIHigh != 50 || pt1.CovXY != 0 {
			t.Fatalf("unexpected n=1 point: %+v", pt1)
		}
	})

	t.Run("n_runs >= 3 clears low_confidence and checks Student-t CI accuracy", func(t *testing.T) {
		xs := []float64{100, 110, 120, 130}
		ys := []float64{24, 22, 20, 18} // negatively correlated: CovXY < 0
		pt, err := ComputeBenchmarkPoint("c=8", xs, ys, 0.95)
		if err != nil {
			t.Fatalf("ComputeBenchmarkPoint(4): %v", err)
		}
		if pt.LowConfidence {
			t.Fatalf("expected LowConfidence=false for n_runs=4, got %+v", pt)
		}
		if pt.CovXY >= 0 {
			t.Fatalf("expected negative CovXY, got %v", pt.CovXY)
		}
		// Student-t 95% CI with df=3: t_{0.025,3} ~= 3.182446, SE_X = sqrt(166.6667/4) ~= 6.45497, HW ~= 20.5426.
		wantHW := 3.182446305 * math.Sqrt((500.0/3.0)/4.0)
		gotHW := pt.XCIHigh - pt.XMean
		if math.Abs(gotHW-wantHW) > 1e-4 {
			t.Fatalf("X CI half-width = %.6f, want %.6f", gotHW, wantHW)
		}
	})

	t.Run("validator rejects ci_low > mean, mean > ci_high, NaN, and inconsistent low_confidence", func(t *testing.T) {
		invalid := []BenchmarkPoint{
			{Label: "bad-x-low", XMean: 10, XCILow: 11, XCIHigh: 12, YMean: 5, YCILow: 4, YCIHigh: 6, NRuns: 3},
			{Label: "bad-y-high", XMean: 10, XCILow: 9, XCIHigh: 11, YMean: 5, YCILow: 4, YCIHigh: 4.5, NRuns: 3},
			{Label: "zero-runs", XMean: 10, XCILow: 10, XCIHigh: 10, YMean: 5, YCILow: 5, YCIHigh: 5, NRuns: 0, LowConfidence: true},
			{Label: "missing-low-conf", XMean: 10, XCILow: 9, XCIHigh: 11, YMean: 5, YCILow: 4, YCIHigh: 6, NRuns: 2, LowConfidence: false},
			{Label: "spurious-low-conf", XMean: 10, XCILow: 9, XCIHigh: 11, YMean: 5, YCILow: 4, YCIHigh: 6, NRuns: 3, LowConfidence: true},
			{Label: "nan-cov", XMean: 10, XCILow: 9, XCIHigh: 11, YMean: 5, YCILow: 4, YCIHigh: 6, CovXY: math.NaN(), NRuns: 3},
		}
		for _, tc := range invalid {
			if err := tc.Validate(); err == nil {
				t.Errorf("%s: expected validation error", tc.Label)
			}
		}
	})
}
