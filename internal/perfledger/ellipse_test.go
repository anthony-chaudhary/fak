package perfledger

import (
	"errors"
	"math"
	"testing"
)

// fak-test:runtime fast est=5ms lane=default
func TestEllipse(t *testing.T) {
	// y = x: perfectly correlated, minor axis collapses to 0 (clamped), angle pi/4.
	xs := []float64{1, 2, 3, 4, 5}
	e, err := ComputeConfidenceEllipse(xs, xs, 0.95)
	if err != nil {
		t.Fatal(err)
	}
	if e.CenterX != 3 || e.CenterY != 3 {
		t.Fatalf("center = (%v, %v), want (3, 3)", e.CenterX, e.CenterY)
	}
	if e.SemiMinor != 0 || math.Abs(e.AngleRad-math.Pi/4) > 1e-9 {
		t.Fatalf("degenerate ellipse minor=%v angle=%v", e.SemiMinor, e.AngleRad)
	}
	// var(x)=2.5 => lambda1 = 5 (sxx+syy+2sxy over two). chi2(95%,2dof) = -2 ln 0.05.
	wantMajor := math.Sqrt(-2 * math.Log(0.05) * 5)
	if math.Abs(e.SemiMajor-wantMajor) > 1e-9 {
		t.Fatalf("semi-major = %v, want %v", e.SemiMajor, wantMajor)
	}
	// Uncorrelated equal-variance sample: axes equal up to rounding.
	e2, err := ComputeConfidenceEllipse([]float64{-1, 1, -1, 1}, []float64{-1, -1, 1, 1}, 0.5)
	if err != nil || math.Abs(e2.SemiMajor-e2.SemiMinor) > 1e-9 {
		t.Fatalf("expected circle, got %+v err=%v", e2, err)
	}
	for _, bad := range []float64{0, 1, -0.1, math.NaN()} {
		if _, err := ComputeConfidenceEllipse(xs, xs, bad); !errors.Is(err, ErrEllipseInput) {
			t.Fatalf("confidence %v must be refused, got %v", bad, err)
		}
	}
	if _, err := ComputeConfidenceEllipse([]float64{1}, []float64{1}, 0.9); !errors.Is(err, ErrEllipseInput) {
		t.Fatalf("n<2 must be refused, got %v", err)
	}
}
