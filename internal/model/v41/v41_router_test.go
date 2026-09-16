package v41

import (
	"math"
	"testing"
)

func TestV41RouterGeometryConstantsMatchPublished(t *testing.T) {
	cfg := V41DefaultRouterConfig()
	if cfg.Experts != 384 || cfg.TopK != 6 || cfg.SharedCount != 1 || cfg.RouteScale != 1.5 {
		t.Fatalf("default geometry = %+v, want 384/6/1/1.5", cfg)
	}
	if V41RouterMoEWidth != 2304 {
		t.Fatalf("V41RouterMoEWidth = %d, want 2304", V41RouterMoEWidth)
	}
}

func TestV41SqrtSoftplusMatchesScalarReference(t *testing.T) {
	for _, z := range []float32{-40, -3, -0.5, 0, 0.25, 5, 40, 1e6} {
		want := float32(math.Sqrt(math.Max(float64(z), 0) + math.Log1p(math.Exp(-math.Abs(float64(z))))))
		got := V41SqrtSoftplus(z)
		if math.Abs(float64(got-want)) > 1e-6 {
			t.Fatalf("V41SqrtSoftplus(%g) = %g, want %g", z, got, want)
		}
	}
}
