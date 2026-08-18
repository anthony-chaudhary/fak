package mathx

import (
	"math"
	"testing"
)

func TestNormalQuantile(t *testing.T) {
	for _, tc := range []struct{ p, want float64 }{{0.5, 0}, {0.025, -1.959963986120195}, {0.975, 1.959963986120195}} {
		if got := NormalQuantile(tc.p); math.Abs(got-tc.want) > 2e-9 {
			t.Fatalf("NormalQuantile(%g) = %.12g, want %.12g", tc.p, got, tc.want)
		}
	}
	if !math.IsInf(NormalQuantile(0), -1) || !math.IsInf(NormalQuantile(1), 1) {
		t.Fatal("NormalQuantile must saturate at the unit-interval boundaries")
	}
}
