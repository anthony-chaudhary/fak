package mathx

import "testing"

func TestMeanBy(t *testing.T) {
	if got := MeanBy([]int{2, 4, 9}, func(v int) float64 { return float64(v + 1) }); got != 6 {
		t.Fatalf("MeanBy = %g, want 6", got)
	}
	if got := MeanBy([]int(nil), func(v int) float64 { return float64(v) }); got != 0 {
		t.Fatalf("empty MeanBy = %g", got)
	}
}
