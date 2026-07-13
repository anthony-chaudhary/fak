package kernel

import (
	"math"
	"testing"
)

func TestFDotFixedReduction(t *testing.T) {
	r := []float32{1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11}
	x := []float32{.5, 1.5, -.25, 2, -.75, .125, 3, -.5, 2.5, -.2, .3}
	var want float32
	// Independent reproduction of the contract's fixed eight-lane tree.
	var lanes [8]float32
	for i := range r {
		lanes[i&7] += r[i] * x[i]
	}
	want = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) +
		((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]))
	if got := FDot(r, x); math.Float32bits(got) != math.Float32bits(want) {
		t.Fatalf("FDot = %v (%08x), want fixed tree %v (%08x)", got, math.Float32bits(got), want, math.Float32bits(want))
	}
}
