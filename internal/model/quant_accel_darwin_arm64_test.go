//go:build fakaccel && darwin && arm64 && cgo

package model

import (
	"math"
	"testing"
)

func TestQGemm8AccelMatchesDequantizedF32Batch(t *testing.T) {
	cases := []struct {
		out, in, P int
	}{
		{4, 32, 3},
		{13, 64, 7},
		{192, 576, 16},
	}
	for _, c := range cases {
		wq := quantizeQ8(mkVec(c.out*c.in, uint64(c.out*1009+c.in*17+c.P)), c.out, c.in)
		X := mkVec(c.P*c.in, uint64(c.P*65537+c.in*31+c.out))
		qp := quantizeBatchPanel(X, c.P, c.in)
		got := make([]float32, c.P*c.out)
		if !qGemm8AccelInto(wq, qp, got) {
			t.Fatalf("out=%d in=%d P=%d: Accelerate path declined", c.out, c.in, c.P)
		}
		want := matMulBatch(dequantQ8ForAccel(wq), X, c.out, c.in, c.P)
		for i := range want {
			if !closeF32Accel(got[i], want[i], 2e-4) {
				t.Fatalf("out=%d in=%d P=%d idx=%d: got %v want %v", c.out, c.in, c.P, i, got[i], want[i])
			}
		}
	}
}

func closeF32Accel(a, b, tol float32) bool {
	if math.IsNaN(float64(a)) || math.IsNaN(float64(b)) {
		return false
	}
	d := a - b
	if d < 0 {
		d = -d
	}
	return d <= tol
}
