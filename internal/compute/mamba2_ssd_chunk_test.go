package compute

import (
	"math"
	"testing"
)

func TestSSDChunkScanVarlen(t *testing.T) {
	t.Parallel()
	// Two varlen sequences: len 3 and len 2, chunkSize=2, dstate=2.
	cu := []int{0, 3, 5}
	const total, dstate, chunk = 5, 2, 2
	x := []float32{1, 2, -1, 0.5, 1.5}
	dt := []float32{0.2, 0.3, 0.1, 0.4, 0.2}
	a := []float32{-0.5, -0.5, -0.5, -1.0, -1.0}
	b := []float32{1, 0.5, 0.5, 1, 1, 1, 2, -1, 0.5, 0.5}
	c := []float32{1, 1, 0.5, 1, 1, 0.5, 1, 1, 0.25, 0.75}
	init := []float32{0.1, 0.2, -0.3, 0.4}

	gotOut, gotFinal, err := SSDChunkScanVarlen(x, dt, a, b, c, cu, init, chunk, dstate)
	if err != nil {
		t.Fatal(err)
	}

	// Sequential recurrence oracle per sequence.
	for seq := 0; seq < 2; seq++ {
		st := []float64{float64(init[seq*2]), float64(init[seq*2+1])}
		for tIdx := cu[seq]; tIdx < cu[seq+1]; tIdx++ {
			dA := math.Exp(float64(a[tIdx]) * float64(dt[tIdx]))
			var y float64
			for s := 0; s < dstate; s++ {
				st[s] = st[s]*dA + float64(dt[tIdx])*float64(b[tIdx*dstate+s])*float64(x[tIdx])
				y += st[s] * float64(c[tIdx*dstate+s])
			}
			if math.Abs(float64(gotOut[tIdx])-y) > 1e-5 {
				t.Fatalf("token %d: got %v, want %v", tIdx, gotOut[tIdx], y)
			}
		}
		for s := 0; s < dstate; s++ {
			if math.Abs(float64(gotFinal[seq*dstate+s])-st[s]) > 1e-5 {
				t.Fatalf("seq %d final[%d]: got %v, want %v", seq, s, gotFinal[seq*dstate+s], st[s])
			}
		}
	}
}
