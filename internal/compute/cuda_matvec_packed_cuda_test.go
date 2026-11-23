//go:build cuda && cgo

package compute

import (
	"math"
	"math/rand"
	"testing"
)

// TestCUDAMatVecPackedMatchesMatMul holds one packed batch over every
// row-view dtype (whole weights and row views, a shared and a private input)
// to per-weight Upload/MatMul/Read: bit-identical, except NVFP4, whose decode
// GEMV splits the K reduction across a warp and so differs by rounding only.
// fak-test:runtime fast est=3s lane=default
func TestCUDAMatVecPackedMatchesMatMul(t *testing.T) {
	cb := cudaOrSkip(t)
	r := rand.New(rand.NewSource(17))
	const out = 48
	for _, c := range rowViewCases(t, r, out) {
		whole := cb.Upload(c.build(cb, 0, out), c.dt)
		ws := []Tensor{whole}
		if v, err := RowView(cb, whole, 16, 16); err == nil {
			ws = append(ws, v)
		}
		ws = append(ws, whole)
		shared, private := rowViewTestX(r, c.in), rowViewTestX(r, c.in)
		inputs := make([][]float32, len(ws))
		for i := range ws {
			inputs[i] = shared
		}
		inputs[len(ws)-1] = private
		seg := (c.in + PackedMatVecAlign - 1) / PackedMatVecAlign * PackedMatVecAlign
		x := make([]float32, 2*seg)
		copy(x, shared)
		copy(x[seg:], private)
		xOff, yOff := make([]int, len(ws)), make([]int, len(ws))
		yLen := 0
		for i, w := range ws {
			if i == len(ws)-1 {
				xOff[i] = seg
			}
			yOff[i] = yLen
			yLen += (w.Shape[0] + PackedMatVecAlign - 1) / PackedMatVecAlign * PackedMatVecAlign
		}
		got, err := cb.MatVecPacked(ws, x, xOff, yOff, yLen)
		if err != nil {
			t.Fatalf("%s: %v", c.dt, err)
		}
		for i, w := range ws {
			dx := cb.Upload(NewF32(cb, []int{c.in}, inputs[i]), F32)
			want := cb.Read(cb.MatMul(w, dx))
			cb.Free(dx)
			for j, v := range want {
				g := got[yOff[i]+j]
				if c.dt == NVFP4 {
					if d := math.Abs(float64(g - v)); d > 1e-5*(1+math.Abs(float64(v))) {
						t.Fatalf("NVFP4 weight %d row %d: packed %v, MatMul %v", i, j, g, v)
					}
					continue
				}
				if g != v {
					t.Fatalf("%s weight %d row %d: packed %v, MatMul %v", c.dt, i, j, g, v)
				}
			}
		}
		cb.Free(whole)
	}
}
