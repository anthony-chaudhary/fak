//go:build arm64

package model

import (
	"math"
	"testing"
)

// TestQMatRows4NEONMatchesCell pins the decode GEMV micro-kernel (qmatrows4NEON) bit-for-bit to
// the deferred-reduction reference qgemm8cell(...,4) — the arm64 anchor that the NEON kernel and
// the portable lane-4 reference compute the same Q8_0 dot in the same order. Skips on an arm64
// part without FEAT_DotProd (where the kernel is never dispatched).
func TestQMatRows4NEONMatchesCell(t *testing.T) {
	if !detectDotProd() {
		t.Skip("FEAT_DotProd (asimddp) not available — NEON kernel inactive")
	}
	dims := []int{32, 64, 192, 576, 1536}
	for _, in := range dims {
		for trial := 0; trial < 8; trial++ {
			w := mkVec(4*in, uint64(in*2654435761+trial*40503+11))
			x := mkVec(in, uint64(in*15485863+trial*2246822519+7))
			qt := quantizeQ8(w, 4, in)
			qv := quantizeVecQ8(x)
			var y [4]float32
			qmatrows4NEON(&qt.q[0], &qv.q[0], &qt.d[0], &qv.d[0], in, qt.nblk, &y[0])
			for i := 0; i < 4; i++ {
				want := qgemm8cell(
					qt.q[i*in:i*in+in], qt.d[i*qt.nblk:i*qt.nblk+qt.nblk],
					qv.q, qv.d, qt.nblk, 4)
				if math.Float32bits(y[i]) != math.Float32bits(want) {
					t.Fatalf("in=%d trial=%d row=%d: NEON %v (%08x) != cell %v (%08x)",
						in, trial, i, y[i], math.Float32bits(y[i]), want, math.Float32bits(want))
				}
			}
		}
	}
	t.Logf("qmatrows4NEON bit-identical to qgemm8cell(...,4) across in=%v", dims)
}

// TestQGemm8TileNEONMatchesCell pins the 4×4 prefill tile micro-kernel bit-for-bit to
// qgemm8cell(...,4) for every cell of an isolated tile — the prefill analogue of the GEMV anchor.
func TestQGemm8TileNEONMatchesCell(t *testing.T) {
	if !detectDotProd() {
		t.Skip("FEAT_DotProd (asimddp) not available — NEON kernel inactive")
	}
	dims := []int{32, 64, 192, 576, 1536}
	for _, in := range dims {
		for trial := 0; trial < 8; trial++ {
			w := mkVec(4*in, uint64(in*1000003+trial*97+1))
			qt := quantizeQ8(w, 4, in)
			X := mkVec(4*in, uint64(in*999983+trial*131+5))
			qp := quantizeBatchPanel(X, 4, in)
			const out = 4 // isolated tile: outStride == 4
			var Y [16]float32
			qgemm8tileNEON(&qt.q[0], &qp.q[0], &qt.d[0], &qp.d[0], in, qt.nblk, out, &Y[0])
			for i := 0; i < 4; i++ {
				for j := 0; j < 4; j++ {
					want := qgemm8cell(
						qt.q[i*in:i*in+in], qt.d[i*qt.nblk:i*qt.nblk+qt.nblk],
						qp.q[j*in:j*in+in], qp.d[j*qt.nblk:j*qt.nblk+qt.nblk], qt.nblk, 4)
					got := Y[j*out+i]
					if math.Float32bits(got) != math.Float32bits(want) {
						t.Fatalf("in=%d trial=%d cell(%d,%d): NEON %v (%08x) != cell %v (%08x)",
							in, trial, i, j, got, math.Float32bits(got), want, math.Float32bits(want))
					}
				}
			}
		}
	}
	t.Logf("qgemm8tileNEON bit-identical to qgemm8cell(...,4) across in=%v", dims)
}

// TestQGemm8IntoMatchesScalarNEON exercises the full arm64 TILE dispatcher (qGemm8TileInto: tiles
// + row and token remainders) against the portable lane-4 reference qGemm8scalar, so the remainder
// handling and the tile geometry agree bit-for-bit at non-tile-aligned shapes too. It targets
// qGemm8TileInto directly because the default qGemm8Into uses the per-cell qdot8asm sweep (the
// faster path on Apple Silicon), whose reduction order is qdot8scalar's, not qgemm8cell(...,4)'s.
func TestQGemm8IntoMatchesScalarNEON(t *testing.T) {
	if !detectDotProd() {
		t.Skip("FEAT_DotProd (asimddp) not available — NEON kernel inactive")
	}
	cases := []struct{ out, in, P int }{
		{4, 32, 4}, {8, 64, 8}, {6, 576, 5}, {13, 192, 7},
		{7, 64, 3}, {1, 32, 1}, {1536, 576, 16}, {576, 1536, 33},
	}
	for _, c := range cases {
		w := mkVec(c.out*c.in, uint64(c.out*131+c.in*7+c.P*3+1))
		qt := quantizeQ8(w, c.out, c.in)
		X := mkVec(c.P*c.in, uint64(c.out*977+c.in*41+c.P*5+9))
		qp := quantizeBatchPanel(X, c.P, c.in)
		want := qGemm8scalar(qt, qp, 4)
		// Both the 4×4 tile and the default qGemm8Into (load-reusing row4 + qgemm8cell remainder)
		// are bit-identical to the lane-4 reference.
		for _, gemm := range []struct {
			name string
			fn   func(*q8Tensor, *q8Panel, []float32)
		}{
			{"tile", qGemm8TileInto},
			{"row4", qGemm8Into},
		} {
			Y := make([]float32, c.P*c.out)
			gemm.fn(qt, qp, Y)
			for k := range Y {
				if math.Float32bits(Y[k]) != math.Float32bits(want[k]) {
					t.Fatalf("%s out=%d in=%d P=%d idx=%d: %08x != %08x",
						gemm.name, c.out, c.in, c.P, k, math.Float32bits(Y[k]), math.Float32bits(want[k]))
				}
			}
		}
	}
}
