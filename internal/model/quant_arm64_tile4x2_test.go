//go:build arm64 && !(fakaccel && darwin && cgo)

package model

import (
	"math"
	"testing"
)

// TestQGemm8Tile4x2NEONMatchesCell pins the authored 4×2 prefill tile micro-kernel
// (qgemm8tile4x2NEON, quant_arm64.s) bit-for-bit to qgemm8cell(...,4) for every cell of an
// isolated tile — the NR=2 twin of TestQGemm8TileNEONMatchesCell. The tile preserves the lane-4
// deferred-reduction order, so the contract is exact-bits equality, not argmax/cosine.
func TestQGemm8Tile4x2NEONMatchesCell(t *testing.T) {
	if !detectDotProd() {
		t.Skip("FEAT_DotProd (asimddp) not available — NEON kernel inactive")
	}
	dims := []int{32, 64, 192, 576, 1536}
	for _, in := range dims {
		for trial := 0; trial < 8; trial++ {
			w := mkVec(4*in, uint64(in*1000003+trial*97+1))
			qt := quantizeQ8(w, 4, in)
			X := mkVec(2*in, uint64(in*999983+trial*131+5))
			qp := quantizeBatchPanel(X, 2, in)
			const out = 4 // isolated tile: outStride == 4 (rows), P == 2 tokens
			var Y [8]float32
			qgemm8tile4x2NEON(&qt.q[0], &qp.q[0], &qt.d[0], &qp.d[0], in, qt.nblk, out, &Y[0])
			for i := 0; i < 4; i++ {
				for j := 0; j < 2; j++ {
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
	t.Logf("qgemm8tile4x2NEON bit-identical to qgemm8cell(...,4) across in=%v", dims)
}

// TestQGemm8Tile4x2IntoMatchesScalar exercises the full 4×2 dispatcher (qGemm8Tile4x2Into: tiles +
// out%4 row and P%2 token remainders) against the portable lane-4 reference qGemm8scalar, so the
// remainder handling and the tile geometry agree bit-for-bit at non-tile-aligned shapes too — the
// NR=2 twin of TestQGemm8IntoMatchesScalarNEON's tile arm.
func TestQGemm8Tile4x2IntoMatchesScalar(t *testing.T) {
	if !detectDotProd() {
		t.Skip("FEAT_DotProd (asimddp) not available — NEON kernel inactive")
	}
	cases := []struct{ out, in, P int }{
		{4, 32, 2}, {8, 64, 8}, {6, 576, 5}, {13, 192, 7},
		{7, 64, 3}, {1, 32, 1}, {1536, 576, 16}, {576, 1536, 33}, {66, 320, 11},
	}
	for _, c := range cases {
		w := mkVec(c.out*c.in, uint64(c.out*131+c.in*7+c.P*3+1))
		qt := quantizeQ8(w, c.out, c.in)
		X := mkVec(c.P*c.in, uint64(c.out*977+c.in*41+c.P*5+9))
		qp := quantizeBatchPanel(X, c.P, c.in)
		want := qGemm8scalar(qt, qp, 4)
		Y := make([]float32, c.P*c.out)
		qGemm8Tile4x2Into(qt, qp, Y)
		for k := range Y {
			if math.Float32bits(Y[k]) != math.Float32bits(want[k]) {
				t.Fatalf("tile4x2 out=%d in=%d P=%d idx=%d: %08x != %08x",
					c.out, c.in, c.P, k, math.Float32bits(Y[k]), math.Float32bits(want[k]))
			}
		}
	}
}
