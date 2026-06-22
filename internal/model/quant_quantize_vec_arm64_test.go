//go:build arm64

package model

import (
	"math"
	"testing"
)

// TestQuantizeVecQ8NEONMatchesScalar pins the issue #476 NEON decode quantizer (quantizeVecQ8NEON)
// to the scalar reference quantizeRowQ8scalar BIT-FOR-BIT — identical codes AND identical scale
// bits — over the same representative vectors the row test uses: a generic block, the real decode
// inner dims (576, 1536), an all-zero (d==0) row, an all-negative row, a denormal row whose amax
// underflows to 0, a row of exact round-half boundaries (x*inv == k+0.5), and a row with one zero
// block among nonzero ones. As a differential oracle it also cross-checks against the PRODUCTION
// NEON row kernel (quantizeRowAsmNEON): two independently-authored NEON quantizers must agree.
//
// Bit-match is the gate (FRINTA's round-to-nearest-ties-away reproduces q8round exactly). The goal
// permits a fallback: if a code genuinely differed, the dequantized vector must still pass
// argmax-exact + high cosine vs the scalar reference — that path is exercised below so a future
// kernel that trades a ULP for speed still has a defined, asserted contract instead of silently
// shifting the Q8 logits. Skips on an arm64 part without FEAT_DotProd (NEON kernel not dispatched).
func TestQuantizeVecQ8NEONMatchesScalar(t *testing.T) {
	if !detectDotProd() {
		t.Skip("FEAT_DotProd (asimddp) not available — NEON quantizer inactive, scalar path only")
	}
	rows := [][]float32{
		mkVec(32, 12345),
		mkVec(64, 67890),
		mkVec(576, 314159),
		mkVec(1536, 271828),
		make([]float32, 96),
		negRow(64, 999),
		denormRow(32),
		halfBoundaryRow(),
		mixedRow(),
	}
	for ri, x := range rows {
		nblk := len(x) / qBlk
		got := quantizeVecQ8NEON(x)

		wantQ := make([]int8, len(x))
		wantD := make([]float32, nblk)
		quantizeRowQ8scalar(x, wantQ, wantD, nblk)

		// Differential cross-check: the production NEON row kernel must agree bit-for-bit too.
		refQ := make([]int8, len(x))
		refD := make([]float32, nblk)
		quantizeRowAsmNEON(&x[0], &refQ[0], &refD[0], nblk)
		for i := range got.q {
			if got.q[i] != refQ[i] {
				t.Fatalf("row %d code[%d]=%d != production NEON kernel %d (block %d lane %d)", ri, i, got.q[i], refQ[i], i/qBlk, i%qBlk)
			}
		}
		for b := range got.d {
			if math.Float32bits(got.d[b]) != math.Float32bits(refD[b]) {
				t.Fatalf("row %d scale[%d] bits %#x != production NEON kernel %#x", ri, b, math.Float32bits(got.d[b]), math.Float32bits(refD[b]))
			}
		}

		if got.nblk != nblk {
			t.Fatalf("row %d nblk=%d != %d", ri, got.nblk, nblk)
		}
		bitMatch := true
		for i := range got.q {
			if got.q[i] != wantQ[i] {
				bitMatch = false
				break
			}
		}
		if bitMatch {
			for b := range got.d {
				if math.Float32bits(got.d[b]) != math.Float32bits(wantD[b]) {
					bitMatch = false
					break
				}
			}
		}
		if bitMatch {
			continue // the preferred outcome: bit-for-bit == scalar
		}
		// Fallback contract (the goal permits non-bit-match only if it still clears this gate):
		// dequantized NEON vector argmax-exact and cosine ~1 vs the scalar dequant.
		gotF := dequantQ8Blocks(got.q, got.d, nblk)
		wantF := dequantQ8Blocks(wantQ, wantD, nblk)
		if argmaxF32(gotF) != argmaxF32(wantF) {
			t.Fatalf("row %d not bit-identical AND argmax differs: NEON %d vs scalar %d", ri, argmaxF32(gotF), argmaxF32(wantF))
		}
		if cs := cosineSimilarity(gotF, wantF); cs < 0.99995 {
			t.Fatalf("row %d not bit-identical AND cosine %.6f < 0.99995 vs scalar", ri, cs)
		}
		t.Logf("row %d not bit-identical to scalar but within argmax-exact + cosine gate", ri)
	}
}

// dequantQ8Blocks reconstructs the float vector a Q8_0 (codes,scales) pair represents: code*scale
// per element, the block scale repeated across its 32 codes — the value a Q8_0 dot reads back.
func dequantQ8Blocks(q []int8, d []float32, nblk int) []float32 {
	out := make([]float32, nblk*qBlk)
	for b := 0; b < nblk; b++ {
		dd := d[b]
		for i := 0; i < qBlk; i++ {
			out[b*qBlk+i] = float32(q[b*qBlk+i]) * dd
		}
	}
	return out
}
