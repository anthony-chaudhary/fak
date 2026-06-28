//go:build amd64

package model

import (
	"math/rand"
	"testing"
)

// TestQ4KReduceAsmMatchesScalar pins the AVX2 reduction kernel to the scalar reference on the
// integer reductions it owns (IS = Σ nibble*qx, SS = Σ qx), bit-for-bit. This is the whole asm
// correctness story for the resident-Q4_K int8 path: the float combine is shared Go, so once the
// int32 reductions match, the full asm-path dot equals the scalar-int8-path dot exactly. VPMADDWD
// and the ones-vector sum are associative with no overflow on these ranges, so any lane order
// yields the same int32 — a mismatch here means a real unpack/sign/ordering bug in the asm. Skips
// on a CPU without AVX2 (where the dispatcher never calls the asm).
func TestQ4KReduceAsmMatchesScalar(t *testing.T) {
	if !detectAVX2() {
		t.Skip("AVX2 not available — q4k asm inactive")
	}
	const out, in = 16, 768 // nblk = 3 super-blocks/row
	rng := rand.New(rand.NewSource(11))
	nblk := in / qkK
	raw := make([]byte, out*nblk*q4kBlockBytes)
	blk := make([]byte, q4kBlockBytes)
	for o := 0; o < out; o++ {
		for b := 0; b < nblk; b++ {
			randQ4KBlock(rng, blk)
			off := (o*nblk + b) * q4kBlockBytes
			copy(raw[off:off+q4kBlockBytes], blk)
		}
	}
	qt := quantizeQ4KFromRaw(raw, out, in)

	x := make([]float32, in)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	qv := quantizeVecQ8(x)

	isAsm := make([]int32, nblk*8)
	ssAsm := make([]int32, nblk*8)
	isSc := make([]int32, nblk*8)
	ssSc := make([]int32, nblk*8)
	rowBytes := qt.q4kRowBytes()
	for o := 0; o < out; o++ {
		row := qt.raw[o*rowBytes : (o+1)*rowBytes]
		q4kReduceRowAsmAVX2(&row[0], nblk, &qv.q[0], &isAsm[0], &ssAsm[0])
		q4kReduceRowScalar(row, nblk, qv.q, isSc, ssSc)
		for i := range isAsm {
			if isAsm[i] != isSc[i] {
				t.Fatalf("row %d IS[%d]: asm=%d scalar=%d (nibble-dot mismatch — unpack/sign bug)", o, i, isAsm[i], isSc[i])
			}
			if ssAsm[i] != ssSc[i] {
				t.Fatalf("row %d SS[%d]: asm=%d scalar=%d (activation-sum mismatch)", o, i, ssAsm[i], ssSc[i])
			}
		}
	}
	t.Logf("q4k AVX2 reduce bit-identical to scalar across %d rows x %d sub-blocks (tier=%d)", out, nblk*8, qtier)
}
