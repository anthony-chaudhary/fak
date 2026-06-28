//go:build amd64

package model

import (
	"encoding/binary"
	"testing"
)

// TestQ5KReduceAsmMatchesScalar pins the AVX2 Q5_K reduction kernel to the scalar reference on the
// integer reductions it owns (IS = Σ q5*qx, SS = Σ qx), bit-for-bit. The float combine is shared
// Go, so once the int32 reductions match, the full asm-path dot equals the scalar-int8-path dot
// exactly. The Q5_K 5th bit is reassembled from qh with a per-chunk constant shift; a mismatch here
// means a real qh-bit / nibble / sign bug in the asm. Skips on a CPU without AVX2.
func TestQ5KReduceAsmMatchesScalar(t *testing.T) {
	if !detectAVX2() {
		t.Skip("AVX2 not available — q5k asm inactive")
	}
	const out, in = 9, 768 // 3 super-blocks/row, odd rows
	nblk := in / qkK
	bb := kindQ5K.blockBytes()
	raw := make([]byte, out*nblk*bb)
	lcgBytes(raw, 0xC5C5A5A5DEADBEEF) // varied bytes exercise all 8 qh bit positions
	for o := 0; o < out; o++ {
		for b := 0; b < nblk; b++ {
			blk := raw[(o*nblk+b)*bb:]
			binary.LittleEndian.PutUint16(blk[0:], f16One)
			binary.LittleEndian.PutUint16(blk[2:], 0)
		}
	}
	qt := quantizeKQuantFromRaw(raw, out, in, kindQ5K)

	x := make([]float32, in)
	for i := range x {
		x[i] = float32((i*7)%23) - 11
	}
	qv := quantizeVecQ8(x)

	isAsm := make([]int32, nblk*8)
	ssAsm := make([]int32, nblk*8)
	isSc := make([]int32, nblk*8)
	ssSc := make([]int32, nblk*8)
	rowBytes := qt.rowBytes()
	for o := 0; o < out; o++ {
		row := qt.raw[o*rowBytes : (o+1)*rowBytes]
		q5kReduceRowAsmAVX2(&row[0], nblk, &qv.q[0], &isAsm[0], &ssAsm[0])
		q5kReduceRowScalar(row, nblk, qv.q, isSc, ssSc)
		for i := range isAsm {
			if isAsm[i] != isSc[i] {
				t.Fatalf("row %d IS[%d]: asm=%d scalar=%d (q5/qh-bit/nibble mismatch)", o, i, isAsm[i], isSc[i])
			}
			if ssAsm[i] != ssSc[i] {
				t.Fatalf("row %d SS[%d]: asm=%d scalar=%d (activation-sum mismatch)", o, i, ssAsm[i], ssSc[i])
			}
		}
	}
	t.Logf("q5k AVX2 reduce bit-identical to scalar across %d rows x %d sub-blocks (tier=%d)", out, nblk*8, qtier)
}
