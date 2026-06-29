//go:build arm64

package model

import (
	"encoding/binary"
	"testing"
)

// TestQ6KReduceAsmMatchesScalar pins the arm64 NEON SDOT Q6_K reduction kernel to the scalar
// reference on the integer reductions it owns (IS = Σ q6*qx, SS = Σ qx), bit-for-bit — the arm64
// twin of the amd64 AVX2/VNNI check. The float combine is shared Go, so once the int32 reductions
// match, the full asm-path dot equals the scalar-int8-path dot exactly. The Q6_K 2-bit qh is
// reassembled per-position with masks+shifts; a mismatch here means a real qh-bit / nibble /
// position bug in the asm. SDOT and the ones-vector sum are associative with no overflow on these
// ranges, so any lane order yields the same int32. Skips on a part without FEAT_DotProd (where the
// dispatcher never calls the asm).
func TestQ6KReduceAsmMatchesScalar(t *testing.T) {
	if !detectDotProd() {
		t.Skip("FEAT_DotProd (asimddp) not available — q6k asm inactive")
	}
	const out, in = 9, 768 // 3 super-blocks/row, odd rows
	nblk := in / qkK
	bb := kindQ6K.blockBytes()
	raw := make([]byte, out*nblk*bb)
	lcgBytes(raw, 0xC5C5A5A5DEADBEEF) // varied bytes exercise all qh bit positions
	for o := 0; o < out; o++ {
		for b := 0; b < nblk; b++ {
			blk := raw[(o*nblk+b)*bb:]
			binary.LittleEndian.PutUint16(blk[q6kBlockBytes-2:], f16One) // d=1.0
		}
	}
	qt := quantizeKQuantFromRaw(raw, out, in, kindQ6K)

	x := make([]float32, in)
	for i := range x {
		x[i] = float32((i*7)%23) - 11
	}
	qv := quantizeVecQ8(x)

	isAsm := make([]int32, nblk*q6kGroupsPerBlock)
	ssAsm := make([]int32, nblk*q6kGroupsPerBlock)
	isSc := make([]int32, nblk*q6kGroupsPerBlock)
	ssSc := make([]int32, nblk*q6kGroupsPerBlock)
	rowBytes := qt.rowBytes()
	for o := 0; o < out; o++ {
		row := qt.raw[o*rowBytes : (o+1)*rowBytes]
		q6kReduceRowAsm(&row[0], nblk, &qv.q[0], &isAsm[0], &ssAsm[0])
		q6kReduceRowScalar(row, nblk, qv.q, isSc, ssSc)
		for i := range isAsm {
			if isAsm[i] != isSc[i] {
				t.Fatalf("row %d IS[%d]: asm=%d scalar=%d (q6/qh-bit/nibble/position mismatch)", o, i, isAsm[i], isSc[i])
			}
			if ssAsm[i] != ssSc[i] {
				t.Fatalf("row %d SS[%d]: asm=%d scalar=%d (activation-sum mismatch)", o, i, ssAsm[i], ssSc[i])
			}
		}
	}
	t.Logf("q6k NEON SDOT reduce bit-identical to scalar across %d rows x %d groups (neonDot=%v)", out, nblk*q6kGroupsPerBlock, neonDot)
}
