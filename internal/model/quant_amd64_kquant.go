//go:build amd64

package model

// quant_amd64_kquant.go — amd64 dispatch for the resident Q5_K/Q6_K int8 decode reductions. The
// q5k/q6k hot GEMV loops (quant_kquant_int8*.go) call q5kReduceRow / q6kReduceRow; this file owns
// the amd64 build of those (quant_noasm_kquant.go owns every other arch). Q5_K has an AVX2 kernel
// (quant_amd64_kquant.s); Q6_K stays scalar until its kernel lands. q5kReduceRowScalar /
// q6kReduceRowScalar are the shared arch-neutral reference; the float combine is shared Go, so a
// reducer is bit-checked against the scalar reference (TestQ5KReduceAsmMatchesScalar).

//go:noescape
func q5kReduceRowAsmAVX2(row *byte, nblk int, qx *int8, Isum, Ssum *int32)

// q5kUseVNNI is read by the q5k asm inner dot (CMPB q5kUseVNNI(SB)) to pick the one-VPDPBUSD-per-dot
// VNNI fast path over the AVX2 sign-extend path — same q5kReduceRowAsmAVX2 entry/unpack either way,
// only the inner reduction differs. 1 when the box has AVX512-VNNI (and the tier isn't pinned down).
// A byte, not a bool, so the asm's CMPB matches its width exactly.
var q5kUseVNNI byte = func() byte {
	if q4kVNNI { // same CPUID gate (AVX512 + VNNI ECX bit 11), same FAK_QKERNEL pin
		return 1
	}
	return 0
}()

// q5kReduceRow dispatches the Q5_K integer reduction to the AVX2/VNNI kernel when the resolved tier
// has AVX2, else the scalar reference. The asm picks VNNI vs AVX2 internally via q5kUseVNNI. IS/SS
// are sized nblk*8 (one I_s/S_s per sub-block).
func q5kReduceRow(row []byte, nblk int, qx []int8, IS, SS []int32) {
	if nblk > 0 && qtier >= tierAVX2 {
		q5kReduceRowAsmAVX2(&row[0], nblk, &qx[0], &IS[0], &SS[0])
		return
	}
	q5kReduceRowScalar(row, nblk, qx, IS, SS)
}

// q6kReduceRow computes the per-group (I_g = Σ q6*qx, S_g = Σ qx) reductions for a Q6_K row.
// Scalar until the Q6_K AVX2 kernel lands (its 16-wide groups + position gather are a separate slice).
func q6kReduceRow(row []byte, nblk int, qx []int8, IS, SS []int32) {
	q6kReduceRowScalar(row, nblk, qx, IS, SS)
}
